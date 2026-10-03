"""
Active sampling training entry point using a COLMAP processed sparse dataset.

This script loads camera parameters and sparse-view images from a COLMAP `sparse`
directory, constructs a simple Gaussian scene from the reconstructed points, and
runs the validation-active sampling loop defined in
`examples.simple_trainer_with_active_sampling`.
"""

from __future__ import annotations

import argparse
import struct
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import time

# 添加项目根目录到PYTHONPATH，解决相对导入问题
_SCRIPT_DIR = Path(__file__).resolve().parent
_EXAMPLES_DIR = _SCRIPT_DIR.parent  # examples目录
_ROOT_DIR = _EXAMPLES_DIR.parent  # GDDN根目录
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# 添加 active_sampling 目录到 Python 路径，支持 GDDN_ControlNet 内部导入（如 controlnet_components）
_active_sampling_dir = _EXAMPLES_DIR / "active_sampling"
if _active_sampling_dir.exists() and str(_active_sampling_dir) not in sys.path:
    sys.path.insert(0, str(_active_sampling_dir))

# 添加 Depth-Anything-V2 到 Python 路径以支持 DACD
_depth_anything_path = Path(__file__).parent.parent / "models" / "Depth-Anything-V2"
if _depth_anything_path.exists() and str(_depth_anything_path) not in sys.path:
    sys.path.insert(0, str(_depth_anything_path))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from examples.active_sampling.gaussian_parameter_contract import (
    EPSILON, field_state, load_field, ply_parameters, validate_field,
)
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from PIL import Image

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from examples.active_sampling.configs.active_sampling_config import ActiveSamplingConfig
from examples.active_sampling.checkpoint_utils import load_state_dict_with_whitelist
from examples.active_sampling.gddn_trainer import ThreeStageTrainer
from examples.active_sampling.uncertainty_estimator import (
    ConditionProvider,
    SparseViewBundle,
)
from examples.active_sampling.utils import ViewCandidate
from examples.active_sampling.viewpoint_sampler import ExistingCamera
from examples.active_sampling.simple_trainer_with_active_sampling import (
    DistributedActiveSamplingContext,
    train_3dgs_with_active_sampling,
    run_phase2_active_sampling,
    ProposalPatch,
    PseudoView,
)
from gsplat.rendering import rasterization
from gsplat.utils import depth_to_normal
from gsplat.distributed import cli
from gsplat import export_splats
import json


CAMERA_MODELS: Dict[int, Tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}

STAGE_RESOLUTION: Dict[str, Tuple[int, int]] = {
    "stage1": (320, 320),
    "stage2b": (320, 320),
    "stage3": (320, 320),
}

ABLATION_PRESETS = (
    "none",
    "full",
    "no_active_selection",
    "no_verified_evidence",
    "no_trust_weights",
    "rgb_only",
    "full_image_pseudo",
    "pseudo_densify_stats",
)


def _set_experiment_seed(seed: int) -> None:
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_ablation_preset(
    config: ActiveSamplingConfig,
    *,
    preset: str,
    seed: int,
    world_rank: int,
) -> None:
    preset = str(preset or "none")
    if preset == "none":
        return
    config.ablation_preset = preset
    config.ablation_random_selection_seed = int(max(seed, 0))

    config.plan6_four_stage_enable = True
    # 组件预设不产生校准证据，保留调用方的校准和准入状态。
    config.phase3_use_plan6_trust_weights = True
    config.phase3_use_plan6_depth_alpha_losses = True
    config.pseudo_require_verified_evidence = True
    config.phase3_use_verified_evidence_only = True
    config.phase3_use_patch_proposals_only = True
    config.allow_pseudo_densification_stats = False
    config.ablation_random_selection = False

    if preset == "no_active_selection":
        config.ablation_random_selection = True
    elif preset == "no_verified_evidence":
        config.pseudo_require_verified_evidence = False
        config.phase3_use_verified_evidence_only = False
        config.pseudo_verified_allow_supported_fallback = True
    elif preset == "no_trust_weights":
        config.phase3_use_plan6_trust_weights = False
        config.phase3_use_plan6_depth_alpha_losses = False
    elif preset == "rgb_only":
        config.phase3_use_plan6_depth_alpha_losses = False
    elif preset == "full_image_pseudo":
        config.phase3_use_patch_proposals_only = False
        config.phase3_use_verified_evidence_only = False
        config.pseudo_require_verified_evidence = False
        config.phase3_use_plan6_trust_weights = False
        config.phase3_use_plan6_depth_alpha_losses = False
    elif preset == "pseudo_densify_stats":
        config.allow_pseudo_densification_stats = True
    elif preset != "full":
        raise ValueError(f"Unknown ablation preset: {preset}")

    if world_rank == 0:
        print(
            "[AblationPreset] "
            f"preset={preset}, random_selection={config.ablation_random_selection}, "
            f"verified_required={config.pseudo_require_verified_evidence}, "
            f"verified_only={config.phase3_use_verified_evidence_only}, "
            f"patch_only={config.phase3_use_patch_proposals_only}, "
            f"trust_weights={config.phase3_use_plan6_trust_weights}, "
            f"depth_alpha={config.phase3_use_plan6_depth_alpha_losses}, "
            f"pseudo_densify_stats={config.allow_pseudo_densification_stats}"
        )


def _export_stage_checkpoint(model: nn.Module, output_dir: Path) -> None:
    # 如果是 DDP 模型，解包出原始模型
    if hasattr(model, "module"):
        model = model.module  # type: ignore
    
    output_dir.mkdir(parents=True, exist_ok=True)
    model.unet.save_pretrained(output_dir / "unet")
    model.vae.save_pretrained(output_dir / "vae")
    model.scheduler.save_pretrained(output_dir / "scheduler")
    extras: Dict[str, object] = {
        "condition_encoder": model.condition_encoder.state_dict(),
        "condition_adapter": model.condition_adapter.state_dict(),
        "view_aggregator": model.view_aggregator.state_dict(),
        "dropout_rate": float(model.dropout.p),
    }
    if getattr(model, "uncertainty_feature_proj", None) is not None:
        extras["uncertainty_feature_proj"] = model.uncertainty_feature_proj.state_dict()
    if getattr(model, "epistemic_head", None) is not None:
        extras["epistemic_head"] = model.epistemic_head.state_dict()
    if getattr(model, "aleatoric_head", None) is not None:
        extras["aleatoric_head"] = model.aleatoric_head.state_dict()
    torch.save(extras, output_dir / "extras.pt")


def _read_bytes(num_bytes: int, fmt: str, file_obj) -> Tuple:
    data = file_obj.read(num_bytes)
    return struct.unpack(fmt, data)


def read_cameras_binary(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    cameras: Dict[int, Dict[str, np.ndarray]] = {}
    with path.open("rb") as fid:
        num_cams = _read_bytes(8, "<Q", fid)[0]
        for _ in range(num_cams):
            cam_id, model_id, width, height = _read_bytes(24, "<iiQQ", fid)
            model_name, num_params = CAMERA_MODELS.get(model_id, ("PINHOLE", 4))
            params = np.array(_read_bytes(8 * num_params, f"<{num_params}d", fid))
            cameras[cam_id] = {
                "model": model_name,
                "width": np.float64(width),
                "height": np.float64(height),
                "params": params,
            }
    return cameras


def read_images_binary(path: Path) -> List[Dict[str, np.ndarray]]:
    images: List[Dict[str, np.ndarray]] = []
    with path.open("rb") as fid:
        num_images = _read_bytes(8, "<Q", fid)[0]
        for _ in range(num_images):
            image_id = _read_bytes(4, "<i", fid)[0]
            qw, qx, qy, qz = _read_bytes(32, "<4d", fid)
            tx, ty, tz = _read_bytes(24, "<3d", fid)
            camera_id = _read_bytes(4, "<i", fid)[0]
            name_bytes = []
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.append(char)
            name = b"".join(name_bytes).decode("utf-8")
            num_points2d = _read_bytes(8, "<Q", fid)[0]
            fid.read(num_points2d * 24)
            images.append(
                {
                    "image_id": np.int64(image_id),
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": np.int64(camera_id),
                    "name": name,
                }
            )
    images.sort(key=lambda item: item["name"])
    return images


def read_points3d_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    points: List[List[float]] = []
    colors: List[List[float]] = []
    with path.open("rb") as fid:
        num_points = _read_bytes(8, "<Q", fid)[0]
        for _ in range(num_points):
            fid.read(8)  # point3D_id
            x, y, z = _read_bytes(24, "<3d", fid)
            r, g, b = _read_bytes(3, "3B", fid)
            fid.read(8)  # reprojection error
            track_length = _read_bytes(8, "<Q", fid)[0]
            fid.read(track_length * 8)
            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])
    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array(
        [
            [
                1.0 - 2.0 * (y**2 + z**2),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x**2 + z**2),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x**2 + y**2),
            ],
        ]
    )


def build_view_matrix(qvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation = qvec_to_rotmat(qvec)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = tvec
    return view


def camera_intrinsics_from_params(
    model: str, params: np.ndarray, scale_x: float, scale_y: float
) -> np.ndarray:
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        fx = fy = f
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
    else:
        base = params[:4]
        if model.startswith("SIMPLE"):
            f, cx, cy = base[:3]
            fx = fy = f
        else:
            fx, fy, cx, cy = base
    intrinsics = np.array(
        [
            [fx * scale_x, 0.0, cx * scale_x],
            [0.0, fy * scale_y, cy * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return intrinsics


def load_sparse_views(
    data_dir: Path,
    device: torch.device,
    max_views: int,
    target_size: int,
    select_indices: Optional[Sequence[int]] = None,
    image_subdir: str = "images",
    holdout_stride: int = 0,
) -> Tuple[List[ExistingCamera], torch.Tensor, SparseViewBundle]:
    cameras = read_cameras_binary(data_dir / "sparse/0/cameras.bin")
    images = read_images_binary(data_dir / "sparse/0/images.bin")
    image_dir = data_dir / image_subdir
    # 若选择的图像子目录与 COLMAP 原始目录不同，则建立名称映射
    name_map: Optional[Dict[str, str]] = None
    try:
        if image_subdir != "images":
            def _get_rel_paths(path_dir: Path) -> List[str]:
                result: List[str] = []
                for dp, dir_names, fns in os.walk(path_dir):
                    dir_names[:] = [dir_name for dir_name in dir_names if not dir_name.startswith(".")]
                    fns[:] = [file_name for file_name in fns if not file_name.startswith(".")]
                    for f in fns:
                        result.append(os.path.relpath(os.path.join(dp, f), path_dir))
                return result
            colmap_dir = data_dir / "images"
            colmap_files = sorted(_get_rel_paths(colmap_dir))
            sub_files = sorted(_get_rel_paths(image_dir))
            # 仅当文件数相同且逐一对应时进行映射
            if len(colmap_files) == len(sub_files) and len(colmap_files) > 0:
                name_map = dict(zip(colmap_files, sub_files))
    except Exception:
        name_map = None
    # 优先按显式索引选择；否则排除测试视角后均匀采样训练视角。
    if select_indices is not None and len(select_indices) > 0:
        # 仅保留指定下标（基于按文件名排序后的顺序）
        filtered: List[Dict[str, np.ndarray]] = []
        total = len(images)
        for idx in select_indices:
            if idx < 0 or idx >= total:
                raise IndexError(f"select_indices 包含越界索引 {idx}（总数={total}）")
            filtered.append(images[idx])
        images = filtered
    else:
        candidate_images = [
            image_info
            for image_index, image_info in enumerate(images)
            if holdout_stride <= 0 or image_index % holdout_stride != 0
        ]
        if max_views > 0 and len(candidate_images) > max_views:
            sampled_positions = np.linspace(
                0,
                len(candidate_images) - 1,
                num=max_views,
                dtype=np.int64,
            )
            images = [candidate_images[int(position)] for position in sampled_positions]
        else:
            images = candidate_images

    sparse_images: List[torch.Tensor] = []
    viewmats: List[torch.Tensor] = []
    existing: List[ExistingCamera] = []

    for info in images:
        # 若存在名称映射，则以映射后的相对路径读取
        rel_name = info["name"]
        if name_map is not None and rel_name in name_map:
            rel_name = name_map[rel_name]
        image_path = image_dir / rel_name
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image file: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = pil_image.size
        if target_size > 0:
            scale = target_size / max(orig_width, orig_height)
            resized_width = max(1, int(round(orig_width * scale)))
            resized_height = max(1, int(round(orig_height * scale)))
            # Stable Diffusion VAE 需要分辨率为8的倍数，这里向下对齐以避免后续解码尺寸不匹配
            resized_width = max(8, (resized_width // 8) * 8)
            resized_height = max(8, (resized_height // 8) * 8)
            new_size = (resized_width, resized_height)
            pil_image = pil_image.resize(new_size, Image.BICUBIC)
        else:
            new_size = (orig_width, orig_height)

        image_tensor = torch.from_numpy(np.array(pil_image)).float().permute(2, 0, 1)
        image_tensor = image_tensor / 255.0
        sparse_images.append(image_tensor.to(device))

        camera = cameras[int(info["camera_id"])]
        scale_x = new_size[0] / camera["width"]
        scale_y = new_size[1] / camera["height"]
        intrinsics = torch.from_numpy(
            camera_intrinsics_from_params(camera["model"], camera["params"], scale_x, scale_y)
        ).to(device)

        view = torch.from_numpy(build_view_matrix(info["qvec"], info["tvec"])).to(device)
        camtoworld = torch.linalg.inv(view)
        position = camtoworld[:3, 3]
        existing.append(ExistingCamera(position=position, viewmat=view, intrinsics=intrinsics))
        viewmats.append(view)

    images_tensor = torch.stack(sparse_images)
    poses_tensor = torch.stack(viewmats)
    bundle = SparseViewBundle(images=images_tensor, poses=poses_tensor)
    return existing, images_tensor, bundle


class GaussianPointField(nn.Module):
    """COLMAP 初始化；scales 为物理标准差，opacities 为物理 alpha。"""

    def __init__(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        device: torch.device,
        scale_init: float = -1.0,  # -1表示自动计算
        opacity_init: float = 0.5,
    ) -> None:
        super().__init__()
        self.means = nn.Parameter(points.to(device))
        quats = torch.zeros(points.shape[0], 4, device=device)
        quats[:, 0] = 1.0
        self.quats = nn.Parameter(quats)

        # 自动计算合适的scale（基于场景尺度）
        if scale_init < 0:
            # 计算点云的边界框对角线长度
            bbox_min = points.min(dim=0)[0]
            bbox_max = points.max(dim=0)[0]
            bbox_diag = torch.norm(bbox_max - bbox_min).item()
            # scale设为对角线长度的1%（经验值）
            scale_init = max(bbox_diag * 0.01, EPSILON)
            print(f"自动计算scale_init={scale_init:.4f} (基于场景对角线{bbox_diag:.2f})")

        self.scales = nn.Parameter(torch.full((points.shape[0], 3), scale_init, device=device))
        self.opacities = nn.Parameter(torch.full((points.shape[0],), opacity_init, device=device))
        clipped_colors = colors.clamp(1e-4, 1.0 - 1e-4)
        color_logits = torch.logit(clipped_colors)
        self._color_logits = nn.Parameter(color_logits.to(device))
        validate_field(self)

    @property
    def colors(self) -> torch.Tensor:
        return torch.sigmoid(self._color_logits)


class StageTrainingDataset(Dataset):
    """针对GDDN三阶段训练准备的数据集，预先缓存渲染与潜变量。"""

    def __init__(
        self,
        *,
        existing_cameras: Sequence[ExistingCamera],
        real_images: torch.Tensor,
        sparse_bundle: SparseViewBundle,
        condition_provider: ConditionProvider,
        gddn_model: nn.Module,
        device: torch.device,
        target_resolution: Tuple[int, int],
    ) -> None:
        super().__init__()
        self._height, self._width = target_resolution
        self._existing = list(existing_cameras)
        self._device = device

        # 预处理稀疏视角数据，统一到目标分辨率，后续由 DataLoader 负责堆叠
        resized_sparse = F.interpolate(
            sparse_bundle.images.float(),
            size=target_resolution,
            mode="bilinear",
            align_corners=False,
        )
        self.sparse_images = resized_sparse.cpu()
        self.sparse_poses = sparse_bundle.poses.float().cpu()
        # 额外缓存稀疏视角与目标视角的内参，供几何一致性损失使用
        try:
            self._all_intrinsics = torch.stack(
                [cam.intrinsics.detach().cpu() for cam in self._existing]
            )
        except Exception:
            # 在缺失内参的极端情况下回退为单位矩阵，避免训练过程中断
            k = self.sparse_images.shape[0]
            self._all_intrinsics = torch.eye(3).unsqueeze(0).repeat(k, 1, 1)
        orig_height, orig_width = sparse_bundle.images.shape[-2:]
        scale_y = self._height / max(orig_height, 1)
        scale_x = self._width / max(orig_width, 1)
        if abs(float(scale_x) - 1.0) > 1e-6 or abs(float(scale_y) - 1.0) > 1e-6:
            intrinsics_scaled = self._all_intrinsics.clone()
            intrinsics_scaled[:, 0, 0] *= scale_x
            intrinsics_scaled[:, 0, 2] *= scale_x
            intrinsics_scaled[:, 1, 1] *= scale_y
            intrinsics_scaled[:, 1, 2] *= scale_y
            self._all_intrinsics = intrinsics_scaled

        self.target_pose = torch.stack(
            [camera.viewmat.detach().cpu() for camera in self._existing]
        )

        # 真实图像统一分辨率并编码潜变量
        gddn_prev_mode = gddn_model.training
        gddn_model.eval()
        with torch.no_grad():
            resized_rgb = F.interpolate(
                real_images.float(),
                size=target_resolution,
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
            latents = gddn_model.encode_rgb_to_latent(resized_rgb.to(device))
            # 为避免 VAE 下采样导致的尺寸对齐问题（H,W 非8整数倍），将目标 RGB 设为 VAE 解码结果
            recon_rgb = gddn_model.decode_latent_to_rgb(latents)
        if gddn_prev_mode:
            gddn_model.train()
        self.target_rgb = recon_rgb.detach().cpu()
        self.target_latent = latents.detach().cpu()

        # 预渲染每个目标视角的几何条件
        rgb_renders: List[torch.Tensor] = []
        depth_renders: List[torch.Tensor] = []
        normal_renders: List[torch.Tensor] = []
        reference_depth_renders: List[torch.Tensor] = []
        for idx, camera in enumerate(self._existing):
            candidate = ViewCandidate(
                position=camera.position.detach().cpu(),
                viewmat=camera.viewmat.detach().cpu(),
                intrinsics=self._all_intrinsics[idx].detach().cpu(),
            )
            with torch.no_grad():
                conditions = condition_provider([candidate])
            rgb_renders.append(conditions["rgb_render"][0].detach().cpu())
            depth_tensor = conditions.get("depth_render")
            if depth_tensor is not None:
                depth_renders.append(depth_tensor[0].detach().cpu())
            else:
                depth_renders.append(torch.zeros(3, self._height, self._width))  # 3通道
            normal_tensor = conditions.get("normal_render")
            if normal_tensor is not None:
                normal_renders.append(normal_tensor[0].detach().cpu())
            else:
                normal_renders.append(torch.zeros(3, self._height, self._width))
            reference_depth_tensor = conditions.get("reference_depth_map")
            if reference_depth_tensor is not None:
                reference_depth_renders.append(reference_depth_tensor[0].detach().cpu())
            else:
                reference_depth_renders.append(torch.zeros(1, self._height, self._width))
        self.rgb_render = torch.stack(rgb_renders)
        self.depth_render = torch.stack(depth_renders)
        self.normal_render = torch.stack(normal_renders)
        self.reference_depth_map = torch.stack(reference_depth_renders)

    def __len__(self) -> int:
        return len(self._existing)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "sparse_images": self.sparse_images,
            "sparse_poses": self.sparse_poses,
            "target_pose": self.target_pose[index],
            "rgb_render": self.rgb_render[index],
            "depth_render": self.depth_render[index],
            "normal_render": self.normal_render[index],
            "reference_depth_map": self.reference_depth_map[index],
            "target_latent": self.target_latent[index],
            "target_rgb": self.target_rgb[index],
            # 新增：几何一致性所需的内参（DataLoader 会自动叠成 [B, ...]）
            "sparse_intrinsics": self._all_intrinsics,
            "target_intrinsics": self._all_intrinsics[index],
            "depth_type": "z-depth",
        }


def build_stage_dataloaders(
    *,
    existing_cameras: Sequence[ExistingCamera],
    real_images: torch.Tensor,
    sparse_bundle: SparseViewBundle,
    condition_provider: ConditionProvider,
    gddn_model: nn.Module,
    device: torch.device,
    train_resolution: Tuple[int, int],
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = True,
    train_repeats: int = 1,
    world_size: int = 1,
    world_rank: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    # 如果是 DDP 模型，解包出原始模型用于 dataset 初始化 (编码 latent)
    if hasattr(gddn_model, "module"):
        gddn_model_for_dataset = gddn_model.module
    else:
        gddn_model_for_dataset = gddn_model

    dataset = StageTrainingDataset(
        existing_cameras=existing_cameras,
        real_images=real_images,
        sparse_bundle=sparse_bundle,
        condition_provider=condition_provider,
        gddn_model=gddn_model_for_dataset, # 使用解包后的模型
        device=device,
        target_resolution=train_resolution,
    )
    total = len(dataset)
    indices = list(range(total))
    if total > 1:
        train_split = max(1, int(0.8 * total))
        train_indices = indices[:train_split]
        val_indices = indices[train_split:] or indices[-1:]
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)
    else:
        train_dataset = dataset
        val_dataset = dataset
    pin_memory = device.type == "cuda"
    
    train_sampler = None
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=world_rank,
            shuffle=shuffle,
            drop_last=False # 显存优先，不强制 drop
        )
    elif train_repeats > 1 and len(train_dataset) > 0:
        total_samples = max(len(train_dataset) * int(train_repeats), len(train_dataset))
        train_sampler = RandomSampler(
            train_dataset,
            replacement=True,
            num_samples=total_samples,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # 验证集通常也建议使用 DistributedSampler 以聚合指标，但不需要 shuffle
    val_sampler = None
    if world_size > 1:
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=world_rank,
            shuffle=False,
            drop_last=False
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_condition_provider(
    gaussians: GaussianPointField,
    image_height: int,
    image_width: int,
    device: torch.device,
    mode: str,
    existing_cameras: Optional[Sequence[ExistingCamera]] = None,
    reference_image_height: Optional[int] = None,
    reference_image_width: Optional[int] = None,
):
    reference_camera_list = list(existing_cameras) if existing_cameras is not None else []
    reference_viewmats: Optional[torch.Tensor] = None
    reference_intrinsics: Optional[torch.Tensor] = None
    reference_positions: Optional[torch.Tensor] = None
    if reference_camera_list:
        reference_viewmats = torch.stack(
            [camera.viewmat.detach().to(device) for camera in reference_camera_list], dim=0
        )
        reference_intrinsics = torch.stack(
            [camera.intrinsics.detach().to(device) for camera in reference_camera_list], dim=0
        )
        reference_positions = torch.stack(
            [camera.position.detach().to(device) for camera in reference_camera_list], dim=0
        )
        reference_source_height = int(reference_image_height or image_height)
        reference_source_width = int(reference_image_width or image_width)
        if (
            reference_source_height != image_height
            or reference_source_width != image_width
        ):
            scaled_reference_intrinsics = reference_intrinsics.clone()
            scale_x = image_width / max(reference_source_width, 1)
            scale_y = image_height / max(reference_source_height, 1)
            scaled_reference_intrinsics[:, 0, 0] *= scale_x
            scaled_reference_intrinsics[:, 0, 2] *= scale_x
            scaled_reference_intrinsics[:, 1, 1] *= scale_y
            scaled_reference_intrinsics[:, 1, 2] *= scale_y
            reference_intrinsics = scaled_reference_intrinsics

    def provider(candidates: Sequence[ViewCandidate]) -> Dict[str, torch.Tensor]:
        if not candidates:
            raise ValueError("Candidate list must be non-empty.")
        viewmats = torch.stack([cand.viewmat.to(device) for cand in candidates])
        intrinsics = torch.stack([cand.intrinsics.to(device) for cand in candidates])
        render_mode = "RGB+ED" if mode == "gddn" else "RGB"
        detached_means = gaussians.means.detach().clone()
        detached_quats = gaussians.quats.detach().clone()
        detached_scales = gaussians.scales.detach().clone()
        detached_opacities = gaussians.opacities.detach().clone()
        detached_colors = gaussians.colors.detach().clone()
        with torch.no_grad():
            render_colors, render_alphas, _ = rasterization(
                means=detached_means,
                quats=detached_quats,
                scales=detached_scales,
                opacities=detached_opacities,
                colors=torch.clamp(detached_colors, 0.0, 1.0),
                viewmats=viewmats,
                Ks=intrinsics,
                width=image_width,
                height=image_height,
                render_mode=render_mode,
            )
        rgb = render_colors[..., :3].permute(0, 3, 1, 2).contiguous()
        accum_render = render_alphas.permute(0, 3, 1, 2).contiguous()
        result: Dict[str, torch.Tensor] = {
            "rgb_render": rgb,
            "accum_render": accum_render,
            "transmittance_render": torch.clamp(1.0 - accum_render, min=0.0, max=1.0),
        }
        if mode == "gddn":
            depth = render_colors[..., 3:].permute(0, 3, 1, 2).contiguous()
            camtoworlds = torch.linalg.inv(viewmats)
            normals = depth_to_normal(
                depth[:, :1].permute(0, 2, 3, 1),  # depth_to_normal需要1通道
                camtoworlds,
                intrinsics,
                z_depth=True,
            ).permute(0, 3, 1, 2)
            latent = torch.zeros(
                len(candidates),
                4,
                max(1, image_height // 8),
                max(1, image_width // 8),
                device=device,
            )
            result.update(
                {
                    "depth_render": depth,  # [B, 1, H, W]
                    "normal_render": normals,
                    "target_latent": latent,
                    "camera_intrinsics": intrinsics,
                }
            )
            if (
                reference_viewmats is not None
                and reference_intrinsics is not None
                and reference_positions is not None
            ):
                candidate_positions = torch.stack(
                    [candidate.position.to(device) for candidate in candidates], dim=0
                ).float()
                nearest_reference_indices = torch.cdist(
                    candidate_positions,
                    reference_positions.float(),
                ).argmin(dim=1)
                reference_viewmats_batch = reference_viewmats[nearest_reference_indices]
                reference_intrinsics_batch = reference_intrinsics[nearest_reference_indices]
                with torch.no_grad():
                    reference_render_colors, _, _ = rasterization(
                        means=detached_means,
                        quats=detached_quats,
                        scales=detached_scales,
                        opacities=detached_opacities,
                        colors=torch.clamp(detached_colors, 0.0, 1.0),
                        viewmats=reference_viewmats_batch,
                        Ks=reference_intrinsics_batch,
                        width=image_width,
                        height=image_height,
                        render_mode="RGB+ED",
                    )
                reference_depth = (
                    reference_render_colors[..., 3:].permute(0, 3, 1, 2).contiguous()
                )
                result["reference_depth_map"] = reference_depth
                result["reference_intrinsics"] = reference_intrinsics_batch
        return result

    return provider


def subsample_points(
    points: torch.Tensor, colors: torch.Tensor, max_points: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    if max_points > 0 and points.shape[0] > max_points:
        indices = torch.randperm(points.shape[0])[:max_points]
        points = points[indices]
        colors = colors[indices]
    return points, colors


# ========== 3DGS兼容输出函数 ==========

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """将RGB颜色转换为零阶球谐系数 (SH0)
    
    Args:
        rgb: RGB颜色张量 [N, 3]，范围[0, 1]
    
    Returns:
        SH0系数张量 [N, 3]
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def export_cameras_json(
    real_cameras: Sequence[ExistingCamera],
    pseudo_cameras: List[Dict[str, Any]],
    width: int,
    height: int,
    save_path: str,
) -> None:
    """导出相机参数为cameras.json (真实视角+伪视角)
    
    Args:
        real_cameras: 真实相机列表
        pseudo_cameras: 伪视角相机列表 (包含viewmat, fx, fy等)
        width: 图像宽度
        height: 图像高度
        save_path: 输出文件路径
    """
    cameras = []
    
    # 导出真实视角
    for i, cam in enumerate(real_cameras):
        viewmat = cam.viewmat.cpu().numpy()
        R = viewmat[:3, :3].T.tolist()  # 转置为rotation矩阵
        t = viewmat[:3, 3]
        position = (-np.array(viewmat[:3, :3]).T @ t).tolist()  # 计算相机位置
        
        # 提取内参（ExistingCamera使用intrinsics属性，不是K）
        if hasattr(cam, 'intrinsics') and cam.intrinsics is not None:
            K = cam.intrinsics.cpu().numpy() if hasattr(cam.intrinsics, 'cpu') else np.array(cam.intrinsics)
            fx = float(K[0, 0]) if K.ndim == 2 else 500.0
            fy = float(K[1, 1]) if K.ndim == 2 else 500.0
        else:
            fx = 500.0
            fy = 500.0
        
        cameras.append({
            "id": i,
            "img_name": f"real_{i:04d}.jpg",
            "type": "real",
            "width": width,
            "height": height,
            "position": position,
            "rotation": R,
            "fx": fx,
            "fy": fy,
        })
    
    # 导出伪视角
    for j, pseudo in enumerate(pseudo_cameras):
        viewmat = pseudo["viewmat"]
        if hasattr(viewmat, 'cpu'):
            viewmat = viewmat.cpu().numpy()
        R = viewmat[:3, :3].T.tolist()
        t = viewmat[:3, 3]
        position = (-np.array(viewmat[:3, :3]).T @ t).tolist()
        
        cameras.append({
            "id": len(real_cameras) + j,
            "img_name": pseudo.get("img_name") or f"pseudo_{j:04d}.png",
            "type": "pseudo",
            "width": pseudo.get("width", width),
            "height": pseudo.get("height", height),
            "position": position,
            "rotation": R,
            "fx": pseudo.get("fx", 500.0),
            "fy": pseudo.get("fy", 500.0),
        })
    
    with open(save_path, "w") as f:
        json.dump(cameras, f, indent=2)
    print(f"[Export] cameras.json 已保存: {save_path} ({len(cameras)} cameras)")


def export_input_ply(
    points: torch.Tensor,
    colors: torch.Tensor,
    save_path: str,
) -> None:
    """导出输入点云为PLY文件
    
    Args:
        points: 点云坐标 [N, 3]
        colors: 点云颜色 [N, 3]，范围[0, 1]
        save_path: 输出文件路径
    """
    points_np = points.cpu().numpy().astype(np.float32)
    colors_np = (colors.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    
    # 写入PLY头部和数据
    num_points = len(points_np)
    header = f"""ply
format binary_little_endian 1.0
element vertex {num_points}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    with open(save_path, "wb") as f:
        f.write(header.encode('ascii'))
        for i in range(num_points):
            f.write(struct.pack('<3f3B', 
                points_np[i, 0], points_np[i, 1], points_np[i, 2],
                colors_np[i, 0], colors_np[i, 1], colors_np[i, 2]))
    
    print(f"[Export] input.ply 已保存: {save_path} ({num_points} points)")


def export_gaussians_to_ply(
    gaussians: "GaussianPointField",
    save_path: str,
) -> None:
    """导出高斯体为标准3DGS PLY文件
    
    Args:
        gaussians: GaussianPointField实例
        save_path: 输出文件路径
    """
    export_splats(**ply_parameters(gaussians), format="ply", save_to=save_path)
    print(f"[Export] point_cloud.ply 已保存: {save_path} ({len(gaussians.means)} gaussians)")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Active sampling training on COLMAP data.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["active", "stage1", "stage2b", "stage3", "pipeline"],
        default="active",
        help="运行模式：active=主动采样；stage*=仅执行对应阶段；pipeline=三阶段训练后进入主动采样。",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/llff/fern"),
        help="Directory containing COLMAP `sparse` and `images` folders.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Torch device to use.")
    parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="固定随机种子；<0 表示沿用当前随机状态。",
    )
    parser.add_argument(
        "--ablation-preset",
        type=str,
        choices=ABLATION_PRESETS,
        default="full",
        help="论文方法或消融预设；full 为论文完整配置。",
    )
    parser.add_argument(
        "--pseudo-exist-region-l1-max",
        type=float,
        default=None,
        help="覆盖Phase 2已观测区域L1一致性准入阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-min-coverage",
        type=float,
        default=None,
        help="覆盖verified evidence最小覆盖率；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-min-support-projection-ratio",
        type=float,
        default=None,
        help="覆盖verified evidence支持投影比例阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-pixel-l1-max",
        type=float,
        default=None,
        help="覆盖pseudo与当前render逐像素L1准入阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-warp-l1-max",
        type=float,
        default=None,
        help="覆盖pseudo与几何warp逐像素L1准入阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-warp-render-l1-max",
        type=float,
        default=None,
        help="覆盖warp与当前render支持区L1准入阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-neighborhood-kernel",
        type=int,
        default=None,
        help="覆盖verified evidence局部一致性核大小；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-verified-neighborhood-ratio",
        type=float,
        default=None,
        help="覆盖verified evidence局部一致性比例；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-degraded-accept-l1-max",
        type=float,
        default=None,
        help="覆盖degraded accept支持区L1阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--pseudo-degraded-accept-max-unsupported-l1",
        type=float,
        default=None,
        help="覆盖degraded accept非支持渲染区L1阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--selection-min-probe-verified-coverage",
        type=float,
        default=None,
        help="覆盖视角选择probe verified coverage阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--phase2-probe-min-verified-coverage",
        type=float,
        default=None,
        help="覆盖Phase 2 probe rerank verified coverage阈值；默认使用配置文件值。",
    )
    parser.add_argument(
        "--image-subdir",
        type=str,
        default="images",
        help="指定使用的数据图像子目录（如 'images', 'images_8'），用于公平对比固定分辨率。",
    )
    parser.add_argument("--max-views", type=int, default=3, help="Number of sparse views to use.")
    parser.add_argument(
        "--select-indices",
        type=str,
        default="",
        help="按逗号分隔的图像下标列表（基于按文件名排序后的顺序），例如 '0,9,19'。若提供则优先使用此选择。",
    )
    parser.add_argument(
        "--holdout-stride",
        type=int,
        default=8,
        help="Exclude every N-th sorted image before uniformly sampling training views; LLFF uses 8. Set 0 for explicit splits.",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=-1,  # 禁用resize，保持原始分辨率504×378
        help="Resize longest image side to this value. Set <=0 to keep original size.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20000,
        help="Maximum number of COLMAP points converted to Gaussians.",
    )
    parser.add_argument("--num-iterations", type=int, default=30000, help="Training iterations.")
    parser.add_argument("--post-densify-iterations", type=int, default=30000, help="阶段B纯致密化迭代次数（无GDDN推理）。")
    parser.add_argument("--guidance-interval", type=int, default=10, help="Active sampling interval.")
    parser.add_argument("--warmup-iterations", type=int, default=20, help="Iterations before enabling guidance.")
    parser.add_argument(
        "--lambda-guide",
        type=float,
        default=0.1,
        help="伪视角引导损失的缩放系数 λ_guide。",
    )
    parser.add_argument(
        "--gddn-gating",
        type=str,
        choices=["product", "geometric"],
        default="product",
        help="软门控组合策略：product=相乘，geometric=加权几何平均。",
    )
    parser.add_argument(
        "--gddn-alpha",
        type=float,
        default=0.5,
        help="软门控几何平均的α系数，仅在geometric策略下生效。",
    )
    parser.add_argument(
        "--gddn-gamma",
        type=float,
        default=1.0,
        help="软门控逆方差项的γ系数，控制对数据不确定性的敏感度。",
    )
    parser.add_argument(
        "--gddn-beta",
        type=float,
        default=10.0,
        help="软门控sigmoid门控的β系数，控制认知不确定性的陡峭度。",
    )
    parser.add_argument(
        "--gddn-tau",
        type=float,
        default=-1.0,
        help="若设置为非负值，则固定使用该τ阈值；默认-1表示启用自动估计。",
    )
    parser.add_argument(
        "--gddn-tau-percentile",
        type=float,
        default=70.0,
        help="自动估计τ时采用的不确定性分位点（百分数）。",
    )
    parser.add_argument(
        "--scene-center-mode",
        type=str,
        choices=["cameras", "points", "hybrid", "lookat"],
        default="lookat",
        help="场景中心计算模式: lookat=相机视线注视点（推荐）, cameras=相机位置均值, points=点云均值, hybrid=两者均值。",
    )
    parser.add_argument(
        "--stage-batch-size",
        type=int,
        default=2,
        help="GDDN 阶段训练批大小，仅在 stage 模式下生效。",
    )
    parser.add_argument(
        "--stage-num-workers",
        type=int,
        default=0,
        help="GDDN 阶段训练 DataLoader 的 worker 数量。",
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=10,
        help="Stage1 预训练的轮次。",
    )
    parser.add_argument(
        "--stage1-lr",
        type=float,
        default=1e-4,
        help="Stage1 学习率。",
    )
    parser.add_argument(
        "--stage1-no-perc",
        action="store_true",
        help="禁用 Stage1 感知损失。",
    )
    parser.add_argument(
        "--stage1-no-consistency",
        action="store_true",
        help="禁用 Stage1 多视角几何一致性损失。",
    )
    parser.add_argument(
        "--stage1-lambda-perc",
        type=float,
        default=0.2,
        help="Stage1 感知损失权重。",
    )
    parser.add_argument(
        "--stage1-lambda-consistency",
        type=float,
        default=1.0,
        help="Stage1 几何一致性损失权重。",
    )
    parser.add_argument(
        "--stage1-perc-layers",
        type=str,
        default="relu1_2,relu2_2,relu3_3,relu4_3",
        help="Stage1 感知损失使用的 VGG 特征层，以逗号分隔。",
    )
    parser.add_argument(
        "--stage1-perc-weights",
        type=str,
        default="0.5,0.75,1.0,1.5",
        help="Stage1 感知损失各层权重，与层次一一对应。",
    )
    parser.add_argument(
        "--stage1-depth-type",
        type=str,
        choices=["z-depth", "ray-length"],
        default="z-depth",
        help="Stage1 几何一致性使用的深度类型。",
    )
    parser.add_argument(
        "--stage1-pose-angle-threshold",
        type=float,
        default=15.0,
        help="Stage1 几何一致性的视角夹角门控阈值（单位：度）。",
    )
    parser.add_argument(
        "--stage1-min-sim-threshold",
        type=float,
        default=0.1,
        help="Stage1 几何一致性使用的最小位姿相似度阈值。",
    )
    parser.add_argument(
        "--stage1-recon-alpha",
        type=float,
        default=0.6,
        help="Stage1 重建损失 L1/L2 混合系数。",
    )
    parser.add_argument(
        "--stage1-ddim-steps",
        type=int,
        default=20,
        help="Stage1 短 DDIM 采样步数。",
    )
    parser.add_argument(
        "--stage1-ddim-interval",
        type=int,
        default=5,
        help="Stage1 触发短 DDIM 采样的间隔（步）。",
    )
    parser.add_argument(
        "--stage1-x0-switch-steps",
        type=int,
        default=2000,
        help="Stage1 切换至 x0 重建路径的全局步数阈值。",
    )
    parser.add_argument(
        "--stage1-adaptive-trigger",
        type=float,
        default=0.1,
        help="Stage1 自适应触发阈值（扩散损失下降比例）。",
    )
    parser.add_argument(
        "--stage1-export-dir",
        type=str,
        default="",
        help="Stage1 结束后导出权重的目录（可选）。",
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=5,
        help="Stage2 校准的轮次。",
    )
    parser.add_argument(
        "--stage2-lr",
        type=float,
        default=1e-4,
        help="Stage2 学习率。",
    )
    parser.add_argument(
        "--stage2-steps",
        type=int,
        default=20,
        help="Stage2 MC-Dropout 采样的扩散步数。",
    )
    parser.add_argument(
        "--stage2-mc-samples",
        type=int,
        default=5,
        help="Stage2 MC-Dropout 采样次数。",
    )
    parser.add_argument(
        "--stage2-export-dir",
        type=str,
        default="",
        help="Stage2 结束后导出权重的目录（可选）。",
    )
    parser.add_argument(
        "--stage3-epochs",
        type=int,
        default=5,
        help="Stage3 联合微调的轮次。",
    )
    parser.add_argument(
        "--stage3-lr",
        type=float,
        default=1e-5,
        help="Stage3 学习率。",
    )
    parser.add_argument(
        "--stage3-steps",
        type=int,
        default=15,
        help="Stage3 渲染引导的扩散步数。",
    )
    parser.add_argument(
        "--stage3-mc-samples",
        type=int,
        default=1,
        help="Stage3 先验一致性约束使用的 MC 样本数。",
    )
    parser.add_argument(
        "--stage3-lambda-calib",
        type=float,
        default=0.3,
        help="Stage3 校准损失权重。",
    )
    parser.add_argument(
        "--stage3-lambda-distill",
        type=float,
        default=0.05,
        help="Stage3 先验一致性损失权重。",
    )
    parser.add_argument(
        "--stage3-distill-interval",
        type=int,
        default=20,
        help="Stage3 先验一致性约束间隔步数。",
    )
    parser.add_argument(
        "--stage3-recon-weight",
        type=float,
        default=0.05,
        help="Stage3 重建损失权重（设为 0 可关闭）。",
    )
    parser.add_argument(
        "--stage3-recon-alpha",
        type=float,
        default=0.6,
        help="Stage3 重建损失的 L1/L2 混合系数，范围 [0,1]。",
    )
    parser.add_argument(
        "--stage3-perc-weight",
        type=float,
        default=0.05,
        help="Stage3 感知损失权重（依赖 VGG19 特征，设为 0 可关闭）。",
    )
    parser.add_argument(
        "--stage3-variance-weight",
        type=float,
        default=0.0,
        help="Stage3 方差正则权重，用于防止 UNet 输出方差塌缩（0 表示禁用）。",
    )
    parser.add_argument(
        "--stage3-variance-target",
        type=float,
        default=1.0,
        help="Stage3 方差正则目标 std（针对噪声预测或 x0 重建）。",
    )
    parser.add_argument(
        "--stage3-variance-source",
        type=str,
        choices=["noise", "rgb"],
        default="noise",
        help="方差正则施加的位置：noise=噪声预测，rgb=重建 x0。",
    )
    parser.add_argument(
        "--stage3-final-lr",
        type=float,
        default=0.0,
        help="Stage3 末段使用的学习率（可选）。",
    )
    parser.add_argument(
        "--stage3-final-epochs",
        type=int,
        default=0,
        help="Stage3 末段使用最终学习率的轮数。",
    )
    parser.add_argument(
        "--sampler-type",
        type=str,
        choices=["ddim", "eulera", "ddpm"],
        default=None,
        help="Stage3 训练/诊断使用的扩散采样器（留空则沿用 --mc-sampler-type）。",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Stage3 扩散步数的别名；若指定，则覆盖 --stage3-steps。",
    )
    parser.add_argument(
        "--scheduler-beta-schedule",
        type=str,
        default=None,
        help="覆盖 GDDN scheduler 的 beta_schedule 字段（例如 linear、squaredcos_cap_v2）。",
    )
    parser.add_argument(
        "--scheduler-beta-start",
        type=float,
        default=None,
        help="覆盖 GDDN scheduler 的 beta_start。",
    )
    parser.add_argument(
        "--scheduler-beta-end",
        type=float,
        default=None,
        help="覆盖 GDDN scheduler 的 beta_end。",
    )
    parser.add_argument(
        "--scheduler-prediction-type",
        type=str,
        default=None,
        help="覆盖 scheduler.config.prediction_type（如 epsilon、sample）。",
    )
    parser.add_argument(
        "--scheduler-timestep-spacing",
        type=str,
        default=None,
        help="覆盖 scheduler.config.timestep_spacing（如 leading、trailing）。",
    )
    parser.add_argument(
        "--scheduler-eta",
        type=float,
        default=None,
        help="DDIM/DDPM 采样的 eta 值，若指定则在 Stage3 采样与诊断中生效。",
    )
    parser.add_argument(
        "--stage3-unfreeze-vae",
        action="store_true",
        help="Stage3 是否解冻 VAE（默认仅解码器）。",
    )
    parser.add_argument(
        "--stage3-unfreeze-vae-full",
        action="store_true",
        help="配合 --stage3-unfreeze-vae 使用，解冻 VAE 全部参数。",
    )
    parser.add_argument(
        "--stage3-vae-lr-scale",
        type=float,
        default=0.2,
        help="Stage3 解冻 VAE 时的学习率缩放倍率（乘以主干学习率）。",
    )
    parser.add_argument(
        "--stage3-auto-unfreeze",
        action="store_true",
        help="Stage3 启用指标驱动的 VAE 自动解冻流程。",
    )
    parser.add_argument(
        "--stage3-auto-unfreeze-full",
        action="store_true",
        help="配合 --stage3-auto-unfreeze 使用，触发时解冻 VAE 全部参数。",
    )
    parser.add_argument(
        "--stage3-auto-ece-threshold",
        type=float,
        default=0.05,
        help="自动解冻判定中 ECE 的上限（单位：绝对值）。",
    )
    parser.add_argument(
        "--stage3-auto-corr-threshold",
        type=float,
        default=0.2,
        help="自动解冻判定中误差-不确定性相关系数的下限。",
    )
    parser.add_argument(
        "--stage3-auto-psnr-delta",
        type=float,
        default=0.1,
        help="自动解冻前允许的验证 PSNR 最大波动幅度（单位：dB）。",
    )
    parser.add_argument(
        "--stage3-auto-warmup-epochs",
        type=int,
        default=6,
        help="自动解冻最短等待的轮数（防止过早解冻）。",
    )
    parser.add_argument(
        "--stage3-auto-patience",
        type=int,
        default=3,
        help="自动解冻需要连续满足 ECE/Corr 阈值的轮数。",
    )
    parser.add_argument(
        "--stage3-auto-psnr-window",
        type=int,
        default=5,
        help="自动解冻检测 PSNR 平台的窗口长度。",
    )
    parser.add_argument(
        "--stage3-export-dir",
        type=str,
        default="",
        help="Stage3 结束后导出权重的目录（可选）。",
    )
    parser.add_argument(
        "--quick-stage3-epochs",
        type=int,
        default=0,
        help="若 > 0，则在进入主动采样前自动追加 Stage3 微调若干轮以产出阶段三权重。",
    )
    parser.add_argument(
        "--quick-stage3-lr",
        type=float,
        default=5e-6,
        help="Quick Stage3 微调时使用的学习率。",
    )
    parser.add_argument(
        "--pipeline-run-stage3",
        action="store_true",
        help="在 pipeline 模式下，在 Stage1 与 Stage2b 之后追加执行 Stage3（默认不执行）。",
    )
    parser.add_argument("--candidate-pool", type=int, default=64, help="Number of candidate viewpoints.")
    parser.add_argument("--coarse-candidates", type=int, default=32, help="Top-M results retained after coarse pass.")
    parser.add_argument("--fine-candidates", type=int, default=16, help="Top-K candidates evaluated in fine pass.")
    parser.add_argument("--views-per-round", type=int, default=4, help="Pseudo views selected each round.")
    parser.add_argument("--mc-samples", type=int, default=6, help="MC-Dropout samples for fine stage.")
    parser.add_argument("--mc-samples-coarse", type=int, default=2, help="MC-Dropout samples for coarse stage.")
    parser.add_argument(
        "--score-type",
        type=str,
        choices=["variance", "mi"],
        default="variance",
        help="Uncertainty score type used for ranking candidates.",
    )
    parser.add_argument(
        "--no-score-mask",
        action="store_true",
        help="Disable depth-based masking when aggregating uncertainty scores.",
    )
    parser.add_argument(
        "--aleatoric-floor",
        type=float,
        default=1e-4,
        help="Floor value for aleatoric variance when approximating mutual information.",
    )
    parser.add_argument(
        "--mc-dropout2d-p",
        type=float,
        default=0.15,
        help="Dropout2d probability applied to conditional latents during MC-Dropout passes.",
    )
    parser.add_argument(
        "--mc-token-dropout-p",
        type=float,
        default=0.08,
        help="Token-level dropout probability applied to encoder hidden states during MC-Dropout.",
    )
    parser.add_argument(
        "--mc-cond-noise",
        type=float,
        default=0.02,
        help="Gaussian noise std added to conditional latents during MC-Dropout.",
    )
    parser.add_argument(
        "--mc-condition-alpha",
        type=float,
        default=0.5,
        help="Blending weight between latent and condition features in MC-Dropout mode.",
    )
    parser.add_argument(
        "--mc-latent-noise",
        type=float,
        default=0.005,
        help="Noise std added to initial latents during MC-Dropout.",
    )
    parser.add_argument(
        "--mc-sampler-type",
        type=str,
        choices=["ddim", "eulera", "ddpm"],
        default="ddim",
        help="Sampler used for MC-Dropout inference.",
    )
    parser.add_argument(
        "--ensemble-sampler-type",
        type=str,
        choices=["ddim", "eulera", "ddpm"],
        default="ddim",
        help="Sampler used for ensemble/bootstrap inference.",
    )
    parser.add_argument(
        "--ensemble-noise-scale",
        type=float,
        default=0.0,
        help="Per-step Gaussian noise injected during ensemble/bootstrap inference.",
    )
    parser.add_argument("--estimator", type=str, choices=["gs_dropout", "gddn"], default="gddn")
    parser.add_argument(
        "--gddn-model-path",
        type=str,
        default=None,
        help="Stable Diffusion模型路径（用于GDDN模式）。如果未指定，使用环境变量GDDN_MODEL_PATH或默认相对路径models/stable-diffusion-2-1-base。",
    )
    parser.add_argument(
        "--gddn-resume-checkpoint",
        type=str,
        default=None,
        help="离线训练的.pt checkpoint路径（来自run_bayesgs_diff.py），加载到GDDN模型作为冻结生成先验模块。",
    )
    parser.add_argument(
        "--primitive-dropout",
        type=float,
        default=0.1,
        help="Primitive dropout probability for GS estimator.",
    )
    parser.add_argument(
        "--opacity-jitter",
        type=float,
        default=0.05,
        help="Gaussian opacity multiplicative jitter for GS estimator.",
    )
    parser.add_argument(
        "--min-angle-deg",
        type=float,
        default=15.0,
        help="Minimum angular separation between selected viewpoints (degrees).",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.025,  # 从0.05降低到0.025，适应当前模型的variation_coeff范围
        help="Coefficient of variation threshold for uncertainty quality gating.",
    )
    parser.add_argument(
        "--pearson-threshold",
        type=float,
        default=0.0,
        help="Correlation threshold. 0.0=允许零相关，只禁止强负相关。",
    )
    parser.add_argument(
        "--dacd-interval",
        type=int,
        default=300,
        help="DACD触发的迭代间隔（默认300）",
    )
    parser.add_argument(
        "--dacd-min-anchor-ratio",
        type=float,
        default=0.03,
        help="DACD尺度对齐的最小锚点占比",
    )
    parser.add_argument(
        "--dacd-min-r2",
        type=float,
        default=0.3,
        help="DACD尺度对齐的最小R^2阈值",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=4.0,
        help="Radius of candidate viewpoint sphere (in scene units). Set <=0 for auto.",
    )
    parser.add_argument(
        "--max-deviation-angle",
        type=float,
        default=90.0,
        help="CACG覆盖锥约束: 候选视角与最近已有视角的最大角度偏差(度)。90=排除背面, 60=侧面扩展, 45=紧凑填充。",
    )
    parser.add_argument(
        "--stability-strategy",
        type=str,
        choices=["none", "alternate", "split"],
        default="split",
        help="Strategy to avoid in-place gradient conflicts.",
    )
    # Validation ensemble/bootstrap and health gating
    parser.add_argument(
        "--validation-bootstrap",
        type=int,
        default=10,
        help="Number of bootstrap replicates (10次提高方差估计稳定性).",
    )
    parser.add_argument(
        "--validation-no-mask",
        action="store_true",
        help="Disable depth masking when aggregating ensemble scores for validation.",
    )
    parser.add_argument(
        "--health-min-std",
        type=float,
        default=0.03,
        help="Minimum per-image std required for GDDN outputs; below this disables GDDN active sampling.",
    )
    parser.add_argument(
        "--health-mean-min",
        type=float,
        default=0.05,
        help="Lower bound on mean brightness for GDDN outputs.",
    )
    parser.add_argument(
        "--health-mean-max",
        type=float,
        default=0.95,
        help="Upper bound on mean brightness for GDDN outputs.",
    )
    # Two-stage diffusion sampling parameters (gddn mode)
    parser.add_argument("--coarse-res", type=int, default=256, help="Coarse stage resolution.")
    parser.add_argument("--coarse-steps", type=int, default=20, help="Coarse stage diffusion steps.")
    parser.add_argument("--fine-res", type=int, default=512, help="Fine stage resolution.")
    parser.add_argument("--fine-steps", type=int, default=50, help="Fine stage diffusion steps.")
    parser.add_argument(
        "--log-selected-images",
        action="store_true",
        help="Log selected pseudo views and variance maps to TensorBoard.",
    )
    parser.add_argument(
        "--save-pseudo-png",
        action="store_true",
        help="Persist selected pseudo views as PNG files under the log directory.",
    )
    parser.add_argument(
        "--candidate-log",
        type=str,
        default="",
        help="将候选视角的位姿、分数及 Pearson 相关性写入的 JSONL 路径（留空表示不记录）。",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="",
        help="TensorBoard log directory. Leave empty to disable logging.",
    )
    parser.add_argument(
        "--gaussian-output",
        type=str,
        default="",
        help="训练结束后将高斯场保存为 .pt 文件的路径（可选）。",
    )
    parser.add_argument(
        "--disable-active-sampling",
        action="store_true",
        help="禁用主动视角选择，用于组件消融。",
    )
    parser.add_argument(
        "--gddn-dtype",
        type=str,
        default="fp32",
        choices=["fp16", "bf16", "fp32"],
        help="加载GDDN权重时使用的数据类型，可选 fp16/bf16/fp32，默认 fp32 以避免在线阶段 dtype 冲突。",
    )
    parser.add_argument(
        "--output-3dgs-format",
        action="store_true",
        help="启用3DGS兼容输出格式（cameras.json, input.ply, point_cloud.ply）。",
    )
    parser.add_argument(
        "--pseudo-mode",
        type=str,
        choices=["direct", "weighted", "guidance_only"],
        default="guidance_only",
        help="Phase 3伪视图参与模式: direct(直接监督), weighted(SoftGating加权), guidance_only(仅引导损失，默认推荐)。",
    )
    return parser.parse_args()




def _resolve_online_single_rank_done_path(args: argparse.Namespace) -> Path:
    if args.log_dir:
        base_dir = Path(args.log_dir)
    elif args.gaussian_output:
        base_dir = Path(args.gaussian_output).expanduser().resolve().parent
    elif args.candidate_log:
        base_dir = Path(args.candidate_log).expanduser().resolve().parent
    else:
        base_dir = _ROOT_DIR / "output"
    master_port = os.environ.get("MASTER_PORT", "single")
    return base_dir / "dist_online_sync" / f"rank0_online_done_{master_port}.json"


def _wait_for_online_single_rank_completion(
    done_path: Path,
    *,
    world_rank: int,
    poll_seconds: float = 2.0,
) -> None:
    wait_start_time = time.time()
    last_log_time = wait_start_time
    while not done_path.exists():
        time.sleep(max(float(poll_seconds), 0.1))
        current_time = time.time()
        if current_time - last_log_time >= 60.0:
            print(
                f"[Distributed][rank{world_rank}] 等待 rank0 完成在线阶段: {done_path}"
            )
            last_log_time = current_time


def main_worker(local_rank: int, world_rank: int, world_size: int, args: argparse.Namespace) -> None:
    cuda_allocator_config = "max_split_size_mb:256,expandable_segments:True"
    os.environ.setdefault("PYTORCH_ALLOC_CONF", cuda_allocator_config)
    # 启用 TF32 以降低显存并加速
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device)
    seed_value = int(getattr(args, "seed", -1))
    if seed_value >= 0:
        _set_experiment_seed(seed_value + int(world_rank))
        if world_rank == 0:
            print(f"[Seed] experiment_seed={seed_value}")

    stage1_export_dir: Optional[Path]
    if args.stage1_export_dir:
        stage1_export_dir = Path(args.stage1_export_dir)
    else:
        stage1_export_dir = None
    stage2_export_dir: Optional[Path]
    if args.stage2_export_dir:
        stage2_export_dir = Path(args.stage2_export_dir)
    else:
        stage2_export_dir = None
    stage3_export_dir: Optional[Path]
    if args.stage3_export_dir:
        stage3_export_dir = Path(args.stage3_export_dir)
    else:
        stage3_export_dir = None
    if getattr(args, "steps", None) is not None:
        args.stage3_steps = max(1, int(args.steps))
    scheduler_override_dict: Dict[str, Any] = {}
    if args.scheduler_beta_schedule:
        scheduler_override_dict["beta_schedule"] = args.scheduler_beta_schedule
    if args.scheduler_beta_start is not None:
        scheduler_override_dict["beta_start"] = float(args.scheduler_beta_start)
    if args.scheduler_beta_end is not None:
        scheduler_override_dict["beta_end"] = float(args.scheduler_beta_end)
    if args.scheduler_prediction_type:
        scheduler_override_dict["prediction_type"] = args.scheduler_prediction_type
    if args.scheduler_timestep_spacing:
        scheduler_override_dict["timestep_spacing"] = args.scheduler_timestep_spacing
    scheduler_eta_value: Optional[float]
    if args.scheduler_eta is not None:
        scheduler_eta_value = float(args.scheduler_eta)
    else:
        scheduler_eta_value = None
    sampler_type_override: Optional[str]
    if args.sampler_type:
        sampler_type_override = args.sampler_type.lower()
    else:
        sampler_type_override = None
    select_indices: Optional[List[int]] = None
    if args.select_indices:
        try:
            select_indices = [
                int(part)
                for part in str(args.select_indices).split(",")
                if part.strip() != ""
            ]
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"--select-indices 解析失败: {args.select_indices}") from exc
    
    data_dir = args.data_dir
    if not (data_dir / "sparse/0").exists():
        raise FileNotFoundError(f"{data_dir}/sparse/0 not found.")

    writer: Optional["SummaryWriter"]
    # 仅 Rank 0 记录日志
    if args.log_dir and SummaryWriter is not None and world_rank == 0:
        writer = SummaryWriter(log_dir=args.log_dir)
    else:
        writer = None

    existing_cameras, real_images, sparse_bundle = load_sparse_views(
        data_dir=data_dir,
        device=device,
        max_views=args.max_views,
        target_size=args.target_size,
        select_indices=select_indices,
        image_subdir=args.image_subdir,
        holdout_stride=max(0, args.holdout_stride),
    )
    height, width = real_images.shape[-2:]

    points_np, colors_np = read_points3d_binary(data_dir / "sparse/0/points3D.bin")
    points = torch.from_numpy(points_np)
    colors = torch.from_numpy(colors_np)
    points, colors = subsample_points(points, colors, args.max_points)

    gaussians = GaussianPointField(points, colors, device=device)
    # 根据模式计算scene_center（解决稀疏视角场景中点云与相机坐标不一致问题）
    if args.scene_center_mode == "lookat":
        # ===== scene_center 的语义（非常关键！）=====
        # scene_center 在 viewpoint_sampler 中是【球心】——伪视角 position = scene_center ± radius
        # 因此 scene_center 必须在【相机群附近】（Z≈0），而不是场景内容中心（Z≈80）
        # 相机前方场景内容（look-at target）用于 viewpoint_sampler.look_at_matrix，
        # 与球心 scene_center 是两个不同的概念！
        #
        # 当前数据集：真实相机 Z≈0 朝 +Z 看，场景（高斯体）Z≈80
        # 正确 scene_center = 真实相机位置均值（Z≈0），不是高斯体均值（Z≈80）
        #
        # 相机射线交点算法：用于估计 look-at target（场景注视点），
        # 但其数值解在 forward-facing 场景下 Z 方向不可解（秩亏），用条件数检测。
        _positions = []
        _directions = []
        for _cam in existing_cameras:
            _vm = _cam.viewmat.to(device)  # W2C [4,4]
            _R_w2c = _vm[:3, :3]
            _t_w2c = _vm[:3, 3]
            _pos = -_R_w2c.T @ _t_w2c
            _forward_c2w = _R_w2c[2, :]  # W2C Z行 = C2W 前方
            _norm = _forward_c2w.norm()
            if _norm > 1e-6:
                _directions.append(_forward_c2w / _norm)
                _positions.append(_pos)
        # 始终以相机位置均值作为球心（scene_center 的物理意义）
        _cam_center = torch.stack(_positions).mean(dim=0) if _positions else torch.zeros(3, device=device)
        scene_center = _cam_center  # 伪视角将分布在相机群附近
        # 尝试计算注视点（用于 viewpoint_sampler 的 look_at 目标）
        # 注意：viewpoint_sampler 内部用 look_at_matrix(position, scene_center)
        # 所以 scene_center 实际上也是 look_at 的目标点
        # 更好的做法：让注视点在相机前方合理深度处（而非相机位置）
        if len(_directions) >= 2:
            _A = torch.zeros(3, 3, device=device)
            _b = torch.zeros(3, device=device)
            for _p, _d in zip(_positions, _directions):
                _dd = _d.unsqueeze(1) @ _d.unsqueeze(0)
                _M = torch.eye(3, device=device) - _dd
                _A += _M
                _b += _M @ _p
            _cond = torch.linalg.cond(_A).item()
            if _cond < 1e4:
                try:
                    _lookat_target = torch.linalg.solve(_A, _b)
                    # 以注视目标为球心（相机环绕注视点）
                    scene_center = _lookat_target
                    print(f"[Config] lookat 注视点求解成功 (cond={_cond:.1f}): {scene_center.tolist()}")
                except Exception:
                    print(f"[Config] lookat 求解失败，使用相机均值: {scene_center.tolist()}")
            else:
                # forward-facing：注视目标 = 相机均值沿前方方向前移适当深度
                # 深度估计：高斯体均值到相机均值的前方分量
                _avg_fwd = torch.stack(_directions).mean(dim=0)
                _avg_fwd = _avg_fwd / (_avg_fwd.norm() + 1e-8)
                _gauss_center = gaussians.means.detach().mean(dim=0)
                # 投影：高斯体中心相对于相机均值沿前方方向的深度
                _depth_proj = torch.dot(_gauss_center - _cam_center, _avg_fwd).item()
                _depth_proj = max(abs(_depth_proj), 1.0)  # 至少 1.0
                # scene_center = 相机均值 + 0.1×深度前移（球心略移向场景，改善 look_at 效果）
                # 注意：不能移太多，否则 position = scene_center + 3*dir 仍会在场景背后
                scene_center = _cam_center + 0.05 * _depth_proj * _avg_fwd
                print(f"[Config] lookat forward-facing 降级: 相机均值={_cam_center.tolist()}, "
                      f"前方深度={_depth_proj:.1f}, scene_center={scene_center.tolist()}")


    elif args.scene_center_mode == "cameras":
        # 使用相机位置均值（推荐稀疏视角场景）
        camera_positions = torch.stack([cam.position for cam in existing_cameras])
        scene_center = camera_positions.mean(dim=0).to(device)
    elif args.scene_center_mode == "hybrid":
        # 使用相机和点云的混合均值
        camera_positions = torch.stack([cam.position for cam in existing_cameras])
        cam_center = camera_positions.mean(dim=0).to(device)
        pts_center = gaussians.means.detach().mean(dim=0)
        scene_center = (cam_center + pts_center) / 2
    else:  # "points"
        scene_center = gaussians.means.detach().mean(dim=0)
    
    if world_rank == 0:
        print(f"[Config] scene_center_mode={args.scene_center_mode}, scene_center={scene_center.tolist()}")

    gating_strategy = "multiply" if args.gddn_gating == "product" else "geometric"
    tau_override = args.gddn_tau
    tau_auto = tau_override < 0.0
    tau_value = tau_override if not tau_auto else 0.5
    tau_percentile = float(min(max(args.gddn_tau_percentile, 0.0), 100.0))

    # Ablation study: Disable active sampling if requested
    actual_views_per_round = 0 if args.disable_active_sampling else args.views_per_round
    if args.disable_active_sampling and world_rank == 0:
        print("[Ablation] 主动采样已禁用（num_views_per_round=0）")

    config = ActiveSamplingConfig(
        candidate_pool_size=args.candidate_pool,
        coarse_candidate_count=args.coarse_candidates,
        fine_candidate_count=args.fine_candidates,
        num_views_per_round=actual_views_per_round,
        mc_dropout_samples_coarse=args.mc_samples_coarse,
        mc_dropout_samples_fine=args.mc_samples,
        score_type=args.score_type,
        use_mask_for_scores=not args.no_score_mask,
        aleatoric_variance_floor=args.aleatoric_floor,
        guidance_interval=args.guidance_interval,
        warmup_iterations=args.warmup_iterations,
        pearson_threshold=args.pearson_threshold,
        sphere_radius=args.sphere_radius,
        max_view_deviation_angle=args.max_deviation_angle,
        estimator_mode=args.estimator,
        primitive_dropout_p=args.primitive_dropout,
        opacity_scale_jitter=args.opacity_jitter,
        min_selection_angle_deg=args.min_angle_deg,
        quality_variation_threshold=args.quality_threshold,
        coarse_resolution=args.coarse_res,
        coarse_steps=args.coarse_steps,
        fine_resolution=args.fine_res,
        fine_steps=args.fine_steps,
        mc_dropout2d_p=args.mc_dropout2d_p,
        mc_token_dropout_p=args.mc_token_dropout_p,
        mc_cond_noise_sigma=args.mc_cond_noise,
        mc_condition_alpha=args.mc_condition_alpha,
        mc_latent_noise_std=args.mc_latent_noise,
        mc_sampler_type=args.mc_sampler_type,
        ensemble_sampler_type=args.ensemble_sampler_type,
        ensemble_noise_scale=args.ensemble_noise_scale,
        validation_use_mask=not args.validation_no_mask,
        # Validation/bootstrap + health gating
        validation_bootstrap_repeats=args.validation_bootstrap,
        health_min_std=args.health_min_std,
        health_min_mean=args.health_mean_min,
        health_max_mean=args.health_mean_max,
        log_selected_images=args.log_selected_images,
        save_pseudo_png=args.save_pseudo_png,
        lambda_guidance=args.lambda_guide,
        gate_strategy=gating_strategy,
        gate_alpha=args.gddn_alpha,
        gate_gamma=args.gddn_gamma,
        gate_beta=args.gddn_beta,
        gate_tau=float(tau_value),
        gate_tau_auto=tau_auto,
        gate_tau_percentile=tau_percentile,
        dacd_interval=int(args.dacd_interval),
        dacd_min_anchor_ratio=float(args.dacd_min_anchor_ratio),
        dacd_min_r2=float(args.dacd_min_r2),
        post_densify_iterations=args.post_densify_iterations,
        ablation_preset=str(args.ablation_preset),
        ablation_random_selection_seed=int(max(seed_value, 0)),
    )
    _apply_ablation_preset(
        config,
        preset=str(args.ablation_preset),
        seed=seed_value,
        world_rank=world_rank,
    )
    ablation_threshold_overrides = {
        "pseudo_exist_region_l1_max": args.pseudo_exist_region_l1_max,
        "pseudo_verified_min_coverage": args.pseudo_verified_min_coverage,
        "pseudo_verified_min_support_projection_ratio": args.pseudo_verified_min_support_projection_ratio,
        "pseudo_verified_pixel_l1_max": args.pseudo_verified_pixel_l1_max,
        "pseudo_verified_warp_l1_max": args.pseudo_verified_warp_l1_max,
        "pseudo_verified_warp_render_l1_max": args.pseudo_verified_warp_render_l1_max,
        "pseudo_verified_neighborhood_kernel": args.pseudo_verified_neighborhood_kernel,
        "pseudo_verified_neighborhood_ratio": args.pseudo_verified_neighborhood_ratio,
        "pseudo_degraded_accept_l1_max": args.pseudo_degraded_accept_l1_max,
        "pseudo_degraded_accept_max_unsupported_l1": args.pseudo_degraded_accept_max_unsupported_l1,
        "selection_min_probe_verified_coverage": args.selection_min_probe_verified_coverage,
        "phase2_probe_min_verified_coverage": args.phase2_probe_min_verified_coverage,
    }
    for _override_name, _override_value in ablation_threshold_overrides.items():
        if _override_value is not None:
            if _override_name == "pseudo_verified_neighborhood_kernel":
                setattr(config, _override_name, int(_override_value))
            else:
                setattr(config, _override_name, float(_override_value))
    if world_rank == 0 and any(value is not None for value in ablation_threshold_overrides.values()):
        print(
            "[AblationThresholds] "
            + ", ".join(
                f"{name}={getattr(config, name)}"
                for name, value in ablation_threshold_overrides.items()
                if value is not None
            )
        )

    mode = args.mode
    is_stage_only = mode in {"stage1", "stage2b", "stage3"}
    is_pipeline = mode == "pipeline"
    require_gddn = is_stage_only or is_pipeline or args.estimator == "gddn"
    gddn_model: Optional[nn.Module]
    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    gddn_torch_dtype = dtype_map.get(args.gddn_dtype, torch.float16)
    if require_gddn:
        from examples.active_sampling.gddn_controlnet import GDDN_ControlNet

        if world_rank == 0:
            print(f"[GDDN] 使用论文 ControlNet backbone: {args.gddn_model_path}")
        gddn_model = GDDN_ControlNet(
            pretrained_model_path=args.gddn_model_path,
            dropout_rate=config.dropout_rate,
            torch_dtype=gddn_torch_dtype,
            use_depth_estimation=True,
        ).to(device)

        # 加载离线训练的.pt checkpoint（冻结生成先验权重）
        gddn_resume_ckpt = getattr(args, "gddn_resume_checkpoint", None)
        if gddn_resume_ckpt:
            if world_rank == 0:
                print(f"[GDDN] 加载离线训练checkpoint: {gddn_resume_ckpt}")
            ckpt_state = torch.load(gddn_resume_ckpt, map_location="cpu")
            checkpoint_report = load_state_dict_with_whitelist(
                gddn_model,
                ckpt_state,
                report_prefix="[GDDN]",
            )
            if world_rank == 0:
                missing_keys = checkpoint_report.missing_keys
                unexpected_keys = checkpoint_report.unexpected_keys
                print(f"[GDDN] checkpoint加载完成: missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
                if missing_keys:
                    print(f"[GDDN] missing keys (前5个): {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"[GDDN] unexpected keys (前5个): {unexpected_keys[:5]}")
                # 关键模块权重验证: 检查核心组件是否被有效加载
                _runtime_loaded_prefixes = [
                    "condition_preprocessor._depth_model.",
                ]
                _critical_missing_keys = [
                    key for key in missing_keys
                    if not any(key.startswith(prefix) for prefix in _runtime_loaded_prefixes)
                ]
                _critical_prefixes = [
                    "pose_encoder.", "controlnet.", "condition_preprocessor.",
                    "uncertainty_head.", "unet.", "vae.",
                ]
                _critical_missing = {}
                for prefix in _critical_prefixes:
                    _cnt = sum(1 for k in _critical_missing_keys if k.startswith(prefix))
                    if _cnt > 0:
                        _critical_missing[prefix.rstrip(".")] = _cnt
                _runtime_loaded_missing = sum(
                    1
                    for key in missing_keys
                    if any(key.startswith(prefix) for prefix in _runtime_loaded_prefixes)
                )
                if _runtime_loaded_missing > 0:
                    print(
                        f"[GDDN] note: {_runtime_loaded_missing} 个DepthAnything运行时模块参数不在离线checkpoint中，"
                        "将由外部权重单独初始化"
                    )
                if _critical_missing:
                    print(f"[GDDN] ⚠️ WARNING: 关键模块权重缺失!")
                    for mod_name, cnt in _critical_missing.items():
                        print(f"[GDDN]   {mod_name}: {cnt} keys missing (随机初始化)")
                    print(f"[GDDN]   这可能导致生成质量严重下降!")
                else:
                    print(f"[GDDN] ✓ 所有关键模块权重已成功加载")
        if scheduler_override_dict:
            gddn_model.apply_scheduler_overrides(scheduler_override_dict)

        # DDP Wrapping
        if world_size > 1:
            gddn_model = DDP(gddn_model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        gddn_model = None

    stage3_has_executed = False

    def _run_stage(stage_name: str, *, epochs: int, lr: float) -> None:
        nonlocal stage3_has_executed
        if gddn_model is None:
            raise ValueError("GDDN 模型未初始化，无法执行阶段训练。")
        resolution = STAGE_RESOLUTION.get(stage_name, (height, width))
        stage_provider = build_condition_provider(
            gaussians=gaussians,
            image_height=resolution[0],
            image_width=resolution[1],
            device=device,
            mode="gddn",
            existing_cameras=existing_cameras,
            reference_image_height=int(sparse_bundle.images.shape[-2]),
            reference_image_width=int(sparse_bundle.images.shape[-1]),
        )
        train_loader, val_loader = build_stage_dataloaders(
            existing_cameras=existing_cameras,
            real_images=real_images,
            sparse_bundle=sparse_bundle,
            condition_provider=stage_provider,
            gddn_model=gddn_model,
            device=device,
            train_resolution=resolution,
            batch_size=args.stage_batch_size,
            num_workers=args.stage_num_workers,
            world_size=world_size,
            world_rank=world_rank,
        )
        trainer = ThreeStageTrainer(
            model=gddn_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device=str(device),
            writer=writer,
            world_rank=world_rank,
            world_size=world_size,
            log_dir=args.log_dir,
        )
        trainer.sampler_type = sampler_type_override or args.mc_sampler_type
        trainer.scheduler_overrides = dict(scheduler_override_dict) if scheduler_override_dict else None
        trainer.scheduler_eta = scheduler_eta_value
        trainer.unfreeze_vae = bool(args.stage3_unfreeze_vae)
        trainer.unfreeze_vae_decoder_only = bool(
            args.stage3_unfreeze_vae and not args.stage3_unfreeze_vae_full
        )
        trainer.stage3_vae_lr_scale = max(args.stage3_vae_lr_scale, 0.0)
        trainer.auto_unfreeze_vae = bool(args.stage3_auto_unfreeze)
        trainer.auto_unfreeze_decoder_only = bool(
            args.stage3_auto_unfreeze and not args.stage3_auto_unfreeze_full
        )
        if trainer.unfreeze_vae and trainer.auto_unfreeze_vae and world_rank == 0:
            print("[Stage3] 已同时指定手动与自动解冻，默认优先采用手动配置并关闭自动解冻。")
            trainer.auto_unfreeze_vae = False
        trainer.auto_unfreeze_ece_threshold = max(args.stage3_auto_ece_threshold, 0.0)
        trainer.auto_unfreeze_corr_threshold = args.stage3_auto_corr_threshold
        trainer.auto_unfreeze_psnr_delta = max(args.stage3_auto_psnr_delta, 0.0)
        trainer.auto_unfreeze_warmup_epochs = max(args.stage3_auto_warmup_epochs, 0)
        trainer.auto_unfreeze_patience = max(args.stage3_auto_patience, 1)
        trainer.auto_unfreeze_psnr_window = max(
            args.stage3_auto_psnr_window,
            trainer.auto_unfreeze_patience,
        )
        trainer.stage3_recon_weight = max(args.stage3_recon_weight, 0.0)
        trainer.stage3_recon_alpha = float(min(max(args.stage3_recon_alpha, 0.0), 1.0))
        trainer.stage3_perceptual_weight = max(args.stage3_perc_weight, 0.0)
        trainer.stage3_variance_weight = max(args.stage3_variance_weight, 0.0)
        trainer.stage3_variance_target = max(args.stage3_variance_target, 0.0)
        trainer.stage3_variance_source = args.stage3_variance_source
        if stage_name == "stage1":
            perc_layers_raw = [item.strip() for item in args.stage1_perc_layers.split(",")] if args.stage1_perc_layers else []
            perc_layers = [item for item in perc_layers_raw if item]
            if not perc_layers:
                perc_layers_opt: Optional[Sequence[str]] = None
            else:
                perc_layers_opt = perc_layers
            perc_weights_opt: Optional[Sequence[float]]
            try:
                perc_weights_parsed = [
                    float(item.strip()) for item in args.stage1_perc_weights.split(",")
                    if item.strip()
                ]
                perc_weights_opt = perc_weights_parsed if perc_layers_opt and len(perc_weights_parsed) == len(perc_layers_opt) else None
            except Exception:
                perc_weights_opt = None
            trainer.stage1_pretrain(
                num_epochs=epochs,
                lr=lr,
                use_perceptual=not args.stage1_no_perc,
                lambda_perceptual=args.stage1_lambda_perc,
                perceptual_layers=perc_layers_opt,
                perceptual_weights=perc_weights_opt,
                use_consistency=not args.stage1_no_consistency,
                lambda_consistency=args.stage1_lambda_consistency,
                consistency_depth_type=args.stage1_depth_type,
                pose_angle_threshold=args.stage1_pose_angle_threshold,
                min_pose_similarity=args.stage1_min_sim_threshold,
                recon_alpha=args.stage1_recon_alpha,
                ddim_steps=max(1, args.stage1_ddim_steps),
                ddim_interval=max(1, args.stage1_ddim_interval),
                x0_switch_step=max(0, args.stage1_x0_switch_steps),
                adaptive_trigger=max(0.0, args.stage1_adaptive_trigger),
            )
            if stage1_export_dir is not None and world_rank == 0:
                _export_stage_checkpoint(gddn_model, stage1_export_dir)
        elif stage_name == "stage2b":
            # 对齐 Stage1 的位姿门控设置
            # DDP 模式下需要通过 module 访问
            raw_model = gddn_model.module if hasattr(gddn_model, "module") else gddn_model
            if hasattr(raw_model, "view_aggregator") and hasattr(raw_model.view_aggregator, "set_pose_gate"):
                try:
                    raw_model.view_aggregator.set_pose_gate(
                        angle_threshold_deg=args.stage1_pose_angle_threshold,
                        min_similarity=args.stage1_min_sim_threshold,
                    )
                except Exception:
                    pass
            trainer.stage2_mc_samples = max(1, args.stage2_mc_samples)
            trainer.stage2_diffusion_steps = max(1, args.stage2_steps)
            trainer.stage2_calibrate(num_epochs=epochs, lr=lr)
            if stage2_export_dir is not None and world_rank == 0:
                _export_stage_checkpoint(gddn_model, stage2_export_dir)
        elif stage_name == "stage3":
            trainer.stage3_mc_samples = max(1, args.stage3_mc_samples)
            trainer.stage3_diffusion_steps = max(1, args.stage3_steps)
            trainer.lambda_calib = args.stage3_lambda_calib
            trainer.distill_interval = max(1, args.stage3_distill_interval)
            trainer.stage3_final_lr = args.stage3_final_lr if args.stage3_final_lr > 0.0 else None
            trainer.stage3_final_epochs = max(0, args.stage3_final_epochs)
            trainer.stage3_finetune(
                num_epochs=epochs,
                lr=lr,
                lambda_distill=args.stage3_lambda_distill,
                recon_weight=trainer.stage3_recon_weight,
                recon_alpha=trainer.stage3_recon_alpha,
                perceptual_weight=trainer.stage3_perceptual_weight,
            )
            if stage3_export_dir is not None and world_rank == 0:
                _export_stage_checkpoint(gddn_model, stage3_export_dir)
        else:
            raise ValueError(f"不支持的阶段名称：{stage_name}")
        if stage_name == "stage3":
            stage3_has_executed = True

    def _maybe_run_quick_stage3() -> None:
        nonlocal stage3_has_executed
        if args.quick_stage3_epochs <= 0:
            return
        if stage3_has_executed:
            return
        if gddn_model is None:
            return
        if world_rank == 0:
            print(
                "[QuickStage3] 触发附加微调："
                f"epochs={args.quick_stage3_epochs}, lr={args.quick_stage3_lr:.3e}"
            )
        _run_stage("stage3", epochs=args.quick_stage3_epochs, lr=args.quick_stage3_lr)

    if is_stage_only or is_pipeline:
        # 最佳顺序：Stage1 → Stage2b 后立刻进入主动采样；可选开启 Stage3
        if is_stage_only:
            stages_sequence = [mode]
        else:
            stages_sequence = ["stage1", "stage2b"]
            if args.pipeline_run_stage3:
                stages_sequence.append("stage3")
        stage_name_map = {
            "stage1": "Stage1_Pretrain",
            "stage2b": "Stage2b_Calibrate",
            "stage3": "Stage3_Finetune",
        }
        for stage_name in stages_sequence:
            stage_label = stage_name_map.get(stage_name, stage_name)
            stage_start_time = time.time()
            if world_rank == 0:
                print(f"[Pipeline] {stage_label} start.")
            if stage_name == "stage1":
                _run_stage("stage1", epochs=args.stage1_epochs, lr=args.stage1_lr)
            elif stage_name == "stage2b":
                _run_stage("stage2b", epochs=args.stage2_epochs, lr=args.stage2_lr)
            else:
                _run_stage("stage3", epochs=args.stage3_epochs, lr=args.stage3_lr)
            stage_elapsed = time.time() - stage_start_time
            if world_rank == 0:
                print(f"[Pipeline] {stage_label} finished. elapsed={stage_elapsed:.2f}s")
                if writer is not None:
                    writer.add_scalar(f"pipeline/{stage_label}_seconds", stage_elapsed)
        _maybe_run_quick_stage3()
        # GDDN 训练结束，切换 eval
        gddn_model.eval()
        if not is_pipeline:
            if writer is not None:
                writer.flush()
                writer.close()
            return
        config.estimator_mode = "gddn"
    else:
        _maybe_run_quick_stage3()

    # Stage 训练结束后同步一次，确保所有 Rank 完成；管线模式下随后切换至单Rank训练
    dist_ctx = DistributedActiveSamplingContext(
        enabled=world_size > 1 and dist.is_available() and dist.is_initialized(),
        rank=world_rank,
        world_size=world_size,
    )
    if dist_ctx.enabled:
        dist.barrier()

    online_single_rank_mode = bool(dist_ctx.enabled and mode in {"active", "pipeline"})
    online_single_rank_done_path: Optional[Path] = None
    active_dist_ctx = dist_ctx
    if online_single_rank_mode:
        online_single_rank_done_path = _resolve_online_single_rank_done_path(args)
        if dist_ctx.is_main:
            online_single_rank_done_path.parent.mkdir(parents=True, exist_ok=True)
            if online_single_rank_done_path.exists():
                online_single_rank_done_path.unlink()
        if dist_ctx.enabled:
            dist.barrier()
        if not dist_ctx.is_main:
            _wait_for_online_single_rank_completion(
                online_single_rank_done_path,
                world_rank=world_rank,
            )
            if writer is not None:
                writer.flush()
                writer.close()
            return
        active_dist_ctx = DistributedActiveSamplingContext()

    runtime_gddn: Optional[nn.Module] = gddn_model
    if runtime_gddn is not None and hasattr(runtime_gddn, "module"):
        runtime_gddn = runtime_gddn.module  # type: ignore[assignment]

    def _pseudo_view_to_cpu(pseudo_view: PseudoView) -> PseudoView:
        cpu_camera = ViewCandidate(
            position=pseudo_view.camera.position.detach().cpu(),
            viewmat=pseudo_view.camera.viewmat.detach().cpu(),
            intrinsics=pseudo_view.camera.intrinsics.detach().cpu(),
        )
        cpu_proposal_patches = [
            ProposalPatch(
                bbox_xyxy=tuple(proposal_patch.bbox_xyxy),
                image=proposal_patch.image.detach().cpu(),
                mask=proposal_patch.mask.detach().cpu(),
                score=float(proposal_patch.score),
                coverage=float(proposal_patch.coverage),
                frontier_coverage=float(proposal_patch.frontier_coverage),
            )
            for proposal_patch in getattr(pseudo_view, "proposal_patches", [])
        ]
        return PseudoView(
            camera=cpu_camera,
            image=pseudo_view.image.detach().cpu(),
            epistemic_map=pseudo_view.epistemic_map.detach().cpu() if pseudo_view.epistemic_map is not None else None,
            aleatoric_map=pseudo_view.aleatoric_map.detach().cpu() if pseudo_view.aleatoric_map is not None else None,
            risk_exist_map=pseudo_view.risk_exist_map.detach().cpu() if pseudo_view.risk_exist_map is not None else None,
            gain_novel_map=pseudo_view.gain_novel_map.detach().cpu() if pseudo_view.gain_novel_map is not None else None,
            exist_mask=pseudo_view.exist_mask.detach().cpu() if pseudo_view.exist_mask is not None else None,
            novel_mask=pseudo_view.novel_mask.detach().cpu() if pseudo_view.novel_mask is not None else None,
            frontier_mask=pseudo_view.frontier_mask.detach().cpu() if pseudo_view.frontier_mask is not None else None,
            supported_exist_mask=(
                pseudo_view.supported_exist_mask.detach().cpu()
                if pseudo_view.supported_exist_mask is not None
                else None
            ),
            verified_evidence_mask=(
                pseudo_view.verified_evidence_mask.detach().cpu()
                if pseudo_view.verified_evidence_mask is not None
                else None
            ),
            verified_evidence_coverage=float(getattr(pseudo_view, "verified_evidence_coverage", 0.0)),
            verified_evidence_score=float(getattr(pseudo_view, "verified_evidence_score", 0.0)),
            unsupported_novel_mask=(
                pseudo_view.unsupported_novel_mask.detach().cpu()
                if pseudo_view.unsupported_novel_mask is not None
                else None
            ),
            score=float(pseudo_view.score),
            q_gen=float(pseudo_view.q_gen),
            q_gain=float(pseudo_view.q_gain),
            q_gain_for_gate=float(pseudo_view.q_gain_for_gate),
            predicted_residual_delta_psnr=float(pseudo_view.predicted_residual_delta_psnr),
            consistency_weight=float(pseudo_view.consistency_weight),
            width=int(pseudo_view.width),
            height=int(pseudo_view.height),
            img_name=str(pseudo_view.img_name),
            proposal_patches=cpu_proposal_patches,
            plan6_rgb_trust_map=(
                pseudo_view.plan6_rgb_trust_map.detach().cpu()
                if pseudo_view.plan6_rgb_trust_map is not None
                else None
            ),
            plan6_depth_trust_map=(
                pseudo_view.plan6_depth_trust_map.detach().cpu()
                if pseudo_view.plan6_depth_trust_map is not None
                else None
            ),
            plan6_alpha_trust_map=(
                pseudo_view.plan6_alpha_trust_map.detach().cpu()
                if pseudo_view.plan6_alpha_trust_map is not None
                else None
            ),
            plan6_phase3_rgb_weight=(
                pseudo_view.plan6_phase3_rgb_weight.detach().cpu()
                if pseudo_view.plan6_phase3_rgb_weight is not None
                else None
            ),
            plan6_phase3_depth_weight=(
                pseudo_view.plan6_phase3_depth_weight.detach().cpu()
                if pseudo_view.plan6_phase3_depth_weight is not None
                else None
            ),
            plan6_phase3_alpha_weight=(
                pseudo_view.plan6_phase3_alpha_weight.detach().cpu()
                if pseudo_view.plan6_phase3_alpha_weight is not None
                else None
            ),
            plan6_joint_trust_mask=(
                pseudo_view.plan6_joint_trust_mask.detach().cpu()
                if pseudo_view.plan6_joint_trust_mask is not None
                else None
            ),
            plan6_proposal_depth=(
                pseudo_view.plan6_proposal_depth.detach().cpu()
                if pseudo_view.plan6_proposal_depth is not None
                else None
            ),
            plan6_proposal_alpha=(
                pseudo_view.plan6_proposal_alpha.detach().cpu()
                if pseudo_view.plan6_proposal_alpha is not None
                else None
            ),
            plan6_online_admission_allowed=bool(pseudo_view.plan6_online_admission_allowed),
            plan6_gate_reasons=list(pseudo_view.plan6_gate_reasons),
            plan6_stage_metrics=dict(pseudo_view.plan6_stage_metrics),
        )

    condition_provider = build_condition_provider(
        gaussians=gaussians,
        image_height=height,
        image_width=width,
        device=device,
        mode=config.estimator_mode,
        existing_cameras=existing_cameras,
        reference_image_height=int(sparse_bundle.images.shape[-2]),
        reference_image_width=int(sparse_bundle.images.shape[-1]),
    )

    pseudo_save_dir: Optional[Path]
    if args.save_pseudo_png and args.log_dir:
        pseudo_save_dir = Path(args.log_dir) / "pseudo_views"
    else:
        pseudo_save_dir = None

    candidate_log_path: Optional[Path]
    if args.candidate_log:
        candidate_log_path = Path(args.candidate_log)
    else:
        candidate_log_path = None

    active_writer = writer if world_rank == 0 else None

    # 将pseudo_mode从命令行传递到config
    config.pseudo_mode = getattr(args, 'pseudo_mode', 'guidance_only')

    # ========== Phase 1: 纯3DGS预训练（收敛检测自动退出） ==========
    phase1_iters = int(config.phase1_max_iterations)
    print(f"\n{'='*60}")
    print(f"[Pipeline] Phase 1: 纯3DGS预训练 (最大 {phase1_iters} 步)")
    print(f"[Pipeline] 收敛条件: loss_delta<{config.phase1_convergence_threshold} 或 PSNR≥{config.phase1_convergence_psnr}")
    print(f"{'='*60}\n")

    pseudo_cameras = train_3dgs_with_active_sampling(
        gaussians=gaussians,
        real_cameras=existing_cameras,
        real_images=real_images,
        gddn_model=runtime_gddn,
        num_iterations=phase1_iters,
        sparse_bundle=sparse_bundle,
        config=config,
        scene_center=scene_center,
        condition_provider=condition_provider,
        writer=active_writer,
        stability_strategy=args.stability_strategy,
        pseudo_save_dir=pseudo_save_dir,
        candidate_log_path=candidate_log_path,
        dist_context=active_dist_ctx,
        current_phase=1,
    )

    # 保存Phase 1 checkpoint
    phase1_state = field_state(gaussians)
    print(f"[Pipeline] Phase 1完成: 高斯体={gaussians.means.shape[0]}")

    # ========== 自动保存 Phase 1 结果（供 eval_uncertainty_correlation.py 使用）==========
    if args.log_dir:
        _eval_save_path = Path(args.log_dir) / "phase1_result.pt"
        try:
            # DDUD: count_accum/grad2d_accum 由 simple_trainer 在 Phase 1 收敛时挂载到 gaussians
            _ddud_count = getattr(gaussians, "_ddud_count_accum", None)
            _ddud_grad = getattr(gaussians, "_ddud_grad2d_accum", None)
            _eval_data = {
                "gaussians_state": phase1_state,
                "training_state": getattr(gaussians, "_training_state", None),
                "count_accum": _ddud_count.cpu().clone() if _ddud_count is not None else torch.zeros(gaussians.means.shape[0]),
                "grad2d_accum": _ddud_grad.cpu().clone() if _ddud_grad is not None else torch.zeros(gaussians.means.shape[0]),
            }
            if _ddud_count is not None:
                print(f"[DDUD-Eval] count_accum: mean={_ddud_count.float().mean():.1f}, "
                      f"max={_ddud_count.max().item()}, nonzero={(_ddud_count > 0).sum().item()}")
            else:
                print("[DDUD-Eval] 警告: count_accum 未附加到 gaussians，值将为全零")
            torch.save(_eval_data, _eval_save_path)
            print(f"[DDUD-Eval] Phase 1 数据已保存: {_eval_save_path}")
            print(f"[DDUD-Eval] 可运行: python eval_uncertainty_correlation.py "
                  f"--log-dir {args.log_dir} --data-dir {args.data_dir}")
        except Exception as _e:
            print(f"[DDUD-Eval] 保存 phase1_result.pt 失败: {_e}")



    # ========== Phase 2: MC-Dropout主动采样 + GDDN伪视图生成 ==========
    phase2_save_dir = Path(args.log_dir) / "phase2_pseudo_views" if args.log_dir else None
    print(f"\n{'='*60}")
    print(f"[Pipeline] Phase 2: MC-Dropout主动采样 ({config.phase2_rounds} 轮)")
    print(f"[Pipeline] 每轮选 {config.num_views_per_round} 个视角")
    print(f"{'='*60}\n")

    if runtime_gddn is not None:
        # DDUD: 将Phase 1的count_accum附加到gaussians对象
        # estimator通过self.gaussians._ddud_count_accum访问精确的U_geo计算
        # count_accum[i] = 第i个高斯体在Phase 1中被观测的累计次数
        # 观测少 → 视角覆盖稀疏 → U_geo高 → 几何信息增益大
        phase1_count_accum = getattr(gaussians, "_ddud_count_accum", None)
        if phase1_count_accum is not None:
            gaussians._ddud_count_accum = phase1_count_accum.detach().clone()
            print(f"[DDUD] Phase 1 count_accum已同步到gaussians: "
                  f"mean={phase1_count_accum.float().mean():.1f}, "
                  f"min={phase1_count_accum.min().item()}, max={phase1_count_accum.max().item()}")
        else:
            print("[DDUD] 警告: Phase 1 未找到 _ddud_count_accum，DDUD将退化为coverage代理")
        if not active_dist_ctx.enabled or active_dist_ctx.is_main:
            generated_pseudo_views = run_phase2_active_sampling(
                gaussians=gaussians,
                real_cameras=existing_cameras,
                gddn_model=runtime_gddn,
                sparse_bundle=sparse_bundle,
                config=config,
                scene_center=scene_center,
                condition_provider=condition_provider,
                writer=active_writer,
                pseudo_save_dir=phase2_save_dir,
            )
            generated_pseudo_views = [_pseudo_view_to_cpu(pseudo_view) for pseudo_view in generated_pseudo_views]
        else:
            generated_pseudo_views = []

        if active_dist_ctx.enabled:
            pseudo_view_payload = [generated_pseudo_views if dist_ctx.is_main else None]
            dist.broadcast_object_list(pseudo_view_payload, src=0)
            generated_pseudo_views = pseudo_view_payload[0] if pseudo_view_payload[0] is not None else []
    else:
        generated_pseudo_views = []
        print("[Pipeline] 无GDDN模型，跳过Phase 2")

    # ========== Phase 3: 伪视图增强3DGS训练 ==========
    phase3_iters = int(config.phase3_iterations)
    print(f"\n{'='*60}")
    print(f"[Pipeline] Phase 3: 伪视图增强训练 ({phase3_iters} 步, 模式={config.pseudo_mode})")
    print(f"[Pipeline] 伪视图数量: {len(generated_pseudo_views)}")
    print(f"{'='*60}\n")

    # 从Phase 1 checkpoint恢复（Phase 2可能修改了gaussians状态）
    load_field(gaussians, phase1_state, device)

    pseudo_cameras = train_3dgs_with_active_sampling(
        gaussians=gaussians,
        real_cameras=existing_cameras,
        real_images=real_images,
        gddn_model=runtime_gddn,
        num_iterations=phase3_iters,
        sparse_bundle=sparse_bundle,
        config=config,
        scene_center=scene_center,
        condition_provider=condition_provider,
        writer=active_writer,
        stability_strategy=args.stability_strategy,
        pseudo_save_dir=pseudo_save_dir,
        candidate_log_path=candidate_log_path,
        dist_context=active_dist_ctx,
        pseudo_views=generated_pseudo_views if generated_pseudo_views else None,
        current_phase=3,
    )
    if world_rank == 0 and args.gaussian_output:
        output_path = Path(args.gaussian_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        state = field_state(gaussians)
        torch.save(state, output_path)
        print(f"[Active] 高斯场已保存至 {output_path}")
    
    # 导出3DGS兼容格式
    if args.output_3dgs_format and world_rank == 0:
        output_dir = Path(args.log_dir) if args.log_dir else Path("./output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. 导出cameras.json（真实视角+伪视角）
        export_cameras_json(
            real_cameras=existing_cameras,
            pseudo_cameras=pseudo_cameras if pseudo_cameras else [],
            width=width,
            height=height,
            save_path=str(output_dir / "cameras.json"),
        )
        
        # 2. 导出input.ply（COLMAP输入点云）
        export_input_ply(
            points=points,
            colors=colors,
            save_path=str(output_dir / "input.ply"),
        )
        
        # 3. 导出point_cloud.ply（最终高斯体）
        ply_dir = output_dir / "point_cloud" / "final"
        ply_dir.mkdir(parents=True, exist_ok=True)
        export_gaussians_to_ply(
            gaussians=gaussians,
            save_path=str(ply_dir / "point_cloud.ply"),
        )
        print(f"[Export] 3DGS兼容格式已保存至: {output_dir}")

    if online_single_rank_done_path is not None and world_rank == 0:
        done_payload = {
            "world_rank": int(world_rank),
            "world_size": int(world_size),
            "mode": str(mode),
            "gaussian_output": str(args.gaussian_output) if args.gaussian_output else "",
            "completed_at_unix_seconds": float(time.time()),
        }
        with online_single_rank_done_path.open("w", encoding="utf-8") as file_handle:
            json.dump(done_payload, file_handle, ensure_ascii=False, indent=2)
    
    if writer is not None and world_rank == 0:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    args = parse_arguments()
    if torch.cuda.is_available():
        cli(main_worker, args, verbose=True)
    else:
        # CPU Fallback，不使用多进程
        main_worker(local_rank=0, world_rank=0, world_size=1, args=args)
