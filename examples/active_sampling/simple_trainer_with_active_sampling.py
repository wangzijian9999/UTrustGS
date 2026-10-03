"""
Example training loop integrating validation-active sampling with gsplat.

This script illustrates how the modules in `examples/active_sampling` may be
composed. It relies on placeholder functions for rendering and optimisation so
that users can adapt them to their own pipelines.
"""

from __future__ import annotations

import math
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from dataclasses import dataclass, field
from enum import IntEnum

# 添加 Depth-Anything-V2 到 Python 路径以支持 DACD
_depth_anything_path = Path(__file__).parent.parent / "models" / "Depth-Anything-V2"
if _depth_anything_path.exists() and str(_depth_anything_path) not in sys.path:
    sys.path.insert(0, str(_depth_anything_path))

import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image

from gsplat.rendering import rasterization  # type: ignore
from gsplat.exporter import export_splats  # PLY/SPLAT导出
from examples.active_sampling.gaussian_parameter_contract import (
    PARAMETERS, accumulate_density_gradients, capture_training_state, field_state, load_field, ply_parameters,
    project_field, refine_field, replace_rows, restore_training_state, validate_field,
)

from examples.active_sampling.active_selector import ActiveSelector
from examples.active_sampling.configs.active_sampling_config import ActiveSamplingConfig
from examples.active_sampling.soft_gating import (
    RobustHeteroscedasticLoss,
    SoftGatedGuidance,
)  # 软门控机制 & 鲁棒NLL
from examples.active_sampling.uncertainty_estimator import (
    QGainCalibrator,
    SparseViewBundle,
    UncertaintyEstimator,
    UncertaintyResult,
)
from examples.active_sampling.utils import ViewCandidate
from examples.active_sampling.viewpoint_sampler import ExistingCamera, ViewpointSampler
from examples.active_sampling.validation import (
    bootstrap_ensemble_scores,
)
from examples.active_sampling.where_wrong_four_stage_protocol import (
    FourStageConfig,
    FourStageResult,
    run_four_stage_protocol,
)

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore


_Q_GAIN_CALIBRATOR_CACHE: Dict[str, Optional[QGainCalibrator]] = {}


def compute_dynamic_lambda(
    iteration: int,
    config: ActiveSamplingConfig
) -> float:
    """
    动态λ调度器 - 三阶段训练策略（模块四）

    实现理论设计中的三阶段动态权重调整：
    - 阶段1 (Warmup, 0 → warmup_iter):  λ = 0
    - 阶段2 (Ramp-up, warmup_iter → peak_iter): λ = lambda_min → lambda_max (线性增长)
    - 阶段3 (Decay, peak_iter → end_iter): λ = lambda_max → lambda_min (线性衰减)

    Args:
        iteration: 当前训练迭代步数
        config: 包含调度参数的配置对象

    Returns:
        当前迭代对应的λ值 ∈ [0, lambda_max]

    Examples:
        >>> config = ActiveSamplingConfig(
        ...     warmup_iterations=5000,
        ...     lambda_min=0.05,
        ...     lambda_max=0.2,
        ...     lambda_peak_iteration=25000,
        ...     lambda_end_iteration=30000
        ... )
        >>> compute_dynamic_lambda(0, config)      # 0.0 (阶段1)
        >>> compute_dynamic_lambda(5000, config)   # 0.05 (阶段2起始)
        >>> compute_dynamic_lambda(15000, config)  # 0.125 (阶段2中期)
        >>> compute_dynamic_lambda(25000, config)  # 0.2 (阶段2峰值/阶段3起始)
        >>> compute_dynamic_lambda(27500, config)  # 0.125 (阶段3中期)
        >>> compute_dynamic_lambda(30000, config)  # 0.05 (阶段3结束)
    """
    if not config.lambda_dynamic_enable:
        # 动态调整未启用，返回固定值
        return config.lambda_guidance

    warmup = config.warmup_iterations
    peak = config.lambda_peak_iteration
    end = config.lambda_end_iteration
    lambda_min = config.lambda_min
    lambda_max = config.lambda_max

    # 阶段1：Warmup阶段，禁用引导
    if iteration < warmup:
        return 0.0

    # 阶段2：渐进式引入引导（线性增长）
    elif iteration < peak:
        progress = (iteration - warmup) / max(peak - warmup, 1)
        lambda_current = lambda_min + (lambda_max - lambda_min) * progress
        return float(lambda_current)

    # 阶段3：精细化调整（线性衰减）
    elif iteration < end:
        progress = (iteration - peak) / max(end - peak, 1)
        lambda_current = lambda_max - (lambda_max - lambda_min) * progress
        return float(lambda_current)

    # 超出总迭代数后，保持最小值
    else:
        return lambda_min


def get_current_phase(iteration: int, config: ActiveSamplingConfig) -> tuple[str, int]:
    """
    获取当前训练阶段名称和对应的整数编码（用于TensorBoard可视化）

    Returns:
        (phase_name, phase_int): 阶段名称和整数编码(0/1/2)
    """
    warmup = config.warmup_iterations
    peak = config.lambda_peak_iteration
    end = config.lambda_end_iteration

    if iteration < warmup:
        return ("scaffold", 0)
    elif iteration < peak:
        return ("guided", 1)
    elif iteration < end:
        return ("refinement", 2)
    else:
        return ("completed", 3)


ConditionProvider = Callable[[Iterable[ViewCandidate]], Dict[str, torch.Tensor]]


@dataclass
class DistributedActiveSamplingContext:
    enabled: bool = False
    rank: int = 0
    world_size: int = 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


class ActiveSamplingCommand(IntEnum):
    SYNC_GAUSSIANS = 1
    EVALUATE = 2
    SHUTDOWN = 3
    BROADCAST_DATA = 4  # 用于广播候选与选择索引


def _command_tensor(value: int = 0) -> torch.Tensor:
    device = torch.device("cpu")
    if dist.is_available() and dist.is_initialized():
        try:
            backend = dist.get_backend()
        except Exception:  # pragma: no cover - backend查询失败时退化为CPU
            backend = None
        if backend == dist.Backend.NCCL or backend == "nccl":
            device = torch.device("cuda", torch.cuda.current_device())
    return torch.tensor([int(value)], dtype=torch.long, device=device)


def _broadcast_command(
    command: ActiveSamplingCommand, context: DistributedActiveSamplingContext
) -> None:
    if not context.enabled:
        return
    tensor = _command_tensor(int(command))
    dist.broadcast(tensor, src=0)


def _serialize_gaussian_state(gaussians: "GaussianPointField") -> Dict[str, Any]:
    return field_state(gaussians)


def _apply_gaussian_state(
    gaussians: "GaussianPointField", state: Dict[str, Any], device: torch.device
) -> None:
    load_field(gaussians, state, device)


def _serialize_candidate(candidate: ViewCandidate) -> Dict[str, torch.Tensor]:
    return {
        "position": candidate.position.detach().cpu(),
        "viewmat": candidate.viewmat.detach().cpu(),
        "intrinsics": candidate.intrinsics.detach().cpu(),
    }


def _deserialize_candidate(serialized: Dict[str, torch.Tensor], device: torch.device) -> ViewCandidate:
    return ViewCandidate(
        position=serialized["position"].to(device),
        viewmat=serialized["viewmat"].to(device),
        intrinsics=serialized["intrinsics"].to(device),
    )


def _partition_indices(total: int, world_size: int, rank: int) -> List[int]:
    if total <= 0:
        return []
    return list(range(rank, total, max(1, world_size)))


def _result_to_cpu(result: UncertaintyResult) -> UncertaintyResult:
    candidate = ViewCandidate(
        position=result.candidate.position.detach().cpu(),
        viewmat=result.candidate.viewmat.detach().cpu(),
        intrinsics=result.candidate.intrinsics.detach().cpu(),
    )
    metrics = result.metrics.copy() if isinstance(result.metrics, dict) else result.metrics
    return UncertaintyResult(
        candidate=candidate,
        mean_rgb=result.mean_rgb.detach().cpu(),
        variance_map=result.variance_map.detach().cpu(),
        score=float(result.score),
        unmasked_score=float(result.unmasked_score),
        metrics=metrics,
        epistemic_map=result.epistemic_map.detach().cpu() if result.epistemic_map is not None else None,
        aleatoric_map=result.aleatoric_map.detach().cpu() if result.aleatoric_map is not None else None,
        risk_exist_map=result.risk_exist_map.detach().cpu() if result.risk_exist_map is not None else None,
        gain_novel_map=result.gain_novel_map.detach().cpu() if result.gain_novel_map is not None else None,
        exist_mask=result.exist_mask.detach().cpu() if result.exist_mask is not None else None,
        novel_mask=result.novel_mask.detach().cpu() if result.novel_mask is not None else None,
        calibrated_depth=result.calibrated_depth.detach().cpu() if result.calibrated_depth is not None else None,
        mono_depth=result.mono_depth.detach().cpu() if result.mono_depth is not None else None,
    )


def _result_to_device(result: UncertaintyResult, device: torch.device) -> UncertaintyResult:
    candidate = ViewCandidate(
        position=result.candidate.position.to(device),
        viewmat=result.candidate.viewmat.to(device),
        intrinsics=result.candidate.intrinsics.to(device),
    )
    return UncertaintyResult(
        candidate=candidate,
        mean_rgb=result.mean_rgb.to(device),
        variance_map=result.variance_map.to(device),
        score=float(result.score),
        unmasked_score=float(result.unmasked_score),
        metrics=result.metrics,
        epistemic_map=result.epistemic_map.to(device) if result.epistemic_map is not None else None,
        aleatoric_map=result.aleatoric_map.to(device) if result.aleatoric_map is not None else None,
        risk_exist_map=result.risk_exist_map.to(device) if result.risk_exist_map is not None else None,
        gain_novel_map=result.gain_novel_map.to(device) if result.gain_novel_map is not None else None,
        exist_mask=result.exist_mask.to(device) if result.exist_mask is not None else None,
        novel_mask=result.novel_mask.to(device) if result.novel_mask is not None else None,
        calibrated_depth=result.calibrated_depth.to(device) if result.calibrated_depth is not None else None,
        mono_depth=result.mono_depth.to(device) if result.mono_depth is not None else None,
    )


# ========== 多卡并行同步辅助函数 ==========


# ========== 分布式同步辅助函数结束 ==========


def _sync_gaussians_to_workers(
    gaussians: "GaussianPointField", context: DistributedActiveSamplingContext
) -> None:
    if not context.enabled:
        return
    _broadcast_command(ActiveSamplingCommand.SYNC_GAUSSIANS, context)
    state = _serialize_gaussian_state(gaussians)
    dist.broadcast_object_list([state], src=0)


def _distributed_estimate_master(
    estimator: UncertaintyEstimator,
    sparse_bundle: SparseViewBundle,
    candidates: Sequence[ViewCandidate],
    *,
    global_step: Optional[int],
    num_samples: Optional[int],
    label: str,
    context: DistributedActiveSamplingContext,
) -> List[UncertaintyResult]:
    if not candidates:
        return []
    serialized = [_serialize_candidate(candidate) for candidate in candidates]
    task_payload = {
        "candidates": serialized,
        "label": label,
        "num_samples": num_samples,
        "global_step": global_step,
        "total": len(serialized),
    }
    _broadcast_command(ActiveSamplingCommand.EVALUATE, context)
    dist.broadcast_object_list([task_payload], src=0)
    local_indices = _partition_indices(len(serialized), context.world_size, context.rank)
    local_candidates = [
        _deserialize_candidate(serialized[idx], estimator.device) for idx in local_indices
    ]
    local_results: List[UncertaintyResult] = []
    if local_candidates:
        with torch.no_grad():
            local_results = estimator.estimate(
                sparse_bundle,
                local_candidates,
                global_step=global_step,
                num_samples=num_samples,
                label=label,
            )
    serializable = [
        (local_indices[idx_pos], _result_to_cpu(result)) for idx_pos, result in enumerate(local_results)
    ]
    gather_list: Optional[List[Optional[List[Tuple[int, UncertaintyResult]]]]] = None
    gather_list = [None for _ in range(context.world_size)]
    dist.gather_object(serializable, gather_list, dst=0)
    merged: List[Tuple[int, UncertaintyResult]] = []
    for chunk in gather_list:
        if not chunk:
            continue
        merged.extend(chunk)
    merged.sort(key=lambda item: item[0])
    return [_result_to_device(result, estimator.device) for _, result in merged]


def _estimate_candidates(
    estimator: UncertaintyEstimator,
    sparse_bundle: SparseViewBundle,
    candidates: Sequence[ViewCandidate],
    *,
    global_step: Optional[int],
    num_samples: Optional[int],
    label: str,
    dist_context: DistributedActiveSamplingContext,
) -> List[UncertaintyResult]:
    if not candidates:
        return []
    if dist_context.enabled and dist_context.is_main:
        return _distributed_estimate_master(
            estimator,
            sparse_bundle,
            candidates,
            global_step=global_step,
            num_samples=num_samples,
            label=label,
            context=dist_context,
        )
    with torch.no_grad():
        return estimator.estimate(
            sparse_bundle,
            candidates,
            global_step=global_step,
            num_samples=num_samples,
            label=label,
        )


def _run_distributed_worker_loop(
    *,
    estimator: UncertaintyEstimator,
    sparse_bundle: SparseViewBundle,
    dist_context: DistributedActiveSamplingContext,
) -> None:
    if not dist_context.enabled or dist_context.is_main:
        return
    device = estimator.device
    while True:
        cmd_tensor = _command_tensor(0)
        dist.broadcast(cmd_tensor, src=0)
        command = ActiveSamplingCommand(int(cmd_tensor.item()))
        if command == ActiveSamplingCommand.SHUTDOWN:
            break
        if command == ActiveSamplingCommand.SYNC_GAUSSIANS:
            payload = [None]
            dist.broadcast_object_list(payload, src=0)
            state = payload[0]
            if state is not None:
                _apply_gaussian_state(estimator.gaussians, state, device)
            continue
        if command != ActiveSamplingCommand.EVALUATE:
            raise RuntimeError(f"未知的主动采样命令: {command}")
        payload = [None]
        dist.broadcast_object_list(payload, src=0)
        task = payload[0]
        if task is None:
            dist.gather_object([], None, dst=0)
            continue
        serialized_candidates = task.get("candidates", [])
        label = task.get("label", "fine")
        num_samples = task.get("num_samples")
        global_step = task.get("global_step")
        total = int(task.get("total", len(serialized_candidates)))
        local_indices = _partition_indices(total, dist_context.world_size, dist_context.rank)
        local_candidates = [
            _deserialize_candidate(serialized_candidates[idx], estimator.device)
            for idx in local_indices
            if idx < len(serialized_candidates)
        ]
        local_results: List[UncertaintyResult] = []
        if local_candidates:
            with torch.no_grad():
                local_results = estimator.estimate(
                    sparse_bundle,
                    local_candidates,
                    global_step=global_step,
                    num_samples=num_samples,
                    label=label,
                )
        serializable = [
            (local_indices[idx_pos], _result_to_cpu(result)) for idx_pos, result in enumerate(local_results)
        ]
        dist.gather_object(serializable, None, dst=0)


def example_condition_provider(candidates: Iterable[ViewCandidate]) -> Dict[str, torch.Tensor]:
    """
    Placeholder renderer that returns dummy conditioning tensors.
    Replace with actual gsplat renders in production.
    """
    candidates = list(candidates)
    batch = len(candidates)
    height, width = 256, 256
    rgb = torch.rand(batch, 3, height, width)
    depth = torch.rand(batch, 1, height, width)
    normal = torch.rand(batch, 3, height, width)
    latent = torch.rand(batch, 4, height // 8, width // 8)
    return {
        "rgb_render": rgb,
        "depth_render": depth,
        "normal_render": normal,
        "target_latent": latent,
    }


def _save_rgb_tensor(path: Path, tensor: torch.Tensor) -> None:
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor.expand(3, -1, -1)
    array = (
        tensor.detach().clamp(0.0, 1.0).mul(255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    Image.fromarray(array).save(path)


def _save_map_tensor(path: Path, tensor: torch.Tensor) -> None:
    data = tensor.detach()
    # 确保张量是2D：移除所有单维度
    while data.dim() > 2:
        data = data.squeeze(0) if data.shape[0] == 1 else data.squeeze(-1)
    if data.dim() == 1:
        # 如果是1D，尝试推断为正方形
        side = int(data.numel() ** 0.5)
        if side * side == data.numel():
            data = data.view(side, side)
        else:
            # 无法推断，保存为1xN
            data = data.unsqueeze(0)
    data = data - data.min()
    data = data / data.max().clamp_min(1e-6)
    array = data.mul(255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(array).save(path)


def _create_comparison_image(
    gddn_rgb: torch.Tensor,
    gsplat_rgb: torch.Tensor,
    path: Path,
    labels: bool = True,
) -> None:
    """
    创建并排对比图：GDDN生成 | gsplat渲染
    
    Args:
        gddn_rgb: GDDN生成的RGB图像, shape [3, H, W]
        gsplat_rgb: gsplat渲染的RGB图像, shape [3, H, W]
        path: 保存路径
        labels: 是否添加标签文字
    """
    import numpy as np

    if gddn_rgb.dim() == 4:
        gddn_rgb = gddn_rgb[0]
    if gsplat_rgb.dim() == 4:
        gsplat_rgb = gsplat_rgb[0]
    if gddn_rgb.shape[-2:] != gsplat_rgb.shape[-2:]:
        gsplat_rgb = F.interpolate(
            gsplat_rgb.unsqueeze(0).float(),
            size=gddn_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )[0].to(dtype=gddn_rgb.dtype)

    # 转换为numpy数组
    gddn_arr = (
        gddn_rgb.detach().clamp(0.0, 1.0).mul(255.0)
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    gsplat_arr = (
        gsplat_rgb.detach().clamp(0.0, 1.0).mul(255.0)
        .to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    )
    
    # 创建分隔线
    H, W = gddn_arr.shape[:2]
    separator = np.ones((H, 4, 3), dtype=np.uint8) * 128  # 灰色分隔线
    
    # 拼接图像
    comparison = np.concatenate([gddn_arr, separator, gsplat_arr], axis=1)
    
    # 添加标签（可选）
    if labels:
        try:
            from PIL import ImageDraw, ImageFont
            img = Image.fromarray(comparison)
            draw = ImageDraw.Draw(img)
            # 尝试使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", 16)
            except Exception:
                font = ImageFont.load_default()
            draw.text((10, 10), "GDDN", fill=(255, 255, 0), font=font)
            draw.text((W + 14, 10), "gsplat", fill=(0, 255, 255), font=font)
            img.save(path)
            return
        except Exception:
            pass  # 如果添加标签失败，直接保存无标签图像
    
    Image.fromarray(comparison).save(path)


@dataclass
class ProposalPatch:
    """经几何验证后的局部伪证据patch。"""
    bbox_xyxy: Tuple[int, int, int, int]
    image: torch.Tensor
    mask: torch.Tensor
    score: float = 0.0
    coverage: float = 0.0
    frontier_coverage: float = 0.0


@dataclass
class PseudoView:
    """Phase 2生成的局部伪证据容器；整图image仅作调试与proposal来源。"""
    camera: "ViewCandidate"           # 视角候选
    image: torch.Tensor               # 伪视图RGB [3,H,W]
    epistemic_map: Optional[torch.Tensor] = None   # 认知不确定性图
    aleatoric_map: Optional[torch.Tensor] = None   # 数据不确定性图
    risk_exist_map: Optional[torch.Tensor] = None
    gain_novel_map: Optional[torch.Tensor] = None
    exist_mask: Optional[torch.Tensor] = None
    novel_mask: Optional[torch.Tensor] = None
    frontier_mask: Optional[torch.Tensor] = None
    supported_exist_mask: Optional[torch.Tensor] = None
    unsupported_novel_mask: Optional[torch.Tensor] = None
    verified_evidence_mask: Optional[torch.Tensor] = None
    verified_evidence_coverage: float = 0.0
    verified_evidence_score: float = 0.0
    score: float = 0.0                # MC-Dropout不确定性得分
    q_gen: float = 1.0                # DDUD维度2: 生成可信度 Q_gen=1/(1+σ²_MC) ∈(0,1]
    q_gain: float = 0.5               # 收益侧 q_gain，用于排序/权重
    q_gain_for_gate: float = 0.5      # 保守 gate 侧 q_gain，仅用于接纳/拒绝
    predicted_residual_delta_psnr: float = 0.0
    consistency_weight: float = 1.0   # Exist域一致性权重
    width: int = 0
    height: int = 0
    img_name: str = ""
    proposal_patches: List[ProposalPatch] = field(default_factory=list)
    plan6_rgb_trust_map: Optional[torch.Tensor] = None
    plan6_depth_trust_map: Optional[torch.Tensor] = None
    plan6_alpha_trust_map: Optional[torch.Tensor] = None
    plan6_phase3_rgb_weight: Optional[torch.Tensor] = None
    plan6_phase3_depth_weight: Optional[torch.Tensor] = None
    plan6_phase3_alpha_weight: Optional[torch.Tensor] = None
    plan6_joint_trust_mask: Optional[torch.Tensor] = None
    plan6_proposal_depth: Optional[torch.Tensor] = None
    plan6_proposal_alpha: Optional[torch.Tensor] = None
    plan6_online_admission_allowed: bool = False
    plan6_gate_reasons: List[str] = field(default_factory=list)
    plan6_stage_metrics: Dict[str, Any] = field(default_factory=dict)


def _masked_l1_value(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    mask: torch.Tensor,
) -> Optional[float]:
    """计算 masked L1，mask为空时返回None。"""
    if lhs.dim() == 3:
        lhs = lhs.unsqueeze(0)
    if rhs.dim() == 3:
        rhs = rhs.unsqueeze(0)
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)

    lhs = lhs.float()
    rhs = rhs.float()
    mask = mask.float()

    if lhs.shape[-2:] != rhs.shape[-2:]:
        lhs = F.interpolate(lhs, size=rhs.shape[-2:], mode="bilinear", align_corners=False)
    if mask.shape[-2:] != rhs.shape[-2:]:
        mask = F.interpolate(mask, size=rhs.shape[-2:], mode="nearest")
    if mask.shape[1] != 1:
        mask = mask[:, :1]

    valid_weight = float(mask.sum().item()) * float(rhs.shape[1])
    if valid_weight <= 1e-6:
        return None

    masked_l1 = ((lhs - rhs).abs() * mask).sum() / (valid_weight + 1e-6)
    return float(masked_l1.item())


def _compute_frontier_region_masks(
    exist_mask: Optional[torch.Tensor],
    novel_mask: Optional[torch.Tensor],
    *,
    band_radius: int,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if exist_mask is None and novel_mask is None:
        return None, None
    normalized_exist_mask = exist_mask.float() if exist_mask is not None else None
    normalized_novel_mask = novel_mask.float() if novel_mask is not None else None
    if normalized_exist_mask is not None:
        if normalized_exist_mask.dim() == 3:
            normalized_exist_mask = normalized_exist_mask.unsqueeze(0)
        if normalized_exist_mask.shape[1] != 1:
            normalized_exist_mask = normalized_exist_mask[:, :1]
    if normalized_novel_mask is not None:
        if normalized_novel_mask.dim() == 3:
            normalized_novel_mask = normalized_novel_mask.unsqueeze(0)
        if normalized_novel_mask.shape[1] != 1:
            normalized_novel_mask = normalized_novel_mask[:, :1]
    if normalized_exist_mask is None and normalized_novel_mask is not None:
        normalized_exist_mask = torch.clamp(1.0 - normalized_novel_mask, min=0.0, max=1.0)
    if normalized_novel_mask is None and normalized_exist_mask is not None:
        normalized_novel_mask = torch.clamp(1.0 - normalized_exist_mask, min=0.0, max=1.0)
    if normalized_exist_mask is None or normalized_novel_mask is None:
        return None, None
    frontier_radius = max(int(band_radius), 1)
    dilated_exist_mask = F.max_pool2d(
        normalized_exist_mask,
        kernel_size=frontier_radius * 2 + 1,
        stride=1,
        padding=frontier_radius,
    )
    frontier_mask = torch.clamp(normalized_novel_mask * dilated_exist_mask, min=0.0, max=1.0)
    unsupported_novel_mask = torch.clamp(normalized_novel_mask - frontier_mask, min=0.0, max=1.0)
    return frontier_mask, unsupported_novel_mask


def _build_phase2_consistency_debug_bundle(
    *,
    gddn_model,
    sparse_bundle: "SparseViewBundle",
    sel_candidate: ViewCandidate,
    rgb_render: torch.Tensor,
    depth_render: torch.Tensor,
    conditions: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """复算单候选一致性诊断所需的 render/warp/mask 中间量。"""
    debug_bundle: Dict[str, torch.Tensor] = {}

    render_rgb = rgb_render[:1].float().to(device)
    render_depth = depth_render[:1].float().to(device)
    if render_depth.dim() == 4 and render_depth.shape[1] > 1:
        render_depth = render_depth[:, :1]
    render_mask = (render_depth > 1e-6).float()
    debug_bundle["render_rgb"] = render_rgb
    debug_bundle["render_depth"] = render_depth
    debug_bundle["render_mask"] = render_mask

    condition_preprocessor = getattr(gddn_model, "condition_preprocessor", None)
    nearest_view_fn = getattr(gddn_model, "_select_nearest_view", None)
    if condition_preprocessor is None or nearest_view_fn is None:
        return debug_bundle

    sparse_images = sparse_bundle.images.to(device)
    sparse_poses = sparse_bundle.poses.to(device)
    target_pose = sel_candidate.viewmat.unsqueeze(0).to(device)

    if sparse_images.dim() == 4:
        sparse_images = sparse_images.unsqueeze(0)
    if sparse_poses.dim() == 3:
        sparse_poses = sparse_poses.unsqueeze(0)

    nearest_idx = nearest_view_fn(sparse_poses, target_pose)
    reference_image = sparse_images[0, nearest_idx[0]].unsqueeze(0).float()
    reference_pose = sparse_poses[0, nearest_idx[0]].unsqueeze(0).float()
    debug_bundle["reference_image"] = reference_image
    debug_bundle["reference_pose"] = reference_pose

    if render_depth.shape[-2:] != reference_image.shape[-2:]:
        target_depth = F.interpolate(
            render_depth,
            size=reference_image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    else:
        target_depth = render_depth
    debug_bundle["target_depth_for_warp"] = target_depth

    target_intrinsics = conditions.get("camera_intrinsics")
    if target_intrinsics is not None:
        target_intrinsics = target_intrinsics[:1].to(device=device, dtype=torch.float32)
    else:
        target_intrinsics = sel_candidate.intrinsics.unsqueeze(0).to(device=device, dtype=torch.float32)

    reference_intrinsics = conditions.get("reference_intrinsics")
    if reference_intrinsics is not None:
        reference_intrinsics = reference_intrinsics[:1].to(device=device, dtype=torch.float32)
    else:
        reference_intrinsics = target_intrinsics

    reference_depth_map = conditions.get("reference_depth_map")
    if reference_depth_map is not None:
        reference_depth_map = reference_depth_map[:1].to(device=device, dtype=torch.float32)

    warped_rgb, warp_valid_mask = condition_preprocessor.perspective_warp(
        image=reference_image,
        depth=target_depth,
        K=target_intrinsics,
        ref_pose=reference_pose,
        target_pose=target_pose.float(),
        ref_K=reference_intrinsics,
        target_K=target_intrinsics,
        reference_depth_map=reference_depth_map,
    )

    if warped_rgb.shape[-2:] != render_rgb.shape[-2:]:
        warped_rgb = F.interpolate(
            warped_rgb,
            size=render_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    if warp_valid_mask.shape[-2:] != render_rgb.shape[-2:]:
        warp_valid_mask = F.interpolate(
            warp_valid_mask.float(),
            size=render_rgb.shape[-2:],
            mode="nearest",
        )

    supported_exist_mask = (render_mask * warp_valid_mask.float()).clamp(0.0, 1.0)
    debug_bundle["warped_rgb"] = warped_rgb
    debug_bundle["warp_valid_mask"] = warp_valid_mask.float()
    debug_bundle["supported_exist_mask"] = supported_exist_mask

    return debug_bundle


def _compute_phase2_consistency_metrics_from_bundle(
    *,
    pseudo_rgb: torch.Tensor,
    debug_bundle: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """基于已复算的 render/warp/mask 中间量输出一致性诊断指标。"""
    debug_metrics: Dict[str, float] = {}

    render_rgb = debug_bundle.get("render_rgb")
    render_mask = debug_bundle.get("render_mask")
    if render_rgb is None or render_mask is None:
        return debug_metrics

    debug_metrics["render_coverage"] = float(render_mask.mean().item())

    render_only_l1 = _masked_l1_value(pseudo_rgb, render_rgb, render_mask)
    if render_only_l1 is not None:
        debug_metrics["exist_l1_render_only"] = render_only_l1

    supported_exist_mask = debug_bundle.get("supported_exist_mask")
    if supported_exist_mask is not None:
        debug_metrics["supported_exist_coverage"] = float(supported_exist_mask.mean().item())
        supported_exist_l1 = _masked_l1_value(pseudo_rgb, render_rgb, supported_exist_mask)
        if supported_exist_l1 is not None:
            debug_metrics["exist_l1_supported_exist"] = supported_exist_l1

        unsupported_render_mask = (render_mask * (1.0 - supported_exist_mask)).clamp(0.0, 1.0)
        debug_metrics["unsupported_render_coverage"] = float(unsupported_render_mask.mean().item())
        unsupported_render_l1 = _masked_l1_value(pseudo_rgb, render_rgb, unsupported_render_mask)
        if unsupported_render_l1 is not None:
            debug_metrics["unsupported_render_l1"] = unsupported_render_l1

        warped_rgb = debug_bundle.get("warped_rgb")
        if warped_rgb is not None:
            warp_vs_render_l1 = _masked_l1_value(warped_rgb, render_rgb, supported_exist_mask)
            if warp_vs_render_l1 is not None:
                debug_metrics["warp_vs_render_l1_on_supported_exist"] = warp_vs_render_l1

    return debug_metrics








def _extract_generated_rgb_tensor(gen_result) -> Optional[torch.Tensor]:
    if isinstance(gen_result, dict):
        pseudo_rgb = gen_result.get("rgb", gen_result.get("image"))
    else:
        pseudo_rgb = gen_result

    if pseudo_rgb is None or not isinstance(pseudo_rgb, torch.Tensor):
        return None
    if pseudo_rgb.dim() == 4:
        pseudo_rgb = pseudo_rgb[0]
    return pseudo_rgb


def _extract_generation_diagnostics(gen_result: Any) -> Dict[str, Any]:
    if not isinstance(gen_result, dict):
        return {}
    generation_diagnostics = gen_result.get("diagnostics", {})
    return generation_diagnostics if isinstance(generation_diagnostics, dict) else {}




def _safe_optional_float(metric_value: Any) -> Optional[float]:
    if metric_value is None:
        return None
    if isinstance(metric_value, torch.Tensor):
        if metric_value.numel() != 1:
            return None
        metric_value = float(metric_value.item())
    try:
        float_value = float(metric_value)
    except (TypeError, ValueError):
        return None
    return float_value if math.isfinite(float_value) else None


def _load_q_gain_calibrator(calibrator_path: str) -> Optional[QGainCalibrator]:
    normalized_path = str(calibrator_path or "").strip()
    if normalized_path == "":
        return None
    if normalized_path not in _Q_GAIN_CALIBRATOR_CACHE:
        try:
            _Q_GAIN_CALIBRATOR_CACHE[normalized_path] = QGainCalibrator.load(normalized_path)
            print(f"[QGain] 已加载校准器: {normalized_path}")
        except Exception as load_error:
            print(f"[QGain] 加载校准器失败，已回退为禁用: {load_error}")
            _Q_GAIN_CALIBRATOR_CACHE[normalized_path] = None
    return _Q_GAIN_CALIBRATOR_CACHE[normalized_path]


def _build_q_gain_runtime_record(
    *,
    selected_metrics: Dict[str, Any],
    consistency_metrics: Optional[Dict[str, Any]],
    health_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    runtime_record: Dict[str, Any] = dict(selected_metrics or {})
    if consistency_metrics:
        runtime_record.update(consistency_metrics)
    if health_metrics:
        runtime_record["health_mean"] = _safe_optional_float(health_metrics.get("per_image_mean"))
        runtime_record["health_std"] = _safe_optional_float(health_metrics.get("per_image_std"))
    return runtime_record






def _compute_consistency_contribution_shares(metrics: Dict[str, Any]) -> Dict[str, float]:
    supported_coverage = metrics.get("supported_exist_coverage")
    supported_l1 = metrics.get("exist_l1_supported_exist")
    unsupported_coverage = metrics.get("unsupported_render_coverage")
    unsupported_l1 = metrics.get("unsupported_render_l1")
    if (
        supported_coverage is None
        or supported_l1 is None
        or unsupported_coverage is None
        or unsupported_l1 is None
    ):
        return {}

    supported_weight = float(supported_coverage) * float(supported_l1)
    unsupported_weight = float(unsupported_coverage) * float(unsupported_l1)
    total_weight = supported_weight + unsupported_weight
    if total_weight <= 1e-8:
        return {}
    return {
        "supported_contribution_pct": 100.0 * supported_weight / total_weight,
        "unsupported_contribution_pct": 100.0 * unsupported_weight / total_weight,
    }


def _normalize_single_channel_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not isinstance(mask, torch.Tensor):
        return None
    normalized_mask = mask.float()
    if normalized_mask.dim() == 4 and normalized_mask.shape[0] == 1:
        normalized_mask = normalized_mask[0]
    if normalized_mask.dim() == 2:
        normalized_mask = normalized_mask.unsqueeze(0)
    if normalized_mask.dim() != 3:
        return None
    if normalized_mask.shape[0] != 1:
        normalized_mask = normalized_mask[:1]
    return normalized_mask.clamp(0.0, 1.0)


def _normalize_rgb_tensor(image: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if not isinstance(image, torch.Tensor):
        return None
    normalized_image = image.float()
    if normalized_image.dim() == 4 and normalized_image.shape[0] == 1:
        normalized_image = normalized_image[0]
    if normalized_image.dim() != 3:
        return None
    if normalized_image.shape[0] == 1:
        normalized_image = normalized_image.repeat(3, 1, 1)
    if normalized_image.shape[0] < 3:
        return None
    if normalized_image.shape[0] > 3:
        normalized_image = normalized_image[:3]
    return normalized_image


def _resize_single_channel_mask(
    mask: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    normalized_mask = _normalize_single_channel_mask(mask)
    if normalized_mask is None:
        return None
    if normalized_mask.shape[-2:] == target_hw:
        return normalized_mask
    resized_mask = F.interpolate(
        normalized_mask.unsqueeze(0),
        size=target_hw,
        mode="nearest",
    ).squeeze(0)
    return resized_mask.clamp(0.0, 1.0)


def _resize_rgb_tensor(
    image: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    normalized_image = _normalize_rgb_tensor(image)
    if normalized_image is None:
        return None
    if normalized_image.shape[-2:] == target_hw:
        return normalized_image
    resized_image = F.interpolate(
        normalized_image.unsqueeze(0),
        size=target_hw,
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized_image


def _compute_single_channel_l1_map(
    lhs: Optional[torch.Tensor],
    rhs: Optional[torch.Tensor],
    target_hw: Tuple[int, int],
) -> Optional[torch.Tensor]:
    lhs_image = _resize_rgb_tensor(lhs, target_hw)
    rhs_image = _resize_rgb_tensor(rhs, target_hw)
    if lhs_image is None or rhs_image is None:
        return None
    return (lhs_image - rhs_image).abs().mean(dim=0, keepdim=True)


def _apply_local_mask_consistency_filter(
    mask: Optional[torch.Tensor],
    *,
    kernel_size: int,
    min_ratio: float,
) -> Optional[torch.Tensor]:
    normalized_mask = _normalize_single_channel_mask(mask)
    if normalized_mask is None:
        return None
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size <= 1:
        return normalized_mask
    pooled_mask = F.avg_pool2d(
        normalized_mask.unsqueeze(0),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    ).squeeze(0)
    filtered_mask = normalized_mask * (pooled_mask >= float(min_ratio)).float()
    if float(filtered_mask.max().item()) <= 1e-6:
        return None
    return filtered_mask.clamp(0.0, 1.0)


def _extract_support_projection_ratio(generation_diagnostics: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(generation_diagnostics, dict):
        return None
    for field_name in (
        "support_projection_ratio",
        "support_projection_ratio_for_schedule",
        "quality_gated_support_projection_coverage",
        "legacy_support_projection_coverage",
    ):
        field_value = _safe_optional_float(generation_diagnostics.get(field_name))
        if field_value is not None:
            return field_value
    return None


def _extract_mask_component_boxes(
    mask: Optional[torch.Tensor],
    *,
    min_pixels: int,
    padding: int,
    max_components: int,
) -> List[Tuple[int, int, int, int, int]]:
    normalized_mask = _normalize_single_channel_mask(mask)
    if normalized_mask is None:
        return []
    mask_array = (normalized_mask[0].detach().cpu().numpy() > 0.5).astype("uint8")
    if mask_array.sum() <= 0:
        return []

    component_boxes: List[Tuple[int, int, int, int, int]] = []
    try:
        import cv2  # type: ignore

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask_array, connectivity=8)
        for label_index in range(1, int(num_labels)):
            x_coord, y_coord, width_value, height_value, area_value = stats[label_index].tolist()
            if int(area_value) < max(int(min_pixels), 1):
                continue
            x0 = max(int(x_coord) - int(padding), 0)
            y0 = max(int(y_coord) - int(padding), 0)
            x1 = min(int(x_coord + width_value) + int(padding), mask_array.shape[1])
            y1 = min(int(y_coord + height_value) + int(padding), mask_array.shape[0])
            component_boxes.append((x0, y0, x1, y1, int(area_value)))
    except Exception:
        ys, xs = mask_array.nonzero()
        if ys.size > 0 and xs.size > 0:
            x0 = max(int(xs.min()) - int(padding), 0)
            y0 = max(int(ys.min()) - int(padding), 0)
            x1 = min(int(xs.max()) + 1 + int(padding), mask_array.shape[1])
            y1 = min(int(ys.max()) + 1 + int(padding), mask_array.shape[0])
            component_boxes.append((x0, y0, x1, y1, int(mask_array.sum())))

    component_boxes.sort(key=lambda box_entry: box_entry[4], reverse=True)
    return component_boxes[: max(int(max_components), 1)]


def _build_verified_proposal_patches(
    *,
    pseudo_rgb: Optional[torch.Tensor],
    verified_evidence_mask: Optional[torch.Tensor],
    frontier_mask: Optional[torch.Tensor],
    config: Any,
) -> Tuple[List[ProposalPatch], Dict[str, float]]:
    proposal_summary: Dict[str, float] = {}
    normalized_image = _normalize_rgb_tensor(pseudo_rgb)
    normalized_verified_mask = _normalize_single_channel_mask(verified_evidence_mask)
    if normalized_image is None or normalized_verified_mask is None:
        return [], proposal_summary

    target_hw = normalized_verified_mask.shape[-2:]
    resized_image = _resize_rgb_tensor(normalized_image, target_hw)
    resized_frontier_mask = _resize_single_channel_mask(frontier_mask, target_hw)
    if resized_image is None:
        return [], proposal_summary

    component_boxes = _extract_mask_component_boxes(
        normalized_verified_mask,
        min_pixels=int(getattr(config, "pseudo_verified_patch_min_pixels", 64)),
        padding=int(getattr(config, "pseudo_verified_patch_padding", 0)),
        max_components=int(getattr(config, "pseudo_verified_max_patches_per_view", 8)),
    )
    if len(component_boxes) == 0:
        return [], proposal_summary

    total_pixels = float(target_hw[0] * target_hw[1])
    frontier_bonus = float(getattr(config, "pseudo_verified_patch_frontier_bonus", 0.5))
    min_patch_score = float(getattr(config, "pseudo_verified_patch_min_score", 0.0))
    proposal_patches: List[ProposalPatch] = []
    total_patch_area_ratio = 0.0

    for x0, y0, x1, y1, _area_pixels in component_boxes:
        patch_mask = normalized_verified_mask[:, y0:y1, x0:x1]
        if patch_mask.numel() == 0 or float(patch_mask.max().item()) <= 1e-6:
            continue
        patch_image = resized_image[:, y0:y1, x0:x1]
        patch_area_ratio = float(patch_mask.sum().item()) / max(total_pixels, 1.0)
        patch_coverage = float(patch_mask.mean().item())
        frontier_coverage = 0.0
        if resized_frontier_mask is not None:
            patch_frontier = resized_frontier_mask[:, y0:y1, x0:x1]
            if patch_frontier.numel() > 0:
                frontier_coverage = float(patch_frontier.mean().item())
        patch_score = patch_area_ratio * (1.0 + frontier_bonus * frontier_coverage)
        if patch_score < min_patch_score:
            continue
        proposal_patches.append(
            ProposalPatch(
                bbox_xyxy=(x0, y0, x1, y1),
                image=patch_image.detach().cpu(),
                mask=patch_mask.detach().cpu(),
                score=float(patch_score),
                coverage=float(patch_coverage),
                frontier_coverage=float(frontier_coverage),
            )
        )
        total_patch_area_ratio += patch_area_ratio

    proposal_patches.sort(key=lambda proposal_patch: proposal_patch.score, reverse=True)
    proposal_summary["proposal_patch_count"] = float(len(proposal_patches))
    proposal_summary["proposal_patch_area_ratio"] = float(total_patch_area_ratio)
    proposal_summary["proposal_patch_best_score"] = (
        float(proposal_patches[0].score) if proposal_patches else 0.0
    )
    return proposal_patches, proposal_summary


def _build_plan6_protocol_config(config: Any, *, current_iter: Optional[int] = None) -> FourStageConfig:
    valid_until_iter = None
    cache_valid_iters = int(getattr(config, "plan6_cache_valid_iters", 0))
    if cache_valid_iters > 0 and current_iter is not None:
        valid_until_iter = int(current_iter) + cache_valid_iters
    gate0_status = str(getattr(config, "plan6_gate0_status", "") or "")
    legacy_trust_is_calibrated = bool(getattr(config, "plan6_trust_is_calibrated", False))
    return FourStageConfig(
        wrong_threshold=float(getattr(config, "plan6_wrong_threshold", 0.35)),
        repairable_threshold=float(getattr(config, "plan6_repairable_threshold", 0.25)),
        trust_threshold=float(getattr(config, "plan6_trust_threshold", 0.50)),
        min_patch_pixels=int(getattr(config, "pseudo_verified_patch_min_pixels", 64)),
        max_patches=int(getattr(config, "pseudo_verified_max_patches_per_view", 8)),
        patch_padding=int(getattr(config, "pseudo_verified_patch_padding", 0)),
        min_trust_coverage=float(getattr(config, "plan6_min_trust_coverage", 0.01)),
        fusion_mode=str(getattr(config, "plan6_fusion_mode", "max")),
        trust_is_calibrated=legacy_trust_is_calibrated,
        rgb_trust_is_calibrated=getattr(config, "plan6_rgb_trust_is_calibrated", None),
        depth_trust_is_calibrated=getattr(config, "plan6_depth_trust_is_calibrated", None),
        alpha_trust_is_calibrated=getattr(config, "plan6_alpha_trust_is_calibrated", None),
        gate0_status=gate0_status or None,
        current_iter=current_iter,
        valid_until_iter=valid_until_iter,
    )


def _extract_plan6_tensor(gen_result: Any, key_name: str) -> Optional[torch.Tensor]:
    if not isinstance(gen_result, dict):
        return None
    value = gen_result.get(key_name)
    if isinstance(value, torch.Tensor):
        return value
    debug_tensors = gen_result.get("debug_tensors")
    if isinstance(debug_tensors, dict):
        value = debug_tensors.get(key_name)
        if isinstance(value, torch.Tensor):
            return value
    return None


def _convert_plan6_patches_to_proposal_patches(
    plan6_result: FourStageResult,
) -> List[ProposalPatch]:
    proposal_patches: List[ProposalPatch] = []
    for plan6_patch in plan6_result.how_repairable.proposal_patches:
        proposal_patches.append(
            ProposalPatch(
                bbox_xyxy=tuple(plan6_patch.bbox_xyxy),
                image=plan6_patch.image.detach().cpu(),
                mask=plan6_patch.mask.detach().cpu(),
                score=float(plan6_patch.score),
                coverage=float(plan6_patch.coverage),
                frontier_coverage=0.0,
            )
        )
    return proposal_patches


def _run_plan6_protocol_for_phase2(
    *,
    gen_result: Any,
    pseudo_rgb: Optional[torch.Tensor],
    render_rgb: torch.Tensor,
    depth_render: torch.Tensor,
    verified_evidence_mask: Optional[torch.Tensor],
    supported_exist_mask: Optional[torch.Tensor],
    frontier_mask: Optional[torch.Tensor],
    epistemic_map: Optional[torch.Tensor],
    aleatoric_map: Optional[torch.Tensor],
    config: Any,
    current_iter: Optional[int],
) -> Optional[FourStageResult]:
    if not bool(getattr(config, "plan6_four_stage_enable", False)):
        return None
    proposal_tensor = _extract_plan6_tensor(gen_result, "proposal_raw_rgb")
    if proposal_tensor is None:
        proposal_tensor = pseudo_rgb
    payload: Dict[str, Any] = {
        "render_rgb": render_rgb.detach().cpu(),
        "warped_rgb": _extract_plan6_tensor(gen_result, "warped_rgb"),
        "support_composed_rgb": _extract_plan6_tensor(gen_result, "support_composed_image"),
        "proposal_rgb": proposal_tensor,
        "render_alpha": (depth_render[:1, :1].detach().cpu() > 1e-6).float(),
        "supported_exist_mask": supported_exist_mask.detach().cpu() if isinstance(supported_exist_mask, torch.Tensor) else None,
        "verified_evidence_mask": verified_evidence_mask.detach().cpu() if isinstance(verified_evidence_mask, torch.Tensor) else None,
        "frontier_mask": frontier_mask.detach().cpu() if isinstance(frontier_mask, torch.Tensor) else None,
        "epistemic_map": epistemic_map.detach().cpu() if isinstance(epistemic_map, torch.Tensor) else None,
        "aleatoric_map": aleatoric_map.detach().cpu() if isinstance(aleatoric_map, torch.Tensor) else None,
        "proposal_acceptance_confidence": _extract_plan6_tensor(gen_result, "proposal_acceptance_confidence"),
        "proposal_verification_confidence": _extract_plan6_tensor(gen_result, "proposal_verification_confidence"),
        "proposal_confidence": _extract_plan6_tensor(gen_result, "proposal_confidence"),
        "proposal_depth": _extract_plan6_tensor(gen_result, "proposal_depth"),
        "proposal_alpha": _extract_plan6_tensor(gen_result, "proposal_alpha"),
    }
    if payload["warped_rgb"] is None:
        payload["warped_rgb"] = render_rgb.detach().cpu()
    if payload["support_composed_rgb"] is None:
        payload["support_composed_rgb"] = pseudo_rgb.detach().cpu() if isinstance(pseudo_rgb, torch.Tensor) else payload["warped_rgb"]
    if payload["proposal_rgb"] is None:
        payload["proposal_rgb"] = pseudo_rgb.detach().cpu() if isinstance(pseudo_rgb, torch.Tensor) else payload["support_composed_rgb"]
    if payload["proposal_rgb"] is None:
        return None
    return run_four_stage_protocol(
        payload,
        _build_plan6_protocol_config(config, current_iter=current_iter),
    )


def _build_verified_evidence_mask(
    *,
    pseudo_rgb: Optional[torch.Tensor],
    debug_bundle: Dict[str, torch.Tensor],
    generation_diagnostics: Optional[Dict[str, Any]],
    config: Any,
) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
    verified_metrics: Dict[str, float] = {}
    supported_exist_mask = _normalize_single_channel_mask(debug_bundle.get("supported_exist_mask"))
    render_rgb = _normalize_rgb_tensor(debug_bundle.get("render_rgb"))
    if supported_exist_mask is None or render_rgb is None:
        return None, verified_metrics

    target_hw = supported_exist_mask.shape[-2:]
    pseudo_rgb_resized = _resize_rgb_tensor(pseudo_rgb, target_hw)
    render_rgb_resized = _resize_rgb_tensor(render_rgb, target_hw)
    warped_rgb_resized = _resize_rgb_tensor(debug_bundle.get("warped_rgb"), target_hw)
    if pseudo_rgb_resized is None or render_rgb_resized is None:
        return None, verified_metrics

    warp_valid_ratio = _safe_optional_float(
        generation_diagnostics.get("warp_valid_ratio") if isinstance(generation_diagnostics, dict) else None
    )
    support_projection_ratio = _extract_support_projection_ratio(generation_diagnostics)
    if warp_valid_ratio is not None:
        verified_metrics["verified_warp_valid_ratio"] = float(warp_valid_ratio)
        if warp_valid_ratio < float(getattr(config, "pseudo_verified_min_warp_valid_ratio", 0.0)):
            return None, verified_metrics
    if support_projection_ratio is not None:
        verified_metrics["verified_support_projection_ratio"] = float(support_projection_ratio)
        if support_projection_ratio < float(
            getattr(config, "pseudo_verified_min_support_projection_ratio", 0.0)
        ):
            return None, verified_metrics

    verified_mask = supported_exist_mask.clone()
    verified_metrics["verified_supported_input_coverage"] = float(verified_mask.mean().item())

    render_l1_map = _compute_single_channel_l1_map(pseudo_rgb_resized, render_rgb_resized, target_hw)
    if render_l1_map is not None:
        render_l1_threshold = float(getattr(config, "pseudo_verified_pixel_l1_max", 0.18))
        verified_mask = verified_mask * (render_l1_map <= render_l1_threshold).float()

    if warped_rgb_resized is not None:
        pseudo_warp_l1_map = _compute_single_channel_l1_map(pseudo_rgb_resized, warped_rgb_resized, target_hw)
        warp_render_l1_map = _compute_single_channel_l1_map(warped_rgb_resized, render_rgb_resized, target_hw)
        if pseudo_warp_l1_map is not None:
            pseudo_warp_l1_threshold = float(getattr(config, "pseudo_verified_warp_l1_max", 0.18))
            verified_mask = verified_mask * (pseudo_warp_l1_map <= pseudo_warp_l1_threshold).float()
        if warp_render_l1_map is not None:
            warp_render_l1_threshold = float(getattr(config, "pseudo_verified_warp_render_l1_max", 0.12))
            verified_mask = verified_mask * (warp_render_l1_map <= warp_render_l1_threshold).float()

    verified_mask = _apply_local_mask_consistency_filter(
        verified_mask,
        kernel_size=int(getattr(config, "pseudo_verified_neighborhood_kernel", 1)),
        min_ratio=float(getattr(config, "pseudo_verified_neighborhood_ratio", 0.0)),
    )
    if verified_mask is None:
        return None, verified_metrics

    verified_coverage = float(verified_mask.mean().item())
    verified_metrics["verified_evidence_coverage"] = verified_coverage

    masked_render_l1 = _masked_l1_value(pseudo_rgb_resized, render_rgb_resized, verified_mask)
    if masked_render_l1 is not None:
        verified_metrics["verified_masked_render_l1"] = float(masked_render_l1)
    masked_pseudo_warp_l1 = None
    if warped_rgb_resized is not None:
        masked_pseudo_warp_l1 = _masked_l1_value(pseudo_rgb_resized, warped_rgb_resized, verified_mask)
        if masked_pseudo_warp_l1 is not None:
            verified_metrics["verified_masked_pseudo_warp_l1"] = float(masked_pseudo_warp_l1)

    agreement_score = 1.0
    render_l1_threshold = float(max(getattr(config, "pseudo_verified_pixel_l1_max", 0.18), 1e-6))
    if masked_render_l1 is not None:
        agreement_score *= max(0.0, 1.0 - float(masked_render_l1) / render_l1_threshold)
    pseudo_warp_l1_threshold = float(max(getattr(config, "pseudo_verified_warp_l1_max", 0.18), 1e-6))
    if masked_pseudo_warp_l1 is not None:
        agreement_score *= max(0.0, 1.0 - float(masked_pseudo_warp_l1) / pseudo_warp_l1_threshold)
    verified_metrics["verified_evidence_score"] = float(verified_coverage * agreement_score)
    return verified_mask, verified_metrics


def _compute_mask_coverage(mask: Optional[torch.Tensor]) -> float:
    normalized_mask = _normalize_single_channel_mask(mask)
    if normalized_mask is None:
        return 0.0
    return float(normalized_mask.mean().item())


def _resolve_verified_evidence_mask(
    *,
    verified_evidence_mask: Optional[torch.Tensor],
    supported_exist_mask: Optional[torch.Tensor],
    config: Any,
) -> Optional[torch.Tensor]:
    normalized_verified_mask = _normalize_single_channel_mask(verified_evidence_mask)
    if normalized_verified_mask is not None and float(normalized_verified_mask.max().item()) > 1e-6:
        return normalized_verified_mask
    if bool(getattr(config, "pseudo_verified_allow_supported_fallback", False)):
        normalized_supported_mask = _normalize_single_channel_mask(supported_exist_mask)
        if normalized_supported_mask is not None and float(normalized_supported_mask.max().item()) > 1e-6:
            return normalized_supported_mask
    return None


def _compute_weighted_masked_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    weight_map: Optional[torch.Tensor] = None,
    hard_mask: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    prediction_rgb = prediction.float()
    target_rgb = target.float()
    if prediction_rgb.dim() == 4 and prediction_rgb.shape[0] == 1:
        prediction_rgb = prediction_rgb[0]
    if target_rgb.dim() == 4 and target_rgb.shape[0] == 1:
        target_rgb = target_rgb[0]
    if prediction_rgb.dim() != 3 or target_rgb.dim() != 3:
        return None
    if prediction_rgb.shape[-2:] != target_rgb.shape[-2:]:
        target_rgb = F.interpolate(
            target_rgb.unsqueeze(0),
            size=prediction_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    effective_weight_map = torch.ones(
        (1, prediction_rgb.shape[-2], prediction_rgb.shape[-1]),
        device=prediction_rgb.device,
        dtype=prediction_rgb.dtype,
    )
    for candidate_weight_map_index, candidate_weight_map in enumerate((weight_map, hard_mask)):
        normalized_mask = _normalize_single_channel_mask(candidate_weight_map)
        if normalized_mask is None:
            continue
        if normalized_mask.shape[-2:] != prediction_rgb.shape[-2:]:
            if candidate_weight_map_index == 0:
                normalized_mask = F.interpolate(
                    normalized_mask.unsqueeze(0),
                    size=prediction_rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            else:
                normalized_mask = F.interpolate(
                    normalized_mask.unsqueeze(0),
                    size=prediction_rgb.shape[-2:],
                    mode="nearest",
                ).squeeze(0)
        effective_weight_map = effective_weight_map * normalized_mask.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )

    effective_weight_sum = effective_weight_map.sum()
    if float(effective_weight_sum.item()) <= 1e-6:
        return None

    expanded_weight_map = effective_weight_map.expand_as(prediction_rgb)
    return (
        ((prediction_rgb - target_rgb).abs() * expanded_weight_map).sum()
        / (expanded_weight_map.sum() + 1e-6)
    )


def _normalize_plan6_single_channel_map(map_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if map_tensor is None:
        return None
    tensor = map_tensor.float()
    while tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        return None
    if tensor.shape[0] != 1 and tensor.shape[-1] == 1:
        tensor = tensor.permute(2, 0, 1).contiguous()
    if tensor.shape[0] != 1:
        tensor = tensor[:1]
    return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_plan6_weighted_map_l1_loss(
    prediction_map: Optional[torch.Tensor],
    target_map: Optional[torch.Tensor],
    weight_map: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    prediction = _normalize_plan6_single_channel_map(prediction_map)
    target = _normalize_plan6_single_channel_map(target_map)
    weight = _normalize_single_channel_mask(weight_map)
    if prediction is None or target is None or weight is None:
        return None
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target.unsqueeze(0), size=prediction.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    if weight.shape[-2:] != prediction.shape[-2:]:
        weight = F.interpolate(weight.unsqueeze(0), size=prediction.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    weight = weight.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    target = target.to(device=prediction.device, dtype=prediction.dtype)
    if float(weight.max().item()) <= 1e-6:
        return None
    return ((prediction - target).abs() * weight).sum() / (weight.sum() + 1e-6)


def _compute_plan6_weighted_alpha_loss(
    prediction_alpha: Optional[torch.Tensor],
    target_alpha: Optional[torch.Tensor],
    weight_map: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    prediction = _normalize_plan6_single_channel_map(prediction_alpha)
    target = _normalize_plan6_single_channel_map(target_alpha)
    weight = _normalize_single_channel_mask(weight_map)
    if prediction is None or target is None or weight is None:
        return None
    if target.shape[-2:] != prediction.shape[-2:]:
        target = F.interpolate(target.unsqueeze(0), size=prediction.shape[-2:], mode="nearest").squeeze(0)
    if weight.shape[-2:] != prediction.shape[-2:]:
        weight = F.interpolate(weight.unsqueeze(0), size=prediction.shape[-2:], mode="bilinear", align_corners=False).squeeze(0)
    prediction = prediction.clamp(1e-4, 1.0 - 1e-4)
    target = target.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    weight = weight.to(device=prediction.device, dtype=prediction.dtype).clamp(0.0, 1.0)
    if float(weight.max().item()) <= 1e-6:
        return None
    bce = F.binary_cross_entropy(prediction, target, reduction="none")
    return (bce * weight).sum() / (weight.sum() + 1e-6)


def _extract_plan6_alpha_from_raster(alpha_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if alpha_tensor is None:
        return None
    tensor = alpha_tensor.float()
    if tensor.dim() == 4:
        if tensor.shape[-1] == 1:
            tensor = tensor[0].permute(2, 0, 1).contiguous()
        else:
            tensor = tensor[0, :1]
    return _normalize_plan6_single_channel_map(tensor)


def _extract_plan6_depth_from_render(render_tensor: torch.Tensor, info: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
    if isinstance(info, dict):
        for key_name in ("render_depth", "depth", "median_depth", "expected_depth"):
            value = info.get(key_name)
            if isinstance(value, torch.Tensor):
                normalized = _normalize_plan6_single_channel_map(value[0] if value.dim() == 4 and value.shape[0] == 1 else value)
                if normalized is not None:
                    return normalized
    if render_tensor.dim() == 4 and render_tensor.shape[-1] > 3:
        return _normalize_plan6_single_channel_map(render_tensor[0, ..., 3].unsqueeze(0))
    return None


def _compute_plan6_depth_alpha_supervision_losses(
    render_tensor: torch.Tensor,
    alpha_tensor: Optional[torch.Tensor],
    render_info: Optional[Dict[str, Any]],
    *,
    proposal_depth: Optional[torch.Tensor],
    proposal_alpha: Optional[torch.Tensor],
    depth_weight_map: Optional[torch.Tensor],
    alpha_weight_map: Optional[torch.Tensor],
    config: Any,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not bool(getattr(config, "phase3_use_plan6_depth_alpha_losses", False)):
        return None, None
    depth_loss = None
    alpha_loss = None
    render_depth = _extract_plan6_depth_from_render(render_tensor, render_info)
    if render_depth is not None:
        depth_l1 = _compute_plan6_weighted_map_l1_loss(
            render_depth,
            proposal_depth,
            depth_weight_map,
        )
        if depth_l1 is not None:
            depth_loss = depth_l1 * float(getattr(config, "phase3_plan6_depth_loss_weight", 0.05))
    render_alpha = _extract_plan6_alpha_from_raster(alpha_tensor)
    if render_alpha is not None:
        alpha_bce = _compute_plan6_weighted_alpha_loss(
            render_alpha,
            proposal_alpha,
            alpha_weight_map,
        )
        if alpha_bce is not None:
            alpha_loss = alpha_bce * float(getattr(config, "phase3_plan6_alpha_loss_weight", 0.02))
    return depth_loss, alpha_loss


def _compute_patch_proposal_guidance_loss(
    prediction: torch.Tensor,
    proposal_patches: Sequence[ProposalPatch],
    *,
    source_width: int,
    source_height: int,
    weight_map: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    prediction_rgb = prediction.float()
    if prediction_rgb.dim() == 4 and prediction_rgb.shape[0] == 1:
        prediction_rgb = prediction_rgb[0]
    if prediction_rgb.dim() != 3 or len(proposal_patches) == 0:
        return None

    prediction_height, prediction_width = prediction_rgb.shape[-2:]
    normalized_weight_map = _normalize_single_channel_mask(weight_map)
    if normalized_weight_map is not None and normalized_weight_map.shape[-2:] != prediction_rgb.shape[-2:]:
        normalized_weight_map = F.interpolate(
            normalized_weight_map.unsqueeze(0),
            size=prediction_rgb.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    if normalized_weight_map is not None:
        normalized_weight_map = normalized_weight_map.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )

    normalized_source_width = max(int(source_width), 1)
    normalized_source_height = max(int(source_height), 1)
    aggregated_patch_loss: Optional[torch.Tensor] = None
    aggregated_patch_weight = 0.0

    for proposal_patch in proposal_patches:
        x0, y0, x1, y1 = proposal_patch.bbox_xyxy
        scaled_x0 = int(math.floor(float(x0) * float(prediction_width) / float(normalized_source_width)))
        scaled_y0 = int(math.floor(float(y0) * float(prediction_height) / float(normalized_source_height)))
        scaled_x1 = int(math.ceil(float(x1) * float(prediction_width) / float(normalized_source_width)))
        scaled_y1 = int(math.ceil(float(y1) * float(prediction_height) / float(normalized_source_height)))
        scaled_x0 = max(min(scaled_x0, prediction_width - 1), 0)
        scaled_y0 = max(min(scaled_y0, prediction_height - 1), 0)
        scaled_x1 = max(min(scaled_x1, prediction_width), scaled_x0 + 1)
        scaled_y1 = max(min(scaled_y1, prediction_height), scaled_y0 + 1)

        rendered_patch = prediction_rgb[:, scaled_y0:scaled_y1, scaled_x0:scaled_x1]
        if rendered_patch.numel() == 0:
            continue

        patch_target = proposal_patch.image.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )
        patch_mask = proposal_patch.mask.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )
        if patch_target.shape[-2:] != rendered_patch.shape[-2:]:
            patch_target = F.interpolate(
                patch_target.unsqueeze(0),
                size=rendered_patch.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if patch_mask.shape[-2:] != rendered_patch.shape[-2:]:
            patch_mask = F.interpolate(
                patch_mask.unsqueeze(0),
                size=rendered_patch.shape[-2:],
                mode="nearest",
            ).squeeze(0)
        if float(patch_mask.max().item()) <= 1e-6:
            continue

        patch_weight_map = None
        if normalized_weight_map is not None:
            patch_weight_map = normalized_weight_map[:, scaled_y0:scaled_y1, scaled_x0:scaled_x1]

        patch_guidance_loss = _compute_weighted_masked_l1_loss(
            rendered_patch,
            patch_target,
            weight_map=patch_weight_map,
            hard_mask=patch_mask,
        )
        if patch_guidance_loss is None:
            continue

        patch_score_weight = max(float(proposal_patch.score), 1e-6)
        aggregated_patch_loss = (
            patch_guidance_loss * patch_score_weight
            if aggregated_patch_loss is None
            else aggregated_patch_loss + patch_guidance_loss * patch_score_weight
        )
        aggregated_patch_weight += patch_score_weight

    if aggregated_patch_loss is None or aggregated_patch_weight <= 1e-6:
        return None
    return aggregated_patch_loss / (aggregated_patch_weight + 1e-6)


def _build_phase3_region_weight_map(
    *,
    risk_map: torch.Tensor,
    gain_map: torch.Tensor,
    exist_mask: torch.Tensor,
    novel_mask: torch.Tensor,
    frontier_mask: Optional[torch.Tensor],
    supported_exist_mask: Optional[torch.Tensor],
    unsupported_novel_mask: Optional[torch.Tensor],
    config: Any,
) -> torch.Tensor:
    if not bool(getattr(config, "phase3_region_weight_enable", False)):
        exist_weights = torch.exp(-torch.clamp(risk_map, min=0.0))
        novel_weights = 1.0 + torch.tanh(torch.clamp(gain_map, min=0.0))
        return exist_mask * exist_weights + novel_mask * novel_weights

    supported_exist_region = exist_mask if supported_exist_mask is None else supported_exist_mask
    if frontier_mask is None:
        frontier_region = torch.zeros_like(exist_mask)
    else:
        frontier_region = frontier_mask
    if unsupported_novel_mask is None:
        unsupported_region = torch.clamp(novel_mask - frontier_region, min=0.0, max=1.0)
    else:
        unsupported_region = unsupported_novel_mask

    supported_exist_scale = float(getattr(config, "phase3_supported_exist_scale", 1.0))
    frontier_scale = float(getattr(config, "phase3_frontier_weight_scale", 1.5))
    unsupported_scale = float(getattr(config, "phase3_unsupported_novel_weight_scale", 0.25))

    supported_exist_weights = supported_exist_scale * torch.exp(-torch.clamp(risk_map, min=0.0))
    frontier_weights = frontier_scale * (1.0 + torch.tanh(torch.clamp(gain_map, min=0.0)))
    unsupported_weights = unsupported_scale * (1.0 + torch.tanh(torch.clamp(gain_map, min=0.0)))
    return (
        supported_exist_region * supported_exist_weights
        + frontier_region * frontier_weights
        + unsupported_region * unsupported_weights
    )


def _compute_expected_default_dynamic_strength(
    inpaint_hole_ratio: Optional[float],
) -> Optional[float]:
    if inpaint_hole_ratio is None:
        return None
    hole_ratio_value = float(inpaint_hole_ratio)
    if hole_ratio_value < 0.3:
        return 0.25
    if hole_ratio_value < 0.6:
        return 0.40
    return 0.70


def _classify_default_dynamic_strength_branch(
    inpaint_hole_ratio: Optional[float],
) -> Optional[str]:
    if inpaint_hole_ratio is None:
        return None
    hole_ratio_value = float(inpaint_hole_ratio)
    if hole_ratio_value < 0.3:
        return "lt_0_30_to_0_25"
    if hole_ratio_value < 0.6:
        return "0_30_to_0_60_to_0_40"
    return "ge_0_60_to_0_70"


def _extract_dynamic_strength_audit_fields(
    generation_diagnostics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(generation_diagnostics, dict):
        return {}

    inpaint_hole_ratio = generation_diagnostics.get("inpaint_hole_ratio")
    effective_img2img_strength = generation_diagnostics.get("effective_img2img_strength")
    expected_default_dynamic_strength = _compute_expected_default_dynamic_strength(inpaint_hole_ratio)
    default_dynamic_strength_branch = _classify_default_dynamic_strength_branch(inpaint_hole_ratio)

    audit_fields: Dict[str, Any] = {}
    if inpaint_hole_ratio is not None:
        audit_fields["inpaint_hole_ratio"] = float(inpaint_hole_ratio)
    if effective_img2img_strength is not None:
        audit_fields["effective_img2img_strength"] = float(effective_img2img_strength)
    if expected_default_dynamic_strength is not None:
        audit_fields["expected_default_dynamic_strength"] = float(expected_default_dynamic_strength)
    if default_dynamic_strength_branch is not None:
        audit_fields["default_dynamic_strength_branch"] = default_dynamic_strength_branch
    if expected_default_dynamic_strength is not None and effective_img2img_strength is not None:
        audit_fields["matches_expected_default_dynamic_strength"] = (
            abs(float(effective_img2img_strength) - float(expected_default_dynamic_strength)) <= 1e-6
        )
    return audit_fields


def _build_round_generation_diagnostic_summary(
    generation_audit_entries: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "candidate_count": int(len(generation_audit_entries)),
        "near_threshold_count_abs_0_005": 0,
        "near_threshold_count_abs_0_01": 0,
        "negative_margin_count": 0,
        "dynamic_strength_branch_counts": {},
        "effective_strength_counts": {},
        "expected_default_dynamic_strength_counts": {},
    }

    gate_margin_values: List[float] = []
    hole_ratio_values: List[float] = []
    for generation_audit_entry in generation_audit_entries:
        gate_margin = generation_audit_entry.get("gate_margin")
        if gate_margin is not None:
            gate_margin_value = float(gate_margin)
            gate_margin_values.append(gate_margin_value)
            if abs(gate_margin_value) <= 0.005:
                summary["near_threshold_count_abs_0_005"] += 1
            if abs(gate_margin_value) <= 0.01:
                summary["near_threshold_count_abs_0_01"] += 1
            if gate_margin_value < 0.0:
                summary["negative_margin_count"] += 1

        hole_ratio = generation_audit_entry.get("inpaint_hole_ratio")
        if hole_ratio is not None:
            hole_ratio_values.append(float(hole_ratio))

        branch_name = generation_audit_entry.get("default_dynamic_strength_branch")
        if branch_name is not None:
            branch_name = str(branch_name)
            summary["dynamic_strength_branch_counts"][branch_name] = (
                int(summary["dynamic_strength_branch_counts"].get(branch_name, 0)) + 1
            )

        effective_strength = generation_audit_entry.get("effective_img2img_strength")
        if effective_strength is not None:
            strength_key = f"{float(effective_strength):.2f}"
            summary["effective_strength_counts"][strength_key] = (
                int(summary["effective_strength_counts"].get(strength_key, 0)) + 1
            )

        expected_strength = generation_audit_entry.get("expected_default_dynamic_strength")
        if expected_strength is not None:
            expected_strength_key = f"{float(expected_strength):.2f}"
            summary["expected_default_dynamic_strength_counts"][expected_strength_key] = (
                int(summary["expected_default_dynamic_strength_counts"].get(expected_strength_key, 0)) + 1
            )

    if len(gate_margin_values) > 0:
        summary["gate_margin_avg"] = float(sum(gate_margin_values) / len(gate_margin_values))
        summary["gate_margin_min"] = float(min(gate_margin_values))
        summary["gate_margin_max"] = float(max(gate_margin_values))
    if len(hole_ratio_values) > 0:
        summary["hole_ratio_min"] = float(min(hole_ratio_values))
        summary["hole_ratio_max"] = float(max(hole_ratio_values))
    return summary










def run_phase2_active_sampling(
    gaussians,
    real_cameras: List["ExistingCamera"],
    gddn_model,
    sparse_bundle: "SparseViewBundle",
    config: "ActiveSamplingConfig",
    *,
    scene_center: Optional[torch.Tensor] = None,
    condition_provider=None,
    writer=None,
    pseudo_save_dir: Optional[Path] = None,
) -> List[PseudoView]:
    """
    Phase 2: 在Phase 1训练好的3DGS上运行MC-Dropout主动采样。
    
    完整保留MC-Dropout不确定性管线:
      粗筛 → 精筛 → Pearson验证 → 贪心多样性选择 → GDDN高质量生成伪视图
    
    Returns:
        List[PseudoView]: 包含伪视图RGB + uncertainty maps的列表
    """
    device = next(iter([p.device for p in gaussians.parameters()]))
    if scene_center is None:
        # 使用真实相机位置均值作为球心（伪视角很相机群周围）
        # 高斯体均值（Z≈1）是场景内容中心，不能用作球心！
        if real_cameras:
            _cam_positions = torch.stack([cam.position.detach() for cam in real_cameras])
            _cam_center = _cam_positions.mean(dim=0).to(device)
            # 计算平均前方方向
            _fwd_dirs = []
            for cam in real_cameras:
                _fwd = cam.viewmat[:3, 2].detach().to(device)
                _fwd = _fwd / (_fwd.norm() + 1e-8)
                _fwd_dirs.append(_fwd)
            _avg_fwd = torch.stack(_fwd_dirs).mean(dim=0)
            _avg_fwd = _avg_fwd / (_avg_fwd.norm() + 1e-8)
            # 高斯体均值到相机均值沿前方的投影深度
            _gauss_center = gaussians.means.detach().mean(dim=0)
            _depth_proj = torch.dot(_gauss_center - _cam_center, _avg_fwd).item()
            _depth_proj = max(abs(float(_depth_proj)), 1.0)
            # 球心小幅偶向场景（避免阿位使 look_at 方向完全错误）
            scene_center = _cam_center + 0.05 * _depth_proj * _avg_fwd
            print(f"[scene_center] 相机均值={_cam_center.tolist()}, 深度={_depth_proj:.1f}, "
                  f"scene_center={scene_center.tolist()}")
        else:
            scene_center = gaussians.means.detach().mean(dim=0).to(device)
    else:
        scene_center = scene_center.to(device)
    default_intrinsics = real_cameras[0].intrinsics.to(device)
    provider = condition_provider or example_condition_provider

    sampler = ViewpointSampler(config)
    estimator = UncertaintyEstimator(
        model=gddn_model if config.estimator_mode == "gddn" else None,
        config=config,
        device=device,
        condition_provider=provider,
        writer=writer,
        mode=config.estimator_mode,
        gaussians=gaussians,
        rasterizer=rasterization,
    )
    selector = ActiveSelector(config)
    all_pseudo_views: List[PseudoView] = []
    rounds = int(config.phase2_rounds)

    print(f"[Phase 2] ===== MC-Dropout主动采样开始 =====")
    print(f"[Phase 2] 采样轮数: {rounds}")
    print(f"[Phase 2] 每轮选视角数: {config.num_views_per_round}")
    for round_idx in range(rounds):
        print(f"\n[Phase 2] --- 第{round_idx+1}/{rounds}轮 ---")

        # 1. 生成候选视角
        # P1修复: 将之前轮次已选出的伪视角加入existing_cameras，
        # 驱动ViewpointSampler的近邻过滤(_is_too_close)排除已覆盖区域，
        # 防止每轮重复选择相同视角（原问题: 15个伪视角仅3-4个唯一位置）
        extended_cameras = list(real_cameras)
        for pv in all_pseudo_views:
            extended_cameras.append(ExistingCamera(
                position=pv.camera.position,
                viewmat=pv.camera.viewmat,
                intrinsics=pv.camera.intrinsics,
            ))

        # 球心 = 真实相机均值（Z≈0）；look_at_target = 高斯体均值（Z≈80属场景内容）
        _real_cam_positions = torch.stack([cam.position.detach() for cam in real_cameras]).to(device)
        _sphere_center = _real_cam_positions.mean(dim=0)  # 盆心：相机群均值，Z≈0
        _gauss_look_target = gaussians.means.detach().mean(dim=0).to(device)  # look-at: 场景内容，Z≈80
        candidates = sampler.generate_candidates(
            scene_center=_sphere_center,
            intrinsics=default_intrinsics,
            existing_cameras=extended_cameras,
            device=device,
            reference_cameras=real_cameras,
            look_at_target=_gauss_look_target,
        )
        print(f"[Phase 2] 生成了 {len(candidates)} 个候选视角")

        # 2. 粗筛 MC-Dropout
        coarse_results = _estimate_candidates(
            estimator, sparse_bundle, candidates,
            global_step=round_idx * 1000,
            num_samples=config.mc_dropout_samples_coarse,
            label="coarse",
            dist_context=DistributedActiveSamplingContext(),
        )
        if not coarse_results:
            print(f"[Phase 2] 粗筛无结果，跳过本轮")
            continue
        coarse_pool = selector.preselect(
            coarse_results,
            config.coarse_candidate_count or len(coarse_results),
        )
        fine_candidates = [
            item.candidate for item in coarse_pool
        ][:config.fine_candidate_count or len(coarse_pool)]
        print(f"[Phase 2] 粗筛后保留 {len(fine_candidates)} 个候选")

        # 3. 精筛 MC-Dropout
        if not fine_candidates:
            continue
        fine_results = _estimate_candidates(
            estimator, sparse_bundle, fine_candidates,
            global_step=round_idx * 1000,
            num_samples=config.mc_dropout_samples_fine,
            label="fine",
            dist_context=DistributedActiveSamplingContext(),
        )
        if not fine_results:
            continue
        print(f"[Phase 2] 精筛返回 {len(fine_results)} 个结果")

        # 4. Pearson验证
        if fine_results:
            fine_pool = selector.preselect(
                fine_results,
                config.fine_candidate_count or len(fine_results),
            )
            ensemble_scores, split_half_reliability = bootstrap_ensemble_scores(
                gddn_model=gddn_model,
                sparse_bundle=sparse_bundle,
                results_batch=list(fine_pool),
                repeats=int(max(2, config.validation_bootstrap_repeats)),
                candidate_batch_size=int(max(1, getattr(config, "validation_candidate_batch_size", 1))),
                condition_provider=provider,
                steps=int(config.fine_steps),
                resolution=int(config.fine_resolution),
                device=device,
                score_type=config.score_type,
                use_mask_for_scores=bool(config.validation_use_mask),
                aleatoric_variance_floor=float(config.aleatoric_variance_floor),
                sampler_type=config.ensemble_sampler_type,
                per_step_noise_scale=float(config.ensemble_noise_scale),
                mc_dropout2d_p=float(config.mc_dropout2d_p),
                mc_token_dropout_p=float(config.mc_token_dropout_p),
                mc_cond_noise_sigma=float(config.mc_cond_noise_sigma),
                mc_condition_alpha=float(config.mc_condition_alpha),
                mc_latent_noise_std=float(config.mc_latent_noise_std),
            )
            # Split-Half Reliability验证：比较同一组MC采样内部两半的一致性
            # 这消除了mc_pass与bootstrap使用不同z_T导致的不一致问题
            pearson_value = split_half_reliability
            pearson_pass = pearson_value >= float(config.pearson_threshold)
            print(f"[Phase 2] Split-Half Reliability={pearson_value:.4f}, pass={pearson_pass}")
            if not pearson_pass:
                print("[Phase 2] Split-Half验证未通过，跳过本轮伪证据")
                continue

        # 5. 贪心多样性选择
        selected = selector.select(
            fine_pool,
            config.num_views_per_round,
        )
        generation_retry_budget = max(
            len(selected),
            int(getattr(config, "phase2_generation_retry_budget", len(selected))),
        )
        generation_queue: List[Any] = list(selected)
        selected_item_ids = {id(selected_item) for selected_item in selected}
        for fine_pool_item in fine_pool:
            if id(fine_pool_item) in selected_item_ids:
                continue
            generation_queue.append(fine_pool_item)
            if len(generation_queue) >= generation_retry_budget:
                break
        print(
            f"[Phase 2] 初始选择 {len(selected)} 个视角，"
            f"生成尝试预算={len(generation_queue)}"
        )

        # 6. GDDN生成伪视图
        _round_accepted = 0
        _round_rejected = 0
        _round_reject_reasons = []
        _round_generation_diagnostics: List[Dict[str, Any]] = []
        q_gain_calibrator = _load_q_gain_calibrator(getattr(config, "q_gain_calibrator_path", ""))
        for generation_attempt_index, selected_item in enumerate(generation_queue):
            if _round_accepted >= int(config.num_views_per_round):
                break
            sel_candidate = selected_item.candidate if hasattr(selected_item, 'candidate') else selected_item
            _selected_metrics = selected_item.metrics if hasattr(selected_item, "metrics") and selected_item.metrics else {}
            _q_gen = float(_selected_metrics.get("q_gen", 1.0))
            _q_gain = float(_selected_metrics.get("q_gain", 0.5))
            _generation_audit_entry: Dict[str, Any] = {
                "selected_index": int(generation_attempt_index),
                "generation_attempt_index": int(generation_attempt_index),
                "selection_score": float(selected_item.score) if hasattr(selected_item, "score") else 0.0,
                "q_gen": _q_gen,
                "q_gain": _q_gain,
                "q_gain_for_gate": _q_gain,
                "gate_threshold": float(config.pseudo_exist_region_l1_max),
                "accepted": False,
                "reject_reason": None,
            }
            try:
                # 使用condition_provider获取条件渲染（rgb_render, depth_render, normal_render等）
                conditions = provider([sel_candidate])
                rgb_render = conditions.get("rgb_render", torch.zeros(1, 3, 256, 256)).to(device)
                depth_render = conditions.get("depth_render", torch.zeros(1, 3, 256, 256)).to(device)
                normal_render = conditions.get("normal_render", torch.zeros(1, 3, 256, 256)).to(device)


                print(
                    f"  [生成] 伪视图尝试 {generation_attempt_index + 1}/{len(generation_queue)}..."
                    f" (本轮已接受={_round_accepted}/{int(config.num_views_per_round)})",
                    flush=True,
                )
                with torch.no_grad():
                    gen_result = gddn_model.generate_view_correctly(
                        sparse_images=sparse_bundle.images.to(device),
                        sparse_poses=sparse_bundle.poses.to(device),
                        target_poses=sel_candidate.viewmat.unsqueeze(0).to(device),
                        rgb_render=rgb_render,
                        depth_render=depth_render,
                        normal_render=normal_render,
                        reference_depth_map=conditions.get("reference_depth_map"),
                        reference_intrinsics=conditions.get("reference_intrinsics"),
                        camera_intrinsics=conditions.get("camera_intrinsics"),
                        steps=int(config.fine_steps),
                        resolution=int(config.fine_resolution),
                        return_diagnostics=True,
                    )

                _generation_diagnostics = _extract_generation_diagnostics(gen_result)
                if len(_generation_diagnostics) > 0:
                    _generation_audit_entry["generation_diagnostics"] = _generation_diagnostics
                    _generation_audit_entry.update(_extract_dynamic_strength_audit_fields(_generation_diagnostics))

                pseudo_rgb = _extract_generated_rgb_tensor(gen_result)
                if isinstance(gen_result, dict):
                    epi_map = gen_result.get("epistemic_map", gen_result.get("epistemic"))
                    ale_map = gen_result.get("aleatoric_map", gen_result.get("aleatoric"))
                    risk_exist_map = gen_result.get("risk_exist_map", gen_result.get("risk_exist", epi_map))
                    gain_novel_map = gen_result.get("gain_novel_map", gen_result.get("gain_novel", ale_map))
                else:
                    epi_map = None
                    ale_map = None
                    risk_exist_map = None
                    gain_novel_map = None

                # ========== 伪视图质量门控 ==========
                # 检查生成的伪视图是否有效（非灰色/非噪声/非全黑）
                _pseudo_valid = True
                _reject_reason = ""
                _consistency_weight = 1.0
                _health_metrics = None
                _consistency_debug: Dict[str, float] = {}
                _consistency_debug_bundle: Dict[str, torch.Tensor] = {}
                _verified_evidence_mask: Optional[torch.Tensor] = None
                _verified_evidence_coverage = 0.0
                _verified_evidence_score = 0.0
                if pseudo_rgb is not None:
                    _rgb_for_check = pseudo_rgb.float()
                    if not torch.isfinite(_rgb_for_check).all():
                        _pseudo_valid = False
                        _reject_reason = "non_finite_values"
                    else:
                        _rgb_std = float(_rgb_for_check.std().item())
                        _rgb_mean = float(_rgb_for_check.mean().item())
                        _generation_audit_entry["rgb_std"] = _rgb_std
                        _generation_audit_entry["rgb_mean"] = _rgb_mean
                        # 均匀灰色检测: std过低表示无有效纹理
                        if _rgb_std < 0.05:
                            _pseudo_valid = False
                            _reject_reason = f"uniform_color(std={_rgb_std:.4f})"
                        # 全黑/全白检测
                        elif _rgb_mean < 0.02 or _rgb_mean > 0.98:
                            _pseudo_valid = False
                            _reject_reason = f"extreme_brightness(mean={_rgb_mean:.4f})"
                else:
                    _pseudo_valid = False
                    _reject_reason = "rgb_is_none"

                if (
                    _pseudo_valid
                    and bool(getattr(config, "pseudo_use_q_gen_gate", False))
                    and _q_gen < float(config.pseudo_min_q_gen)
                ):
                    _pseudo_valid = False
                    _reject_reason = f"low_q_gen({_q_gen:.4f})"

                if _pseudo_valid:
                    _health_batch = pseudo_rgb.unsqueeze(0) if pseudo_rgb.dim() == 3 else pseudo_rgb
                    _health_ok, _health_reason, _health_metrics = gddn_model.evaluate_health(
                        _health_batch,
                        min_mean=float(config.health_min_mean),
                        max_mean=float(config.health_max_mean),
                        min_std=float(config.health_min_std),
                    )
                    if not _health_ok:
                        _pseudo_valid = False
                        _reject_reason = f"health_{_health_reason}"

                if _pseudo_valid:
                    _render_rgb = rgb_render[:1].float()
                    _render_depth = depth_render[:1].float()
                    if _render_depth.dim() == 4 and _render_depth.shape[1] > 1:
                        _render_depth = _render_depth[:, :1]
                    _render_mask = (_render_depth > 1e-6).float()
                    _render_coverage = float(_render_mask.mean().item())
                    if _render_coverage >= float(config.pseudo_exist_region_min_coverage):
                        try:
                            _consistency_debug_bundle = _build_phase2_consistency_debug_bundle(
                                gddn_model=gddn_model,
                                sparse_bundle=sparse_bundle,
                                sel_candidate=sel_candidate,
                                rgb_render=rgb_render,
                                depth_render=depth_render,
                                conditions=conditions,
                                device=device,
                            )
                            _consistency_debug = _compute_phase2_consistency_metrics_from_bundle(
                                pseudo_rgb=pseudo_rgb,
                                debug_bundle=_consistency_debug_bundle,
                            )
                            _consistency_debug.update(_compute_consistency_contribution_shares(_consistency_debug))
                        except Exception as debug_exc:
                            print(f"[Phase 2][ConsistencyDebug] 诊断复算失败: {debug_exc}")

                        if len(_consistency_debug) > 0:
                            _generation_audit_entry["consistency_metrics"] = dict(_consistency_debug)

                        _masked_l1_value = _consistency_debug.get("exist_l1_render_only")
                        if _masked_l1_value is None:
                            _pseudo_for_consistency = pseudo_rgb.unsqueeze(0).float() if pseudo_rgb.dim() == 3 else pseudo_rgb.float()
                            if _pseudo_for_consistency.shape[-2:] != _render_rgb.shape[-2:]:
                                _pseudo_for_consistency = F.interpolate(
                                    _pseudo_for_consistency,
                                    size=_render_rgb.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            _masked_l1 = (
                                (_pseudo_for_consistency - _render_rgb).abs() * _render_mask
                            ).sum() / (_render_mask.sum() * _render_rgb.shape[1] + 1e-6)
                            _masked_l1_value = float(_masked_l1.item())
                        if _masked_l1_value is not None:
                            _gate_margin = float(config.pseudo_exist_region_l1_max) - float(_masked_l1_value)
                            _generation_audit_entry["gate_margin"] = _gate_margin
                            _generation_audit_entry["near_threshold_abs_0_005"] = abs(_gate_margin) <= 0.005
                            _generation_audit_entry["near_threshold_abs_0_01"] = abs(_gate_margin) <= 0.01

                        _supported_exist_l1 = _consistency_debug.get("exist_l1_supported_exist")
                        _warp_vs_render_l1 = _consistency_debug.get("warp_vs_render_l1_on_supported_exist")
                        _supported_exist_coverage = _consistency_debug.get("supported_exist_coverage", 0.0)
                        _unsupported_render_coverage = _consistency_debug.get("unsupported_render_coverage", 0.0)
                        _unsupported_render_l1 = _consistency_debug.get("unsupported_render_l1")
                        _warp_valid_ratio = _generation_diagnostics.get("warp_valid_ratio")
                        _support_projection_ratio = _generation_diagnostics.get("support_projection_ratio")
                        if _support_projection_ratio is None:
                            _support_projection_ratio = _generation_diagnostics.get("support_projection_ratio_for_schedule")
                        _supported_exist_l1_str = (
                            f"{_supported_exist_l1:.4f}" if _supported_exist_l1 is not None else "NA"
                        )
                        _unsupported_render_l1_str = (
                            f"{_unsupported_render_l1:.4f}" if _unsupported_render_l1 is not None else "NA"
                        )
                        _warp_vs_render_l1_str = (
                            f"{_warp_vs_render_l1:.4f}" if _warp_vs_render_l1 is not None else "NA"
                        )
                        print(
                            "[Phase 2][ConsistencyDebug] "
                            f"render_cov={_render_coverage:.3f}, "
                            f"supported_cov={_supported_exist_coverage:.3f}, "
                            f"unsupported_render_cov={_unsupported_render_coverage:.3f}, "
                            f"exist_l1_render_only={_masked_l1_value:.4f}, "
                            f"exist_l1_supported_exist={_supported_exist_l1_str}, "
                            f"unsupported_render_l1={_unsupported_render_l1_str}, "
                            f"warp_vs_render_l1_on_supported_exist={_warp_vs_render_l1_str}"
                        )
                        _consistency_weight = max(
                            0.0,
                            1.0 - _masked_l1_value / max(float(config.pseudo_exist_region_l1_max), 1e-6),
                        )
                        if _masked_l1_value > float(config.pseudo_exist_region_l1_max):
                            _enable_degraded_accept = bool(
                                getattr(config, "pseudo_degraded_accept_enable", False)
                            )
                            if _enable_degraded_accept:
                                _degraded_l1_max = float(
                                    getattr(
                                        config,
                                        "pseudo_degraded_accept_l1_max",
                                        float(config.pseudo_exist_region_l1_max) * 1.5,
                                    )
                                )
                                _degraded_weight_scale = float(
                                    getattr(config, "pseudo_degraded_accept_weight_scale", 0.3)
                                )
                                _max_unsupported_l1 = float(
                                    getattr(config, "pseudo_degraded_accept_max_unsupported_l1", 0.35)
                                )
                                _min_supported_coverage = float(
                                    getattr(config, "pseudo_degraded_accept_min_supported_coverage", 0.02)
                                )
                                _degraded_accept = (
                                    _supported_exist_l1 is not None
                                    and float(_supported_exist_l1) <= _degraded_l1_max
                                    and float(_supported_exist_coverage) >= _min_supported_coverage
                                    and (
                                        _unsupported_render_l1 is None
                                        or float(_unsupported_render_l1) <= _max_unsupported_l1
                                    )
                                    and (
                                        _warp_valid_ratio is None
                                        or float(_warp_valid_ratio) >= 0.05
                                    )
                                    and (
                                        _support_projection_ratio is None
                                        or float(_support_projection_ratio) >= 0.01
                                    )
                                )
                                if _degraded_accept:
                                    _consistency_weight = max(
                                        0.0,
                                        (1.0 - float(_supported_exist_l1) / max(_degraded_l1_max, 1e-6))
                                        * _degraded_weight_scale,
                                    )
                                    _generation_audit_entry["degraded_accept"] = True
                                    _generation_audit_entry["degraded_accept_supported_exist_l1"] = float(_supported_exist_l1)
                                    _generation_audit_entry["degraded_accept_unsupported_render_l1"] = (
                                        float(_unsupported_render_l1) if _unsupported_render_l1 is not None else None
                                    )
                                else:
                                    _pseudo_valid = False
                                    _reject_reason = f"exist_l1({_masked_l1_value:.4f})"
                            else:
                                _pseudo_valid = False
                                _reject_reason = f"exist_l1({_masked_l1_value:.4f})"

                _verified_evidence_mask, _verified_evidence_metrics = _build_verified_evidence_mask(
                    pseudo_rgb=pseudo_rgb,
                    debug_bundle=_consistency_debug_bundle,
                    generation_diagnostics=_generation_diagnostics,
                    config=config,
                )
                if len(_verified_evidence_metrics) > 0:
                    _generation_audit_entry["verified_evidence_metrics"] = {
                        metric_name: float(metric_value)
                        for metric_name, metric_value in _verified_evidence_metrics.items()
                    }
                _verified_evidence_coverage = float(
                    _verified_evidence_metrics.get(
                        "verified_evidence_coverage",
                        _compute_mask_coverage(_verified_evidence_mask),
                    )
                )
                _verified_evidence_score = float(
                    _verified_evidence_metrics.get(
                        "verified_evidence_score",
                        _verified_evidence_coverage * max(float(_consistency_weight), 0.0),
                    )
                )
                _generation_audit_entry["verified_evidence_coverage"] = float(_verified_evidence_coverage)
                _generation_audit_entry["verified_evidence_score"] = float(_verified_evidence_score)
                if _verified_evidence_mask is not None:
                    _verified_supported_input_coverage = float(
                        _verified_evidence_metrics.get("verified_supported_input_coverage", 0.0)
                    )
                    print(
                        "[Phase 2][VerifiedEvidence] "
                        f"coverage={_verified_evidence_coverage:.3f}, "
                        f"score={_verified_evidence_score:.4f}, "
                        f"supported_input_cov={_verified_supported_input_coverage:.3f}"
                    )
                if (
                    _pseudo_valid
                    and bool(getattr(config, "pseudo_require_verified_evidence", False))
                ):
                    _verified_min_coverage = float(
                        getattr(config, "pseudo_verified_min_coverage", 0.08)
                    )
                    if _verified_evidence_mask is None:
                        _pseudo_valid = False
                        _reject_reason = "missing_verified_evidence"
                    elif float(_verified_evidence_coverage) < _verified_min_coverage:
                        _pseudo_valid = False
                        _reject_reason = f"low_verified_coverage({_verified_evidence_coverage:.4f})"

                if _pseudo_valid and q_gain_calibrator is not None:
                    _q_gain_runtime_record = _build_q_gain_runtime_record(
                        selected_metrics=_selected_metrics,
                        consistency_metrics=_consistency_debug,
                        health_metrics=_health_metrics,
                    )
                    _q_gain_prediction = q_gain_calibrator.predict_from_record(_q_gain_runtime_record)
                    _q_gain = float(_q_gain_prediction.get("q_gain", _q_gain))
                    _q_gain_for_gate = float(_q_gain_prediction.get("q_gain_for_gate", _q_gain))
                    _generation_audit_entry["q_gain"] = _q_gain
                    _generation_audit_entry["q_gain_for_gate"] = _q_gain_for_gate
                    _generation_audit_entry["predicted_residual_delta_psnr"] = _q_gain_prediction.get(
                        "predicted_residual_delta_psnr"
                    )
                    _generation_audit_entry["gate_predicted_residual_delta_psnr"] = _q_gain_prediction.get(
                        "gate_predicted_residual_delta_psnr"
                    )
                    _generation_audit_entry["predicted_positive_gain_delta_psnr"] = _q_gain_prediction.get(
                        "predicted_positive_gain_delta_psnr"
                    )
                    _generation_audit_entry["predicted_harmful_risk_delta_psnr"] = _q_gain_prediction.get(
                        "predicted_harmful_risk_delta_psnr"
                    )
                    _generation_audit_entry["predicted_delta_psnr"] = _q_gain_prediction.get(
                        "predicted_delta_psnr"
                    )
                    _generation_audit_entry["baseline_delta_psnr_from_u_geo"] = _q_gain_prediction.get(
                        "baseline_delta_psnr_from_u_geo"
                    )
                    _generation_audit_entry["q_gain_feature_coverage"] = _q_gain_prediction.get(
                        "feature_coverage"
                    )
                    _generation_audit_entry["q_gain_missing_feature_count"] = _q_gain_prediction.get(
                        "missing_feature_count"
                    )
                    _generation_audit_entry["q_gain_gate_feature_coverage"] = _q_gain_prediction.get(
                        "gate_feature_coverage"
                    )
                    _generation_audit_entry["q_gain_gate_missing_feature_count"] = _q_gain_prediction.get(
                        "gate_missing_feature_count"
                    )
                    _generation_audit_entry["q_gain_risk_feature_coverage"] = _q_gain_prediction.get(
                        "risk_feature_coverage"
                    )
                    _generation_audit_entry["q_gain_risk_missing_feature_count"] = _q_gain_prediction.get(
                        "risk_missing_feature_count"
                    )
                    if (
                        bool(getattr(config, "pseudo_use_q_gain_gate", False))
                        and _q_gain_for_gate < float(getattr(config, "pseudo_min_q_gain", 0.5))
                    ):
                        _pseudo_valid = False
                        _reject_reason = f"low_q_gain({_q_gain_for_gate:.4f})"

                if not _pseudo_valid:
                    _round_rejected += 1
                    _round_reject_reasons.append(_reject_reason)
                    _generation_audit_entry["accepted"] = False
                    _generation_audit_entry["reject_reason"] = _reject_reason
                    _round_generation_diagnostics.append(_generation_audit_entry)
                    continue
                # ========== 质量门控结束 ==========

                _round_accepted += 1
                _generation_audit_entry["accepted"] = True
                _generation_audit_entry["consistency_weight"] = float(_consistency_weight)

                # 使用MC-Dropout方差作为uncertainty（如果模型未返回）
                if epi_map is None and hasattr(selected_item, 'epistemic_map'):
                    epi_map = selected_item.epistemic_map
                if ale_map is None and hasattr(selected_item, 'aleatoric_map'):
                    ale_map = selected_item.aleatoric_map
                if epi_map is None and hasattr(selected_item, 'variance_map'):
                    epi_map = selected_item.variance_map
                if risk_exist_map is None:
                    risk_exist_map = epi_map
                if gain_novel_map is None:
                    gain_novel_map = ale_map
                _phase2_exist_mask = (depth_render[:1, :1] > 1e-6).float()
                _phase2_novel_mask = torch.clamp(1.0 - _phase2_exist_mask, min=0.0, max=1.0)
                _phase2_frontier_mask, _phase2_unsupported_novel_mask = _compute_frontier_region_masks(
                    _phase2_exist_mask,
                    _phase2_novel_mask,
                    band_radius=int(getattr(config, "phase2_frontier_band_radius", 3)),
                )
                _phase2_supported_exist_mask = _consistency_debug_bundle.get("supported_exist_mask")
                _phase2_verified_evidence_mask = _resolve_verified_evidence_mask(
                    verified_evidence_mask=_verified_evidence_mask,
                    supported_exist_mask=_phase2_supported_exist_mask,
                    config=config,
                )
                _phase2_proposal_patches, _phase2_proposal_summary = _build_verified_proposal_patches(
                    pseudo_rgb=pseudo_rgb,
                    verified_evidence_mask=_phase2_verified_evidence_mask,
                    frontier_mask=_phase2_frontier_mask,
                    config=config,
                )
                _plan6_result = _run_plan6_protocol_for_phase2(
                    gen_result=gen_result,
                    pseudo_rgb=pseudo_rgb,
                    render_rgb=rgb_render,
                    depth_render=depth_render,
                    verified_evidence_mask=_phase2_verified_evidence_mask,
                    supported_exist_mask=_phase2_supported_exist_mask,
                    frontier_mask=_phase2_frontier_mask,
                    epistemic_map=epi_map,
                    aleatoric_map=ale_map,
                    config=config,
                    current_iter=round_idx,
                )
                _plan6_rgb_trust_map = None
                _plan6_depth_trust_map = None
                _plan6_alpha_trust_map = None
                _plan6_phase3_rgb_weight = None
                _plan6_phase3_depth_weight = None
                _plan6_phase3_alpha_weight = None
                _plan6_joint_trust_mask = None
                _plan6_proposal_depth = None
                _plan6_proposal_alpha = None
                _plan6_online_allowed = False
                _plan6_gate_reasons: List[str] = []
                _plan6_stage_metrics: Dict[str, Any] = {}
                if _plan6_result is not None:
                    _plan6_patches = _convert_plan6_patches_to_proposal_patches(_plan6_result)
                    if len(_plan6_patches) > 0:
                        _phase2_proposal_patches = _plan6_patches
                        _phase2_proposal_summary = {
                            "proposal_patch_count": float(len(_phase2_proposal_patches)),
                            "proposal_patch_area_ratio": float(sum(patch.score for patch in _phase2_proposal_patches)),
                            "proposal_patch_best_score": float(_phase2_proposal_patches[0].score),
                        }
                    _plan6_rgb_trust_map = _plan6_result.when_trust.p_rgb_trust.detach().cpu()
                    _plan6_depth_trust_map = _plan6_result.when_trust.p_depth_trust.detach().cpu()
                    _plan6_alpha_trust_map = _plan6_result.when_trust.p_alpha_trust.detach().cpu()
                    _plan6_phase3_rgb_weight = _plan6_result.when_trust.phase3_rgb_weight.detach().cpu()
                    _plan6_phase3_depth_weight = _plan6_result.when_trust.phase3_depth_weight.detach().cpu()
                    _plan6_phase3_alpha_weight = _plan6_result.when_trust.phase3_alpha_weight.detach().cpu()
                    _plan6_joint_trust_mask = _plan6_result.when_trust.joint_trust_mask.detach().cpu()
                    _plan6_proposal_depth = _extract_plan6_tensor(gen_result, "proposal_depth")
                    _plan6_proposal_alpha = _extract_plan6_tensor(gen_result, "proposal_alpha")
                    if isinstance(_plan6_proposal_depth, torch.Tensor):
                        _plan6_proposal_depth = _plan6_proposal_depth.detach().cpu()
                    if isinstance(_plan6_proposal_alpha, torch.Tensor):
                        _plan6_proposal_alpha = _plan6_proposal_alpha.detach().cpu()
                    _plan6_online_allowed = bool(_plan6_result.when_trust.online_admission_allowed)
                    _plan6_gate_reasons = list(_plan6_result.when_trust.gate_reasons)
                    _plan6_stage_metrics = {
                        "where_wrong": dict(_plan6_result.where_wrong.metrics),
                        "where_repairable": dict(_plan6_result.where_repairable.metrics),
                        "how_repairable": dict(_plan6_result.how_repairable.metrics),
                        "when_trust": dict(_plan6_result.when_trust.metrics),
                        "where_wrong_source": _plan6_result.where_wrong.source,
                        "proposal_source": _plan6_result.how_repairable.proposal_source,
                        "online_admission_allowed": _plan6_online_allowed,
                        "gate_reasons": _plan6_gate_reasons,
                    }
                    _generation_audit_entry["plan6_four_stage"] = _plan6_stage_metrics
                if len(_phase2_proposal_summary) > 0:
                    _generation_audit_entry["proposal_patch_summary"] = {
                        summary_name: float(summary_value)
                        for summary_name, summary_value in _phase2_proposal_summary.items()
                    }
                    _generation_audit_entry["proposal_patch_count"] = int(
                        _phase2_proposal_summary.get("proposal_patch_count", 0.0)
                    )
                if (
                    bool(getattr(config, "phase3_use_patch_proposals_only", False))
                    and len(_phase2_proposal_patches) == 0
                ):
                    _round_accepted -= 1
                    _round_rejected += 1
                    _reject_reason = "empty_verified_proposal"
                    _round_reject_reasons.append(_reject_reason)
                    _generation_audit_entry["accepted"] = False
                    _generation_audit_entry["reject_reason"] = _reject_reason
                    _round_generation_diagnostics.append(_generation_audit_entry)
                    continue

                pv = PseudoView(
                    camera=sel_candidate,
                    image=pseudo_rgb.detach().cpu() if pseudo_rgb is not None else torch.zeros(3, 256, 256),
                    epistemic_map=epi_map.detach().cpu() if epi_map is not None else None,
                    aleatoric_map=ale_map.detach().cpu() if ale_map is not None else None,
                    risk_exist_map=risk_exist_map.detach().cpu() if risk_exist_map is not None else None,
                    gain_novel_map=gain_novel_map.detach().cpu() if gain_novel_map is not None else None,
                    exist_mask=_phase2_exist_mask.detach().cpu(),
                    novel_mask=_phase2_novel_mask.detach().cpu(),
                    frontier_mask=_phase2_frontier_mask.detach().cpu() if _phase2_frontier_mask is not None else None,
                    supported_exist_mask=(
                        _phase2_supported_exist_mask.detach().cpu()
                        if isinstance(_phase2_supported_exist_mask, torch.Tensor)
                        else None
                    ),
                    unsupported_novel_mask=(
                        _phase2_unsupported_novel_mask.detach().cpu()
                        if _phase2_unsupported_novel_mask is not None
                        else None
                    ),
                    verified_evidence_mask=(
                        _phase2_verified_evidence_mask.detach().cpu()
                        if isinstance(_phase2_verified_evidence_mask, torch.Tensor)
                        else None
                    ),
                    verified_evidence_coverage=float(_verified_evidence_coverage),
                    verified_evidence_score=float(_verified_evidence_score),
                    score=selected_item.score if hasattr(selected_item, 'score') else 0.0,
                    q_gen=_q_gen,
                    q_gain=_q_gain,
                    q_gain_for_gate=_generation_audit_entry.get("q_gain_for_gate", _q_gain),
                    predicted_residual_delta_psnr=float(
                        _generation_audit_entry.get("predicted_residual_delta_psnr", 0.0) or 0.0
                    ),
                    consistency_weight=_consistency_weight,
                    width=int(pseudo_rgb.shape[-1]) if pseudo_rgb is not None else 0,
                    height=int(pseudo_rgb.shape[-2]) if pseudo_rgb is not None else 0,
                    proposal_patches=_phase2_proposal_patches,
                    plan6_rgb_trust_map=_plan6_rgb_trust_map,
                    plan6_depth_trust_map=_plan6_depth_trust_map,
                    plan6_alpha_trust_map=_plan6_alpha_trust_map,
                    plan6_phase3_rgb_weight=_plan6_phase3_rgb_weight,
                    plan6_phase3_depth_weight=_plan6_phase3_depth_weight,
                    plan6_phase3_alpha_weight=_plan6_phase3_alpha_weight,
                    plan6_joint_trust_mask=_plan6_joint_trust_mask,
                    plan6_proposal_depth=_plan6_proposal_depth,
                    plan6_proposal_alpha=_plan6_proposal_alpha,
                    plan6_online_admission_allowed=_plan6_online_allowed,
                    plan6_gate_reasons=_plan6_gate_reasons,
                    plan6_stage_metrics=_plan6_stage_metrics,
                )
                all_pseudo_views.append(pv)

                # 保存到磁盘
                if pseudo_save_dir is not None:
                    pseudo_save_dir.mkdir(parents=True, exist_ok=True)
                    idx = len(all_pseudo_views) - 1
                    file_name = f"pseudo_{idx:04d}.png"
                    pv.img_name = file_name
                    if pseudo_rgb is not None:
                        img_tensor = pseudo_rgb.detach().cpu().clamp(0, 1).permute(1, 2, 0)
                        img_pil = Image.fromarray((img_tensor * 255).byte().numpy())
                        img_pil.save(pseudo_save_dir / file_name)
                    if epi_map is not None:
                        torch.save(epi_map.detach().cpu(), pseudo_save_dir / f"epistemic_{idx:04d}.pt")
                    if ale_map is not None:
                        torch.save(ale_map.detach().cpu(), pseudo_save_dir / f"aleatoric_{idx:04d}.pt")
                    torch.save(sel_candidate.viewmat.cpu(), pseudo_save_dir / f"viewmat_{idx:04d}.pt")
                    torch.save(sel_candidate.intrinsics.cpu(), pseudo_save_dir / f"intrinsics_{idx:04d}.pt")
                    evidence_dir = pseudo_save_dir / "evidence_maps"
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    evidence_tag = f"round{round_idx + 1:02d}_pseudo{idx:04d}"
                    if pseudo_rgb is not None:
                        _save_rgb_tensor(evidence_dir / f"{evidence_tag}_pseudo_repair.png", pseudo_rgb.detach().cpu())
                    _save_rgb_tensor(evidence_dir / f"{evidence_tag}_sparse_render.png", rgb_render.detach().cpu())
                    if isinstance(_phase2_verified_evidence_mask, torch.Tensor):
                        _save_map_tensor(
                            evidence_dir / f"{evidence_tag}_verified_evidence_mask.png",
                            _phase2_verified_evidence_mask.detach().cpu(),
                        )
                    if _plan6_result is not None:
                        _save_map_tensor(
                            evidence_dir / f"{evidence_tag}_typed_failure_map.png",
                            _plan6_result.where_wrong.p_wrong.detach().cpu(),
                        )
                    if isinstance(_plan6_rgb_trust_map, torch.Tensor):
                        _save_map_tensor(
                            evidence_dir / f"{evidence_tag}_rgb_trust.png",
                            _plan6_rgb_trust_map.detach().cpu(),
                        )
                    if isinstance(_plan6_depth_trust_map, torch.Tensor):
                        _save_map_tensor(
                            evidence_dir / f"{evidence_tag}_depth_trust.png",
                            _plan6_depth_trust_map.detach().cpu(),
                        )
                    if isinstance(_plan6_alpha_trust_map, torch.Tensor):
                        _save_map_tensor(
                            evidence_dir / f"{evidence_tag}_alpha_trust.png",
                            _plan6_alpha_trust_map.detach().cpu(),
                        )

                _round_generation_diagnostics.append(_generation_audit_entry)

            except Exception as e:
                _round_rejected += 1
                _round_reject_reasons.append(f"exception({e})")
                _generation_audit_entry["accepted"] = False
                _generation_audit_entry["reject_reason"] = f"exception({e})"
                _generation_audit_entry["error"] = str(e)
                _round_generation_diagnostics.append(_generation_audit_entry)
                continue

        if pseudo_save_dir is not None and len(_round_generation_diagnostics) > 0:
            generation_diagnostic_dir = pseudo_save_dir / "generation_diagnostics"
            generation_diagnostic_dir.mkdir(parents=True, exist_ok=True)
            generation_diagnostic_payload = {
                "round_index_1based": round_idx + 1,
                "gate_threshold": float(config.pseudo_exist_region_l1_max),
                "accepted_count": int(_round_accepted),
                "rejected_count": int(_round_rejected),
                "diagnostic_summary": _build_round_generation_diagnostic_summary(_round_generation_diagnostics),
                "candidates": _round_generation_diagnostics,
            }
            with (generation_diagnostic_dir / f"round_{round_idx + 1:02d}.json").open(
                "w",
                encoding="utf-8",
            ) as file_handle:
                json.dump(generation_diagnostic_payload, file_handle, ensure_ascii=False, indent=2)

        # 轮次汇总（替代逐条警告）
        if _round_rejected > 0:
            # 统计拒绝原因
            from collections import Counter
            _reason_counts = Counter(_round_reject_reasons)
            _reason_summary = ", ".join(f"{r}×{c}" for r, c in _reason_counts.most_common(3))
            print(f"[Phase 2] 第{round_idx+1}轮汇总: ✅接受={_round_accepted}, ❌拒绝={_round_rejected} ({_reason_summary})")
        else:
            print(f"[Phase 2] 第{round_idx+1}轮汇总: ✅全部接受={_round_accepted}")

    print(f"\n[Phase 2] ===== 主动采样完成 =====")
    print(f"[Phase 2] 共生成 {len(all_pseudo_views)} 个伪视图")
    return all_pseudo_views


def train_3dgs_with_active_sampling(
    gaussians,
    real_cameras: List[ExistingCamera],
    real_images: torch.Tensor,
    gddn_model: torch.nn.Module,
    num_iterations: int,
    sparse_bundle: SparseViewBundle,
    config: ActiveSamplingConfig,
    *,
    scene_center: Optional[torch.Tensor] = None,
    condition_provider: Optional[ConditionProvider] = None,
    writer: Optional["SummaryWriter"] = None,
    stability_strategy: str = "none",
    pseudo_save_dir: Optional[Path] = None,
    candidate_log_path: Optional[Path] = None,
    dist_context: Optional[DistributedActiveSamplingContext] = None,
    pseudo_views: Optional[List[PseudoView]] = None,
    current_phase: int = 0,
    resume_training_state: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Illustrative training loop with periodic active sampling.
    
    Args:
        pseudo_views: Phase 2生成的伪视图列表（Phase 3使用）
        current_phase: 当前Phase（1=纯3DGS预训练, 3=伪视图增强训练, 0=原始模式）
        resume_training_state: 单进程 Phase 1 的完整状态；num_iterations 是总步数。
            结束后 gaussians._training_state 含参数、Adam、密度统计、RNG 和输入/配置。
            多进程和伪视图阶段不声明完整续训支持。
    
    Returns:
        List of pseudo camera dicts for 3DGS export.
    """
    validate_field(gaussians)
    if hasattr(gaussians, "_training_state"):
        del gaussians._training_state
    device = real_images.device
    dist_ctx = dist_context or DistributedActiveSamplingContext()
    is_main_rank = (not dist_ctx.enabled) or dist_ctx.is_main
    if scene_center is None:
        # 使用真实相机位置均值作为球心（伪视角很相机群局）
        if real_cameras:
            _cam_positions = torch.stack([cam.position.detach() for cam in real_cameras])
            _cam_center2 = _cam_positions.mean(dim=0).to(device)
            _fwd_dirs2 = []
            for cam in real_cameras:
                _fwd2 = cam.viewmat[:3, 2].detach().to(device)
                _fwd2 = _fwd2 / (_fwd2.norm() + 1e-8)
                _fwd_dirs2.append(_fwd2)
            _avg_fwd2 = torch.stack(_fwd_dirs2).mean(dim=0)
            _avg_fwd2 = _avg_fwd2 / (_avg_fwd2.norm() + 1e-8)
            if hasattr(gaussians, "means"):
                _gauss_center2 = gaussians.means.detach().mean(dim=0)
                _depth_proj2 = torch.dot(_gauss_center2 - _cam_center2, _avg_fwd2).item()
            else:
                _depth_proj2 = 1.0
            _depth_proj2 = max(abs(float(_depth_proj2)), 1.0)
            scene_center = _cam_center2 + 0.05 * _depth_proj2 * _avg_fwd2
        elif hasattr(gaussians, "means"):
            scene_center = gaussians.means.detach().mean(dim=0)
        else:
            scene_center = torch.zeros(3, device=device)
    else:
        scene_center = scene_center.detach()
    scene_center = scene_center.to(device)
    default_intrinsics = real_cameras[0].intrinsics.to(device)
    provider = condition_provider or example_condition_provider

    sampler = ViewpointSampler(config)
    estimator = UncertaintyEstimator(
        model=gddn_model if config.estimator_mode == "gddn" else None,
        config=config,
        device=device,
        condition_provider=provider,
        writer=writer,
        mode=config.estimator_mode,
        gaussians=gaussians,
        rasterizer=rasterization,
    )
    selector = ActiveSelector(config)
    # 初始化软门控机制（模块三）
    soft_gating = SoftGatedGuidance(config) if config.gate_enable else None
    nll_loss = (
        RobustHeteroscedasticLoss(
            epsilon=float(config.nll_variance_epsilon),
            huber_delta=float(config.nll_huber_delta),
        )
        if getattr(config, "nll_enable", False)
        else None
    )
    tau_ema_value: Optional[float] = None

    base_lr = 1e-3
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=base_lr)
    selected_views: List[ViewCandidate] = []
    if is_main_rank and pseudo_save_dir is not None and config.save_pseudo_png:
        pseudo_save_dir.mkdir(parents=True, exist_ok=True)
    candidate_log_file = None
    if is_main_rank and candidate_log_path is not None:
        candidate_log_path.parent.mkdir(parents=True, exist_ok=True)
        candidate_log_file = candidate_log_path.open("a", encoding="utf-8")

    if dist_ctx.enabled and not is_main_rank:
        _run_distributed_worker_loop(
            estimator=estimator,
            sparse_bundle=sparse_bundle,
            dist_context=dist_ctx,
        )
        if candidate_log_file is not None:
            candidate_log_file.close()
        return  # Worker rank返回，只有rank 0继续执行主循环

    def _resize_map_to_hw(tensor: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if tensor.shape[-2:] == target_hw:
            return tensor
        if tensor.dim() == 2:
            expand = tensor.unsqueeze(0).unsqueeze(0)
        elif tensor.dim() == 3:
            expand = tensor.unsqueeze(0)
        else:
            expand = tensor
        resized = F.interpolate(
            expand.unsqueeze(0) if expand.dim() == 2 else expand,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze()

    def _unproject_pixels(
        candidate: ViewCandidate,
        x_idx: torch.Tensor,
        y_idx: torch.Tensor,
        depth_values: torch.Tensor,
    ) -> torch.Tensor:
        intr = candidate.intrinsics.to(device)
        viewmat = candidate.viewmat.to(device)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        z_vals = depth_values.to(device)
        xs = (x_idx.float() - cx) * z_vals / fx
        ys = (y_idx.float() - cy) * z_vals / fy
        ones = torch.ones_like(z_vals)
        cam_coords = torch.stack([xs, ys, z_vals, ones], dim=-1)
        c2w = torch.inverse(viewmat)
        world_pts = (cam_coords @ c2w.T)[..., :3]
        return world_pts

    def _append_gaussians_to_scene(
        new_positions: torch.Tensor, new_colors: torch.Tensor
    ) -> None:
        nonlocal grad2d_accum, count_accum
        if new_positions.numel() == 0:
            return
        count = new_positions.shape[0]
        with torch.no_grad():
            new_positions = new_positions.to(gaussians.means.device)
            base_scales = gaussians.scales.detach()
            if base_scales.numel() == 0:
                scale_template = torch.full(
                    (3,), 0.01, device=device, dtype=gaussians.scales.dtype
                )
            else:
                scale_template = torch.median(base_scales, dim=0).values.to(
                    device=device, dtype=gaussians.scales.dtype
                )
            replicated_scales = scale_template.unsqueeze(0).repeat(count, 1)
            new_opacity = torch.full(
                (count,),
                float(config.dacd_spawn_opacity),
                device=device,
                dtype=gaussians.opacities.dtype,
            )
            new_quats = torch.zeros(count, 4, device=device)
            new_quats[:, 0] = 1.0
            clipped_colors = new_colors.clamp(1e-4, 1.0 - 1e-4)
            color_logits = torch.logit(clipped_colors).to(gaussians._color_logits.dtype)
            additions = dict(zip(PARAMETERS, (
                new_positions, new_quats.to(gaussians.quats.dtype), replicated_scales,
                new_opacity, color_logits.to(device),
            )))
            old_count = len(gaussians.means)
            values = {name: torch.cat((getattr(gaussians, name).detach(), additions[name]))
                      for name in PARAMETERS}
            rows = torch.cat((torch.arange(old_count, device=device),
                              torch.full((count,), -1, device=device, dtype=torch.long)))
            replace_rows(gaussians, optimizer, values, rows)
            # 新增点尚未积累屏幕梯度；保留旧点统计。
            grad2d_accum = torch.cat((grad2d_accum, grad2d_accum.new_zeros(count)))
            count_accum = torch.cat((count_accum, count_accum.new_zeros(count)))

    def _maybe_run_dacd(
        selected_result: UncertaintyResult,
        opacity_map: torch.Tensor,
        iteration_idx: int,
    ) -> None:
        metrics = getattr(selected_result, "metrics", None)
        anchor_ratio = -1.0
        r2_value = -1.0
        if isinstance(metrics, dict):
            anchor_ratio = float(metrics.get("depth_anchor_ratio", -1.0))
            r2_value = float(metrics.get("depth_r2", -1.0))

        if not config.dacd_enable:
            return
        if iteration_idx % max(1, int(config.dacd_interval)) != 0:
            return
        if selected_result.calibrated_depth is None or selected_result.epistemic_map is None:
            return
        target_hw = opacity_map.shape[-2:]
        teacher_epistemic = torch.nan_to_num(
            _resize_map_to_hw(
                selected_result.epistemic_map.detach().to(device), target_hw
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        depth_calib = torch.nan_to_num(
            _resize_map_to_hw(
                selected_result.calibrated_depth.detach().to(device), target_hw
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        opacity_local = opacity_map.to(device)
        void_mask = opacity_local < float(config.dacd_void_opacity_threshold)
        confident_mask = teacher_epistemic < float(config.dacd_safe_epistemic_threshold)
        foreground_mask = depth_calib < float(config.dacd_far_plane)
        spawn_mask = void_mask & confident_mask & foreground_mask
        if not spawn_mask.any():
            return
        max_spawn = int(max(1, config.dacd_max_spawn_per_view))
        indices = torch.nonzero(spawn_mask, as_tuple=False)
        if indices.shape[0] > max_spawn:
            choice = torch.randperm(indices.shape[0], device=indices.device)[:max_spawn]
            indices = indices[choice]
        y_idx = indices[:, 0]
        x_idx = indices[:, 1]
        depths = depth_calib[y_idx, x_idx]
        positive_mask = depths > 1e-3
        if not positive_mask.any():
            return
        y_idx = y_idx[positive_mask]
        x_idx = x_idx[positive_mask]
        depths = depths[positive_mask]
        teacher_rgb = selected_result.mean_rgb.detach().to(device)
        if teacher_rgb.shape[-2:] != target_hw:
            teacher_rgb = F.interpolate(
                teacher_rgb.unsqueeze(0),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        spawn_colors = teacher_rgb[:, y_idx, x_idx].permute(1, 0)
        world_points = _unproject_pixels(
            selected_result.candidate,
            x_idx.to(device),
            y_idx.to(device),
            depths,
        )
        _append_gaussians_to_scene(world_points, spawn_colors)
        print(
            f"[DACD] iter {iteration_idx}: spawned {world_points.shape[0]} gaussians "
            f"(anchor_ratio={anchor_ratio:.4f}, r2={r2_value:.4f})"
        )

    # ========== Clone/Split致密化策略初始化 ==========
    # 策略参数（参考DefaultStrategy默认值）
    refine_start_iter = getattr(config, 'refine_start_iter', 100)
    refine_every = getattr(config, 'refine_every', 100)
    refine_stop_iter = getattr(config, 'refine_stop_iter', 15000)
    grow_grad2d = getattr(config, 'grow_grad2d', 0.0002)
    grow_scale3d = getattr(config, 'grow_scale3d', 0.01)
    prune_opa = getattr(config, 'prune_opa', 0.005)
    
    # 计算场景尺度
    scene_scale = 1.0
    if hasattr(gaussians, 'means') and gaussians.means.numel() > 0:
        scene_extent = gaussians.means.detach().max(dim=0)[0] - gaussians.means.detach().min(dim=0)[0]
        scene_scale = scene_extent.norm().item() / 3.0
    
    # 梯度累积变量
    grad2d_accum = torch.zeros(gaussians.means.shape[0], device=device)
    count_accum = torch.zeros(gaussians.means.shape[0], device=device, dtype=torch.int32)
    
    def _maybe_grow_gs(iteration_idx: int, info: Dict) -> Tuple[int, int, int]:
        """执行Clone/Split/Prune操作
        
        Returns:
            (n_dupli, n_split, n_prune): 重复、分裂、剪枝的高斯体数量
        """
        nonlocal optimizer, gaussians, grad2d_accum, count_accum

        
        if iteration_idx < refine_start_iter or iteration_idx >= refine_stop_iter:
            return 0, 0, 0
        if iteration_idx % refine_every != 0:
            return 0, 0, 0
        
        n_gs = gaussians.means.shape[0]
        if n_gs == 0:
            return 0, 0, 0
        
        # 无增长候选时仍允许独立裁剪低 alpha 基元。
        grads = grad2d_accum / count_accum.clamp_min(1).float()
        n_dupli, n_split, n_prune = refine_field(
            gaussians, optimizer, grads, grow_scale3d * scene_scale, grow_grad2d, prune_opa
        )

        # 重置累积变量
        grad2d_accum = torch.zeros(gaussians.means.shape[0], device=device)
        count_accum = torch.zeros(gaussians.means.shape[0], device=device, dtype=torch.int32)
        
        print(f"[Densify] iter {iteration_idx}: {n_dupli} duplicated, {n_split} split, {n_prune} pruned. "
              f"Now having {gaussians.means.shape[0]} gaussians.")
        
        return n_dupli, n_split, n_prune

    # ========== Phase 3伪视图设置（如果可用） ==========
    pseudo_start_idx = len(real_cameras)  # 伪视角起始索引
    pseudo_uncertainty_maps = {}          # 伪视角的不确定性图
    pseudo_mode = getattr(config, 'pseudo_mode', 'guidance_only')
    phase3_real_only_training = bool(getattr(config, "phase3_real_only_training", True))
    phase3_patch_proposals_only = bool(getattr(config, "phase3_use_patch_proposals_only", False))
    effective_pseudo_mode = pseudo_mode
    if (
        current_phase == 3
        and pseudo_views
        and (phase3_real_only_training or phase3_patch_proposals_only)
        and pseudo_mode in ("direct", "weighted")
    ):
        force_reasons: List[str] = []
        if phase3_real_only_training:
            force_reasons.append("phase3_real_only_training=True")
        if phase3_patch_proposals_only:
            force_reasons.append("phase3_use_patch_proposals_only=True")
        print(
            f"[Phase 3] {' 且 '.join(force_reasons)}，"
            f"伪视图模式已从 {pseudo_mode} 强制切换为 guidance_only"
        )
        effective_pseudo_mode = "guidance_only"

    if pseudo_views and current_phase == 3 and effective_pseudo_mode in ("direct", "weighted"):
        print(f"[Phase 3] 加载 {len(pseudo_views)} 个伪视图 (模式={effective_pseudo_mode})")
        pseudo_cameras_list = []
        pseudo_images_list = []
        for pv_idx, pv in enumerate(pseudo_views):
            pseudo_cameras_list.append(
                ExistingCamera(
                    position=pv.camera.position.to(device),
                    viewmat=pv.camera.viewmat.to(device),
                    intrinsics=pv.camera.intrinsics.to(device),
                )
            )
            pv_img = pv.image.to(device)
            if pv_img.dim() == 4 and pv_img.shape[0] == 1:
                pv_img = pv_img[0]
            if pv_img.dim() != 3:
                raise ValueError(
                    f"PseudoView.image must be [3,H,W] or [1,3,H,W], got shape={tuple(pv_img.shape)}"
                )
            if pv_img.shape[-2:] != real_images.shape[-2:]:
                pv_img = F.interpolate(
                    pv_img.unsqueeze(0), size=real_images.shape[-2:],
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
            pseudo_images_list.append(pv_img)
            pseudo_uncertainty_entry: Dict[str, Any] = {
                "verified_evidence_mask": pv.verified_evidence_mask,
                "verified_evidence_coverage": float(getattr(pv, "verified_evidence_coverage", 0.0)),
                "verified_evidence_score": float(getattr(pv, "verified_evidence_score", 0.0)),
                "supported_exist_mask": pv.supported_exist_mask,
                "plan6_rgb_trust_map": getattr(pv, "plan6_rgb_trust_map", None),
                "plan6_depth_trust_map": getattr(pv, "plan6_depth_trust_map", None),
                "plan6_alpha_trust_map": getattr(pv, "plan6_alpha_trust_map", None),
                "plan6_phase3_rgb_weight": getattr(pv, "plan6_phase3_rgb_weight", None),
                "plan6_phase3_depth_weight": getattr(pv, "plan6_phase3_depth_weight", None),
                "plan6_phase3_alpha_weight": getattr(pv, "plan6_phase3_alpha_weight", None),
                "plan6_joint_trust_mask": getattr(pv, "plan6_joint_trust_mask", None),
                "plan6_proposal_depth": getattr(pv, "plan6_proposal_depth", None),
                "plan6_proposal_alpha": getattr(pv, "plan6_proposal_alpha", None),
                "plan6_online_admission_allowed": bool(getattr(pv, "plan6_online_admission_allowed", False)),
                "plan6_gate_reasons": list(getattr(pv, "plan6_gate_reasons", [])),
            }
            if effective_pseudo_mode == "weighted":
                pseudo_uncertainty_entry.update({
                    "epistemic": pv.epistemic_map,
                    "aleatoric": pv.aleatoric_map,
                    "risk_exist": pv.risk_exist_map,
                    "gain_novel": pv.gain_novel_map,
                    "exist_mask": pv.exist_mask,
                    "novel_mask": pv.novel_mask,
                    "frontier_mask": pv.frontier_mask,
                    "unsupported_novel_mask": pv.unsupported_novel_mask,
                    "q_gen": pv.q_gen,  # DDUD维度2: 生成可信度，用于Phase 3 loss降权
                    "q_gain": pv.q_gain,  # 收益侧 q_gain，用于 Phase 3 权重
                    "q_gain_for_gate": pv.q_gain_for_gate,
                    "predicted_residual_delta_psnr": pv.predicted_residual_delta_psnr,
                    "consistency_weight": pv.consistency_weight,
                })
            pseudo_uncertainty_maps[pseudo_start_idx + pv_idx] = pseudo_uncertainty_entry
        real_cameras = list(real_cameras) + pseudo_cameras_list
        real_images = torch.cat([real_images, torch.stack(pseudo_images_list)], dim=0)
        print(f"[Phase 3] 训练视角扩展为 {len(real_cameras)} 个 "
              f"(真实={pseudo_start_idx}, 伪={len(pseudo_views)})")
    elif pseudo_views and current_phase == 3:
        print(
            f"[Phase 3] 加载 {len(pseudo_views)} 个伪视图 (模式={effective_pseudo_mode})，"
            "仅作为verified guidance，不加入训练相机池"
        )

    # ========== Phase 1收敛检测初始化 ==========
    loss_history: List[float] = []
    psnr_history: List[float] = []

    start_iteration = 0
    completed_iterations = 0
    if resume_training_state is not None:
        if current_phase != 1 or dist_ctx.enabled or pseudo_views:
            raise ValueError("完整续训目前仅支持单进程、无伪视图的 Phase 1")
        loop = resume_training_state["loop"]
        if loop["current_phase"] != 1 or loop["config"] != vars(config):
            raise ValueError("续训阶段或配置与保存状态不一致")
        if loop["stability_strategy"] != stability_strategy:
            raise ValueError("续训优化策略不一致")
        if not torch.equal(loop["real_images"], real_images.detach().cpu()) or any(
            not torch.equal(saved, camera.viewmat.detach().cpu()) or
            not torch.equal(intrinsics, camera.intrinsics.detach().cpu())
            for (saved, intrinsics), camera in zip(loop["cameras"], real_cameras)
        ) or len(loop["cameras"]) != len(real_cameras):
            raise ValueError("续训图像或相机与保存状态不一致")
        if num_iterations < loop["completed_iterations"]:
            raise ValueError("num_iterations 必须是大于等于已完成步数的总预算")
        loop = restore_training_state(gaussians, optimizer, resume_training_state)
        start_iteration = completed_iterations = loop["completed_iterations"]
        grad2d_accum = loop["grad2d_accum"].to(device)
        count_accum = loop["count_accum"].to(device)
        scene_scale = loop["scene_scale"]
        loss_history, psnr_history = loop["loss_history"], loop["psnr_history"]

    # ========== 训练循环启动日志 ==========
    phase_label = f"Phase {current_phase}" if current_phase > 0 else "Standard"
    print(f"[Training] ========== 3DGS训练循环开始 ({phase_label}) ==========")
    print(f"[Training] num_iterations={num_iterations}")
    print(f"[Training] warmup_iterations={config.warmup_iterations}")
    print(f"[Training] guidance_interval={config.guidance_interval}")
    print(f"[Training] num_views_per_round={config.num_views_per_round}")
    print(f"[Training] estimator_mode={config.estimator_mode}")
    if current_phase == 3 and pseudo_views:
        print(
            f"[Training] pseudo_mode={effective_pseudo_mode}, pseudo_views={len(pseudo_views)}, "
            f"real_only_training={phase3_real_only_training}"
        )
    print(f"[Training] =============================================")
    
    try:
        from tqdm import tqdm as _tqdm
        _pbar = _tqdm(range(start_iteration, num_iterations), desc="Training", unit="it",
                      dynamic_ncols=True, leave=True)
    except ImportError:
        _pbar = None

    for iteration in ((_pbar) if _pbar is not None else range(start_iteration, num_iterations)):
        
        # 计算当前迭代的动态λ值（模块四）
        if current_phase == 1:
            schedule_iteration = 0
            lambda_current = 0.0
            phase_name, phase_int = ("scaffold", 0)
        elif current_phase == 3:
            schedule_iteration = int(config.lambda_peak_iteration) + iteration
            lambda_current = compute_dynamic_lambda(schedule_iteration, config)
            phase_name, phase_int = ("refinement", 2)
        else:
            schedule_iteration = iteration
            lambda_current = compute_dynamic_lambda(schedule_iteration, config)
            phase_name, phase_int = get_current_phase(schedule_iteration, config)

        # Phase 3: 真实视角优先采样策略
        # 原来均匀轮换时真实视角采样率仅 3/(3+20)=13%，伪视角梯度贡献
        # 0.87×0.2=0.174 > 真实视角 0.13×1.0，会破坏真实纹理
        # 修复：Phase 3 中提高真实视角采样率，避免伪视角梯度淹没真实纹理
        if (
            current_phase == 3
            and pseudo_views
            and pseudo_start_idx > 0
            and effective_pseudo_mode in ("direct", "weighted")
        ):
            real_view_probability = float(getattr(config, "phase3_real_view_probability", 0.5))
            if random.random() < real_view_probability:
                camera_index = iteration % pseudo_start_idx
            else:
                camera_index = pseudo_start_idx + random.randint(0, len(real_cameras) - pseudo_start_idx - 1)
        else:
            camera_index = iteration % len(real_cameras)
        camera = real_cameras[camera_index]
        gt_image = real_images[camera_index]
        
        # Clone trainable tensors before passing to CUDA kernels to avoid
        # potential in-place mutations on leaf variables during forward.
        # 使用absgrad=True来获取means2d的绝对梯度用于致密化
        means_clone = gaussians.means.clone()
        quats_clone = gaussians.quats.clone()
        scales_clone = gaussians.scales.clone()
        opacities_clone = gaussians.opacities.clone()
        colors_clone = gaussians.colors.clone()
        is_phase3_pseudo_camera = (
            current_phase == 3
            and camera_index >= pseudo_start_idx
            and effective_pseudo_mode in ("direct", "weighted")
        )
        render_mode_for_iteration = (
            "RGB+ED"
            if (
                is_phase3_pseudo_camera
                and bool(getattr(config, "phase3_use_plan6_depth_alpha_losses", False))
            )
            else "RGB"
        )
        
        render_colors, render_alphas, info = rasterization(
            means=means_clone,
            quats=quats_clone,
            scales=scales_clone,
            opacities=opacities_clone,
            colors=colors_clone,
            viewmats=camera.viewmat.unsqueeze(0),
            Ks=camera.intrinsics.unsqueeze(0),
            width=gt_image.shape[-1],
            height=gt_image.shape[-2],
            render_mode=render_mode_for_iteration,
            absgrad=True,  # 启用绝对梯度计算
        )
        # 修复：means2d是从clone()派生的非叶张量，必须显式retain_grad()
        # 否则backward()不会填充其.grad，导致densification梯度累积永远为0
        if "means2d" in info:
            info["means2d"].retain_grad()
        render_rgb = render_colors[0, ..., :3].permute(2, 0, 1).contiguous()
        loss_photo = F.l1_loss(render_rgb, gt_image)
        photo_optim_loss = loss_photo
        loss_guidance = torch.tensor(0.0, device=device)
        guidance_scale = lambda_current

        # ========== Phase 3: 伪视角加权处理 ==========
        # 使用固定权重0.2（独立于λ调度），避免warmup_iterations=phase3_iters
        # 导致lambda_current=0使伪视图loss完全无效化
        _pseudo_loss_weight = float(getattr(config, "pseudo_loss_weight", 0.1))
        is_pseudo_view = (current_phase == 3 and camera_index >= pseudo_start_idx
                          and effective_pseudo_mode in ("direct", "weighted"))
        if is_pseudo_view:
            pseudo_runtime_data = pseudo_uncertainty_maps.get(camera_index, {})
            verified_only_enabled = bool(getattr(config, "phase3_use_verified_evidence_only", False))
            verified_evidence_mask_dev = None
            if verified_only_enabled:
                verified_evidence_mask_map = _resolve_verified_evidence_mask(
                    verified_evidence_mask=pseudo_runtime_data.get("verified_evidence_mask"),
                    supported_exist_mask=pseudo_runtime_data.get("supported_exist_mask"),
                    config=config,
                )
                if verified_evidence_mask_map is not None:
                    verified_evidence_mask_dev = _resize_map_to_hw(
                        verified_evidence_mask_map.to(device),
                        render_rgb.shape[-2:],
                    )
            plan6_weight_dev = None
            plan6_blocked = False
            plan6_online_allowed = bool(
                pseudo_runtime_data.get("plan6_online_admission_allowed", False)
            )
            if bool(getattr(config, "phase3_use_plan6_trust_weights", False)):
                plan6_weight_map = pseudo_runtime_data.get("plan6_phase3_rgb_weight")
                if not plan6_online_allowed or plan6_weight_map is None:
                    plan6_blocked = True
                else:
                    plan6_weight_dev = _resize_map_to_hw(
                        plan6_weight_map.to(device),
                        render_rgb.shape[-2:],
                    )
            if effective_pseudo_mode == "weighted" and camera_index in pseudo_uncertainty_maps:
                # weighted模式：使用SoftGating加权
                unc_data = pseudo_runtime_data
                epi = unc_data.get("epistemic")
                ale = unc_data.get("aleatoric")
                risk_exist = unc_data.get("risk_exist", epi)
                gain_novel = unc_data.get("gain_novel", ale)
                exist_mask_map = unc_data.get("exist_mask")
                novel_mask_map = unc_data.get("novel_mask")
                frontier_mask_map = unc_data.get("frontier_mask")
                supported_exist_mask_map = unc_data.get("supported_exist_mask")
                unsupported_novel_mask_map = unc_data.get("unsupported_novel_mask")
                if plan6_blocked:
                    photo_optim_loss = torch.zeros_like(loss_photo)
                elif plan6_weight_dev is not None:
                    photo_optim_loss = torch.zeros_like(loss_photo)
                    _plan6_guidance_value = _compute_weighted_masked_l1_loss(
                        render_rgb,
                        gt_image,
                        weight_map=plan6_weight_dev,
                    )
                    if _plan6_guidance_value is not None:
                        _plan6_guidance_loss = _plan6_guidance_value * _pseudo_loss_weight
                        loss_guidance = loss_guidance + _plan6_guidance_loss
                        guidance_scale = 1.0
                        if writer is not None and iteration % 100 == 0:
                            writer.add_scalar("loss/phase3_plan6_guidance", float(_plan6_guidance_loss.item()), iteration)
                            writer.add_scalar("gating/phase3_plan6_weight_coverage", float((plan6_weight_dev > 1e-6).float().mean().item()), iteration)
                elif risk_exist is not None and gain_novel is not None:
                    risk_dev = _resize_map_to_hw(risk_exist.to(device), render_rgb.shape[-2:])
                    gain_dev = _resize_map_to_hw(gain_novel.to(device), render_rgb.shape[-2:])
                    if exist_mask_map is None:
                        exist_mask_dev = torch.ones_like(risk_dev)
                    else:
                        exist_mask_dev = _resize_map_to_hw(exist_mask_map.to(device), render_rgb.shape[-2:])
                    if novel_mask_map is None:
                        novel_mask_dev = torch.clamp(1.0 - exist_mask_dev, min=0.0, max=1.0)
                    else:
                        novel_mask_dev = _resize_map_to_hw(novel_mask_map.to(device), render_rgb.shape[-2:])
                    frontier_mask_dev = (
                        None
                        if frontier_mask_map is None
                        else _resize_map_to_hw(frontier_mask_map.to(device), render_rgb.shape[-2:])
                    )
                    supported_exist_mask_dev = (
                        None
                        if supported_exist_mask_map is None
                        else _resize_map_to_hw(supported_exist_mask_map.to(device), render_rgb.shape[-2:])
                    )
                    unsupported_novel_mask_dev = (
                        None
                        if unsupported_novel_mask_map is None
                        else _resize_map_to_hw(unsupported_novel_mask_map.to(device), render_rgb.shape[-2:])
                    )
                    weights = _build_phase3_region_weight_map(
                        risk_map=risk_dev,
                        gain_map=gain_dev,
                        exist_mask=exist_mask_dev,
                        novel_mask=novel_mask_dev,
                        frontier_mask=frontier_mask_dev,
                        supported_exist_mask=supported_exist_mask_dev,
                        unsupported_novel_mask=unsupported_novel_mask_dev,
                        config=config,
                    )
                    _raw_q_gen_weight = float(unc_data.get("q_gen", 1.0))
                    _q_gen_weight = (
                        _raw_q_gen_weight
                        if bool(getattr(config, "phase3_use_q_gen_weight", False))
                        else 1.0
                    )
                    _raw_predicted_residual = float(unc_data.get("predicted_residual_delta_psnr", 0.0))
                    _raw_q_gain_for_gate = float(unc_data.get("q_gain_for_gate", unc_data.get("q_gain", 0.5)))
                    _q_gain_weight = (
                        max(1.0 + _raw_predicted_residual, 0.0) * _raw_q_gain_for_gate
                        if bool(getattr(config, "phase3_use_q_gain_weight", False))
                        else 1.0
                    )
                    _consistency_weight = float(unc_data.get("consistency_weight", 1.0))
                    _phase3_weight = (
                        _pseudo_loss_weight
                        * _q_gen_weight
                        * _q_gain_weight
                        * _consistency_weight
                    )
                    photo_optim_loss = torch.zeros_like(loss_photo)
                    _phase3_guidance_value = (
                        None
                        if verified_only_enabled and verified_evidence_mask_dev is None
                        else _compute_weighted_masked_l1_loss(
                            render_rgb,
                            gt_image,
                            weight_map=weights,
                            hard_mask=verified_evidence_mask_dev,
                        )
                    )
                    if _phase3_guidance_value is not None:
                        _phase3_guidance_loss = _phase3_guidance_value * _phase3_weight
                        # weighted 模式下，伪视图监督本质上是 uncertainty-aware guidance，
                        # 不是与真实图同等语义的 photo loss。这里显式归入 guidance，
                        # 同时保持总优化量级与旧实现一致（scale=1.0）。
                        loss_guidance = loss_guidance + _phase3_guidance_loss
                        guidance_scale = 1.0
                        if writer is not None and iteration % 100 == 0:
                            writer.add_scalar("gating/phase3_q_gen_raw", _raw_q_gen_weight, iteration)
                            writer.add_scalar("gating/phase3_q_gen", _q_gen_weight, iteration)
                            writer.add_scalar("gating/phase3_predicted_residual_delta_psnr", _raw_predicted_residual, iteration)
                            writer.add_scalar("gating/phase3_q_gain_for_gate_raw", _raw_q_gain_for_gate, iteration)
                            writer.add_scalar("gating/phase3_q_gain", _q_gain_weight, iteration)
                            writer.add_scalar("gating/phase3_consistency", _consistency_weight, iteration)
                            writer.add_scalar("gating/phase3_weight", _phase3_weight, iteration)
                            writer.add_scalar("loss/phase3_guidance", float(_phase3_guidance_loss.item()), iteration)
                            writer.add_scalar(
                                "gating/phase3_verified_evidence_coverage",
                                float(verified_evidence_mask_dev.mean().item()) if verified_evidence_mask_dev is not None else 0.0,
                                iteration,
                            )
                elif soft_gating is not None and epi is not None:
                    epi_dev = epi.to(device)
                    ale_dev = ale.to(device) if ale is not None else None
                    if epi_dev.shape[-2:] != render_rgb.shape[-2:]:
                        epi_dev = F.interpolate(
                            epi_dev.unsqueeze(0).unsqueeze(0) if epi_dev.dim() == 2
                            else epi_dev.unsqueeze(0),
                            size=render_rgb.shape[-2:], mode="bilinear", align_corners=False,
                        ).squeeze(0)
                    if ale_dev is not None and ale_dev.shape[-2:] != render_rgb.shape[-2:]:
                        ale_dev = F.interpolate(
                            ale_dev.unsqueeze(0).unsqueeze(0) if ale_dev.dim() == 2
                            else ale_dev.unsqueeze(0),
                                size=render_rgb.shape[-2:], mode="bilinear", align_corners=False,
                            ).squeeze(0)
                    weights = soft_gating.compute_weights(epi_dev, ale_dev)
                    photo_optim_loss = torch.zeros_like(loss_photo)
                    _soft_guidance_value = (
                        None
                        if verified_only_enabled and verified_evidence_mask_dev is None
                        else _compute_weighted_masked_l1_loss(
                            render_rgb,
                            gt_image,
                            weight_map=weights,
                            hard_mask=verified_evidence_mask_dev,
                        )
                    )
                    if _soft_guidance_value is not None:
                        loss_guidance = loss_guidance + _soft_guidance_value * _pseudo_loss_weight
                else:
                    _masked_pseudo_loss = (
                        None
                        if verified_only_enabled and verified_evidence_mask_dev is None
                        else _compute_weighted_masked_l1_loss(
                            render_rgb,
                            gt_image,
                            hard_mask=verified_evidence_mask_dev,
                        )
                    )
                    photo_optim_loss = (
                        torch.zeros_like(loss_photo)
                        if _masked_pseudo_loss is None
                        else _masked_pseudo_loss * _pseudo_loss_weight
                    )
            else:
                # direct模式：使用固定权重
                if plan6_blocked:
                    _masked_pseudo_loss = None
                elif plan6_weight_dev is not None:
                    _masked_pseudo_loss = _compute_weighted_masked_l1_loss(
                        render_rgb,
                        gt_image,
                        weight_map=plan6_weight_dev,
                    )
                else:
                    _masked_pseudo_loss = (
                        None
                        if verified_only_enabled and verified_evidence_mask_dev is None
                        else _compute_weighted_masked_l1_loss(
                            render_rgb,
                            gt_image,
                            hard_mask=verified_evidence_mask_dev,
                        )
                    )
                photo_optim_loss = (
                    torch.zeros_like(loss_photo)
                    if _masked_pseudo_loss is None
                    else _masked_pseudo_loss * _pseudo_loss_weight
                )
            if (
                plan6_online_allowed
                and not plan6_blocked
                and bool(getattr(config, "phase3_use_plan6_depth_alpha_losses", False))
            ):
                _plan6_depth_loss, _plan6_alpha_loss = _compute_plan6_depth_alpha_supervision_losses(
                    render_colors,
                    render_alphas,
                    info,
                    proposal_depth=pseudo_runtime_data.get("plan6_proposal_depth"),
                    proposal_alpha=pseudo_runtime_data.get("plan6_proposal_alpha"),
                    depth_weight_map=pseudo_runtime_data.get("plan6_phase3_depth_weight"),
                    alpha_weight_map=pseudo_runtime_data.get("plan6_phase3_alpha_weight"),
                    config=config,
                )
                if _plan6_depth_loss is not None:
                    loss_guidance = loss_guidance + _plan6_depth_loss
                    guidance_scale = 1.0
                    if writer is not None and iteration % 100 == 0:
                        writer.add_scalar("loss/phase3_plan6_depth", float(_plan6_depth_loss.item()), iteration)
                if _plan6_alpha_loss is not None:
                    loss_guidance = loss_guidance + _plan6_alpha_loss
                    guidance_scale = 1.0
                    if writer is not None and iteration % 100 == 0:
                        writer.add_scalar("loss/phase3_plan6_alpha", float(_plan6_alpha_loss.item()), iteration)

        mse = F.mse_loss(render_rgb, gt_image)
        psnr = 10.0 * torch.log10(1.0 / torch.clamp(mse, min=1e-8))

        # ========== Phase 1 收敛检测 ==========
        loss_history.append(float(photo_optim_loss.item()))
        psnr_history.append(float(psnr.item()))
        if current_phase == 1:
            conv_window = int(getattr(config, 'phase1_convergence_window', 100))
            conv_thresh = float(getattr(config, 'phase1_convergence_threshold', 0.02))
            conv_psnr = float(getattr(config, 'phase1_convergence_psnr', 18.0))
            conv_min = int(getattr(config, 'phase1_convergence_min_iter', 500))
            if (iteration >= conv_min and iteration % 50 == 0
                    and len(loss_history) >= 2 * conv_window):
                recent_loss = sum(loss_history[-conv_window:]) / conv_window
                prev_loss = sum(loss_history[-2 * conv_window:-conv_window]) / conv_window
                delta_ratio = abs(recent_loss - prev_loss) / (prev_loss + 1e-8)
                avg_psnr = sum(psnr_history[-conv_window:]) / conv_window
                if delta_ratio < conv_thresh and avg_psnr >= conv_psnr:
                    print(f"[Phase 1] Converged at iter {iteration}: "
                          f"delta_ratio={delta_ratio:.4f}, avg_psnr={avg_psnr:.2f}, "
                          f"gaussians={gaussians.means.shape[0]}")
                    # DDUD: 将 Phase 1 的观测统计挂载到 gaussians，
                    # 用于 train_active_sampling.py 保存到 phase1_result.pt
                    gaussians._ddud_count_accum = count_accum.detach().clone()
                    gaussians._ddud_grad2d_accum = grad2d_accum.detach().clone()
                    loss_history.pop()
                    psnr_history.pop()
                    break

                elif iteration % 200 == 0:
                    print(f"[Phase 1] 收敛监控 iter {iteration}: "
                          f"delta_ratio={delta_ratio:.4f}, avg_psnr={avg_psnr:.2f}")

        if writer is not None:
            writer.add_scalar("train/loss_photo", float(photo_optim_loss.item()), iteration)
            writer.add_scalar("train/loss_photo_raw", float(loss_photo.item()), iteration)
            writer.add_scalar("train/psnr", float(psnr.item()), iteration)
            writer.add_scalar("train/num_gaussians", gaussians.means.shape[0], iteration)

        selected_results: List[UncertaintyResult] = []
        fine_results: List[UncertaintyResult] = []
        pearson_value: Optional[float] = None
        if (
            current_phase == 0
            and
            iteration >= config.warmup_iterations
            and iteration % config.guidance_interval == 0
        ):
            admission_failure_reason: Optional[str] = None
            # 候选生成（只有rank 0执行主循环，不需要广播）
            # 球心 = 真实相机均值（Z≈0）；look_at_target = 高斯体均值（Z≈80）
            _loop_cam_positions = torch.stack([cam.position.detach() for cam in real_cameras]).to(device)
            _loop_sphere_center = _loop_cam_positions.mean(dim=0)
            _loop_gauss_target = gaussians.means.detach().mean(dim=0).to(device)
            candidates = sampler.generate_candidates(
                scene_center=_loop_sphere_center,
                intrinsics=default_intrinsics,
                existing_cameras=real_cameras,
                device=device,
                look_at_target=_loop_gauss_target,
            )
            coarse_results: List[UncertaintyResult] = []
            if dist_ctx.enabled:
                _sync_gaussians_to_workers(gaussians, dist_ctx)
            coarse_results = _estimate_candidates(
                estimator,
                sparse_bundle,
                candidates,
                global_step=iteration,
                num_samples=config.mc_dropout_samples_coarse,
                label="coarse",
                dist_context=dist_ctx,
            )
            if coarse_results:
                coarse_pool = selector.preselect(
                    coarse_results,
                    config.coarse_candidate_count or len(coarse_results),
                )
                fine_candidates = [
                    item.candidate for item in coarse_pool
                ][: config.fine_candidate_count or len(coarse_pool)]
            else:
                coarse_pool = []
                fine_candidates = []
            if fine_candidates:
                fine_results = _estimate_candidates(
                    estimator,
                    sparse_bundle,
                    fine_candidates,
                    global_step=iteration,
                    num_samples=config.mc_dropout_samples_fine,
                    label="fine",
                    dist_context=dist_ctx,
                )
            else:
                fine_results = []
            if fine_results and gddn_model is not None:
                stacked_rgb = torch.stack([r.mean_rgb for r in fine_results]).to(device)
                health_ok, health_reason, health_metrics = gddn_model.evaluate_health(
                    stacked_rgb,
                    min_mean=float(config.health_min_mean),
                    max_mean=float(config.health_max_mean),
                    min_std=float(config.health_min_std),
                )
                if writer is not None:
                    writer.add_scalar("health/per_image_mean", health_metrics.get("per_image_mean", 0.0), iteration)
                    writer.add_scalar("health/per_image_std", health_metrics.get("per_image_std", 0.0), iteration)
                if not health_ok:
                    admission_failure_reason = health_reason or "gddn_health_below_threshold"
            if fine_results and admission_failure_reason is None:
                # 筛选和选择（只有rank 0执行主循环，不需要广播）
                fine_pool = selector.preselect(
                    fine_results,
                    config.fine_candidate_count or len(fine_results),
                )
                selected_results = selector.select(
                    fine_pool, config.num_views_per_round
                )
            # 获取原始不确定性分数用于Pearson验证（不使用NIMVS混合后的score）
            mc_scores = []
            for result in fine_results:
                if result.metrics and "original_uncertainty_score" in result.metrics:
                    # 优先使用保存的原始分数
                    mc_scores.append(float(result.metrics["original_uncertainty_score"]))
                elif result.metrics and "variance_masked" in result.metrics:
                    # 备选：使用方差均值
                    mc_scores.append(float(result.metrics["variance_masked"]))
                else:
                    # 最后：使用variance_map的均值
                    mc_scores.append(float(result.variance_map.mean().item()))
            if mc_scores and admission_failure_reason is None:
                # 在gs_dropout模式下，如果没有GDDN模型，跳过validation检查
                # 因为ensemble_scores会全是0，导致validation必然失败
                if gddn_model is None:
                    ensemble_scores = [0.0 for _ in fine_results]
                    # gs_dropout模式：跳过validation，直接使用selected_results
                    validation_pass = True  # 强制通过validation
                else:
                    ensemble_scores, split_half_reliability = bootstrap_ensemble_scores(
                        gddn_model=gddn_model,
                        sparse_bundle=sparse_bundle,
                        results_batch=fine_results,
                        repeats=int(max(2, config.validation_bootstrap_repeats)),
                        candidate_batch_size=int(max(1, getattr(config, "validation_candidate_batch_size", 1))),
                        condition_provider=provider,
                        steps=int(config.fine_steps),
                        resolution=int(config.fine_resolution),
                        device=device,
                        score_type=config.score_type,
                        use_mask_for_scores=bool(config.validation_use_mask),
                        aleatoric_variance_floor=float(config.aleatoric_variance_floor),
                        sampler_type=config.ensemble_sampler_type,
                        per_step_noise_scale=float(config.ensemble_noise_scale),
                        mc_dropout2d_p=float(config.mc_dropout2d_p),
                        mc_token_dropout_p=float(config.mc_token_dropout_p),
                        mc_cond_noise_sigma=float(config.mc_cond_noise_sigma),
                        mc_condition_alpha=float(config.mc_condition_alpha),
                        mc_latent_noise_std=float(config.mc_latent_noise_std),
                    )
                    # 使用split-half reliability替代cross-method Pearson
                    pearson_value = split_half_reliability
                    if writer is not None:
                        writer.add_scalar(
                            "validation/pearson",
                            validation.pearson_correlation,
                            iteration,
                        )
                        writer.add_scalar(
                            "validation/spearman",
                            validation.spearman_correlation,
                            iteration,
                        )
                        writer.add_scalar("validation/ece", validation.ece, iteration)
                        writer.add_scalar(
                            "validation/pass_threshold",
                            1.0 if validation.pass_threshold else 0.0,
                            iteration,
                        )
                    validation_pass = validation.pass_threshold

                if not validation_pass:
                    admission_failure_reason = "validation_correlation_below_threshold"
            else:
                admission_failure_reason = "insufficient_scores"
            if not selected_results and admission_failure_reason is None:
                admission_failure_reason = "no_candidate_passed_quality"
            if admission_failure_reason is not None:
                selected_results = []
            # ========== 处理通过主动选择与证据准入的候选 ==========
            if candidate_log_file is not None:
                selected_ids = {id(item) for item in selected_results}
                if fine_results:
                    source_results = fine_results
                elif selected_results:
                    source_results = selected_results
                else:
                    source_results = []
                candidate_payload = []
                for cand_idx, result in enumerate(source_results):
                    candidate = result.candidate
                    entry = {
                        "index": cand_idx,
                        "score": float(result.score),
                        "selected": id(result) in selected_ids,
                        "position": candidate.position.detach().cpu().tolist(),
                        "viewmat": candidate.viewmat.detach().cpu().tolist(),
                        "intrinsics": candidate.intrinsics.detach().cpu().tolist(),
                    }
                    candidate_payload.append(entry)
                log_record = {
                    "iteration": int(iteration),
                    "pearson": pearson_value,
                    "admission_failure_reason": admission_failure_reason,
                    "candidates": candidate_payload,
                }
                candidate_log_file.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                candidate_log_file.flush()

            # Pearson门控：验证mc_scores和ensemble_scores的相关性
            enforce_pearson_gate = (
                getattr(config, "estimator_mode", "").lower() == "gddn"
                and getattr(config, "pearson_threshold", None) is not None
                and admission_failure_reason is None
            )
            if enforce_pearson_gate:
                pearson_ok = pearson_value is not None and pearson_value >= float(config.pearson_threshold)
                if not pearson_ok and selected_results:
                    selected_results = []
                    if writer is not None:
                        writer.add_text(
                            "active_sampling/skip_guidance",
                            f"pearson<{getattr(config, 'pearson_threshold', 0.0):.3f} @ iter {iteration}",
                            iteration,
                        )
            if selected_results:
                tau_for_iteration = float(config.gate_tau)
                if (
                    soft_gating is not None
                    and config.gate_enable
                    and config.gate_tau_auto
                    and selected_results
                ):
                    uncertainty_vectors: List[torch.Tensor] = []
                    for result in selected_results:
                        if hasattr(result, "variance_map") and result.variance_map is not None:
                            uncertainty_vectors.append(
                                torch.nan_to_num(result.variance_map.detach(), nan=0.0, posinf=0.0, neginf=0.0).view(-1)
                            )
                    if uncertainty_vectors:
                        flattened = [vec.float() for vec in uncertainty_vectors if vec.numel() > 0]
                        if flattened:
                            stacked_unc = torch.cat(flattened, dim=0)
                            quantile = torch.quantile(
                                stacked_unc,
                                max(0.0, min(1.0, config.gate_tau_percentile / 100.0)),
                            ).item()
                            if math.isfinite(quantile):
                                if writer is not None:
                                    writer.add_scalar("gating/tau_raw", quantile, iteration)
                                if tau_ema_value is None:
                                    tau_ema_value = quantile
                                else:
                                    alpha = float(config.gate_tau_ema_alpha)
                                    tau_ema_value = (1.0 - alpha) * tau_ema_value + alpha * quantile
                                tau_for_iteration = tau_ema_value
                tau_for_iteration = max(0.0, float(tau_for_iteration))
                if soft_gating is not None:
                    soft_gating.tau = float(tau_for_iteration)
                    if writer is not None:
                        writer.add_scalar("gating/tau", tau_for_iteration, iteration)
                for sel_index, selected in enumerate(selected_results):
                    # 在循环开始时初始化depth和normal变量
                    depth_render = None
                    normal_render = None
                    depth_vis = None
                    normal_vis = None

                    # 获取depth和normal渲染（用于TensorBoard和PNG保存）
                    try:
                        with torch.no_grad():
                            cond_maps = provider([selected.candidate])
                    except Exception:
                        cond_maps = None
                    if cond_maps is not None:
                        depth_render = cond_maps.get("depth_render")
                        normal_render = cond_maps.get("normal_render")
                    if depth_render is not None:
                        depth_vis = depth_render[0].detach()
                        depth_vis = depth_vis - depth_vis.min()
                        depth_vis = depth_vis / depth_vis.max().clamp_min(1e-6)
                    if normal_render is not None:
                        normal_vis = normal_render[0].detach()
                        normal_vis = (normal_vis.clamp(-1.0, 1.0) + 1.0) * 0.5

                    if writer is not None:
                        writer.add_scalar(
                            "active_sampling/selected_score",
                            selected.score,
                            iteration,
                        )
                        if config.log_selected_images:
                            clamped_rgb = selected.mean_rgb.detach().clamp(0.0, 1.0)
                            writer.add_image(
                                f"guidance/pseudo_rgb/iter_{iteration}/cand_{sel_index}",
                                clamped_rgb,
                                iteration,
                            )
                            variance_map = torch.nan_to_num(selected.variance_map.detach())
                            var_min = variance_map.min()
                            var_max = variance_map.max()
                            denom = torch.clamp(var_max - var_min, min=1e-6)
                            normalized_var = ((variance_map - var_min) / denom).unsqueeze(0)
                            writer.add_image(
                                f"guidance/variance_map/iter_{iteration}/cand_{sel_index}",
                                normalized_var,
                                iteration,
                            )
                            if depth_vis is not None:
                                writer.add_image(
                                    f"guidance/depth_render/iter_{iteration}/cand_{sel_index}",
                                    depth_vis,
                                    iteration,
                                )
                            if normal_vis is not None:
                                writer.add_image(
                                    f"guidance/normal_render/iter_{iteration}/cand_{sel_index}",
                                    normal_vis,
                                    iteration,
                                )
                    if config.save_pseudo_png and pseudo_save_dir is not None:
                        rgb_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_score{selected.score:.4f}.png"
                        _save_rgb_tensor(rgb_path, selected.mean_rgb)
                        var_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_var.png"
                        _save_map_tensor(var_path, selected.variance_map)
                        # 保存depth渲染图
                        if depth_vis is not None:
                            depth_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_depth.png"
                            _save_map_tensor(depth_path, depth_vis.squeeze(0) if depth_vis.dim() == 3 else depth_vis)
                        # 保存normal渲染图
                        if normal_vis is not None:
                            normal_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_normal.png"
                            normal_tensor = normal_vis
                            if normal_tensor.dim() == 2:
                                normal_tensor = normal_tensor.unsqueeze(0)
                            _save_rgb_tensor(normal_path, normal_tensor)
                    pseudo_rgb = selected.mean_rgb
                    pseudo_colors, pseudo_alpha, _ = rasterization(
                        means=gaussians.means.clone(),
                        quats=gaussians.quats.clone(),
                        scales=gaussians.scales.clone(),
                        opacities=gaussians.opacities.clone(),
                        colors=gaussians.colors.clone(),
                        viewmats=selected.candidate.viewmat.unsqueeze(0),
                        Ks=selected.candidate.intrinsics.unsqueeze(0),
                        width=pseudo_rgb.shape[-1],
                        height=pseudo_rgb.shape[-2],
                    )
                    rendered_pseudo = (
                        pseudo_colors[0, ..., :3].permute(2, 0, 1).contiguous()
                    )
                    opacity_map = pseudo_alpha[0, ..., 0].contiguous()
                    
                    # ========== 增强伪视图保存功能 ==========
                    if config.save_pseudo_png and pseudo_save_dir is not None:
                        # 保存gsplat渲染结果（用于对比）
                        if getattr(config, 'save_gsplat_render', True):
                            gsplat_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_gsplat.png"
                            _save_rgb_tensor(gsplat_path, rendered_pseudo)
                        
                        # 分别保存epistemic和aleatoric不确定性图
                        if getattr(config, 'save_uncertainty_maps', True):
                            if selected.epistemic_map is not None:
                                epi_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_epistemic.png"
                                _save_map_tensor(epi_path, selected.epistemic_map)
                            if selected.aleatoric_map is not None:
                                ale_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_aleatoric.png"
                                _save_map_tensor(ale_path, selected.aleatoric_map)
                        
                        # 保存对比拼接图（GDDN vs gsplat）
                        if getattr(config, 'save_comparison_png', True):
                            comparison_path = pseudo_save_dir / f"iter{iteration:06d}_cand{sel_index}_comparison.png"
                            _create_comparison_image(selected.mean_rgb, rendered_pseudo, comparison_path)
                    
                    _maybe_run_dacd(selected, opacity_map, iteration)

                    # ========== 鲁棒NLL / 软门控引导损失 ==========
                    if (
                        nll_loss is not None
                        and selected.aleatoric_map is not None
                        and selected.epistemic_map is not None
                    ):
                        target_hw = (pseudo_rgb.shape[-2], pseudo_rgb.shape[-1])
                        ale_map = _resize_map_to_hw(
                            selected.aleatoric_map.detach().to(device), target_hw
                        ).unsqueeze(0).unsqueeze(0)
                        epi_map = _resize_map_to_hw(
                            selected.epistemic_map.detach().to(device), target_hw
                        ).unsqueeze(0).unsqueeze(0)
                        student = rendered_pseudo.unsqueeze(0)
                        teacher = pseudo_rgb.unsqueeze(0)
                        nll_value = nll_loss(
                            pred_rgb=student,
                            target_rgb=teacher,
                            aleatoric_var=ale_map,
                            epistemic_var=epi_map,
                            ood_threshold=float(config.nll_epistemic_threshold),
                        )
                        loss_guidance = loss_guidance + nll_value
                    elif (
                        soft_gating is not None
                        and hasattr(selected, "metrics")
                        and selected.metrics is not None
                        and hasattr(selected, "variance_map")
                        and selected.variance_map is not None
                    ):
                        u_epistemic = torch.nan_to_num(
                            selected.variance_map, nan=0.0, posinf=0.0, neginf=0.0
                        ).to(rendered_pseudo.device)
                        if "aleatoric_mean" in selected.metrics:
                            u_aleatoric = torch.full_like(
                                u_epistemic,
                                float(selected.metrics["aleatoric_mean"]),
                            )
                        else:
                            u_aleatoric = u_epistemic * 0.3
                        weights = soft_gating.compute_weights(
                            u_epistemic, u_aleatoric
                        ).to(rendered_pseudo.device)
                        if weights.shape != pseudo_rgb.shape[-2:]:
                            weights = F.interpolate(
                                weights.unsqueeze(0).unsqueeze(0),
                                size=pseudo_rgb.shape[-2:],
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        pixel_loss = (rendered_pseudo - pseudo_rgb).abs()
                        pixel_loss_mean = pixel_loss.mean(dim=0)
                        weighted_loss = (
                            (weights * pixel_loss_mean).sum()
                            / (weights.sum() + 1e-8)
                        )
                        loss_guidance = loss_guidance + weighted_loss
                        if writer is not None and iteration % 100 == 0:
                            writer.add_scalar(
                                f"gating/weight_mean/cand_{sel_index}",
                                weights.mean().item(),
                                iteration,
                            )
                            writer.add_scalar(
                                f"gating/weight_std/cand_{sel_index}",
                                weights.std().item(),
                                iteration,
                            )
                            writer.add_histogram(
                                f"gating/weight_distribution/cand_{sel_index}",
                                weights,
                                iteration,
                            )
                            norm_weights = weights.unsqueeze(0).clamp(0, 1)
                            writer.add_image(
                                f"gating/weights_iter{iteration}/cand_{sel_index}",
                                norm_weights,
                                iteration,
                            )
                    else:
                        loss_guidance = loss_guidance + F.l1_loss(
                            rendered_pseudo, pseudo_rgb
                        )
                    # ========== 引导损失结束 ==========

                    selected_views.append(selected.candidate)
            else:
                # selected_results为空，跳过本轮guidance计算
                if writer is not None and admission_failure_reason is not None:
                    writer.add_text(
                        "active_sampling/skip_guidance",
                        f"skip at iter {iteration}: {admission_failure_reason}",
                        iteration,
                    )
        # ========== Phase 3 guidance_only模式：伪视图引导损失 ==========
        guidance_iteration_uses_pseudo_supervision = False
        if (current_phase == 3 and effective_pseudo_mode == "guidance_only"
                and pseudo_views and iteration % max(1, config.guidance_interval) == 0):
            verified_only_enabled = bool(getattr(config, "phase3_use_verified_evidence_only", False))
            for pv in pseudo_views:
                try:
                    pv_camera = pv.camera
                    pv_image = pv.image.to(device)
                    if pv_image.dim() == 4 and pv_image.shape[0] == 1:
                        pv_image = pv_image[0]
                    if pv_image.dim() != 3:
                        raise ValueError(
                            f"PseudoView.image must be [3,H,W] or [1,3,H,W], got shape={tuple(pv_image.shape)}"
                        )
                    if pv_image.shape[-2:] != gt_image.shape[-2:]:
                        pv_image = F.interpolate(
                            pv_image.unsqueeze(0), size=gt_image.shape[-2:],
                            mode="bilinear", align_corners=False,
                        ).squeeze(0)
                    pv_render, pv_alpha, pv_info = rasterization(
                        means=gaussians.means.clone(),
                        quats=gaussians.quats.clone(),
                        scales=gaussians.scales.clone(),
                        opacities=gaussians.opacities.clone(),
                        colors=gaussians.colors.clone(),
                        viewmats=pv_camera.viewmat.unsqueeze(0).to(device),
                        Ks=pv_camera.intrinsics.unsqueeze(0).to(device),
                        width=pv_image.shape[-1],
                        height=pv_image.shape[-2],
                        render_mode=(
                            "RGB+ED"
                            if bool(getattr(config, "phase3_use_plan6_depth_alpha_losses", False))
                            else "RGB"
                        ),
                    )
                    pv_rendered = pv_render[0, ..., :3].permute(2, 0, 1).contiguous()
                    proposal_patches = list(getattr(pv, "proposal_patches", []))
                    if phase3_patch_proposals_only and len(proposal_patches) == 0:
                        continue
                    verified_evidence_mask_dev = None
                    if verified_only_enabled:
                        verified_evidence_mask = _resolve_verified_evidence_mask(
                            verified_evidence_mask=getattr(pv, "verified_evidence_mask", None),
                            supported_exist_mask=pv.supported_exist_mask,
                            config=config,
                        )
                        if verified_evidence_mask is not None:
                            verified_evidence_mask_dev = _resize_map_to_hw(
                                verified_evidence_mask.to(device),
                                pv_rendered.shape[-2:],
                            )

                    def _compute_phase3_verified_guidance(
                        patch_weight_map: Optional[torch.Tensor] = None,
                    ) -> Optional[torch.Tensor]:
                        if phase3_patch_proposals_only:
                            return _compute_patch_proposal_guidance_loss(
                                pv_rendered,
                                proposal_patches,
                                source_width=int(getattr(pv, "width", pv_image.shape[-1])),
                                source_height=int(getattr(pv, "height", pv_image.shape[-2])),
                                weight_map=patch_weight_map,
                            )
                        if verified_only_enabled and verified_evidence_mask_dev is None:
                            return None
                        return _compute_weighted_masked_l1_loss(
                            pv_rendered,
                            pv_image,
                            weight_map=patch_weight_map,
                            hard_mask=verified_evidence_mask_dev,
                        )

                    if bool(getattr(config, "phase3_use_plan6_trust_weights", False)):
                        if not bool(getattr(pv, "plan6_online_admission_allowed", False)):
                            continue
                        plan6_weight_map = getattr(pv, "plan6_phase3_rgb_weight", None)
                        if plan6_weight_map is None:
                            continue
                        plan6_weight_dev = _resize_map_to_hw(
                            plan6_weight_map.to(device),
                            pv_rendered.shape[-2:],
                        )
                        _pv_guidance_value = _compute_phase3_verified_guidance(plan6_weight_dev)
                        if _pv_guidance_value is None:
                            continue
                        pv_loss = _pv_guidance_value
                    elif pv.risk_exist_map is not None and pv.gain_novel_map is not None:
                        risk_dev = _resize_map_to_hw(pv.risk_exist_map.to(device), pv_rendered.shape[-2:])
                        gain_dev = _resize_map_to_hw(pv.gain_novel_map.to(device), pv_rendered.shape[-2:])
                        exist_mask_dev = (
                            torch.ones_like(risk_dev)
                            if pv.exist_mask is None
                            else _resize_map_to_hw(pv.exist_mask.to(device), pv_rendered.shape[-2:])
                        )
                        novel_mask_dev = (
                            torch.clamp(1.0 - exist_mask_dev, min=0.0, max=1.0)
                            if pv.novel_mask is None
                            else _resize_map_to_hw(pv.novel_mask.to(device), pv_rendered.shape[-2:])
                        )
                        frontier_mask_dev = (
                            None
                            if pv.frontier_mask is None
                            else _resize_map_to_hw(pv.frontier_mask.to(device), pv_rendered.shape[-2:])
                        )
                        supported_exist_mask_dev = (
                            None
                            if pv.supported_exist_mask is None
                            else _resize_map_to_hw(pv.supported_exist_mask.to(device), pv_rendered.shape[-2:])
                        )
                        unsupported_novel_mask_dev = (
                            None
                            if pv.unsupported_novel_mask is None
                            else _resize_map_to_hw(pv.unsupported_novel_mask.to(device), pv_rendered.shape[-2:])
                        )
                        w = _build_phase3_region_weight_map(
                            risk_map=risk_dev,
                            gain_map=gain_dev,
                            exist_mask=exist_mask_dev,
                            novel_mask=novel_mask_dev,
                            frontier_mask=frontier_mask_dev,
                            supported_exist_mask=supported_exist_mask_dev,
                            unsupported_novel_mask=unsupported_novel_mask_dev,
                            config=config,
                        )
                        _guidance_consistency_weight = float(getattr(pv, "consistency_weight", 1.0))
                        _guidance_q_gain_weight = 1.0
                        if bool(getattr(config, "phase3_use_q_gain_weight", False)):
                            _guidance_q_gain_weight = max(
                                1.0 + float(getattr(pv, "predicted_residual_delta_psnr", 0.0)),
                                0.0,
                            ) * float(getattr(pv, "q_gain_for_gate", getattr(pv, "q_gain", 0.5)))
                        _pv_guidance_value = _compute_phase3_verified_guidance(w)
                        if _pv_guidance_value is None:
                            continue
                        pv_loss = (
                            _pv_guidance_value
                            * _guidance_consistency_weight
                            * _guidance_q_gain_weight
                        )
                    elif soft_gating is not None and pv.epistemic_map is not None:
                        epi_dev = pv.epistemic_map.to(device)
                        ale_dev = pv.aleatoric_map.to(device) if pv.aleatoric_map is not None else None
                        if epi_dev.shape[-2:] != pv_rendered.shape[-2:]:
                            epi_dev = F.interpolate(
                                epi_dev.unsqueeze(0).unsqueeze(0) if epi_dev.dim() == 2
                                else epi_dev.unsqueeze(0),
                                size=pv_rendered.shape[-2:], mode="bilinear", align_corners=False,
                            ).squeeze(0)
                        w = soft_gating.compute_weights(epi_dev, ale_dev).to(device)
                        _pv_guidance_value = _compute_phase3_verified_guidance(w)
                        if _pv_guidance_value is None:
                            continue
                        pv_loss = _pv_guidance_value
                    else:
                        _pv_guidance_value = _compute_phase3_verified_guidance()
                        if _pv_guidance_value is None:
                            continue
                        pv_loss = _pv_guidance_value
                    if (
                        bool(getattr(config, "phase3_use_plan6_depth_alpha_losses", False))
                        and bool(getattr(pv, "plan6_online_admission_allowed", False))
                    ):
                        _pv_depth_loss, _pv_alpha_loss = _compute_plan6_depth_alpha_supervision_losses(
                            pv_render,
                            pv_alpha,
                            pv_info,
                            proposal_depth=getattr(pv, "plan6_proposal_depth", None),
                            proposal_alpha=getattr(pv, "plan6_proposal_alpha", None),
                            depth_weight_map=getattr(pv, "plan6_phase3_depth_weight", None),
                            alpha_weight_map=getattr(pv, "plan6_phase3_alpha_weight", None),
                            config=config,
                        )
                        if _pv_depth_loss is not None:
                            pv_loss = pv_loss + _pv_depth_loss
                            if writer is not None and iteration % 100 == 0:
                                writer.add_scalar("loss/phase3_plan6_depth_guidance_only", float(_pv_depth_loss.item()), iteration)
                        if _pv_alpha_loss is not None:
                            pv_loss = pv_loss + _pv_alpha_loss
                            if writer is not None and iteration % 100 == 0:
                                writer.add_scalar("loss/phase3_plan6_alpha_guidance_only", float(_pv_alpha_loss.item()), iteration)
                    guidance_iteration_uses_pseudo_supervision = True
                    loss_guidance = loss_guidance + pv_loss
                except Exception:
                    pass  # 跳过失败的伪视角

        total_loss = photo_optim_loss + guidance_scale * loss_guidance
        has_photo_grad = bool(photo_optim_loss.requires_grad)
        has_guidance_grad = bool(loss_guidance.requires_grad)
        if stability_strategy == "alternate":
            if has_photo_grad and (iteration % 2 == 0 or not has_guidance_grad):
                optimizer.zero_grad()
                photo_optim_loss.backward()
                optimizer.step()
                project_field(gaussians)
            elif has_guidance_grad:
                optimizer.zero_grad()
                (guidance_scale * loss_guidance).backward()
                optimizer.step()
                project_field(gaussians)
        elif stability_strategy == "split":
            if has_photo_grad:
                optimizer.zero_grad()
                photo_optim_loss.backward()
                optimizer.step()
                project_field(gaussians)
            if has_guidance_grad:
                optimizer.zero_grad()
                (guidance_scale * loss_guidance).backward()
                optimizer.step()
                project_field(gaussians)
        else:
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                project_field(gaussians)
        # 进度条更新（替代逐行 print）
        if _pbar is not None:
            _pbar.set_postfix(
                photo=f"{photo_optim_loss.item():.4f}",
                guidance=f"{loss_guidance.item():.4f}",
                lam=f"{guidance_scale:.4f}",
                gs=gaussians.means.shape[0],
            )
        elif iteration % 200 == 0:
            print(
                f"iter {iteration}/{num_iterations}: "
                f"photo={photo_optim_loss.item():.4f}, guidance={loss_guidance.item():.4f}, "
                f"λ={guidance_scale:.4f}, gs={gaussians.means.shape[0]}"
            )
        if writer is not None:
            writer.add_scalar("loss/guidance", float(loss_guidance.item()), iteration)
            writer.add_scalar("loss/total", float(total_loss.item()), iteration)
            writer.add_scalar("training/lambda_schedule", float(lambda_current), iteration)
            writer.add_scalar("training/lambda_effective", float(guidance_scale), iteration)
            # 记录当前训练阶段
            writer.add_scalar("training/phase", phase_int, iteration)
        
        # ========== 梯度累积与Clone/Split致密化 ==========
        # 对齐gsplat DefaultStrategy._update_state的梯度累积逻辑
        # Phase 3伪视角保护: 伪视角梯度不参与densification判断
        # guidance_only 下的 verified pseudo guidance 也不能把扩散噪声写入 densification 统计
        _is_pseudo_view = (current_phase == 3 and pseudo_views
                           and camera_index >= pseudo_start_idx)
        allow_pseudo_densification_stats = bool(
            getattr(config, "allow_pseudo_densification_stats", False)
        )
        if (
            (_is_pseudo_view or guidance_iteration_uses_pseudo_supervision)
            and not allow_pseudo_densification_stats
        ):
            # 伪视角: 只贡献photo loss, 不驱动densification
            pass  # 跳过梯度累积
        elif "means2d" in info:
            accumulate_density_gradients(
                info, grad2d_accum, count_accum, gt_image.shape[-1], gt_image.shape[-2]
            )

        # 执行Clone/Split/Prune
        _maybe_grow_gs(iteration, info)
        completed_iterations = iteration + 1

    gaussians._ddud_count_accum = count_accum.detach().clone()
    gaussians._ddud_grad2d_accum = grad2d_accum.detach().clone()
    if current_phase == 1 and not dist_ctx.enabled and not pseudo_views:
        gaussians._training_state = capture_training_state(
            gaussians, optimizer, current_phase=1, completed_iterations=completed_iterations,
            config=dict(vars(config)), stability_strategy=stability_strategy,
            grad2d_accum=grad2d_accum.detach().cpu().clone(),
            count_accum=count_accum.detach().cpu().clone(), scene_scale=scene_scale,
            loss_history=loss_history, psnr_history=psnr_history,
            real_images=real_images.detach().cpu().clone(),
            cameras=[(camera.viewmat.detach().cpu().clone(), camera.intrinsics.detach().cpu().clone())
                     for camera in real_cameras],
        )

    # ========== 训练循环完成日志 ==========
    print(f"[Training] ========== 3DGS训练循环完成 ==========")
    print(f"[Training] 完成优化步数: {completed_iterations}")
    print(f"[Training] 最终高斯体数量: {gaussians.means.shape[0]}")
    print(f"[Training] 已选择伪视角数量: {len(selected_views)}")
    print(f"[Training] =============================================")

    if dist_ctx.enabled and dist_ctx.is_main:
        _broadcast_command(ActiveSamplingCommand.SHUTDOWN, dist_ctx)
    if candidate_log_file is not None:
        candidate_log_file.close()
    
    # 收集并返回伪视角信息用于3DGS导出
    collected_pseudo_cameras: List[Dict[str, Any]] = []
    # 1. 从在线采样选出的视角收集
    for view in selected_views:
        K = view.intrinsics.cpu().numpy() if hasattr(view.intrinsics, 'cpu') else view.intrinsics
        collected_pseudo_cameras.append({
            "viewmat": view.viewmat,
            "width": config.fine_resolution,
            "height": config.fine_resolution,
            "fx": float(K[0, 0]) if K.ndim == 2 else 500.0,
            "fy": float(K[1, 1]) if K.ndim == 2 else 500.0,
            "img_name": "",
        })
    # 2. 如果在线采样未产出视角（如Phase 3），从传入的pseudo_views收集
    if not collected_pseudo_cameras and pseudo_views:
        for pv in pseudo_views:
            pv_cam = pv.camera
            pv_viewmat = pv_cam.viewmat
            pv_K = pv_cam.intrinsics.cpu().numpy() if hasattr(pv_cam.intrinsics, 'cpu') else pv_cam.intrinsics
            collected_pseudo_cameras.append({
                "viewmat": pv_viewmat,
                "width": pv.width or int(pv.image.shape[-1]),
                "height": pv.height or int(pv.image.shape[-2]),
                "fx": float(pv_K[0, 0]) if pv_K.ndim == 2 else 500.0,
                "fy": float(pv_K[1, 1]) if pv_K.ndim == 2 else 500.0,
                "img_name": pv.img_name,
            })
    
    # ========== 导出3DGS模型 ==========
    if getattr(config, 'output_dir', None) is not None:
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出PLY格式的高斯体点云
        ply_dir = output_dir / "point_cloud"
        ply_dir.mkdir(parents=True, exist_ok=True)
        ply_path = ply_dir / "point_cloud.ply"
        
        export_splats(**ply_parameters(gaussians), format="ply", save_to=str(ply_path))
        print(f"[INFO] Exported PLY to {ply_path}")
        
        # 导出cameras.json
        cameras_path = output_dir / "cameras.json"
        cameras_data = []
        
        # 添加训练相机
        train_cameras = list(real_cameras[:pseudo_start_idx]) if pseudo_start_idx > 0 else list(real_cameras)
        for i, cam in enumerate(train_cameras):
            cameras_data.append({
                "id": i,
                "type": "train",
                "width": int(real_images.shape[-1]),
                "height": int(real_images.shape[-2]),
                "fx": float(cam.intrinsics[0, 0]) if hasattr(cam, 'intrinsics') else 500.0,
                "fy": float(cam.intrinsics[1, 1]) if hasattr(cam, 'intrinsics') else 500.0,
                "position": cam.position.cpu().numpy().tolist() if hasattr(cam, 'position') else [0, 0, 0],
            })
        
        # 添加伪视角相机
        for i, pc in enumerate(collected_pseudo_cameras):
            cameras_data.append({
                "id": len(train_cameras) + i,
                "type": "pseudo",
                "width": pc["width"],
                "height": pc["height"],
                "fx": pc["fx"],
                "fy": pc["fy"],
                "img_name": pc.get("img_name", ""),
            })
        
        with open(cameras_path, 'w') as f:
            json.dump(cameras_data, f, indent=2)
        print(f"[INFO] Exported cameras to {cameras_path}")
    
    return collected_pseudo_cameras


__all__ = [
    "train_3dgs_with_active_sampling",
    "run_phase2_active_sampling",
    "PseudoView",
    "DistributedActiveSamplingContext",
]
