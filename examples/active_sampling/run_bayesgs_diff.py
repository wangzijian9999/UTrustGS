#!/usr/bin/env python
"""Train the frozen UTrustGS generative prior.

The paper configuration uses Stable Diffusion 2.1 with depth ControlNet and
supports single- or multi-GPU training.

Usage:
    python run_bayesgs_diff.py \\
        --backbone controlnet \\
        --local-path /path/to/stable-diffusion-2-1-base \\
        --data-dir /path/to/dataset
"""

from __future__ import annotations

import os
import sys
import re
import math
import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Set, Sequence, Iterator, Union

# 添加当前目录到PYTHONPATH，解决相对导入问题
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
# 添加 examples 目录与项目根目录，以支持 examples.active_sampling.* 包式导入
_EXAMPLES_DIR = _SCRIPT_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))
_ROOT_DIR = _EXAMPLES_DIR.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
if __name__ == "__main__":
    sys.modules.setdefault("examples.active_sampling.run_bayesgs_diff", sys.modules[__name__])

import torch
from examples.active_sampling.gaussian_parameter_contract import field_state, load_field
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

from examples.active_sampling.checkpoint_utils import load_state_dict_with_whitelist


def _unproject_pixels(
    camera_to_world: torch.Tensor,
    x_pixels: torch.Tensor,
    y_pixels: torch.Tensor,
    depths: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Back-project image pixels to world coordinates."""
    x_camera = (x_pixels.float() - intrinsics[0, 2]) / intrinsics[0, 0] * depths
    y_camera = (y_pixels.float() - intrinsics[1, 2]) / intrinsics[1, 1] * depths
    camera_points = torch.stack((x_camera, y_camera, depths), dim=-1)
    rotation = camera_to_world[:3, :3]
    translation = camera_to_world[:3, 3]
    return camera_points @ rotation.T + translation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the UTrustGS generative prior"
    )
    
    # Backbone selection
    parser.add_argument(
        "--backbone", type=str, default="controlnet",
        choices=["controlnet"],
        help="Paper backbone: Stable Diffusion 2.1 with depth ControlNet."
    )
    parser.add_argument(
        "--local-path", type=str, default=None,
        help="Local path to the pretrained Stable Diffusion 2.1 model."
    )
    
    # Data
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to training dataset directory"
    )
    parser.add_argument(
        "--val-dir", type=str, default=None,
        help="Path to validation dataset (optional)"
    )
    parser.add_argument(
        "--llff-multiscene-root",
        type=str,
        default=None,
        help="LLFF 多场景根目录；设置后启用跨场景生成先验训练协议，并忽略单场景 val-dir。",
    )
    parser.add_argument(
        "--llff-train-scenes",
        type=str,
        default="",
        help="显式指定 LLFF 训练场景名，逗号分隔；留空则按 holdout/val 自动推导。",
    )
    parser.add_argument(
        "--llff-val-scenes",
        type=str,
        default="",
        help="显式指定 LLFF 验证场景名，逗号分隔；留空时按 auto-val-scene-count 自动保留。",
    )
    parser.add_argument(
        "--llff-holdout-scene",
        type=str,
        default="",
        help="当测试场景来自 LLFF 时，用于 leave-one-scene-out 的 holdout 场景名；该场景不会进入生成先验训练/验证。",
    )
    parser.add_argument(
        "--llff-auto-val-scene-count",
        type=int,
        default=2,
        help="在未显式指定 llff-val-scenes 时，自动保留的跨场景验证集数量，默认 2。",
    )
    parser.add_argument(
        "--llff-sparse-views",
        type=int,
        default=3,
        help="LLFF 生成先验训练时每个样本使用的 support 视角数，默认严格对齐在线阶段的 3 视角。",
    )
    parser.add_argument(
        "--llff-pretrain-sparse-views",
        type=int,
        default=8,
        help="LLFF Stage1 几何预训练时每个样本使用的 support 视角数；默认 8，用更强几何条件学习 proposal 先验。",
    )
    parser.add_argument(
        "--llff-alignment-sparse-views",
        type=int,
        default=3,
        help="LLFF Stage2a/Stage2/Stage3 sparse-view alignment 时每个样本使用的 support 视角数；默认 3，对齐在线测试协议。",
    )
    parser.add_argument(
        "--geometry-aware-trainable-tokens",
        type=str,
        default="",
        help=(
            "显式限制 LLFF geometry-aware finetune 的可训练模块 token，逗号分隔；"
            "留空保持自动推导默认行为。用于 Gate0 诊断时可只训练 frontier_proposal_head。"
        ),
    )
    parser.add_argument(
        "--disable-geometry-aware-defaults",
        action="store_true",
        help="禁用 LLFF 多场景协议下的几何感知默认配置（support losses、模块冻结策略等）。",
    )
    parser.add_argument(
        "--disable-llff-scene-balanced-sampling",
        action="store_true",
        help="禁用 LLFF 多场景训练的 scene-balanced sampling；默认开启以防样本多的场景主导生成先验。",
    )
    parser.add_argument(
        "--llff-balanced-samples-per-scene",
        type=int,
        default=0,
        help="LLFF 多场景训练时每个 epoch 对每个 scene 采样的样本数；0 表示自动对齐到最大场景样本数。",
    )
    parser.add_argument(
        "--keep-llff-warp-condition-source",
        action="store_true",
        help="在 LLFF 多场景 geometry-aware 协议下保留旧的 warp 条件源；默认会自动切换到 phase1_3dgs。",
    )
    parser.add_argument(
        "--offline-condition-source",
        type=str,
        default="warp",
        choices=["warp", "corrupt", "phase1_3dgs"],
        help="离线生成先验条件源：warp/corrupt 或基于 3DGS render/depth 条件缓存的 phase1_3dgs。",
    )
    parser.add_argument(
        "--offline-phase1-result",
        type=str,
        default=None,
        help="可选的 train_active_sampling.py Phase1 导出的 phase1_result.pt；未提供时，phase1_3dgs 将回退到离线 sparse COLMAP bootstrap。",
    )
    parser.add_argument(
        "--offline-3dgs-cache-dir",
        type=str,
        default=None,
        help="离线生成先验的 3DGS render/depth 条件缓存目录；未提供时默认写入输出目录下的 offline_condition_cache。",
    )
    parser.add_argument(
        "--relaxed-geometry-contract", action="store_true",
        help="允许缺失 poses/reference depth 时继续退化训练；默认关闭以避免几何链静默退化"
    )
    parser.add_argument(
        "--teacher-guidance-scale",
        type=float,
        default=1.0,
        help="离线生成先验采样 guidance scale；默认 1.0 关闭 CFG，避免 novel-view 条件链被过强 cond-uncond 差异破坏。",
    )
    parser.add_argument(
        "--enable-plucker-conditioning",
        action="store_true",
        help="显式启用 Plucker 逐像素位姿 token；默认关闭，便于与全局 PoseEncoder 做单独消融。",
    )
    parser.add_argument(
        "--controlnet-condition-channels",
        type=int,
        default=3,
        choices=[3, 5],
        help="ControlNet 条件通道数：3=保守 blended 条件，5=显式 warped_rgb+depth+confidence。",
    )
    parser.add_argument(
        "--enable-sfm-depth-alignment",
        action="store_true",
        help="显式启用基于 COLMAP sparse points 的 SfM metric depth alignment；默认关闭，避免未经消融直接并入主线。",
    )
    parser.add_argument(
        "--teacher-final-sample-support-fidelity-weight",
        type=float,
        default=None,
        help="显式开启 final_sample support 保真项；None 保持当前行为不变，建议作为独立主线实验单独消融。",
    )
    parser.add_argument(
        "--teacher-final-sample-support-consistency-weight",
        type=float,
        default=None,
        help="为全 support-projected 区域增加相对 rgb_render 的一致性约束；None 保持当前行为不变。",
    )
    parser.add_argument(
        "--teacher-final-sample-support-confidence-boost",
        type=float,
        default=None,
        help="对低置信 support 区域施加更高 final_sample 保真权重；None 保持当前行为不变。",
    )
    parser.add_argument(
        "--teacher-final-sample-support-low-confidence-anchor-weight",
        type=float,
        default=None,
        help="为 editable 邻近的低置信 support 区域增加窄锚点监督；None 保持当前行为不变。",
    )
    parser.add_argument(
        "--teacher-relative-effect-weight",
        type=float,
        default=None,
        help="显式设置 geometry-aware 生成先验的 relative effect / utility calibration 权重；None 时使用协议默认值。",
    )
    parser.add_argument(
        "--disable-geometry-aware-frontier-proposal-focus",
        action="store_true",
        help="关闭 geometry-aware frontier proposal focus；默认开启，使 novel supervision 聚焦在 support 邻域的 proposal 区域。",
    )
    parser.add_argument(
        "--teacher-frontier-proposal-kernel-size",
        type=int,
        default=None,
        help="frontier proposal mask 的膨胀核大小；None 时使用协议默认值。",
    )
    parser.add_argument(
        "--teacher-frontier-proposal-support-threshold",
        type=float,
        default=None,
        help="frontier proposal mask 从 support_projection_mask 二值化时使用的阈值；None 时使用协议默认值。",
    )
    parser.add_argument(
        "--teacher-proposal-residual-source",
        type=str,
        default="head",
        choices=["head", "teacher_delta"],
        help="Frontier proposal residual source: learned head or raw model delta minus base RGB.",
    )
    parser.add_argument(
        "--teacher-proposal-safe-confidence-threshold",
        type=float,
        default=None,
        help="覆盖 proposal residual gate 的 safe confidence threshold；None 保持协议默认值。",
    )
    parser.add_argument(
        "--teacher-support-projection-conflict-target-scheduler-delta",
        type=float,
        default=None,
        help="覆盖 support projection conflict gate 的目标 scheduler delta；None 保持模型默认值。",
    )
    parser.add_argument(
        "--teacher-support-projection-conflict-min-scale",
        type=float,
        default=None,
        help="覆盖 support projection conflict gate 的最小缩放比例；None 保持模型默认值。",
    )
    parser.add_argument(
        "--teacher-support-projection-conflict-gamma",
        type=float,
        default=None,
        help="覆盖 support projection conflict gate 的 gamma；None 保持模型默认值。",
    )
    parser.add_argument(
        "--fixed-visualization-sample-indices",
        type=str,
        default="0",
        help="固定样本组索引，逗号分隔；第一个索引保持旧单样本可视化兼容。",
    )
    parser.add_argument(
        "--teacher-image-size",
        type=int,
        default=512,
        help="Offline generative-prior LLFF dataloader square image size; default 512 preserves existing behavior.",
    )
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--stage1-batch-size",
        type=int,
        default=None,
        help="Stage1 几何预训练 dataloader batch size；None 时回退到 --batch-size。",
    )
    parser.add_argument(
        "--alignment-batch-size",
        type=int,
        default=None,
        help="Stage2a/Stage2/Stage3 sparse-view alignment dataloader batch size；None 时回退到 --batch-size。",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--verbose-dataset-logs",
        action="store_true",
        help="打印 LLFFDataset 的逐场景/逐缓存详细日志；默认关闭，仅保留必要摘要。",
    )
    parser.add_argument(
        "--verbose-epoch-stats",
        action="store_true",
        help="打印 Stage1 的详细 epoch 统计字典；默认关闭，仅保留 loss 主线日志。",
    )
    parser.add_argument(
        "--prewarm-image-cache",
        action="store_true",
        help="启动时预热整场景 RGB 图像缓存；默认关闭以减少启动耗时。",
    )
    parser.add_argument(
        "--disable-lazy-alignment-dataloaders",
        action="store_true",
        help="禁用 sparse-view alignment dataloader 延迟构建；默认按需构建以减少启动耗时。",
    )
    parser.add_argument(
        "--enable-amp",
        action="store_true",
        help="启用 AMP 混合精度以换取更大 batch 和更高吞吐；默认关闭以保持历史稳定性。",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16"],
        help="AMP 精度类型；5090 建议优先使用 bf16。",
    )
    parser.add_argument(
        "--vae-decode-chunk-size",
        type=int,
        default=None,
        help="VAE 解码 chunk 大小；None 保持当前默认策略，增大可提升 batch>1 时吞吐。",
    )
    parser.add_argument("--stage1-epochs", type=int, default=15)
    parser.add_argument("--stage2a-epochs", type=int, default=8)
    parser.add_argument("--stage2-epochs", type=int, default=10)
    parser.add_argument("--stage3-epochs", type=int, default=30)
    parser.add_argument("--stage1-lr", type=float, default=1e-5)
    parser.add_argument("--stage2a-lr", type=float, default=5e-6)
    parser.add_argument("--stage2-lr", type=float, default=1e-4)
    parser.add_argument("--stage3-lr", type=float, default=1e-6)
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=0,
        help="统一启用基于主验证指标的早停；0 表示关闭。patience 以“监控事件次数”计数，不是自然 epoch 数。",
    )
    parser.add_argument(
        "--early-stop-min-epochs",
        type=int,
        default=0,
        help="早停至少等待的最少 epoch 数；与 --early-stop-patience 配合使用。",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="判定指标提升所需的最小增量；默认 0.0。",
    )
    parser.add_argument(
        "--stage1-validation-interval",
        type=int,
        default=None,
        help="Stage1 full validation 的 epoch 间隔；LLFF 多场景协议下默认自动放宽到 5。",
    )
    parser.add_argument(
        "--stage1-visualization-interval",
        type=int,
        default=None,
        help="Stage1 full sample visualization 的 epoch 间隔；LLFF 多场景协议下默认自动放宽到 5。",
    )
    parser.add_argument(
        "--stage2a-validation-interval",
        type=int,
        default=None,
        help="Stage2a full validation 的 epoch 间隔；LLFF 多场景协议下默认自动放宽到 5。",
    )
    parser.add_argument(
        "--stage2a-visualization-interval",
        type=int,
        default=None,
        help="Stage2a full sample visualization 的 epoch 间隔；LLFF 多场景协议下默认自动放宽到 5。",
    )
    parser.add_argument(
        "--stage1-x0-warmup-steps",
        type=int,
        default=None,
        help="Stage1 使用 x0_decode 的 warmup step 数；0 表示立即切到 trainable_sample。",
    )
    parser.add_argument(
        "--stage2a-x0-warmup-steps",
        type=int,
        default=None,
        help="Stage2a 使用 x0_decode 的 warmup step 数；0 表示立即切到 trainable_sample。",
    )
    parser.add_argument(
        "--stage1-validation-mode",
        type=str,
        default=None,
        choices=["full_generate", "trainable_sample"],
        help="Stage1 validation 使用的生成模式；LLFF 多场景协议默认使用 trainable_sample。",
    )
    parser.add_argument(
        "--stage2a-validation-mode",
        type=str,
        default=None,
        choices=["full_generate", "trainable_sample"],
        help="Stage2a validation 使用的生成模式；LLFF 多场景协议默认使用 trainable_sample。",
    )
    parser.add_argument(
        "--stage1-validation-max-batches",
        type=int,
        default=None,
        help="Stage1 每次 validation 最多评估的 batch 数；0 表示完整验证集。",
    )
    parser.add_argument(
        "--stage2a-validation-max-batches",
        type=int,
        default=None,
        help="Stage2a 每次 validation 最多评估的 batch 数；0 表示完整验证集。",
    )
    parser.add_argument(
        "--stage1-sample-path-interval",
        type=int,
        default=None,
        help="Stage1 触发 trainable_sample 辅助监督的 batch 间隔；LLFF 多场景协议下默认更高频。",
    )
    parser.add_argument(
        "--stage2a-sample-path-interval",
        type=int,
        default=None,
        help="Stage2a 触发 trainable_sample 辅助监督的 batch 间隔；LLFF 多场景协议下默认每步触发。",
    )
    parser.add_argument(
        "--stage1-sample-path-weight",
        type=float,
        default=None,
        help="Stage1 中 trainable_sample 辅助损失的相对放大系数。",
    )
    parser.add_argument(
        "--stage2a-sample-path-weight",
        type=float,
        default=None,
        help="Stage2a 中 trainable_sample 辅助损失的相对放大系数。",
    )
    
    parser.add_argument(
        "--multi-gpu", action="store_true",
        help="Enable multi-GPU component distribution for ControlNet."
    )
    parser.add_argument(
        "--disable-tf32", action="store_true", default=True,
        help="Disable TF32 to prevent NaN issues (recommended for multi-GPU)"
    )
    
    # Output
    parser.add_argument(
        "--output-dir", type=str, default="./output",
        help="Directory for checkpoints and logs"
    )
    parser.add_argument("--exp-name", type=str, default="utrustgs_prior")
    # Distributed training
    parser.add_argument("--local-rank", type=int, default=-1)

    # Resume & Stage control
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--stage", type=str, default="1,2,3",
        help="Stages to run (comma separated, e.g., '2,3' or '3')"
    )
    
    # 学术创新方向
    
    return parser.parse_args()


def setup_distributed() -> tuple:
    """Initialize distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        return rank, world_size, local_rank, True
    
    return 0, 1, 0, False


def _parse_scene_name_list(raw_value: Optional[str]) -> List[str]:
    if raw_value is None:
        return []
    parsed_scene_names: List[str] = []
    seen_scene_names: Set[str] = set()
    for raw_token in str(raw_value).split(","):
        scene_name = raw_token.strip()
        if not scene_name or scene_name in seen_scene_names:
            continue
        seen_scene_names.add(scene_name)
        parsed_scene_names.append(scene_name)
    return parsed_scene_names


def _discover_llff_scene_dirs(root_dir: Path) -> Dict[str, Path]:
    discovered_scene_dirs: Dict[str, Path] = {}
    if not root_dir.exists():
        raise FileNotFoundError(f"LLFF 多场景根目录不存在: {root_dir}")
    for candidate_scene_dir in sorted(root_dir.iterdir()):
        if not candidate_scene_dir.is_dir():
            continue
        resolved_scene_dir: Optional[Path] = None
        if (candidate_scene_dir / "poses_bounds.npy").exists():
            resolved_scene_dir = candidate_scene_dir
        else:
            nested_pose_candidates = [
                nested_dir
                for nested_dir in sorted(candidate_scene_dir.iterdir())
                if nested_dir.is_dir() and (nested_dir / "poses_bounds.npy").exists()
            ]
            if nested_pose_candidates:
                resolved_scene_dir = nested_pose_candidates[0]
        if resolved_scene_dir is not None:
            discovered_scene_dirs[candidate_scene_dir.name] = resolved_scene_dir
    if not discovered_scene_dirs:
        raise RuntimeError(f"在 {root_dir} 下未发现可用 LLFF 场景（缺少 poses_bounds.npy）。")
    return discovered_scene_dirs


def _resolve_llff_multiscene_splits(
    *,
    llff_root_dir: Path,
    explicit_train_scenes: Sequence[str],
    explicit_val_scenes: Sequence[str],
    holdout_scene_name: str,
    auto_val_scene_count: int,
) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, Path]]]:
    discovered_scene_dirs = _discover_llff_scene_dirs(llff_root_dir)
    holdout_scene_name = str(holdout_scene_name).strip()
    if holdout_scene_name and holdout_scene_name not in discovered_scene_dirs:
        raise ValueError(
            f"指定的 llff-holdout-scene 不存在: {holdout_scene_name}; "
            f"可选场景={sorted(discovered_scene_dirs.keys())}"
        )

    explicit_train_scene_names = list(explicit_train_scenes)
    explicit_val_scene_names = list(explicit_val_scenes)
    unknown_scene_names = [
        scene_name
        for scene_name in explicit_train_scene_names + explicit_val_scene_names
        if scene_name not in discovered_scene_dirs
    ]
    if unknown_scene_names:
        raise ValueError(
            f"存在未找到的 LLFF 场景: {unknown_scene_names}; "
            f"可选场景={sorted(discovered_scene_dirs.keys())}"
        )

    blocked_scene_names = {holdout_scene_name} if holdout_scene_name else set()
    available_scene_names = [
        scene_name
        for scene_name in sorted(discovered_scene_dirs.keys())
        if scene_name not in blocked_scene_names
    ]
    if not available_scene_names:
        raise RuntimeError("可用于生成先验训练的 LLFF 场景为空。")

    if explicit_train_scene_names:
        overlapping_blocked = [
            scene_name for scene_name in explicit_train_scene_names if scene_name in blocked_scene_names
        ]
        if overlapping_blocked:
            raise ValueError(
                f"训练场景与 holdout 场景冲突: {overlapping_blocked}"
            )
        resolved_train_scene_names = list(explicit_train_scene_names)
    else:
        resolved_train_scene_names = []

    if explicit_val_scene_names:
        overlapping_blocked = [
            scene_name for scene_name in explicit_val_scene_names if scene_name in blocked_scene_names
        ]
        if overlapping_blocked:
            raise ValueError(
                f"验证场景与 holdout 场景冲突: {overlapping_blocked}"
            )
        resolved_val_scene_names = list(explicit_val_scene_names)
    else:
        resolved_val_scene_names = []

    if resolved_train_scene_names and resolved_val_scene_names:
        overlapping_scene_names = sorted(
            set(resolved_train_scene_names).intersection(resolved_val_scene_names)
        )
        if overlapping_scene_names:
            raise ValueError(
                f"训练/验证场景存在重叠: {overlapping_scene_names}"
            )

    if not resolved_train_scene_names:
        if resolved_val_scene_names:
            resolved_train_scene_names = [
                scene_name
                for scene_name in available_scene_names
                if scene_name not in resolved_val_scene_names
            ]
        else:
            reserved_val_scene_count = max(int(auto_val_scene_count), 0)
            reserved_val_scene_count = min(
                reserved_val_scene_count,
                max(len(available_scene_names) - 1, 0),
            )
            if reserved_val_scene_count > 0:
                resolved_val_scene_names = available_scene_names[-reserved_val_scene_count:]
            resolved_train_scene_names = [
                scene_name
                for scene_name in available_scene_names
                if scene_name not in resolved_val_scene_names
            ]
    elif not resolved_val_scene_names:
        remaining_scene_names = [
            scene_name
            for scene_name in available_scene_names
            if scene_name not in resolved_train_scene_names
        ]
        reserved_val_scene_count = max(int(auto_val_scene_count), 0)
        reserved_val_scene_count = min(
            reserved_val_scene_count,
            len(remaining_scene_names),
        )
        if reserved_val_scene_count > 0:
            resolved_val_scene_names = remaining_scene_names[:reserved_val_scene_count]

    if not resolved_train_scene_names:
        raise RuntimeError("自动推导 LLFF 训练场景失败，训练集为空。")

    resolved_train_scene_specs = [
        (scene_name, discovered_scene_dirs[scene_name])
        for scene_name in resolved_train_scene_names
    ]
    resolved_val_scene_specs = [
        (scene_name, discovered_scene_dirs[scene_name])
        for scene_name in resolved_val_scene_names
    ]
    return resolved_train_scene_specs, resolved_val_scene_specs


class LLFFMultiSceneDataset(Dataset):
    """Thin wrapper that keeps per-scene metadata for balanced sampling."""

    def __init__(
        self,
        scene_datasets: Sequence[Dataset],
        scene_names: Sequence[str],
    ) -> None:
        if not scene_datasets:
            raise ValueError("LLFFMultiSceneDataset requires at least one scene dataset.")
        if len(scene_datasets) != len(scene_names):
            raise ValueError("scene_datasets and scene_names must have the same length.")
        self.scene_datasets = list(scene_datasets)
        self.scene_names = [str(scene_name) for scene_name in scene_names]
        self.scene_lengths = [len(scene_dataset) for scene_dataset in self.scene_datasets]
        self.scene_offsets: List[int] = []
        running_offset = 0
        for scene_length in self.scene_lengths:
            self.scene_offsets.append(running_offset)
            running_offset += int(scene_length)
        self.total_length = running_offset
        self.scene_sample_counts: Dict[str, int] = {
            scene_name: int(scene_length)
            for scene_name, scene_length in zip(self.scene_names, self.scene_lengths)
        }

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0:
            index += self.total_length
        if index < 0 or index >= self.total_length:
            raise IndexError(f"LLFFMultiSceneDataset index out of range: {index}")
        scene_index = 0
        for next_scene_index, scene_offset in enumerate(self.scene_offsets):
            scene_length = self.scene_lengths[next_scene_index]
            if index < scene_offset + scene_length:
                scene_index = next_scene_index
                break
        local_index = index - self.scene_offsets[scene_index]
        sample = self.scene_datasets[scene_index][local_index]
        if isinstance(sample, dict):
            sample = dict(sample)
            sample.setdefault("scene_index", torch.tensor(scene_index, dtype=torch.long))
            sample.setdefault("scene_sample_index", torch.tensor(local_index, dtype=torch.long))
        return sample


class SceneBalancedDistributedSampler(Sampler[int]):
    """Balance LLFF multi-scene training so large scenes do not dominate prior-model updates."""

    def __init__(
        self,
        dataset: LLFFMultiSceneDataset,
        *,
        shuffle: bool = True,
        seed: int = 0,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        samples_per_scene: int = 0,
    ) -> None:
        if len(dataset.scene_lengths) <= 1:
            raise ValueError("SceneBalancedDistributedSampler requires at least 2 scenes.")
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"Invalid rank {rank} for num_replicas={num_replicas}.")
        self.dataset = dataset
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.epoch = 0
        requested_samples_per_scene = int(samples_per_scene)
        if requested_samples_per_scene <= 0:
            requested_samples_per_scene = max(int(scene_length) for scene_length in dataset.scene_lengths)
        self.samples_per_scene = requested_samples_per_scene
        self.balanced_total_size = self.samples_per_scene * len(self.dataset.scene_lengths)
        self.num_samples = int(math.ceil(self.balanced_total_size / float(self.num_replicas)))
        self.total_size = self.num_samples * self.num_replicas

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _build_balanced_scene_indices(
        self,
        *,
        generator: torch.Generator,
    ) -> List[int]:
        per_scene_global_indices: List[torch.Tensor] = []
        for scene_offset, scene_length in zip(self.dataset.scene_offsets, self.dataset.scene_lengths):
            scene_length = int(scene_length)
            if scene_length <= 0:
                continue
            if self.shuffle:
                local_indices = []
                repeat_count = int(math.ceil(self.samples_per_scene / float(scene_length)))
                for _ in range(max(repeat_count, 1)):
                    local_indices.append(torch.randperm(scene_length, generator=generator))
                stacked_local_indices = torch.cat(local_indices, dim=0)[: self.samples_per_scene]
            else:
                stacked_local_indices = (
                    torch.arange(self.samples_per_scene, dtype=torch.long) % scene_length
                )
            per_scene_global_indices.append(stacked_local_indices + int(scene_offset))

        if not per_scene_global_indices:
            return []

        interleaved_indices: List[int] = []
        ordered_scene_ids = torch.arange(len(per_scene_global_indices), dtype=torch.long)
        for sample_index in range(self.samples_per_scene):
            if self.shuffle:
                scene_order = torch.randperm(len(per_scene_global_indices), generator=generator)
            else:
                scene_order = ordered_scene_ids
            for scene_id in scene_order.tolist():
                interleaved_indices.append(int(per_scene_global_indices[scene_id][sample_index].item()))
        return interleaved_indices

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = self._build_balanced_scene_indices(generator=generator)
        if not indices:
            return iter([])
        if len(indices) < self.total_size:
            padding = indices[: self.total_size - len(indices)]
            indices = indices + padding
        else:
            indices = indices[: self.total_size]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices)


def _apply_llff_multiscene_runtime_defaults(
    *,
    args: argparse.Namespace,
    is_main: bool,
) -> None:
    if not args.llff_multiscene_root or args.disable_geometry_aware_defaults:
        return
    if (
        str(getattr(args, "offline_condition_source", "")).strip().lower() == "warp"
        and not bool(getattr(args, "keep_llff_warp_condition_source", False))
    ):
        args.offline_condition_source = "phase1_3dgs"
        if is_main:
            print(
                "[GeometryAware] LLFF 多场景协议已自动将 offline_condition_source 从 warp "
                "切换为 phase1_3dgs，以避免生成先验主学整图外观补全。"
            )
    if (
        str(getattr(args, "offline_condition_source", "")).strip().lower() == "warp"
        and not bool(getattr(args, "enable_sfm_depth_alignment", False))
    ):
        args.enable_sfm_depth_alignment = True
        if is_main:
            print(
                "[GeometryAware] 当前仍使用 warp 条件源，已自动启用 SfM depth alignment，"
                "减少跨场景生成先验因相对深度尺度漂移导致的 support 退化。"
            )


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


import numpy as np

from controlnet_components import (
    _DA_MODEL_CONFIGS,
    _find_da_model_spec,
    _get_depth_anything_v2_class,
)


class LLFFDataset(torch.utils.data.Dataset):
    """
    LLFF格式数据集加载器（NeRF标准格式）。
    
    支持的目录结构:
        data_dir/
        ├── images/ 或 images_8/ (降采样版本)
        │   ├── image_001.png
        │   ├── image_002.png
        │   └── ...
        └── poses_bounds.npy  (相机位姿和边界)
    
    输出格式:
        - sparse_images: [V, 3, H, W] 稀疏视角图像
        - sparse_poses: [V, 4, 4] 稀疏视角位姿
        - target_pose: [4, 4] 目标位姿
        - target_rgb: [3, H, W] 目标图像
        - rgb_render: [3, H, W] 渲染RGB（warp proxy 或真实3DGS render）
        - depth_render: [1, H, W] 深度渲染（伪深度估计或真实3DGS深度）
        - accum_render: [1, H, W] 3DGS 累积透明度 / coverage
        - transmittance_render: [1, H, W] 3DGS 透射率
        - normal_render: [3, H, W] 法线渲染（占位符）
        - target_latent: [4, H//8, W//8] 目标latent（占位符）
        
    条件生成模式 (condition_mode):
        - warp: 使用透视变换从最近视角warp到目标视角（推荐）
        - corrupt: 在target_rgb上添加随机遮挡和噪声
        - phase1_3dgs: 使用 3DGS render/depth/alpha 条件；优先使用真实 Phase1 缓存，缺失时回退到离线 sparse COLMAP bootstrap
    """
    _shared_depth_model: Optional[nn.Module] = None
    _shared_depth_model_checkpoint: Optional[str] = None
    _shared_depth_model_device: Optional[str] = None
    _shared_depth_model_type: Optional[str] = None
    _shared_aligned_depth_map_cache: Dict[str, torch.Tensor] = {}
    _shared_image_tensor_cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    _shared_image_tensor_cache_limit: Optional[int] = None
    
    def __init__(
        self,
        data_dir: str,
        num_sparse_views: int = 3,
        image_size: tuple = (512, 512),
        is_train: bool = True,
        train_ratio: float = 1.0,
        condition_mode: str = "warp",  # warp, corrupt, phase1_3dgs
        corruption_ratio: float = 0.3,  # 用于corrupt模式
        strict_geometry_contract: bool = True,
        offline_phase1_result: Optional[str] = None,
        offline_3dgs_cache_dir: Optional[str] = None,
        enable_sfm_depth_alignment: bool = False,
        support_template_stage_tag: Optional[str] = None,
        hard_episode_top_fraction: float = 1.0,
        hard_episode_repeat_factor: int = 1,
        verbose_dataset_logs: bool = False,
        prewarm_image_cache: bool = False,
    ):
        super().__init__()
        self.input_data_dir = Path(data_dir)
        self.strict_geometry_contract = strict_geometry_contract
        self.scene_dir = self._resolve_scene_dir()
        self.data_dir = self.scene_dir
        self.num_sparse_views = num_sparse_views
        self.image_size = image_size
        self.is_train = is_train
        self.condition_mode = condition_mode
        self.corruption_ratio = corruption_ratio
        self.offline_phase1_result = None
        if offline_phase1_result is not None:
            resolved_phase1_result_path = Path(offline_phase1_result).expanduser().resolve()
            if resolved_phase1_result_path.is_dir():
                scene_specific_phase1_result_path = (
                    resolved_phase1_result_path / self.scene_dir.name / "phase1_result.pt"
                )
                if scene_specific_phase1_result_path.exists():
                    resolved_phase1_result_path = scene_specific_phase1_result_path
                else:
                    resolved_phase1_result_path = resolved_phase1_result_path / "phase1_result.pt"
            self.offline_phase1_result = resolved_phase1_result_path
        self.offline_3dgs_cache_dir = (
            Path(offline_3dgs_cache_dir).expanduser().resolve()
            if offline_3dgs_cache_dir is not None
            else None
        )
        self.verbose_dataset_logs = bool(verbose_dataset_logs)
        self.prewarm_image_cache = bool(prewarm_image_cache)
        self.enable_sfm_depth_alignment = bool(enable_sfm_depth_alignment)
        self.use_phase1_3dgs_conditions = self.condition_mode == "phase1_3dgs"
        self.hard_episode_top_fraction = min(
            max(float(hard_episode_top_fraction), 0.0),
            1.0,
        )
        self.hard_episode_repeat_factor = max(int(hard_episode_repeat_factor), 1)
        
        # target 监督图像池来自完整 scene；sparse reference 图像池来自输入稀疏子集目录。
        self.images_dir = self._find_images_dir()
        self.sparse_images_dir = self._find_sparse_images_dir()
        
        # 加载 target 图像列表
        self.image_files = self._collect_image_files_from_dir(self.images_dir)
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {self.images_dir}")
        
        # 加载 sparse reference 图像列表
        self.sparse_image_files = self._collect_image_files_from_dir(self.sparse_images_dir)
        if len(self.sparse_image_files) == 0:
            raise ValueError(f"No sparse reference images found in {self.sparse_images_dir}")
        
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Found {len(self.image_files)} target images in {self.images_dir}")
            if self.sparse_images_dir != self.images_dir:
                print(
                    f"[LLFFDataset] Found {len(self.sparse_image_files)} sparse reference images in "
                    f"{self.sparse_images_dir}"
                )
            if self.scene_dir != self.input_data_dir:
                print(f"[LLFFDataset] Scene root resolved to: {self.scene_dir}")
        
        self.effective_num_sparse_views = min(
            self.num_sparse_views,
            max(len(self.sparse_image_files), 1),
        )
        if self.effective_num_sparse_views != self.num_sparse_views and self.verbose_dataset_logs:
            print(
                f"[LLFFDataset] Requested {self.num_sparse_views} sparse views but only "
                f"{self.effective_num_sparse_views} unique reference views are available; "
                f"using the effective value without duplication."
            )
        self.support_template_stage_tag = self._resolve_support_template_stage_tag(
            support_template_stage_tag
        )
        self.support_template_scope = self._resolve_support_template_scope()
        
        # 加载位姿与LLFF相机内参(H, W, focal)
        self.poses, self.bounds, self.pose_hwf = self._load_poses()
        self.camera_intrinsics = self._build_resized_camera_intrinsics(self.pose_hwf)
        self.pose_indices_for_images = self._build_pose_index_mapping(
            self.image_files,
            image_directory=self.images_dir,
        )
        self.pose_indices_for_sparse_images = self._build_pose_index_mapping(
            self.sparse_image_files,
            image_directory=self.sparse_images_dir,
        )
        self._nearest_sparse_indices_by_target = self._build_nearest_sparse_indices_by_target()
        
        # 划分训练/验证集
        n_total = len(self.image_files)
        n_train = int(n_total * train_ratio)
        
        if is_train:
            self.indices = list(range(n_train))
        else:
            if train_ratio >= 1.0 or n_train >= n_total:
                self.indices = list(range(n_total))
            else:
                self.indices = list(range(n_train, n_total))
        
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Using {len(self.indices)} samples for {'training' if is_train else 'validation'}")
        
        # 初始化Depth-Anything-V2模型
        self.depth_model = None
        self.depth_model_device = self._resolve_depth_model_device()
        self._cached_image_tensors_by_file_path: Dict[str, torch.Tensor] = {}
        self._cached_depth_maps_by_image_index: Dict[str, torch.Tensor] = {}
        self._cached_depth_maps_by_file_path: Dict[str, torch.Tensor] = {}
        self._persistent_depth_cache_dir = self._resolve_persistent_depth_cache_dir()
        self._phase1_3dgs_condition_cache: Optional[Dict[str, Any]] = None
        self._phase1_3dgs_runtime_geometry_bank: Optional[Dict[str, Any]] = None
        self._sfm_depth_alignment_assets: Optional[Dict[str, Any]] = None
        self._last_colmap_sparse_dir_candidates: List[Path] = []
        self._sfm_depth_alignment_logged: bool = False
        self._sfm_depth_alignment_attempted_images: Set[str] = set()
        self._sfm_depth_alignment_successful_images: Set[str] = set()
        self._sfm_depth_alignment_inlier_sum: int = 0
        self.relative_depth_consistency_tolerance = 0.12
        self.forward_backward_pixel_tolerance = 2.5
        self.phase1_3dgs_min_render_coverage = 0.01
        self.phase1_3dgs_min_render_std = 5.0e-4
        self.phase1_3dgs_max_dark_render_value = 1.0e-3
        self.phase1_3dgs_frontier_support_threshold = 0.05
        self.phase1_3dgs_frontier_kernel_size = 31
        self.phase1_3dgs_privileged_depth_stride = 8
        self.phase1_3dgs_privileged_max_points = 250000
        self._phase1_3dgs_fallback_log_count = 0
        self._phase1_3dgs_fallback_log_limit = 1 if self.verbose_dataset_logs else 0
        self._phase1_support_condition_provider_cache: Dict[str, Dict[str, Any]] = {}
        self._phase1_support_condition_provider_cache_limit = 512
        self._phase1_support_template_bank: Dict[int, List[Dict[str, Any]]] = {}
        self._phase1_episode_bank_metadata: Optional[Dict[str, Any]] = None
        self._phase1_episode_file_lookup: Dict[Tuple[int, int], Path] = {}
        self._phase1_episode_sample_specs: List[Dict[str, Any]] = []
        self._phase1_episode_tensor_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._phase1_episode_tensor_cache_limit = 24
        self._phase1_total_episode_sample_spec_count: int = 0
        self._phase1_selected_episode_sample_spec_count: int = 0
        self.supports_multiprocess_dataloader: bool = False
        if self.use_phase1_3dgs_conditions:
            self._phase1_support_template_bank = self._load_or_build_support_template_bank()
            self._phase1_episode_bank_metadata = self._load_phase1_episode_bank_metadata()
            if self._phase1_episode_bank_metadata is None:
                self._init_depth_model()
                self._warmup_scene_depth_cache()
                self._phase1_3dgs_condition_cache = self._load_or_build_phase1_3dgs_condition_cache()
                self._phase1_3dgs_runtime_geometry_bank = self._materialize_phase1_3dgs_geometry_bank(
                    self._phase1_3dgs_condition_cache
                )
                self._phase1_episode_bank_metadata = self._build_phase1_episode_bank()
            self._index_phase1_episode_bank_metadata(self._phase1_episode_bank_metadata)
            self._phase1_episode_sample_specs = self._build_phase1_episode_sample_specs()
            self._phase1_support_condition_provider_cache.clear()
            self._release_phase1_build_only_resources()
            self.supports_multiprocess_dataloader = bool(
                self._phase1_episode_bank_metadata is not None
            )
            if self.supports_multiprocess_dataloader and self.prewarm_image_cache:
                self._warmup_scene_image_tensor_cache()
            print(
                "[LLFFDataset] 已启用 support-conditioned Phase1 episode bank: "
                f"stage={self.support_template_stage_tag}, "
                f"template_scope={self.support_template_scope}, "
                f"episodes={len(self._phase1_episode_sample_specs)}"
            )
            if self.hard_episode_top_fraction < 0.999 or self.hard_episode_repeat_factor > 1:
                print(
                    "[LLFFDataset] 已启用 hard-episode 采样池: "
                    f"stage={self.support_template_stage_tag}, "
                    f"top_fraction={self.hard_episode_top_fraction:.2f}, "
                    f"base_selected={self._phase1_selected_episode_sample_spec_count}, "
                    f"base_total={self._phase1_total_episode_sample_spec_count}, "
                    f"repeat_factor={self.hard_episode_repeat_factor}, "
                    f"effective_episodes={len(self._phase1_episode_sample_specs)}"
                )
        else:
            if self.enable_sfm_depth_alignment:
                self._sfm_depth_alignment_assets = self._load_sfm_depth_alignment_assets()
            elif self.verbose_dataset_logs:
                print(
                    "[LLFFDataset] SfM metric depth alignment disabled; "
                    "当前使用 poses_bounds 线性拉伸作为默认深度尺度。"
                )
            self._init_depth_model()

    def _resolve_scene_dir(self) -> Path:
        """解析真正的 scene root，避免把 images 子目录误当成场景根目录。"""
        candidate_dirs = [
            self.input_data_dir,
            self.input_data_dir.parent,
            self.input_data_dir.parent.parent,
        ]
        for candidate_dir in candidate_dirs:
            if (candidate_dir / "poses_bounds.npy").exists():
                return candidate_dir
        matched_scene_dir = self._find_scene_dir_from_matching_images()
        if matched_scene_dir is not None:
            return matched_scene_dir
        return self.input_data_dir

    def _resolve_persistent_depth_cache_dir(self) -> Optional[Path]:
        if self.offline_3dgs_cache_dir is None:
            return None
        cache_dir = (
            self.offline_3dgs_cache_dir
            / "depth_cache_v1"
            / self.scene_dir.name
            / f"h{int(self.image_size[0])}_w{int(self.image_size[1])}_sfm{int(self.enable_sfm_depth_alignment)}"
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _resolve_offline_condition_cache_root(self) -> Path:
        if self.offline_3dgs_cache_dir is None:
            cache_root = self.scene_dir / "offline_condition_cache"
        else:
            cache_root = self.offline_3dgs_cache_dir / self.scene_dir.name
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root

    def _release_phase1_build_only_resources(self) -> None:
        """episode bank 构建完成后释放仅初始化阶段需要的资源，允许 DataLoader 多进程。"""
        self.depth_model = None
        self._phase1_3dgs_condition_cache = None
        self._phase1_3dgs_runtime_geometry_bank = None
        self._cached_depth_maps_by_image_index.clear()
        self._cached_depth_maps_by_file_path.clear()

    def _build_depth_cache_key(
        self,
        *,
        image_file: Path,
        pose_index: int,
    ) -> str:
        resolved_image_path = str(image_file.resolve()).lower()
        return (
            f"{self.scene_dir.resolve()}::"
            f"{resolved_image_path}::"
            f"pose{int(pose_index)}::"
            f"h{int(self.image_size[0])}_w{int(self.image_size[1])}::"
            f"sfm{int(self.enable_sfm_depth_alignment)}"
        )

    def _resolve_persistent_depth_cache_file(
        self,
        *,
        image_file: Path,
        pose_index: int,
    ) -> Optional[Path]:
        if self._persistent_depth_cache_dir is None:
            return None
        persistent_cache_stem = f"{image_file.stem}_pose{int(pose_index)}"
        return self._persistent_depth_cache_dir / f"{persistent_cache_stem}.pt"

    def _build_nearest_sparse_indices_by_target(self) -> Dict[int, np.ndarray]:
        nearest_sparse_indices_by_target: Dict[int, np.ndarray] = {}
        if len(self.sparse_image_files) == 0 or len(self.image_files) == 0:
            return nearest_sparse_indices_by_target

        sparse_camera_centers: List[torch.Tensor] = []
        for sparse_image_index in range(len(self.sparse_image_files)):
            sparse_pose_index = int(self.pose_indices_for_sparse_images[sparse_image_index].item())
            sparse_pose = self.poses[sparse_pose_index]
            sparse_rotation = sparse_pose[:3, :3]
            sparse_translation = sparse_pose[:3, 3]
            sparse_camera_center = (-sparse_rotation.transpose(0, 1) @ sparse_translation).float()
            sparse_camera_centers.append(sparse_camera_center)
        sparse_camera_centers_tensor = torch.stack(sparse_camera_centers, dim=0)

        for target_image_index in range(len(self.image_files)):
            target_pose_index = int(self.pose_indices_for_images[target_image_index].item())
            target_pose = self.poses[target_pose_index]
            target_rotation = target_pose[:3, :3]
            target_translation = target_pose[:3, 3]
            target_camera_center = (-target_rotation.transpose(0, 1) @ target_translation).float()
            camera_distances = torch.norm(
                sparse_camera_centers_tensor - target_camera_center.unsqueeze(0),
                dim=1,
            )
            sorted_sparse_indices = torch.argsort(camera_distances, dim=0).cpu().numpy().astype(np.int64)
            filtered_sparse_indices = [
                int(sparse_image_index)
                for sparse_image_index in sorted_sparse_indices.tolist()
                if int(self.pose_indices_for_sparse_images[int(sparse_image_index)].item())
                != target_pose_index
            ]
            nearest_sparse_indices_by_target[target_image_index] = np.asarray(
                filtered_sparse_indices,
                dtype=np.int64,
            )
        return nearest_sparse_indices_by_target

    def _warmup_scene_depth_cache(self) -> None:
        if self.depth_model is None:
            return
        unique_image_specs: Dict[str, Tuple[Path, int]] = {}
        for target_image_index, image_file in enumerate(self.image_files):
            pose_index = int(self.pose_indices_for_images[target_image_index].item())
            unique_image_specs[str(image_file.resolve()).lower()] = (image_file, pose_index)
        for sparse_image_index, image_file in enumerate(self.sparse_image_files):
            pose_index = int(self.pose_indices_for_sparse_images[sparse_image_index].item())
            unique_image_specs[str(image_file.resolve()).lower()] = (image_file, pose_index)

        uncached_image_specs: List[Tuple[Path, int]] = []
        for image_file, pose_index in unique_image_specs.values():
            shared_depth_cache_key = self._build_depth_cache_key(
                image_file=image_file,
                pose_index=pose_index,
            )
            persistent_depth_cache_file = self._resolve_persistent_depth_cache_file(
                image_file=image_file,
                pose_index=pose_index,
            )
            if shared_depth_cache_key in LLFFDataset._shared_aligned_depth_map_cache:
                continue
            if persistent_depth_cache_file is not None and persistent_depth_cache_file.exists():
                continue
            uncached_image_specs.append((image_file, pose_index))

        if not uncached_image_specs:
            return

        if self.verbose_dataset_logs:
            print(
                f"[LLFFDataset] 预热场景深度缓存: scene={self.scene_dir.name}, "
                f"uncached_images={len(uncached_image_specs)}"
            )
        for image_file, pose_index in uncached_image_specs:
            self._get_cached_depth_map_for_image_file(
                image_file,
                pose_index=pose_index,
            )

    def _collect_image_files_from_dir(self, base_dir: Path) -> list[Path]:
        """从目录或其 images 子目录收集图像文件。"""
        candidate_dirs = [
            base_dir,
            base_dir / "images",
            base_dir / "input",
        ]
        for candidate_dir in candidate_dirs:
            if not candidate_dir.exists():
                continue
            image_files = sorted(
                list(candidate_dir.glob("*.png"))
                + list(candidate_dir.glob("*.jpg"))
                + list(candidate_dir.glob("*.JPG"))
            )
            if image_files:
                return image_files
        return []

    def _find_scene_dir_from_matching_images(self) -> Optional[Path]:
        """当输入是稀疏子集目录时，通过图像名匹配回推真实 scene root。"""
        input_image_files = self._collect_image_files_from_dir(self.input_data_dir)
        if not input_image_files:
            return None
        input_image_names = {image_file.name for image_file in input_image_files}
        search_roots = [self.input_data_dir.parent, self.input_data_dir.parent.parent]
        candidate_scene_dirs: list[Path] = []
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for candidate_dir in search_root.iterdir():
                if not candidate_dir.is_dir():
                    continue
                if (candidate_dir / "poses_bounds.npy").exists():
                    candidate_scene_dirs.append(candidate_dir)
                for nested_candidate_dir in candidate_dir.iterdir():
                    if (
                        nested_candidate_dir.is_dir()
                        and (nested_candidate_dir / "poses_bounds.npy").exists()
                    ):
                        candidate_scene_dirs.append(nested_candidate_dir)
        candidate_scene_dirs = list(dict.fromkeys(candidate_scene_dirs))
        preferred_image_dir_name = self.input_data_dir.name
        for candidate_scene_dir in candidate_scene_dirs:
            candidate_image_dir = candidate_scene_dir / preferred_image_dir_name
            candidate_image_files = self._collect_image_files_from_dir(candidate_image_dir)
            if not candidate_image_files:
                candidate_image_files = self._collect_image_files_from_dir(candidate_scene_dir)
            candidate_image_names = {image_file.name for image_file in candidate_image_files}
            if input_image_names and input_image_names.issubset(candidate_image_names):
                return candidate_scene_dir
        return None
    
    def _find_images_dir(self) -> Path:
        """查找 target 监督图像目录，优先返回完整 scene 图像池。"""
        candidates = [
            self.scene_dir / self.input_data_dir.name,
            self.scene_dir / "images_8",  # 8x降采样
            self.scene_dir / "images_4",  # 4x降采样
            self.scene_dir / "images_2",  # 2x降采样
            self.scene_dir / "images",    # 原始分辨率
            self.scene_dir,               # 直接在根目录
            self.input_data_dir / "images",
            self.input_data_dir / "input",
            self.input_data_dir,
        ]
        
        for candidate in candidates:
            if self._collect_image_files_from_dir(candidate):
                return candidate
        
        # 如果找不到，返回data_dir本身
        return self.data_dir

    def _find_sparse_images_dir(self) -> Path:
        """查找稀疏 reference 图像目录，优先使用输入子集目录。"""
        candidates = [
            self.input_data_dir / "images",
            self.input_data_dir / "input",
            self.input_data_dir,
            self.images_dir,
        ]
        for candidate in candidates:
            if self._collect_image_files_from_dir(candidate):
                return candidate
        return self.input_data_dir

    def _resolve_phase1_3dgs_cache_file(self) -> Path:
        cache_root = self._resolve_offline_condition_cache_root()
        target_height, target_width = self.image_size
        if self.offline_phase1_result is not None and self.offline_phase1_result.exists():
            phase1_condition_source_tag = self.offline_phase1_result.stem
        else:
            phase1_condition_source_tag = "support_conditioned_privileged_bank"
        cache_file_name = (
            f"phase1_3dgs_geom_bank_v2_{phase1_condition_source_tag}"
            f"_h{target_height}_w{target_width}"
            f"_targets{len(self.image_files)}"
            f"_refs{len(self.sparse_image_files)}.pt"
            f"_stride{int(self.phase1_3dgs_privileged_depth_stride)}"
            f"_maxpts{int(self.phase1_3dgs_privileged_max_points)}"
        )
        return cache_root / cache_file_name

    def _resolve_support_template_stage_tag(self, raw_stage_tag: Optional[str]) -> str:
        normalized_stage_tag = str(raw_stage_tag or "").strip().lower()
        if normalized_stage_tag:
            return normalized_stage_tag
        if self.effective_num_sparse_views >= 8:
            return "stage1_pretrain"
        if self.effective_num_sparse_views <= 3:
            return "stage2_alignment"
        return f"support_{int(self.effective_num_sparse_views)}view"

    def _resolve_support_template_scope(self) -> str:
        return "scene_global" if self._use_scene_global_support_templates() else "target_local"

    def _resolve_support_template_bank_file(self) -> Path:
        cache_root = self._resolve_offline_condition_cache_root()
        target_height, target_width = self.image_size
        cache_file_name = (
            f"phase1_support_template_bank_v2_{self.support_template_stage_tag}"
            f"_h{target_height}_w{target_width}"
            f"_targets{len(self.image_files)}"
            f"_refs{len(self.sparse_image_files)}"
            f"_views{int(self.effective_num_sparse_views)}.pt"
        )
        return cache_root / cache_file_name

    def _resolve_phase1_episode_bank_dir(self) -> Path:
        cache_root = self._resolve_offline_condition_cache_root()
        target_height, target_width = self.image_size
        episode_bank_dir = cache_root / (
            f"phase1_episode_bank_v2_{self.support_template_stage_tag}"
            f"_h{target_height}_w{target_width}"
            f"_targets{len(self.image_files)}"
            f"_refs{len(self.sparse_image_files)}"
            f"_views{int(self.effective_num_sparse_views)}"
        )
        episode_bank_dir.mkdir(parents=True, exist_ok=True)
        return episode_bank_dir

    def _resolve_phase1_episode_bank_metadata_file(self) -> Path:
        return self._resolve_phase1_episode_bank_dir() / "metadata.pt"

    def _resolve_phase1_episode_tensor_file(
        self,
        *,
        target_idx: int,
        support_template_id: int,
    ) -> Path:
        return self._resolve_phase1_episode_bank_dir() / (
            f"target_{int(target_idx):04d}_template_{int(support_template_id):02d}.pt"
        )

    def _get_support_template_strategy_specs(self) -> List[Tuple[str, int]]:
        effective_view_count = max(1, int(self.effective_num_sparse_views))
        if "stage1" in self.support_template_stage_tag:
            return [
                ("nearest", effective_view_count),
                ("spread", effective_view_count * 2),
                ("wide", effective_view_count * 4),
            ]
        return [
            ("nearest", effective_view_count),
            ("spread", effective_view_count * 3),
            ("wide", effective_view_count * 5),
            ("shifted", effective_view_count * 2),
        ]

    def _use_scene_global_support_templates(self) -> bool:
        normalized_stage_tag = str(self.support_template_stage_tag).strip().lower()
        return (
            "stage2_alignment" in normalized_stage_tag
            or "stage3" in normalized_stage_tag
        ) and int(self.effective_num_sparse_views) <= 3

    def _build_sparse_camera_centers_tensor(self) -> torch.Tensor:
        sparse_camera_centers: List[torch.Tensor] = []
        for sparse_image_index in range(len(self.sparse_image_files)):
            sparse_pose_index = int(self.pose_indices_for_sparse_images[sparse_image_index].item())
            sparse_pose = self.poses[sparse_pose_index]
            sparse_rotation = sparse_pose[:3, :3]
            sparse_translation = sparse_pose[:3, 3]
            sparse_camera_center = (-sparse_rotation.transpose(0, 1) @ sparse_translation).float()
            sparse_camera_centers.append(sparse_camera_center)
        if not sparse_camera_centers:
            return torch.zeros(0, 3, dtype=torch.float32)
        return torch.stack(sparse_camera_centers, dim=0)

    def _build_scene_coverage_sparse_order(self) -> np.ndarray:
        sparse_camera_centers_tensor = self._build_sparse_camera_centers_tensor()
        total_sparse_views = int(sparse_camera_centers_tensor.shape[0])
        if total_sparse_views == 0:
            return np.asarray([], dtype=np.int64)
        if total_sparse_views == 1:
            return np.asarray([0], dtype=np.int64)

        pairwise_camera_distances = torch.cdist(
            sparse_camera_centers_tensor,
            sparse_camera_centers_tensor,
            p=2.0,
        )
        mean_camera_distances = pairwise_camera_distances.mean(dim=1)
        selected_sparse_indices: List[int] = [int(torch.argmax(mean_camera_distances).item())]
        remaining_sparse_indices: Set[int] = set(range(total_sparse_views))
        remaining_sparse_indices.discard(selected_sparse_indices[0])
        while remaining_sparse_indices:
            best_sparse_index = None
            best_sparse_distance = None
            for candidate_sparse_index in remaining_sparse_indices:
                candidate_min_distance = float(
                    pairwise_camera_distances[
                        int(candidate_sparse_index),
                        torch.as_tensor(selected_sparse_indices, dtype=torch.long),
                    ].min().item()
                )
                if best_sparse_distance is None or candidate_min_distance > best_sparse_distance:
                    best_sparse_distance = candidate_min_distance
                    best_sparse_index = int(candidate_sparse_index)
            if best_sparse_index is None:
                break
            selected_sparse_indices.append(int(best_sparse_index))
            remaining_sparse_indices.discard(int(best_sparse_index))
        return np.asarray(selected_sparse_indices, dtype=np.int64)

    def _sample_ordered_sparse_pool(
        self,
        *,
        ordered_sparse_indices: np.ndarray,
        offset: int,
        desired_count: int,
    ) -> np.ndarray:
        if ordered_sparse_indices.size == 0:
            return np.asarray([], dtype=np.int64)
        bounded_desired_count = max(1, min(int(desired_count), int(ordered_sparse_indices.size)))
        if int(ordered_sparse_indices.size) <= bounded_desired_count:
            return np.asarray(ordered_sparse_indices, dtype=np.int64)
        selected_sparse_indices: List[int] = []
        used_sparse_indices: Set[int] = set()
        scene_stride = float(ordered_sparse_indices.size) / float(bounded_desired_count)
        for sample_position in range(bounded_desired_count):
            raw_sparse_position = int(
                round(
                    (float(sample_position) + float(offset)) * scene_stride
                )
            ) % int(ordered_sparse_indices.size)
            candidate_sparse_index = int(ordered_sparse_indices[raw_sparse_position])
            if candidate_sparse_index in used_sparse_indices:
                continue
            used_sparse_indices.add(candidate_sparse_index)
            selected_sparse_indices.append(candidate_sparse_index)
        for candidate_sparse_index in ordered_sparse_indices.tolist():
            normalized_sparse_index = int(candidate_sparse_index)
            if normalized_sparse_index in used_sparse_indices:
                continue
            used_sparse_indices.add(normalized_sparse_index)
            selected_sparse_indices.append(normalized_sparse_index)
            if len(selected_sparse_indices) >= bounded_desired_count:
                break
        return np.asarray(selected_sparse_indices[:bounded_desired_count], dtype=np.int64)

    def _build_scene_global_template_candidate_specs(self) -> List[Dict[str, Any]]:
        effective_view_count = max(1, int(self.effective_num_sparse_views))
        scene_global_pool_size = max(effective_view_count + 2, effective_view_count * 4)
        sequential_sparse_indices = np.arange(len(self.sparse_image_files), dtype=np.int64)
        coverage_sparse_order = self._build_scene_coverage_sparse_order()
        candidate_specs: List[Dict[str, Any]] = []
        strategy_specs = [
            ("scene_uniform", sequential_sparse_indices, 0),
            ("scene_shifted", sequential_sparse_indices, 1),
            ("scene_coverage", coverage_sparse_order, 0),
            ("scene_coverage_shifted", coverage_sparse_order, 1),
        ]
        for strategy_name, ordered_sparse_indices, offset in strategy_specs:
            sampled_sparse_pool = self._sample_ordered_sparse_pool(
                ordered_sparse_indices=ordered_sparse_indices,
                offset=offset,
                desired_count=scene_global_pool_size,
            )
            if sampled_sparse_pool.size == 0:
                continue
            candidate_specs.append(
                {
                    "template_strategy": strategy_name,
                    "ordered_candidate_sparse_indices": [
                        int(sparse_index) for sparse_index in sampled_sparse_pool.tolist()
                    ],
                }
            )
        return candidate_specs

    def _materialize_scene_global_support_template_indices(
        self,
        *,
        target_pose_index: int,
        ordered_candidate_sparse_indices: Sequence[int],
    ) -> np.ndarray:
        selected_sparse_indices: List[int] = []
        seen_sparse_indices: Set[int] = set()
        for sparse_index in ordered_candidate_sparse_indices:
            normalized_sparse_index = int(sparse_index)
            if normalized_sparse_index in seen_sparse_indices:
                continue
            if int(self.pose_indices_for_sparse_images[normalized_sparse_index].item()) == target_pose_index:
                continue
            seen_sparse_indices.add(normalized_sparse_index)
            selected_sparse_indices.append(normalized_sparse_index)
            if len(selected_sparse_indices) >= int(self.effective_num_sparse_views):
                break
        if len(selected_sparse_indices) < int(self.effective_num_sparse_views):
            for fallback_sparse_index in range(len(self.sparse_image_files)):
                if int(self.pose_indices_for_sparse_images[int(fallback_sparse_index)].item()) == target_pose_index:
                    continue
                if int(fallback_sparse_index) in seen_sparse_indices:
                    continue
                seen_sparse_indices.add(int(fallback_sparse_index))
                selected_sparse_indices.append(int(fallback_sparse_index))
                if len(selected_sparse_indices) >= int(self.effective_num_sparse_views):
                    break
        return np.asarray(
            selected_sparse_indices[: int(self.effective_num_sparse_views)],
            dtype=np.int64,
        )

    def _build_scene_global_support_template_bank(self) -> Dict[str, Any]:
        scene_global_candidate_specs = self._build_scene_global_template_candidate_specs()
        support_templates_by_target: Dict[int, List[Dict[str, Any]]] = {}
        for target_idx in self.indices:
            target_pose_index = int(self.pose_indices_for_images[int(target_idx)].item())
            target_support_templates: List[Dict[str, Any]] = []
            seen_support_templates: Set[Tuple[int, ...]] = set()
            for candidate_spec in scene_global_candidate_specs:
                materialized_sparse_indices = self._materialize_scene_global_support_template_indices(
                    target_pose_index=target_pose_index,
                    ordered_candidate_sparse_indices=candidate_spec["ordered_candidate_sparse_indices"],
                )
                if materialized_sparse_indices.size < int(self.effective_num_sparse_views):
                    continue
                support_template_tuple = tuple(
                    int(sparse_index) for sparse_index in materialized_sparse_indices.tolist()
                )
                if support_template_tuple in seen_support_templates:
                    continue
                seen_support_templates.add(support_template_tuple)
                target_support_templates.append(
                    {
                        "support_template_id": len(target_support_templates),
                        "template_strategy": str(candidate_spec["template_strategy"]),
                        "template_scope": "scene_global",
                        "sparse_indices": [
                            int(sparse_index) for sparse_index in materialized_sparse_indices.tolist()
                        ],
                        "support_image_names": [
                            self.sparse_image_files[int(sparse_index)].name
                            for sparse_index in materialized_sparse_indices.tolist()
                        ],
                        "support_condition_key": self._make_support_condition_key(
                            materialized_sparse_indices
                        ),
                    }
                )
            support_templates_by_target[int(target_idx)] = target_support_templates
        return {
            "cache_version": 2,
            "support_template_stage_tag": self.support_template_stage_tag,
            "support_template_scope": self.support_template_scope,
            "effective_num_sparse_views": int(self.effective_num_sparse_views),
            "target_image_names": [image_file.name for image_file in self.image_files],
            "sparse_image_names": [image_file.name for image_file in self.sparse_image_files],
            "support_templates_by_target": support_templates_by_target,
        }

    def _finalize_support_template_indices(
        self,
        *,
        initial_sparse_indices: Sequence[int],
        ranked_sparse_indices: np.ndarray,
    ) -> np.ndarray:
        unique_sparse_indices: List[int] = []
        seen_sparse_indices: Set[int] = set()
        for sparse_index in list(initial_sparse_indices) + ranked_sparse_indices.tolist():
            normalized_sparse_index = int(sparse_index)
            if normalized_sparse_index in seen_sparse_indices:
                continue
            seen_sparse_indices.add(normalized_sparse_index)
            unique_sparse_indices.append(normalized_sparse_index)
            if len(unique_sparse_indices) >= self.effective_num_sparse_views:
                break
        return np.asarray(unique_sparse_indices[: self.effective_num_sparse_views], dtype=np.int64)

    def _select_support_template_indices(
        self,
        *,
        ranked_sparse_indices: np.ndarray,
        strategy_name: str,
        candidate_pool_size: int,
    ) -> np.ndarray:
        if ranked_sparse_indices.size == 0:
            return np.asarray([], dtype=np.int64)
        effective_view_count = max(1, int(self.effective_num_sparse_views))
        bounded_pool_size = min(
            ranked_sparse_indices.size,
            max(int(candidate_pool_size), effective_view_count),
        )
        candidate_pool = ranked_sparse_indices[:bounded_pool_size]
        if candidate_pool.size == 0:
            return np.asarray([], dtype=np.int64)

        if strategy_name == "nearest":
            initial_sparse_indices = candidate_pool[:effective_view_count].tolist()
        elif strategy_name == "spread":
            sampled_positions = np.linspace(
                0,
                max(candidate_pool.size - 1, 0),
                num=effective_view_count,
            ).round().astype(np.int64)
            initial_sparse_indices = [int(candidate_pool[position]) for position in sampled_positions.tolist()]
        elif strategy_name == "wide":
            wide_pool = ranked_sparse_indices[
                : min(ranked_sparse_indices.size, max(candidate_pool_size, effective_view_count * 6))
            ]
            sampled_positions = np.linspace(
                0,
                max(wide_pool.size - 1, 0),
                num=effective_view_count,
            ).round().astype(np.int64)
            initial_sparse_indices = [int(wide_pool[position]) for position in sampled_positions.tolist()]
        elif strategy_name == "shifted":
            shifted_start_index = 1 if ranked_sparse_indices.size > effective_view_count else 0
            shifted_pool = ranked_sparse_indices[shifted_start_index : shifted_start_index + effective_view_count]
            initial_sparse_indices = shifted_pool.tolist()
        else:
            initial_sparse_indices = candidate_pool[:effective_view_count].tolist()

        return self._finalize_support_template_indices(
            initial_sparse_indices=initial_sparse_indices,
            ranked_sparse_indices=ranked_sparse_indices,
        )

    def _build_support_template_bank(self) -> Dict[str, Any]:
        if self._use_scene_global_support_templates():
            return self._build_scene_global_support_template_bank()
        support_template_strategy_specs = self._get_support_template_strategy_specs()
        support_templates_by_target: Dict[int, List[Dict[str, Any]]] = {}
        for target_idx in self.indices:
            ranked_sparse_indices = self._nearest_sparse_indices_by_target.get(
                int(target_idx),
                np.asarray([], dtype=np.int64),
            )
            target_support_templates: List[Dict[str, Any]] = []
            seen_support_templates: Set[Tuple[int, ...]] = set()
            for strategy_name, candidate_pool_size in support_template_strategy_specs:
                selected_sparse_indices = self._select_support_template_indices(
                    ranked_sparse_indices=ranked_sparse_indices,
                    strategy_name=strategy_name,
                    candidate_pool_size=candidate_pool_size,
                )
                if selected_sparse_indices.size < self.effective_num_sparse_views:
                    continue
                support_template_tuple = tuple(int(index) for index in selected_sparse_indices.tolist())
                if support_template_tuple in seen_support_templates:
                    continue
                seen_support_templates.add(support_template_tuple)
                target_support_templates.append(
                    {
                        "support_template_id": len(target_support_templates),
                        "template_strategy": strategy_name,
                        "template_scope": "target_local",
                        "sparse_indices": [int(index) for index in selected_sparse_indices.tolist()],
                        "support_image_names": [
                            self.sparse_image_files[int(index)].name
                            for index in selected_sparse_indices.tolist()
                        ],
                        "support_condition_key": self._make_support_condition_key(selected_sparse_indices),
                    }
                )
            if not target_support_templates and ranked_sparse_indices.size > 0:
                fallback_sparse_indices = self._finalize_support_template_indices(
                    initial_sparse_indices=ranked_sparse_indices[: self.effective_num_sparse_views].tolist(),
                    ranked_sparse_indices=ranked_sparse_indices,
                )
                if fallback_sparse_indices.size >= self.effective_num_sparse_views:
                    target_support_templates.append(
                        {
                            "support_template_id": 0,
                            "template_strategy": "fallback_nearest",
                            "template_scope": "target_local",
                            "sparse_indices": [int(index) for index in fallback_sparse_indices.tolist()],
                            "support_image_names": [
                                self.sparse_image_files[int(index)].name
                                for index in fallback_sparse_indices.tolist()
                            ],
                            "support_condition_key": self._make_support_condition_key(fallback_sparse_indices),
                        }
                    )
            support_templates_by_target[int(target_idx)] = target_support_templates
        return {
            "cache_version": 2,
            "support_template_stage_tag": self.support_template_stage_tag,
            "effective_num_sparse_views": int(self.effective_num_sparse_views),
            "target_image_names": [image_file.name for image_file in self.image_files],
            "sparse_image_names": [image_file.name for image_file in self.sparse_image_files],
            "support_templates_by_target": support_templates_by_target,
        }

    def _validate_support_template_poses(
        self,
        support_templates_by_target: Dict[int, List[Dict[str, Any]]],
        source: str,
    ) -> None:
        for raw_target_idx, templates in support_templates_by_target.items():
            try:
                target_idx = int(raw_target_idx)
                if target_idx < 0 or target_idx >= len(self.pose_indices_for_images):
                    raise ValueError(f"target_idx={target_idx} is out of range")
                target_pose_index = int(self.pose_indices_for_images[target_idx].item())
                if not isinstance(templates, list):
                    raise ValueError("template list is invalid")
                for template in templates:
                    if not isinstance(template, dict) or not isinstance(template.get("sparse_indices"), list):
                        raise ValueError("sparse_indices is invalid")
                    for raw_sparse_idx in template["sparse_indices"]:
                        sparse_idx = int(raw_sparse_idx)
                        if sparse_idx < 0 or sparse_idx >= len(self.pose_indices_for_sparse_images):
                            raise ValueError(f"sparse_idx={sparse_idx} is out of range")
                        sparse_pose_index = int(self.pose_indices_for_sparse_images[sparse_idx].item())
                        if sparse_pose_index == target_pose_index:
                            raise ValueError(
                                f"target_idx={target_idx}, sparse_idx={sparse_idx}, pose_idx={target_pose_index}"
                            )
            except (TypeError, ValueError, IndexError) as error:
                raise RuntimeError(
                    f"[LLFFDataset] Rejected support template with invalid or self-view pose in {source}: {error}"
                ) from error

    def _load_support_template_bank(
        self,
        cache_path: Path,
    ) -> Optional[Dict[int, List[Dict[str, Any]]]]:
        if not cache_path.exists():
            return None
        cache_payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache_payload, dict):
            return None
        if int(cache_payload.get("cache_version", 0)) < 2:
            return None
        if str(cache_payload.get("support_template_stage_tag", "")) != self.support_template_stage_tag:
            return None
        cached_support_template_scope = str(
            cache_payload.get("support_template_scope", "target_local")
        ).strip().lower()
        if cached_support_template_scope != self.support_template_scope:
            return None
        if int(cache_payload.get("effective_num_sparse_views", 0)) != int(self.effective_num_sparse_views):
            return None
        current_target_names = [image_file.name for image_file in self.image_files]
        current_sparse_names = [image_file.name for image_file in self.sparse_image_files]
        if cache_payload.get("target_image_names") != current_target_names:
            return None
        if cache_payload.get("sparse_image_names") != current_sparse_names:
            return None
        support_templates_by_target = cache_payload.get("support_templates_by_target")
        if not isinstance(support_templates_by_target, dict):
            return None
        self._validate_support_template_poses(support_templates_by_target, str(cache_path))
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Loaded support template bank from {cache_path}")
        return {
            int(target_idx): list(template_list)
            for target_idx, template_list in support_templates_by_target.items()
        }

    def _load_or_build_support_template_bank(self) -> Dict[int, List[Dict[str, Any]]]:
        cache_path = self._resolve_support_template_bank_file()
        cached_support_template_bank = self._load_support_template_bank(cache_path)
        if cached_support_template_bank is not None:
            return cached_support_template_bank
        built_payload = self._build_support_template_bank()
        self._validate_support_template_poses(built_payload["support_templates_by_target"], "new support template bank")
        torch.save(built_payload, cache_path)
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Saved support template bank to {cache_path}")
        return {
            int(target_idx): list(template_list)
            for target_idx, template_list in built_payload["support_templates_by_target"].items()
        }

    def _serialize_phase1_episode_package(
        self,
        episode_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        serialized_episode_package: Dict[str, Any] = {}
        for package_key, package_value in episode_package.items():
            if torch.is_tensor(package_value):
                if package_value.dtype.is_floating_point:
                    serialized_episode_package[package_key] = package_value.detach().cpu().to(torch.float16)
                else:
                    serialized_episode_package[package_key] = package_value.detach().cpu()
            else:
                serialized_episode_package[package_key] = package_value
        return serialized_episode_package

    def _deserialize_phase1_episode_package(
        self,
        serialized_episode_package: Dict[str, Any],
    ) -> Dict[str, Any]:
        deserialized_episode_package: Dict[str, Any] = {}
        for package_key, package_value in serialized_episode_package.items():
            if torch.is_tensor(package_value):
                if package_value.dtype.is_floating_point:
                    deserialized_episode_package[package_key] = package_value.detach().cpu().float()
                else:
                    deserialized_episode_package[package_key] = package_value.detach().cpu()
            else:
                deserialized_episode_package[package_key] = package_value
        return deserialized_episode_package

    def _tensor_mean_as_float(
        self,
        tensor: Optional[torch.Tensor],
    ) -> float:
        if tensor is None or not torch.is_tensor(tensor):
            return 0.0
        return float(
            torch.nan_to_num(
                tensor.detach().float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).mean().item()
        )

    def _build_phase1_episode_stats(
        self,
        phase1_episode_package: Dict[str, Any],
    ) -> Dict[str, float]:
        support_valid_coverage = self._tensor_mean_as_float(
            phase1_episode_package.get("warp_valid_mask")
        )
        support_confidence_mean = self._tensor_mean_as_float(
            phase1_episode_package.get("warp_support_confidence")
        )
        frontier_coverage = self._tensor_mean_as_float(
            phase1_episode_package.get("frontier_mask")
        )
        verification_prior_mean = self._tensor_mean_as_float(
            phase1_episode_package.get("verification_prior")
        )
        novel_coverage = float(max(0.0, min(1.0, 1.0 - support_valid_coverage)))
        proposal_hardness_score = float(
            novel_coverage
            + 0.75 * frontier_coverage
            + 0.25 * max(0.0, 1.0 - support_confidence_mean)
        )
        return {
            "support_valid_coverage": support_valid_coverage,
            "support_confidence_mean": support_confidence_mean,
            "frontier_coverage": frontier_coverage,
            "novel_coverage": novel_coverage,
            "verification_prior_mean": verification_prior_mean,
            "proposal_hardness_score": proposal_hardness_score,
        }

    def _index_phase1_episode_bank_metadata(
        self,
        metadata_payload: Dict[str, Any],
    ) -> None:
        self._phase1_episode_file_lookup = {}
        episode_specs = metadata_payload.get("episode_specs", [])
        episode_bank_dir = self._resolve_phase1_episode_bank_dir()
        for episode_spec in episode_specs:
            target_idx = int(episode_spec["target_idx"])
            support_template_id = int(episode_spec["support_template_id"])
            episode_file_name = str(episode_spec["episode_file_name"])
            self._phase1_episode_file_lookup[(target_idx, support_template_id)] = (
                episode_bank_dir / episode_file_name
            )

    def _load_phase1_episode_bank_metadata(self) -> Optional[Dict[str, Any]]:
        metadata_path = self._resolve_phase1_episode_bank_metadata_file()
        if not metadata_path.exists():
            return None
        metadata_payload = torch.load(metadata_path, map_location="cpu")
        if not isinstance(metadata_payload, dict):
            return None
        if int(metadata_payload.get("cache_version", 0)) < 4:
            return None
        if str(metadata_payload.get("support_template_stage_tag", "")) != self.support_template_stage_tag:
            return None
        cached_support_template_scope = str(
            metadata_payload.get("support_template_scope", "target_local")
        ).strip().lower()
        if cached_support_template_scope != self.support_template_scope:
            return None
        if int(metadata_payload.get("effective_num_sparse_views", 0)) != int(self.effective_num_sparse_views):
            return None
        if metadata_payload.get("target_image_names") != [image_file.name for image_file in self.image_files]:
            return None
        if metadata_payload.get("sparse_image_names") != [image_file.name for image_file in self.sparse_image_files]:
            return None
        cached_support_templates_by_target = metadata_payload.get("support_templates_by_target")
        if isinstance(cached_support_templates_by_target, dict):
            self._validate_support_template_poses(cached_support_templates_by_target, str(metadata_path))
        if cached_support_templates_by_target != self._phase1_support_template_bank:
            return None
        cached_episode_specs = metadata_payload.get("episode_specs")
        if not isinstance(cached_episode_specs, list) or len(cached_episode_specs) == 0:
            return None
        episode_bank_dir = self._resolve_phase1_episode_bank_dir()
        for episode_spec in cached_episode_specs:
            episode_file_name = str(episode_spec.get("episode_file_name", ""))
            if not episode_file_name:
                return None
            if not (episode_bank_dir / episode_file_name).exists():
                return None
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Loaded Phase1 episode bank metadata from {metadata_path}")
        return metadata_payload

    def _build_phase1_episode_bank(self) -> Dict[str, Any]:
        if self._phase1_3dgs_runtime_geometry_bank is None:
            raise RuntimeError("[LLFFDataset] 构建 Phase1 episode bank 前缺少 runtime geometry bank。")
        episode_specs: List[Dict[str, Any]] = []
        for target_idx in sorted(self._phase1_support_template_bank.keys()):
            target_support_templates = self._phase1_support_template_bank.get(int(target_idx), [])
            if not target_support_templates:
                continue
            target_pose_index = int(self.pose_indices_for_images[int(target_idx)].item())
            target_pose = self.poses[target_pose_index]
            target_intrinsics = self.camera_intrinsics[target_pose_index]
            for support_template in target_support_templates:
                support_template_id = int(support_template["support_template_id"])
                sparse_indices = np.asarray(support_template["sparse_indices"], dtype=np.int64)
                sparse_images = torch.stack([
                    self._load_image(self.sparse_image_files[int(sparse_index)])
                    for sparse_index in sparse_indices
                ])
                sparse_poses = torch.stack([
                    self.poses[int(self.pose_indices_for_sparse_images[int(sparse_index)].item())]
                    for sparse_index in sparse_indices
                ])
                sparse_intrinsics = torch.stack([
                    self.camera_intrinsics[int(self.pose_indices_for_sparse_images[int(sparse_index)].item())]
                    for sparse_index in sparse_indices
                ])
                phase1_episode_package = self._build_support_conditioned_phase1_package(
                    target_idx=int(target_idx),
                    target_pose=target_pose,
                    target_intrinsics=target_intrinsics,
                    sparse_indices=sparse_indices,
                    sparse_images=sparse_images,
                    sparse_poses=sparse_poses,
                    sparse_intrinsics=sparse_intrinsics,
                )
                episode_tensor_file = self._resolve_phase1_episode_tensor_file(
                    target_idx=int(target_idx),
                    support_template_id=support_template_id,
                )
                torch.save(
                    self._serialize_phase1_episode_package(phase1_episode_package),
                    episode_tensor_file,
                )
                episode_stats = self._build_phase1_episode_stats(phase1_episode_package)
                episode_specs.append(
                    {
                        "target_idx": int(target_idx),
                        "support_template_id": support_template_id,
                        "template_strategy": str(support_template["template_strategy"]),
                        "support_condition_key": str(support_template["support_condition_key"]),
                        "episode_file_name": episode_tensor_file.name,
                        **episode_stats,
                    }
                )
        metadata_payload = {
            "cache_version": 4,
            "support_template_stage_tag": self.support_template_stage_tag,
            "support_template_scope": self.support_template_scope,
            "effective_num_sparse_views": int(self.effective_num_sparse_views),
            "target_image_names": [image_file.name for image_file in self.image_files],
            "sparse_image_names": [image_file.name for image_file in self.sparse_image_files],
            "support_templates_by_target": self._phase1_support_template_bank,
            "episode_specs": episode_specs,
        }
        metadata_path = self._resolve_phase1_episode_bank_metadata_file()
        torch.save(metadata_payload, metadata_path)
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Saved Phase1 episode bank metadata to {metadata_path}")
        return metadata_payload

    def _build_phase1_episode_sample_specs(self) -> List[Dict[str, Any]]:
        indexed_target_indices = {int(target_idx) for target_idx in self.indices}
        metadata_episode_specs = []
        if isinstance(self._phase1_episode_bank_metadata, dict):
            metadata_episode_specs = list(
                self._phase1_episode_bank_metadata.get("episode_specs", [])
            )

        episode_sample_specs: List[Dict[str, Any]] = []
        if metadata_episode_specs:
            for episode_spec in metadata_episode_specs:
                target_idx = int(episode_spec.get("target_idx", -1))
                if target_idx not in indexed_target_indices:
                    continue
                episode_sample_specs.append(
                    {
                        "target_idx": target_idx,
                        "support_template_id": int(episode_spec.get("support_template_id", -1)),
                        "support_valid_coverage": float(
                            episode_spec.get("support_valid_coverage", 0.0)
                        ),
                        "support_confidence_mean": float(
                            episode_spec.get("support_confidence_mean", 0.0)
                        ),
                        "frontier_coverage": float(
                            episode_spec.get("frontier_coverage", 0.0)
                        ),
                        "novel_coverage": float(
                            episode_spec.get("novel_coverage", 0.0)
                        ),
                        "verification_prior_mean": float(
                            episode_spec.get("verification_prior_mean", 0.0)
                        ),
                        "proposal_hardness_score": float(
                            episode_spec.get("proposal_hardness_score", 0.0)
                        ),
                    }
                )
            episode_sample_specs.sort(
                key=lambda episode_spec: (
                    float(episode_spec.get("proposal_hardness_score", 0.0)),
                    float(episode_spec.get("frontier_coverage", 0.0)),
                    float(episode_spec.get("novel_coverage", 0.0)),
                    -float(episode_spec.get("support_valid_coverage", 0.0)),
                ),
                reverse=True,
            )
            self._phase1_total_episode_sample_spec_count = len(episode_sample_specs)
            selected_episode_sample_specs = list(episode_sample_specs)
            if self.hard_episode_top_fraction < 0.999 and selected_episode_sample_specs:
                selected_episode_count = max(
                    1,
                    int(
                        math.ceil(
                            float(len(selected_episode_sample_specs))
                            * float(self.hard_episode_top_fraction)
                        )
                    ),
                )
                selected_episode_sample_specs = selected_episode_sample_specs[:selected_episode_count]
            self._phase1_selected_episode_sample_spec_count = len(selected_episode_sample_specs)
            if self.is_train and self.hard_episode_repeat_factor > 1 and selected_episode_sample_specs:
                expanded_episode_sample_specs: List[Dict[str, Any]] = []
                for _ in range(self.hard_episode_repeat_factor):
                    expanded_episode_sample_specs.extend(
                        dict(episode_sample_spec)
                        for episode_sample_spec in selected_episode_sample_specs
                    )
                return expanded_episode_sample_specs
            return selected_episode_sample_specs

        for target_idx in self.indices:
            target_support_templates = self._phase1_support_template_bank.get(int(target_idx), [])
            for support_template in target_support_templates:
                episode_sample_specs.append(
                    {
                        "target_idx": int(target_idx),
                        "support_template_id": int(support_template["support_template_id"]),
                        "support_valid_coverage": 0.0,
                        "support_confidence_mean": 0.0,
                        "frontier_coverage": 0.0,
                        "novel_coverage": 0.0,
                        "verification_prior_mean": 0.0,
                        "proposal_hardness_score": 0.0,
                    }
                )
        self._phase1_total_episode_sample_spec_count = len(episode_sample_specs)
        self._phase1_selected_episode_sample_spec_count = len(episode_sample_specs)
        return episode_sample_specs

    def _lookup_prebuilt_phase1_episode_package(
        self,
        *,
        target_idx: int,
        support_template_id: int,
    ) -> Dict[str, Any]:
        episode_cache_key = f"{int(target_idx)}::{int(support_template_id)}"
        cached_episode_package = self._phase1_episode_tensor_cache.get(episode_cache_key)
        if cached_episode_package is not None:
            self._phase1_episode_tensor_cache.move_to_end(episode_cache_key)
            return self._deserialize_phase1_episode_package(cached_episode_package)
        episode_tensor_file = self._phase1_episode_file_lookup.get(
            (int(target_idx), int(support_template_id))
        )
        if episode_tensor_file is None or not episode_tensor_file.exists():
            raise RuntimeError(
                "[LLFFDataset] 缺少预构建 Phase1 episode: "
                f"target_idx={int(target_idx)}, support_template_id={int(support_template_id)}"
            )
        serialized_episode_package = torch.load(episode_tensor_file, map_location="cpu")
        if not isinstance(serialized_episode_package, dict):
            raise RuntimeError(f"[LLFFDataset] 非法 Phase1 episode 文件: {episode_tensor_file}")
        self._phase1_episode_tensor_cache[episode_cache_key] = serialized_episode_package
        self._phase1_episode_tensor_cache.move_to_end(episode_cache_key)
        while len(self._phase1_episode_tensor_cache) > self._phase1_episode_tensor_cache_limit:
            self._phase1_episode_tensor_cache.popitem(last=False)
        return self._deserialize_phase1_episode_package(serialized_episode_package)

    def _build_existing_camera_from_pose_index(self, pose_index: int):
        from examples.active_sampling.viewpoint_sampler import ExistingCamera

        pose_index = int(pose_index)
        view_matrix = self.poses[pose_index].detach().cpu().float()
        camera_intrinsics = self.camera_intrinsics[pose_index].detach().cpu().float()
        camera_to_world = torch.linalg.inv(view_matrix)
        camera_position = camera_to_world[:3, 3].contiguous()
        return ExistingCamera(
            position=camera_position,
            viewmat=view_matrix,
            intrinsics=camera_intrinsics,
        )

    def _load_phase1_3dgs_condition_cache(
        self,
        cache_path: Path,
    ) -> Optional[Dict[str, Any]]:
        if not cache_path.exists():
            return None
        cache_payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(cache_payload, dict):
            return None
        cached_target_names = cache_payload.get("target_image_names")
        cached_sparse_names = cache_payload.get("sparse_image_names")
        current_target_names = [image_file.name for image_file in self.image_files]
        current_sparse_names = [image_file.name for image_file in self.sparse_image_files]
        if cached_target_names != current_target_names or cached_sparse_names != current_sparse_names:
            return None
        required_tensor_keys = [
            "gaussians_state",
        ]
        for required_key in required_tensor_keys:
            if required_key not in cache_payload:
                return None
        cache_version = int(cache_payload.get("cache_version", 0))
        if cache_version < 2:
            return None
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Loaded Phase1 3DGS geometry bank cache from {cache_path}")
        return cache_payload

    def _extract_gaussians_state_from_field(self, gaussians) -> Dict[str, Any]:
        return field_state(gaussians)

    def _build_privileged_dense_point_bank(
        self,
        *,
        sparse_points_world: np.ndarray,
        sparse_point_colors: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        self._ensure_warp_fallback_ready()

        sparse_points_tensor = torch.from_numpy(sparse_points_world.astype(np.float32))
        sparse_colors_tensor = torch.from_numpy(sparse_point_colors.astype(np.float32)).clamp(
            1e-4,
            1.0 - 1.0e-4,
        )
        dense_point_chunks: List[torch.Tensor] = []
        dense_color_chunks: List[torch.Tensor] = []
        stride = max(1, int(self.phase1_3dgs_privileged_depth_stride))

        for image_index, image_file in enumerate(self.image_files):
            pose_index = int(self.pose_indices_for_images[image_index].item())
            image_tensor = self._load_image(image_file).cpu().float()
            depth_tensor = self._get_cached_depth_map_for_image_index(int(image_index)).cpu().float()
            if depth_tensor.ndim != 3 or depth_tensor.shape[0] != 1:
                continue
            _, image_height, image_width = depth_tensor.shape
            y_coords = torch.arange(0, image_height, stride, dtype=torch.long)
            x_coords = torch.arange(0, image_width, stride, dtype=torch.long)
            if y_coords.numel() == 0 or x_coords.numel() == 0:
                continue
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
            sampled_depths = depth_tensor[0, grid_y, grid_x]
            valid_mask = torch.isfinite(sampled_depths) & (sampled_depths > 1.0e-4)
            if not bool(valid_mask.any().item()):
                continue
            sampled_x = grid_x[valid_mask].reshape(-1).to(dtype=torch.float32)
            sampled_y = grid_y[valid_mask].reshape(-1).to(dtype=torch.float32)
            sampled_depths = sampled_depths[valid_mask].reshape(-1).to(dtype=torch.float32)
            camera_to_world = torch.linalg.inv(self.poses[pose_index].detach().cpu().float())
            camera_intrinsics = self.camera_intrinsics[pose_index].detach().cpu().float()
            world_points = _unproject_pixels(
                camera_to_world=camera_to_world,
                x_pixels=sampled_x,
                y_pixels=sampled_y,
                depths=sampled_depths,
                intrinsics=camera_intrinsics,
            ).cpu()
            sampled_colors = image_tensor[:, grid_y[valid_mask], grid_x[valid_mask]].permute(1, 0).contiguous()
            dense_point_chunks.append(world_points)
            dense_color_chunks.append(sampled_colors.clamp(1e-4, 1.0 - 1.0e-4))

        if dense_point_chunks:
            dense_points_tensor = torch.cat(dense_point_chunks, dim=0)
            dense_colors_tensor = torch.cat(dense_color_chunks, dim=0)
            remaining_capacity = max(
                int(self.phase1_3dgs_privileged_max_points) - int(sparse_points_tensor.shape[0]),
                0,
            )
            if remaining_capacity > 0 and dense_points_tensor.shape[0] > remaining_capacity:
                dense_keep_indices = torch.randperm(dense_points_tensor.shape[0])[:remaining_capacity]
                dense_points_tensor = dense_points_tensor[dense_keep_indices]
                dense_colors_tensor = dense_colors_tensor[dense_keep_indices]
            elif remaining_capacity <= 0:
                dense_points_tensor = dense_points_tensor[:0]
                dense_colors_tensor = dense_colors_tensor[:0]
            combined_points_tensor = torch.cat([sparse_points_tensor, dense_points_tensor], dim=0)
            combined_colors_tensor = torch.cat([sparse_colors_tensor, dense_colors_tensor], dim=0)
            condition_source_description = "privileged_dense_bank:sparse_colmap+all_view_depth_backprojection"
        else:
            combined_points_tensor = sparse_points_tensor
            combined_colors_tensor = sparse_colors_tensor
            condition_source_description = "sparse_colmap_bootstrap"
        return combined_points_tensor, combined_colors_tensor, condition_source_description

    def _materialize_phase1_3dgs_geometry_bank(
        self,
        cache_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        from examples.active_sampling.train_active_sampling import GaussianPointField

        render_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gaussians_state = cache_payload.get("gaussians_state")
        if not isinstance(gaussians_state, dict):
            raise RuntimeError("[LLFFDataset] geometry bank cache 缺少 gaussians_state。")
        means = gaussians_state["means"].float()
        colors = gaussians_state["colors"].float().clamp(1e-4, 1.0 - 1.0e-4)
        gaussians = GaussianPointField(points=means, colors=colors, device=render_device)
        load_field(gaussians, gaussians_state, render_device)
        gaussians.eval()
        return {
            "gaussians": gaussians,
            "render_device": render_device,
            "condition_source": str(cache_payload.get("condition_source", "unknown")),
        }

    def _build_phase1_3dgs_condition_cache(self) -> Dict[str, Any]:
        from examples.active_sampling.train_active_sampling import (
            GaussianPointField,
            read_points3d_binary,
        )

        render_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        phase1_condition_source_description = "unknown"
        if self.offline_phase1_result is not None and self.offline_phase1_result.exists():
            phase1_payload = torch.load(self.offline_phase1_result, map_location="cpu")
            gaussians_state = phase1_payload.get("gaussians_state")
            if not isinstance(gaussians_state, dict):
                raise RuntimeError(
                    f"[LLFFDataset] phase1_result.pt 中缺少 gaussians_state: {self.offline_phase1_result}"
                )

            means = gaussians_state["means"].float()
            colors = gaussians_state["colors"].float().clamp(1e-4, 1.0 - 1.0e-4)
            gaussians = GaussianPointField(points=means, colors=colors, device=render_device)
            load_field(gaussians, gaussians_state, render_device)
            phase1_condition_source_description = (
                f"phase1_result:{self.offline_phase1_result.name}"
            )
            gaussians_state = self._extract_gaussians_state_from_field(gaussians)
        else:
            sparse_reconstruction_dir = self._find_colmap_sparse_dir()
            if sparse_reconstruction_dir is None:
                checked_candidate_paths = ", ".join(
                    str(candidate_path) for candidate_path in self._last_colmap_sparse_dir_candidates[:12]
                )
                raise FileNotFoundError(
                    "[LLFFDataset] phase1_3dgs 离线 bootstrap 失败：未找到 COLMAP sparse/0 资产。"
                    f" checked_candidates=[{checked_candidate_paths}]"
                )
            sparse_points_world, sparse_point_colors = read_points3d_binary(
                sparse_reconstruction_dir / "points3D.bin"
            )
            if sparse_points_world.size == 0:
                raise RuntimeError(
                    f"[LLFFDataset] phase1_3dgs 离线 bootstrap 失败：点云为空 {sparse_reconstruction_dir / 'points3D.bin'}"
                )
            privileged_points, privileged_colors, privileged_source_description = (
                self._build_privileged_dense_point_bank(
                    sparse_points_world=sparse_points_world,
                    sparse_point_colors=sparse_point_colors,
                )
            )
            gaussians = GaussianPointField(
                points=privileged_points,
                colors=privileged_colors,
                device=render_device,
            )
            phase1_condition_source_description = (
                f"{privileged_source_description}:{sparse_reconstruction_dir}"
            )
            if self.verbose_dataset_logs:
                print(
                    "[LLFFDataset] 未提供有效 phase1_result.pt，"
                    "phase1_3dgs 已回退到 support-conditioned privileged geometry bank 构建流程。"
                )
            gaussians_state = self._extract_gaussians_state_from_field(gaussians)

        if self.verbose_dataset_logs:
            print(
                "[LLFFDataset] 已构建 Phase1 geometry bank 源: "
                f"{phase1_condition_source_description}"
            )
        return {
            "cache_version": 2,
            "target_image_names": [image_file.name for image_file in self.image_files],
            "sparse_image_names": [image_file.name for image_file in self.sparse_image_files],
            "condition_source": phase1_condition_source_description,
            "gaussians_state": gaussians_state,
        }

    def _load_or_build_phase1_3dgs_condition_cache(self) -> Dict[str, Any]:
        cache_path = self._resolve_phase1_3dgs_cache_file()
        cached_payload = self._load_phase1_3dgs_condition_cache(cache_path)
        if cached_payload is not None:
            return cached_payload
        built_payload = self._build_phase1_3dgs_condition_cache()
        torch.save(built_payload, cache_path)
        if self.verbose_dataset_logs:
            print(f"[LLFFDataset] Saved Phase1 3DGS geometry bank cache to {cache_path}")
        return built_payload

    def _build_pose_index_mapping(
        self,
        image_files: Optional[list[Path]] = None,
        *,
        image_directory: Optional[Path] = None,
    ) -> torch.Tensor:
        """构建给定 image_files 到全局 poses 的索引映射。"""
        if image_files is None:
            image_files = self.image_files
        num_images = len(image_files)
        num_poses = int(self.poses.shape[0])
        if num_images == num_poses:
            return torch.arange(num_images, dtype=torch.long)
        input_image_names = [image_file.name for image_file in image_files]
        preferred_reference_name_to_pose_index: Optional[dict[str, int]] = None
        candidate_reference_dirs = [
            self.scene_dir / self.input_data_dir.name,
            self.scene_dir / "images_8",
            self.scene_dir / "images_4",
            self.scene_dir / "images_2",
            self.scene_dir / "images",
            self.scene_dir,
        ]
        reference_name_to_pose_index: Optional[dict[str, int]] = None
        for candidate_reference_dir in candidate_reference_dirs:
            candidate_image_files = self._collect_image_files_from_dir(candidate_reference_dir)
            if not candidate_image_files:
                continue
            candidate_image_names = [image_file.name for image_file in candidate_image_files]
            if all(image_name in candidate_image_names for image_name in input_image_names):
                candidate_name_to_pose_index = {
                    image_name: pose_index
                    for pose_index, image_name in enumerate(candidate_image_names)
                }
                if len(candidate_image_names) == num_poses:
                    reference_name_to_pose_index = candidate_name_to_pose_index
                    break
                if preferred_reference_name_to_pose_index is None:
                    preferred_reference_name_to_pose_index = candidate_name_to_pose_index
        if reference_name_to_pose_index is None:
            reference_name_to_pose_index = preferred_reference_name_to_pose_index
        if reference_name_to_pose_index is None:
            parsed_pose_indices: list[int] = []
            for image_name in input_image_names:
                image_stem = Path(image_name).stem
                matched_index = re.search(r"(\d+)$", image_stem)
                if matched_index is None:
                    parsed_pose_indices = []
                    break
                pose_index = int(matched_index.group(1))
                if pose_index < 0 or pose_index >= num_poses:
                    parsed_pose_indices = []
                    break
                parsed_pose_indices.append(pose_index)
            if len(parsed_pose_indices) == len(input_image_names):
                return torch.tensor(parsed_pose_indices, dtype=torch.long)
        if reference_name_to_pose_index is None:
            raise RuntimeError(
                "[LLFFDataset] Unable to align input images with poses. "
                f"input_data_dir={self.input_data_dir}, "
                f"images_dir={image_directory or self.images_dir}, scene_dir={self.scene_dir}"
            )
        pose_indices = [
            reference_name_to_pose_index[image_name] for image_name in input_image_names
        ]
        return torch.tensor(pose_indices, dtype=torch.long)

    def _is_valid_colmap_sparse_dir(self, candidate_sparse_dir: Path) -> bool:
        required_binary_assets = ("cameras.bin", "images.bin", "points3D.bin")
        return all((candidate_sparse_dir / asset_name).exists() for asset_name in required_binary_assets)

    def _score_colmap_sparse_dir(self, candidate_sparse_dir: Path) -> int:
        reference_image_names = {image_file.name.lower() for image_file in self.sparse_image_files}
        target_image_names = {image_file.name.lower() for image_file in self.image_files}
        candidate_root_dir = candidate_sparse_dir.parent.parent
        candidate_image_names: Set[str] = set()
        for candidate_image_dir in (
            candidate_root_dir / "images",
            candidate_root_dir / "input",
            candidate_root_dir,
        ):
            candidate_image_files = self._collect_image_files_from_dir(candidate_image_dir)
            if candidate_image_files:
                candidate_image_names = {
                    candidate_image_file.name.lower() for candidate_image_file in candidate_image_files
                }
                break

        reference_overlap_count = len(reference_image_names & candidate_image_names)
        target_overlap_count = len(target_image_names & candidate_image_names)
        score = reference_overlap_count * 100 + target_overlap_count * 10

        candidate_root_name = candidate_root_dir.name.lower()
        preferred_root_names = {
            self.input_data_dir.name.lower(),
            self.images_dir.name.lower(),
            self.sparse_images_dir.name.lower(),
        }
        if candidate_root_name in preferred_root_names:
            score += 3
        if candidate_root_dir == self.sparse_images_dir:
            score += 5
        if candidate_root_dir == self.images_dir:
            score += 2
        return score

    def _find_colmap_sparse_dir(self) -> Optional[Path]:
        candidate_sparse_dirs: List[Path] = []
        candidate_root_dirs = [
            self.scene_dir,
            self.input_data_dir,
            self.images_dir,
            self.sparse_images_dir,
        ]
        llff_nested_root_names = [
            "images",
            "images_8",
            "images_4",
            "images_2",
            "input",
            "test1",
            "2_views",
            "3_views",
        ]

        for candidate_root_dir in candidate_root_dirs:
            candidate_sparse_dirs.extend(
                [
                    candidate_root_dir / "sparse" / "0",
                    candidate_root_dir / "distorted" / "sparse" / "0",
                ]
            )
        for nested_root_name in llff_nested_root_names:
            candidate_sparse_dirs.extend(
                [
                    self.scene_dir / nested_root_name / "sparse" / "0",
                    self.scene_dir / nested_root_name / "distorted" / "sparse" / "0",
                ]
            )

        unique_candidate_sparse_dirs: List[Path] = []
        seen_candidate_sparse_dirs: Set[Path] = set()
        for candidate_sparse_dir in candidate_sparse_dirs:
            resolved_candidate_sparse_dir = candidate_sparse_dir.resolve()
            if resolved_candidate_sparse_dir in seen_candidate_sparse_dirs:
                continue
            seen_candidate_sparse_dirs.add(resolved_candidate_sparse_dir)
            unique_candidate_sparse_dirs.append(resolved_candidate_sparse_dir)

        self._last_colmap_sparse_dir_candidates = unique_candidate_sparse_dirs
        valid_sparse_dirs = [
            candidate_sparse_dir
            for candidate_sparse_dir in unique_candidate_sparse_dirs
            if self._is_valid_colmap_sparse_dir(candidate_sparse_dir)
        ]
        if not valid_sparse_dirs:
            return None
        return max(valid_sparse_dirs, key=self._score_colmap_sparse_dir)

    def _load_sfm_depth_alignment_assets(self) -> Optional[Dict[str, Any]]:
        sparse_reconstruction_dir = self._find_colmap_sparse_dir()
        if sparse_reconstruction_dir is None:
            print("[LLFFDataset] 未找到 COLMAP sparse/0 资产，深度对齐将回退到 poses_bounds 线性拉伸。")
            return None

        from examples.active_sampling.train_active_sampling import (
            camera_intrinsics_from_params,
            qvec_to_rotmat,
            read_cameras_binary,
            read_images_binary,
            read_points3d_binary,
        )

        try:
            colmap_cameras = read_cameras_binary(sparse_reconstruction_dir / "cameras.bin")
            colmap_images = read_images_binary(sparse_reconstruction_dir / "images.bin")
            colmap_points_world, _ = read_points3d_binary(sparse_reconstruction_dir / "points3D.bin")
        except Exception as exc:
            print(
                f"[LLFFDataset] 读取 COLMAP sparse 资产失败，深度对齐回退为 poses_bounds 线性拉伸: {exc}"
            )
            return None

        image_entries_by_name: Dict[str, Dict[str, Any]] = {}
        image_entries_by_stem: Dict[str, Dict[str, Any]] = {}
        for image_entry in colmap_images:
            entry_name = Path(str(image_entry["name"])).name
            image_entries_by_name[entry_name.lower()] = image_entry
            image_entries_by_stem[Path(entry_name).stem.lower()] = image_entry

        print(
            f"[LLFFDataset] 已加载 COLMAP sparse 深度对齐资产: "
            f"dir={sparse_reconstruction_dir}, images={len(colmap_images)}, points={len(colmap_points_world)}"
        )
        return {
            "sparse_reconstruction_dir": sparse_reconstruction_dir,
            "cameras": colmap_cameras,
            "image_entries_by_name": image_entries_by_name,
            "image_entries_by_stem": image_entries_by_stem,
            "points_world": torch.from_numpy(colmap_points_world.astype(np.float32)),
            "camera_intrinsics_from_params": camera_intrinsics_from_params,
            "qvec_to_rotmat": qvec_to_rotmat,
        }

    def _resolve_colmap_image_entry(
        self,
        image_file: Path,
    ) -> Optional[Dict[str, Any]]:
        if self._sfm_depth_alignment_assets is None:
            return None
        image_name_key = image_file.name.lower()
        image_entry = self._sfm_depth_alignment_assets["image_entries_by_name"].get(image_name_key)
        if image_entry is not None:
            return image_entry
        image_stem_key = image_file.stem.lower()
        return self._sfm_depth_alignment_assets["image_entries_by_stem"].get(image_stem_key)

    def _fit_depth_scale_shift_with_ransac(
        self,
        relative_depth_samples: torch.Tensor,
        metric_depth_samples: torch.Tensor,
        *,
        max_iterations: int = 96,
    ) -> Optional[Tuple[float, float, int]]:
        relative_depth_samples = relative_depth_samples.detach().cpu().float().flatten()
        metric_depth_samples = metric_depth_samples.detach().cpu().float().flatten()
        valid_mask = (
            torch.isfinite(relative_depth_samples)
            & torch.isfinite(metric_depth_samples)
            & (metric_depth_samples > 1e-4)
        )
        relative_depth_samples = relative_depth_samples[valid_mask]
        metric_depth_samples = metric_depth_samples[valid_mask]
        if relative_depth_samples.numel() < 8:
            return None

        if relative_depth_samples.numel() > 4096:
            sample_indices = torch.linspace(
                0,
                relative_depth_samples.numel() - 1,
                steps=4096,
            ).long()
            relative_depth_samples = relative_depth_samples.index_select(0, sample_indices)
            metric_depth_samples = metric_depth_samples.index_select(0, sample_indices)

        residual_threshold = max(
            float(metric_depth_samples.median().item()) * 0.05,
            0.05,
        )
        random_generator = torch.Generator(device="cpu")
        random_generator.manual_seed(0)
        best_inlier_mask: Optional[torch.Tensor] = None
        best_inlier_count = 0
        best_residual = float("inf")
        total_samples = relative_depth_samples.numel()
        num_iterations = min(max_iterations, max(total_samples * 2, 16))

        for _ in range(num_iterations):
            sample_indices = torch.randperm(total_samples, generator=random_generator)[:2]
            sampled_relative = relative_depth_samples.index_select(0, sample_indices)
            sampled_metric = metric_depth_samples.index_select(0, sample_indices)
            denominator = float((sampled_relative[1] - sampled_relative[0]).item())
            if abs(denominator) < 1e-6:
                continue
            scale_value = float((sampled_metric[1] - sampled_metric[0]).item() / denominator)
            if not np.isfinite(scale_value) or scale_value <= 0.0:
                continue
            shift_value = float(sampled_metric[0].item() - scale_value * sampled_relative[0].item())
            predicted_metric_depth = scale_value * relative_depth_samples + shift_value
            residual = torch.abs(predicted_metric_depth - metric_depth_samples)
            inlier_mask = residual <= residual_threshold
            inlier_count = int(inlier_mask.sum().item())
            if inlier_count < 4:
                continue
            mean_residual = float(residual[inlier_mask].mean().item())
            if (
                inlier_count > best_inlier_count
                or (inlier_count == best_inlier_count and mean_residual < best_residual)
            ):
                best_inlier_mask = inlier_mask
                best_inlier_count = inlier_count
                best_residual = mean_residual

        if best_inlier_mask is None or best_inlier_count < 4:
            return None

        inlier_relative = relative_depth_samples[best_inlier_mask]
        inlier_metric = metric_depth_samples[best_inlier_mask]
        centered_relative = inlier_relative - inlier_relative.mean()
        centered_metric = inlier_metric - inlier_metric.mean()
        relative_variance = float((centered_relative * centered_relative).mean().item())
        if relative_variance < 1e-8:
            return None
        refined_scale = float((centered_relative * centered_metric).mean().item() / relative_variance)
        if not np.isfinite(refined_scale) or refined_scale <= 0.0:
            return None
        refined_shift = float(inlier_metric.mean().item() - refined_scale * inlier_relative.mean().item())
        return refined_scale, refined_shift, best_inlier_count

    def _align_depth_with_sfm_points(
        self,
        normalized_relative_depth_map: torch.Tensor,
        *,
        image_file: Path,
    ) -> Optional[torch.Tensor]:
        if self._sfm_depth_alignment_assets is None:
            return None

        resolved_image_key = str(image_file.resolve())
        alignment_attempted = True
        alignment_succeeded = False
        alignment_inlier_count = 0
        aligned_depth_map: Optional[torch.Tensor] = None

        colmap_image_entry = self._resolve_colmap_image_entry(image_file)
        if colmap_image_entry is None:
            self._record_sfm_depth_alignment_statistics(
                image_key=resolved_image_key,
                image_name=image_file.name,
                success=False,
                inlier_count=0,
            )
            return None

        colmap_cameras = self._sfm_depth_alignment_assets["cameras"]
        camera_intrinsics_from_params = self._sfm_depth_alignment_assets["camera_intrinsics_from_params"]
        qvec_to_rotmat = self._sfm_depth_alignment_assets["qvec_to_rotmat"]
        points_world = self._sfm_depth_alignment_assets["points_world"]

        camera_id = int(colmap_image_entry["camera_id"])
        if camera_id in colmap_cameras:
            camera_entry = colmap_cameras[camera_id]
            source_width = float(camera_entry["width"])
            source_height = float(camera_entry["height"])
            if source_width > 1.0 and source_height > 1.0:
                target_height, target_width = self.image_size
                scale_x = float(target_width) / source_width
                scale_y = float(target_height) / source_height
                camera_intrinsics = torch.from_numpy(
                    camera_intrinsics_from_params(
                        str(camera_entry["model"]),
                        np.asarray(camera_entry["params"]),
                        scale_x,
                        scale_y,
                    )
                ).float()

                rotation_matrix = torch.from_numpy(
                    qvec_to_rotmat(np.asarray(colmap_image_entry["qvec"]))
                ).float()
                translation_vector = torch.from_numpy(
                    np.asarray(colmap_image_entry["tvec"], dtype=np.float32)
                ).view(3, 1)

                points_camera = (rotation_matrix @ points_world.t() + translation_vector).t()
                metric_depth_values = points_camera[:, 2]
                positive_depth_mask = metric_depth_values > 1e-4
                if int(positive_depth_mask.sum().item()) >= 8:
                    points_camera = points_camera[positive_depth_mask]
                    metric_depth_values = metric_depth_values[positive_depth_mask]

                    projected_points = (camera_intrinsics @ points_camera.t()).t()
                    projected_depth = projected_points[:, 2].clamp(min=1e-6)
                    pixel_u = projected_points[:, 0] / projected_depth
                    pixel_v = projected_points[:, 1] / projected_depth

                    valid_pixel_mask = (
                        (pixel_u >= 0.0)
                        & (pixel_u <= float(target_width - 1))
                        & (pixel_v >= 0.0)
                        & (pixel_v <= float(target_height - 1))
                    )
                    if int(valid_pixel_mask.sum().item()) >= 8:
                        pixel_u = pixel_u[valid_pixel_mask].round().long().clamp(0, target_width - 1)
                        pixel_v = pixel_v[valid_pixel_mask].round().long().clamp(0, target_height - 1)
                        metric_depth_values = metric_depth_values[valid_pixel_mask]
                        relative_depth_samples = normalized_relative_depth_map[0, pixel_v, pixel_u]
                        fitted_parameters = self._fit_depth_scale_shift_with_ransac(
                            relative_depth_samples,
                            metric_depth_values,
                        )
                        if fitted_parameters is not None:
                            depth_scale, depth_shift, inlier_count = fitted_parameters
                            if inlier_count >= 4:
                                alignment_succeeded = True
                                alignment_inlier_count = int(inlier_count)
                                aligned_depth_map = depth_scale * normalized_relative_depth_map + depth_shift
                                aligned_depth_map = aligned_depth_map.clamp(min=1e-3)
                                if not self._sfm_depth_alignment_logged:
                                    print(
                                        "[LLFFDataset] SfM metric depth alignment enabled: "
                                        f"image={image_file.name}, depth_alignment_inliers={inlier_count}, "
                                        f"depth_scale={depth_scale:.4f}, depth_shift={depth_shift:.4f}"
                                    )
                                    self._sfm_depth_alignment_logged = True

        if alignment_attempted:
            self._record_sfm_depth_alignment_statistics(
                image_key=resolved_image_key,
                image_name=image_file.name,
                success=alignment_succeeded,
                inlier_count=alignment_inlier_count,
            )
        return aligned_depth_map

    def _record_sfm_depth_alignment_statistics(
        self,
        *,
        image_key: str,
        image_name: str,
        success: bool,
        inlier_count: int,
    ) -> None:
        if image_key in self._sfm_depth_alignment_attempted_images:
            return
        self._sfm_depth_alignment_attempted_images.add(image_key)
        if success:
            self._sfm_depth_alignment_successful_images.add(image_key)
            self._sfm_depth_alignment_inlier_sum += int(inlier_count)
        success_count = len(self._sfm_depth_alignment_successful_images)
        attempt_count = len(self._sfm_depth_alignment_attempted_images)
        success_ratio = float(success_count / max(attempt_count, 1))
        mean_inliers = float(self._sfm_depth_alignment_inlier_sum / max(success_count, 1))
        print(
            "[LLFFDataset] SfM alignment stats: "
            f"image={image_name}, success={'yes' if success else 'no'}, "
            f"sfm_alignment_success_count={success_count}, "
            f"sfm_alignment_success_ratio={success_ratio:.4f}, "
            f"mean_inliers={mean_inliers:.2f}"
        )
    
    def _build_resized_camera_intrinsics(self, pose_hwf: torch.Tensor) -> torch.Tensor:
        """将LLFF的 H/W/focal 缩放到当前训练图像分辨率。"""
        target_height, target_width = self.image_size
        pose_hwf = pose_hwf.to(dtype=torch.float32)
        source_height = pose_hwf[:, 0].clamp_min(1.0)
        source_width = pose_hwf[:, 1].clamp_min(1.0)
        source_focal = pose_hwf[:, 2].clamp_min(1.0)

        focal_x = source_focal * (float(target_width) / source_width)
        focal_y = source_focal * (float(target_height) / source_height)
        principal_x = torch.full_like(focal_x, float(target_width) / 2.0)
        principal_y = torch.full_like(focal_y, float(target_height) / 2.0)

        intrinsics = torch.zeros((pose_hwf.shape[0], 3, 3), dtype=torch.float32)
        intrinsics[:, 0, 0] = focal_x
        intrinsics[:, 1, 1] = focal_y
        intrinsics[:, 0, 2] = principal_x
        intrinsics[:, 1, 2] = principal_y
        intrinsics[:, 2, 2] = 1.0
        return intrinsics

    def _resolve_depth_model_device(self) -> torch.device:
        """为数据集内的深度模型选择设备。

        数据集 __getitem__ 在主进程运行，若把大号 DA-V2 常驻 cuda:0，
        会与训练主干共享同一 CUDA 上下文并争抢显存。默认放到 CPU，
        如需显式指定可通过 GDDN_DATASET_DEPTH_DEVICE 覆盖。
        """
        requested_device = os.environ.get("GDDN_DATASET_DEPTH_DEVICE")
        if requested_device:
            try:
                return torch.device(requested_device)
            except Exception:
                pass
        return torch.device("cpu")

    @classmethod
    def _resolve_shared_image_tensor_cache_limit(cls) -> int:
        if cls._shared_image_tensor_cache_limit is None:
            raw_limit = os.environ.get("GDDN_DATASET_IMAGE_CACHE_LIMIT", "").strip()
            resolved_limit = 1024
            if raw_limit:
                try:
                    resolved_limit = max(0, int(raw_limit))
                except ValueError:
                    resolved_limit = 1024
            cls._shared_image_tensor_cache_limit = resolved_limit
        return int(cls._shared_image_tensor_cache_limit)

    def _build_image_tensor_cache_key(self, path: Path) -> str:
        resolved_path = str(path.expanduser().resolve())
        return f"{resolved_path}|{int(self.image_size[0])}x{int(self.image_size[1])}"

    @classmethod
    def _store_shared_image_tensor(cls, cache_key: str, image_tensor: torch.Tensor) -> None:
        cls._shared_image_tensor_cache[cache_key] = image_tensor
        cls._shared_image_tensor_cache.move_to_end(cache_key)
        cache_limit = cls._resolve_shared_image_tensor_cache_limit()
        if cache_limit <= 0:
            return
        while len(cls._shared_image_tensor_cache) > cache_limit:
            cls._shared_image_tensor_cache.popitem(last=False)

    def _load_image_tensor_from_disk(self, path: Path) -> torch.Tensor:
        from PIL import Image
        import torchvision.transforms.functional as TF

        image = Image.open(path).convert("RGB")
        image = TF.resize(image, self.image_size)
        image_tensor = TF.to_tensor(image).detach().cpu().float()
        return image_tensor

    def _get_or_load_cached_image_tensor(self, path: Path) -> torch.Tensor:
        cache_key = self._build_image_tensor_cache_key(path)
        cached_tensor = self._cached_image_tensors_by_file_path.get(cache_key)
        if cached_tensor is not None:
            return cached_tensor
        shared_cached_tensor = LLFFDataset._shared_image_tensor_cache.get(cache_key)
        if shared_cached_tensor is not None:
            LLFFDataset._shared_image_tensor_cache.move_to_end(cache_key)
            self._cached_image_tensors_by_file_path[cache_key] = shared_cached_tensor
            return shared_cached_tensor
        loaded_tensor = self._load_image_tensor_from_disk(path)
        self._cached_image_tensors_by_file_path[cache_key] = loaded_tensor
        LLFFDataset._store_shared_image_tensor(cache_key, loaded_tensor)
        return loaded_tensor

    def _warmup_scene_image_tensor_cache(self) -> None:
        unique_image_files: List[Path] = []
        seen_cache_keys: Set[str] = set()
        for image_file in list(self.image_files) + list(self.sparse_image_files):
            cache_key = self._build_image_tensor_cache_key(image_file)
            if cache_key in seen_cache_keys:
                continue
            seen_cache_keys.add(cache_key)
            unique_image_files.append(image_file)
        for image_file in unique_image_files:
            self._get_or_load_cached_image_tensor(image_file)
        if self.verbose_dataset_logs:
            print(
                "[LLFFDataset] 预热 RGB 图像缓存完成: "
                f"scene={self.scene_dir.name}, cached_images={len(unique_image_files)}"
            )

    def _load_poses(self) -> tuple:
        """加载LLFF格式的位姿文件"""
        poses_file = self.scene_dir / "poses_bounds.npy"
        
        if poses_file.exists():
            poses_bounds = np.load(str(poses_file))
            poses = poses_bounds[:, :-2].reshape(-1, 3, 5)
            bounds = poses_bounds[:, -2:]
            
            # 转换为4x4矩阵
            poses_4x4 = []
            pose_hwf = []
            for pose_3x5 in poses:
                # LLFF 原始 poses_bounds.npy 的旋转列语义是 [down, right, back]。
                # 本项目 warp / reprojection / pinhole 投影链统一假设 OpenCV 相机坐标：
                #   x 向右, y 向下, z 向前(正深度)。
                # 因此这里显式转换为 [right, down, forward]，再取逆得到 world->cam。
                canonical_pose_3x5 = np.concatenate(
                    [
                        pose_3x5[:, 1:2],
                        pose_3x5[:, 0:1],
                        -pose_3x5[:, 2:3],
                        pose_3x5[:, 3:5],
                    ],
                    axis=1,
                )
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :4] = canonical_pose_3x5[:, :4]
                pose_hwf.append(pose_3x5[:, 4].copy())
                # 项目其余几何链（warp / multiview consistency / GCD）统一按 world->cam 使用位姿，
                # 而 LLFF/NeRF 标准 poses_bounds.npy 存的是 cam->world，因此这里显式取逆。
                pose_4x4 = np.linalg.inv(pose_4x4)
                poses_4x4.append(pose_4x4)
            
            poses = np.stack(poses_4x4, axis=0)
            pose_hwf = np.stack(pose_hwf, axis=0)
            if self.verbose_dataset_logs:
                print(f"[LLFFDataset] Loaded {len(poses)} poses from {poses_file}")
            return (
                torch.tensor(poses, dtype=torch.float32),
                torch.tensor(bounds, dtype=torch.float32),
                torch.tensor(pose_hwf, dtype=torch.float32),
            )
        else:
            if self.strict_geometry_contract:
                raise FileNotFoundError(
                    f"[LLFFDataset] strict geometry contract enabled, but poses_bounds.npy was not found. "
                    f"input_data_dir={self.input_data_dir}, resolved_scene_dir={self.scene_dir}"
                )
            # 如果没有位姿文件，生成带扰动的identity矩阵避免数值问题
            print(f"[LLFFDataset] Warning: No poses_bounds.npy found, using perturbed identity poses")
            n_images = len(self.image_files)
            poses = torch.eye(4).unsqueeze(0).expand(n_images, 4, 4).clone()
            # 添加小的平移扰动避免完全相同的pose导致数值问题
            for i in range(n_images):
                poses[i, :3, 3] = torch.randn(3) * 0.1
            bounds = torch.tensor([[0.1, 10.0]]).expand(n_images, 2).clone()
            fallback_hwf = torch.tensor(
                [[float(self.image_size[0]), float(self.image_size[1]), float(max(self.image_size))]],
                dtype=torch.float32,
            ).expand(n_images, 3).clone()
            return poses, bounds, fallback_hwf

    
    def _load_image(self, path: Path) -> torch.Tensor:
        """加载并预处理图像"""
        return self._get_or_load_cached_image_tensor(path).clone()
    
    def _init_depth_model(self):
        """初始化Depth-Anything-V2模型"""
        try:
            import sys
            import os
            
            # 禁用xformers，使用标准PyTorch注意力
            # xformers在新GPU(capability>=12.0)上不支持float32
            os.environ["XFORMERS_DISABLED"] = "1"
            
            # 添加Depth-Anything-V2路径
            da2_path = Path(__file__).parent.parent.parent / "models" / "Depth-Anything-V2"
            if str(da2_path) not in sys.path:
                sys.path.insert(0, str(da2_path))
            
            DepthAnythingV2 = _get_depth_anything_v2_class()
            if DepthAnythingV2 is None:
                print("[LLFFDataset] WARNING: DA-V2 module unavailable, using simple depth estimation")
                return
            
            model_spec = _find_da_model_spec(
                preferred_model_types=("vitg", "vitl", "vitb", "vits")
            )
            if model_spec is None:
                print("[LLFFDataset] WARNING: DA-V2 checkpoint not found, using simple depth estimation")
                return
            checkpoint, resolved_depth_model_type = model_spec
            resolved_checkpoint_path = str(Path(checkpoint).expanduser().resolve())
            resolved_depth_model_device = str(self.depth_model_device)
            if (
                LLFFDataset._shared_depth_model is not None
                and LLFFDataset._shared_depth_model_checkpoint == resolved_checkpoint_path
                and LLFFDataset._shared_depth_model_device == resolved_depth_model_device
                and LLFFDataset._shared_depth_model_type == resolved_depth_model_type
            ):
                self.depth_model = LLFFDataset._shared_depth_model
                print(
                    f"[LLFFDataset] Reusing shared Depth-Anything-V2 ({resolved_depth_model_type}) "
                    f"on {self.depth_model_device}"
                )
                return
            
            depth_model_config = _DA_MODEL_CONFIGS.get(
                resolved_depth_model_type,
                _DA_MODEL_CONFIGS["vitb"],
            )
            self.depth_model = DepthAnythingV2(**depth_model_config)
            self.depth_model.load_state_dict(torch.load(str(checkpoint), map_location='cpu'))
            self.depth_model.eval()
            
            self.depth_model = self.depth_model.to(
                device=self.depth_model_device,
                dtype=torch.float32,
            )
            LLFFDataset._shared_depth_model = self.depth_model
            LLFFDataset._shared_depth_model_checkpoint = resolved_checkpoint_path
            LLFFDataset._shared_depth_model_device = resolved_depth_model_device
            LLFFDataset._shared_depth_model_type = resolved_depth_model_type
            
            print(
                f"[LLFFDataset] Loaded Depth-Anything-V2 ({resolved_depth_model_type}) from {checkpoint} "
                f"on {self.depth_model_device}"
            )
            
        except Exception as e:
            print(f"[LLFFDataset] WARNING: Failed to load DA-V2: {e}, using simple depth estimation")
            self.depth_model = None
    
    def _estimate_depth_da2(self, image: torch.Tensor) -> torch.Tensor:
        """使用Depth-Anything-V2估计深度
        
        Args:
            image: [3, H, W] RGB图像, 范围[0,1]
            
        Returns:
            [3, H, W] 深度图（复制到3通道以匹配ControlNet输入）
        """
        if self.depth_model is None:
            # Fallback到简单估计并复制到3通道
            depth_1ch = self._estimate_depth_simple(image)
            return depth_1ch.repeat(3, 1, 1)  # [3, H, W]
        
        with torch.no_grad():
            # DA-V2期望 [B, 3, H, W] 输入
            # 注意：DA-V2的patch size是14，输入尺寸必须是14的倍数
            # 512不是14的倍数，需要resize到518 (= 14 * 37)
            H, W = image.shape[-2:]
            target_H = ((H + 13) // 14) * 14  # 向上取整到14的倍数
            target_W = ((W + 13) // 14) * 14
            
            img_resized = F.interpolate(
                image.unsqueeze(0), 
                size=(target_H, target_W), 
                mode='bilinear', 
                align_corners=False
            )  # [1, 3, target_H, target_W]
            
            # 将输入移到与模型相同的设备
            depth_model_device = next(self.depth_model.parameters()).device
            img_resized = img_resized.to(depth_model_device, dtype=torch.float32)
            
            # 推理（已禁用xformers，可使用float32）
            depth = self.depth_model(img_resized)  # [1, H, W] 或 [H, W]
            
            if depth.dim() == 2:
                depth = depth.unsqueeze(0)  # [1, H', W']
            
            # 将深度图resize回原尺寸
            if depth.shape[-2:] != (H, W):
                depth = F.interpolate(
                    depth.unsqueeze(0), 
                    size=(H, W), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)  # [1, H, W]
            
            # 归一化到[0, 1]
            depth = depth.squeeze(0)  # [H, W]
            depth_min, depth_max = depth.min(), depth.max()
            if depth_max - depth_min > 1e-6:
                depth = (depth - depth_min) / (depth_max - depth_min)
            
            # 复制到3通道
            depth_3ch = depth.unsqueeze(0).repeat(3, 1, 1).cpu()  # [3, H, W]
            
        return depth_3ch

    def _get_cached_depth_map_for_image_file(
        self,
        image_file: Path,
        *,
        pose_index: int,
        cache_key: Optional[str] = None,
    ) -> torch.Tensor:
        """按图像文件缓存原始图像深度，避免数据加载阶段重复运行 DA-V2。"""
        resolved_cache_key = cache_key or str(image_file.resolve())
        resolved_image_path_key = str(image_file.resolve())
        shared_depth_cache_key = self._build_depth_cache_key(
            image_file=image_file,
            pose_index=int(pose_index),
        )
        cached_depth_map = self._cached_depth_maps_by_image_index.get(resolved_cache_key)
        if cached_depth_map is not None:
            return cached_depth_map.clone()
        cached_depth_map = self._cached_depth_maps_by_file_path.get(resolved_image_path_key)
        if cached_depth_map is not None:
            self._cached_depth_maps_by_image_index[resolved_cache_key] = cached_depth_map
            return cached_depth_map.clone()
        cached_depth_map = LLFFDataset._shared_aligned_depth_map_cache.get(shared_depth_cache_key)
        if cached_depth_map is not None:
            self._cached_depth_maps_by_image_index[resolved_cache_key] = cached_depth_map
            self._cached_depth_maps_by_file_path[resolved_image_path_key] = cached_depth_map
            return cached_depth_map.clone()

        persistent_depth_cache_file = self._resolve_persistent_depth_cache_file(
            image_file=image_file,
            pose_index=int(pose_index),
        )
        if persistent_depth_cache_file is not None and persistent_depth_cache_file.exists():
            cached_depth_map = torch.load(persistent_depth_cache_file, map_location="cpu")
            if torch.is_tensor(cached_depth_map):
                cached_depth_map = cached_depth_map.detach().cpu().float()
                self._cached_depth_maps_by_image_index[resolved_cache_key] = cached_depth_map
                self._cached_depth_maps_by_file_path[resolved_image_path_key] = cached_depth_map
                LLFFDataset._shared_aligned_depth_map_cache[shared_depth_cache_key] = cached_depth_map
                return cached_depth_map.clone()

        image_tensor = self._load_image(image_file)
        relative_depth_map = self._estimate_depth_da2(image_tensor)[0:1].cpu()
        cached_depth_map = self._align_depth_map_to_scene_bounds(
            relative_depth_map,
            pose_index=pose_index,
            image_file=image_file,
        ).cpu()
        self._cached_depth_maps_by_image_index[resolved_cache_key] = cached_depth_map
        self._cached_depth_maps_by_file_path[resolved_image_path_key] = cached_depth_map
        LLFFDataset._shared_aligned_depth_map_cache[shared_depth_cache_key] = cached_depth_map
        if persistent_depth_cache_file is not None:
            torch.save(cached_depth_map, persistent_depth_cache_file)
        return cached_depth_map.clone()

    def _get_cached_depth_map_for_image_index(self, image_index: int) -> torch.Tensor:
        """缓存 target 图像池深度，避免数据加载阶段重复运行 DA-V2。"""
        image_index = int(image_index)
        image_file = self.image_files[image_index]
        pose_index = int(self.pose_indices_for_images[image_index].item())
        return self._get_cached_depth_map_for_image_file(
            image_file,
            pose_index=pose_index,
            cache_key=f"target::{image_index}",
        )

    def _align_depth_map_to_scene_bounds(
        self,
        depth_map: torch.Tensor,
        *,
        pose_index: int,
        image_file: Optional[Path] = None,
    ) -> torch.Tensor:
        """将相对深度映射到当前视角对应的 LLFF scene bounds 尺度。

        DA-V2 / 亮度 fallback 当前都只提供逐图相对深度，若直接与 pose 平移量
        混用，会导致投影尺度严重失配。这里使用该视角的 near/far bounds 将
        相对深度拉回场景尺度，确保 warp 使用的 z 深度与相机位姿处于同一量纲。
        """
        aligned_depth_map = depth_map.to(dtype=torch.float32)
        depth_min_value = aligned_depth_map.amin()
        depth_max_value = aligned_depth_map.amax()
        if float(depth_max_value - depth_min_value) > 1e-6:
            aligned_depth_map = (
                aligned_depth_map - depth_min_value
            ) / (depth_max_value - depth_min_value)
        else:
            aligned_depth_map = torch.zeros_like(aligned_depth_map)

        if image_file is not None:
            sfm_aligned_depth_map = self._align_depth_with_sfm_points(
                aligned_depth_map,
                image_file=image_file,
            )
            if sfm_aligned_depth_map is not None:
                return sfm_aligned_depth_map

        if (
            self.bounds is None
            or pose_index < 0
            or pose_index >= int(self.bounds.shape[0])
        ):
            return aligned_depth_map + 1.0

        pose_bounds = self.bounds[pose_index].to(dtype=torch.float32)
        near_bound = float(torch.min(pose_bounds).item())
        far_bound = float(torch.max(pose_bounds).item())
        near_bound = max(near_bound, 1e-3)
        far_bound = max(far_bound, near_bound + 1e-3)
        return near_bound + aligned_depth_map * (far_bound - near_bound)
    
    def _depth_to_normal(self, depth: torch.Tensor) -> torch.Tensor:
        """从深度图估计法线（使用Sobel算子）
        
        Args:
            depth: [3, H, W] 深度图（3通道相同）
            
        Returns:
            [3, H, W] 法线图，RGB编码
        """
        # 使用第一个通道
        depth_1ch = depth[0:1]  # [1, H, W]
        
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0
        
        # 计算梯度
        depth_batch = depth_1ch.unsqueeze(0)  # [1, 1, H, W]
        dx = F.conv2d(depth_batch, sobel_x, padding=1).squeeze(0)  # [1, H, W]
        dy = F.conv2d(depth_batch, sobel_y, padding=1).squeeze(0)  # [1, H, W]
        
        # 构建法线向量 (nx, ny, nz)
        # 假设z方向为常数1
        ones = torch.ones_like(dx)
        normal = torch.cat([-dx, -dy, ones], dim=0)  # [3, H, W]
        
        # 归一化
        norm = torch.norm(normal, dim=0, keepdim=True).clamp(min=1e-6)
        normal = normal / norm
        
        # 从[-1,1]映射到[0,1]用于可视化
        normal = (normal + 1.0) / 2.0
        
        return normal
    
    def _estimate_depth_simple(self, image: torch.Tensor) -> torch.Tensor:
        """基于亮度的简化深度估计（CPU端，无需深度模型）
        
        Args:
            image: [3, H, W] RGB图像, 范围[0,1]
            
        Returns:
            [1, H, W] 伪深度图, 范围[0,1]
        """
        # 使用亮度作为伪深度（较暗区域假设更远）
        depth = image.mean(dim=0, keepdim=True)  # [1, H, W]
        # 归一化到[0,1]
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max - depth_min > 1e-6:
            depth = (depth - depth_min) / (depth_max - depth_min)
        # 反转：亮度高的区域深度浅（更近）
        depth = 1.0 - depth
        # 缩放到合理深度范围 [1, 10]
        depth = depth * 9.0 + 1.0
        return depth
    
    def _perspective_warp_simple(
        self,
        image: torch.Tensor,       # [3, H, W]
        depth: torch.Tensor,       # [1, H, W]，目标视角深度
        ref_pose: torch.Tensor,    # [4, 4]
        target_pose: torch.Tensor, # [4, 4]
        ref_intrinsics: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        reference_depth_map: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """简化的目标驱动 backward warp（纯PyTorch，无需CUDA）。
        
        Args:
            image: 参考图像 [3, H, W]
            depth: 目标视角深度图 [1, H, W]
            ref_pose: 参考位姿 [4, 4]
            target_pose: 目标位姿 [4, 4]
            
        Returns:
            warped: 变换后图像 [3, H, W]
            valid_mask: 有效像素掩码 [1, H, W]
        """
        C, H, W = image.shape
        dtype = image.dtype
        warp_diagnostics: Dict[str, torch.Tensor] = {}
        
        # 默认内参
        if ref_intrinsics is None:
            fx = fy = max(H, W)
            cx, cy = W / 2.0, H / 2.0
            ref_intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=dtype)
        else:
            ref_intrinsics = ref_intrinsics.to(dtype=dtype)
        if target_intrinsics is None:
            target_intrinsics = ref_intrinsics
        else:
            target_intrinsics = target_intrinsics.to(dtype=dtype)
        
        # 创建目标视角像素网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        ones = torch.ones_like(x_coords)
        pixels = torch.stack([x_coords, y_coords, ones], dim=-1)  # [H, W, 3]
        pixels_flat = pixels.reshape(-1, 3)  # [HW, 3]
        
        # 目标像素反投影到目标相机坐标系
        try:
            target_intrinsics_inverse = torch.linalg.pinv(target_intrinsics.float())
        except:
            target_intrinsics_inverse = torch.inverse(target_intrinsics.float())
        target_rays = pixels_flat @ target_intrinsics_inverse.T  # [HW, 3]
        
        depth_flat = depth.reshape(-1).float()
        depth_flat = torch.nan_to_num(depth_flat, nan=0.0, posinf=0.0, neginf=0.0)
        depth_valid_mask = depth_flat > 1e-4
        depth_flat = depth_flat.clamp(min=1e-4)
        target_points_3d = target_rays * depth_flat.unsqueeze(-1)  # [HW, 3]
        target_points_homogeneous = torch.cat(
            [
                target_points_3d,
                torch.ones(target_points_3d.shape[0], 1, dtype=torch.float32),
            ],
            dim=-1,
        )
        
        # 目标相机 -> 世界 -> 参考相机
        ref_pose_f32 = ref_pose.float()
        target_pose_f32 = target_pose.float()
        try:
            target_to_world = torch.linalg.pinv(target_pose_f32)
        except:
            target_to_world = torch.inverse(target_pose_f32)
        world_points = target_points_homogeneous @ target_to_world.T
        reference_points = world_points @ ref_pose_f32.T
        reference_points = reference_points[:, :3]
        
        # 投影到参考图像平面，构造 target pixel -> source pixel 的采样网格
        points_2d_homo = reference_points @ ref_intrinsics.float().T  # [HW, 3]
        z_values = points_2d_homo[:, 2:].clamp(min=1e-4)
        points_2d = points_2d_homo[:, :2] / z_values  # [HW, 2]
        
        # 归一化到[-1, 1]用于grid_sample
        grid_x = 2.0 * points_2d[:, 0] / max(W - 1, 1) - 1.0
        grid_y = 2.0 * points_2d[:, 1] / max(H - 1, 1) - 1.0
        grid_x = torch.clamp(grid_x, min=-10.0, max=10.0)
        grid_y = torch.clamp(grid_y, min=-10.0, max=10.0)
        grid_x = torch.nan_to_num(grid_x, nan=0.0)
        grid_y = torch.nan_to_num(grid_y, nan=0.0)
        
        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(H, W, 2)  # [H, W, 2]
        
        # 采样
        image_batch = image.unsqueeze(0)  # [1, 3, H, W]
        grid_batch = grid.unsqueeze(0)    # [1, H, W, 2]
        warped = F.grid_sample(
            image_batch, 
            grid_batch.to(dtype), 
            mode='bilinear', 
            padding_mode='zeros', 
            align_corners=True
        )[0]  # [3, H, W]
        
        # 计算有效像素掩码
        valid_x = (grid[..., 0] >= -1) & (grid[..., 0] <= 1)
        valid_y = (grid[..., 1] >= -1) & (grid[..., 1] <= 1)
        valid_z = points_2d_homo[:, 2].reshape(H, W) > 1e-4
        valid_depth = depth_valid_mask.reshape(H, W)
        valid_mask = (valid_x & valid_y & valid_z & valid_depth).float()

        if reference_depth_map is not None:
            if reference_depth_map.dim() == 2:
                reference_depth_map = reference_depth_map.unsqueeze(0)
            if reference_depth_map.dim() == 3:
                reference_depth_map = reference_depth_map.unsqueeze(0)
            reference_depth_map = reference_depth_map.to(device=image.device, dtype=torch.float32)
            sampled_reference_depth = F.grid_sample(
                reference_depth_map,
                grid_batch,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0).squeeze(0)
            sampled_reference_depth = torch.nan_to_num(
                sampled_reference_depth,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(min=0.0)
            projected_reference_depth = reference_points[:, 2].reshape(H, W).clamp(min=1e-4)
            reference_depth_valid_mask = sampled_reference_depth > 1e-4

            relative_depth_error = (
                sampled_reference_depth - projected_reference_depth
            ).abs() / torch.maximum(
                torch.maximum(sampled_reference_depth, projected_reference_depth),
                torch.full_like(projected_reference_depth, 1e-4),
            )
            depth_consistency_hard_mask = (
                relative_depth_error <= self.relative_depth_consistency_tolerance
            )
            depth_consistency_confidence = torch.exp(
                -relative_depth_error / max(self.relative_depth_consistency_tolerance, 1e-4)
            )

            sampled_reference_u = points_2d[:, 0]
            sampled_reference_v = points_2d[:, 1]
            reference_pixels = torch.stack(
                [sampled_reference_u, sampled_reference_v, torch.ones_like(sampled_reference_u)],
                dim=-1,
            )
            try:
                ref_intrinsics_inverse = torch.linalg.pinv(ref_intrinsics.float())
            except Exception:
                ref_intrinsics_inverse = torch.inverse(ref_intrinsics.float())
            reference_rays = reference_pixels @ ref_intrinsics_inverse.T
            reference_points_3d = reference_rays * sampled_reference_depth.reshape(-1, 1)
            reference_points_homogeneous = torch.cat(
                [
                    reference_points_3d,
                    torch.ones(reference_points_3d.shape[0], 1, dtype=torch.float32),
                ],
                dim=-1,
            )
            try:
                reference_to_world = torch.linalg.pinv(ref_pose_f32)
            except Exception:
                reference_to_world = torch.inverse(ref_pose_f32)
            backward_world_points = reference_points_homogeneous @ reference_to_world.T
            backward_target_points = backward_world_points @ target_pose_f32.T
            backward_target_points = backward_target_points[:, :3]
            backward_target_homogeneous = backward_target_points @ target_intrinsics.float().T
            backward_target_depth = backward_target_homogeneous[:, 2:].clamp(min=1e-4)
            backward_target_uv = backward_target_homogeneous[:, :2] / backward_target_depth
            original_target_u = pixels_flat[:, 0]
            original_target_v = pixels_flat[:, 1]
            backward_reprojection_error = torch.sqrt(
                (backward_target_uv[:, 0] - original_target_u) ** 2
                + (backward_target_uv[:, 1] - original_target_v) ** 2
            ).reshape(H, W)
            forward_backward_hard_mask = (
                backward_reprojection_error <= self.forward_backward_pixel_tolerance
            )
            forward_backward_confidence = torch.exp(
                -backward_reprojection_error / max(self.forward_backward_pixel_tolerance, 1e-4)
            )
            warp_diagnostics = {
                "relative_depth_consistency_error": torch.nan_to_num(
                    relative_depth_error.unsqueeze(0),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "relative_depth_consistency_confidence": torch.nan_to_num(
                    depth_consistency_confidence.unsqueeze(0),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0),
                "forward_backward_reprojection_error": torch.nan_to_num(
                    backward_reprojection_error.unsqueeze(0),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "forward_backward_confidence": torch.nan_to_num(
                    forward_backward_confidence.unsqueeze(0),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0),
            }

            valid_mask = (
                valid_mask
                * reference_depth_valid_mask.float()
                * depth_consistency_hard_mask.float()
                * forward_backward_hard_mask.float()
                * depth_consistency_confidence
                * forward_backward_confidence
            )

        valid_mask = valid_mask.unsqueeze(0)  # [1, H, W]
        if return_diagnostics:
            return warped, valid_mask, warp_diagnostics
        return warped, valid_mask
    
    def _add_corruption(
        self, 
        image: torch.Tensor, 
        corruption_ratio: float = 0.3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """在图像上添加随机遮挡和噪声，模拟3DGS粗糙渲染
        
        Args:
            image: [3, H, W] 原始图像
            corruption_ratio: 遮挡比例
            
        Returns:
            corrupted: 添加遮挡后的图像 [3, H, W]
            valid_mask: 有效像素掩码 [1, H, W]
        """
        C, H, W = image.shape
        corrupted = image.clone()
        
        # 1. 随机矩形遮挡
        num_patches = int(corruption_ratio * 10)  # 遮挡块数量
        valid_mask = torch.ones(1, H, W)
        
        for _ in range(num_patches):
            # 随机遮挡块大小和位置
            patch_h = torch.randint(H // 8, H // 3, (1,)).item()
            patch_w = torch.randint(W // 8, W // 3, (1,)).item()
            top = torch.randint(0, H - patch_h, (1,)).item()
            left = torch.randint(0, W - patch_w, (1,)).item()
            
            # 应用遮挡（设为0）
            corrupted[:, top:top+patch_h, left:left+patch_w] = 0.0
            valid_mask[:, top:top+patch_h, left:left+patch_w] = 0.0
        
        # 2. 添加高斯噪声
        noise = torch.randn_like(corrupted) * 0.05
        corrupted = (corrupted + noise).clamp(0.0, 1.0)
        
        # 3. 轻微模糊（模拟渲染不准确）
        if self.is_train and torch.rand(1).item() > 0.5:
            corrupted = corrupted.unsqueeze(0)  # [1, 3, H, W]
            corrupted = F.avg_pool2d(corrupted, kernel_size=3, stride=1, padding=1)
            corrupted = corrupted.squeeze(0)  # [3, H, W]
        
        return corrupted, valid_mask

    def _degrade_online_conditions(
        self,
        rgb_render: torch.Tensor,
        depth_render: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将离线条件退化到更接近在线 3DGS render/depth 的分布。"""
        if not self.is_train:
            return rgb_render, depth_render

        degraded_rgb = rgb_render.clone()
        degraded_depth = depth_render.clone()
        condition_quality = float(torch.rand(1).item())

        blur_kernel = 1
        if condition_quality < 0.33:
            blur_kernel = 5
        elif condition_quality < 0.66:
            blur_kernel = 3

        if blur_kernel > 1:
            degraded_rgb = F.avg_pool2d(
                degraded_rgb.unsqueeze(0),
                kernel_size=blur_kernel,
                stride=1,
                padding=blur_kernel // 2,
            ).squeeze(0)
            degraded_depth = F.avg_pool2d(
                degraded_depth.unsqueeze(0),
                kernel_size=blur_kernel,
                stride=1,
                padding=blur_kernel // 2,
            ).squeeze(0)

        rgb_noise_std = 0.01 + 0.06 * condition_quality
        depth_noise_std = 0.005 + 0.03 * condition_quality
        degraded_rgb = torch.clamp(
            degraded_rgb + torch.randn_like(degraded_rgb) * rgb_noise_std,
            0.0,
            1.0,
        )
        degraded_depth = torch.clamp(
            degraded_depth + torch.randn_like(degraded_depth) * depth_noise_std,
            min=0.0,
        )

        dropout_probability = 0.03 + 0.22 * condition_quality
        dropout_mask = (
            torch.rand(
                1,
                degraded_rgb.shape[-2],
                degraded_rgb.shape[-1],
                device=degraded_rgb.device,
                dtype=degraded_rgb.dtype,
            ) > dropout_probability
        ).float()
        degraded_rgb = degraded_rgb * dropout_mask
        degraded_depth = degraded_depth * dropout_mask

        if condition_quality < 0.5:
            low_frequency_rgb = F.avg_pool2d(
                degraded_rgb.unsqueeze(0),
                kernel_size=7,
                stride=1,
                padding=3,
            ).squeeze(0)
            degraded_rgb = 0.7 * degraded_rgb + 0.3 * low_frequency_rgb

        return degraded_rgb, degraded_depth
    
    def _compute_warped_condition(
        self,
        sparse_images: torch.Tensor,   # [V, 3, H, W]
        sparse_poses: torch.Tensor,    # [V, 4, 4]
        sparse_intrinsics: torch.Tensor,  # [V, 3, 3]
        target_pose: torch.Tensor,     # [4, 4]
        target_intrinsics: torch.Tensor,  # [3, 3]
        target_depth_map: torch.Tensor,  # [1, H, W]
        sparse_depth_maps: Optional[torch.Tensor] = None,  # [V, 1, H, W]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算warped条件图像

        将所有稀疏视角分别warp到目标视平面，并按视角距离做逐像素联合融合。
        这样在3-view稀疏设定下，support base 不再退化成单参考warp。
        
        Returns:
            rgb_render: 联合warp图像 [3, H, W]
            reference_depth_map: 最近参考视角深度图 [1, H, W]
            reference_image: 最近参考图像 [3, H, W]
            valid_mask: 联合Warp有效区域 [1, H, W]
            reference_intrinsics: 最近参考视角内参 [3, 3]
            support_confidence: 联合support置信度 [1, H, W]
        """
        V = sparse_images.shape[0]

        target_rotation = target_pose[:3, :3]
        target_translation = target_pose[:3, 3]
        target_camera_center = -target_rotation.T @ target_translation
        view_distances = torch.stack([
            torch.norm(
                target_camera_center
                - (-sparse_poses[view_index, :3, :3].T @ sparse_poses[view_index, :3, 3])
            )
            for view_index in range(V)
        ])
        nearest_view_index = int(view_distances.argmin().item())
        nearest_reference_image = sparse_images[nearest_view_index]
        nearest_reference_intrinsics = sparse_intrinsics[nearest_view_index]

        warped_images: list[torch.Tensor] = []
        valid_masks: list[torch.Tensor] = []
        reference_depth_maps: list[torch.Tensor] = []

        for view_index in range(V):
            reference_image = sparse_images[view_index]
            reference_pose = sparse_poses[view_index]
            reference_intrinsics = sparse_intrinsics[view_index]
            if sparse_depth_maps is not None:
                reference_depth_map = sparse_depth_maps[view_index].to(dtype=reference_image.dtype)
            else:
                reference_depth_map = self._estimate_depth_da2(reference_image)[0:1]
            warped_image, valid_mask = self._perspective_warp_simple(
                reference_image,
                target_depth_map,
                reference_pose,
                target_pose,
                ref_intrinsics=reference_intrinsics,
                target_intrinsics=target_intrinsics,
                reference_depth_map=reference_depth_map,
            )
            warped_images.append(warped_image)
            valid_masks.append(valid_mask)
            reference_depth_maps.append(reference_depth_map)

        stacked_warped_images = torch.stack(warped_images, dim=0)  # [V, 3, H, W]
        stacked_valid_masks = torch.stack(valid_masks, dim=0)      # [V, 1, H, W]
        nearest_reference_depth_map = reference_depth_maps[nearest_view_index]

        inverse_distance_weights = 1.0 / torch.clamp(view_distances.float(), min=1e-4)
        inverse_distance_weights = inverse_distance_weights / torch.clamp(
            inverse_distance_weights.max(),
            min=1e-6,
        )
        per_view_support_scores = stacked_valid_masks * inverse_distance_weights.view(V, 1, 1, 1)
        best_view_indices = per_view_support_scores.squeeze(1).argmax(dim=0)  # [H, W]

        fused_warped_rgb = torch.zeros_like(stacked_warped_images[0])
        for view_index in range(V):
            selected_pixels = (best_view_indices == view_index).unsqueeze(0)
            fused_warped_rgb = torch.where(
                selected_pixels,
                stacked_warped_images[view_index],
                fused_warped_rgb,
            )

        joint_valid_mask = stacked_valid_masks.amax(dim=0)
        support_confidence = per_view_support_scores.amax(dim=0) * joint_valid_mask
        fused_warped_rgb = fused_warped_rgb * joint_valid_mask

        # 无效区域仍使用最近参考视角作弱填充，但support语义只由 joint_valid_mask 决定。
        blend_factor = 0.7
        rgb_render = (
            fused_warped_rgb * joint_valid_mask * blend_factor
            + nearest_reference_image * (1 - joint_valid_mask * blend_factor)
        )

        return (
            rgb_render,
            nearest_reference_depth_map,
            nearest_reference_image,
            joint_valid_mask,
            nearest_reference_intrinsics,
            support_confidence,
        )

    def _should_fallback_phase1_3dgs_condition(
        self,
        rgb_render: torch.Tensor,
        accum_render: torch.Tensor,
    ) -> Tuple[bool, Dict[str, float]]:
        render_coverage = float(accum_render.mean().item())
        render_std = float(rgb_render.std(unbiased=False).item())
        render_max = float(rgb_render.max().item())
        render_mean = float(rgb_render.mean().item())
        should_fallback = bool(
            render_coverage < self.phase1_3dgs_min_render_coverage
            or (
                render_std < self.phase1_3dgs_min_render_std
                and render_max < self.phase1_3dgs_max_dark_render_value
            )
        )
        return should_fallback, {
            "coverage": render_coverage,
            "std": render_std,
            "max": render_max,
            "mean": render_mean,
        }

    def _ensure_warp_fallback_ready(self) -> None:
        if self.enable_sfm_depth_alignment and self._sfm_depth_alignment_assets is None:
            self._sfm_depth_alignment_assets = self._load_sfm_depth_alignment_assets()
        if self.depth_model is None:
            self._init_depth_model()

    def _build_warp_fallback_condition(
        self,
        *,
        target_idx: int,
        target_pose: torch.Tensor,
        target_intrinsics: torch.Tensor,
        sparse_indices: np.ndarray,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        sparse_intrinsics: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        self._ensure_warp_fallback_ready()
        target_geometry_depth_map = self._get_cached_depth_map_for_image_index(int(target_idx)).clone()
        sparse_depth_maps = torch.stack([
            self._get_cached_depth_map_for_image_file(
                self.sparse_image_files[int(i)],
                pose_index=int(self.pose_indices_for_sparse_images[int(i)].item()),
                cache_key=f"sparse::{int(i)}",
            )
            for i in sparse_indices
        ])
        (
            rgb_render,
            reference_depth_map,
            _,
            warp_valid_mask,
            reference_intrinsics,
            warp_support_confidence,
        ) = self._compute_warped_condition(
            sparse_images=sparse_images,
            sparse_poses=sparse_poses,
            sparse_intrinsics=sparse_intrinsics,
            target_pose=target_pose,
            target_intrinsics=target_intrinsics,
            target_depth_map=target_geometry_depth_map,
            sparse_depth_maps=sparse_depth_maps,
        )
        support_anchor_rgb = rgb_render.clone()
        geometry_depth_render = target_geometry_depth_map.clone()
        depth_render = geometry_depth_render.clone()
        rgb_render, depth_render = self._degrade_online_conditions(
            rgb_render,
            depth_render,
        )
        normal_render = self._depth_to_normal(depth_render.repeat(3, 1, 1))
        accum_render = warp_support_confidence.clamp(0.0, 1.0)
        transmittance_render = (1.0 - accum_render).clamp(0.0, 1.0)
        return {
            "sparse_depth_maps": sparse_depth_maps,
            "rgb_render": rgb_render,
            "support_anchor_rgb": support_anchor_rgb,
            "geometry_depth_render": geometry_depth_render,
            "depth_render": depth_render,
            "normal_render": normal_render,
            "reference_depth_map": reference_depth_map,
            "reference_intrinsics": reference_intrinsics,
            "accum_render": accum_render,
            "transmittance_render": transmittance_render,
            "warp_support_confidence": warp_support_confidence,
            "warp_valid_mask": warp_valid_mask,
        }

    def _compute_phase1_frontier_mask(self, support_confidence: torch.Tensor) -> torch.Tensor:
        support_threshold = float(self.phase1_3dgs_frontier_support_threshold)
        support_mask = (support_confidence > support_threshold).float()
        kernel_size = max(3, int(self.phase1_3dgs_frontier_kernel_size))
        if kernel_size % 2 == 0:
            kernel_size += 1
        dilated_support_mask = F.max_pool2d(
            support_mask.unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        ).squeeze(0)
        frontier_mask = (dilated_support_mask - support_mask).clamp(0.0, 1.0)
        return frontier_mask

    def _compute_phase1_verification_prior(
        self,
        support_confidence: torch.Tensor,
        frontier_mask: torch.Tensor,
    ) -> torch.Tensor:
        clamped_support_confidence = support_confidence.clamp(0.0, 1.0)
        frontier_bonus = frontier_mask.clamp(0.0, 1.0) * 0.35
        verification_prior = (
            0.75 * clamped_support_confidence
            + 0.25 * torch.clamp(clamped_support_confidence + frontier_bonus, 0.0, 1.0)
        )
        return verification_prior.clamp(0.0, 1.0)

    def _phase1_source_uses_real_phase1_result(
        self,
        phase1_condition_source_tag: str,
    ) -> bool:
        normalized_phase1_condition_source_tag = str(phase1_condition_source_tag).strip().lower()
        return "phase1_result:" in normalized_phase1_condition_source_tag

    def _make_support_condition_key(self, sparse_indices: np.ndarray) -> str:
        support_image_names = sorted(
            self.sparse_image_files[int(sparse_index)].name
            for sparse_index in sparse_indices
        )
        return f"{self.scene_dir.name}::{'|'.join(support_image_names)}"

    def _remember_support_condition_provider(
        self,
        support_condition_key: str,
        provider_payload: Dict[str, Any],
    ) -> None:
        if support_condition_key in self._phase1_support_condition_provider_cache:
            del self._phase1_support_condition_provider_cache[support_condition_key]
        elif len(self._phase1_support_condition_provider_cache) >= self._phase1_support_condition_provider_cache_limit:
            oldest_support_condition_key = next(iter(self._phase1_support_condition_provider_cache))
            del self._phase1_support_condition_provider_cache[oldest_support_condition_key]
        self._phase1_support_condition_provider_cache[support_condition_key] = provider_payload

    def _get_or_build_support_conditioned_phase1_provider(
        self,
        sparse_indices: np.ndarray,
    ) -> Dict[str, Any]:
        from examples.active_sampling.train_active_sampling import build_condition_provider

        if self._phase1_3dgs_runtime_geometry_bank is None:
            raise RuntimeError("[LLFFDataset] support-conditioned geometry bank 未初始化。")
        support_condition_key = self._make_support_condition_key(sparse_indices)
        cached_provider_payload = self._phase1_support_condition_provider_cache.get(support_condition_key)
        if cached_provider_payload is not None:
            return cached_provider_payload

        support_existing_cameras = [
            self._build_existing_camera_from_pose_index(
                int(self.pose_indices_for_sparse_images[int(sparse_index)].item())
            )
            for sparse_index in sparse_indices
        ]
        condition_provider = build_condition_provider(
            gaussians=self._phase1_3dgs_runtime_geometry_bank["gaussians"],
            image_height=self.image_size[0],
            image_width=self.image_size[1],
            device=self._phase1_3dgs_runtime_geometry_bank["render_device"],
            mode="gddn",
            existing_cameras=support_existing_cameras,
            reference_image_height=self.image_size[0],
            reference_image_width=self.image_size[1],
        )
        provider_payload = {
            "support_condition_key": support_condition_key,
            "support_image_names": [
                self.sparse_image_files[int(sparse_index)].name
                for sparse_index in sparse_indices
            ],
            "condition_provider": condition_provider,
            "condition_source": str(
                self._phase1_3dgs_runtime_geometry_bank.get("condition_source", "unknown")
            ),
        }
        self._remember_support_condition_provider(
            support_condition_key=support_condition_key,
            provider_payload=provider_payload,
        )
        return provider_payload

    def _build_support_conditioned_phase1_package(
        self,
        *,
        target_idx: int,
        target_pose: torch.Tensor,
        target_intrinsics: torch.Tensor,
        sparse_indices: np.ndarray,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        sparse_intrinsics: torch.Tensor,
    ) -> Dict[str, Any]:
        from examples.active_sampling.utils import ViewCandidate

        provider_payload = self._get_or_build_support_conditioned_phase1_provider(sparse_indices)
        target_view_matrix = target_pose.detach().cpu().float()
        target_camera_to_world = torch.linalg.inv(target_view_matrix)
        target_candidate = ViewCandidate(
            position=target_camera_to_world[:3, 3].contiguous(),
            viewmat=target_view_matrix,
            intrinsics=target_intrinsics.detach().cpu().float(),
        )
        with torch.no_grad():
            condition_tensors = provider_payload["condition_provider"]([target_candidate])

        reference_intrinsics_tensor = condition_tensors.get("reference_intrinsics")
        if reference_intrinsics_tensor is None:
            reference_intrinsics_tensor = target_intrinsics.detach().cpu().float().unsqueeze(0)
        phase1_package: Dict[str, Any] = {
            "rgb_render": condition_tensors["rgb_render"][0].detach().cpu().float(),
            "geometry_depth_render": condition_tensors["depth_render"][0].detach().cpu().float(),
            "depth_render": condition_tensors["depth_render"][0].detach().cpu().float(),
            "normal_render": condition_tensors["normal_render"][0].detach().cpu().float(),
            "reference_depth_map": condition_tensors["reference_depth_map"][0].detach().cpu().float(),
            "reference_intrinsics": reference_intrinsics_tensor[0].detach().cpu().float(),
            "accum_render": condition_tensors["accum_render"][0].detach().cpu().float(),
            "transmittance_render": condition_tensors["transmittance_render"][0].detach().cpu().float(),
            "support_condition_key": provider_payload["support_condition_key"],
            "phase1_condition_source_tag": (
                "support_conditioned_geometry_bank::"
                f"{provider_payload['condition_source']}"
            ),
        }
        uses_real_phase1_result = self._phase1_source_uses_real_phase1_result(
            phase1_package["phase1_condition_source_tag"]
        )
        warp_support_condition = self._build_warp_fallback_condition(
            target_idx=int(target_idx),
            target_pose=target_pose,
            target_intrinsics=target_intrinsics,
            sparse_indices=sparse_indices,
            sparse_images=sparse_images,
            sparse_poses=sparse_poses,
            sparse_intrinsics=sparse_intrinsics,
        )
        phase1_package["warp_support_confidence"] = warp_support_condition[
            "warp_support_confidence"
        ].clone()
        phase1_package["warp_valid_mask"] = warp_support_condition["warp_valid_mask"].clone()
        phase1_package["support_anchor_rgb"] = warp_support_condition["support_anchor_rgb"].clone()
        phase1_package["reference_depth_map"] = warp_support_condition["reference_depth_map"].clone()
        phase1_package["reference_intrinsics"] = warp_support_condition["reference_intrinsics"].clone()
        if not uses_real_phase1_result:
            phase1_package["geometry_depth_render"] = warp_support_condition[
                "geometry_depth_render"
            ].clone()
            phase1_package["depth_render"] = warp_support_condition["depth_render"].clone()
            phase1_package["normal_render"] = warp_support_condition["normal_render"].clone()
        phase1_package["enable_3dgs_residual_refiner"] = bool(uses_real_phase1_result)

        should_fallback_to_warp, phase1_render_stats = self._should_fallback_phase1_3dgs_condition(
            rgb_render=phase1_package["rgb_render"],
            accum_render=phase1_package["accum_render"],
        )
        if should_fallback_to_warp:
            confident_phase1_mask = (
                phase1_package["accum_render"] > self.phase1_3dgs_frontier_support_threshold
            ).float()
            phase1_package["rgb_render"] = (
                phase1_package["rgb_render"] * confident_phase1_mask
                + warp_support_condition["rgb_render"] * (1.0 - confident_phase1_mask)
            )
            if uses_real_phase1_result:
                phase1_package["geometry_depth_render"] = (
                    phase1_package["geometry_depth_render"] * confident_phase1_mask
                    + warp_support_condition["geometry_depth_render"] * (1.0 - confident_phase1_mask)
                )
                phase1_package["depth_render"] = (
                    phase1_package["depth_render"] * confident_phase1_mask
                    + warp_support_condition["depth_render"] * (1.0 - confident_phase1_mask)
                )
                phase1_package["normal_render"] = (
                    phase1_package["normal_render"] * confident_phase1_mask
                    + warp_support_condition["normal_render"] * (1.0 - confident_phase1_mask)
                )
            phase1_package["accum_render"] = torch.maximum(
                phase1_package["accum_render"],
                warp_support_condition["accum_render"],
            )
            phase1_package["transmittance_render"] = (
                1.0 - phase1_package["accum_render"]
            ).clamp(0.0, 1.0)
            phase1_package["phase1_condition_source_tag"] = (
                f"{phase1_package['phase1_condition_source_tag']}+support_conditioned_warp_fill"
            )
            phase1_package["enable_3dgs_residual_refiner"] = False
            if self._phase1_3dgs_fallback_log_count < self._phase1_3dgs_fallback_log_limit:
                print(
                    "[LLFFDataset] support-conditioned geometry bank render 偏弱，当前样本已执行 support-conditioned warp 填补: "
                    f"scene={self.scene_dir.name}, image={self.image_files[int(target_idx)].name}, "
                    f"coverage={phase1_render_stats['coverage']:.4f}, "
                    f"std={phase1_render_stats['std']:.6f}, "
                    f"max={phase1_render_stats['max']:.6f}, "
                    f"mean={phase1_render_stats['mean']:.6f}"
                )
                self._phase1_3dgs_fallback_log_count += 1

        phase1_package["frontier_mask"] = self._compute_phase1_frontier_mask(
            phase1_package["warp_support_confidence"]
        )
        phase1_package["verification_prior"] = self._compute_phase1_verification_prior(
            phase1_package["warp_support_confidence"],
            phase1_package["frontier_mask"],
        )
        return phase1_package
    
    def __len__(self) -> int:
        if self.use_phase1_3dgs_conditions:
            return len(self._phase1_episode_sample_specs)
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> dict:
        support_template_id = -1
        episode_sample_spec: Optional[Dict[str, Any]] = None
        if self.use_phase1_3dgs_conditions:
            episode_sample_spec = self._phase1_episode_sample_specs[idx]
            target_idx = int(episode_sample_spec["target_idx"])
            support_template_id = int(episode_sample_spec["support_template_id"])
        else:
            target_idx = self.indices[idx]
        target_pose_index = int(self.pose_indices_for_images[target_idx].item())
        
        # 加载目标图像和位姿
        target_rgb = self._load_image(self.image_files[target_idx])
        target_pose = self.poses[target_pose_index]
        target_intrinsics = self.camera_intrinsics[target_pose_index]
        
        # 选择稀疏视角（排除目标视角）
        if self.use_phase1_3dgs_conditions:
            target_support_templates = self._phase1_support_template_bank.get(int(target_idx), [])
            support_template_metadata = None
            for candidate_template_metadata in target_support_templates:
                if int(candidate_template_metadata["support_template_id"]) == support_template_id:
                    support_template_metadata = candidate_template_metadata
                    break
            if support_template_metadata is None:
                raise RuntimeError(
                    "[LLFFDataset] 缺少 support template metadata: "
                    f"target_idx={int(target_idx)}, support_template_id={int(support_template_id)}"
                )
            sparse_indices = np.asarray(
                support_template_metadata["sparse_indices"],
                dtype=np.int64,
            )
        else:
            all_indices = list(range(len(self.sparse_image_files)))
            filtered_sparse_indices: list[int] = []
            for sparse_index in all_indices:
                if int(self.pose_indices_for_sparse_images[sparse_index].item()) == target_pose_index:
                    continue
                filtered_sparse_indices.append(sparse_index)
            all_indices = filtered_sparse_indices
        
        if not self.use_phase1_3dgs_conditions:
            if len(all_indices) >= self.effective_num_sparse_views:
                if self.is_train:
                    # 随机选择稀疏视角
                    sparse_indices = np.random.choice(all_indices, self.effective_num_sparse_views, replace=False)
                else:
                    # 验证时使用固定的稀疏视角
                    sparse_indices = all_indices[:self.effective_num_sparse_views]
            else:
                if self.strict_geometry_contract and len(all_indices) == 0:
                    raise RuntimeError(
                        "[LLFFDataset] strict geometry contract enabled, but no reference views are available."
                    )
                sparse_indices = np.asarray(all_indices, dtype=np.int64)
        else:
            if sparse_indices.size < self.effective_num_sparse_views:
                raise RuntimeError(
                    "[LLFFDataset] support template sparse_indices 数量不足: "
                    f"target_idx={int(target_idx)}, support_template_id={int(support_template_id)}, "
                    f"count={int(sparse_indices.size)}, required={int(self.effective_num_sparse_views)}"
                )

        for sparse_index in sparse_indices:
            if int(self.pose_indices_for_sparse_images[int(sparse_index)].item()) == target_pose_index:
                raise RuntimeError(
                    "[LLFFDataset] Selected support view has the target pose: "
                    f"target_idx={int(target_idx)}, sparse_idx={int(sparse_index)}, pose_idx={target_pose_index}"
                )
        
        # 加载稀疏视角图像和位姿
        sparse_images = torch.stack([
            self._load_image(self.sparse_image_files[int(i)]) for i in sparse_indices
        ])  # [V, 3, H, W]
        using_phase1_3dgs_conditions_for_sample = False
        if self.use_phase1_3dgs_conditions:
            sparse_depth_maps = torch.zeros(
                len(sparse_indices),
                1,
                self.image_size[0],
                self.image_size[1],
                dtype=target_rgb.dtype,
            )
        else:
            sparse_depth_maps = torch.stack([
                self._get_cached_depth_map_for_image_file(
                    self.sparse_image_files[int(i)],
                    pose_index=int(self.pose_indices_for_sparse_images[int(i)].item()),
                    cache_key=f"sparse::{int(i)}",
                )
                for i in sparse_indices
            ])  # [V, 1, H, W]
        sparse_poses = torch.stack([
            self.poses[int(self.pose_indices_for_sparse_images[int(i)].item())] for i in sparse_indices
        ])  # [V, 4, 4]
        sparse_intrinsics = torch.stack([
            self.camera_intrinsics[int(self.pose_indices_for_sparse_images[int(i)].item())] for i in sparse_indices
        ])  # [V, 3, 3]
        target_geometry_depth_map = None
        if not self.use_phase1_3dgs_conditions:
            target_geometry_depth_map = self._get_cached_depth_map_for_image_index(int(target_idx)).clone()
        
        H, W = self.image_size

        reference_depth_map = None
        reference_intrinsics = None
        accum_render = torch.zeros(1, H, W, dtype=target_rgb.dtype)
        transmittance_render = torch.ones(1, H, W, dtype=target_rgb.dtype)
        warp_valid_mask = torch.ones(1, H, W, dtype=target_rgb.dtype)
        warp_support_confidence = torch.ones(1, H, W, dtype=target_rgb.dtype)
        support_anchor_rgb = target_rgb.clone()
        frontier_mask = torch.zeros(1, H, W, dtype=target_rgb.dtype)
        verification_prior = torch.zeros(1, H, W, dtype=target_rgb.dtype)
        phase1_condition_source_tag = "none"
        episode_support_valid_coverage = 0.0
        episode_support_confidence_mean = 0.0
        episode_frontier_coverage = 0.0
        episode_novel_coverage = 0.0
        episode_verification_prior_mean = 0.0
        episode_proposal_hardness_score = 0.0
        # 根据 condition_mode 选择离线条件源。
        if self.use_phase1_3dgs_conditions:
            phase1_condition_package = self._lookup_prebuilt_phase1_episode_package(
                target_idx=int(target_idx),
                support_template_id=int(support_template_id),
            )
            rgb_render = phase1_condition_package["rgb_render"].clone()
            support_anchor_rgb = phase1_condition_package.get(
                "support_anchor_rgb",
                rgb_render,
            ).clone()
            geometry_depth_render = phase1_condition_package["geometry_depth_render"].clone()
            depth_render = phase1_condition_package["depth_render"].clone()
            normal_render = phase1_condition_package["normal_render"].clone()
            reference_depth_map = phase1_condition_package["reference_depth_map"].clone()
            reference_intrinsics = phase1_condition_package["reference_intrinsics"].clone()
            accum_render = phase1_condition_package["accum_render"].clone()
            transmittance_render = phase1_condition_package["transmittance_render"].clone()
            warp_support_confidence = phase1_condition_package["warp_support_confidence"].clone()
            warp_valid_mask = phase1_condition_package["warp_valid_mask"].clone()
            frontier_mask = phase1_condition_package["frontier_mask"].clone()
            verification_prior = phase1_condition_package["verification_prior"].clone()
            phase1_condition_source_tag = str(
                phase1_condition_package.get("phase1_condition_source_tag", "support_conditioned_geometry_bank")
            )
            using_phase1_3dgs_conditions_for_sample = bool(
                phase1_condition_package.get("enable_3dgs_residual_refiner", False)
            )
            if episode_sample_spec is not None:
                episode_support_valid_coverage = float(
                    episode_sample_spec.get("support_valid_coverage", 0.0)
                )
                episode_support_confidence_mean = float(
                    episode_sample_spec.get("support_confidence_mean", 0.0)
                )
                episode_frontier_coverage = float(
                    episode_sample_spec.get("frontier_coverage", 0.0)
                )
                episode_novel_coverage = float(
                    episode_sample_spec.get("novel_coverage", 0.0)
                )
                episode_verification_prior_mean = float(
                    episode_sample_spec.get("verification_prior_mean", 0.0)
                )
                episode_proposal_hardness_score = float(
                    episode_sample_spec.get("proposal_hardness_score", 0.0)
                )
        elif self.condition_mode == "warp":
            (
                rgb_render,
                reference_depth_map,
                _,
                warp_valid_mask,
                reference_intrinsics,
                warp_support_confidence,
            ) = self._compute_warped_condition(
                sparse_images,
                sparse_poses,
                sparse_intrinsics,
                target_pose,
                target_intrinsics,
                target_depth_map=target_geometry_depth_map,
                sparse_depth_maps=sparse_depth_maps,
            )
            support_anchor_rgb = rgb_render.clone()
        elif self.condition_mode == "corrupt":
            rgb_render, _ = self._add_corruption(
                target_rgb, self.corruption_ratio
            )
            support_anchor_rgb = rgb_render.clone()
        else:
            raise ValueError(f"不支持的条件生成模式: {self.condition_mode}")

        if not self.use_phase1_3dgs_conditions:
            if reference_depth_map is None:
                reference_depth_map = sparse_depth_maps[0].clone()
            if reference_intrinsics is None:
                reference_intrinsics = sparse_intrinsics[0]

            # 几何 warp / support base 需要使用未退化的干净目标深度；
            # 条件生成则继续消费退化后的在线风格 depth_render。
            if target_geometry_depth_map is None:
                raise RuntimeError("[LLFFDataset] warp/corrupt 模式下缺少 target_geometry_depth_map。")
            geometry_depth_render = target_geometry_depth_map.clone()
            depth_render = geometry_depth_render.clone()
            rgb_render, depth_render = self._degrade_online_conditions(
                rgb_render,
                depth_render,
            )

            # 从退化后的 depth 条件生成法线，而不是再回到 target_rgb 派生条件。
            normal_render = self._depth_to_normal(depth_render.repeat(3, 1, 1))
        
        return {
            "sparse_images": sparse_images,
            "sparse_depth_maps": sparse_depth_maps,
            "sparse_poses": sparse_poses,
            "sparse_intrinsics": sparse_intrinsics,
            "scene_name": str(self.scene_dir.name),
            "target_image_index": torch.tensor(int(target_idx), dtype=torch.long),
            "support_template_id": torch.tensor(int(support_template_id), dtype=torch.long),
            "support_template_stage_tag": str(self.support_template_stage_tag),
            "target_pose": target_pose,
            "target_intrinsics": target_intrinsics,
            "target_rgb": target_rgb,
            "rgb_render": rgb_render,
            "support_anchor_rgb": support_anchor_rgb,
            "geometry_depth_render": geometry_depth_render,  # [1, H, W]
            "depth_render": depth_render,  # [1, H, W]
            "reference_depth_map": reference_depth_map,  # [1, H, W]
            "reference_intrinsics": reference_intrinsics,  # [3, 3]
            "warp_valid_mask": warp_valid_mask,  # [1, H, W]
            "warp_support_confidence": warp_support_confidence,  # [1, H, W]
            "supported_exist_mask": warp_valid_mask,  # [1, H, W]
            "accum_render": accum_render,  # [1, H, W] or None
            "transmittance_render": transmittance_render,  # [1, H, W] or None
            "use_3dgs_residual_refiner": torch.tensor(
                1.0 if using_phase1_3dgs_conditions_for_sample else 0.0,
                dtype=target_rgb.dtype,
            ),
            "frontier_mask": frontier_mask,
            "verification_prior": verification_prior,
            "episode_support_valid_coverage": torch.tensor(
                episode_support_valid_coverage,
                dtype=target_rgb.dtype,
            ),
            "episode_support_confidence_mean": torch.tensor(
                episode_support_confidence_mean,
                dtype=target_rgb.dtype,
            ),
            "episode_frontier_coverage": torch.tensor(
                episode_frontier_coverage,
                dtype=target_rgb.dtype,
            ),
            "episode_novel_coverage": torch.tensor(
                episode_novel_coverage,
                dtype=target_rgb.dtype,
            ),
            "episode_verification_prior_mean": torch.tensor(
                episode_verification_prior_mean,
                dtype=target_rgb.dtype,
            ),
            "episode_proposal_hardness_score": torch.tensor(
                episode_proposal_hardness_score,
                dtype=target_rgb.dtype,
            ),
            "phase1_condition_source_tag": phase1_condition_source_tag,
            "phase1_episode_conditioned": torch.tensor(
                1.0 if self.use_phase1_3dgs_conditions else 0.0,
                dtype=target_rgb.dtype,
            ),
            "normal_render": normal_render,  # [3, H, W] 从深度估计的法线
            "target_latent": torch.zeros(4, H // 8, W // 8),  # 占位符
        }


def _dataset_supports_multiprocess_dataloader(dataset: Dataset) -> bool:
    if isinstance(dataset, LLFFMultiSceneDataset):
        return all(
            bool(getattr(scene_dataset, "supports_multiprocess_dataloader", False))
            for scene_dataset in dataset.scene_datasets
        )
    return bool(getattr(dataset, "supports_multiprocess_dataloader", False))


def _build_dataloader_runtime_kwargs(
    *,
    dataset: Dataset,
    requested_num_workers: int,
    is_train: bool,
    loader_tag: str,
) -> Dict[str, Any]:
    requested_num_workers = max(0, int(requested_num_workers))
    multiprocess_safe = _dataset_supports_multiprocess_dataloader(dataset)
    effective_num_workers = requested_num_workers if multiprocess_safe else 0
    runtime_kwargs: Dict[str, Any] = {
        "num_workers": effective_num_workers,
        "pin_memory": True,
    }
    if effective_num_workers > 0:
        runtime_kwargs["persistent_workers"] = True
        runtime_kwargs["prefetch_factor"] = 2 if is_train else 1
        print(
            f"[DataLoader] {loader_tag}: num_workers={effective_num_workers}, "
            f"persistent_workers=True, prefetch_factor={int(runtime_kwargs['prefetch_factor'])}"
        )
    elif requested_num_workers > 0 and not multiprocess_safe:
        print(
            f"[DataLoader] {loader_tag}: 数据集仍包含主进程依赖路径，"
            "已回退到 num_workers=0 以保持稳定。"
        )
    return runtime_kwargs


def create_dataloader(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    is_train: bool = True,
    strict_geometry_contract: bool = True,
    condition_mode: str = "warp",
    offline_phase1_result: Optional[str] = None,
    offline_3dgs_cache_dir: Optional[str] = None,
    enable_sfm_depth_alignment: bool = False,
    num_sparse_views: int = 3,
    image_size: int = 512,
    verbose_dataset_logs: bool = False,
    prewarm_image_cache: bool = False,
) -> DataLoader:
    """Create data loader for training/validation."""
    
    # 检查数据目录是否存在
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"[create_dataloader] 数据目录不存在: {data_dir}")

    # 使用LLFF数据集
    print(f"[INFO] Loading LLFF dataset from: {data_dir}")
    dataset = LLFFDataset(
        data_dir=data_dir,
        num_sparse_views=num_sparse_views,
        image_size=(int(image_size), int(image_size)),
        is_train=is_train,
        train_ratio=1.0,  # 小数据集使用全部图像训练
        condition_mode=condition_mode,
        strict_geometry_contract=strict_geometry_contract,
        offline_phase1_result=offline_phase1_result,
        offline_3dgs_cache_dir=offline_3dgs_cache_dir,
        enable_sfm_depth_alignment=enable_sfm_depth_alignment,
        verbose_dataset_logs=verbose_dataset_logs,
        prewarm_image_cache=prewarm_image_cache,
    )
    
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=is_train)

    dataloader_runtime_kwargs = _build_dataloader_runtime_kwargs(
        dataset=dataset,
        requested_num_workers=num_workers,
        is_train=is_train,
        loader_tag="single-scene",
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not is_distributed and is_train),
        sampler=sampler,
        drop_last=is_train,
        **dataloader_runtime_kwargs,
    )
    setattr(
        dataloader,
        "support_template_stage_tag",
        getattr(dataset, "support_template_stage_tag", "unknown"),
    )
    setattr(
        dataloader,
        "support_template_scope",
        getattr(dataset, "support_template_scope", "unknown"),
    )
    setattr(
        dataloader,
        "effective_num_sparse_views",
        int(getattr(dataset, "effective_num_sparse_views", num_sparse_views)),
    )
    setattr(
        dataloader,
        "use_phase1_3dgs_conditions",
        bool(getattr(dataset, "use_phase1_3dgs_conditions", False)),
    )
    return dataloader


def create_multiscene_dataloader(
    scene_specs: Sequence[Tuple[str, Path]],
    batch_size: int,
    num_workers: int,
    is_distributed: bool,
    *,
    is_train: bool,
    strict_geometry_contract: bool,
    condition_mode: str,
    offline_phase1_result: Optional[str],
    offline_3dgs_cache_dir: Optional[str],
    enable_sfm_depth_alignment: bool,
    num_sparse_views: int,
    image_size: int = 512,
    support_template_stage_tag: Optional[str] = None,
    scene_balanced_sampling: bool = True,
    balanced_samples_per_scene: int = 0,
    hard_episode_top_fraction: float = 1.0,
    hard_episode_repeat_factor: int = 1,
    verbose_dataset_logs: bool = False,
    prewarm_image_cache: bool = False,
) -> Optional[DataLoader]:
    if not scene_specs:
        return None

    scene_datasets: List[LLFFDataset] = []
    scene_sample_counts: Dict[str, int] = {}
    for scene_name, scene_dir in scene_specs:
        scene_dataset = LLFFDataset(
            data_dir=str(scene_dir),
            num_sparse_views=num_sparse_views,
            image_size=(int(image_size), int(image_size)),
            is_train=is_train,
            train_ratio=1.0,
            condition_mode=condition_mode,
            strict_geometry_contract=strict_geometry_contract,
            offline_phase1_result=offline_phase1_result,
            offline_3dgs_cache_dir=offline_3dgs_cache_dir,
            enable_sfm_depth_alignment=enable_sfm_depth_alignment,
            support_template_stage_tag=support_template_stage_tag,
            hard_episode_top_fraction=hard_episode_top_fraction,
            hard_episode_repeat_factor=hard_episode_repeat_factor,
            verbose_dataset_logs=verbose_dataset_logs,
            prewarm_image_cache=prewarm_image_cache,
        )
        scene_datasets.append(scene_dataset)
        scene_sample_counts[scene_name] = len(scene_dataset)

    if len(scene_datasets) == 1:
        combined_dataset: Dataset = scene_datasets[0]
    else:
        combined_dataset = LLFFMultiSceneDataset(
            scene_datasets=scene_datasets,
            scene_names=[scene_name for scene_name, _ in scene_specs],
        )

    sampler = None
    if (
        is_train
        and scene_balanced_sampling
        and isinstance(combined_dataset, LLFFMultiSceneDataset)
        and len(combined_dataset.scene_lengths) > 1
    ):
        sampler = SceneBalancedDistributedSampler(
            combined_dataset,
            shuffle=True,
            num_replicas=(dist.get_world_size() if is_distributed and dist.is_initialized() else 1),
            rank=(dist.get_rank() if is_distributed and dist.is_initialized() else 0),
            samples_per_scene=balanced_samples_per_scene,
        )
    elif is_distributed:
        sampler = DistributedSampler(combined_dataset, shuffle=is_train)

    dataloader_runtime_kwargs = _build_dataloader_runtime_kwargs(
        dataset=combined_dataset,
        requested_num_workers=num_workers,
        is_train=is_train,
        loader_tag="multi-scene",
    )

    dataloader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None and not is_distributed and is_train),
        sampler=sampler,
        drop_last=is_train,
        **dataloader_runtime_kwargs,
    )
    setattr(dataloader, "scene_names", [scene_name for scene_name, _ in scene_specs])
    setattr(dataloader, "scene_sample_counts", scene_sample_counts)
    reference_dataset = scene_datasets[0] if scene_datasets else None
    setattr(
        dataloader,
        "support_template_stage_tag",
        getattr(reference_dataset, "support_template_stage_tag", "unknown"),
    )
    setattr(
        dataloader,
        "support_template_scope",
        getattr(reference_dataset, "support_template_scope", "unknown"),
    )
    setattr(
        dataloader,
        "effective_num_sparse_views",
        int(getattr(reference_dataset, "effective_num_sparse_views", num_sparse_views)),
    )
    setattr(
        dataloader,
        "use_phase1_3dgs_conditions",
        bool(getattr(reference_dataset, "use_phase1_3dgs_conditions", False)),
    )
    setattr(
        dataloader,
        "scene_balanced_sampling",
        bool(isinstance(sampler, SceneBalancedDistributedSampler)),
    )
    setattr(
        dataloader,
        "effective_num_workers",
        int(dataloader_runtime_kwargs.get("num_workers", 0)),
    )
    setattr(dataloader, "hard_episode_top_fraction", float(hard_episode_top_fraction))
    setattr(dataloader, "hard_episode_repeat_factor", int(hard_episode_repeat_factor))
    if isinstance(sampler, SceneBalancedDistributedSampler):
        setattr(dataloader, "balanced_samples_per_scene", int(sampler.samples_per_scene))
    return dataloader


def _resolve_expected_support_template_scope(
    *,
    support_template_stage_tag: str,
    effective_num_sparse_views: int,
) -> str:
    normalized_stage_tag = str(support_template_stage_tag).strip().lower()
    if (
        ("stage2_alignment" in normalized_stage_tag or "stage3" in normalized_stage_tag)
        and int(effective_num_sparse_views) <= 3
    ):
        return "scene_global"
    return "target_local"


def _assert_dataloader_protocol(
    *,
    dataloader: Optional[Iterable[Dict[str, torch.Tensor]]],
    loader_name: str,
    expected_stage_tag: str,
    expected_num_sparse_views: int,
    expected_template_scope: str,
    require_phase1_episode_bank: bool,
) -> None:
    if dataloader is None:
        raise RuntimeError(
            f"[ProtocolGuard] {loader_name} dataloader is None; expected "
            f"stage_tag={expected_stage_tag}, support_views={int(expected_num_sparse_views)}."
        )
    dataset = getattr(dataloader, "dataset", None)
    resolved_stage_tag = str(
        getattr(
            dataloader,
            "support_template_stage_tag",
            getattr(dataset, "support_template_stage_tag", "unknown"),
        )
    ).strip().lower()
    resolved_template_scope = str(
        getattr(
            dataloader,
            "support_template_scope",
            getattr(dataset, "support_template_scope", "unknown"),
        )
    ).strip().lower()
    resolved_num_sparse_views = int(
        getattr(
            dataloader,
            "effective_num_sparse_views",
            getattr(dataset, "effective_num_sparse_views", -1),
        )
    )
    resolved_use_phase1_episode_bank = bool(
        getattr(
            dataloader,
            "use_phase1_3dgs_conditions",
            getattr(dataset, "use_phase1_3dgs_conditions", False),
        )
    )
    protocol_errors: List[str] = []
    if resolved_stage_tag != str(expected_stage_tag).strip().lower():
        protocol_errors.append(
            f"stage_tag={resolved_stage_tag} != expected {str(expected_stage_tag).strip().lower()}"
        )
    if resolved_num_sparse_views != int(expected_num_sparse_views):
        protocol_errors.append(
            f"support_views={resolved_num_sparse_views} != expected {int(expected_num_sparse_views)}"
        )
    if resolved_template_scope != str(expected_template_scope).strip().lower():
        protocol_errors.append(
            f"template_scope={resolved_template_scope} != expected {str(expected_template_scope).strip().lower()}"
        )
    if require_phase1_episode_bank and not resolved_use_phase1_episode_bank:
        protocol_errors.append("phase1_episode_bank=False but expected True")
    if protocol_errors:
        raise RuntimeError(
            f"[ProtocolGuard] {loader_name} protocol mismatch: "
            + "; ".join(protocol_errors)
        )


def _infer_geometry_aware_trainable_tokens(model: nn.Module) -> Tuple[str, ...]:
    available_parameter_names = [name for name, _ in model.named_parameters()]
    preferred_tokens = [
        "controlnet",
        "pose_encoder",
        "encoder_proj",
        "_pose_proj",
        "geometry_auxiliary_router",
        "frontier_proposal_head",
        "plucker_embedder",
        "condition_encoder",
        "condition_adapter",
        "view_aggregator",
        "residual_gates",
    ]
    resolved_tokens: List[str] = []
    for preferred_token in preferred_tokens:
        if any(preferred_token in parameter_name for parameter_name in available_parameter_names):
            resolved_tokens.append(preferred_token)
    return tuple(dict.fromkeys(resolved_tokens))


def _configure_geometry_aware_teacher_training(
    *,
    trainer: "ThreeStageTrainer",
    args: argparse.Namespace,
    is_main: bool,
) -> None:
    if not args.llff_multiscene_root or args.disable_geometry_aware_defaults:
        return

    explicit_trainable_tokens = _parse_scene_name_list(
        getattr(args, "geometry_aware_trainable_tokens", "")
    )
    trainable_name_tokens = (
        tuple(explicit_trainable_tokens)
        if explicit_trainable_tokens
        else _infer_geometry_aware_trainable_tokens(trainer.raw_model)
    )
    if hasattr(trainer, "configure_geometry_aware_trainable_filter"):
        trainer.configure_geometry_aware_trainable_filter(
            trainable_name_tokens,
            include_uncertainty=False,
        )
    trainer.stage3_force_controlnet_trainable = True

    applied_defaults: Dict[str, Any] = {}
    trainer.exist_diffusion_weight = 1.00
    trainer.novel_diffusion_weight = 0.30
    trainer.novel_reconstruction_weight = 0.25
    trainer.sample_path_surrogate_reconstruction_decay = 0.25
    trainer.final_sample_supervision_weight = min(
        float(getattr(trainer, "final_sample_supervision_weight", 0.20)),
        0.25,
    )
    trainer.final_sample_supervision_max_weight = min(
        float(getattr(trainer, "final_sample_supervision_max_weight", 0.35)),
        0.40,
    )
    trainer.trainable_sample_supervision_weight = min(
        float(getattr(trainer, "trainable_sample_supervision_weight", 0.0)),
        0.18,
    )
    trainer.trainable_sample_supervision_max_weight = min(
        float(getattr(trainer, "trainable_sample_supervision_max_weight", 0.0)),
        0.25,
    )
    applied_defaults["exist_diffusion_weight"] = trainer.exist_diffusion_weight
    applied_defaults["novel_diffusion_weight"] = trainer.novel_diffusion_weight
    applied_defaults["novel_reconstruction_weight"] = trainer.novel_reconstruction_weight
    applied_defaults["sample_path_surrogate_reconstruction_decay"] = (
        trainer.sample_path_surrogate_reconstruction_decay
    )
    if args.teacher_final_sample_support_fidelity_weight is None:
        trainer.final_sample_support_fidelity_weight = 0.35
        applied_defaults["final_sample_support_fidelity_weight"] = 0.35
    if args.teacher_final_sample_support_consistency_weight is None:
        trainer.final_sample_support_consistency_weight = 0.15
        applied_defaults["final_sample_support_consistency_weight"] = 0.15
    if args.teacher_final_sample_support_confidence_boost is None:
        trainer.final_sample_support_confidence_boost = 1.00
        applied_defaults["final_sample_support_confidence_boost"] = 1.00
    if args.teacher_final_sample_support_low_confidence_anchor_weight is None:
        trainer.final_sample_support_low_confidence_anchor_weight = 0.08
        applied_defaults["final_sample_support_low_confidence_anchor_weight"] = 0.08
    if args.teacher_relative_effect_weight is None:
        trainer.relative_teacher_effect_weight = 1.25
        applied_defaults["relative_teacher_effect_weight"] = 1.25
    else:
        trainer.relative_teacher_effect_weight = max(
            float(args.teacher_relative_effect_weight),
            0.0,
        )
    trainer.teacher_warp_margin_weight = 1.0
    trainer.teacher_warp_margin = 0.01
    trainer.teacher_warp_editable_weight = 0.5
    trainer.validation_use_warp_delta_score = True
    trainer.stage2a_frontier_residual_weight = 1.0
    trainer.stage3_frontier_residual_weight = 1.0
    trainer.frontier_residual_anchor_weight = 1.00
    trainer.frontier_residual_recon_alpha = 0.75
    trainer.proposal_confidence_supervision_weight = 0.50
    trainer.proposal_warp_error_supervision_weight = 0.50
    trainer.proposal_repairability_supervision_weight = 0.50
    trainer.proposal_verification_supervision_weight = 0.25
    trainer.proposal_acceptance_supervision_weight = 0.50
    trainer.proposal_confidence_temperature = 0.05
    trainer.proposal_opportunity_error_threshold = 0.0
    trainer.proposal_opportunity_top_fraction = 0.30
    trainer.proposal_repair_prior_quantile = 0.90
    trainer.proposal_verification_min_support = 0.05
    trainer.proposal_repairability_empirical_max_weight = 1.00
    trainer.proposal_repairability_empirical_transition_epochs = 1
    trainer.proposal_empirical_patch_kernel_size = 15
    trainer.proposal_empirical_candidate_count = 3
    trainer.proposal_empirical_candidate_mc_dropout_p = 0.15
    trainer.proposal_empirical_disable_hard_gate = True
    trainer.proposal_empirical_include_decoded_teacher = True
    trainer.proposal_empirical_include_local_repair_candidates = True
    trainer.proposal_empirical_local_repair_kernel_sizes = (9, 21, 41)
    trainer.proposal_head_positive_balance_max = 8.0
    trainer.proposal_warp_error_rank_blend = 0.50
    trainer.proposal_head_mean_calibration_weight = 0.25
    trainer.proposal_selective_risk_weight = 1.0
    trainer.stage2a_localization_phase_ratio = 0.33
    trainer.stage2a_localization_sample_path_weight = 0.0
    trainer.stage2a_residual_sample_path_weight = 0.25
    trainer.stage2a_episode_novel_reconstruction_scale = 0.10
    trainer.proposal_validation_acceptance_threshold = 0.50
    if hasattr(trainer.raw_model, "proposal_safe_confidence_threshold"):
        trainer.raw_model.proposal_safe_confidence_threshold = 0.50
    if hasattr(trainer.raw_model, "proposal_safe_confidence_gate_power"):
        trainer.raw_model.proposal_safe_confidence_gate_power = 1.0
    if hasattr(trainer.raw_model, "proposal_use_hard_acceptance_gate"):
        trainer.raw_model.proposal_use_hard_acceptance_gate = True
    if hasattr(trainer.raw_model, "proposal_hard_acceptance_straight_through"):
        trainer.raw_model.proposal_hard_acceptance_straight_through = True
    if (
        args.teacher_proposal_safe_confidence_threshold is not None
        and hasattr(trainer.raw_model, "proposal_safe_confidence_threshold")
    ):
        trainer.raw_model.proposal_safe_confidence_threshold = min(
            max(float(args.teacher_proposal_safe_confidence_threshold), 0.0),
            0.99,
        )
    if hasattr(trainer.raw_model, "proposal_warp_error_gate_power"):
        trainer.raw_model.proposal_warp_error_gate_power = 1.0
    if hasattr(trainer.raw_model, "proposal_repairability_gate_power"):
        trainer.raw_model.proposal_repairability_gate_power = 1.0
    if hasattr(trainer.raw_model, "proposal_verification_gate_power"):
        trainer.raw_model.proposal_verification_gate_power = 1.0
    trainer.stage2a_surrogate_reconstruction_decay = 0.35
    trainer.stage3_surrogate_reconstruction_decay = 0.25
    trainer.stage2a_exist_diffusion_weight = 0.0
    trainer.stage2a_novel_diffusion_weight = 1.0
    trainer.stage3_exist_diffusion_weight = 0.0
    trainer.stage3_novel_diffusion_weight = 1.0
    applied_defaults["teacher_warp_margin_weight"] = trainer.teacher_warp_margin_weight
    applied_defaults["teacher_warp_margin"] = trainer.teacher_warp_margin
    applied_defaults["teacher_warp_editable_weight"] = trainer.teacher_warp_editable_weight
    applied_defaults["validation_use_warp_delta_score"] = bool(
        trainer.validation_use_warp_delta_score
    )
    applied_defaults["stage2a_frontier_residual_weight"] = trainer.stage2a_frontier_residual_weight
    applied_defaults["stage3_frontier_residual_weight"] = trainer.stage3_frontier_residual_weight
    applied_defaults["frontier_residual_anchor_weight"] = trainer.frontier_residual_anchor_weight
    applied_defaults["frontier_residual_recon_alpha"] = trainer.frontier_residual_recon_alpha
    applied_defaults["proposal_confidence_supervision_weight"] = (
        trainer.proposal_confidence_supervision_weight
    )
    applied_defaults["proposal_warp_error_supervision_weight"] = (
        trainer.proposal_warp_error_supervision_weight
    )
    applied_defaults["proposal_repairability_supervision_weight"] = (
        trainer.proposal_repairability_supervision_weight
    )
    applied_defaults["proposal_verification_supervision_weight"] = (
        trainer.proposal_verification_supervision_weight
    )
    applied_defaults["proposal_acceptance_supervision_weight"] = (
        trainer.proposal_acceptance_supervision_weight
    )
    applied_defaults["proposal_confidence_temperature"] = (
        trainer.proposal_confidence_temperature
    )
    applied_defaults["proposal_opportunity_error_threshold"] = (
        trainer.proposal_opportunity_error_threshold
    )
    applied_defaults["proposal_opportunity_top_fraction"] = (
        trainer.proposal_opportunity_top_fraction
    )
    applied_defaults["proposal_repair_prior_quantile"] = (
        trainer.proposal_repair_prior_quantile
    )
    applied_defaults["proposal_verification_min_support"] = (
        trainer.proposal_verification_min_support
    )
    applied_defaults["proposal_repairability_empirical_max_weight"] = (
        trainer.proposal_repairability_empirical_max_weight
    )
    applied_defaults["proposal_repairability_empirical_transition_epochs"] = (
        trainer.proposal_repairability_empirical_transition_epochs
    )
    applied_defaults["proposal_empirical_patch_kernel_size"] = (
        trainer.proposal_empirical_patch_kernel_size
    )
    applied_defaults["proposal_empirical_candidate_count"] = (
        trainer.proposal_empirical_candidate_count
    )
    applied_defaults["proposal_empirical_candidate_mc_dropout_p"] = (
        trainer.proposal_empirical_candidate_mc_dropout_p
    )
    applied_defaults["proposal_empirical_disable_hard_gate"] = bool(
        trainer.proposal_empirical_disable_hard_gate
    )
    applied_defaults["proposal_empirical_include_decoded_teacher"] = bool(
        trainer.proposal_empirical_include_decoded_teacher
    )
    applied_defaults["proposal_empirical_include_local_repair_candidates"] = bool(
        trainer.proposal_empirical_include_local_repair_candidates
    )
    applied_defaults["proposal_empirical_local_repair_kernel_sizes"] = tuple(
        trainer.proposal_empirical_local_repair_kernel_sizes
    )
    applied_defaults["proposal_head_positive_balance_max"] = (
        trainer.proposal_head_positive_balance_max
    )
    applied_defaults["proposal_warp_error_rank_blend"] = (
        trainer.proposal_warp_error_rank_blend
    )
    applied_defaults["proposal_head_mean_calibration_weight"] = (
        trainer.proposal_head_mean_calibration_weight
    )
    applied_defaults["proposal_selective_risk_weight"] = (
        trainer.proposal_selective_risk_weight
    )
    applied_defaults["stage2a_localization_phase_ratio"] = (
        trainer.stage2a_localization_phase_ratio
    )
    applied_defaults["stage2a_localization_sample_path_weight"] = (
        trainer.stage2a_localization_sample_path_weight
    )
    applied_defaults["stage2a_residual_sample_path_weight"] = (
        trainer.stage2a_residual_sample_path_weight
    )
    applied_defaults["stage2a_episode_novel_reconstruction_scale"] = (
        trainer.stage2a_episode_novel_reconstruction_scale
    )
    applied_defaults["proposal_validation_acceptance_threshold"] = (
        trainer.proposal_validation_acceptance_threshold
    )
    applied_defaults["proposal_safe_confidence_threshold"] = getattr(
        trainer.raw_model,
        "proposal_safe_confidence_threshold",
        None,
    )
    applied_defaults["proposal_safe_confidence_gate_power"] = getattr(
        trainer.raw_model,
        "proposal_safe_confidence_gate_power",
        None,
    )
    applied_defaults["proposal_use_hard_acceptance_gate"] = getattr(
        trainer.raw_model,
        "proposal_use_hard_acceptance_gate",
        None,
    )
    applied_defaults["proposal_hard_acceptance_straight_through"] = getattr(
        trainer.raw_model,
        "proposal_hard_acceptance_straight_through",
        None,
    )
    applied_defaults["proposal_warp_error_gate_power"] = getattr(
        trainer.raw_model,
        "proposal_warp_error_gate_power",
        None,
    )
    applied_defaults["proposal_repairability_gate_power"] = getattr(
        trainer.raw_model,
        "proposal_repairability_gate_power",
        None,
    )
    applied_defaults["proposal_verification_gate_power"] = getattr(
        trainer.raw_model,
        "proposal_verification_gate_power",
        None,
    )
    applied_defaults["stage2a_surrogate_reconstruction_decay"] = (
        trainer.stage2a_surrogate_reconstruction_decay
    )
    applied_defaults["stage3_surrogate_reconstruction_decay"] = (
        trainer.stage3_surrogate_reconstruction_decay
    )
    applied_defaults["stage2a_exist_diffusion_weight"] = trainer.stage2a_exist_diffusion_weight
    applied_defaults["stage2a_novel_diffusion_weight"] = trainer.stage2a_novel_diffusion_weight
    applied_defaults["stage3_exist_diffusion_weight"] = trainer.stage3_exist_diffusion_weight
    applied_defaults["stage3_novel_diffusion_weight"] = trainer.stage3_novel_diffusion_weight
    trainer.geometry_aware_frontier_proposal_focus = not bool(
        args.disable_geometry_aware_frontier_proposal_focus
    )
    if args.teacher_frontier_proposal_kernel_size is None:
        trainer.geometry_aware_frontier_kernel_size = 31
        applied_defaults["geometry_aware_frontier_kernel_size"] = 31
    else:
        trainer.geometry_aware_frontier_kernel_size = max(
            int(args.teacher_frontier_proposal_kernel_size),
            3,
        )
    if args.teacher_frontier_proposal_support_threshold is None:
        trainer.geometry_aware_frontier_support_threshold = 0.05
        applied_defaults["geometry_aware_frontier_support_threshold"] = 0.05
    else:
        trainer.geometry_aware_frontier_support_threshold = min(
            max(float(args.teacher_frontier_proposal_support_threshold), 0.0),
            1.0,
        )
    applied_defaults["geometry_aware_frontier_proposal_focus"] = bool(
        trainer.geometry_aware_frontier_proposal_focus
    )

    if is_main:
        trainable_parameter_count = sum(
            parameter.numel() for parameter in trainer.model.parameters() if parameter.requires_grad
        )
        print("[GeometryAware] 已启用 LLFF 多场景 geometry-aware 生成先验 finetune")
        print(f"[GeometryAware] trainable_name_tokens={list(trainable_name_tokens)}")
        print(f"[GeometryAware] trainable_params={trainable_parameter_count:,}")
        if applied_defaults:
            print(f"[GeometryAware] applied_defaults={applied_defaults}")


def main():
    args = parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank, is_distributed = setup_distributed()
    is_main = rank == 0
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    _apply_llff_multiscene_runtime_defaults(args=args, is_main=is_main)
    
    if is_main:
        print("=" * 60)
        print("UTrustGS Generative-Prior Training")
        print("=" * 60)
        print(f"Backbone: {args.backbone}")
        print(f"World size: {world_size}")
        print(f"Device: {device}")
        print("=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)



    
    # Create model
    if is_main:
        print(f"[1/4] Creating {args.backbone.upper()} model...")
    
    from gddn_factory import create_gddn_model
    
    model = create_gddn_model(
        backbone=args.backbone,
        pretrained_path=args.local_path,
        torch_dtype=torch.float32,
        default_guidance_scale=args.teacher_guidance_scale,
        enable_plucker_conditioning=args.enable_plucker_conditioning,
        controlnet_condition_channels=args.controlnet_condition_channels,
    )
    if hasattr(model, "require_reference_depth_map"):
        model.require_reference_depth_map = not args.relaxed_geometry_contract
    else:
        setattr(model, "require_reference_depth_map", not args.relaxed_geometry_contract)
    if hasattr(model, "proposal_residual_source"):
        model.proposal_residual_source = str(args.teacher_proposal_residual_source)
    if args.teacher_support_projection_conflict_target_scheduler_delta is not None:
        model.support_projection_conflict_target_scheduler_delta = float(
            args.teacher_support_projection_conflict_target_scheduler_delta
        )
    if args.teacher_support_projection_conflict_min_scale is not None:
        model.support_projection_conflict_min_scale = float(
            args.teacher_support_projection_conflict_min_scale
        )
    if args.teacher_support_projection_conflict_gamma is not None:
        model.support_projection_conflict_gamma = float(
            args.teacher_support_projection_conflict_gamma
        )
    model.to(device)
    if is_main:
        print(f"Model created. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(
            "[Config] Generative-prior sampling defaults: "
            f"guidance_scale={args.teacher_guidance_scale:.2f}, "
            f"plucker={'on' if args.enable_plucker_conditioning else 'off'}, "
            f"cond_channels={args.controlnet_condition_channels}, "
            f"offline_condition_source={args.offline_condition_source}, "
            f"sfm_depth_alignment={'on' if args.enable_sfm_depth_alignment else 'off'}, "
            f"proposal_residual_source={getattr(model, 'proposal_residual_source', 'head')}"
        )
        if (
            args.teacher_support_projection_conflict_target_scheduler_delta is not None
            or args.teacher_support_projection_conflict_min_scale is not None
            or args.teacher_support_projection_conflict_gamma is not None
        ):
            print(
                "[Config] Projection conflict gate overrides: "
                "target_scheduler_delta="
                f"{float(getattr(model, 'support_projection_conflict_target_scheduler_delta', 0.03)):.4f}, "
                "min_scale="
                f"{float(getattr(model, 'support_projection_conflict_min_scale', 0.65)):.4f}, "
                "gamma="
                f"{float(getattr(model, 'support_projection_conflict_gamma', 1.0)):.4f}"
            )
    
    # Disable TF32 for numerical stability
    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if is_main:
            print("[Config] TF32 disabled for numerical stability")
    
    # 自动检测：如果有多个GPU可用且未使用分布式训练，自动启用多GPU
    num_available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    auto_multi_gpu = num_available_gpus >= 2 and not is_distributed
    
    if args.multi_gpu or auto_multi_gpu:
        if num_available_gpus >= 2:
            if is_main:
                print(f"[Config] Enabling multi-GPU with {num_available_gpus} GPUs")
            model.enable_multi_gpu()
            # 注意：不要在enable_multi_gpu()后调用model.to(device)，否则会覆盖设备分配
        elif is_main and args.multi_gpu:
            print(f"[Config] --multi-gpu specified but only {num_available_gpus} GPU(s) available")
    
    
    # Wrap with DDP for distributed training
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    if is_main:
        print(f"  - Model loaded successfully")
    if args.vae_decode_chunk_size is not None:
        resolved_vae_decode_chunk_size = max(1, int(args.vae_decode_chunk_size))
        setattr(model.module if hasattr(model, "module") else model, "vae_decode_chunk_size", resolved_vae_decode_chunk_size)
        if is_main:
            print(f"[Config] vae_decode_chunk_size={resolved_vae_decode_chunk_size}")
    
    # Create dataloaders
    if is_main:
        print(f"[2/4] Creating dataloaders...")

    if args.llff_multiscene_root and args.offline_condition_source not in {"warp", "phase1_3dgs"}:
        raise ValueError(
            "LLFF 多场景 geometry-aware 生成先验训练仅支持 offline-condition-source=warp 或 phase1_3dgs。"
        )
    if args.llff_multiscene_root and args.offline_phase1_result is not None:
        raise ValueError(
            "LLFF 多场景生成先验训练不支持单一 offline_phase1_result；"
            "请留空以按场景使用 sparse COLMAP bootstrap，或后续扩展为 per-scene cache。"
        )

    if args.offline_3dgs_cache_dir:
        offline_cache_root = str(Path(args.offline_3dgs_cache_dir).expanduser().resolve())
    elif args.llff_multiscene_root and str(args.offline_condition_source) == "phase1_3dgs":
        offline_cache_root = str((Path(args.output_dir) / "llff_shared_offline_condition_cache").resolve())
    else:
        offline_cache_root = str(output_dir / "offline_condition_cache")
    resolved_stage1_batch_size = max(
        1,
        int(args.stage1_batch_size if args.stage1_batch_size is not None else args.batch_size),
    )
    resolved_alignment_batch_size = max(
        1,
        int(args.alignment_batch_size if args.alignment_batch_size is not None else args.batch_size),
    )
    train_loader: DataLoader
    val_loader: Optional[DataLoader] = None
    sparse_alignment_train_loader: Optional[DataLoader] = None
    sparse_alignment_val_loader: Optional[DataLoader] = None
    build_sparse_alignment_dataloaders = None
    if args.llff_multiscene_root:
        llff_root_dir = Path(args.llff_multiscene_root).expanduser().resolve()
        pretrain_sparse_views = max(1, int(args.llff_pretrain_sparse_views))
        alignment_sparse_views = max(1, int(args.llff_alignment_sparse_views))
        alignment_hard_episode_top_fraction_train = 1.0
        alignment_hard_episode_top_fraction_val = 1.0
        alignment_hard_episode_repeat_factor = 1
        if not args.disable_geometry_aware_defaults:
            alignment_hard_episode_top_fraction_train = 0.50
            alignment_hard_episode_top_fraction_val = 0.25
            alignment_hard_episode_repeat_factor = 2
        train_scene_specs, val_scene_specs = _resolve_llff_multiscene_splits(
            llff_root_dir=llff_root_dir,
            explicit_train_scenes=_parse_scene_name_list(args.llff_train_scenes),
            explicit_val_scenes=_parse_scene_name_list(args.llff_val_scenes),
            holdout_scene_name=args.llff_holdout_scene,
            auto_val_scene_count=args.llff_auto_val_scene_count,
        )
        if is_main:
            print(f"[LLFF-MultiScene] root={llff_root_dir}")
            print(f"[LLFF-MultiScene] train_scenes={[scene_name for scene_name, _ in train_scene_specs]}")
            print(f"[LLFF-MultiScene] val_scenes={[scene_name for scene_name, _ in val_scene_specs]}")
            if args.val_dir:
                print("[LLFF-MultiScene] 已忽略单场景 val-dir，改用跨场景验证集。")
            print(
                "[LLFF-MultiScene] support-view protocol: "
                f"stage1_pretrain={pretrain_sparse_views}, "
                f"stage2a_alignment={alignment_sparse_views}"
            )
            print(
                "[LLFF-MultiScene] hard-episode protocol: "
                f"alignment_train_top_fraction={alignment_hard_episode_top_fraction_train:.2f}, "
                f"alignment_val_top_fraction={alignment_hard_episode_top_fraction_val:.2f}, "
                f"alignment_repeat_factor={alignment_hard_episode_repeat_factor}"
            )
            print(f"[Config] Offline condition cache root: {offline_cache_root}")

        train_loader = create_multiscene_dataloader(
            train_scene_specs,
            resolved_stage1_batch_size,
            args.num_workers,
            is_distributed,
            is_train=True,
            strict_geometry_contract=not args.relaxed_geometry_contract,
            condition_mode=args.offline_condition_source,
            offline_phase1_result=None,
            offline_3dgs_cache_dir=offline_cache_root,
            enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
            num_sparse_views=pretrain_sparse_views,
            image_size=max(32, int(args.teacher_image_size)),
            support_template_stage_tag="stage1_pretrain",
            scene_balanced_sampling=(not args.disable_llff_scene_balanced_sampling),
            balanced_samples_per_scene=max(0, int(args.llff_balanced_samples_per_scene)),
            hard_episode_top_fraction=1.0,
            hard_episode_repeat_factor=1,
            verbose_dataset_logs=bool(args.verbose_dataset_logs),
            prewarm_image_cache=bool(args.prewarm_image_cache),
        )
        val_loader = create_multiscene_dataloader(
            val_scene_specs,
            resolved_stage1_batch_size,
            args.num_workers,
            is_distributed,
            is_train=False,
            strict_geometry_contract=not args.relaxed_geometry_contract,
            condition_mode=args.offline_condition_source,
            offline_phase1_result=None,
            offline_3dgs_cache_dir=offline_cache_root,
            enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
            num_sparse_views=pretrain_sparse_views,
            image_size=max(32, int(args.teacher_image_size)),
            support_template_stage_tag="stage1_pretrain",
            scene_balanced_sampling=False,
            balanced_samples_per_scene=0,
            hard_episode_top_fraction=1.0,
            hard_episode_repeat_factor=1,
            verbose_dataset_logs=bool(args.verbose_dataset_logs),
            prewarm_image_cache=bool(args.prewarm_image_cache),
        )

        def _build_sparse_alignment_dataloaders() -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
            nonlocal sparse_alignment_train_loader, sparse_alignment_val_loader
            if sparse_alignment_train_loader is not None or sparse_alignment_val_loader is not None:
                return sparse_alignment_train_loader, sparse_alignment_val_loader
            if is_main:
                print(
                    "[LLFF-MultiScene] 正在按需构建 sparse-view alignment dataloader: "
                    f"support_views={alignment_sparse_views}"
                )
            sparse_alignment_train_loader = create_multiscene_dataloader(
                train_scene_specs,
                resolved_alignment_batch_size,
                args.num_workers,
                is_distributed,
                is_train=True,
                strict_geometry_contract=not args.relaxed_geometry_contract,
                condition_mode=args.offline_condition_source,
                offline_phase1_result=None,
                offline_3dgs_cache_dir=offline_cache_root,
                enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
                num_sparse_views=alignment_sparse_views,
                image_size=max(32, int(args.teacher_image_size)),
                support_template_stage_tag="stage2_alignment",
                scene_balanced_sampling=(not args.disable_llff_scene_balanced_sampling),
                balanced_samples_per_scene=max(0, int(args.llff_balanced_samples_per_scene)),
                hard_episode_top_fraction=alignment_hard_episode_top_fraction_train,
                hard_episode_repeat_factor=alignment_hard_episode_repeat_factor,
                verbose_dataset_logs=bool(args.verbose_dataset_logs),
                prewarm_image_cache=bool(args.prewarm_image_cache),
            )
            sparse_alignment_val_loader = create_multiscene_dataloader(
                val_scene_specs,
                resolved_alignment_batch_size,
                args.num_workers,
                is_distributed,
                is_train=False,
                strict_geometry_contract=not args.relaxed_geometry_contract,
                condition_mode=args.offline_condition_source,
                offline_phase1_result=None,
                offline_3dgs_cache_dir=offline_cache_root,
                enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
                num_sparse_views=alignment_sparse_views,
                image_size=max(32, int(args.teacher_image_size)),
                support_template_stage_tag="stage2_alignment",
                scene_balanced_sampling=False,
                balanced_samples_per_scene=0,
                hard_episode_top_fraction=alignment_hard_episode_top_fraction_val,
                hard_episode_repeat_factor=1,
                verbose_dataset_logs=bool(args.verbose_dataset_logs),
                prewarm_image_cache=bool(args.prewarm_image_cache),
            )
            if str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
                expected_alignment_template_scope = _resolve_expected_support_template_scope(
                    support_template_stage_tag="stage2_alignment",
                    effective_num_sparse_views=alignment_sparse_views,
                )
                _assert_dataloader_protocol(
                    dataloader=sparse_alignment_train_loader,
                    loader_name="stage2_alignment/train",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=alignment_sparse_views,
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
                if sparse_alignment_val_loader is not None:
                    _assert_dataloader_protocol(
                        dataloader=sparse_alignment_val_loader,
                        loader_name="stage2_alignment/val",
                        expected_stage_tag="stage2_alignment",
                        expected_num_sparse_views=alignment_sparse_views,
                        expected_template_scope=expected_alignment_template_scope,
                        require_phase1_episode_bank=True,
                    )
            return sparse_alignment_train_loader, sparse_alignment_val_loader

        build_sparse_alignment_dataloaders = _build_sparse_alignment_dataloaders

        if args.disable_lazy_alignment_dataloaders:
            _build_sparse_alignment_dataloaders()
    else:
        train_loader = create_dataloader(
            args.data_dir,
            args.batch_size,
            args.num_workers,
            is_distributed,
            is_train=True,
            strict_geometry_contract=not args.relaxed_geometry_contract,
            condition_mode=args.offline_condition_source,
            offline_phase1_result=args.offline_phase1_result,
            offline_3dgs_cache_dir=offline_cache_root,
            enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
            num_sparse_views=max(1, int(args.llff_sparse_views)),
            image_size=max(32, int(args.teacher_image_size)),
            verbose_dataset_logs=bool(args.verbose_dataset_logs),
            prewarm_image_cache=bool(args.prewarm_image_cache),
        )
        
        if args.val_dir:
            val_loader = create_dataloader(
                args.val_dir,
                args.batch_size,
                args.num_workers,
                is_distributed,
                is_train=False,
                strict_geometry_contract=not args.relaxed_geometry_contract,
                condition_mode=args.offline_condition_source,
                offline_phase1_result=args.offline_phase1_result,
                offline_3dgs_cache_dir=offline_cache_root,
                enable_sfm_depth_alignment=args.enable_sfm_depth_alignment,
                num_sparse_views=max(1, int(args.llff_sparse_views)),
                image_size=max(32, int(args.teacher_image_size)),
                verbose_dataset_logs=bool(args.verbose_dataset_logs),
                prewarm_image_cache=bool(args.prewarm_image_cache),
            )
    
    if args.llff_multiscene_root and str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
        expected_stage1_template_scope = _resolve_expected_support_template_scope(
            support_template_stage_tag="stage1_pretrain",
            effective_num_sparse_views=pretrain_sparse_views,
        )
        _assert_dataloader_protocol(
            dataloader=train_loader,
            loader_name="stage1_pretrain/train",
            expected_stage_tag="stage1_pretrain",
            expected_num_sparse_views=pretrain_sparse_views,
            expected_template_scope=expected_stage1_template_scope,
            require_phase1_episode_bank=True,
        )
        if val_loader is not None:
            _assert_dataloader_protocol(
                dataloader=val_loader,
                loader_name="stage1_pretrain/val",
                expected_stage_tag="stage1_pretrain",
                expected_num_sparse_views=pretrain_sparse_views,
                expected_template_scope=expected_stage1_template_scope,
                require_phase1_episode_bank=True,
            )

    if is_main:
        print(
            "  - Batch protocol: "
            f"stage1_batch_size={resolved_stage1_batch_size}, "
            f"alignment_batch_size={resolved_alignment_batch_size}"
        )
        train_scene_names = getattr(train_loader, "scene_names", None)
        train_scene_sample_counts = getattr(train_loader, "scene_sample_counts", None)
        if train_scene_names:
            print(f"  - Train scenes: {train_scene_names}")
            if args.verbose_dataset_logs:
                print(f"  - Train scene sample counts: {train_scene_sample_counts}")
        if getattr(train_loader, "scene_balanced_sampling", False) and args.verbose_dataset_logs:
            print(
                "  - Train scene-balanced sampling: "
                f"enabled (samples_per_scene={getattr(train_loader, 'balanced_samples_per_scene', 'auto')})"
            )
        val_scene_names = getattr(val_loader, "scene_names", None) if val_loader is not None else None
        val_scene_sample_counts = getattr(val_loader, "scene_sample_counts", None) if val_loader is not None else None
        if val_scene_names:
            print(f"  - Val scenes: {val_scene_names}")
            if args.verbose_dataset_logs:
                print(f"  - Val scene sample counts: {val_scene_sample_counts}")
    
    # Create trainer
    if is_main:
        print(f"[3/4] Initializing ThreeStageTrainer...")
    
    from gddn_trainer import ThreeStageTrainer
    
    writer = None
    if is_main and SummaryWriter is not None:
        log_dir = output_dir / "tensorboard"
        writer = SummaryWriter(str(log_dir))
    
    trainer = ThreeStageTrainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        device=device,
        writer=writer,
        world_rank=rank,
        world_size=world_size,
        log_dir=str(output_dir),
    )
    trainer.verbose_epoch_stats = bool(args.verbose_epoch_stats)
    trainer.early_stop_patience = max(0, int(args.early_stop_patience))
    trainer.early_stop_min_epochs = max(0, int(args.early_stop_min_epochs))
    trainer.early_stop_min_delta = max(0.0, float(args.early_stop_min_delta))
    if args.enable_amp:
        trainer._amp_enabled = True
        trainer._amp_dtype = (
            torch.bfloat16 if str(args.amp_dtype).strip().lower() == "bf16" else torch.float16
        )
        if is_main:
            print(
                "[Config] AMP enabled: "
                f"dtype={str(args.amp_dtype).strip().lower()}, "
                "Stage1/Stage2a/Stage2 将使用 autocast，Stage3 保持既有 FP32 策略。"
            )
    trainer.sample_guidance_scale = float(args.teacher_guidance_scale)
    _configure_geometry_aware_teacher_training(
        trainer=trainer,
        args=args,
        is_main=is_main,
    )
    if args.teacher_final_sample_support_fidelity_weight is not None:
        trainer.final_sample_support_fidelity_weight = max(
            float(args.teacher_final_sample_support_fidelity_weight),
            0.0,
        )
        if is_main:
            print(
                "[Config] final_sample_support_fidelity_weight="
                f"{trainer.final_sample_support_fidelity_weight:.4f}"
            )
    if args.teacher_final_sample_support_consistency_weight is not None:
        trainer.final_sample_support_consistency_weight = max(
            float(args.teacher_final_sample_support_consistency_weight),
            0.0,
        )
        if is_main:
            print(
                "[Config] final_sample_support_consistency_weight="
                f"{trainer.final_sample_support_consistency_weight:.4f}"
            )
    if args.teacher_final_sample_support_confidence_boost is not None:
        trainer.final_sample_support_confidence_boost = max(
            float(args.teacher_final_sample_support_confidence_boost),
            0.0,
        )
        if is_main:
            print(
                "[Config] final_sample_support_confidence_boost="
                f"{trainer.final_sample_support_confidence_boost:.4f}"
            )
    if args.teacher_final_sample_support_low_confidence_anchor_weight is not None:
        trainer.final_sample_support_low_confidence_anchor_weight = max(
            float(args.teacher_final_sample_support_low_confidence_anchor_weight),
            0.0,
        )
        if is_main:
            print(
                "[Config] final_sample_support_low_confidence_anchor_weight="
                f"{trainer.final_sample_support_low_confidence_anchor_weight:.4f}"
            )
    if args.teacher_relative_effect_weight is not None:
        trainer.relative_teacher_effect_weight = max(
            float(args.teacher_relative_effect_weight),
            0.0,
        )
        if is_main:
            print(
                "[Config] relative_teacher_effect_weight="
                f"{trainer.relative_teacher_effect_weight:.4f}"
            )
    if args.teacher_frontier_proposal_kernel_size is not None:
        trainer.geometry_aware_frontier_kernel_size = max(
            int(args.teacher_frontier_proposal_kernel_size),
            3,
        )
    if args.teacher_frontier_proposal_support_threshold is not None:
        trainer.geometry_aware_frontier_support_threshold = min(
            max(float(args.teacher_frontier_proposal_support_threshold), 0.0),
            1.0,
        )
    trainer.geometry_aware_frontier_proposal_focus = not bool(
        args.disable_geometry_aware_frontier_proposal_focus
    )
    if is_main and args.llff_multiscene_root and not args.disable_geometry_aware_defaults:
        print(
            "[Config] geometry-aware frontier proposal focus: "
            f"enabled={bool(trainer.geometry_aware_frontier_proposal_focus)}, "
            f"kernel={int(getattr(trainer, 'geometry_aware_frontier_kernel_size', 31))}, "
            "support_threshold="
            f"{float(getattr(trainer, 'geometry_aware_frontier_support_threshold', 0.05)):.4f}"
        )
    fixed_visualization_sample_indices: List[int] = []
    seen_fixed_visualization_indices: Set[int] = set()
    for raw_sample_index in str(args.fixed_visualization_sample_indices).split(","):
        sample_index_token = raw_sample_index.strip()
        if not sample_index_token:
            continue
        try:
            resolved_sample_index = max(int(sample_index_token), 0)
        except ValueError:
            if is_main:
                print(
                    "[FixedVisualization] WARNING: ignore invalid sample index token: "
                    f"{sample_index_token}"
                )
            continue
        if resolved_sample_index in seen_fixed_visualization_indices:
            continue
        seen_fixed_visualization_indices.add(resolved_sample_index)
        fixed_visualization_sample_indices.append(resolved_sample_index)
    if not fixed_visualization_sample_indices:
        fixed_visualization_sample_indices = [0]
    trainer.fixed_visualization_sample_indices = fixed_visualization_sample_indices
    trainer.fixed_visualization_sample_index = fixed_visualization_sample_indices[0]
    default_stage_interval = 5 if args.llff_multiscene_root else 1
    default_validation_mode = "trainable_sample" if args.llff_multiscene_root else "full_generate"
    default_validation_max_batches = 8 if args.llff_multiscene_root else 0
    default_stage1_x0_warmup_steps = (
        64 if args.llff_multiscene_root and str(args.offline_condition_source) == "phase1_3dgs" else 0
    )
    default_stage2a_x0_warmup_steps = 0
    default_stage1_sample_path_interval = (
        2 if args.llff_multiscene_root and str(args.offline_condition_source) == "phase1_3dgs" else 5
    )
    default_stage2a_sample_path_interval = 1 if args.llff_multiscene_root else 5
    default_stage1_sample_path_weight = 1.5 if args.llff_multiscene_root else 1.0
    default_stage2a_sample_path_weight = 2.0 if args.llff_multiscene_root else 1.0
    trainer.stage1_validation_interval = max(
        0,
        int(
            args.stage1_validation_interval
            if args.stage1_validation_interval is not None
            else default_stage_interval
        ),
    )
    trainer.stage1_visualization_interval = max(
        0,
        int(
            args.stage1_visualization_interval
            if args.stage1_visualization_interval is not None
            else default_stage_interval
        ),
    )
    trainer.stage2a_validation_interval = max(
        0,
        int(
            args.stage2a_validation_interval
            if args.stage2a_validation_interval is not None
            else default_stage_interval
        ),
    )
    trainer.stage2a_visualization_interval = max(
        0,
        int(
            args.stage2a_visualization_interval
            if args.stage2a_visualization_interval is not None
            else default_stage_interval
        ),
    )
    trainer.stage1_x0_warmup_steps = max(
        0,
        int(
            args.stage1_x0_warmup_steps
            if args.stage1_x0_warmup_steps is not None
            else default_stage1_x0_warmup_steps
        ),
    )
    trainer.stage2a_x0_warmup_steps = max(
        0,
        int(
            args.stage2a_x0_warmup_steps
            if args.stage2a_x0_warmup_steps is not None
            else default_stage2a_x0_warmup_steps
        ),
    )
    trainer.stage1_validation_mode = str(
        args.stage1_validation_mode
        if args.stage1_validation_mode is not None
        else default_validation_mode
    ).strip().lower()
    trainer.stage2a_validation_mode = str(
        args.stage2a_validation_mode
        if args.stage2a_validation_mode is not None
        else default_validation_mode
    ).strip().lower()
    trainer.stage1_validation_max_batches = max(
        0,
        int(
            args.stage1_validation_max_batches
            if args.stage1_validation_max_batches is not None
            else default_validation_max_batches
        ),
    )
    trainer.stage2a_validation_max_batches = max(
        0,
        int(
            args.stage2a_validation_max_batches
            if args.stage2a_validation_max_batches is not None
            else default_validation_max_batches
        ),
    )
    trainer.stage1_sample_path_interval = max(
        1,
        int(
            args.stage1_sample_path_interval
            if args.stage1_sample_path_interval is not None
            else default_stage1_sample_path_interval
        ),
    )
    trainer.stage2a_sample_path_interval = max(
        1,
        int(
            args.stage2a_sample_path_interval
            if args.stage2a_sample_path_interval is not None
            else default_stage2a_sample_path_interval
        ),
    )
    trainer.stage1_sample_path_weight = max(
        0.0,
        float(
            args.stage1_sample_path_weight
            if args.stage1_sample_path_weight is not None
            else default_stage1_sample_path_weight
        ),
    )
    trainer.stage2a_sample_path_weight = max(
        0.0,
        float(
            args.stage2a_sample_path_weight
            if args.stage2a_sample_path_weight is not None
            else default_stage2a_sample_path_weight
        ),
    )
    if is_main:
        print(
            "[Config] Stage1 intervals: "
            f"validation_every={int(trainer.stage1_validation_interval)}, "
            f"visualization_every={int(trainer.stage1_visualization_interval)}"
        )
        print(
            "[Config] Stage2a intervals: "
            f"validation_every={int(trainer.stage2a_validation_interval)}, "
            f"visualization_every={int(trainer.stage2a_visualization_interval)}"
        )
        print(
            "[Config] Stage1 objective path: "
            f"x0_warmup_steps={int(trainer.stage1_x0_warmup_steps)}, "
            f"validation_mode={trainer.stage1_validation_mode}, "
            f"validation_max_batches={int(trainer.stage1_validation_max_batches)}, "
            f"sample_path_interval={int(trainer.stage1_sample_path_interval)}, "
            f"sample_path_weight={float(trainer.stage1_sample_path_weight):.2f}"
        )
        print(
            "[Config] Stage2a objective path: "
            f"x0_warmup_steps={int(trainer.stage2a_x0_warmup_steps)}, "
            f"validation_mode={trainer.stage2a_validation_mode}, "
            f"validation_max_batches={int(trainer.stage2a_validation_max_batches)}, "
            f"sample_path_interval={int(trainer.stage2a_sample_path_interval)}, "
            f"sample_path_weight={float(trainer.stage2a_sample_path_weight):.2f}"
        )
        if trainer.early_stop_patience > 0:
            print(
                "[Config] Early stopping: "
                f"patience={int(trainer.early_stop_patience)}, "
                f"min_epochs={int(trainer.early_stop_min_epochs)}, "
                f"min_delta={float(trainer.early_stop_min_delta):.4f}"
            )
    if sparse_alignment_train_loader is not None:
        trainer.sparse_alignment_train_dataloader = sparse_alignment_train_loader
        trainer.sparse_alignment_val_dataloader = sparse_alignment_val_loader
    
    # 默认启用 GST + CBT + IBC 训练链路
    trainer.gst_enabled = True    # 启用GST信息分解 (已修复归一化)
    trainer.cbt_enabled = True    # 启用CBT冲突惩罚
    trainer.ibc_enabled = True    # 启用IBC交替优化
    
    # IBC内层校准配置
    trainer.ibc_inner_steps = 5   # 内层迭代次数
    trainer.ibc_inner_lr = 5e-4   # 内层学习率
    
    if is_main:
        print("[Config] GST + CBT + IBC training path enabled")
    
    # Run three-stage training
    stages_to_run = [int(s) for s in args.stage.split(",")]
    run_stage1 = 1 in stages_to_run
    run_stage2 = 2 in stages_to_run
    run_stage3 = 3 in stages_to_run
    run_stage2a = bool(
        args.llff_multiscene_root
        and int(args.stage2a_epochs) > 0
        and run_stage1
        and (run_stage2 or run_stage3)
    )
    if is_main:
        print(f"[4/4] Starting training for stages: {stages_to_run} (raw: '{args.stage}')")
        print(f"  - Stage 1: {args.stage1_epochs} epochs @ lr={args.stage1_lr}")
        if run_stage2a:
            print(f"  - Stage 2a: {args.stage2a_epochs} epochs @ lr={args.stage2a_lr}")
        print(f"  - Stage 2: {args.stage2_epochs} epochs @ lr={args.stage2_lr}")
        print(f"  - Stage 3: {args.stage3_epochs} epochs @ lr={args.stage3_lr}")

    # Load checkpoint if specified
    if args.resume_from:
        if is_main:
            print(f"loading checkpoint from {args.resume_from}")
        # Map location to cpu to avoid gpu memory spikes during load
        checkpoint = torch.load(args.resume_from, map_location="cpu")
        if hasattr(trainer.raw_model, "load_state_dict"):
            load_state_dict_with_whitelist(
                trainer.raw_model,
                checkpoint,
                report_prefix="[run_bayesgs_diff]",
            )
        
        if is_main:
            print("Checkpoint loaded successfully.")


    # Stage 1: RGB generation pretraining
    if run_stage1:
        if is_main:
            print("\n" + "=" * 40)
            print("STAGE 1: RGB Generation Pretraining")
            print("=" * 40)
        
        trainer.stage1_pretrain(
            num_epochs=args.stage1_epochs,
            lr=args.stage1_lr,
            use_consistency=True,
            x0_switch_step=int(trainer.stage1_x0_warmup_steps),
            validation_interval=int(trainer.stage1_validation_interval),
            visualization_interval=int(trainer.stage1_visualization_interval),
            validation_mode=trainer.stage1_validation_mode,
            validation_max_batches=int(trainer.stage1_validation_max_batches),
            sample_path_interval=int(trainer.stage1_sample_path_interval),
            sample_path_weight=float(trainer.stage1_sample_path_weight),
        )
        
        if is_main:
            torch.save(
                trainer.raw_model.state_dict() if hasattr(trainer, "raw_model") else model.state_dict(),
                output_dir / "stage1_checkpoint.pt"
            )
    else:
        if is_main:
             print("\nSkipping Stage 1 (not in requested stages)")

    if run_stage2a:
        if sparse_alignment_train_loader is None and build_sparse_alignment_dataloaders is not None:
            sparse_alignment_train_loader, sparse_alignment_val_loader = build_sparse_alignment_dataloaders()
            trainer.sparse_alignment_train_dataloader = sparse_alignment_train_loader
            trainer.sparse_alignment_val_dataloader = sparse_alignment_val_loader
        if sparse_alignment_train_loader is not None:
            if is_main:
                print(
                    "[Stage2a] 切换到 sparse-view alignment dataloader: "
                    f"support_views={int(args.llff_alignment_sparse_views)}"
                )
            trainer.set_stage_dataloaders(
                sparse_alignment_train_loader,
                sparse_alignment_val_loader,
            )
            if str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
                expected_alignment_template_scope = _resolve_expected_support_template_scope(
                    support_template_stage_tag="stage2_alignment",
                    effective_num_sparse_views=int(args.llff_alignment_sparse_views),
                )
                _assert_dataloader_protocol(
                    dataloader=trainer.train_dataloader,
                    loader_name="trainer/stage2a/train",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
                if trainer.val_dataloader is not None:
                    _assert_dataloader_protocol(
                        dataloader=trainer.val_dataloader,
                        loader_name="trainer/stage2a/val",
                        expected_stage_tag="stage2_alignment",
                        expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                        expected_template_scope=expected_alignment_template_scope,
                        require_phase1_episode_bank=True,
                    )
        if is_main:
            print("\n" + "=" * 40)
            print("STAGE 2A: Sparse-View Alignment")
            print("=" * 40)
        trainer.stage2a_sparse_alignment(
            num_epochs=args.stage2a_epochs,
            lr=args.stage2a_lr,
            validation_interval=int(trainer.stage2a_validation_interval),
            visualization_interval=int(trainer.stage2a_visualization_interval),
        )
        if is_main:
            torch.save(
                trainer.raw_model.state_dict() if hasattr(trainer, "raw_model") else model.state_dict(),
                output_dir / "stage2a_checkpoint.pt"
            )
    elif sparse_alignment_train_loader is not None and (run_stage2 or run_stage3):
        if is_main:
            print(
                "[Stage2a] 未执行显式 alignment 训练，但将切换到 sparse-view alignment dataloader: "
                f"support_views={int(args.llff_alignment_sparse_views)}"
            )
        trainer.set_stage_dataloaders(
            sparse_alignment_train_loader,
            sparse_alignment_val_loader,
        )
        if str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
            expected_alignment_template_scope = _resolve_expected_support_template_scope(
                support_template_stage_tag="stage2_alignment",
                effective_num_sparse_views=int(args.llff_alignment_sparse_views),
            )
            _assert_dataloader_protocol(
                dataloader=trainer.train_dataloader,
                loader_name="trainer/stage2+/train",
                expected_stage_tag="stage2_alignment",
                expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                expected_template_scope=expected_alignment_template_scope,
                require_phase1_episode_bank=True,
            )
            if trainer.val_dataloader is not None:
                _assert_dataloader_protocol(
                    dataloader=trainer.val_dataloader,
                    loader_name="trainer/stage2+/val",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
        if is_main:
            print(
                "[Stage2a] 已跳过显式 sparse-view alignment 训练，"
                "但 Stage2/Stage3 将继续使用 sparse-view alignment dataloader。"
            )
    elif build_sparse_alignment_dataloaders is not None and (run_stage2 or run_stage3):
        sparse_alignment_train_loader, sparse_alignment_val_loader = build_sparse_alignment_dataloaders()
        trainer.sparse_alignment_train_dataloader = sparse_alignment_train_loader
        trainer.sparse_alignment_val_dataloader = sparse_alignment_val_loader
        if sparse_alignment_train_loader is not None:
            if is_main:
                print(
                    "[Stage2a] 未执行显式 alignment 训练，但已按需切换到 sparse-view alignment dataloader: "
                    f"support_views={int(args.llff_alignment_sparse_views)}"
                )
            trainer.set_stage_dataloaders(
                sparse_alignment_train_loader,
                sparse_alignment_val_loader,
            )
            if str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
                expected_alignment_template_scope = _resolve_expected_support_template_scope(
                    support_template_stage_tag="stage2_alignment",
                    effective_num_sparse_views=int(args.llff_alignment_sparse_views),
                )
                _assert_dataloader_protocol(
                    dataloader=trainer.train_dataloader,
                    loader_name="trainer/stage2+/train",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
                if trainer.val_dataloader is not None:
                    _assert_dataloader_protocol(
                        dataloader=trainer.val_dataloader,
                        loader_name="trainer/stage2+/val",
                        expected_stage_tag="stage2_alignment",
                        expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                        expected_template_scope=expected_alignment_template_scope,
                        require_phase1_episode_bank=True,
                    )
            if is_main:
                print(
                    "[Stage2a] 已跳过显式 sparse-view alignment 训练，"
                    "但 Stage2/Stage3 将继续使用 sparse-view alignment dataloader。"
                )
    
    # Stage 2: Uncertainty calibration
    if run_stage2:
        if args.llff_multiscene_root and str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
            expected_alignment_template_scope = _resolve_expected_support_template_scope(
                support_template_stage_tag="stage2_alignment",
                effective_num_sparse_views=int(args.llff_alignment_sparse_views),
            )
            _assert_dataloader_protocol(
                dataloader=trainer.train_dataloader,
                loader_name="stage2/train",
                expected_stage_tag="stage2_alignment",
                expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                expected_template_scope=expected_alignment_template_scope,
                require_phase1_episode_bank=True,
            )
            if trainer.val_dataloader is not None:
                _assert_dataloader_protocol(
                    dataloader=trainer.val_dataloader,
                    loader_name="stage2/val",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
        if is_main:
            print("\n" + "=" * 40)
            print("STAGE 2: Uncertainty Calibration")
            print("=" * 40)
        
        trainer.stage2_calibrate(
            num_epochs=args.stage2_epochs,
            lr=args.stage2_lr,
        )
        
        if is_main:
            torch.save(
                trainer.raw_model.state_dict() if hasattr(trainer, "raw_model") else model.state_dict(),
                output_dir / "stage2_checkpoint.pt"
            )
    else:
        if is_main:
             print("\nSkipping Stage 2 (not in requested stages)")
    
    # Stage 3: Joint fine-tuning
    if run_stage3:
        if args.llff_multiscene_root and str(args.offline_condition_source).strip().lower() == "phase1_3dgs":
            expected_alignment_template_scope = _resolve_expected_support_template_scope(
                support_template_stage_tag="stage2_alignment",
                effective_num_sparse_views=int(args.llff_alignment_sparse_views),
            )
            _assert_dataloader_protocol(
                dataloader=trainer.train_dataloader,
                loader_name="stage3/train",
                expected_stage_tag="stage2_alignment",
                expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                expected_template_scope=expected_alignment_template_scope,
                require_phase1_episode_bank=True,
            )
            if trainer.val_dataloader is not None:
                _assert_dataloader_protocol(
                    dataloader=trainer.val_dataloader,
                    loader_name="stage3/val",
                    expected_stage_tag="stage2_alignment",
                    expected_num_sparse_views=int(args.llff_alignment_sparse_views),
                    expected_template_scope=expected_alignment_template_scope,
                    require_phase1_episode_bank=True,
                )
        if is_main:
            print("\n" + "=" * 40)
            print("STAGE 3: Joint Fine-tuning")
            print("=" * 40)
        
        trainer.stage3_finetune(
            num_epochs=args.stage3_epochs,
            lr=1e-6,  # 修复: 降低学习率防止loss爆炸
            lambda_distill=0.01,  # 修复: 降低一致性损失权重
            recon_weight=1.0,  # 修复: 确保重建损失为主
        )
        
        if is_main:
            torch.save(
                trainer.raw_model.state_dict() if hasattr(trainer, "raw_model") else model.state_dict(),
                output_dir / "final_checkpoint.pt"
            )
    else:
         if is_main:
             print("\nSkipping Stage 3 (not in requested stages)")

    if is_main and stages_to_run:
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Checkpoints saved to: {output_dir}")
        print("=" * 60)
    
    # Cleanup
    if writer:
        writer.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
