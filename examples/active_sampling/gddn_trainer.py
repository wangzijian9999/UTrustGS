"""
Three-stage trainer for the GDDN diffusion model.

This module follows the staged procedure discussed in the project notes:
    1. Stage 1: optimise RGB generation quality under multi-modal conditioning.
    2. Stage 2: freeze the backbone and train epistemic / aleatoric uncertainty
       heads using MC-Dropout supervision.
    3. Stage 3: jointly fine-tune the entire model with a small learning rate and
       an uncertainty-consistency regulariser to keep uncertainties calibrated.

The trainer is intentionally lightweight and focuses on clarity rather than
production-ready efficiency. Users may extend or adapt it for custom pipelines.
"""

from __future__ import annotations

import json
import os
import math
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS = os.environ.get(
    "GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS",
    "",
).strip().lower() in {"1", "true", "yes", "on"}

if _SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS:
    Image = None  # type: ignore
else:
    try:  # pragma: no cover - optional dependency
        from PIL import Image
    except Exception:  # pragma: no cover - optional dependency
        Image = None  # type: ignore

if _SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS:
    SummaryWriter = None  # type: ignore
else:
    try:  # pragma: no cover - optional dependency
        from torch.utils.tensorboard import SummaryWriter
    except Exception:  # pragma: no cover - tensorboard not installed
        SummaryWriter = None  # type: ignore

import torch
import torch.nn.functional as F
import torch.distributed as dist

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable, *args, **kwargs):
        return iterable

# 可选依赖：用于感知损失的VGG特征
if _SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS:
    torchvision = None  # type: ignore
    vgg19 = None  # type: ignore
    VGG19_Weights = None  # type: ignore
else:
    try:  # pragma: no cover - optional dependency
        import torchvision
        from torchvision.models import vgg19, VGG19_Weights
    except Exception:  # pragma: no cover - optional dependency
        torchvision = None  # type: ignore
        vgg19 = None  # type: ignore
        VGG19_Weights = None  # type: ignore

# 可选依赖：用于感知质量评估的LPIPS
if _SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS:
    lpips = None  # type: ignore
    LPIPS_AVAILABLE = False
else:
    try:
        import lpips
        LPIPS_AVAILABLE = True
    except ImportError:
        lpips = None  # type: ignore
        LPIPS_AVAILABLE = False


FIXED_RUNTIME_PREDECODE_METRIC_NAMES: Tuple[str, ...] = (
    "predecode_initial_latent_overall_abs_mean",
    "predecode_initial_latent_overall_variance",
    "predecode_controlnet_condition_abs_mean",
    "predecode_controlnet_condition_variance",
    "predecode_start_latent_overall_abs_mean",
    "predecode_start_latent_overall_variance",
    "predecode_start_latent_support_abs_mean",
    "predecode_start_latent_editable_abs_mean",
    "predecode_start_latent_novel_abs_mean",
    "predecode_final_latent_overall_abs_mean",
    "predecode_final_latent_overall_variance",
    "predecode_final_latent_support_abs_mean",
    "predecode_final_latent_editable_abs_mean",
    "predecode_final_latent_novel_abs_mean",
    "predecode_controlnet_down_residual_abs_mean_avg",
    "predecode_controlnet_mid_residual_abs_mean_avg",
    "predecode_cfg_noise_delta_abs_mean_avg",
    "predecode_cfg_noise_delta_variance_avg",
    "predecode_scheduler_step_delta_abs_mean_avg",
    "predecode_gcd_delta_abs_mean_avg",
    "predecode_gcd_delta_novel_abs_mean_avg",
    "predecode_cags_delta_abs_mean_avg",
    "predecode_cags_delta_exist_abs_mean_avg",
    "predecode_projection_delta_abs_mean_avg",
    "predecode_projection_delta_support_abs_mean_avg",
)

FIXED_RUNTIME_SEQUENCE_DIAGNOSTIC_NAMES: Tuple[str, ...] = (
    "per_view_valid_coverage",
    "per_view_distance_weight",
    "per_view_raw_inverse_distance_weight",
    "per_view_coverage_distance_weight",
    "per_view_fusion_distance_weight",
)

FIXED_RUNTIME_COMPACT_LOG_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("fixed_sample_psnr", "sample_psnr", "psnr"),
    ("fixed_runtime_pre_compose_teacher_masked_inpaint_editable_psnr", "pre_compose_editable_psnr", "psnr"),
    ("fixed_runtime_pre_compose_teacher_novel_psnr", "pre_compose_novel_psnr", "psnr"),
    ("fixed_runtime_support_projection_coverage", "support_projection_coverage", "ratio"),
    ("fixed_runtime_masked_inpaint_editable_coverage", "editable_coverage", "ratio"),
    ("fixed_runtime_novel_coverage", "novel_coverage", "ratio"),
    ("fixed_runtime_predecode_controlnet_down_residual_abs_mean_avg", "predecode_down_residual_avg", "ratio"),
    ("fixed_runtime_predecode_controlnet_mid_residual_abs_mean_avg", "predecode_mid_residual_avg", "ratio"),
    ("fixed_runtime_predecode_cfg_noise_delta_abs_mean_avg", "predecode_cfg_delta_avg", "ratio"),
    ("fixed_runtime_predecode_scheduler_step_delta_abs_mean_avg", "predecode_scheduler_delta_avg", "ratio"),
    ("fixed_runtime_predecode_gcd_delta_abs_mean_avg", "predecode_gcd_delta_avg", "ratio"),
    ("fixed_runtime_predecode_cags_delta_abs_mean_avg", "predecode_cags_delta_avg", "ratio"),
    ("fixed_runtime_predecode_projection_delta_abs_mean_avg", "predecode_projection_delta_avg", "ratio"),
)

FIXED_RUNTIME_DEBUG_TENSOR_NAMES: Tuple[str, ...] = (
    "warped_rgb",
    "target_depth_proxy",
    "warp_depth",
    "warp_alpha",
    "target_alpha",
    "proposal_depth",
    "proposal_alpha",
    "proposal_raw_rgb",
    "decoded_teacher_rgb",
    "support_composed_image",
    "support_projection_mask",
    "masked_inpaint_editable_mask",
    "predecode_controlnet_condition",
    "predecode_start_latent_rgb",
    "predecode_step_start_rgb",
    "predecode_step_mid_rgb",
    "predecode_step_supportswitch_rgb",
)

class ThreeStageTrainer:
    """Utility class that orchestrates the staged training of GDDN."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: Iterable[Dict[str, torch.Tensor]],
        val_dataloader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
        device: str = "cuda",
        writer: Optional["SummaryWriter"] = None,
        world_rank: int = 0,
        world_size: int = 1,
        log_dir: Optional[str] = None,
    ) -> None:
        self.device = torch.device(device)
        # 检查是否已启用多GPU，避免覆盖设备分配
        raw = model.module if hasattr(model, "module") else model
        if getattr(raw, "_multi_gpu_enabled", False):
            self.model = model  # 保持多GPU设备分配
        else:
            self.model = model.to(self.device)
        # 如果是 DDP 模型，获取原始模型引用以访问自定义方法/属性
        self.raw_model = self.model.module if hasattr(self.model, "module") else self.model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.sparse_alignment_train_dataloader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        self.sparse_alignment_val_dataloader: Optional[Iterable[Dict[str, torch.Tensor]]] = None
        self.writer = writer
        self.world_rank = world_rank
        self.world_size = world_size
        self._distributed = (
            self.world_size > 1 and dist.is_available() and dist.is_initialized()
        )
        self._is_main_rank = self.world_rank == 0

        # PNG文件保存目录
        self.log_dir: Optional[Path] = Path(log_dir) if log_dir is not None else None
        self.stage_samples_dir: Optional[Path] = None
        if self.log_dir is not None and self._is_main_rank:
            self.stage_samples_dir = self.log_dir / "stage_samples"
            self.stage_samples_dir.mkdir(parents=True, exist_ok=True)
        self._fixed_visualization_batch_cpu: Optional[Dict[str, Any]] = None
        self._fixed_visualization_batch_source: Optional[str] = None
        self._fixed_visualization_batches_cpu: Optional[List[Dict[str, Any]]] = None
        self._fixed_visualization_batch_sources: Optional[List[str]] = None
        self.fixed_visualization_sample_index: int = 0
        self.fixed_visualization_sample_indices: List[int] = [0]
        self.stage1_validation_interval: int = 1
        self.stage1_visualization_interval: int = 1
        self.stage2a_validation_interval: int = 1
        self.stage2a_visualization_interval: int = 1
        self.stage1_x0_warmup_steps: int = 0
        self.stage2a_x0_warmup_steps: int = 0
        self.stage1_validation_mode: str = "full_generate"
        self.stage2a_validation_mode: str = "full_generate"
        self.stage1_validation_max_batches: int = 0
        self.stage2a_validation_max_batches: int = 0
        self.stage1_sample_path_interval: int = 5
        self.stage2a_sample_path_interval: int = 5
        self.stage1_sample_path_weight: float = 1.0
        self.stage2a_sample_path_weight: float = 1.0
        self.stage1_aux_batch_chunk_size: int = 0
        self.stage2a_aux_batch_chunk_size: int = 0
        self.early_stop_patience: int = 0
        self.early_stop_min_epochs: int = 0
        self.early_stop_min_delta: float = 0.0
        self.verbose_epoch_stats: bool = False
        self._target_latent_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        self._target_latent_cache_logged: bool = False
        self.where_wrong_last_result: Optional[Dict[str, Any]] = None
        
        self._amp_dtype: torch.dtype = self._detect_model_dtype()
        # 强制禁用AMP混合精度以避免Stage1 NaN问题（FP16精度不足导致梯度溢出）
        self._amp_enabled: bool = False
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.sampler_type: str = "ddim"
        self.stage2_mc_samples: int = 8  # 增加采样次数提升不确定性估计质量
        self.stage3_mc_samples: int = 8
        self.stage2_diffusion_steps: int = 20
        self.stage3_diffusion_steps: int = 20
        self.distill_interval: int = 10
        self.lambda_calib: float = 0.2  # Stage3校准损失权重（增加到0.2以防止ECE恶化）
        self.lambda_var_preserve: float = 0.1  # Variance preservation 正则化权重
        self.min_pred_var: float = 0.01  # 最小预测方差阈值
        self._stage3_step: int = 0
        self._mc_dropout_nan_logged: bool = False
        self.stage3_base_lr: float = 0.0
        self.stage3_final_lr: Optional[float] = None
        self.stage3_final_epochs: int = 0
        self.unfreeze_vae: bool = False
        self.unfreeze_vae_decoder_only: bool = True
        self.stage3_vae_lr_scale: float = 0.2
        self.auto_unfreeze_vae: bool = False
        self.auto_unfreeze_decoder_only: bool = True
        self.auto_unfreeze_ece_threshold: float = 0.05
        self.auto_unfreeze_corr_threshold: float = 0.2
        self.auto_unfreeze_psnr_delta: float = 0.1
        self.auto_unfreeze_warmup_epochs: int = 6
        self.auto_unfreeze_patience: int = 3
        self.auto_unfreeze_psnr_window: int = 5
        self._auto_unfreeze_candidates: List[torch.nn.Parameter] = []
        self._auto_unfreeze_activated: bool = False
        self._stage3_metric_history: List[Dict[str, float]] = []
        self._stage3_psnr_history: List[float] = []
        self.stage3_recon_weight: float = 0.3  # 增加6倍，强化重建目标
        self.stage3_recon_alpha: float = 0.6
        self.stage3_perceptual_weight: float = 0.05
        self.stage3_vae_weight_decay: float = 1e-4
        self.stage3_downscale_factor: float = 1.0
        self.stage3_min_hw: int = 256
        self.stage3_variance_weight: float = 0.0
        self.stage3_variance_target: float = 1.0
        self.stage3_variance_source: str = "noise"
        self.scheduler_overrides: Optional[Dict[str, Any]] = None
        self.scheduler_eta: Optional[float] = None
        self.supports_native_masked_inpainting: bool = bool(
            getattr(self.raw_model, "enable_true_masked_latent_inpainting", False)
        )
        self.supports_mask_aware_fallback_sampling: bool = bool(
            getattr(self.raw_model, "supports_mask_aware_fallback_sampling", False)
        )
        self.supports_sample_centric_training: bool = bool(
            self.supports_native_masked_inpainting
            or self.supports_mask_aware_fallback_sampling
        )
        self.sample_path_primary_alignment_enabled: bool = bool(
            self.supports_sample_centric_training
        )
        self.sample_path_supervision_priority_weight: float = (
            1.50 if self.supports_native_masked_inpainting else (
                1.25 if self.supports_mask_aware_fallback_sampling else 1.0
            )
        )
        self.sample_path_surrogate_reconstruction_decay: float = (
            0.45 if self.supports_sample_centric_training else 1.0
        )
        self.use_final_sample_for_teacher_effect_targets: bool = (
            self.supports_sample_centric_training
        )
        self.exist_diffusion_weight: float = 0.15
        self.novel_diffusion_weight: float = 1.0
        self.exist_identity_weight: float = 1.0
        self.novel_reconstruction_weight: float = 1.0
        self.stage2_final_sample_target_interval: int = 1
        self.stage2_final_sample_target_steps: int = 20
        self.final_sample_supervision_interval: int = (
            1 if self.supports_native_masked_inpainting else (
                1 if self.supports_mask_aware_fallback_sampling else 8
            )
        )
        self.final_sample_supervision_steps: int = 15
        self.final_sample_supervision_weight: float = (
            0.55 if self.supports_native_masked_inpainting else (
                0.45 if self.supports_mask_aware_fallback_sampling else 0.20
            )
        )
        self.final_sample_supervision_max_weight: float = (
            0.85 if self.supports_native_masked_inpainting else (
                0.70 if self.supports_mask_aware_fallback_sampling else 0.35
            )
        )
        self.final_sample_support_fidelity_weight: float = 0.0
        self.final_sample_support_consistency_weight: float = 0.0
        self.final_sample_support_confidence_boost: float = 0.0
        self.final_sample_support_low_confidence_anchor_weight: float = 0.0
        self.relative_teacher_effect_weight: Optional[float] = None
        self.teacher_warp_margin_weight: float = 0.0
        self.teacher_warp_margin: float = 0.0
        self.teacher_warp_editable_weight: float = 1.0
        self.stage2a_frontier_residual_weight: float = 0.0
        self.stage3_frontier_residual_weight: float = 0.0
        self.frontier_residual_anchor_weight: float = 0.0
        self.frontier_residual_recon_alpha: float = 0.75
        self.proposal_confidence_supervision_weight: float = 0.0
        self.proposal_warp_error_supervision_weight: float = 0.0
        self.proposal_repairability_supervision_weight: float = 0.0
        self.proposal_verification_supervision_weight: float = 0.0
        self.proposal_acceptance_supervision_weight: float = 0.0
        self.proposal_confidence_temperature: float = 0.05
        self.proposal_opportunity_error_threshold: float = 0.0
        self.proposal_opportunity_top_fraction: float = 1.0
        self.proposal_repair_prior_quantile: float = 0.90
        self.proposal_verification_min_support: float = 0.05
        self.proposal_repairability_empirical_max_weight: float = 0.0
        self.proposal_repairability_empirical_transition_epochs: int = 1
        self.proposal_empirical_patch_kernel_size: int = 15
        self.proposal_empirical_candidate_count: int = 1
        self.proposal_empirical_candidate_mc_dropout_p: float = 0.15
        self.proposal_empirical_disable_hard_gate: bool = True
        self.proposal_empirical_include_decoded_teacher: bool = True
        self.proposal_empirical_include_local_repair_candidates: bool = True
        self.proposal_empirical_local_repair_kernel_sizes: Tuple[int, ...] = (9, 21, 41)
        self.proposal_head_positive_balance_max: float = 8.0
        self.proposal_warp_error_rank_blend: float = 0.50
        self.proposal_head_mean_calibration_weight: float = 0.25
        self.proposal_selective_risk_weight: float = 1.0
        self.stage2a_localization_phase_ratio: float = 0.0
        self.stage2a_localization_sample_path_weight: float = 0.0
        self.stage2a_residual_sample_path_weight: float = 0.25
        self.stage2a_episode_novel_reconstruction_scale: float = 0.25
        self.proposal_validation_acceptance_threshold: float = 0.50
        self.stage2a_surrogate_reconstruction_decay: float = 1.0
        self.stage3_surrogate_reconstruction_decay: float = 1.0
        self.stage2a_exist_diffusion_weight: Optional[float] = None
        self.stage2a_novel_diffusion_weight: Optional[float] = None
        self.stage3_exist_diffusion_weight: Optional[float] = None
        self.stage3_novel_diffusion_weight: Optional[float] = None
        self.validation_use_warp_delta_score: bool = False
        self.geometry_aware_frontier_proposal_focus: bool = False
        self.geometry_aware_frontier_kernel_size: int = 31
        self.geometry_aware_frontier_support_threshold: float = 0.05
        self.stage1_trainable_sample_steps: int = 8
        self.trainable_sample_supervision_interval: int = (
            2 if self.supports_native_masked_inpainting else (
                2 if self.supports_mask_aware_fallback_sampling else 0
            )
        )
        self.trainable_sample_supervision_steps: int = 8
        self.trainable_sample_supervision_weight: float = (
            0.35 if self.supports_native_masked_inpainting else (
                0.22 if self.supports_mask_aware_fallback_sampling else 0.0
            )
        )
        self.trainable_sample_supervision_max_weight: float = (
            0.55 if self.supports_native_masked_inpainting else (
                0.38 if self.supports_mask_aware_fallback_sampling else 0.0
            )
        )
        self.final_sample_large_hole_ratio_threshold: float = 0.55
        self.final_sample_medium_hole_step_boost: int = 4
        self.final_sample_large_hole_step_boost: int = 8
        self.stage2_primary_metric_use_final_sample: bool = True
        self.stage3_primary_metric_use_final_sample: bool = True
        self.stage3_adaptive_controlnet_freeze: bool = True
        self.stage3_freeze_controlnet_min_support_coverage: float = 0.45
        self._stage3_freeze_controlnet: bool = True
        self.stage3_force_controlnet_trainable: bool = False
        self.geometry_aware_trainable_name_tokens: Tuple[str, ...] = ()
        self.sample_masked_dual_path_support_step_offset_override: Optional[int] = None
        if self._is_main_rank and not self.supports_native_masked_inpainting:
            print(
                "[ThreeStageTrainer] 当前底座非inpainting checkpoint，"
                "已启用 mask-aware fallback 的 sample-centric 监督模式。"
            )
        
        # ========== 创新算法参数 ==========
        # VPDT: 方差保持扩散训练（暂时禁用以确保基础训练稳定）
        self.lambda_vpdt: float = 0.0  # 完全禁用VPDT损失
        self.vpdt_target_std: float = 1.0  # 噪声预测的目标标准差
        
        # IBC: 交替双层校准
        self.ibc_enabled: bool = True  # 是否启用IBC
        self.ibc_inner_steps: int = 2  # 内层校准迭代次数（减少以节省显存）
        self.ibc_inner_lr: float = 1e-4  # 内层校准学习率
        
        # Stage1训练配置
        self.stage1_freeze_up_blocks_3: bool = False  # 是否冻结up_blocks.3层（FP32下无需冻结）
        self.enable_min_snr_weighting: bool = True
        self.min_snr_gamma: float = 5.0
        self._last_diffusion_prediction_type: str = "epsilon"
        self._last_diffusion_supervision_label: str = "noise"
        self.sample_guidance_scale: float = float(
            getattr(self.raw_model, "default_guidance_scale", 1.0)
        )
        self.sample_disable_latent_blending: bool = False
        self.sample_inpaint_img2img_strength_override: Optional[float] = None
        self.default_sample_output_mode: str = "composed"
        self.episode_conditioned_sample_output_mode: str = "proposal_raw"



    def _reset_fixed_visualization_cache(self) -> None:
        self._fixed_visualization_batch_cpu = None
        self._fixed_visualization_batch_source = None
        self._fixed_visualization_batches_cpu = None
        self._fixed_visualization_batch_sources = None

    def set_stage_dataloaders(
        self,
        train_dataloader: Iterable[Dict[str, torch.Tensor]],
        val_dataloader: Optional[Iterable[Dict[str, torch.Tensor]]] = None,
    ) -> None:
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self._reset_fixed_visualization_cache()
        if self._is_main_rank:
            def _describe_loader_protocol(
                dataloader: Optional[Iterable[Dict[str, torch.Tensor]]],
                loader_name: str,
            ) -> Optional[str]:
                if dataloader is None:
                    return None
                dataset = getattr(dataloader, "dataset", None)
                support_template_stage_tag = getattr(
                    dataloader,
                    "support_template_stage_tag",
                    getattr(dataset, "support_template_stage_tag", "unknown"),
                )
                support_template_scope = getattr(
                    dataloader,
                    "support_template_scope",
                    getattr(dataset, "support_template_scope", "unknown"),
                )
                effective_num_sparse_views = getattr(
                    dataloader,
                    "effective_num_sparse_views",
                    getattr(dataset, "effective_num_sparse_views", -1),
                )
                use_phase1_3dgs_conditions = getattr(
                    dataloader,
                    "use_phase1_3dgs_conditions",
                    getattr(dataset, "use_phase1_3dgs_conditions", False),
                )
                hard_episode_top_fraction = getattr(
                    dataloader,
                    "hard_episode_top_fraction",
                    getattr(dataset, "hard_episode_top_fraction", 1.0),
                )
                hard_episode_repeat_factor = getattr(
                    dataloader,
                    "hard_episode_repeat_factor",
                    getattr(dataset, "hard_episode_repeat_factor", 1),
                )
                dataset_length = len(dataset) if dataset is not None and hasattr(dataset, "__len__") else -1
                return (
                    f"{loader_name}: stage_tag={support_template_stage_tag}, "
                    f"template_scope={support_template_scope}, "
                    f"support_views={int(effective_num_sparse_views)}, "
                    f"phase1_episode_bank={bool(use_phase1_3dgs_conditions)}, "
                    f"hard_top_fraction={float(hard_episode_top_fraction):.2f}, "
                    f"hard_repeat={int(hard_episode_repeat_factor)}, "
                    f"dataset_len={int(dataset_length)}"
                )

            loader_descriptions = []
            train_loader_description = _describe_loader_protocol(train_dataloader, "train")
            if train_loader_description is not None:
                loader_descriptions.append(train_loader_description)
            val_loader_description = _describe_loader_protocol(val_dataloader, "val")
            if val_loader_description is not None:
                loader_descriptions.append(val_loader_description)
            if loader_descriptions:
                print("[Trainer] 当前阶段 dataloader 协议: " + " | ".join(loader_descriptions))

    def _resolve_sample_output_mode(
        self,
        batch: Dict[str, Any],
    ) -> str:
        if self._batch_uses_phase1_3dgs_conditions(batch):
            return str(self.episode_conditioned_sample_output_mode or "proposal_raw")
        return str(self.default_sample_output_mode or "composed")

    def _resolve_frontier_proposal_objective_config(
        self,
        stage_tag: str,
    ) -> Dict[str, float]:
        normalized_stage_tag = str(stage_tag).strip().lower()
        if "stage2a" in normalized_stage_tag:
            return {
                "frontier_residual_weight": max(
                    float(getattr(self, "stage2a_frontier_residual_weight", 0.0)),
                    0.0,
                ),
                "surrogate_reconstruction_decay": min(
                    max(
                        float(getattr(self, "stage2a_surrogate_reconstruction_decay", 1.0)),
                        0.0,
                    ),
                    1.0,
                ),
                "exist_diffusion_weight": (
                    None
                    if getattr(self, "stage2a_exist_diffusion_weight", None) is None
                    else max(float(getattr(self, "stage2a_exist_diffusion_weight", 0.0)), 0.0)
                ),
                "novel_diffusion_weight": (
                    None
                    if getattr(self, "stage2a_novel_diffusion_weight", None) is None
                    else max(float(getattr(self, "stage2a_novel_diffusion_weight", 0.0)), 0.0)
                ),
            }
        if "stage3" in normalized_stage_tag:
            return {
                "frontier_residual_weight": max(
                    float(getattr(self, "stage3_frontier_residual_weight", 0.0)),
                    0.0,
                ),
                "surrogate_reconstruction_decay": min(
                    max(
                        float(getattr(self, "stage3_surrogate_reconstruction_decay", 1.0)),
                        0.0,
                    ),
                    1.0,
                ),
                "exist_diffusion_weight": (
                    None
                    if getattr(self, "stage3_exist_diffusion_weight", None) is None
                    else max(float(getattr(self, "stage3_exist_diffusion_weight", 0.0)), 0.0)
                ),
                "novel_diffusion_weight": (
                    None
                    if getattr(self, "stage3_novel_diffusion_weight", None) is None
                    else max(float(getattr(self, "stage3_novel_diffusion_weight", 0.0)), 0.0)
                ),
            }
        return {
            "frontier_residual_weight": 0.0,
            "surrogate_reconstruction_decay": 1.0,
            "exist_diffusion_weight": None,
            "novel_diffusion_weight": None,
        }

    def configure_geometry_aware_trainable_filter(
        self,
        name_tokens: Optional[Sequence[str]],
        *,
        include_uncertainty: bool = False,
    ) -> None:
        normalized_name_tokens = tuple(
            str(name_token).strip()
            for name_token in (name_tokens or [])
            if str(name_token).strip()
        )
        self.geometry_aware_trainable_name_tokens = normalized_name_tokens
        if normalized_name_tokens:
            self._apply_geometry_aware_trainable_filter(
                include_uncertainty=include_uncertainty,
            )

    def _parameter_matches_geometry_aware_filter(self, parameter_name: str) -> bool:
        allowed_name_tokens = tuple(getattr(self, "geometry_aware_trainable_name_tokens", ()) or ())
        if not allowed_name_tokens:
            return True
        return any(name_token in parameter_name for name_token in allowed_name_tokens)

    def _apply_geometry_aware_trainable_filter(
        self,
        *,
        include_uncertainty: bool,
    ) -> None:
        allowed_name_tokens = tuple(getattr(self, "geometry_aware_trainable_name_tokens", ()) or ())
        if not allowed_name_tokens:
            return
        for parameter_name, parameter in self.model.named_parameters():
            is_uncertainty_parameter = any(
                keyword in parameter_name for keyword in ("epistemic", "aleatoric", "uncertainty")
            )
            parameter.requires_grad = bool(
                self._parameter_matches_geometry_aware_filter(parameter_name)
                or (include_uncertainty and is_uncertainty_parameter)
            )

    def _detect_model_dtype(self) -> torch.dtype:
        for param in self.model.parameters():
            return param.dtype
        return torch.float32

    def _resolve_auxiliary_loss_device(self) -> torch.device:
        auxiliary_device = getattr(self.raw_model, "_auxiliary_loss_device", None)
        if isinstance(auxiliary_device, torch.device):
            return auxiliary_device
        decode_device = getattr(self.raw_model, "_vae_decode_device", None)
        if isinstance(decode_device, torch.device):
            return decode_device
        if isinstance(self.device, torch.device):
            return self.device
        return torch.device(self.device)

    def _resolve_stage_auxiliary_batch_chunk_size(
        self,
        *,
        stage_tag: str,
        batch_size: int,
        configured_chunk_size: Optional[int] = None,
    ) -> int:
        normalized_stage_tag = str(stage_tag).strip().lower()
        if configured_chunk_size is None:
            if normalized_stage_tag == "stage2a":
                configured_chunk_size = getattr(self, "stage2a_aux_batch_chunk_size", 0)
            else:
                configured_chunk_size = getattr(self, "stage1_aux_batch_chunk_size", 0)
        resolved_chunk_size = int(configured_chunk_size or 0)
        if resolved_chunk_size > 0:
            return max(1, min(int(batch_size), resolved_chunk_size))
        if self._amp_enabled:
            return 1
        if int(batch_size) <= 2:
            return max(1, int(batch_size))
        return max(1, min(int(batch_size), 2))

    def _slice_batch_along_batch_dim(
        self,
        batch: Dict[str, Any],
        *,
        batch_start_index: int,
        batch_end_index: int,
        reference_batch_size: int,
    ) -> Dict[str, Any]:
        if batch_start_index <= 0 and batch_end_index >= int(reference_batch_size):
            return batch
        sliced_batch: Dict[str, Any] = {}
        for key, value in batch.items():
            if (
                isinstance(value, torch.Tensor)
                and value.dim() > 0
                and int(value.shape[0]) == int(reference_batch_size)
            ):
                sliced_batch[key] = value[batch_start_index:batch_end_index]
            else:
                sliced_batch[key] = value
        return sliced_batch

    def _resolve_training_noise_scheduler(self) -> Optional[Any]:
        scheduler = getattr(self.raw_model, "noise_scheduler", None)
        if scheduler is not None:
            return scheduler
        return getattr(self.raw_model, "scheduler", None)

    def _resolve_scheduler_prediction_type(
        self,
        scheduler: Optional[Any] = None,
    ) -> str:
        resolved_scheduler = scheduler if scheduler is not None else self._resolve_training_noise_scheduler()
        if resolved_scheduler is None:
            return "epsilon"
        scheduler_config = getattr(resolved_scheduler, "config", None)
        prediction_type = getattr(scheduler_config, "prediction_type", None)
        if prediction_type is None and isinstance(scheduler_config, dict):
            prediction_type = scheduler_config.get("prediction_type")
        if prediction_type is None:
            return "epsilon"
        return str(prediction_type)

    def _get_scheduler_alphas_cumprod(
        self,
        *,
        scheduler: Optional[Any],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        resolved_scheduler = scheduler if scheduler is not None else self._resolve_training_noise_scheduler()
        if resolved_scheduler is None or not hasattr(resolved_scheduler, "alphas_cumprod"):
            return None
        alpha_values = resolved_scheduler.alphas_cumprod
        if not torch.is_tensor(alpha_values):
            alpha_values = torch.tensor(alpha_values, device=device, dtype=dtype)
        else:
            alpha_values = alpha_values.to(device=device, dtype=dtype)
        return alpha_values

    def _compute_diffusion_supervision_target(
        self,
        *,
        target_latent: torch.Tensor,
        sampled_noise: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: Optional[Any] = None,
    ) -> torch.Tensor:
        prediction_type = self._resolve_scheduler_prediction_type(scheduler)
        self._last_diffusion_prediction_type = prediction_type
        alpha_values = self._get_scheduler_alphas_cumprod(
            scheduler=scheduler,
            device=target_latent.device,
            dtype=target_latent.dtype,
        )
        if alpha_values is None:
            self._last_diffusion_supervision_label = "noise"
            return sampled_noise

        step_indices = timesteps.to(device=target_latent.device, dtype=torch.long)
        if step_indices.ndim == 0:
            step_indices = step_indices.unsqueeze(0)
        max_index = alpha_values.shape[0] - 1
        step_indices = step_indices.clamp_min(0).clamp_max(max_index)
        alpha_t = alpha_values.index_select(0, step_indices).view(-1, 1, 1, 1)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-12))

        if prediction_type == "v_prediction":
            self._last_diffusion_supervision_label = "v_prediction"
            return sqrt_alpha_t * sampled_noise - sqrt_one_minus_alpha_t * target_latent
        if prediction_type == "sample":
            self._last_diffusion_supervision_label = "sample"
            return target_latent

        self._last_diffusion_supervision_label = "noise"
        return sampled_noise

    def _predict_x0_from_model_output(
        self,
        *,
        noisy_latents: torch.Tensor,
        model_output: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler: Optional[Any] = None,
        alpha_fallback: float = 0.99,
    ) -> torch.Tensor:
        alpha_values = self._get_scheduler_alphas_cumprod(
            scheduler=scheduler,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        if alpha_values is not None:
            step_indices = timesteps.to(device=noisy_latents.device, dtype=torch.long)
            if step_indices.ndim == 0:
                step_indices = step_indices.unsqueeze(0)
            max_index = alpha_values.shape[0] - 1
            step_indices = step_indices.clamp_min(0).clamp_max(max_index)
            alpha_t = alpha_values.index_select(0, step_indices).view(-1, 1, 1, 1)
        else:
            alpha_t = torch.full(
                (noisy_latents.shape[0], 1, 1, 1),
                float(alpha_fallback),
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            )

        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-12))
        prediction_type = self._resolve_scheduler_prediction_type(scheduler)
        aligned_model_output = model_output.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        if prediction_type == "v_prediction":
            return sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * aligned_model_output
        if prediction_type == "sample":
            return aligned_model_output
        return (noisy_latents - sqrt_one_minus_alpha_t * aligned_model_output) / sqrt_alpha_t

    def _compute_render_only_psnr_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        render_rgb = batch["rgb_render"]
        target_rgb = batch["target_rgb"]
        if render_rgb.shape[-2:] != target_rgb.shape[-2:]:
            render_rgb = F.interpolate(
                render_rgb,
                size=target_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        return self._compute_psnr_from_rgb(render_rgb, target_rgb)

    def _get_support_anchor_rgb_from_batch(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        support_anchor_rgb = batch.get("support_anchor_rgb")
        if isinstance(support_anchor_rgb, torch.Tensor):
            return support_anchor_rgb
        render_rgb = batch.get("rgb_render")
        if isinstance(render_rgb, torch.Tensor):
            return render_rgb
        return None

    def _batch_uses_phase1_3dgs_conditions(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> bool:
        residual_refiner_flag = batch.get("use_3dgs_residual_refiner")
        if isinstance(residual_refiner_flag, torch.Tensor) and bool(
            (residual_refiner_flag > 0.5).any().item()
        ):
            return True
        episode_conditioned_flag = batch.get("phase1_episode_conditioned")
        if isinstance(episode_conditioned_flag, torch.Tensor) and bool(
            (episode_conditioned_flag > 0.5).any().item()
        ):
            return True
        phase1_source_tag = batch.get("phase1_condition_source_tag")
        if isinstance(phase1_source_tag, str):
            normalized_source_tag = phase1_source_tag.strip().lower()
            return any(
                token in normalized_source_tag
                for token in ("phase1", "3dgs", "geometry_bank", "episode")
            )
        if isinstance(phase1_source_tag, (list, tuple)):
            for source_tag_item in phase1_source_tag:
                if not isinstance(source_tag_item, str):
                    continue
                normalized_source_tag = source_tag_item.strip().lower()
                if any(
                    token in normalized_source_tag
                    for token in ("phase1", "3dgs", "geometry_bank", "episode")
                ):
                    return True
        return False

    def _compose_render_residual_refined_rgb(
        self,
        prediction_rgb: torch.Tensor,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        accum_render = batch.get("accum_render")
        render_rgb = batch.get("rgb_render")
        if not isinstance(accum_render, torch.Tensor) or not isinstance(render_rgb, torch.Tensor):
            return prediction_rgb
        residual_base_rgb = render_rgb.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )
        if residual_base_rgb.shape[-2:] != prediction_rgb.shape[-2:]:
            residual_base_rgb = F.interpolate(
                residual_base_rgb,
                size=prediction_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        residual_blend_mask = accum_render.to(
            device=prediction_rgb.device,
            dtype=prediction_rgb.dtype,
        )
        if residual_blend_mask.dim() == 3:
            residual_blend_mask = residual_blend_mask.unsqueeze(1)
        if residual_blend_mask.shape[1] > 1:
            residual_blend_mask = residual_blend_mask[:, :1]
        if residual_blend_mask.shape[-2:] != prediction_rgb.shape[-2:]:
            residual_blend_mask = F.interpolate(
                residual_blend_mask,
                size=prediction_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        residual_blend_mask = torch.nan_to_num(
            residual_blend_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return prediction_rgb * (1.0 - residual_blend_mask) + residual_base_rgb * residual_blend_mask

    def _resolve_proposal_reference_rgb_from_debug_tensors(
        self,
        sample_debug_tensors: Dict[str, Any],
        fallback_rgb: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        proposal_raw_rgb = sample_debug_tensors.get("proposal_raw_rgb")
        if isinstance(proposal_raw_rgb, torch.Tensor):
            return proposal_raw_rgb
        decoded_teacher_rgb = sample_debug_tensors.get("decoded_teacher_rgb")
        if isinstance(decoded_teacher_rgb, torch.Tensor):
            return decoded_teacher_rgb
        return fallback_rgb

    def _select_teacher_prediction_rgb(
        self,
        outputs: Any,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self._batch_uses_phase1_3dgs_conditions(batch):
            proposal_rgb = getattr(outputs, "predicted_proposal_image", None)
            if proposal_rgb is not None:
                return proposal_rgb
            residual_refined_rgb = getattr(outputs, "predicted_residual_refined_image", None)
            if residual_refined_rgb is not None:
                return residual_refined_rgb
            residual_or_predicted_rgb = getattr(outputs, "predicted_image", None)
            if residual_or_predicted_rgb is not None:
                return residual_or_predicted_rgb
        pure_x0_rgb = getattr(outputs, "predicted_pure_x0_image", None)
        if pure_x0_rgb is not None:
            return pure_x0_rgb
        return getattr(outputs, "predicted_image", None)

    def _trainable_params_are_fp32(self) -> bool:
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if param.dtype != torch.float32:
                return False
        return True

    def _set_epoch(self, epoch: int) -> None:
        sampler = getattr(self.train_dataloader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def _reduce_values(self, values: Sequence[float]) -> List[float]:
        tensor = torch.tensor(values, device=self.device, dtype=torch.float64)
        if self._distributed:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.tolist()

    def _vpdt_variance_loss(
        self,
        noise_pred: torch.Tensor,
        target_std: float = 1.0,
    ) -> torch.Tensor:
        """
        VPDT: 方差保持扩散训练损失
        
        理论依据：
        在标准扩散中，ε ~ N(0, I)，即 Std[ε] = 1
        如果模型预测的噪声方差 < 1，说明模型在"保守"预测
        这会导致生成图像的动态范围被压缩
        
        Args:
            noise_pred: [B, C, H, W] 模型预测的噪声
            target_std: 目标标准差，通常为1.0
            
        Returns:
            方差匹配损失 (Std[noise_pred] - target_std)^2
        """
        # 计算每个batch样本的标准差，然后取均值
        # 使用全局std而非per-sample，更稳定
        pred_std = noise_pred.std()
        
        # MSE损失：(预测std - 目标std)^2
        variance_loss = (pred_std - target_std) ** 2
        
        return variance_loss
    
    # ========== Stage 0: Domain Adaptation (新增) ==========
    def stage1_pretrain(
        self,
        num_epochs: int = 10,
        lr: float = 1e-4,
        *,
        # Stage1 额外损失的开关与权重（默认开启，若VGG权重缺失会自动跳过）
        use_perceptual: bool = True,
        lambda_perceptual: float = 0.2,
        perceptual_layers: Optional[Sequence[str]] = None,
        perceptual_weights: Optional[Sequence[float]] = None,
        use_consistency: bool = False,  # 默认禁用: 稀疏视角下out_of_frame>50%导致loss振荡
        lambda_consistency: float = 1.0,
        consistency_depth_type: str = "z-depth",
        pose_angle_threshold: float = 15.0,
        min_pose_similarity: float = 0.1,
        # 重建项 L1/L2 的混合系数
        recon_alpha: float = 0.1,  # 降低x0重建权重: 0.6→0.1, 防止x0 loss主导总loss
        # 采样步数（为控制开销，建议小于训练/推理默认步数）
        ddim_steps: int = 20,
        ddim_interval: int = 5,
        x0_switch_step: int = 0,  # 默认0：直接使用可微 sample-path；>0 时仅在 warmup 内使用 x0_decode
        adaptive_trigger: float = 0.0,  # 禁用x0切换: x0辅助损失(recon+perceptual)是振荡源
        validation_interval: Optional[int] = None,
        visualization_interval: Optional[int] = None,
        validation_mode: Optional[str] = None,
        validation_max_batches: Optional[int] = None,
        sample_path_interval: Optional[int] = None,
        sample_path_weight: Optional[float] = None,
        aux_batch_chunk_size: Optional[int] = None,
        stage_tag: str = "stage1",
        stage_display_name: str = "Stage1",
    ) -> None:
        """
        Stage 1: optimise RGB generation (diffusion objective + optional losses).
        """
        # ========== 强制使用FP32以避免NaN问题 ==========
        fp16_params = []
        for name, param in self.model.named_parameters():
            if param.dtype == torch.float16:
                fp16_params.append(name)
        if fp16_params:
            if self._is_main_rank:
                print(f"[{stage_display_name}] 检测到 {len(fp16_params)} 个FP16参数，正在转换为FP32...")
            self.model.float()  # 将整个模型转换为FP32
            if self._is_main_rank:
                print(f"[{stage_display_name}] 模型已转换为FP32，这将解决NaN问题")
        
        self.raw_model.epistemic_head = None
        self.raw_model.aleatoric_head = None
        if hasattr(self.raw_model, "vae"):
            for param in self.raw_model.vae.parameters():
                param.requires_grad = False
        
        # 处理UNet的up_blocks.3层（这些层在FP16下容易产生NaN）
        # 由于已经转换为FP32，可以选择不冻结这些层以获得更好的训练效果
        freeze_up_blocks_3 = getattr(self, 'stage1_freeze_up_blocks_3', False)  # 默认不冻结
        if freeze_up_blocks_3:
            frozen_count = 0
            if hasattr(self.raw_model, "unet"):
                for name, param in self.raw_model.unet.named_parameters():
                    if 'up_blocks.3' in name:
                        param.requires_grad = False
                        frozen_count += 1
            if self._is_main_rank and frozen_count > 0:
                print(f"[{stage_display_name}] 已冻结UNet up_blocks.3层 ({frozen_count}个参数) 以防止NaN")
        else:
            # 确保这些层保持FP32精度（已通过self.model.float()实现）
            if self._is_main_rank:
                print(f"[{stage_display_name}] up_blocks.3层保持可训练状态（FP32精度，无需冻结）")
        
        if hasattr(self.raw_model, "view_aggregator") and self.raw_model.view_aggregator is not None:
            self.raw_model.view_aggregator.set_pose_gate(
                angle_threshold_deg=pose_angle_threshold, min_similarity=min_pose_similarity
            )
        self.optimizer = torch.optim.AdamW(
            filter(lambda param: param.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )
        # P1b: Warmup + CosineAnnealing — 前2个epoch从lr/10线性增长到lr
        # 使用LambdaLR实现warmup，兼容所有PyTorch版本
        warmup_epochs = min(2, max(num_epochs - 1, 1))
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return 0.1 + 0.9 * epoch / warmup_epochs  # 从0.1*lr线性增长到lr
            else:
                # Cosine annealing
                import math
                progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
                return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        # 初始化感知损失（若可用）
        perceptual_fn: Optional[_PerceptualLoss] = None
        if use_perceptual:
            perceptual_fn = _PerceptualLoss(
                device=str(self._resolve_auxiliary_loss_device()),
                layers=perceptual_layers,
                weights=perceptual_weights,
            )
            if not perceptual_fn.available:
                if self.world_rank == 0:
                    print(f"[{stage_display_name}] VGG 权重不可用，已跳过感知损失。")
                perceptual_fn = None
        stage_sample_prefix = str(stage_tag).strip().lower() or "stage1"
        resolved_x0_switch_step = max(0, int(x0_switch_step))
        resolved_validation_mode = str(
            self.stage1_validation_mode if validation_mode is None else validation_mode
        ).strip().lower()
        if resolved_validation_mode not in {"full_generate", "trainable_sample"}:
            resolved_validation_mode = "full_generate"
        resolved_validation_max_batches = max(
            0,
            int(
                self.stage1_validation_max_batches
                if validation_max_batches is None
                else validation_max_batches
            ),
        )
        resolved_sample_path_interval = max(
            1,
            int(
                self.stage1_sample_path_interval
                if sample_path_interval is None
                else sample_path_interval
            ),
        )
        resolved_sample_path_weight = max(
            0.0,
            float(
                self.stage1_sample_path_weight
                if sample_path_weight is None
                else sample_path_weight
            ),
        )
        resolved_frontier_objective_config = self._resolve_frontier_proposal_objective_config(
            stage_sample_prefix
        )
        resolved_frontier_residual_weight = float(
            resolved_frontier_objective_config["frontier_residual_weight"]
        )
        resolved_surrogate_reconstruction_decay = float(
            resolved_frontier_objective_config["surrogate_reconstruction_decay"]
        )
        resolved_exist_diffusion_weight = resolved_frontier_objective_config[
            "exist_diffusion_weight"
        ]
        resolved_novel_diffusion_weight = resolved_frontier_objective_config[
            "novel_diffusion_weight"
        ]
        configured_aux_batch_chunk_size = (
            None if aux_batch_chunk_size is None else max(0, int(aux_batch_chunk_size))
        )
        config_summary = {
            "use_perceptual": bool(perceptual_fn is not None),
            "lambda_perceptual": float(lambda_perceptual),
            "use_consistency": bool(use_consistency),
            "lambda_consistency": float(lambda_consistency),
            "recon_alpha": float(recon_alpha),
            "ddim_steps": int(ddim_steps),
            "ddim_interval": int(max(ddim_interval, 1)),
            "x0_switch_step": int(resolved_x0_switch_step),
            "adaptive_trigger": float(max(adaptive_trigger, 0.0)),
            "pose_angle_threshold": float(pose_angle_threshold),
            "min_pose_similarity": float(min_pose_similarity),
            "consistency_depth_type": consistency_depth_type,
            "prediction_type": self._resolve_scheduler_prediction_type(),
            "min_snr_gamma": float(self.min_snr_gamma) if self.enable_min_snr_weighting else 0.0,
            "sample_guidance_scale": float(self.sample_guidance_scale),
            "sample_disable_latent_blending": bool(self.sample_disable_latent_blending),
            "sample_inpaint_img2img_strength_override": (
                float(self.sample_inpaint_img2img_strength_override)
                if self.sample_inpaint_img2img_strength_override is not None
                else None
            ),
            "relative_teacher_effect_weight": (
                None
                if getattr(self, "relative_teacher_effect_weight", None) is None
                else float(getattr(self, "relative_teacher_effect_weight"))
            ),
            "teacher_warp_margin_weight": float(
                getattr(self, "teacher_warp_margin_weight", 0.0)
            ),
            "teacher_warp_margin": float(getattr(self, "teacher_warp_margin", 0.0)),
            "validation_use_warp_delta_score": bool(
                getattr(self, "validation_use_warp_delta_score", False)
            ),
            "frontier_proposal_focus": bool(
                getattr(self, "geometry_aware_frontier_proposal_focus", False)
            ),
            "frontier_kernel_size": int(
                getattr(self, "geometry_aware_frontier_kernel_size", 31)
            ),
            "controlnet_condition_channels": int(
                getattr(getattr(self.raw_model, "condition_preprocessor", None), "output_channels", 3)
            ),
            "enable_plucker_conditioning": bool(
                getattr(self.raw_model, "enable_plucker_conditioning", False)
            ),
            "validation_interval": int(
                self.stage1_validation_interval if validation_interval is None else validation_interval
            ),
            "visualization_interval": int(
                self.stage1_visualization_interval if visualization_interval is None else visualization_interval
            ),
            "validation_mode": resolved_validation_mode,
            "validation_max_batches": int(resolved_validation_max_batches),
            "sample_path_interval": int(resolved_sample_path_interval),
            "sample_path_weight": float(resolved_sample_path_weight),
            "frontier_residual_weight": float(resolved_frontier_residual_weight),
            "surrogate_reconstruction_decay": float(resolved_surrogate_reconstruction_decay),
            "exist_diffusion_weight": (
                None if resolved_exist_diffusion_weight is None else float(resolved_exist_diffusion_weight)
            ),
            "novel_diffusion_weight": (
                None if resolved_novel_diffusion_weight is None else float(resolved_novel_diffusion_weight)
            ),
            "aux_batch_chunk_size": (
                "auto" if configured_aux_batch_chunk_size is None else int(configured_aux_batch_chunk_size)
            ),
            "stage_tag": stage_sample_prefix,
            "stage_display_name": stage_display_name,
        }
        if self.world_rank == 0:
            print(f"[{stage_display_name}] 配置摘要:", config_summary)

        global_step = 0
        early_stop_best_metric: Optional[float] = None
        early_stop_plateau_count = 0
        ddim_compute_count = 0
        x0_usage_count = 0
        adaptive_trigger_count = 0
        geometry_stats = {
            "valid_view_ratio": 0.0,
            "gate_trigger_ratio": 0.0,
            "out_of_frame_ratio": 0.0,
            "support_coverage_ratio": 0.0,
            "support_confidence_ratio": 0.0,
        }
        geometry_calls = 0
        last_diffusion_loss: Optional[float] = None
        current_ddim_interval = max(resolved_sample_path_interval, 1)
        consistency_enabled = bool(use_consistency)
        resolved_relative_teacher_weight = getattr(
            self,
            "relative_teacher_effect_weight",
            None,
        )
        if resolved_relative_teacher_weight is None:
            resolved_relative_teacher_weight = (
                float(lambda_consistency)
                if consistency_enabled and lambda_consistency > 0.0
                else 0.0
            )
        current_relative_teacher_effect_weight = max(
            0.0,
            float(resolved_relative_teacher_weight),
        )
        aux_interval_backoff = False
        aux_forced_off = False
        aux_downscale_factor = 1.0
        min_aux_downscale = 0.35
        resolved_validation_interval = max(
            0,
            int(self.stage1_validation_interval if validation_interval is None else validation_interval),
        )
        resolved_visualization_interval = max(
            0,
            int(self.stage1_visualization_interval if visualization_interval is None else visualization_interval),
        )

        def _resize_for_aux(
            tensor: Optional[torch.Tensor],
            target_hw: Tuple[int, int],
            *,
            is_mask: bool = False,
        ) -> Optional[torch.Tensor]:
            if tensor is None or tensor.shape[-2:] == target_hw:
                return tensor
            if tensor.shape[-2:] == target_hw[::-1]:
                return tensor.transpose(-1, -2).contiguous()
            mode = "nearest" if is_mask else "bilinear"
            align = None if mode == "nearest" else False
            if tensor.dim() == 5:
                bsz, views = tensor.shape[:2]
                flattened = tensor.reshape(bsz * views, *tensor.shape[2:])
                resized = F.interpolate(
                    flattened,
                    size=target_hw,
                    mode=mode,
                    align_corners=align,
                )
                return resized.view(bsz, views, *resized.shape[1:])
            return F.interpolate(tensor, size=target_hw, mode=mode, align_corners=align)

        def _compute_stage1_auxiliary_losses_for_chunk(
            batch_chunk: Dict[str, Any],
            rgb_prediction_chunk: torch.Tensor,
            *,
            aux_hw: Tuple[int, int],
            target_hw: Tuple[int, int],
            aux_target_device: torch.device,
        ) -> Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            Optional[Dict[str, float]],
            bool,
        ]:
            target_rgb_chunk = (
                batch_chunk["target_rgb"]
                if aux_hw == target_hw
                else F.interpolate(
                    batch_chunk["target_rgb"], size=aux_hw, mode="bilinear", align_corners=False
                )
            )
            depth_info_chunk = batch_chunk.get("geometry_depth_render")
            if depth_info_chunk is None:
                depth_info_chunk = batch_chunk.get("depth_render")
            if (
                depth_info_chunk is not None
                and depth_info_chunk.dim() >= 2
                and depth_info_chunk.shape[1] == 3
            ):
                depth_info_chunk = depth_info_chunk[:, :1, ...]
            depth_for_chunk = (
                depth_info_chunk if aux_hw == target_hw else _resize_for_aux(depth_info_chunk, aux_hw)
            )
            target_rgb_chunk = target_rgb_chunk.to(aux_target_device).detach()
            if depth_for_chunk is not None:
                depth_for_chunk = depth_for_chunk.to(aux_target_device).detach()

            support_mask_for_chunk = (
                batch_chunk.get("warp_valid_mask")
                if aux_hw == target_hw
                else _resize_for_aux(batch_chunk.get("warp_valid_mask"), aux_hw)
            )
            support_confidence_for_chunk = (
                batch_chunk.get("warp_support_confidence")
                if aux_hw == target_hw
                else _resize_for_aux(batch_chunk.get("warp_support_confidence"), aux_hw)
            )
            if support_mask_for_chunk is not None:
                support_mask_for_chunk = support_mask_for_chunk.to(aux_target_device).detach()
            if support_confidence_for_chunk is not None:
                support_confidence_for_chunk = support_confidence_for_chunk.to(
                    aux_target_device
                ).detach()
            sampler_aligned_support_signals = self._build_sampler_aligned_support_signals(
                support_mask=support_mask_for_chunk,
                support_confidence=support_confidence_for_chunk,
                target_hw=rgb_prediction_chunk.shape[-2:],
            )
            sampler_aligned_valid_mask = sampler_aligned_support_signals.get("warp_valid_mask")
            sampler_aligned_projection_mask = sampler_aligned_support_signals.get(
                "support_projection_mask"
            )
            sampler_aligned_support_confidence = sampler_aligned_support_signals.get(
                "support_confidence"
            )
            exist_mask_for_chunk, novel_mask_for_chunk = self._build_projection_aligned_exist_novel_masks(
                support_mask=support_mask_for_chunk,
                support_confidence=support_confidence_for_chunk,
                target_hw=rgb_prediction_chunk.shape[-2:],
            )
            if exist_mask_for_chunk is None and novel_mask_for_chunk is None:
                exist_mask_for_chunk, novel_mask_for_chunk = self._build_exist_novel_masks(
                    support_mask_for_chunk,
                    depth_for_chunk,
                    rgb_prediction_chunk.shape[-2:],
                )
            if sampler_aligned_valid_mask is None:
                sampler_aligned_valid_mask = support_mask_for_chunk
            if sampler_aligned_support_confidence is None:
                sampler_aligned_support_confidence = support_confidence_for_chunk
            if sampler_aligned_projection_mask is None:
                sampler_aligned_projection_mask = sampler_aligned_support_confidence

            support_anchor_rgb_for_chunk = self._get_support_anchor_rgb_from_batch(batch_chunk)
            base_rgb_for_chunk = (
                support_anchor_rgb_for_chunk
                if aux_hw == target_hw
                else F.interpolate(
                    support_anchor_rgb_for_chunk,
                    size=aux_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            )
            if base_rgb_for_chunk is not None:
                base_rgb_for_chunk = base_rgb_for_chunk.to(
                    device=aux_target_device,
                    dtype=rgb_prediction_chunk.dtype,
                ).detach()

            loss_reconstruction = self._region_aware_reconstruction_loss(
                rgb_prediction_chunk,
                target_rgb_chunk,
                base_rgb_for_chunk,
                exist_mask=exist_mask_for_chunk,
                novel_mask=novel_mask_for_chunk,
                recon_alpha=recon_alpha,
            )
            if perceptual_fn is not None and lambda_perceptual > 0.0:
                loss_perceptual = perceptual_fn(rgb_prediction_chunk, target_rgb_chunk) * float(
                    lambda_perceptual
                )
            else:
                loss_perceptual = loss_reconstruction.new_zeros(())

            if consistency_enabled and lambda_consistency > 0.0:
                loss_consistency, geometry_stat_dict = _joint_support_consistency_loss(
                    rgb_gen=rgb_prediction_chunk,
                    support_base_rgb=base_rgb_for_chunk,
                    support_mask=sampler_aligned_valid_mask,
                    support_confidence=sampler_aligned_projection_mask,
                    return_stats=True,
                )
                geometry_stats_enabled = True
            else:
                loss_consistency = loss_reconstruction.new_zeros(())
                geometry_stat_dict = None
                geometry_stats_enabled = False

            relative_teacher_loss = loss_reconstruction.new_zeros(())
            frontier_residual_loss = loss_reconstruction.new_zeros(())
            relative_effect_targets = None
            if (
                current_relative_teacher_effect_weight > 0.0
                or (
                    resolved_frontier_residual_weight > 0.0
                    and self._batch_uses_phase1_3dgs_conditions(batch_chunk)
                )
            ):
                render_rgb_for_chunk = (
                    batch_chunk["rgb_render"]
                    if aux_hw == target_hw
                    else F.interpolate(
                        batch_chunk["rgb_render"],
                        size=aux_hw,
                        mode="bilinear",
                        align_corners=False,
                    )
                )
                render_rgb_for_chunk = render_rgb_for_chunk.to(
                    device=aux_target_device,
                    dtype=rgb_prediction_chunk.dtype,
                ).detach()
                relative_effect_targets = self._compute_teacher_effect_targets(
                    teacher_rgb=rgb_prediction_chunk,
                    render_rgb=render_rgb_for_chunk,
                    target_rgb=target_rgb_chunk,
                    depth_render=depth_for_chunk,
                    support_mask=support_mask_for_chunk,
                    support_confidence=support_confidence_for_chunk,
                    frontier_mask=batch_chunk.get("frontier_mask"),
                    verification_prior=batch_chunk.get("verification_prior"),
                    use_projection_aligned_support=True,
                )
            if current_relative_teacher_effect_weight > 0.0 and relative_effect_targets is not None:
                relative_teacher_loss = self._compute_relative_teacher_effect_loss(
                    relative_effect_targets
                )
            if (
                resolved_frontier_residual_weight > 0.0
                and self._batch_uses_phase1_3dgs_conditions(batch_chunk)
                and relative_effect_targets is not None
            ):
                frontier_residual_loss = self._compute_frontier_residual_proposal_loss(
                    relative_effect_targets,
                    proposal_rgb=rgb_prediction_chunk,
                )
            return (
                loss_reconstruction,
                loss_perceptual,
                loss_consistency,
                relative_teacher_loss,
                frontier_residual_loss,
                geometry_stat_dict,
                geometry_stats_enabled,
            )

        def _stage1_env_flag(env_name: str, default_value: bool) -> bool:
            raw_env_value = os.environ.get(env_name)
            if raw_env_value is None:
                return default_value
            return raw_env_value.strip().lower() not in {"0", "false", "off", "no"}

        def _stage1_env_int(env_name: str, default_value: int) -> int:
            raw_env_value = os.environ.get(env_name)
            if raw_env_value is None:
                return default_value
            try:
                return int(raw_env_value)
            except (TypeError, ValueError):
                return default_value

        stage1_timing_enabled = _stage1_env_flag("GDDN_STAGE1_TIMING", False)
        stage1_timing_sync_cuda = _stage1_env_flag("GDDN_STAGE1_TIMING_SYNC_CUDA", False)
        stage1_timing_verbose = _stage1_env_flag("GDDN_STAGE1_TIMING_VERBOSE", False)
        stage1_timing_batch_limit = max(
            1, _stage1_env_int("GDDN_STAGE1_TIMING_BATCH_LIMIT", 8)
        )
        stage1_global_batch_index = 0

        if self._is_main_rank and stage1_timing_enabled:
            print(
                f"[Stage1][Timer] 已启用Stage1计时: batch_limit={stage1_timing_batch_limit}, "
                f"sync_cuda={stage1_timing_sync_cuda}, verbose={stage1_timing_verbose}"
            )

        def _stage1_sync_for_timing() -> None:
            if stage1_timing_sync_cuda and torch.cuda.is_available():
                torch.cuda.synchronize()

        def _stage1_time_now() -> float:
            _stage1_sync_for_timing()
            return time.perf_counter()

        @contextmanager
        def _stage1_timed_region(
            timing_store: Dict[str, float],
            region_name: str,
            *,
            enabled: bool,
            batch_index: int,
            emit_progress: bool = False,
            progress_suffix: Optional[str] = None,
        ):
            if not enabled:
                yield
                return
            start_message = f"[Stage1][Timer] batch={batch_index} 开始 {region_name}"
            if progress_suffix:
                start_message = f"{start_message} ({progress_suffix})"
            if emit_progress and stage1_timing_verbose:
                print(start_message)
            start_time = _stage1_time_now()
            try:
                yield
            finally:
                elapsed_seconds = _stage1_time_now() - start_time
                timing_store[region_name] = timing_store.get(region_name, 0.0) + elapsed_seconds
                end_message = (
                    f"[Stage1][Timer] batch={batch_index} 完成 {region_name}: "
                    f"{elapsed_seconds:.3f}s"
                )
                if progress_suffix:
                    end_message = f"{end_message} ({progress_suffix})"
                if emit_progress and stage1_timing_verbose:
                    print(end_message)

        def _emit_stage1_timing_summary(
            *,
            batch_index: int,
            batch_timings: Dict[str, float],
            batch_process_start_time: Optional[float],
            compute_aux: bool,
            aux_mode: str,
            current_loss: Optional[float],
            note: str = "ok",
        ) -> None:
            if not stage1_timing_enabled or not self._is_main_rank:
                return
            ordered_region_names = (
                "fetch_batch",
                "prepare_batch",
                "encode_target_latent",
                "sample_diffusion_inputs",
                "model_forward",
                "diffusion_loss",
                "vpdt_loss",
                "aux_total",
                "aux_sample_path",
                "aux_resize_align",
                "aux_reconstruction",
                "aux_perceptual",
                "aux_consistency",
                "aux_teacher_effect",
                "aux_image_logging",
                "backward",
                "clip_and_step",
            )
            summary_entries: List[str] = []
            for region_name in ordered_region_names:
                if region_name in batch_timings:
                    summary_entries.append(
                        f"{region_name}={batch_timings[region_name]:.3f}s"
                    )
            if not compute_aux and "aux_total" not in batch_timings:
                summary_entries.append("aux_total=skipped")
            if batch_process_start_time is not None:
                batch_process_total = _stage1_time_now() - batch_process_start_time
                summary_entries.append(f"batch_process_total={batch_process_total:.3f}s")
            meta_entries = [
                f"compute_aux={compute_aux}",
                f"aux_mode={aux_mode}",
                f"note={note}",
            ]
            if current_loss is not None:
                meta_entries.append(f"loss={current_loss:.4f}")
            print(
                f"[Stage1][Timer] batch={batch_index} summary ({', '.join(meta_entries)}): "
                + ", ".join(summary_entries)
            )

        for epoch in range(num_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.train()
            # 在分布式模式下，为了保证每个 epoch 的随机性，需要设置 sampler 的 epoch
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)
                
            epoch_loss = 0.0
            epoch_batches = 0
            diffusion_loss_accum = 0.0
            image_logged = False
            epoch_validation_enabled = bool(
                self.val_dataloader is not None
                and resolved_validation_interval > 0
                and (
                    (epoch + 1) % resolved_validation_interval == 0
                    or epoch == max(num_epochs - 1, 0)
                )
            )
            epoch_visualization_enabled = bool(
                resolved_visualization_interval > 0
                and (
                    (epoch + 1) % resolved_visualization_interval == 0
                    or epoch == max(num_epochs - 1, 0)
                )
            )
            train_batch_iterator = iter(self.train_dataloader)
            while True:
                next_batch_index = stage1_global_batch_index + 1
                batch_timing_enabled = (
                    stage1_timing_enabled
                    and self._is_main_rank
                    and next_batch_index <= stage1_timing_batch_limit
                )
                batch_timings: Dict[str, float] = {}
                with _stage1_timed_region(
                    batch_timings,
                    "fetch_batch",
                    enabled=batch_timing_enabled,
                    batch_index=next_batch_index,
                    emit_progress=True,
                ):
                    try:
                        batch = next(train_batch_iterator)
                    except StopIteration:
                        batch = None
                if batch is None:
                    break
                stage1_global_batch_index = next_batch_index
                batch_process_start_time = (
                    _stage1_time_now() if batch_timing_enabled else None
                )
                aux_mode = "not_triggered"

                with _stage1_timed_region(
                    batch_timings,
                    "prepare_batch",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                ):
                    batch = self._to_device(batch)

                    # 确保输入张量也是FP32（与模型权重匹配）
                    for key in batch:
                        if (
                            isinstance(batch[key], torch.Tensor)
                            and batch[key].dtype == torch.float16
                        ):
                            batch[key] = batch[key].float()

                target_latent_needs_encode = bool(batch["target_latent"].abs().sum() < 1e-6)
                with _stage1_timed_region(
                    batch_timings,
                    "encode_target_latent",
                    enabled=batch_timing_enabled and target_latent_needs_encode,
                    batch_index=stage1_global_batch_index,
                    emit_progress=target_latent_needs_encode,
                ):
                    if target_latent_needs_encode:
                        self._ensure_batch_target_latents(batch)

                with _stage1_timed_region(
                    batch_timings,
                    "sample_diffusion_inputs",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                ):
                    timesteps, diffusion_target, noisy_latents = self._sample_diffusion_inputs(batch)
                with _stage1_timed_region(
                    batch_timings,
                    "model_forward",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                    emit_progress=True,
                ):
                    outputs = self.model(

                        sparse_images=batch["sparse_images"],
                        sparse_poses=batch["sparse_poses"],
                        target_pose=batch["target_pose"],
                        rgb_render=batch["rgb_render"],
                        depth_render=batch["depth_render"],
                        normal_render=batch["normal_render"],
                        reference_depth_map=batch.get("reference_depth_map"),
                        reference_intrinsics=batch.get("reference_intrinsics"),
                        target_intrinsics=batch.get("target_intrinsics"),
                        sparse_intrinsics=batch.get("sparse_intrinsics"),
                        support_confidence=batch.get("warp_support_confidence"),
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        return_uncertainty=False,
                    )
                # 标准扩散去噪损失
                # 多GPU模式下需要确保张量在同一设备
                diffusion_target_aligned = diffusion_target.to(outputs.noise_pred.device)
                with _stage1_timed_region(
                    batch_timings,
                    "diffusion_loss",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                ):
                    diffusion_loss = self._masked_diffusion_loss(
                        outputs.noise_pred,
                        diffusion_target_aligned,
                        batch.get("depth_render"),
                        support_mask=batch.get("warp_valid_mask"),
                        support_confidence=batch.get("warp_support_confidence"),
                        timesteps=timesteps,
                        exist_diffusion_weight=resolved_exist_diffusion_weight,
                        novel_diffusion_weight=resolved_novel_diffusion_weight,
                    )

                # VPDT: 方差保持损失，防止动态范围坍塌
                with _stage1_timed_region(
                    batch_timings,
                    "vpdt_loss",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                ):
                    vpdt_loss = self._vpdt_variance_loss(
                        outputs.noise_pred,
                        target_std=self.vpdt_target_std,
                    )

                loss = diffusion_loss + self.lambda_vpdt * vpdt_loss
                
                # 残差正则化：已禁用（lambda=0），因为它会阻止gates学习增长
                # lambda_residual = 0.01  # 正则化权重
                # if hasattr(outputs, 'residual_penalty') and outputs.residual_penalty is not None:
                #     loss = loss + lambda_residual * outputs.residual_penalty
                
                diffusion_loss_accum += float(diffusion_loss.item())
                
                # ========== 诊断日志：Stage1张量范围检查 ==========
                # 可选：基于解码图像的重建/感知/几何一致性损失
                add_losses = 0.0
                compute_aux = False
                global_step += 1
                if current_ddim_interval > 0 and global_step % current_ddim_interval == 0:
                    compute_aux = True
                if last_diffusion_loss is not None and adaptive_trigger > 0.0:
                    drop = (last_diffusion_loss - float(diffusion_loss.item())) / (
                        abs(last_diffusion_loss) + 1e-8
                    )
                    if drop > adaptive_trigger:
                        compute_aux = True
                        adaptive_trigger_count += 1
                last_diffusion_loss = float(diffusion_loss.item())

                if compute_aux:
                    try:
                        with _stage1_timed_region(
                            batch_timings,
                            "aux_total",
                            enabled=batch_timing_enabled,
                            batch_index=stage1_global_batch_index,
                            emit_progress=True,
                        ):
                            target_hw = batch["target_rgb"].shape[-2:]  # 提前定义，避免OOM时未定义
                            x0_warmup_steps = int(resolved_x0_switch_step)
                            use_x0 = x0_warmup_steps > 0 and global_step <= x0_warmup_steps
                            aux_mode = "x0_decode" if use_x0 else "trainable_sample"
                            with _stage1_timed_region(
                                batch_timings,
                                "aux_sample_path",
                                enabled=batch_timing_enabled,
                                batch_index=stage1_global_batch_index,
                                emit_progress=True,
                                progress_suffix=aux_mode,
                            ):
                                if use_x0:
                                    rgb_gen = self._decode_predicted_x0(
                                        noisy_latents, outputs.noise_pred, timesteps, track_grad=True
                                    )
                                    x0_usage_count += 1
                                else:
                                    generation = self._generate_trainable_sample(
                                        batch,
                                        steps=max(1, min(int(ddim_steps), self.stage1_trainable_sample_steps)),
                                        sampler_type=self.sampler_type,
                                        return_diagnostics=False,
                                    )
                                    if generation is None:
                                        aux_mode = "fallback_sample"
                                        generation = self._generate_sample(
                                            batch,
                                            steps=max(1, int(ddim_steps)),
                                            sampler_type=self.sampler_type,
                                            return_diagnostics=False,
                                        )
                                    if isinstance(generation, tuple):
                                        rgb_gen = generation[0]
                                    elif isinstance(generation, dict):
                                        rgb_gen = generation.get("rgb")
                                        if rgb_gen is None:
                                            rgb_gen = generation.get("image")
                                        if rgb_gen is None:
                                            raise RuntimeError("Stage1 可微采样未返回 rgb/image。")
                                    else:
                                        rgb_gen = generation
                                    ddim_compute_count += 1

                            with _stage1_timed_region(
                                batch_timings,
                                "aux_resize_align",
                                enabled=batch_timing_enabled,
                                batch_index=stage1_global_batch_index,
                            ):
                                target_hw = batch["target_rgb"].shape[-2:]
                                if rgb_gen.shape[-2:] != target_hw:
                                    if rgb_gen.shape[-2:] == target_hw[::-1]:
                                        rgb_gen = rgb_gen.transpose(-1, -2).contiguous()
                                    else:
                                        rgb_gen = F.interpolate(
                                            rgb_gen, size=target_hw, mode="bilinear", align_corners=False
                                        )
                                if self._batch_uses_phase1_3dgs_conditions(batch):
                                    rgb_gen = self._compose_render_residual_refined_rgb(rgb_gen, batch)
                                aux_target_device = self._resolve_auxiliary_loss_device()
                                if rgb_gen.device != aux_target_device:
                                    rgb_gen = rgb_gen.to(
                                        device=aux_target_device,
                                        dtype=rgb_gen.dtype,
                                    )
                                aux_hw = target_hw
                                apply_downscale = aux_downscale_factor < 0.999
                                if apply_downscale:
                                    aux_hw = tuple(
                                        max(32, int(max(dim, 1) * aux_downscale_factor)) for dim in target_hw
                                    )
                                rgb_for_loss = rgb_gen if aux_hw == target_hw else F.interpolate(
                                    rgb_gen, size=aux_hw, mode="bilinear", align_corners=False
                                )
                                effective_aux_batch_chunk_size = self._resolve_stage_auxiliary_batch_chunk_size(
                                    stage_tag=stage_sample_prefix,
                                    batch_size=int(rgb_for_loss.shape[0]),
                                    configured_chunk_size=configured_aux_batch_chunk_size,
                                )

                            with torch.cuda.amp.autocast(
                                dtype=self._amp_dtype, enabled=self._amp_enabled
                            ):
                                with _stage1_timed_region(
                                    batch_timings,
                                    "aux_reconstruction",
                                    enabled=batch_timing_enabled,
                                    batch_index=stage1_global_batch_index,
                                    emit_progress=effective_aux_batch_chunk_size < int(rgb_for_loss.shape[0]),
                                ):
                                    loss_recon = torch.zeros((), device=aux_target_device, dtype=rgb_for_loss.dtype)
                                    loss_perc = torch.zeros((), device=aux_target_device, dtype=rgb_for_loss.dtype)
                                    loss_cons = torch.zeros((), device=aux_target_device, dtype=rgb_for_loss.dtype)
                                    relative_teacher_loss = torch.zeros((), device=aux_target_device, dtype=rgb_for_loss.dtype)
                                    frontier_residual_loss = torch.zeros(
                                        (),
                                        device=aux_target_device,
                                        dtype=rgb_for_loss.dtype,
                                    )
                                    stats: Optional[Dict[str, float]] = None
                                    stats_enabled = False
                                    total_aux_batch_size = max(1, int(rgb_for_loss.shape[0]))
                                    for batch_start_index in range(
                                        0,
                                        total_aux_batch_size,
                                        max(1, int(effective_aux_batch_chunk_size)),
                                    ):
                                        batch_end_index = min(
                                            total_aux_batch_size,
                                            batch_start_index + max(1, int(effective_aux_batch_chunk_size)),
                                        )
                                        batch_chunk = self._slice_batch_along_batch_dim(
                                            batch,
                                            batch_start_index=batch_start_index,
                                            batch_end_index=batch_end_index,
                                            reference_batch_size=total_aux_batch_size,
                                        )
                                        rgb_prediction_chunk = rgb_for_loss[
                                            batch_start_index:batch_end_index
                                        ].to(
                                            device=aux_target_device,
                                            dtype=rgb_for_loss.dtype,
                                        )
                                        chunk_weight = float(rgb_prediction_chunk.shape[0]) / float(
                                            total_aux_batch_size
                                        )
                                        (
                                            loss_recon_chunk,
                                            loss_perc_chunk,
                                            loss_cons_chunk,
                                            relative_teacher_loss_chunk,
                                            frontier_residual_loss_chunk,
                                            stats_chunk,
                                            stats_enabled_chunk,
                                        ) = _compute_stage1_auxiliary_losses_for_chunk(
                                            batch_chunk,
                                            rgb_prediction_chunk,
                                            aux_hw=aux_hw,
                                            target_hw=target_hw,
                                            aux_target_device=aux_target_device,
                                        )
                                        loss_recon = loss_recon + loss_recon_chunk * chunk_weight
                                        loss_perc = loss_perc + loss_perc_chunk * chunk_weight
                                        loss_cons = loss_cons + loss_cons_chunk * chunk_weight
                                        relative_teacher_loss = (
                                            relative_teacher_loss
                                            + relative_teacher_loss_chunk * chunk_weight
                                        )
                                        frontier_residual_loss = (
                                            frontier_residual_loss
                                            + frontier_residual_loss_chunk * chunk_weight
                                        )
                                        if stats_enabled_chunk and stats_chunk is not None:
                                            stats_enabled = True
                                            if stats is None:
                                                stats = {}
                                            for key, value in stats_chunk.items():
                                                stats[key] = stats.get(key, 0.0) + float(value) * chunk_weight
                                        del batch_chunk
                                        del rgb_prediction_chunk
                            log_image_this_batch = (
                                epoch_visualization_enabled
                                and
                                not image_logged
                                and self.writer is not None
                                and self.world_rank == 0
                                and isinstance(rgb_gen, torch.Tensor)
                                and rgb_gen.dim() == 4
                                and rgb_gen.size(0) > 0
                            )
                            with _stage1_timed_region(
                                batch_timings,
                                "aux_image_logging",
                                enabled=batch_timing_enabled and log_image_this_batch,
                                batch_index=stage1_global_batch_index,
                            ):
                                if log_image_this_batch:
                                    rgb_to_log = torch.clamp(rgb_gen[0], 0.0, 1.0).detach().cpu()
                                    self.writer.add_image(f"{stage_sample_prefix}/generated", rgb_to_log, epoch)
                                    # 保存PNG文件
                                    if self.stage_samples_dir is not None and self._is_main_rank:
                                        self._save_tensor_as_png(
                                            rgb_to_log,
                                            self.stage_samples_dir / f"{stage_sample_prefix}_epoch{epoch:03d}_{aux_mode}_rgb.png",
                                            normalize=False,
                                        )
                                        with torch.no_grad():
                                            x0_debug_rgb = self._decode_predicted_x0(
                                                noisy_latents.detach(),
                                                outputs.noise_pred.detach(),
                                                timesteps.detach(),
                                                track_grad=False,
                                            )
                                            self._save_tensor_as_png(
                                                torch.clamp(x0_debug_rgb[0], 0.0, 1.0).detach().cpu(),
                                                self.stage_samples_dir / f"{stage_sample_prefix}_epoch{epoch:03d}_x0_decode_rgb.png",
                                                normalize=False,
                                            )
                                            full_generate_output = self._generate_sample(
                                                batch,
                                                steps=max(1, int(ddim_steps)),
                                                sampler_type=self.sampler_type,
                                                return_diagnostics=True,
                                            )
                                            full_generate_rgb = None
                                            full_generate_support_composed_rgb = None
                                            if isinstance(full_generate_output, dict):
                                                full_generate_rgb = full_generate_output.get(
                                                    "rgb",
                                                    full_generate_output.get("image"),
                                                )
                                                full_generate_debug_tensors = full_generate_output.get(
                                                    "debug_tensors",
                                                    {},
                                                )
                                                if isinstance(full_generate_debug_tensors, dict):
                                                    full_generate_support_composed_rgb = (
                                                        full_generate_debug_tensors.get(
                                                            "support_composed_image"
                                                        )
                                                    )
                                            elif isinstance(full_generate_output, torch.Tensor):
                                                full_generate_rgb = full_generate_output
                                            if isinstance(full_generate_rgb, torch.Tensor):
                                                self._save_tensor_as_png(
                                                    torch.clamp(full_generate_rgb[0], 0.0, 1.0).detach().cpu(),
                                                    self.stage_samples_dir / f"{stage_sample_prefix}_epoch{epoch:03d}_full_generate_rgb.png",
                                                    normalize=False,
                                                )
                                            if isinstance(full_generate_support_composed_rgb, torch.Tensor):
                                                self._save_tensor_as_png(
                                                    torch.clamp(
                                                        full_generate_support_composed_rgb[0],
                                                        0.0,
                                                        1.0,
                                                    ).detach().cpu(),
                                                    self.stage_samples_dir / f"{stage_sample_prefix}_epoch{epoch:03d}_full_generate_support_composed_rgb.png",
                                                    normalize=False,
                                                )
                                    image_logged = True
                        reconstruction_term = loss_recon.float() + loss_perc.float()
                        if (
                            resolved_frontier_residual_weight > 0.0
                            and self._batch_uses_phase1_3dgs_conditions(batch)
                        ):
                            reconstruction_term = (
                                reconstruction_term * float(resolved_surrogate_reconstruction_decay)
                            )
                        add_losses = add_losses + reconstruction_term
                        if consistency_enabled and lambda_consistency > 0.0:
                            add_losses = add_losses + loss_cons.float() * float(lambda_consistency)
                            if stats_enabled and stats is not None:
                                for key, value in stats.items():
                                    geometry_stats[key] = geometry_stats.get(key, 0.0) + float(value)
                                geometry_calls += 1
                        if current_relative_teacher_effect_weight > 0.0:
                            add_losses = add_losses + (
                                relative_teacher_loss.float() * float(current_relative_teacher_effect_weight)
                            )
                        if (
                            resolved_frontier_residual_weight > 0.0
                            and self._batch_uses_phase1_3dgs_conditions(batch)
                        ):
                            add_losses = add_losses + (
                                frontier_residual_loss.float()
                                * float(resolved_frontier_residual_weight)
                            )
                        if aux_mode != "x0_decode" and resolved_sample_path_weight > 0.0:
                            add_losses = add_losses * float(resolved_sample_path_weight)
                    except Exception as exc:
                        print(f"[Stage1] 附加损失计算失败，已跳过。reason={exc}")
                        is_oom_error = isinstance(exc, RuntimeError) and "CUDA out of memory" in str(exc)
                        if is_oom_error:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            if aux_downscale_factor > min_aux_downscale + 1e-3:
                                prev_factor = aux_downscale_factor
                                aux_downscale_factor = max(aux_downscale_factor * 0.75, min_aux_downscale)
                                scaled_hw = tuple(
                                    max(32, int(max(dim, 1) * aux_downscale_factor)) for dim in target_hw
                                )
                                print(
                                    f"[Stage1] 显存紧张，已将附加损失分辨率降至 {scaled_hw[0]}x{scaled_hw[1]} "
                                    f"(scale={aux_downscale_factor:.2f}, 之前为{prev_factor:.2f})。"
                                )
                                if batch_timing_enabled:
                                    _emit_stage1_timing_summary(
                                        batch_index=stage1_global_batch_index,
                                        batch_timings=batch_timings,
                                        batch_process_start_time=batch_process_start_time,
                                        compute_aux=compute_aux,
                                        aux_mode=aux_mode,
                                        current_loss=float(loss.item()),
                                        note="aux_oom_downscale_retry",
                                    )
                                continue
                            if not aux_interval_backoff:
                                current_ddim_interval = max(current_ddim_interval * 2, current_ddim_interval + 4)
                                aux_interval_backoff = True
                                print(
                                    f"[Stage1] 已将附加损失采样间隔增至 {current_ddim_interval} 步，以降低显存峰值。"
                                )
                            elif current_relative_teacher_effect_weight > 0.0:
                                current_relative_teacher_effect_weight = 0.0
                                print(
                                    "[Stage1] 多次 OOM，已关闭 relative prior effect 项，"
                                    "以降低附加损失显存峰值。"
                                )
                            elif not aux_forced_off:
                                perceptual_fn = None
                                consistency_enabled = False
                                aux_forced_off = True
                                print(
                                    "[Stage1] 多次 OOM，已永久关闭感知/几何一致性项，仅保留重建损失。"
                                )

                # 多GPU模式：对齐add_losses到loss的设备（add_losses可能是Tensor或float）
                if isinstance(add_losses, torch.Tensor):
                    loss = loss + add_losses.to(loss.device)
                else:
                    loss = loss + add_losses
                current_loss_value = float(loss.item())
                
                # NaN检测和跳过机制
                if torch.isnan(loss) or torch.isinf(loss):
                    if self._is_main_rank and not getattr(self, '_nan_warned', False):
                        print(f"[Stage1] Warning: loss is NaN/Inf, skipping batch")
                        self._nan_warned = True
                    if batch_timing_enabled:
                        _emit_stage1_timing_summary(
                            batch_index=stage1_global_batch_index,
                            batch_timings=batch_timings,
                            batch_process_start_time=batch_process_start_time,
                            compute_aux=compute_aux,
                            aux_mode=aux_mode,
                            current_loss=current_loss_value,
                            note="skip_nan_loss",
                        )
                    continue
                
                # 保存参数状态以便恢复
                if not hasattr(self, '_param_backup'):
                    self._param_backup = {}
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self._param_backup[name] = param.data.clone()
                
                with _stage1_timed_region(
                    batch_timings,
                    "backward",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                    emit_progress=True,
                ):
                    self.optimizer.zero_grad()
                    loss.backward()
                
                # 在裁剪前检查梯度是否包含NaN/Inf
                nan_grad_params = []
                inf_grad_params = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        if torch.isnan(param.grad).any():
                            nan_grad_params.append(name)
                        if torch.isinf(param.grad).any():
                            inf_grad_params.append(name)
                
                if nan_grad_params or inf_grad_params:
                    if self._is_main_rank and not getattr(self, '_grad_nan_warned', False):
                        print(f"[Stage1 诊断] 梯度包含NaN的参数: {nan_grad_params[:5]}...")
                        print(f"[Stage1 诊断] 梯度包含Inf的参数: {inf_grad_params[:5]}...")
                        self._grad_nan_warned = True
                    # 将NaN/Inf梯度置零，尝试继续训练
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and param.grad is not None:
                            param.grad = torch.nan_to_num(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 更激进的梯度裁剪
                with _stage1_timed_region(
                    batch_timings,
                    "clip_and_step",
                    enabled=batch_timing_enabled,
                    batch_index=stage1_global_batch_index,
                    emit_progress=True,
                ):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                
                # 检查参数是否变成NaN，如果是则恢复
                has_nan = False
                nan_param_names = []
                for name, param in self.model.named_parameters():
                    if param.requires_grad and (torch.isnan(param.data).any() or torch.isinf(param.data).any()):
                        has_nan = True
                        nan_param_names.append(name)
                
                if has_nan:
                    if self._is_main_rank and not getattr(self, '_param_nan_warned', False):
                        print(f"[Stage1] Warning: parameters contain NaN after update")
                        print(f"[Stage1 诊断] NaN参数列表: {nan_param_names[:10]}...")
                        self._param_nan_warned = True
                    # 恢复参数
                    for name, param in self.model.named_parameters():
                        if name in self._param_backup:
                            param.data.copy_(self._param_backup[name])
                    if batch_timing_enabled:
                        _emit_stage1_timing_summary(
                            batch_index=stage1_global_batch_index,
                            batch_timings=batch_timings,
                            batch_process_start_time=batch_process_start_time,
                            compute_aux=compute_aux,
                            aux_mode=aux_mode,
                            current_loss=current_loss_value,
                            note="restore_nan_params",
                        )
                    continue
                
                epoch_loss += loss.item()
                epoch_batches += 1
                if batch_timing_enabled:
                    _emit_stage1_timing_summary(
                        batch_index=stage1_global_batch_index,
                        batch_timings=batch_timings,
                        batch_process_start_time=batch_process_start_time,
                        compute_aux=compute_aux,
                        aux_mode=aux_mode,
                        current_loss=current_loss_value,
                        note="ok",
                    )
            validation_metrics: Dict[str, float] = {}
            validation_score: Optional[float] = None
            scheduler.step()
            if epoch_validation_enabled:
                validation_metrics = self.validate(
                    generation_mode=resolved_validation_mode,
                    max_batches=resolved_validation_max_batches,
                )
                if validation_metrics:
                    validation_score = float(
                        validation_metrics.get(
                            "validation_score",
                            validation_metrics.get("whole_image_psnr", 0.0),
                        )
                    )
            mean_epoch_loss_for_logging = epoch_loss / max(epoch_batches, 1)
            _log_stage_progress(stage=stage_display_name, epoch=epoch, loss=mean_epoch_loss_for_logging)
            if epoch_batches > 0:
                sums = self._reduce_values(
                    [epoch_loss, diffusion_loss_accum, float(epoch_batches)]
                )
                total_loss, total_diffusion, total_batches = sums
                denom = max(total_batches, 1.0)
                mean_total_loss = total_loss / denom
                mean_diffusion_loss = total_diffusion / denom
                avg_ddim = ddim_compute_count / max(global_step, 1)
                avg_x0 = x0_usage_count / max(global_step, 1)
                avg_adaptive = adaptive_trigger_count / max(global_step, 1)
                stats_message = {
                    "ddim_call_ratio": round(avg_ddim, 4),
                    "x0_usage_ratio": round(avg_x0, 4),
                    "adaptive_trigger_ratio": round(avg_adaptive, 4),
                }
                if geometry_calls > 0:
                    stats_message.update(
                        {
                            key: round(val / geometry_calls, 4)
                            for key, val in geometry_stats.items()
                        }
                    )
                if self.writer is not None:
                    writer_stage_prefix = stage_sample_prefix
                    for stat_name, stat_value in stats_message.items():
                        self.writer.add_scalar(f"{writer_stage_prefix}/{stat_name}", float(stat_value), epoch)
                    self.writer.add_scalar(f"{writer_stage_prefix}/loss", float(mean_total_loss), epoch)
                    self.writer.add_scalar(f"{writer_stage_prefix}/diffusion_loss", float(mean_diffusion_loss), epoch)
                    self._write_validation_metrics(
                        writer_prefix=writer_stage_prefix,
                        validation_metrics=validation_metrics,
                        epoch=epoch,
                    )
                if self._is_main_rank and self.verbose_epoch_stats:
                    print(f"[{stage_display_name}] Epoch {epoch} stats: {stats_message}")
            if self._is_main_rank and validation_metrics:
                print(
                    f"[{stage_display_name}] Epoch {epoch} validation: "
                    f"{self._format_validation_metrics_for_log(validation_metrics)}"
                )
            (
                early_stop_best_metric,
                early_stop_plateau_count,
                should_stop_early,
            ) = self._update_early_stop_state(
                stage_name=stage_display_name,
                metric_name="validation_score",
                current_metric=validation_score,
                best_metric=early_stop_best_metric,
                plateau_count=early_stop_plateau_count,
                epoch=epoch,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if should_stop_early:
                break

    def stage2_calibrate(self, num_epochs: int = 5, lr: float = 1e-4) -> None:
        """
        Stage 2: attach uncertainty heads and calibrate them with MC-Dropout.
        """
        self.raw_model.add_uncertainty_heads()
        
        # 兼容不同的不确定性头结构:
        # - GDDN原版: epistemic_head, aleatoric_head
        # - GDDN_ControlNet: uncertainty_head.epistemic_net, uncertainty_head.aleatoric_net
        uncertainty_param_patterns = [
            "epistemic_head", "aleatoric_head",  # GDDN原版
            "uncertainty_head",  # GDDN_ControlNet
        ]
        
        for name, param in self.model.named_parameters():
            is_uncertainty_param = any(pattern in name for pattern in uncertainty_param_patterns)
            if not is_uncertainty_param:
                param.requires_grad = False
            else:
                param.requires_grad = True

        # 收集不确定性相关参数
        params = []
        if hasattr(self.raw_model, 'epistemic_head') and self.raw_model.epistemic_head is not None:
            params.extend(list(self.raw_model.epistemic_head.parameters()))
        if hasattr(self.raw_model, 'aleatoric_head') and self.raw_model.aleatoric_head is not None:
            params.extend(list(self.raw_model.aleatoric_head.parameters()))
        # 兼容GDDN_ControlNet的uncertainty_head
        if hasattr(self.raw_model, 'uncertainty_head') and self.raw_model.uncertainty_head is not None:
            params.extend(list(self.raw_model.uncertainty_head.parameters()))
        
        # 去重（避免重复添加参数）
        seen_ids = set()
        unique_params = []
        for p in params:
            if id(p) not in seen_ids:
                seen_ids.add(id(p))
                unique_params.append(p)
        
        self.optimizer = torch.optim.AdamW(unique_params, lr=lr)
        early_stop_best_metric: Optional[float] = None
        early_stop_plateau_count = 0

        for epoch in range(num_epochs):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.eval()
            
            # DistributedSampler set_epoch
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            assert self.raw_model.epistemic_head is not None
            assert self.raw_model.aleatoric_head is not None
            self.raw_model.epistemic_head.train()
            self.raw_model.aleatoric_head.train()
            epoch_loss = 0.0
            epoch_psnr = 0.0
            epoch_surrogate_psnr = 0.0
            epoch_render_only_psnr = 0.0
            epoch_sample_surrogate_gap = 0.0
            epoch_ece = 0.0
            epoch_corr = 0.0
            epoch_projection_coverage = 0.0
            epoch_exist_regret_vs_render = 0.0
            metric_batches = 0
            scatter_logged = False
            image_logged = False
            for batch_index, batch in enumerate(self.train_dataloader):
                batch = self._to_device(batch)
                timesteps, _, noisy_latents = self._sample_diffusion_inputs(batch)

                outputs = self.model(
                    sparse_images=batch["sparse_images"],
                    sparse_poses=batch["sparse_poses"],
                    target_pose=batch["target_pose"],
                    rgb_render=batch["rgb_render"],
                    depth_render=batch["depth_render"],
                    normal_render=batch["normal_render"],
                    reference_depth_map=batch.get("reference_depth_map"),
                    reference_intrinsics=batch.get("reference_intrinsics"),
                    target_intrinsics=batch.get("target_intrinsics"),
                    sparse_intrinsics=batch.get("sparse_intrinsics"),
                    support_confidence=batch.get("warp_support_confidence"),
                    accum_render=batch.get("accum_render"),
                    transmittance_render=batch.get("transmittance_render"),
                    timesteps=timesteps,
                    noisy_latents=noisy_latents,
                    return_uncertainty=True,
                )
                assert outputs.uncertainty_epistemic is not None
                assert outputs.uncertainty_aleatoric is not None
                mean_rgb = self._select_teacher_prediction_rgb(outputs, batch)
                if mean_rgb is None:
                    mean_rgb = self._decode_predicted_x0(
                        noisy_latents=noisy_latents,
                        noise_pred=outputs.noise_pred,
                        timesteps=timesteps,
                    )
                mean_rgb = torch.nan_to_num(mean_rgb, nan=0.0, posinf=1.0, neginf=0.0)
                final_sample_reference = None
                final_sample_config = self._resolve_final_sample_supervision_config(
                    batch,
                    base_steps=self.stage2_final_sample_target_steps,
                )
                if (
                    self.stage2_final_sample_target_interval > 0
                    and batch_index % max(self.stage2_final_sample_target_interval, 1) == 0
                ):
                    final_sample_reference = self._generate_final_sample_reference(
                        batch,
                        steps=max(1, int(final_sample_config["steps"] or self.stage2_final_sample_target_steps)),
                    )
                teacher_effect_reference_rgb = (
                    final_sample_reference
                    if (
                        self.use_final_sample_for_teacher_effect_targets
                        and final_sample_reference is not None
                    )
                    else mean_rgb.detach()
                )
                effect_targets = self._compute_teacher_effect_targets(
                    teacher_rgb=teacher_effect_reference_rgb,
                    render_rgb=batch["rgb_render"],
                    target_rgb=batch["target_rgb"],
                    depth_render=batch.get("depth_render"),
                    support_mask=batch.get("warp_valid_mask"),
                    support_confidence=batch.get("warp_support_confidence"),
                    use_projection_aligned_support=True,
                )
                epistemic_target = effect_targets["risk_exist_target"]
                aleatoric_target = effect_targets["gain_novel_target"]
                exist_mask = effect_targets["exist_mask"]
                novel_mask = effect_targets["novel_mask"]
                metric_mask = None if novel_mask is not None else exist_mask

                unc_epistemic = torch.nan_to_num(
                    outputs.uncertainty_epistemic, nan=0.0, posinf=1.0, neginf=0.0
                )
                unc_aleatoric = torch.nan_to_num(
                    outputs.uncertainty_aleatoric, nan=0.0, posinf=1.0, neginf=0.0
                )
                # 尺寸对齐到 MC 目标尺寸
                target_hw = epistemic_target.shape[-2:]
                if unc_epistemic.shape[-2:] != target_hw:
                    unc_epistemic = F.interpolate(
                        unc_epistemic, size=target_hw, mode="bilinear", align_corners=False
                    )
                if unc_aleatoric.shape[-2:] != target_hw:
                    unc_aleatoric = F.interpolate(
                        unc_aleatoric, size=target_hw, mode="bilinear", align_corners=False
                    )
                # 尺寸对齐：将预测 effect maps 插值到目标尺寸
                target_hw = epistemic_target.shape[-2:]
                if unc_epistemic.shape[-2:] != target_hw:
                    unc_epistemic = F.interpolate(
                        unc_epistemic, size=target_hw, mode="bilinear", align_corners=False
                    )
                if unc_aleatoric.shape[-2:] != target_hw:
                    unc_aleatoric = F.interpolate(
                        unc_aleatoric, size=target_hw, mode="bilinear", align_corners=False
                    )
                loss_epist = self._calibration_loss(
                    unc_epistemic,
                    epistemic_target,
                    exist_mask,
                )
                loss_aleat = self._calibration_loss(
                    unc_aleatoric,
                    aleatoric_target,
                    novel_mask,
                )
                
                # Variance preservation 正则化：惩罚 uncertainty 输出方差过小
                unc_epist_var = unc_epistemic.var()
                unc_aleat_var = unc_aleatoric.var()
                var_preserve_loss = (
                    F.relu(self.min_pred_var - unc_epist_var) + 
                    F.relu(self.min_pred_var - unc_aleat_var)
                )
                
                loss = loss_epist + loss_aleat + self.lambda_var_preserve * var_preserve_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
                surrogate_psnr_value = self._compute_psnr_from_rgb(mean_rgb, batch["target_rgb"])
                metric_rgb = (
                    final_sample_reference
                    if self.stage2_primary_metric_use_final_sample and final_sample_reference is not None
                    else mean_rgb
                )
                psnr_value = self._compute_psnr_from_rgb(metric_rgb, batch["target_rgb"])
                render_only_psnr_value = self._compute_render_only_psnr_from_batch(batch)
                sample_surrogate_gap_value = psnr_value - surrogate_psnr_value
                epoch_psnr += float(psnr_value.item())
                epoch_surrogate_psnr += float(surrogate_psnr_value.item())
                epoch_render_only_psnr += float(render_only_psnr_value.item())
                epoch_sample_surrogate_gap += float(sample_surrogate_gap_value.item())
                projection_coverage_value = self._estimate_batch_support_coverage(batch)
                if projection_coverage_value is not None:
                    epoch_projection_coverage += float(projection_coverage_value)
                exist_regret_vs_render_value = self._compute_exist_regret_vs_render(
                    teacher_rgb=metric_rgb.detach(),
                    render_rgb=batch["rgb_render"],
                    target_rgb=batch["target_rgb"],
                    depth_render=batch.get("depth_render"),
                    support_mask=batch.get("warp_valid_mask"),
                    support_confidence=batch.get("warp_support_confidence"),
                )
                if exist_regret_vs_render_value is not None:
                    epoch_exist_regret_vs_render += float(
                        exist_regret_vs_render_value.item()
                    )
                
                # ========== 诊断日志：Stage2 RGB范围检查 ==========
                total_uncertainty = unc_epistemic + unc_aleatoric
                total_target = epistemic_target + aleatoric_target
                error_map = torch.abs(metric_rgb - batch["target_rgb"]).mean(dim=1, keepdim=True)
                pred_std = torch.sqrt(torch.clamp(total_uncertainty, min=0.0))
                batch_ece = self._expected_calibration_error(
                    total_uncertainty,
                    total_target,
                    metric_mask,
                )
                epoch_ece += batch_ece
                corr_value = self._uncertainty_error_correlation(pred_std, error_map, metric_mask)
                epoch_corr += corr_value
                if self.writer is not None and self.world_rank == 0:
                    if not scatter_logged:
                        self._log_uncertainty_scatter(
                            stage_tag="stage2b",
                            epoch=epoch,
                            prediction_std=pred_std,
                            error_map=error_map,
                            mask=metric_mask,
                        )
                        scatter_logged = True
                    if not image_logged:
                        self._log_uncertainty_heatmaps(
                            stage_tag="stage2b",
                            epoch=epoch,
                            error_map=error_map,
                            epistemic=unc_epistemic,
                            aleatoric=unc_aleatoric,
                        )
                        rgb_to_log = torch.clamp(metric_rgb[0], 0.0, 1.0).detach().cpu()
                        self.writer.add_image(
                            "stage2b/generated",
                            rgb_to_log,
                            epoch,
                        )
                        # 保存PNG文件
                        if self.stage_samples_dir is not None and self._is_main_rank:
                            self._save_tensor_as_png(
                                rgb_to_log,
                                self.stage_samples_dir / f"stage2b_epoch{epoch:03d}_rgb.png",
                                normalize=False,
                            )
                        image_logged = True
                metric_batches += 1
            mean_epoch_loss_for_logging = epoch_loss / max(metric_batches, 1)
            _log_stage_progress(stage="Stage2", epoch=epoch, loss=mean_epoch_loss_for_logging)
            if metric_batches > 0:
                metric_sums = self._reduce_values(
                    [
                        epoch_psnr,
                        epoch_surrogate_psnr,
                        epoch_render_only_psnr,
                        epoch_sample_surrogate_gap,
                        epoch_ece,
                        epoch_corr,
                        epoch_projection_coverage,
                        epoch_exist_regret_vs_render,
                        float(metric_batches),
                    ]
                )
                (
                    total_psnr,
                    total_surrogate_psnr,
                    total_render_only_psnr,
                    total_sample_surrogate_gap,
                    total_ece,
                    total_corr,
                    total_projection_coverage,
                    total_exist_regret_vs_render,
                    total_batches,
                ) = metric_sums
                denom = max(total_batches, 1.0)
                avg_psnr = total_psnr / denom
                avg_surrogate_psnr = total_surrogate_psnr / denom
                avg_render_only_psnr = total_render_only_psnr / denom
                avg_sample_surrogate_gap = total_sample_surrogate_gap / denom
                avg_ece = total_ece / denom
                avg_corr = total_corr / denom
                avg_projection_coverage = total_projection_coverage / denom
                avg_exist_regret_vs_render = total_exist_regret_vs_render / denom
                
                if self._is_main_rank:
                    print(
                        f"[Stage2] Epoch {epoch} metrics: "
                        f"sample_psnr={avg_psnr:.2f}dB, surrogate_psnr={avg_surrogate_psnr:.2f}dB, "
                        f"render_only_psnr={avg_render_only_psnr:.2f}dB, "
                        f"sample_surrogate_gap={avg_sample_surrogate_gap:.2f}dB, "
                        f"ece={avg_ece:.4f}, corr={avg_corr:.4f}, "
                        f"projection_coverage_ratio={avg_projection_coverage:.4f}, "
                        f"exist_regret_vs_render={avg_exist_regret_vs_render:.4f}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("stage2b/psnr", avg_psnr, epoch)
                        self.writer.add_scalar("stage2b/surrogate_psnr", avg_surrogate_psnr, epoch)
                        self.writer.add_scalar("stage2b/render_only_psnr", avg_render_only_psnr, epoch)
                        self.writer.add_scalar("stage2b/sample_surrogate_gap", avg_sample_surrogate_gap, epoch)
                        self.writer.add_scalar("stage2b/ece", avg_ece, epoch)
                        self.writer.add_scalar("stage2b/uncert_error_corr", avg_corr, epoch)
                        self.writer.add_scalar(
                            "stage2b/projection_coverage_ratio",
                            avg_projection_coverage,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage2b/exist_regret_vs_render",
                            avg_exist_regret_vs_render,
                            epoch,
                        )
                    self._log_fixed_visualization_runtime_metrics(
                        stage_tag="stage2b",
                        epoch=epoch,
                        base_steps=self.stage2_final_sample_target_steps,
                    )
                (
                    early_stop_best_metric,
                    early_stop_plateau_count,
                    should_stop_early,
                ) = self._update_early_stop_state(
                    stage_name="Stage2",
                    metric_name="sample_psnr",
                    current_metric=float(avg_psnr),
                    best_metric=early_stop_best_metric,
                    plateau_count=early_stop_plateau_count,
                    epoch=epoch,
                )
                if should_stop_early:
                    break
                 
                # 保存Stage 2最终ECE快照，用于Stage 3对比
                self.stage2_final_ece = avg_ece
        
        # Stage 2结束时打印快照信息
        if self._is_main_rank and hasattr(self, 'stage2_final_ece'):
            print(f"\n[Stage2 完成] 最终ECE={self.stage2_final_ece:.4f}，此值将用于Stage3监控参考\n")

    def stage3_finetune(
        self,
        num_epochs: int = 5,
        lr: float = 5e-6,
        lambda_distill: float = 0.05,
        recon_weight: Optional[float] = None,
        recon_alpha: Optional[float] = None,
        perceptual_weight: Optional[float] = None,
    ) -> None:
        """
        Stage 3: jointly fine-tune backbone and uncertainty heads with uncertainty consistency.
        """
        if self.raw_model.epistemic_head is None or self.raw_model.aleatoric_head is None:
            self.raw_model.add_uncertainty_heads()
        if self.geometry_aware_trainable_name_tokens:
            self._apply_geometry_aware_trainable_filter(include_uncertainty=True)
        else:
            for param in self.model.parameters():
                param.requires_grad = True

        estimated_support_coverage = self._estimate_train_support_coverage(max_batches=2)
        self._stage3_freeze_controlnet = True
        if self.stage3_force_controlnet_trainable:
            self._stage3_freeze_controlnet = False
        elif (
            self.stage3_adaptive_controlnet_freeze
            and estimated_support_coverage is not None
            and estimated_support_coverage < self.stage3_freeze_controlnet_min_support_coverage
        ):
            self._stage3_freeze_controlnet = False
        if hasattr(self.raw_model, "controlnet") and self.raw_model.controlnet is not None:
            for param in self.raw_model.controlnet.parameters():
                param.requires_grad = not self._stage3_freeze_controlnet
            if self.world_rank == 0:
                if self._stage3_freeze_controlnet:
                    print("[Stage3] ControlNet 已冻结以提升训练稳定性。")
                else:
                    print(
                        "[Stage3] 检测到support覆盖不足，ControlNet保持可训练，"
                        f"estimated_support_coverage={estimated_support_coverage:.4f}"
                    )
        
        # Stage3强制使用fp32：避免mixed precision导致的dtype不匹配
        # 强制转换整个模型（包括所有子模块如ControlNet、UNet、VAE）为fp32
        # 不依赖dtype检查，因为不同子模块可能有不同dtype
        if self.world_rank == 0:
            print("[Stage3] 强制将整个模型转换为 float32...")
        self.raw_model.to(dtype=torch.float32)
        
        # 验证关键子模块的dtype
        if self.world_rank == 0:
            unet_dtype = next(self.raw_model.unet.parameters()).dtype if hasattr(self.raw_model, 'unet') else 'N/A'
            cn_dtype = next(self.raw_model.controlnet.parameters()).dtype if hasattr(self.raw_model, 'controlnet') and self.raw_model.controlnet is not None else 'N/A'
            vae_dtype = next(self.raw_model.vae.parameters()).dtype if hasattr(self.raw_model, 'vae') else 'N/A'
            print(f"[Stage3] dtype验证: UNet={unet_dtype}, ControlNet={cn_dtype}, VAE={vae_dtype}")

        recon_weight = (
            self.stage3_recon_weight if recon_weight is None else float(max(recon_weight, 0.0))
        )
        recon_alpha = (
            self.stage3_recon_alpha if recon_alpha is None else float(min(max(recon_alpha, 0.0), 1.0))
        )
        perceptual_weight = (
            self.stage3_perceptual_weight
            if perceptual_weight is None
            else float(max(perceptual_weight, 0.0))
        )
        perceptual_fn: Optional[_PerceptualLoss] = None
        if perceptual_weight > 0.0:
            perceptual_fn = _PerceptualLoss(device=str(self._resolve_auxiliary_loss_device()))
            if not perceptual_fn.available:
                if self.world_rank == 0:
                    print("[Stage3] VGG 权重不可用，Stage3 感知损失已自动关闭。")
                perceptual_fn = None
                perceptual_weight = 0.0

        resolved_stage3_frontier_objective_config = (
            self._resolve_frontier_proposal_objective_config("stage3")
        )
        resolved_stage3_frontier_residual_weight = float(
            resolved_stage3_frontier_objective_config["frontier_residual_weight"]
        )
        resolved_stage3_surrogate_reconstruction_decay = float(
            resolved_stage3_frontier_objective_config["surrogate_reconstruction_decay"]
        )
        resolved_stage3_exist_diffusion_weight = resolved_stage3_frontier_objective_config[
            "exist_diffusion_weight"
        ]
        resolved_stage3_novel_diffusion_weight = resolved_stage3_frontier_objective_config[
            "novel_diffusion_weight"
        ]
        resolved_stage3_proposal_warp_error_supervision_weight = max(
            float(getattr(self, "proposal_warp_error_supervision_weight", 0.0)),
            0.0,
        )
        resolved_stage3_proposal_repairability_supervision_weight = max(
            float(getattr(self, "proposal_repairability_supervision_weight", 0.0)),
            0.0,
        )
        resolved_stage3_proposal_verification_supervision_weight = max(
            float(getattr(self, "proposal_verification_supervision_weight", 0.0)),
            0.0,
        )
        resolved_stage3_proposal_acceptance_supervision_weight = max(
            float(getattr(self, "proposal_acceptance_supervision_weight", 0.0)),
            0.0,
        )
        if self.world_rank == 0:
            print(
                "[Stage3] frontier proposal objective: "
                f"frontier_residual_weight={resolved_stage3_frontier_residual_weight:.4f}, "
                "surrogate_reconstruction_decay="
                f"{resolved_stage3_surrogate_reconstruction_decay:.4f}, "
                "exist_diffusion_weight="
                f"{resolved_stage3_exist_diffusion_weight if resolved_stage3_exist_diffusion_weight is not None else 'default'}, "
                "novel_diffusion_weight="
                f"{resolved_stage3_novel_diffusion_weight if resolved_stage3_novel_diffusion_weight is not None else 'default'}, "
                "proposal_warp_error_weight="
                f"{resolved_stage3_proposal_warp_error_supervision_weight:.4f}, "
                "proposal_repairability_weight="
                f"{resolved_stage3_proposal_repairability_supervision_weight:.4f}, "
                "proposal_verification_weight="
                f"{resolved_stage3_proposal_verification_supervision_weight:.4f}, "
                "proposal_acceptance_weight="
                f"{resolved_stage3_proposal_acceptance_supervision_weight:.4f}"
            )

        self._auto_unfreeze_candidates = []
        self._auto_unfreeze_activated = False
        self._stage3_metric_history = []
        self._stage3_psnr_history = []
        vae_param_pool: List[torch.nn.Parameter] = []
        vae_status_msg: Optional[str] = None
        if hasattr(self.raw_model, "vae") and self.raw_model.vae is not None:
            vae_module = self.raw_model.vae
            manual_unfreeze = bool(self.unfreeze_vae)
            if manual_unfreeze:
                if self.unfreeze_vae_decoder_only:
                    if hasattr(vae_module, "encoder"):
                        for param in vae_module.encoder.parameters():
                            param.requires_grad = False
                    if hasattr(vae_module, "quant_conv") and getattr(vae_module, "quant_conv") is not None:
                        for param in vae_module.quant_conv.parameters():
                            param.requires_grad = False
                    if hasattr(vae_module, "decoder"):
                        for param in vae_module.decoder.parameters():
                            param.requires_grad = True
                            vae_param_pool.append(param)
                    if hasattr(vae_module, "post_quant_conv") and getattr(vae_module, "post_quant_conv") is not None:
                        for param in vae_module.post_quant_conv.parameters():
                            param.requires_grad = True
                            vae_param_pool.append(param)
                    vae_status_msg = "[Stage3] 按需解冻：仅开放 VAE 解码路径，已自动缩放学习率。"
                else:
                    for param in vae_module.parameters():
                        param.requires_grad = True
                        vae_param_pool.append(param)
                    vae_status_msg = "[Stage3] 已按配置解冻全部 VAE 参数，请注意显存与稳定性。"
            else:
                for param in vae_module.parameters():
                    param.requires_grad = False
                if self.auto_unfreeze_vae:
                    self._auto_unfreeze_candidates = self._collect_vae_params(
                        vae_module, decoder_only=self.auto_unfreeze_decoder_only
                    )
                    vae_status_msg = (
                        "[Stage3] 当前冻结 VAE，已启用自动解冻监控：等待指标稳定后解冻以提升画质。"
                    )
                else:
                    vae_status_msg = "[Stage3] 出于显存考虑，VAE 参数仍保持冻结（仅优化 UNet/条件分支与不确定性头）。"
        if vae_status_msg and self.world_rank == 0:
            print(vae_status_msg)

        vae_param_ids = {id(param) for param in vae_param_pool}
        base_params: List[torch.nn.Parameter] = []
        vae_params: List[torch.nn.Parameter] = []
        for param in self.model.parameters():
            if not param.requires_grad:
                continue
            if vae_param_ids and id(param) in vae_param_ids:
                vae_params.append(param)
            else:
                base_params.append(param)
        optimizer_groups = []
        if base_params:
            optimizer_groups.append({"params": base_params, "weight_decay": 0.01})
        if vae_params:
            scaled_lr = float(lr * max(self.stage3_vae_lr_scale, 0.0))
            if scaled_lr <= 0.0:
                scaled_lr = lr
            optimizer_groups.append(
                {
                    "params": vae_params,
                    "lr": scaled_lr,
                    "weight_decay": float(max(self.stage3_vae_weight_decay, 0.0)),
                }
            )
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=lr,
            weight_decay=0.0,
        )
        self.stage3_base_lr = lr
        trainable_params_fp32 = self._trainable_params_are_fp32()
        autocast_enabled = self._amp_enabled
        
        # Stage3禁用autocast以避免dtype不匹配问题
        # 混合精度训练在Stage3存在冻结模块时会导致forward/backward dtype冲突
        # 使用纯fp32训练虽然显存占用略高，但更稳定
        if autocast_enabled and self._is_main_rank:
            print("[Stage3] 检测到混合精度模式，为避免dtype冲突，Stage3将使用纯fp32训练。")
        autocast_enabled = False  # 强制禁用Stage3的autocast
        
        grad_scaler: Optional[torch.cuda.amp.GradScaler]
        if autocast_enabled and trainable_params_fp32:
            grad_scaler = torch.cuda.amp.GradScaler(enabled=True)
        else:
            grad_scaler = None
        can_unscale_grads = grad_scaler is not None

        # ========== IBC: 交替双层校准 ==========
        # Stage 3采用IBC策略：
        # - 外层（慢）：每个epoch更新扩散模型backbone
        # - 内层（快）：epoch结束后快速重校准不确定性头
        if self._is_main_rank:
            if self.ibc_enabled:
                print(f"[Stage3] IBC双层优化已启用: 内层校准步数={self.ibc_inner_steps}, 内层lr={self.ibc_inner_lr}")
            else:
                print("[Stage3] IBC已禁用，使用传统联合训练")
        
        # 创建不确定性头专用优化器（用于IBC内层）
        uncertainty_params = []
        for name, param in self.model.named_parameters():
            if any(kw in name for kw in ["epistemic", "aleatoric", "uncertainty"]):
                uncertainty_params.append(param)
        
        ibc_inner_optimizer = None
        if self.ibc_enabled and uncertainty_params:
            ibc_inner_optimizer = torch.optim.AdamW(
                uncertainty_params,
                lr=self.ibc_inner_lr,
                weight_decay=0.0,
            )
        early_stop_best_metric: Optional[float] = None
        early_stop_plateau_count = 0

        for epoch in range(num_epochs):
            self._current_stage3_epoch = int(epoch)
            self._current_proposal_effect_epoch = int(epoch)
            # ===== IBC外层：仅训练扩散模型backbone =====
            # 冻结不确定性头，只更新backbone
            if self.ibc_enabled:
                for name, param in self.model.named_parameters():
                    is_uncertainty = any(kw in name for kw in ["epistemic", "aleatoric", "uncertainty"])
                    # 外层：backbone可训练，不确定性头冻结
                    allow_backbone = self._parameter_matches_geometry_aware_filter(name)
                    param.requires_grad = bool(allow_backbone and not is_uncertainty)
                if epoch == 0 and self._is_main_rank:
                    print("[IBC外层] 冻结不确定性头，仅训练backbone")
            else:
                # 非IBC模式：所有参数可训练（除VAE/ControlNet）
                if self.geometry_aware_trainable_name_tokens:
                    self._apply_geometry_aware_trainable_filter(include_uncertainty=True)
                else:
                    for param in self.model.parameters():
                        param.requires_grad = True
            
            # 保持VAE和ControlNet冻结
            if hasattr(self.raw_model, "vae") and self.raw_model.vae is not None:
                for param in self.raw_model.vae.parameters():
                    param.requires_grad = False
            if hasattr(self.raw_model, "controlnet") and self.raw_model.controlnet is not None:
                for param in self.raw_model.controlnet.parameters():
                    param.requires_grad = not self._stage3_freeze_controlnet

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model.train()
            
            if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            current_lr = self.stage3_base_lr
            if self.stage3_final_lr is not None and self.stage3_final_epochs > 0:
                if epoch >= max(num_epochs - self.stage3_final_epochs, 0):
                    current_lr = self.stage3_final_lr
                else:
                    current_lr = self.stage3_base_lr
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
            epoch_loss = 0.0
            epoch_psnr = 0.0
            epoch_surrogate_psnr = 0.0
            epoch_render_only_psnr = 0.0
            epoch_sample_surrogate_gap = 0.0
            epoch_ece = 0.0
            epoch_corr = 0.0
            epoch_projection_coverage = 0.0
            epoch_exist_regret_vs_render = 0.0
            epoch_variance = 0.0
            epoch_proposal_warp_error_loss = 0.0
            epoch_proposal_repairability_loss = 0.0
            epoch_proposal_verification_loss = 0.0
            epoch_proposal_acceptance_loss = 0.0
            metric_batches = 0
            scatter_logged = False
            image_logged = False
            for batch in self.train_dataloader:
                batch = self._to_device(batch)
                recompute_latent = self.stage3_downscale_factor < 0.999
                batch, stage3_hw = self._maybe_downscale_batch(
                    batch,
                    self.stage3_downscale_factor,
                    recompute_latent=recompute_latent,
                )
                timesteps, diffusion_target, noisy_latents = self._sample_diffusion_inputs(batch)
                self.optimizer.zero_grad()
                exist_mask, novel_mask = self._build_projection_aligned_exist_novel_masks(
                    support_mask=batch.get("warp_valid_mask"),
                    support_confidence=batch.get("warp_support_confidence"),
                    target_hw=batch["target_rgb"].shape[-2:],
                )
                metric_mask = None if novel_mask is not None else exist_mask
                
                # IBC外层：正常计算diffusion loss并backward
                # 第一次前向：仅计算扩散损失，避免长时间保留不确定性图激活
                # 确保所有输入都是float32（避免autocast或之前操作产生的Half）
                _f32 = torch.float32
                with torch.cuda.amp.autocast(dtype=self._amp_dtype, enabled=autocast_enabled):
                    diffusion_outputs = self.model(
                        sparse_images=batch["sparse_images"].to(dtype=_f32),
                        sparse_poses=batch["sparse_poses"].to(dtype=_f32),
                        target_pose=batch["target_pose"].to(dtype=_f32),
                        rgb_render=batch["rgb_render"].to(dtype=_f32),
                        depth_render=batch["depth_render"].to(dtype=_f32),
                        normal_render=batch["normal_render"].to(dtype=_f32),
                        reference_depth_map=batch.get("reference_depth_map"),
                        reference_intrinsics=batch.get("reference_intrinsics"),
                        target_intrinsics=batch.get("target_intrinsics"),
                        sparse_intrinsics=batch.get("sparse_intrinsics"),
                        support_confidence=batch.get("warp_support_confidence"),
                        timesteps=timesteps,
                        noisy_latents=noisy_latents.to(dtype=_f32),
                        return_uncertainty=False,
                    )
                    # 多GPU模式：对齐noise到noise_pred的设备
                    noise_device = diffusion_outputs.noise_pred.device
                    loss_diffusion = self._masked_diffusion_loss(
                        diffusion_outputs.noise_pred,
                        diffusion_target.to(device=noise_device, dtype=_f32),
                        batch.get("depth_render"),
                        support_mask=batch.get("warp_valid_mask"),
                        support_confidence=batch.get("warp_support_confidence"),
                        timesteps=timesteps,
                        exist_diffusion_weight=resolved_stage3_exist_diffusion_weight,
                        novel_diffusion_weight=resolved_stage3_novel_diffusion_weight,
                    )
                    
                    # VPDT: 方差保持损失
                    vpdt_loss = self._vpdt_variance_loss(
                        diffusion_outputs.noise_pred,
                        target_std=self.vpdt_target_std,
                    )
                    
                    variance_term = torch.zeros_like(loss_diffusion)
                    variance_loss_value = 0.0
                    if self.stage3_variance_weight > 0.0 and diffusion_outputs.noise_pred is not None:
                        # Determine Source and Target
                        target_std: torch.Tensor
                        variance_source_tensor: Optional[torch.Tensor] = None

                        if self.stage3_variance_source == "rgb":
                            variance_source_tensor = self._decode_predicted_x0(
                                noisy_latents=noisy_latents,
                                noise_pred=diffusion_outputs.noise_pred,
                                timesteps=timesteps,
                                track_grad=True,
                            )
                            # Adaptive Target: Match Ground Truth RGB Variance (per-image)
                            with torch.no_grad():
                                target_std = batch["target_rgb"].std(dim=[1, 2, 3], keepdim=True)
                        else:
                            variance_source_tensor = diffusion_outputs.noise_pred
                            # Fixed Target for Noise: N(0, 1) -> std should be 1.0
                            # Use configured target (default 1.0)
                            target_std = torch.full(
                                (diffusion_outputs.noise_pred.shape[0], 1, 1, 1),
                                self.stage3_variance_target,
                                device=self.device,
                                dtype=diffusion_outputs.noise_pred.dtype
                            )

                        if variance_source_tensor is not None:
                            # Compute Predicted Std [B, 1, 1, 1]
                            pred_std = torch.std(
                                variance_source_tensor.float(),
                                dim=[1, 2, 3],
                                keepdim=True,
                                unbiased=False
                            )

                            # Loss: MSE between predicted std and target std
                            variance_loss_element = F.mse_loss(pred_std, target_std.to(pred_std.dtype))
                            variance_term = variance_loss_element.to(loss_diffusion.dtype)

                            variance_loss_value = float(
                                (variance_term.detach() * self.stage3_variance_weight).item()
                            )
                
                total_diffusion_loss = loss_diffusion + variance_term * self.stage3_variance_weight + self.lambda_vpdt * vpdt_loss
                final_sample_reference = None
                final_sample_config = self._resolve_final_sample_supervision_config(
                    batch,
                    base_steps=self.final_sample_supervision_steps,
                    base_weight=self.final_sample_supervision_weight,
                )
                effective_final_sample_steps = max(
                    1,
                    int(final_sample_config["steps"] or self.final_sample_supervision_steps),
                )
                effective_final_sample_weight = float(
                    final_sample_config["weight"]
                    if final_sample_config["weight"] is not None
                    else self.final_sample_supervision_weight
                )
                should_apply_final_sample_supervision = (
                    effective_final_sample_weight > 0.0
                    and self.final_sample_supervision_interval > 0
                    and (self._stage3_step % max(self.final_sample_supervision_interval, 1) == 0)
                    and diffusion_outputs.noise_pred is not None
                )
                should_generate_metric_final_sample = bool(
                    self.stage3_primary_metric_use_final_sample
                )
                if should_generate_metric_final_sample or should_apply_final_sample_supervision:
                    final_sample_reference = self._generate_final_sample_reference(
                        batch,
                        steps=effective_final_sample_steps,
                    )
                    if should_apply_final_sample_supervision and final_sample_reference is not None:
                        stage3_predicted_rgb = self._decode_predicted_x0(
                            noisy_latents=noisy_latents,
                            noise_pred=diffusion_outputs.noise_pred,
                            timesteps=timesteps,
                            track_grad=True,
                        )
                        final_sample_guided_term = self._compute_final_sample_guided_reconstruction_loss(
                            prediction_rgb=stage3_predicted_rgb,
                            target_rgb=batch["target_rgb"],
                            base_rgb=self._get_support_anchor_rgb_from_batch(batch),
                            final_sample_rgb=final_sample_reference,
                            support_mask=batch.get("warp_valid_mask"),
                            support_confidence=batch.get("warp_support_confidence"),
                            novel_mask=novel_mask,
                        )
                        total_diffusion_loss = total_diffusion_loss + (
                            final_sample_guided_term.to(
                                device=total_diffusion_loss.device,
                                dtype=total_diffusion_loss.dtype,
                            )
                            * effective_final_sample_weight
                        )
                
                if grad_scaler is not None:
                    grad_scaler.scale(total_diffusion_loss).backward()
                else:
                    total_diffusion_loss.backward()
                self._detach_parameter_grads()
                epoch_variance += variance_loss_value
                
                if hasattr(diffusion_outputs, "noise_pred"):
                    diffusion_outputs.noise_pred = None  # type: ignore[assignment]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # IBC外层：跳过不确定性损失的计算和backward
                # 不确定性校准只在IBC内层进行
                if self.ibc_enabled:
                    # IBC外层除 diffusion loss 外，继续显式优化 proposal 相关目标，
                    # 避免 Stage2a 学到的 frontier/proposal 能力在 Stage3 被冲掉。
                    outputs = self.model(
                        sparse_images=batch["sparse_images"].to(dtype=_f32),
                        sparse_poses=batch["sparse_poses"].to(dtype=_f32),
                        target_pose=batch["target_pose"].to(dtype=_f32),
                        rgb_render=batch["rgb_render"].to(dtype=_f32),
                        depth_render=batch["depth_render"].to(dtype=_f32),
                        normal_render=batch["normal_render"].to(dtype=_f32),
                        reference_depth_map=batch.get("reference_depth_map"),
                        reference_intrinsics=batch.get("reference_intrinsics"),
                        target_intrinsics=batch.get("target_intrinsics"),
                        sparse_intrinsics=batch.get("sparse_intrinsics"),
                        support_confidence=batch.get("warp_support_confidence"),
                        accum_render=batch.get("accum_render"),
                        transmittance_render=batch.get("transmittance_render"),
                        timesteps=timesteps,
                        noisy_latents=noisy_latents.to(dtype=_f32),
                        return_uncertainty=True,
                    )
                    proposal_warp_error_loss = batch["target_rgb"].new_zeros(())
                    proposal_repairability_loss = batch["target_rgb"].new_zeros(())
                    proposal_verification_loss = batch["target_rgb"].new_zeros(())
                    proposal_acceptance_loss = batch["target_rgb"].new_zeros(())
                    frontier_residual_term = batch["target_rgb"].new_zeros(())
                    relative_teacher_loss = batch["target_rgb"].new_zeros(())
                    mean_rgb = self._select_teacher_prediction_rgb(outputs, batch)
                    if mean_rgb is None:
                        mean_rgb = self._decode_predicted_x0(
                            noisy_latents=noisy_latents,
                            noise_pred=outputs.noise_pred,
                            timesteps=timesteps,
                            track_grad=True,
                        )
                    mean_rgb = torch.nan_to_num(mean_rgb, nan=0.0, posinf=1.0, neginf=0.0)
                    effect_targets = self._compute_teacher_effect_targets(
                        teacher_rgb=mean_rgb,
                        render_rgb=batch["rgb_render"],
                        target_rgb=batch["target_rgb"],
                        depth_render=batch.get("depth_render"),
                        support_mask=batch.get("warp_valid_mask"),
                        support_confidence=batch.get("warp_support_confidence"),
                        frontier_mask=batch.get("frontier_mask"),
                        verification_prior=batch.get("verification_prior"),
                        proposal_residual_rgb=getattr(
                            outputs,
                            "predicted_proposal_residual",
                            None,
                        ),
                        proposal_confidence=getattr(
                            outputs,
                            "predicted_proposal_confidence",
                            None,
                        ),
                        proposal_warp_error_logit=getattr(
                            outputs,
                            "predicted_proposal_warp_error_logit",
                            None,
                        ),
                        proposal_repairability_logit=getattr(
                            outputs,
                            "predicted_proposal_repairability_logit",
                            None,
                        ),
                        proposal_verification_logit=getattr(
                            outputs,
                            "predicted_proposal_verification_logit",
                            None,
                        ),
                        proposal_acceptance_logit=getattr(
                            outputs,
                            "predicted_proposal_acceptance_logit",
                            None,
                        ),
                        proposal_warp_error_confidence=getattr(
                            outputs,
                            "predicted_proposal_warp_error",
                            None,
                        ),
                        proposal_repairability_confidence=getattr(
                            outputs,
                            "predicted_proposal_repairability",
                            None,
                        ),
                        proposal_verification_confidence=getattr(
                            outputs,
                            "predicted_proposal_verification",
                            None,
                        ),
                        proposal_acceptance_confidence=getattr(
                            outputs,
                            "predicted_proposal_acceptance",
                            None,
                        ),
                        use_projection_aligned_support=True,
                    )
                    if self._batch_uses_phase1_3dgs_conditions(batch):
                        resolved_relative_teacher_weight = getattr(
                            self,
                            "relative_teacher_effect_weight",
                            None,
                        )
                        if resolved_relative_teacher_weight is None:
                            resolved_relative_teacher_weight = 0.0
                        relative_teacher_loss = self._compute_relative_teacher_effect_loss(
                            effect_targets
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                        if resolved_stage3_frontier_residual_weight > 0.0:
                            frontier_residual_term = self._compute_frontier_residual_proposal_loss(
                                effect_targets,
                                proposal_rgb=mean_rgb,
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype) * float(
                                resolved_stage3_frontier_residual_weight
                            )
                        if resolved_stage3_proposal_warp_error_supervision_weight > 0.0:
                            warp_error_prediction = effect_targets.get(
                                "proposal_warp_error_logit"
                            )
                            warp_error_prediction_is_logit = isinstance(
                                warp_error_prediction,
                                torch.Tensor,
                            )
                            if not warp_error_prediction_is_logit:
                                warp_error_prediction = effect_targets.get(
                                    "proposal_warp_error_confidence"
                                )
                            proposal_warp_error_loss = self._compute_probability_head_supervision_loss(
                                warp_error_prediction,
                                effect_targets.get("proposal_warp_error_target"),
                                effect_targets.get("proposal_training_mask"),
                                prediction_is_logits=warp_error_prediction_is_logit,
                            ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                        if resolved_stage3_proposal_repairability_supervision_weight > 0.0:
                            repairability_prediction = effect_targets.get(
                                "proposal_repairability_logit"
                            )
                            repairability_prediction_is_logit = isinstance(
                                repairability_prediction,
                                torch.Tensor,
                            )
                            if not repairability_prediction_is_logit:
                                repairability_prediction = effect_targets.get(
                                    "proposal_repairability_confidence"
                                )
                            proposal_repairability_loss = self._compute_probability_head_supervision_loss(
                                repairability_prediction,
                                effect_targets.get("proposal_repairability_target"),
                                effect_targets.get("proposal_training_mask"),
                                prediction_is_logits=repairability_prediction_is_logit,
                                positive_balance=True,
                            ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                        if resolved_stage3_proposal_verification_supervision_weight > 0.0:
                            verification_prediction = effect_targets.get(
                                "proposal_verification_logit"
                            )
                            verification_prediction_is_logit = isinstance(
                                verification_prediction,
                                torch.Tensor,
                            )
                            if not verification_prediction_is_logit:
                                verification_prediction = effect_targets.get(
                                    "proposal_verification_confidence"
                                )
                            proposal_verification_loss = self._compute_probability_head_supervision_loss(
                                verification_prediction,
                                effect_targets.get("proposal_verification_target"),
                                effect_targets.get("proposal_training_mask"),
                                prediction_is_logits=verification_prediction_is_logit,
                                positive_balance=True,
                            ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                        if resolved_stage3_proposal_acceptance_supervision_weight > 0.0:
                            acceptance_prediction = effect_targets.get(
                                "proposal_acceptance_logit"
                            )
                            acceptance_prediction_is_logit = isinstance(
                                acceptance_prediction,
                                torch.Tensor,
                            )
                            if not acceptance_prediction_is_logit:
                                acceptance_prediction = effect_targets.get(
                                    "proposal_acceptance_confidence"
                                )
                            proposal_acceptance_loss = self._compute_probability_head_supervision_loss(
                                acceptance_prediction,
                                effect_targets.get("proposal_acceptance_target"),
                                effect_targets.get("proposal_training_mask"),
                                prediction_is_logits=acceptance_prediction_is_logit,
                                positive_balance=True,
                            ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                        ibc_outer_alignment_loss = (
                            relative_teacher_loss * float(max(resolved_relative_teacher_weight, 0.0))
                            + frontier_residual_term
                        )
                        if resolved_stage3_proposal_warp_error_supervision_weight > 0.0:
                            ibc_outer_alignment_loss = ibc_outer_alignment_loss + (
                                proposal_warp_error_loss
                                * float(resolved_stage3_proposal_warp_error_supervision_weight)
                            )
                        if resolved_stage3_proposal_repairability_supervision_weight > 0.0:
                            ibc_outer_alignment_loss = ibc_outer_alignment_loss + (
                                proposal_repairability_loss
                                * float(resolved_stage3_proposal_repairability_supervision_weight)
                            )
                        if resolved_stage3_proposal_verification_supervision_weight > 0.0:
                            ibc_outer_alignment_loss = ibc_outer_alignment_loss + (
                                proposal_verification_loss
                                * float(resolved_stage3_proposal_verification_supervision_weight)
                            )
                        if resolved_stage3_proposal_acceptance_supervision_weight > 0.0:
                            ibc_outer_alignment_loss = ibc_outer_alignment_loss + (
                                proposal_acceptance_loss
                                * float(resolved_stage3_proposal_acceptance_supervision_weight)
                            )
                        if torch.isfinite(ibc_outer_alignment_loss):
                            if grad_scaler is not None:
                                grad_scaler.scale(ibc_outer_alignment_loss).backward()
                            else:
                                ibc_outer_alignment_loss.backward()
                            epoch_proposal_warp_error_loss += float(
                                proposal_warp_error_loss.item()
                            )
                            epoch_proposal_repairability_loss += float(
                                proposal_repairability_loss.item()
                            )
                            epoch_proposal_verification_loss += float(
                                proposal_verification_loss.item()
                            )
                            epoch_proposal_acceptance_loss += float(
                                proposal_acceptance_loss.item()
                            )
                            total_diffusion_loss = total_diffusion_loss + ibc_outer_alignment_loss.detach().to(
                                device=total_diffusion_loss.device,
                                dtype=total_diffusion_loss.dtype,
                            )
                    
                    # IBC外层：直接执行optimizer step（diffusion loss已在前面backward）
                    if grad_scaler is not None:
                        if can_unscale_grads:
                            grad_scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        grad_scaler.step(self.optimizer)
                        grad_scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optimizer.step()
                    
                    # 记录metrics
                    total_loss_value = float(total_diffusion_loss.item())
                    epoch_loss += total_loss_value
                    self._stage3_step += 1

                    metric_rgb = (
                        final_sample_reference
                        if self.stage3_primary_metric_use_final_sample and final_sample_reference is not None
                        else mean_rgb.detach()
                    )
                    teacher_effect_reference_rgb = (
                        final_sample_reference
                        if (
                            self.use_final_sample_for_teacher_effect_targets
                            and final_sample_reference is not None
                        )
                        else mean_rgb.detach()
                    )
                    epistemic_target = effect_targets["risk_exist_target"]
                    aleatoric_target = effect_targets["gain_novel_target"]
                    unc_epistemic = (
                        outputs.uncertainty_epistemic.detach()
                        if outputs.uncertainty_epistemic is not None
                        else None
                    )
                    unc_aleatoric = (
                        outputs.uncertainty_aleatoric.detach()
                        if outputs.uncertainty_aleatoric is not None
                        else None
                    )
                    if unc_epistemic is None:
                        unc_epistemic = torch.zeros_like(epistemic_target)
                    if unc_aleatoric is None:
                        unc_aleatoric = torch.zeros_like(aleatoric_target)

                    surrogate_psnr_value = self._compute_psnr_from_rgb(mean_rgb, batch["target_rgb"])
                    psnr_value = self._compute_psnr_from_rgb(metric_rgb, batch["target_rgb"])
                    render_only_psnr_value = self._compute_render_only_psnr_from_batch(batch)
                    sample_surrogate_gap_value = psnr_value - surrogate_psnr_value
                    epoch_psnr += float(psnr_value.item())
                    epoch_surrogate_psnr += float(surrogate_psnr_value.item())
                    epoch_render_only_psnr += float(render_only_psnr_value.item())
                    epoch_sample_surrogate_gap += float(sample_surrogate_gap_value.item())
                    projection_coverage_value = self._estimate_batch_support_coverage(batch)
                    if projection_coverage_value is not None:
                        epoch_projection_coverage += float(projection_coverage_value)
                    exist_regret_vs_render_value = self._compute_exist_regret_vs_render(
                        teacher_rgb=metric_rgb.detach(),
                        render_rgb=batch["rgb_render"],
                        target_rgb=batch["target_rgb"],
                        depth_render=batch.get("depth_render"),
                        support_mask=batch.get("warp_valid_mask"),
                        support_confidence=batch.get("warp_support_confidence"),
                    )
                    if exist_regret_vs_render_value is not None:
                        epoch_exist_regret_vs_render += float(
                            exist_regret_vs_render_value.item()
                        )

                    total_uncertainty = unc_epistemic + unc_aleatoric
                    total_target = epistemic_target + aleatoric_target
                    error_map = torch.abs(metric_rgb - batch["target_rgb"]).mean(dim=1, keepdim=True)

                    if self._is_main_rank and epoch == 0 and not image_logged:
                        rgb_to_log = torch.clamp(metric_rgb[0], 0.0, 1.0).detach().cpu()
                        if self.stage_samples_dir is not None and self._is_main_rank:
                            self._save_tensor_as_png(
                                rgb_to_log,
                                self.stage_samples_dir / f"stage3_epoch{epoch:03d}_rgb.png",
                                normalize=False,
                            )
                        image_logged = True
                    
                    # Resize uncertainty到error_map的尺寸
                    if total_uncertainty.shape[-2:] != error_map.shape[-2:]:
                        total_uncertainty = F.interpolate(
                            total_uncertainty,
                            size=error_map.shape[-2:],
                            mode="bilinear",
                            align_corners=True,
                        )
                    
                    pred_std = torch.sqrt(torch.clamp(total_uncertainty, min=0.0))
                    # 多GPU模式：对齐error_map到pred_std的设备
                    error_map = error_map.to(pred_std.device)
                    error_scaled = error_map / (error_map.max() + 1e-8)
                    pred_std_scaled = pred_std / (pred_std.max() + 1e-8)
                    ece = torch.abs(pred_std_scaled - error_scaled).mean()
                    epoch_ece += float(ece.item())
                    
                    flat_unc = total_uncertainty.flatten()
                    flat_err = error_map.flatten()
                    corr = torch.corrcoef(torch.stack([flat_unc, flat_err]))[0, 1]
                    epoch_corr += float(corr.item()) if not torch.isnan(corr) else 0.0
                    
                    metric_batches += 1
                    continue  # 跳过后续的非IBC代码
                
                # 第二次前向：专注不确定性分支，允许其独立构建计算图（非IBC模式）
                with torch.cuda.amp.autocast(dtype=self._amp_dtype, enabled=autocast_enabled):
                    outputs = self.model(
                        sparse_images=batch["sparse_images"],
                        sparse_poses=batch["sparse_poses"],
                        target_pose=batch["target_pose"],
                        rgb_render=batch["rgb_render"],
                        depth_render=batch["depth_render"],
                        normal_render=batch["normal_render"],
                        reference_depth_map=batch.get("reference_depth_map"),
                        reference_intrinsics=batch.get("reference_intrinsics"),
                        target_intrinsics=batch.get("target_intrinsics"),
                        sparse_intrinsics=batch.get("sparse_intrinsics"),
                        support_confidence=batch.get("warp_support_confidence"),
                        accum_render=batch.get("accum_render"),
                        transmittance_render=batch.get("transmittance_render"),
                        timesteps=timesteps,
                        noisy_latents=noisy_latents,
                        return_uncertainty=True,
                    )
                recon_term = torch.zeros((), device=self.device, dtype=noisy_latents.dtype)
                perc_term = torch.zeros((), device=self.device, dtype=noisy_latents.dtype)
                final_sample_guided_term = torch.zeros((), device=self.device, dtype=noisy_latents.dtype)
                direct_sample_path_term = torch.zeros((), device=self.device, dtype=noisy_latents.dtype)
                frontier_residual_term = torch.zeros((), device=self.device, dtype=noisy_latents.dtype)
                predicted_noise = getattr(outputs, "noise_pred", None)
                # Timestep Masking: 仅在 t < 500 时计算 recon_loss
                # 高时间步的x0预测有15-30倍放大效应，会产生不稳定梯度
                safe_timesteps = timesteps < 500
                should_compute_recon = (
                    predicted_noise is not None 
                    and (recon_weight > 0.0 or perceptual_weight > 0.0)
                    and safe_timesteps.all()  # 仅当batch内所有样本都满足条件时计算
                )
                if should_compute_recon:
                    rgb_reconstruction = self._decode_predicted_x0(
                        noisy_latents=noisy_latents,
                        noise_pred=predicted_noise,
                        timesteps=timesteps,
                        track_grad=True,
                    )
                    if self._batch_uses_phase1_3dgs_conditions(batch):
                        rgb_reconstruction = self._compose_render_residual_refined_rgb(
                            rgb_reconstruction,
                            batch,
                        )
                    target_rgb = batch["target_rgb"]
                    if recon_weight > 0.0:
                        recon_term = self._region_aware_reconstruction_loss(
                            rgb_reconstruction,
                            target_rgb,
                            self._get_support_anchor_rgb_from_batch(batch),
                            exist_mask=exist_mask,
                            novel_mask=novel_mask,
                            recon_alpha=recon_alpha,
                        ) * recon_weight
                    if perceptual_fn is not None and perceptual_weight > 0.0:
                        perc_term = perceptual_fn(rgb_reconstruction, target_rgb) * perceptual_weight
                mean_rgb = self._select_teacher_prediction_rgb(outputs, batch)
                if (
                    (mean_rgb is None or not bool(getattr(mean_rgb, "requires_grad", False)))
                    and predicted_noise is not None
                ):
                    mean_rgb = self._decode_predicted_x0(
                        noisy_latents=noisy_latents,
                        noise_pred=predicted_noise,
                        timesteps=timesteps,
                        track_grad=True,
                    )
                    if self._batch_uses_phase1_3dgs_conditions(batch):
                        mean_rgb = self._compose_render_residual_refined_rgb(mean_rgb, batch)
                if mean_rgb is None:
                    raise RuntimeError("Stage3 effect 目标构造失败：predicted_image 不可用。")
                mean_rgb = torch.nan_to_num(mean_rgb, nan=0.0, posinf=1.0, neginf=0.0)
                final_sample_reference = None
                trainable_sample_rgb = None
                final_sample_config = self._resolve_final_sample_supervision_config(
                    batch,
                    base_steps=self.final_sample_supervision_steps,
                    base_weight=self.final_sample_supervision_weight,
                )
                effective_final_sample_steps = max(
                    1,
                    int(final_sample_config["steps"] or self.final_sample_supervision_steps),
                )
                effective_final_sample_weight = float(
                    final_sample_config["weight"]
                    if final_sample_config["weight"] is not None
                    else self.final_sample_supervision_weight
                )
                trainable_sample_config = self._resolve_final_sample_supervision_config(
                    batch,
                    base_steps=self.trainable_sample_supervision_steps,
                    base_weight=self.trainable_sample_supervision_weight,
                    max_weight=self.trainable_sample_supervision_max_weight,
                )
                effective_trainable_sample_steps = max(
                    1,
                    int(trainable_sample_config["steps"] or self.trainable_sample_supervision_steps),
                )
                effective_trainable_sample_weight = float(
                    trainable_sample_config["weight"]
                    if trainable_sample_config["weight"] is not None
                    else self.trainable_sample_supervision_weight
                )
                should_apply_final_sample_supervision = (
                    effective_final_sample_weight > 0.0
                    and self.final_sample_supervision_interval > 0
                    and (self._stage3_step % max(self.final_sample_supervision_interval, 1) == 0)
                )
                should_apply_trainable_sample_supervision = (
                    effective_trainable_sample_weight > 0.0
                    and self.trainable_sample_supervision_interval > 0
                    and (self._stage3_step % max(self.trainable_sample_supervision_interval, 1) == 0)
                )
                should_generate_metric_final_sample = bool(
                    self.stage3_primary_metric_use_final_sample
                )
                if should_generate_metric_final_sample or should_apply_final_sample_supervision:
                    final_sample_reference = self._generate_final_sample_reference(
                        batch,
                        steps=effective_final_sample_steps,
                    )
                    if should_apply_final_sample_supervision and final_sample_reference is not None:
                        final_sample_guided_term = self._compute_final_sample_guided_reconstruction_loss(
                            prediction_rgb=mean_rgb,
                            target_rgb=batch["target_rgb"],
                            base_rgb=self._get_support_anchor_rgb_from_batch(batch),
                            final_sample_rgb=final_sample_reference,
                            support_mask=batch.get("warp_valid_mask"),
                            support_confidence=batch.get("warp_support_confidence"),
                            novel_mask=novel_mask,
                        ).to(
                            device=mean_rgb.device,
                            dtype=mean_rgb.dtype,
                        ) * effective_final_sample_weight
                if should_apply_trainable_sample_supervision:
                    trainable_sample_generation = self._generate_trainable_sample(
                        batch,
                        steps=effective_trainable_sample_steps,
                        sampler_type=self.sampler_type,
                        return_diagnostics=False,
                    )
                    if isinstance(trainable_sample_generation, dict):
                        trainable_sample_rgb = trainable_sample_generation.get("rgb")
                        if trainable_sample_rgb is None:
                            trainable_sample_rgb = trainable_sample_generation.get("image")
                    else:
                        trainable_sample_rgb = trainable_sample_generation
                    if trainable_sample_rgb is not None:
                        direct_sample_path_term = self._compute_direct_sample_path_reconstruction_loss(
                            sample_rgb=trainable_sample_rgb,
                            target_rgb=batch["target_rgb"],
                            base_rgb=self._get_support_anchor_rgb_from_batch(batch),
                            support_mask=batch.get("warp_valid_mask"),
                            support_confidence=batch.get("warp_support_confidence"),
                            depth_render=batch.get("depth_render"),
                            recon_alpha=recon_alpha,
                        ).to(
                            device=mean_rgb.device,
                            dtype=mean_rgb.dtype,
                        ) * effective_trainable_sample_weight
                sample_alignment_reference_rgb = None
                if trainable_sample_rgb is not None:
                    sample_alignment_reference_rgb = torch.nan_to_num(
                        trainable_sample_rgb.detach(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                elif final_sample_reference is not None:
                    sample_alignment_reference_rgb = torch.nan_to_num(
                        final_sample_reference.detach(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                metric_rgb = (
                    sample_alignment_reference_rgb
                    if self.stage3_primary_metric_use_final_sample and sample_alignment_reference_rgb is not None
                    else mean_rgb
                )
                teacher_effect_reference_rgb = (
                    sample_alignment_reference_rgb
                    if (
                        self.use_final_sample_for_teacher_effect_targets
                        and sample_alignment_reference_rgb is not None
                    )
                    else mean_rgb.detach()
                )
                empirical_candidate_tensors = {"candidate_rgbs": [], "candidate_residuals": []}
                if self._batch_uses_phase1_3dgs_conditions(batch):
                    empirical_candidate_tensors = self._build_empirical_repairability_candidates(
                        batch,
                        steps=effective_trainable_sample_steps,
                        sampler_type=self.sampler_type,
                    )
                effect_targets = self._compute_teacher_effect_targets(
                    teacher_rgb=teacher_effect_reference_rgb,
                    render_rgb=batch["rgb_render"],
                    target_rgb=batch["target_rgb"],
                    depth_render=batch.get("depth_render"),
                    support_mask=batch.get("warp_valid_mask"),
                    support_confidence=batch.get("warp_support_confidence"),
                    frontier_mask=batch.get("frontier_mask"),
                    verification_prior=batch.get("verification_prior"),
                    proposal_residual_rgb=getattr(
                        outputs,
                        "predicted_proposal_residual",
                        None,
                    ),
                    proposal_confidence=getattr(
                        outputs,
                        "predicted_proposal_confidence",
                        None,
                    ),
                    proposal_warp_error_logit=getattr(
                        outputs,
                        "predicted_proposal_warp_error_logit",
                        None,
                    ),
                    proposal_repairability_logit=getattr(
                        outputs,
                        "predicted_proposal_repairability_logit",
                        None,
                    ),
                    proposal_verification_logit=getattr(
                        outputs,
                        "predicted_proposal_verification_logit",
                        None,
                    ),
                    proposal_acceptance_logit=getattr(
                        outputs,
                        "predicted_proposal_acceptance_logit",
                        None,
                    ),
                    proposal_warp_error_confidence=getattr(
                        outputs,
                        "predicted_proposal_warp_error",
                        None,
                    ),
                    proposal_repairability_confidence=getattr(
                        outputs,
                        "predicted_proposal_repairability",
                        None,
                    ),
                    proposal_verification_confidence=getattr(
                        outputs,
                        "predicted_proposal_verification",
                        None,
                    ),
                    proposal_acceptance_confidence=getattr(
                        outputs,
                        "predicted_proposal_acceptance",
                        None,
                    ),
                    empirical_candidate_rgbs=empirical_candidate_tensors.get(
                        "candidate_rgbs"
                    ),
                    empirical_candidate_residuals=empirical_candidate_tensors.get(
                        "candidate_residuals"
                    ),
                    use_projection_aligned_support=True,
                )
                epistemic_target = effect_targets["risk_exist_target"]
                aleatoric_target = effect_targets["gain_novel_target"]
                exist_mask = effect_targets["exist_mask"]
                novel_mask = effect_targets["novel_mask"]
                metric_mask = None if novel_mask is not None else exist_mask
                relative_teacher_loss = self._compute_relative_teacher_effect_loss(
                    effect_targets
                )
                proposal_warp_error_loss = mean_rgb.new_zeros(())
                proposal_repairability_loss = mean_rgb.new_zeros(())
                proposal_verification_loss = mean_rgb.new_zeros(())
                proposal_acceptance_loss = mean_rgb.new_zeros(())
                if (
                    resolved_stage3_frontier_residual_weight > 0.0
                    and self._batch_uses_phase1_3dgs_conditions(batch)
                ):
                    frontier_residual_term = self._compute_frontier_residual_proposal_loss(
                        effect_targets,
                        proposal_rgb=mean_rgb,
                    ).to(
                        device=mean_rgb.device,
                        dtype=mean_rgb.dtype,
                    ) * float(resolved_stage3_frontier_residual_weight)
                if self._batch_uses_phase1_3dgs_conditions(batch):
                    if resolved_stage3_proposal_warp_error_supervision_weight > 0.0:
                        warp_error_prediction = effect_targets.get(
                            "proposal_warp_error_logit"
                        )
                        warp_error_prediction_is_logit = isinstance(
                            warp_error_prediction,
                            torch.Tensor,
                        )
                        if not warp_error_prediction_is_logit:
                            warp_error_prediction = effect_targets.get(
                                "proposal_warp_error_confidence"
                            )
                        proposal_warp_error_loss = self._compute_probability_head_supervision_loss(
                            warp_error_prediction,
                            effect_targets.get("proposal_warp_error_target"),
                            effect_targets.get("proposal_training_mask"),
                            prediction_is_logits=warp_error_prediction_is_logit,
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                    if resolved_stage3_proposal_repairability_supervision_weight > 0.0:
                        repairability_prediction = effect_targets.get(
                            "proposal_repairability_logit"
                        )
                        repairability_prediction_is_logit = isinstance(
                            repairability_prediction,
                            torch.Tensor,
                        )
                        if not repairability_prediction_is_logit:
                            repairability_prediction = effect_targets.get(
                                "proposal_repairability_confidence"
                            )
                        proposal_repairability_loss = self._compute_probability_head_supervision_loss(
                            repairability_prediction,
                            effect_targets.get("proposal_repairability_target"),
                            effect_targets.get("proposal_training_mask"),
                            prediction_is_logits=repairability_prediction_is_logit,
                            positive_balance=True,
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                    if resolved_stage3_proposal_verification_supervision_weight > 0.0:
                        verification_prediction = effect_targets.get(
                            "proposal_verification_logit"
                        )
                        verification_prediction_is_logit = isinstance(
                            verification_prediction,
                            torch.Tensor,
                        )
                        if not verification_prediction_is_logit:
                            verification_prediction = effect_targets.get(
                                "proposal_verification_confidence"
                            )
                        proposal_verification_loss = self._compute_probability_head_supervision_loss(
                            verification_prediction,
                            effect_targets.get("proposal_verification_target"),
                            effect_targets.get("proposal_training_mask"),
                            prediction_is_logits=verification_prediction_is_logit,
                            positive_balance=True,
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                    if resolved_stage3_proposal_acceptance_supervision_weight > 0.0:
                        acceptance_prediction = effect_targets.get(
                            "proposal_acceptance_logit"
                        )
                        acceptance_prediction_is_logit = isinstance(
                            acceptance_prediction,
                            torch.Tensor,
                        )
                        if not acceptance_prediction_is_logit:
                            acceptance_prediction = effect_targets.get(
                                "proposal_acceptance_confidence"
                            )
                        proposal_acceptance_loss = self._compute_probability_head_supervision_loss(
                            acceptance_prediction,
                            effect_targets.get("proposal_acceptance_target"),
                            effect_targets.get("proposal_training_mask"),
                            prediction_is_logits=acceptance_prediction_is_logit,
                            positive_balance=True,
                        ).to(device=mean_rgb.device, dtype=mean_rgb.dtype)
                if hasattr(outputs, "noise_pred"):
                    outputs.noise_pred = None  # type: ignore[assignment]
                assert outputs.uncertainty_epistemic is not None
                assert outputs.uncertainty_aleatoric is not None
                unc_epistemic = torch.nan_to_num(
                    outputs.uncertainty_epistemic, nan=0.0, posinf=1.0, neginf=0.0
                )
                unc_aleatoric = torch.nan_to_num(
                    outputs.uncertainty_aleatoric, nan=0.0, posinf=1.0, neginf=0.0
                )
                loss_epist = self._calibration_loss(
                    unc_epistemic,
                    epistemic_target,
                    exist_mask,
                )
                loss_aleat = self._calibration_loss(
                    unc_aleatoric,
                    aleatoric_target,
                    novel_mask,
                )
                do_distill = (self._stage3_step % max(self.distill_interval, 1)) == 0
                if do_distill:
                    pred_for_distill = unc_epistemic
                    if pred_for_distill.shape[-2:] != epistemic_target.shape[-2:]:
                        pred_for_distill = F.interpolate(
                            pred_for_distill,
                            size=epistemic_target.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    loss_distill = F.mse_loss(
                        pred_for_distill,
                        epistemic_target.detach(),
                    )
                else:
                    loss_distill = torch.tensor(0.0, device=self.device)
                uncertainty_loss = self.lambda_calib * (loss_epist + loss_aleat) + lambda_distill * loss_distill
                final_sample_guided_term = final_sample_guided_term.to(
                    device=uncertainty_loss.device,
                    dtype=uncertainty_loss.dtype,
                )
                direct_sample_path_term = direct_sample_path_term.to(
                    device=uncertainty_loss.device,
                    dtype=uncertainty_loss.dtype,
                )
                sample_path_supervision_active = bool(
                    should_apply_final_sample_supervision
                    or should_apply_trainable_sample_supervision
                )
                sample_path_alignment_term = (
                    final_sample_guided_term + direct_sample_path_term
                )
                if (
                    sample_path_supervision_active
                    and self.sample_path_primary_alignment_enabled
                ):
                    sample_path_alignment_term = (
                        sample_path_alignment_term
                        * float(self.sample_path_supervision_priority_weight)
                    )
                    recon_term = recon_term * float(
                        self.sample_path_surrogate_reconstruction_decay
                    )
                if (
                    resolved_stage3_frontier_residual_weight > 0.0
                    and self._batch_uses_phase1_3dgs_conditions(batch)
                ):
                    recon_term = recon_term * float(resolved_stage3_surrogate_reconstruction_decay)
                    perc_term = perc_term * float(resolved_stage3_surrogate_reconstruction_decay)
                 
                # Variance preservation 正则化：惩罚 uncertainty 输出方差过小
                unc_epist_var = unc_epistemic.var()
                unc_aleat_var = unc_aleatoric.var()
                var_preserve_loss = (
                    F.relu(self.min_pred_var - unc_epist_var) + 
                    F.relu(self.min_pred_var - unc_aleat_var)
                )
                uncertainty_loss = uncertainty_loss + self.lambda_var_preserve * var_preserve_loss
                resolved_relative_teacher_weight = getattr(
                    self,
                    "relative_teacher_effect_weight",
                    None,
                )
                if resolved_relative_teacher_weight is None:
                    resolved_relative_teacher_weight = max(float(recon_weight), 0.0)
                
                total_stage3_loss = (
                    uncertainty_loss
                    + recon_term
                    + perc_term
                    + frontier_residual_term
                    + relative_teacher_loss * float(max(resolved_relative_teacher_weight, 0.0))
                    + sample_path_alignment_term
                )
                if resolved_stage3_proposal_warp_error_supervision_weight > 0.0:
                    total_stage3_loss = total_stage3_loss + (
                        proposal_warp_error_loss
                        * float(resolved_stage3_proposal_warp_error_supervision_weight)
                    )
                if resolved_stage3_proposal_repairability_supervision_weight > 0.0:
                    total_stage3_loss = total_stage3_loss + (
                        proposal_repairability_loss
                        * float(resolved_stage3_proposal_repairability_supervision_weight)
                    )
                if resolved_stage3_proposal_verification_supervision_weight > 0.0:
                    total_stage3_loss = total_stage3_loss + (
                        proposal_verification_loss
                        * float(resolved_stage3_proposal_verification_supervision_weight)
                    )
                if resolved_stage3_proposal_acceptance_supervision_weight > 0.0:
                    total_stage3_loss = total_stage3_loss + (
                        proposal_acceptance_loss
                        * float(resolved_stage3_proposal_acceptance_supervision_weight)
                    )
                if grad_scaler is not None:
                    grad_scaler.scale(total_stage3_loss).backward()
                    if can_unscale_grads:
                        grad_scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                else:
                    total_stage3_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss_value = float(total_diffusion_loss.item() + total_stage3_loss.item())
                epoch_loss += total_loss_value
                epoch_proposal_warp_error_loss += float(proposal_warp_error_loss.item())
                epoch_proposal_repairability_loss += float(
                    proposal_repairability_loss.item()
                )
                epoch_proposal_verification_loss += float(
                    proposal_verification_loss.item()
                )
                epoch_proposal_acceptance_loss += float(
                    proposal_acceptance_loss.item()
                )
                self._stage3_step += 1
                surrogate_psnr_value = self._compute_psnr_from_rgb(mean_rgb, batch["target_rgb"])
                psnr_value = self._compute_psnr_from_rgb(metric_rgb, batch["target_rgb"])
                render_only_psnr_value = self._compute_render_only_psnr_from_batch(batch)
                sample_surrogate_gap_value = psnr_value - surrogate_psnr_value
                epoch_psnr += float(psnr_value.item())
                epoch_surrogate_psnr += float(surrogate_psnr_value.item())
                epoch_render_only_psnr += float(render_only_psnr_value.item())
                epoch_sample_surrogate_gap += float(sample_surrogate_gap_value.item())
                projection_coverage_value = self._estimate_batch_support_coverage(batch)
                if projection_coverage_value is not None:
                    epoch_projection_coverage += float(projection_coverage_value)
                exist_regret_vs_render_value = self._compute_exist_regret_vs_render(
                    teacher_rgb=metric_rgb.detach(),
                    render_rgb=batch["rgb_render"],
                    target_rgb=batch["target_rgb"],
                    depth_render=batch.get("depth_render"),
                    support_mask=batch.get("warp_valid_mask"),
                    support_confidence=batch.get("warp_support_confidence"),
                )
                if exist_regret_vs_render_value is not None:
                    epoch_exist_regret_vs_render += float(
                        exist_regret_vs_render_value.item()
                    )
                total_uncertainty = unc_epistemic + unc_aleatoric
                total_target = epistemic_target + aleatoric_target
                error_map = torch.abs(metric_rgb - batch["target_rgb"]).mean(dim=1, keepdim=True)
                pred_std = torch.sqrt(torch.clamp(total_uncertainty, min=0.0))
                epoch_ece += self._expected_calibration_error(total_uncertainty, total_target, metric_mask)
                corr_value = self._uncertainty_error_correlation(pred_std, error_map, metric_mask)
                epoch_corr += corr_value
                if self.writer is not None and self.world_rank == 0:
                    if not scatter_logged:
                        self._log_uncertainty_scatter(
                            stage_tag="stage3",
                            epoch=epoch,
                            prediction_std=pred_std,
                            error_map=error_map,
                            mask=metric_mask,
                        )
                        scatter_logged = True
                    if not image_logged:
                        self._log_uncertainty_heatmaps(
                            stage_tag="stage3",
                            epoch=epoch,
                            error_map=error_map,
                            epistemic=unc_epistemic,
                            aleatoric=unc_aleatoric,
                        )
                        rgb_to_log = torch.clamp(metric_rgb[0], 0.0, 1.0).detach().cpu()
                        self.writer.add_image(
                            "stage3/generated",
                            rgb_to_log,
                            epoch,
                        )
                if not image_logged:
                    rgb_to_log = torch.clamp(metric_rgb[0], 0.0, 1.0).detach().cpu()
                    if self.stage_samples_dir is not None and self._is_main_rank:
                        self._save_tensor_as_png(
                            rgb_to_log,
                            self.stage_samples_dir / f"stage3_epoch{epoch:03d}_rgb.png",
                            normalize=False,
                        )
                    image_logged = True
                metric_batches += 1
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            validation_metrics: Dict[str, float] = {}
            validation_score: Optional[float] = None
            if self.val_dataloader is not None:
                validation_metrics = self.validate()
                if validation_metrics:
                    validation_score = float(
                        validation_metrics.get(
                            "validation_score",
                            validation_metrics.get("whole_image_psnr", 0.0),
                        )
                    )
            mean_epoch_loss_for_logging = epoch_loss / max(metric_batches, 1)
            _log_stage_progress(stage="Stage3", epoch=epoch, loss=mean_epoch_loss_for_logging)
            if metric_batches > 0:
                metric_sums = self._reduce_values(
                    [
                        epoch_psnr,
                        epoch_surrogate_psnr,
                        epoch_render_only_psnr,
                        epoch_sample_surrogate_gap,
                        epoch_ece,
                        epoch_corr,
                        epoch_projection_coverage,
                        epoch_exist_regret_vs_render,
                        epoch_proposal_warp_error_loss,
                        epoch_proposal_repairability_loss,
                        epoch_proposal_verification_loss,
                        epoch_proposal_acceptance_loss,
                        float(metric_batches),
                    ]
                )
                (
                    total_psnr,
                    total_surrogate_psnr,
                    total_render_only_psnr,
                    total_sample_surrogate_gap,
                    total_ece,
                    total_corr,
                    total_projection_coverage,
                    total_exist_regret_vs_render,
                    total_proposal_warp_error_loss,
                    total_proposal_repairability_loss,
                    total_proposal_verification_loss,
                    total_proposal_acceptance_loss,
                    total_batches,
                ) = metric_sums
                denom = max(total_batches, 1.0)
                avg_psnr = total_psnr / denom
                avg_surrogate_psnr = total_surrogate_psnr / denom
                avg_render_only_psnr = total_render_only_psnr / denom
                avg_sample_surrogate_gap = total_sample_surrogate_gap / denom
                avg_ece = total_ece / denom
                avg_corr = total_corr / denom
                avg_projection_coverage = total_projection_coverage / denom
                avg_exist_regret_vs_render = total_exist_regret_vs_render / denom
                avg_variance = epoch_variance / denom if denom > 0 else 0.0
                avg_proposal_warp_error_loss = total_proposal_warp_error_loss / denom
                avg_proposal_repairability_loss = (
                    total_proposal_repairability_loss / denom
                )
                avg_proposal_verification_loss = total_proposal_verification_loss / denom
                avg_proposal_acceptance_loss = total_proposal_acceptance_loss / denom

                if self._is_main_rank:
                    print(
                        f"[Stage3] Epoch {epoch} metrics: "
                        f"sample_psnr={avg_psnr:.2f}dB, surrogate_psnr={avg_surrogate_psnr:.2f}dB, "
                        f"render_only_psnr={avg_render_only_psnr:.2f}dB, "
                        f"sample_surrogate_gap={avg_sample_surrogate_gap:.2f}dB, "
                        f"ece={avg_ece:.4f}, corr={avg_corr:.4f}, "
                        f"projection_coverage_ratio={avg_projection_coverage:.4f}, "
                        f"exist_regret_vs_render={avg_exist_regret_vs_render:.4f}, "
                        f"proposal_warp_error_loss={avg_proposal_warp_error_loss:.4f}, "
                        f"proposal_repairability_loss={avg_proposal_repairability_loss:.4f}, "
                        f"proposal_verification_loss={avg_proposal_verification_loss:.4f}, "
                        f"proposal_acceptance_loss={avg_proposal_acceptance_loss:.4f}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar("stage3/psnr", avg_psnr, epoch)
                        self.writer.add_scalar("stage3/surrogate_psnr", avg_surrogate_psnr, epoch)
                        self.writer.add_scalar("stage3/render_only_psnr", avg_render_only_psnr, epoch)
                        self.writer.add_scalar("stage3/sample_surrogate_gap", avg_sample_surrogate_gap, epoch)
                        self.writer.add_scalar("stage3/ece", avg_ece, epoch)
                        self.writer.add_scalar("stage3/uncert_error_corr", avg_corr, epoch)
                        self.writer.add_scalar(
                            "stage3/projection_coverage_ratio",
                            avg_projection_coverage,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage3/exist_regret_vs_render",
                            avg_exist_regret_vs_render,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage3/epoch_avg_proposal_warp_error_loss",
                            avg_proposal_warp_error_loss,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage3/epoch_avg_proposal_repairability_loss",
                            avg_proposal_repairability_loss,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage3/epoch_avg_proposal_verification_loss",
                            avg_proposal_verification_loss,
                            epoch,
                        )
                        self.writer.add_scalar(
                            "stage3/epoch_avg_proposal_acceptance_loss",
                            avg_proposal_acceptance_loss,
                            epoch,
                        )
                        if self.stage3_variance_weight > 0.0:
                            self.writer.add_scalar("stage3/variance_loss", avg_variance, epoch)
                    self._log_fixed_visualization_runtime_metrics(
                        stage_tag="stage3",
                        epoch=epoch,
                        base_steps=self.final_sample_supervision_steps,
                        base_weight=self.final_sample_supervision_weight,
                    )
            self._write_validation_metrics(
                writer_prefix="stage3",
                validation_metrics=validation_metrics,
                epoch=epoch,
            )
            if self._is_main_rank and validation_metrics:
                print(
                    "[Stage3] Validation: "
                    + self._format_validation_metrics_for_log(validation_metrics)
                )
            if metric_batches > 0:
                self._record_stage3_metrics(
                    psnr=float(avg_psnr),
                    ece=float(avg_ece),
                    corr=float(avg_corr),
                    validation_score=validation_score,
                )
                self._maybe_trigger_auto_unfreeze(current_epoch=epoch)
                monitored_metric_name = (
                    "validation_score" if validation_score is not None else "sample_psnr"
                )
                monitored_metric_value = (
                    float(validation_score) if validation_score is not None else float(avg_psnr)
                )
                (
                    early_stop_best_metric,
                    early_stop_plateau_count,
                    should_stop_early,
                ) = self._update_early_stop_state(
                    stage_name="Stage3",
                    metric_name=monitored_metric_name,
                    current_metric=monitored_metric_value,
                    best_metric=early_stop_best_metric,
                    plateau_count=early_stop_plateau_count,
                    epoch=epoch,
                )
                if should_stop_early:
                    break
            
            # ========== IBC内层：快速重校准不确定性头 ==========
            if self.ibc_enabled and ibc_inner_optimizer is not None:
                # 解冻不确定性头，冻结其他参数
                for name, param in self.model.named_parameters():
                    is_uncertainty = any(kw in name for kw in ["epistemic", "aleatoric", "uncertainty"])
                    param.requires_grad = is_uncertainty
                
                inner_loss_sum = 0.0
                inner_loss_count = 0
                
                for inner_step in range(self.ibc_inner_steps):
                    # 遍历数据计算MC目标并校准
                    for batch in self.train_dataloader:
                        batch = self._to_device(batch)
                        
                        # 前向计算不确定性头的输出
                        # 显式转换为float32以避免dtype不匹配（Half vs Float）
                        timesteps_inner, _, noisy_latents_inner = self._sample_diffusion_inputs(batch)
                        _f32_inner = torch.float32
                        with torch.cuda.amp.autocast(dtype=self._amp_dtype, enabled=autocast_enabled):
                            outputs = self.model(
                                sparse_images=batch["sparse_images"].to(dtype=_f32_inner),
                                sparse_poses=batch["sparse_poses"].to(dtype=_f32_inner),
                                target_pose=batch["target_pose"].to(dtype=_f32_inner),
                                rgb_render=batch["rgb_render"].to(dtype=_f32_inner),
                                depth_render=batch["depth_render"].to(dtype=_f32_inner),
                                normal_render=batch["normal_render"].to(dtype=_f32_inner),
                                reference_depth_map=batch.get("reference_depth_map"),
                                reference_intrinsics=batch.get("reference_intrinsics"),
                                target_intrinsics=batch.get("target_intrinsics"),
                                sparse_intrinsics=batch.get("sparse_intrinsics"),
                                support_confidence=batch.get("warp_support_confidence"),
                                accum_render=batch.get("accum_render"),
                                transmittance_render=batch.get("transmittance_render"),
                                timesteps=timesteps_inner,
                                noisy_latents=noisy_latents_inner.to(dtype=_f32_inner),
                                return_uncertainty=True,
                            )
                        predicted_inner_rgb = self._select_teacher_prediction_rgb(outputs, batch)
                        if predicted_inner_rgb is None:
                            predicted_inner_rgb = self._decode_predicted_x0(
                                noisy_latents=noisy_latents_inner,
                                noise_pred=outputs.noise_pred,
                                timesteps=timesteps_inner,
                            )
                            if self._batch_uses_phase1_3dgs_conditions(batch):
                                predicted_inner_rgb = self._compose_render_residual_refined_rgb(
                                    predicted_inner_rgb,
                                    batch,
                                )
                        effect_targets = self._compute_teacher_effect_targets(
                            teacher_rgb=predicted_inner_rgb.detach(),
                            render_rgb=batch["rgb_render"],
                            target_rgb=batch["target_rgb"],
                            depth_render=batch.get("depth_render"),
                            support_mask=batch.get("warp_valid_mask"),
                            support_confidence=batch.get("warp_support_confidence"),
                            frontier_mask=batch.get("frontier_mask"),
                            verification_prior=batch.get("verification_prior"),
                            use_projection_aligned_support=True,
                        )
                        epistemic_target = effect_targets["risk_exist_target"]
                        aleatoric_target = effect_targets["gain_novel_target"]
                        exist_mask = effect_targets["exist_mask"]
                        novel_mask = effect_targets["novel_mask"]

                        if outputs.uncertainty_epistemic is not None:
                            calib_loss = self._calibration_loss(
                                outputs.uncertainty_epistemic,
                                epistemic_target,
                                exist_mask,
                            )
                            if outputs.uncertainty_aleatoric is not None:
                                calib_loss = calib_loss + self._calibration_loss(
                                    outputs.uncertainty_aleatoric,
                                    aleatoric_target,
                                    novel_mask,
                                )
                            
                            # 检查损失是否有效
                            if torch.isfinite(calib_loss):
                                ibc_inner_optimizer.zero_grad()
                                calib_loss.backward()
                                ibc_inner_optimizer.step()
                                inner_loss_sum += calib_loss.item()
                                inner_loss_count += 1
                        
                        # 清理本次迭代的临时张量
                        del epistemic_target, aleatoric_target, outputs
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        break  # 每个内层step只用一个batch
                
                avg_inner_loss = inner_loss_sum / max(inner_loss_count, 1)
                if self._is_main_rank and epoch % 10 == 0:
                    print(f"[IBC内层] Epoch {epoch}: 平均校准损失={avg_inner_loss:.4f}")

    # MVALF-Pro: Phase 1 + Phase 2 全部改进集成
    # ================================================================
    
    
    
    
    
    
    # ========== MVALF: Multi-View Adaptive Latent Fusion ==========
    
    # ========== 几何Warping + 选择性扩散 (Repaint) ==========
    
    # ========== 综合最优: Latent Warp + 频率分解 ==========
    
    # ========== 方向2: 频率分解双分支 ==========
    
    # ========== 方向3: 区域自适应噪声 ==========
    
    # ========== 方向1: Latent Warping ==========
    
    # ========== 方向4: Epipolar Cross-View Attention ==========
    
    


    def validate(
        self,
        *,
        generation_mode: str = "full_generate",
        max_batches: int = 0,
    ) -> Dict[str, float]:
        """Compute proposal-first validation metrics on the validation set."""
        if self.val_dataloader is None:
            return {}
        previous_training_state = self.model.training
        self.model.eval()
        aggregated_metric_values = {
            "whole_image_squared_error_sum": 0.0,
            "whole_image_weight_sum": 0.0,
            "proposal_novel_squared_error_sum": 0.0,
            "proposal_novel_weight_sum": 0.0,
            "proposal_editable_squared_error_sum": 0.0,
            "proposal_editable_weight_sum": 0.0,
            "warp_novel_squared_error_sum": 0.0,
            "warp_novel_weight_sum": 0.0,
            "warp_editable_squared_error_sum": 0.0,
            "warp_editable_weight_sum": 0.0,
            "support_composed_novel_squared_error_sum": 0.0,
            "support_composed_novel_weight_sum": 0.0,
            "support_composed_editable_squared_error_sum": 0.0,
            "support_composed_editable_weight_sum": 0.0,
            "pure_x0_novel_squared_error_sum": 0.0,
            "pure_x0_novel_weight_sum": 0.0,
            "pure_x0_editable_squared_error_sum": 0.0,
            "pure_x0_editable_weight_sum": 0.0,
            "accepted_patch_proposal_squared_error_sum": 0.0,
            "accepted_patch_proposal_weight_sum": 0.0,
            "accepted_patch_warp_squared_error_sum": 0.0,
            "accepted_patch_warp_weight_sum": 0.0,
            "accepted_patch_coverage_sum": 0.0,
            "residual_outside_acceptance_abs_sum": 0.0,
            "residual_outside_acceptance_weight_sum": 0.0,
            "support_projection_coverage_sum": 0.0,
            "editable_coverage_sum": 0.0,
            "novel_coverage_sum": 0.0,
            "validation_batch_count": 0.0,
            "validation_item_count": 0.0,
        }
        resolved_generation_mode = str(generation_mode).strip().lower()
        if resolved_generation_mode not in {"full_generate", "trainable_sample"}:
            resolved_generation_mode = "full_generate"
        resolved_max_batches = max(0, int(max_batches))
        per_sample_metric_records: List[Dict[str, float]] = []

        def _slice_batch_value(
            batch_value: Any,
            sample_index: int,
            batch_size: int,
        ) -> Any:
            if not isinstance(batch_value, torch.Tensor):
                return batch_value
            if batch_value.dim() == 0 or batch_value.shape[0] != batch_size:
                return batch_value
            return batch_value[sample_index : sample_index + 1]

        def _resolve_sample_scalar(
            batch_value: Any,
            sample_index: int,
            batch_size: int,
        ) -> Optional[float]:
            extracted_values = self._extract_float_sequence(
                _slice_batch_value(batch_value, sample_index, batch_size)
            )
            if not extracted_values and not isinstance(batch_value, torch.Tensor):
                fallback_values = self._extract_float_sequence(batch_value)
                if len(fallback_values) == batch_size:
                    extracted_values = [fallback_values[sample_index]]
            if not extracted_values:
                return None
            return float(sum(extracted_values) / float(len(extracted_values)))

        def _compute_record_psnr(
            metric_records: List[Dict[str, float]],
            squared_error_key: str,
            weight_key: str,
        ) -> Optional[float]:
            return self._compute_psnr_from_error_statistics(
                squared_error_sum=sum(
                    float(metric_record.get(squared_error_key, 0.0))
                    for metric_record in metric_records
                ),
                weight_sum=sum(
                    float(metric_record.get(weight_key, 0.0))
                    for metric_record in metric_records
                ),
            )

        def _compute_record_mean(
            metric_records: List[Dict[str, float]],
            metric_key: str,
        ) -> Optional[float]:
            if not metric_records:
                return None
            return float(
                sum(float(metric_record.get(metric_key, 0.0)) for metric_record in metric_records)
                / float(len(metric_records))
            )
        try:
            with torch.no_grad():
                for batch in self.val_dataloader:
                    if (
                        resolved_max_batches > 0
                        and aggregated_metric_values["validation_batch_count"] >= resolved_max_batches
                    ):
                        break
                    batch = self._to_device(batch)
                    if resolved_generation_mode == "trainable_sample":
                        generated_output = self._generate_trainable_sample(
                            batch,
                            steps=self.stage1_trainable_sample_steps,
                            sampler_type=self.sampler_type,
                            return_diagnostics=True,
                        )
                        if generated_output is None:
                            generated_output = self._generate_sample(
                                batch,
                                return_diagnostics=True,
                            )
                    else:
                        generated_output = self._generate_sample(
                            batch,
                            return_diagnostics=True,
                        )
                    if isinstance(generated_output, tuple):
                        proposal_rgb = generated_output[0]
                        sample_debug_tensors: Dict[str, Any] = {}
                    elif isinstance(generated_output, dict):
                        proposal_rgb = generated_output.get(
                            "rgb",
                            generated_output.get("image"),
                        )
                        sample_debug_tensors = generated_output.get("debug_tensors", {})
                        if not isinstance(sample_debug_tensors, dict):
                            sample_debug_tensors = {}
                    else:
                        proposal_rgb = generated_output
                        sample_debug_tensors = {}
                    if not isinstance(proposal_rgb, torch.Tensor):
                        continue
                    target_rgb = batch.get("target_rgb")
                    if not isinstance(target_rgb, torch.Tensor):
                        continue
                    proposal_reference_rgb = self._resolve_proposal_reference_rgb_from_debug_tensors(
                        sample_debug_tensors,
                        fallback_rgb=proposal_rgb,
                    )
                    if proposal_reference_rgb is None:
                        proposal_reference_rgb = proposal_rgb
                    support_composed_image = sample_debug_tensors.get("support_composed_image")
                    warp_reference_rgb = sample_debug_tensors.get("warped_rgb")
                    if not isinstance(warp_reference_rgb, torch.Tensor):
                        warp_reference_rgb = batch.get("rgb_render")
                    true_novel_inpaint_mask = sample_debug_tensors.get("true_novel_inpaint_mask")
                    if true_novel_inpaint_mask is None:
                        true_novel_inpaint_mask = sample_debug_tensors.get("inpaint_mask")
                    editable_mask = sample_debug_tensors.get("masked_inpaint_editable_mask")
                    support_projection_mask = sample_debug_tensors.get("support_projection_mask")
                    pure_x0_reference_rgb = sample_debug_tensors.get("decoded_teacher_rgb")
                    proposal_acceptance_confidence = sample_debug_tensors.get(
                        "proposal_acceptance_confidence"
                    )
                    proposal_acceptance_gate = sample_debug_tensors.get(
                        "proposal_acceptance_gate"
                    )
                    proposal_residual_rgb = sample_debug_tensors.get("proposal_residual_rgb")
                    proposal_applied_residual_rgb = sample_debug_tensors.get(
                        "proposal_applied_residual_rgb"
                    )
                    batch_size = (
                        int(target_rgb.shape[0])
                        if isinstance(target_rgb, torch.Tensor) and target_rgb.dim() >= 4
                        else 1
                    )
                    for sample_index in range(batch_size):
                        sample_target_rgb = _slice_batch_value(
                            target_rgb,
                            sample_index,
                            batch_size,
                        )
                        sample_proposal_reference_rgb = _slice_batch_value(
                            proposal_reference_rgb,
                            sample_index,
                            batch_size,
                        )
                        sample_support_composed_image = _slice_batch_value(
                            support_composed_image,
                            sample_index,
                            batch_size,
                        )
                        sample_warp_reference_rgb = _slice_batch_value(
                            warp_reference_rgb,
                            sample_index,
                            batch_size,
                        )
                        sample_true_novel_inpaint_mask = _slice_batch_value(
                            true_novel_inpaint_mask,
                            sample_index,
                            batch_size,
                        )
                        sample_editable_mask = _slice_batch_value(
                            editable_mask,
                            sample_index,
                            batch_size,
                        )
                        sample_support_projection_mask = _slice_batch_value(
                            support_projection_mask,
                            sample_index,
                            batch_size,
                        )
                        sample_pure_x0_reference_rgb = _slice_batch_value(
                            pure_x0_reference_rgb,
                            sample_index,
                            batch_size,
                        )
                        sample_proposal_acceptance_confidence = _slice_batch_value(
                            proposal_acceptance_confidence,
                            sample_index,
                            batch_size,
                        )
                        sample_proposal_acceptance_gate = _slice_batch_value(
                            proposal_acceptance_gate,
                            sample_index,
                            batch_size,
                        )
                        sample_proposal_residual_rgb = _slice_batch_value(
                            proposal_residual_rgb,
                            sample_index,
                            batch_size,
                        )
                        sample_proposal_applied_residual_rgb = _slice_batch_value(
                            proposal_applied_residual_rgb,
                            sample_index,
                            batch_size,
                        )

                        whole_image_error_sum, whole_image_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_proposal_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=None,
                            )
                        )
                        aggregated_metric_values["whole_image_squared_error_sum"] += whole_image_error_sum
                        aggregated_metric_values["whole_image_weight_sum"] += whole_image_weight_sum

                        proposal_novel_error_sum, proposal_novel_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_proposal_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_true_novel_inpaint_mask,
                            )
                        )
                        aggregated_metric_values["proposal_novel_squared_error_sum"] += proposal_novel_error_sum
                        aggregated_metric_values["proposal_novel_weight_sum"] += proposal_novel_weight_sum

                        proposal_editable_error_sum, proposal_editable_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_proposal_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_editable_mask,
                            )
                        )
                        aggregated_metric_values["proposal_editable_squared_error_sum"] += proposal_editable_error_sum
                        aggregated_metric_values["proposal_editable_weight_sum"] += proposal_editable_weight_sum

                        warp_novel_error_sum, warp_novel_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_warp_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_true_novel_inpaint_mask,
                            )
                        )
                        aggregated_metric_values["warp_novel_squared_error_sum"] += warp_novel_error_sum
                        aggregated_metric_values["warp_novel_weight_sum"] += warp_novel_weight_sum

                        warp_editable_error_sum, warp_editable_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_warp_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_editable_mask,
                            )
                        )
                        aggregated_metric_values["warp_editable_squared_error_sum"] += (
                            warp_editable_error_sum
                        )
                        aggregated_metric_values["warp_editable_weight_sum"] += (
                            warp_editable_weight_sum
                        )

                        support_composed_novel_error_sum, support_composed_novel_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_support_composed_image,
                                target_rgb=sample_target_rgb,
                                mask=sample_true_novel_inpaint_mask,
                            )
                        )
                        aggregated_metric_values["support_composed_novel_squared_error_sum"] += (
                            support_composed_novel_error_sum
                        )
                        aggregated_metric_values["support_composed_novel_weight_sum"] += (
                            support_composed_novel_weight_sum
                        )

                        support_composed_editable_error_sum, support_composed_editable_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_support_composed_image,
                                target_rgb=sample_target_rgb,
                                mask=sample_editable_mask,
                            )
                        )
                        aggregated_metric_values["support_composed_editable_squared_error_sum"] += (
                            support_composed_editable_error_sum
                        )
                        aggregated_metric_values["support_composed_editable_weight_sum"] += (
                            support_composed_editable_weight_sum
                        )

                        pure_x0_novel_error_sum, pure_x0_novel_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_pure_x0_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_true_novel_inpaint_mask,
                            )
                        )
                        aggregated_metric_values["pure_x0_novel_squared_error_sum"] += (
                            pure_x0_novel_error_sum
                        )
                        aggregated_metric_values["pure_x0_novel_weight_sum"] += (
                            pure_x0_novel_weight_sum
                        )
                        pure_x0_editable_error_sum, pure_x0_editable_weight_sum = (
                            self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_pure_x0_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=sample_editable_mask,
                            )
                        )
                        aggregated_metric_values["pure_x0_editable_squared_error_sum"] += (
                            pure_x0_editable_error_sum
                        )
                        aggregated_metric_values["pure_x0_editable_weight_sum"] += (
                            pure_x0_editable_weight_sum
                        )

                        accepted_patch_mask = None
                        accepted_patch_source = (
                            sample_proposal_acceptance_gate
                            if isinstance(sample_proposal_acceptance_gate, torch.Tensor)
                            else sample_proposal_acceptance_confidence
                        )
                        if isinstance(accepted_patch_source, torch.Tensor):
                            accepted_patch_mask = accepted_patch_source
                            if accepted_patch_mask.dim() == 3:
                                accepted_patch_mask = accepted_patch_mask.unsqueeze(1)
                            if accepted_patch_mask.shape[1] > 1:
                                accepted_patch_mask = accepted_patch_mask[:, :1]
                            accepted_patch_mask = torch.nan_to_num(
                                accepted_patch_mask.to(
                                    device=sample_target_rgb.device,
                                    dtype=sample_target_rgb.dtype,
                                ),
                                nan=0.0,
                                posinf=1.0,
                                neginf=0.0,
                            ).clamp(0.0, 1.0)
                            if accepted_patch_mask.shape[-2:] != sample_target_rgb.shape[-2:]:
                                accepted_patch_mask = F.interpolate(
                                    accepted_patch_mask,
                                    size=sample_target_rgb.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                )
                            accepted_patch_threshold = min(
                                max(
                                    float(
                                        getattr(
                                            self,
                                            "proposal_validation_acceptance_threshold",
                                            0.5,
                                        )
                                    ),
                                    0.0,
                                ),
                                1.0,
                            )
                            accepted_patch_mask = (
                                accepted_patch_mask >= accepted_patch_threshold
                            ).to(dtype=sample_target_rgb.dtype)
                            if isinstance(sample_editable_mask, torch.Tensor):
                                editable_acceptance_mask = sample_editable_mask
                                if editable_acceptance_mask.dim() == 3:
                                    editable_acceptance_mask = editable_acceptance_mask.unsqueeze(1)
                                if editable_acceptance_mask.shape[1] > 1:
                                    editable_acceptance_mask = editable_acceptance_mask[:, :1]
                                editable_acceptance_mask = editable_acceptance_mask.to(
                                    device=sample_target_rgb.device,
                                    dtype=sample_target_rgb.dtype,
                                )
                                if editable_acceptance_mask.shape[-2:] != sample_target_rgb.shape[-2:]:
                                    editable_acceptance_mask = F.interpolate(
                                        editable_acceptance_mask,
                                        size=sample_target_rgb.shape[-2:],
                                        mode="nearest",
                                    )
                                accepted_patch_mask = accepted_patch_mask * editable_acceptance_mask
                            if float(accepted_patch_mask.sum().item()) < 1.0:
                                accepted_patch_mask = None

                        accepted_patch_proposal_error_sum = 0.0
                        accepted_patch_proposal_weight_sum = 0.0
                        accepted_patch_warp_error_sum = 0.0
                        accepted_patch_warp_weight_sum = 0.0
                        accepted_patch_coverage = 0.0
                        residual_outside_acceptance_abs_sum = 0.0
                        residual_outside_acceptance_weight_sum = 0.0
                        if isinstance(accepted_patch_mask, torch.Tensor):
                            accepted_patch_coverage = float(
                                accepted_patch_mask.detach().float().mean().item()
                            )
                            (
                                accepted_patch_proposal_error_sum,
                                accepted_patch_proposal_weight_sum,
                            ) = self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_proposal_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=accepted_patch_mask,
                            )
                            (
                                accepted_patch_warp_error_sum,
                                accepted_patch_warp_weight_sum,
                            ) = self._compute_rgb_squared_error_statistics(
                                predicted_rgb=sample_warp_reference_rgb,
                                target_rgb=sample_target_rgb,
                                mask=accepted_patch_mask,
                            )
                            residual_for_rejection_metric = (
                                sample_proposal_applied_residual_rgb
                                if isinstance(sample_proposal_applied_residual_rgb, torch.Tensor)
                                else sample_proposal_residual_rgb
                            )
                            if isinstance(residual_for_rejection_metric, torch.Tensor):
                                aligned_residual_rgb = residual_for_rejection_metric.to(
                                    device=sample_target_rgb.device,
                                    dtype=sample_target_rgb.dtype,
                                )
                                if aligned_residual_rgb.shape[-2:] != sample_target_rgb.shape[-2:]:
                                    aligned_residual_rgb = F.interpolate(
                                        aligned_residual_rgb,
                                        size=sample_target_rgb.shape[-2:],
                                        mode="bilinear",
                                        align_corners=False,
                                    )
                                residual_rejection_mask = torch.clamp(
                                    1.0 - accepted_patch_mask,
                                    min=0.0,
                                    max=1.0,
                                )
                                residual_outside_acceptance_abs_sum = float(
                                    (
                                        aligned_residual_rgb.abs()
                                        * residual_rejection_mask
                                    ).sum().item()
                                )
                                residual_outside_acceptance_weight_sum = float(
                                    residual_rejection_mask.sum().item()
                                    * aligned_residual_rgb.shape[1]
                                )
                        aggregated_metric_values["accepted_patch_proposal_squared_error_sum"] += (
                            accepted_patch_proposal_error_sum
                        )
                        aggregated_metric_values["accepted_patch_proposal_weight_sum"] += (
                            accepted_patch_proposal_weight_sum
                        )
                        aggregated_metric_values["accepted_patch_warp_squared_error_sum"] += (
                            accepted_patch_warp_error_sum
                        )
                        aggregated_metric_values["accepted_patch_warp_weight_sum"] += (
                            accepted_patch_warp_weight_sum
                        )
                        aggregated_metric_values["accepted_patch_coverage_sum"] += float(
                            accepted_patch_coverage
                        )
                        aggregated_metric_values["residual_outside_acceptance_abs_sum"] += (
                            residual_outside_acceptance_abs_sum
                        )
                        aggregated_metric_values["residual_outside_acceptance_weight_sum"] += (
                            residual_outside_acceptance_weight_sum
                        )

                        support_projection_coverage = self._tensor_mean_value(
                            sample_support_projection_mask
                        )
                        if support_projection_coverage is not None:
                            aggregated_metric_values["support_projection_coverage_sum"] += float(
                                support_projection_coverage
                            )
                        editable_coverage = self._tensor_mean_value(sample_editable_mask)
                        if editable_coverage is not None:
                            aggregated_metric_values["editable_coverage_sum"] += float(
                                editable_coverage
                            )
                        novel_coverage = self._tensor_mean_value(sample_true_novel_inpaint_mask)
                        if novel_coverage is not None:
                            aggregated_metric_values["novel_coverage_sum"] += float(
                                novel_coverage
                            )

                        sample_hardness_score = _resolve_sample_scalar(
                            batch.get("episode_proposal_hardness_score"),
                            sample_index,
                            batch_size,
                        )
                        if sample_hardness_score is None:
                            fallback_hardness_terms: List[float] = []
                            frontier_coverage = _resolve_sample_scalar(
                                batch.get("episode_frontier_coverage"),
                                sample_index,
                                batch_size,
                            )
                            if frontier_coverage is not None:
                                fallback_hardness_terms.append(float(frontier_coverage))
                            if novel_coverage is not None:
                                fallback_hardness_terms.append(float(novel_coverage))
                            if editable_coverage is not None:
                                fallback_hardness_terms.append(float(editable_coverage))
                            if fallback_hardness_terms:
                                sample_hardness_score = float(
                                    sum(fallback_hardness_terms) / float(len(fallback_hardness_terms))
                                )

                        per_sample_metric_records.append(
                            {
                                "whole_image_squared_error_sum": float(whole_image_error_sum),
                                "whole_image_weight_sum": float(whole_image_weight_sum),
                                "proposal_novel_squared_error_sum": float(proposal_novel_error_sum),
                                "proposal_novel_weight_sum": float(proposal_novel_weight_sum),
                                "proposal_editable_squared_error_sum": float(proposal_editable_error_sum),
                                "proposal_editable_weight_sum": float(proposal_editable_weight_sum),
                                "warp_novel_squared_error_sum": float(warp_novel_error_sum),
                                "warp_novel_weight_sum": float(warp_novel_weight_sum),
                                "warp_editable_squared_error_sum": float(warp_editable_error_sum),
                                "warp_editable_weight_sum": float(warp_editable_weight_sum),
                                "support_composed_novel_squared_error_sum": float(
                                    support_composed_novel_error_sum
                                ),
                                "support_composed_novel_weight_sum": float(
                                    support_composed_novel_weight_sum
                                ),
                                "support_composed_editable_squared_error_sum": float(
                                    support_composed_editable_error_sum
                                ),
                                "support_composed_editable_weight_sum": float(
                                    support_composed_editable_weight_sum
                                ),
                                "pure_x0_novel_squared_error_sum": float(
                                    pure_x0_novel_error_sum
                                ),
                                "pure_x0_novel_weight_sum": float(
                                    pure_x0_novel_weight_sum
                                ),
                                "pure_x0_editable_squared_error_sum": float(
                                    pure_x0_editable_error_sum
                                ),
                                "pure_x0_editable_weight_sum": float(
                                    pure_x0_editable_weight_sum
                                ),
                                "accepted_patch_proposal_squared_error_sum": float(
                                    accepted_patch_proposal_error_sum
                                ),
                                "accepted_patch_proposal_weight_sum": float(
                                    accepted_patch_proposal_weight_sum
                                ),
                                "accepted_patch_warp_squared_error_sum": float(
                                    accepted_patch_warp_error_sum
                                ),
                                "accepted_patch_warp_weight_sum": float(
                                    accepted_patch_warp_weight_sum
                                ),
                                "accepted_patch_coverage": float(accepted_patch_coverage),
                                "residual_outside_acceptance_abs_sum": float(
                                    residual_outside_acceptance_abs_sum
                                ),
                                "residual_outside_acceptance_weight_sum": float(
                                    residual_outside_acceptance_weight_sum
                                ),
                                "support_projection_coverage": float(
                                    support_projection_coverage or 0.0
                                ),
                                "editable_coverage": float(editable_coverage or 0.0),
                                "novel_coverage": float(novel_coverage or 0.0),
                                "hardness_score": float(sample_hardness_score or 0.0),
                            }
                        )
                        aggregated_metric_values["validation_item_count"] += 1.0
                    aggregated_metric_values["validation_batch_count"] += 1.0

            reduced_metric_names = list(aggregated_metric_values.keys())
            reduced_metric_tensor = torch.tensor(
                [aggregated_metric_values[metric_name] for metric_name in reduced_metric_names],
                device=self.device,
                dtype=torch.float64,
            )
            if self.world_size > 1:
                dist.all_reduce(reduced_metric_tensor, op=dist.ReduceOp.SUM)
            reduced_metric_values = {
                metric_name: float(metric_value)
                for metric_name, metric_value in zip(
                    reduced_metric_names,
                    reduced_metric_tensor.tolist(),
                )
            }
            validation_item_count = max(
                reduced_metric_values["validation_item_count"],
                1.0,
            )
            validation_metrics: Dict[str, float] = {
                "whole_image_psnr": float(
                    self._compute_psnr_from_error_statistics(
                        squared_error_sum=reduced_metric_values["whole_image_squared_error_sum"],
                        weight_sum=reduced_metric_values["whole_image_weight_sum"],
                    )
                    or 0.0
                ),
                "mean_support_projection_coverage": float(
                    reduced_metric_values["support_projection_coverage_sum"] / validation_item_count
                ),
                "mean_editable_coverage": float(
                    reduced_metric_values["editable_coverage_sum"] / validation_item_count
                ),
                "mean_novel_coverage": float(
                    reduced_metric_values["novel_coverage_sum"] / validation_item_count
                ),
                "validation_batches": float(reduced_metric_values["validation_batch_count"]),
                "validation_items": float(reduced_metric_values["validation_item_count"]),
            }
            proposal_novel_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["proposal_novel_squared_error_sum"],
                weight_sum=reduced_metric_values["proposal_novel_weight_sum"],
            )
            if proposal_novel_psnr is not None:
                validation_metrics["proposal_novel_psnr"] = float(proposal_novel_psnr)
            proposal_editable_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["proposal_editable_squared_error_sum"],
                weight_sum=reduced_metric_values["proposal_editable_weight_sum"],
            )
            if proposal_editable_psnr is not None:
                validation_metrics["proposal_editable_psnr"] = float(proposal_editable_psnr)
            warp_novel_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["warp_novel_squared_error_sum"],
                weight_sum=reduced_metric_values["warp_novel_weight_sum"],
            )
            if warp_novel_psnr is not None:
                validation_metrics["warp_novel_psnr"] = float(warp_novel_psnr)
            warp_editable_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["warp_editable_squared_error_sum"],
                weight_sum=reduced_metric_values["warp_editable_weight_sum"],
            )
            if warp_editable_psnr is not None:
                validation_metrics["warp_editable_psnr"] = float(warp_editable_psnr)
            support_composed_novel_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["support_composed_novel_squared_error_sum"],
                weight_sum=reduced_metric_values["support_composed_novel_weight_sum"],
            )
            if support_composed_novel_psnr is not None:
                validation_metrics["support_composed_novel_psnr"] = float(
                    support_composed_novel_psnr
                )
            support_composed_editable_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["support_composed_editable_squared_error_sum"],
                weight_sum=reduced_metric_values["support_composed_editable_weight_sum"],
            )
            if support_composed_editable_psnr is not None:
                validation_metrics["support_composed_editable_psnr"] = float(
                    support_composed_editable_psnr
                )
            pure_x0_novel_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["pure_x0_novel_squared_error_sum"],
                weight_sum=reduced_metric_values["pure_x0_novel_weight_sum"],
            )
            if pure_x0_novel_psnr is not None:
                validation_metrics["pure_x0_novel_psnr"] = float(pure_x0_novel_psnr)
            pure_x0_editable_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values["pure_x0_editable_squared_error_sum"],
                weight_sum=reduced_metric_values["pure_x0_editable_weight_sum"],
            )
            if pure_x0_editable_psnr is not None:
                validation_metrics["pure_x0_editable_psnr"] = float(pure_x0_editable_psnr)
            accepted_patch_proposal_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values[
                    "accepted_patch_proposal_squared_error_sum"
                ],
                weight_sum=reduced_metric_values["accepted_patch_proposal_weight_sum"],
            )
            if accepted_patch_proposal_psnr is not None:
                validation_metrics["accepted_patch_proposal_psnr"] = float(
                    accepted_patch_proposal_psnr
                )
            accepted_patch_warp_psnr = self._compute_psnr_from_error_statistics(
                squared_error_sum=reduced_metric_values[
                    "accepted_patch_warp_squared_error_sum"
                ],
                weight_sum=reduced_metric_values["accepted_patch_warp_weight_sum"],
            )
            if accepted_patch_warp_psnr is not None:
                validation_metrics["accepted_patch_warp_psnr"] = float(
                    accepted_patch_warp_psnr
                )
            if (
                accepted_patch_proposal_psnr is not None
                and accepted_patch_warp_psnr is not None
            ):
                validation_metrics["accepted_patch_minus_warp_psnr"] = float(
                    accepted_patch_proposal_psnr - accepted_patch_warp_psnr
                )
            validation_metrics["mean_accepted_patch_coverage"] = float(
                reduced_metric_values["accepted_patch_coverage_sum"] / validation_item_count
            )
            residual_outside_acceptance_weight_sum = reduced_metric_values[
                "residual_outside_acceptance_weight_sum"
            ]
            if residual_outside_acceptance_weight_sum > 0.0:
                validation_metrics["residual_outside_acceptance_abs_mean"] = float(
                    reduced_metric_values["residual_outside_acceptance_abs_sum"]
                    / residual_outside_acceptance_weight_sum
                )
            if (
                proposal_novel_psnr is not None
                and support_composed_novel_psnr is not None
            ):
                validation_metrics["compose_gain_vs_pre_novel_psnr"] = float(
                    support_composed_novel_psnr - proposal_novel_psnr
                )
            if (
                proposal_editable_psnr is not None
                and support_composed_editable_psnr is not None
            ):
                validation_metrics["compose_gain_vs_pre_editable_psnr"] = float(
                    support_composed_editable_psnr - proposal_editable_psnr
                )
            if proposal_novel_psnr is not None and warp_novel_psnr is not None:
                validation_metrics["proposal_minus_warp_novel_psnr"] = float(
                    proposal_novel_psnr - warp_novel_psnr
                )
            if proposal_editable_psnr is not None and warp_editable_psnr is not None:
                validation_metrics["proposal_minus_warp_editable_psnr"] = float(
                    proposal_editable_psnr - warp_editable_psnr
                )
            validation_score_terms: List[float] = []
            if proposal_novel_psnr is not None:
                validation_score_terms.append(float(proposal_novel_psnr))
            if proposal_editable_psnr is not None:
                validation_score_terms.append(float(proposal_editable_psnr))
            if validation_score_terms:
                validation_metrics["proposal_validation_score"] = float(
                    sum(validation_score_terms) / float(len(validation_score_terms))
                )
            else:
                validation_metrics["proposal_validation_score"] = float(
                    validation_metrics["whole_image_psnr"]
                )
            all_sample_metric_records = per_sample_metric_records
            if self.world_size > 1:
                gathered_sample_metric_records: List[Any] = [None for _ in range(self.world_size)]
                dist.all_gather_object(gathered_sample_metric_records, per_sample_metric_records)
                all_sample_metric_records = []
                for gathered_rank_records in gathered_sample_metric_records:
                    if not isinstance(gathered_rank_records, list):
                        continue
                    for gathered_metric_record in gathered_rank_records:
                        if isinstance(gathered_metric_record, dict):
                            all_sample_metric_records.append(gathered_metric_record)
            hard_metric_records: List[Dict[str, float]] = []
            if all_sample_metric_records:
                ranked_metric_records = sorted(
                    all_sample_metric_records,
                    key=lambda metric_record: float(metric_record.get("hardness_score", 0.0)),
                    reverse=True,
                )
                resolved_hard_episode_count = max(
                    1,
                    int(math.ceil(float(len(ranked_metric_records)) * 0.5)),
                )
                hard_metric_records = ranked_metric_records[:resolved_hard_episode_count]
            if hard_metric_records:
                validation_metrics["hard_episode_count"] = float(len(hard_metric_records))
                validation_metrics["hard_episode_ratio"] = float(
                    len(hard_metric_records) / float(max(len(all_sample_metric_records), 1))
                )
                hard_mean_novel_coverage = _compute_record_mean(
                    hard_metric_records,
                    "novel_coverage",
                )
                if hard_mean_novel_coverage is not None:
                    validation_metrics["hard_mean_novel_coverage"] = float(
                        hard_mean_novel_coverage
                    )
                hard_proposal_novel_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "proposal_novel_squared_error_sum",
                    "proposal_novel_weight_sum",
                )
                if hard_proposal_novel_psnr is not None:
                    validation_metrics["hard_proposal_novel_psnr"] = float(
                        hard_proposal_novel_psnr
                    )
                hard_proposal_editable_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "proposal_editable_squared_error_sum",
                    "proposal_editable_weight_sum",
                )
                if hard_proposal_editable_psnr is not None:
                    validation_metrics["hard_proposal_editable_psnr"] = float(
                        hard_proposal_editable_psnr
                    )
                hard_warp_novel_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "warp_novel_squared_error_sum",
                    "warp_novel_weight_sum",
                )
                if hard_warp_novel_psnr is not None:
                    validation_metrics["hard_warp_novel_psnr"] = float(
                        hard_warp_novel_psnr
                    )
                hard_warp_editable_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "warp_editable_squared_error_sum",
                    "warp_editable_weight_sum",
                )
                if hard_warp_editable_psnr is not None:
                    validation_metrics["hard_warp_editable_psnr"] = float(
                        hard_warp_editable_psnr
                    )
                hard_support_composed_novel_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "support_composed_novel_squared_error_sum",
                    "support_composed_novel_weight_sum",
                )
                if hard_support_composed_novel_psnr is not None:
                    validation_metrics["hard_support_composed_novel_psnr"] = float(
                        hard_support_composed_novel_psnr
                    )
                hard_support_composed_editable_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "support_composed_editable_squared_error_sum",
                    "support_composed_editable_weight_sum",
                )
                if hard_support_composed_editable_psnr is not None:
                    validation_metrics["hard_support_composed_editable_psnr"] = float(
                        hard_support_composed_editable_psnr
                    )
                hard_pure_x0_novel_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "pure_x0_novel_squared_error_sum",
                    "pure_x0_novel_weight_sum",
                )
                if hard_pure_x0_novel_psnr is not None:
                    validation_metrics["hard_pure_x0_novel_psnr"] = float(
                        hard_pure_x0_novel_psnr
                    )
                hard_pure_x0_editable_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "pure_x0_editable_squared_error_sum",
                    "pure_x0_editable_weight_sum",
                )
                if hard_pure_x0_editable_psnr is not None:
                    validation_metrics["hard_pure_x0_editable_psnr"] = float(
                        hard_pure_x0_editable_psnr
                    )
                hard_accepted_patch_proposal_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "accepted_patch_proposal_squared_error_sum",
                    "accepted_patch_proposal_weight_sum",
                )
                if hard_accepted_patch_proposal_psnr is not None:
                    validation_metrics["hard_accepted_patch_proposal_psnr"] = float(
                        hard_accepted_patch_proposal_psnr
                    )
                hard_accepted_patch_warp_psnr = _compute_record_psnr(
                    hard_metric_records,
                    "accepted_patch_warp_squared_error_sum",
                    "accepted_patch_warp_weight_sum",
                )
                if hard_accepted_patch_warp_psnr is not None:
                    validation_metrics["hard_accepted_patch_warp_psnr"] = float(
                        hard_accepted_patch_warp_psnr
                    )
                if (
                    hard_accepted_patch_proposal_psnr is not None
                    and hard_accepted_patch_warp_psnr is not None
                ):
                    validation_metrics["hard_accepted_patch_minus_warp_psnr"] = float(
                        hard_accepted_patch_proposal_psnr
                        - hard_accepted_patch_warp_psnr
                    )
                hard_accepted_patch_coverage = _compute_record_mean(
                    hard_metric_records,
                    "accepted_patch_coverage",
                )
                if hard_accepted_patch_coverage is not None:
                    validation_metrics["hard_mean_accepted_patch_coverage"] = float(
                        hard_accepted_patch_coverage
                    )
                if (
                    hard_proposal_novel_psnr is not None
                    and hard_support_composed_novel_psnr is not None
                ):
                    validation_metrics["hard_compose_gain_vs_pre_novel_psnr"] = float(
                        hard_support_composed_novel_psnr - hard_proposal_novel_psnr
                    )
                if (
                    hard_proposal_editable_psnr is not None
                    and hard_support_composed_editable_psnr is not None
                ):
                    validation_metrics["hard_compose_gain_vs_pre_editable_psnr"] = float(
                        hard_support_composed_editable_psnr - hard_proposal_editable_psnr
                    )
                if hard_proposal_novel_psnr is not None and hard_warp_novel_psnr is not None:
                    validation_metrics["hard_proposal_minus_warp_novel_psnr"] = float(
                        hard_proposal_novel_psnr - hard_warp_novel_psnr
                    )
                if (
                    hard_proposal_editable_psnr is not None
                    and hard_warp_editable_psnr is not None
                ):
                    validation_metrics["hard_proposal_minus_warp_editable_psnr"] = float(
                        hard_proposal_editable_psnr - hard_warp_editable_psnr
                    )
                hard_validation_score_terms: List[float] = []
                if hard_proposal_novel_psnr is not None:
                    hard_validation_score_terms.append(float(hard_proposal_novel_psnr))
                if hard_proposal_editable_psnr is not None:
                    hard_validation_score_terms.append(float(hard_proposal_editable_psnr))
                if hard_validation_score_terms:
                    validation_metrics["hard_validation_score"] = float(
                        sum(hard_validation_score_terms)
                        / float(len(hard_validation_score_terms))
                    )
            if bool(getattr(self, "validation_use_warp_delta_score", False)):
                warp_delta_score_terms: List[float] = []
                if "hard_accepted_patch_minus_warp_psnr" in validation_metrics:
                    warp_delta_score_terms.append(
                        float(validation_metrics["hard_accepted_patch_minus_warp_psnr"])
                    )
                if "hard_proposal_minus_warp_novel_psnr" in validation_metrics:
                    warp_delta_score_terms.append(
                        float(validation_metrics["hard_proposal_minus_warp_novel_psnr"])
                    )
                if "hard_proposal_minus_warp_editable_psnr" in validation_metrics:
                    warp_delta_score_terms.append(
                        float(validation_metrics["hard_proposal_minus_warp_editable_psnr"])
                    )
                if not warp_delta_score_terms:
                    if "proposal_minus_warp_novel_psnr" in validation_metrics:
                        warp_delta_score_terms.append(
                            float(validation_metrics["proposal_minus_warp_novel_psnr"])
                        )
                    if "proposal_minus_warp_editable_psnr" in validation_metrics:
                        warp_delta_score_terms.append(
                            float(validation_metrics["proposal_minus_warp_editable_psnr"])
                        )
                    if "accepted_patch_minus_warp_psnr" in validation_metrics:
                        warp_delta_score_terms.append(
                            float(validation_metrics["accepted_patch_minus_warp_psnr"])
                        )
                if warp_delta_score_terms:
                    validation_metrics["validation_score"] = float(
                        sum(warp_delta_score_terms) / float(len(warp_delta_score_terms))
                    )
                else:
                    validation_metrics["validation_score"] = float(
                        validation_metrics.get(
                            "hard_validation_score",
                            validation_metrics.get(
                                "proposal_validation_score",
                                validation_metrics["whole_image_psnr"],
                            ),
                        )
                    )
            else:
                validation_metrics["validation_score"] = float(
                    validation_metrics.get(
                        "hard_validation_score",
                        validation_metrics.get(
                            "proposal_validation_score",
                            validation_metrics["whole_image_psnr"],
                        ),
                    )
                )
            validation_metrics["val_psnr"] = float(validation_metrics["whole_image_psnr"])
            return validation_metrics
        finally:
            if previous_training_state:
                self.model.train()
            else:
                self.model.eval()

    def _collect_vae_params(
        self,
        vae_module: torch.nn.Module,
        *,
        decoder_only: bool,
    ) -> List[torch.nn.Parameter]:

        params: List[torch.nn.Parameter] = []
        if decoder_only:
            decoder = getattr(vae_module, "decoder", None)
            if decoder is not None:
                params.extend(list(decoder.parameters()))
            post_quant = getattr(vae_module, "post_quant_conv", None)
            if post_quant is not None:
                params.extend(list(post_quant.parameters()))
        else:
            params.extend(list(vae_module.parameters()))
        return params

    def _record_stage3_metrics(
        self,
        *,
        psnr: float,
        ece: float,
        corr: float,
        validation_score: Optional[float],
    ) -> None:
        self._stage3_metric_history.append({"psnr": psnr, "ece": ece, "corr": corr})
        reference_psnr = validation_score if validation_score is not None else psnr
        self._stage3_psnr_history.append(reference_psnr)

    def _write_validation_metrics(
        self,
        *,
        writer_prefix: str,
        validation_metrics: Dict[str, float],
        epoch: int,
    ) -> None:
        if self.writer is None or not self._is_main_rank:
            return
        for metric_name, metric_value in validation_metrics.items():
            self.writer.add_scalar(
                f"{writer_prefix}/validation/{metric_name}",
                float(metric_value),
                epoch,
            )
        if "whole_image_psnr" in validation_metrics:
            self.writer.add_scalar(
                f"{writer_prefix}/val_psnr",
                float(validation_metrics["whole_image_psnr"]),
                epoch,
            )
        if "validation_score" in validation_metrics:
            self.writer.add_scalar(
                f"{writer_prefix}/validation_score",
                float(validation_metrics["validation_score"]),
                epoch,
            )

    def _format_validation_metrics_for_log(
        self,
        validation_metrics: Dict[str, float],
    ) -> str:
        if not validation_metrics:
            return "validation=unavailable"
        formatted_segments: List[str] = []
        metric_display_order = (
            "hard_validation_score",
            "hard_proposal_minus_warp_novel_psnr",
            "hard_proposal_minus_warp_editable_psnr",
            "hard_accepted_patch_minus_warp_psnr",
            "hard_accepted_patch_proposal_psnr",
            "hard_accepted_patch_warp_psnr",
            "hard_proposal_novel_psnr",
            "hard_proposal_editable_psnr",
            "hard_pure_x0_novel_psnr",
            "hard_pure_x0_editable_psnr",
            "hard_compose_gain_vs_pre_novel_psnr",
            "hard_compose_gain_vs_pre_editable_psnr",
            "hard_mean_novel_coverage",
            "hard_mean_accepted_patch_coverage",
            "hard_episode_ratio",
            "validation_score",
            "proposal_minus_warp_novel_psnr",
            "proposal_minus_warp_editable_psnr",
            "accepted_patch_minus_warp_psnr",
            "accepted_patch_proposal_psnr",
            "accepted_patch_warp_psnr",
            "proposal_validation_score",
            "whole_image_psnr",
            "proposal_novel_psnr",
            "proposal_editable_psnr",
            "pure_x0_novel_psnr",
            "pure_x0_editable_psnr",
            "warp_novel_psnr",
            "warp_editable_psnr",
            "compose_gain_vs_pre_novel_psnr",
            "compose_gain_vs_pre_editable_psnr",
            "mean_novel_coverage",
            "mean_editable_coverage",
            "mean_accepted_patch_coverage",
            "residual_outside_acceptance_abs_mean",
        )
        for metric_name in metric_display_order:
            if metric_name not in validation_metrics:
                continue
            metric_value = float(validation_metrics[metric_name])
            if "psnr" in metric_name or metric_name.endswith("_score"):
                formatted_segments.append(f"{metric_name}={metric_value:.2f}")
            else:
                formatted_segments.append(f"{metric_name}={metric_value:.4f}")
        if not formatted_segments:
            return "validation=unavailable"
        return ", ".join(formatted_segments)

    def _maybe_trigger_auto_unfreeze(self, *, current_epoch: int) -> None:
        if not self.auto_unfreeze_vae:
            return
        if self._auto_unfreeze_activated:
            return
        if not self._auto_unfreeze_candidates:
            return
        if len(self._stage3_metric_history) < self.auto_unfreeze_patience:
            return
        if current_epoch + 1 < self.auto_unfreeze_warmup_epochs:
            return
        recent_metrics = self._stage3_metric_history[-self.auto_unfreeze_patience :]
        if not all(metric["ece"] <= self.auto_unfreeze_ece_threshold for metric in recent_metrics):
            return
        if not all(metric["corr"] >= self.auto_unfreeze_corr_threshold for metric in recent_metrics):
            return
        psnr_window_size = max(self.auto_unfreeze_psnr_window, self.auto_unfreeze_patience)
        psnr_window = self._stage3_psnr_history[-psnr_window_size:]
        if len(psnr_window) < psnr_window_size:
            return
        psnr_delta = max(psnr_window) - min(psnr_window)
        if psnr_delta > self.auto_unfreeze_psnr_delta:
            return
        for param in self._auto_unfreeze_candidates:
            param.requires_grad = True
        scaled_lr = float(self.stage3_base_lr * max(self.stage3_vae_lr_scale, 0.0))
        if scaled_lr <= 0.0:
            scaled_lr = self.stage3_base_lr
        self.optimizer.add_param_group({"params": self._auto_unfreeze_candidates, "lr": scaled_lr})
        self._auto_unfreeze_activated = True
        self.unfreeze_vae = True
        message = "[Stage3] 自动解冻：校准稳定，已纳入 VAE 解码路径以微调画质。"
        if not self.auto_unfreeze_decoder_only:
            message = "[Stage3] 自动解冻：校准稳定，已纳入 VAE 全量参数以微调画质。"
        if self.world_rank == 0:
            print(message)
        if self.writer is not None:
            self.writer.add_scalar("stage3/auto_unfreeze_epoch", current_epoch)

    # ----------------------------------------------------------------------
    # Helper utilities
    # ----------------------------------------------------------------------

    def _compute_target_hw(
        self,
        original_hw: Tuple[int, int],
        factor: float,
    ) -> Tuple[int, int]:
        if factor >= 0.999:
            return original_hw
        height = max(self.stage3_min_hw, int(original_hw[0] * factor))
        width = max(self.stage3_min_hw, int(original_hw[1] * factor))
        height = min(original_hw[0], max(32, height))
        width = min(original_hw[1], max(32, width))
        return (height, width)

    def _resize_tensor_to_hw(
        self,
        tensor: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        *,
        mode: str = "bilinear",
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.shape[-2:] == target_hw:
            return tensor
        align = None if mode == "nearest" else False
        original_shape = tensor.shape
        need_squeeze = False
        view_tensor = tensor
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)
            need_squeeze = True
        elif view_tensor.dim() > 4:
            view_tensor = view_tensor.view(-1, original_shape[-3], original_shape[-2], original_shape[-1])
        resized = F.interpolate(view_tensor, size=target_hw, mode=mode, align_corners=align)
        if view_tensor.dim() > 4:
            resized = resized.view(*original_shape[:-3], resized.shape[-3], *target_hw)
        if need_squeeze:
            resized = resized.squeeze(0)
        return resized

    def _scale_intrinsics_tensor(
        self,
        intrinsics: Optional[torch.Tensor],
        source_hw: Tuple[int, int],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if intrinsics is None or source_hw == target_hw:
            return intrinsics
        scaled_intrinsics = intrinsics.clone()
        scale_y = float(target_hw[0]) / max(float(source_hw[0]), 1.0)
        scale_x = float(target_hw[1]) / max(float(source_hw[1]), 1.0)
        scaled_intrinsics[..., 0, 0] = scaled_intrinsics[..., 0, 0] * scale_x
        scaled_intrinsics[..., 1, 1] = scaled_intrinsics[..., 1, 1] * scale_y
        scaled_intrinsics[..., 0, 2] = scaled_intrinsics[..., 0, 2] * scale_x
        scaled_intrinsics[..., 1, 2] = scaled_intrinsics[..., 1, 2] * scale_y
        return scaled_intrinsics

    def _resize_batch_to_hw(
        self,
        batch: Dict[str, torch.Tensor],
        target_hw: Tuple[int, int],
        *,
        recompute_latent: bool,
    ) -> Dict[str, torch.Tensor]:
        resized = dict(batch)
        source_hw = batch["target_rgb"].shape[-2:]
        current_hw = target_hw
        resized["target_rgb"] = self._resize_tensor_to_hw(batch["target_rgb"], target_hw, mode="bilinear")
        resized["rgb_render"] = self._resize_tensor_to_hw(batch["rgb_render"], target_hw, mode="bilinear")
        if batch.get("support_anchor_rgb") is not None:
            resized["support_anchor_rgb"] = self._resize_tensor_to_hw(
                batch["support_anchor_rgb"],
                target_hw,
                mode="bilinear",
            )
        resized["depth_render"] = self._resize_tensor_to_hw(
            batch.get("depth_render"),
            target_hw,
            mode="nearest",
        )
        if batch.get("geometry_depth_render") is not None:
            resized["geometry_depth_render"] = self._resize_tensor_to_hw(
                batch["geometry_depth_render"],
                target_hw,
                mode="nearest",
            )
        if batch.get("reference_depth_map") is not None:
            resized["reference_depth_map"] = self._resize_tensor_to_hw(
                batch["reference_depth_map"],
                target_hw,
                mode="nearest",
            )
        if batch.get("warp_valid_mask") is not None:
            resized["warp_valid_mask"] = self._resize_tensor_to_hw(
                batch["warp_valid_mask"],
                target_hw,
                mode="nearest",
            )
        if batch.get("warp_support_confidence") is not None:
            resized["warp_support_confidence"] = self._resize_tensor_to_hw(
                batch["warp_support_confidence"],
                target_hw,
                mode="bilinear",
            )
        if batch.get("supported_exist_mask") is not None:
            resized["supported_exist_mask"] = self._resize_tensor_to_hw(
                batch["supported_exist_mask"],
                target_hw,
                mode="nearest",
            )
        if batch.get("normal_render") is not None:
            resized["normal_render"] = self._resize_tensor_to_hw(
                batch["normal_render"],
                target_hw,
                mode="bilinear",
            )
        if batch.get("sparse_images") is not None:
            resized["sparse_images"] = self._resize_tensor_to_hw(
                batch["sparse_images"],
                target_hw,
                mode="bilinear",
            )
            resized["sparse_images"] = self._ensure_sparse_images_shape(
                resized["sparse_images"],
                batch.get("sparse_poses"),
            )
        if batch.get("sparse_depth_maps") is not None:
            resized["sparse_depth_maps"] = self._resize_tensor_to_hw(
                batch["sparse_depth_maps"],
                target_hw,
                mode="nearest",
            )
        if batch.get("target_intrinsics") is not None:
            resized["target_intrinsics"] = self._scale_intrinsics_tensor(
                batch["target_intrinsics"],
                source_hw,
                target_hw,
            )
        if batch.get("reference_intrinsics") is not None:
            resized["reference_intrinsics"] = self._scale_intrinsics_tensor(
                batch["reference_intrinsics"],
                source_hw,
                target_hw,
            )
        if batch.get("sparse_intrinsics") is not None:
            resized["sparse_intrinsics"] = self._scale_intrinsics_tensor(
                batch["sparse_intrinsics"],
                source_hw,
                target_hw,
            )
        if recompute_latent and batch.get("target_rgb") is not None:
            with torch.no_grad():
                # use raw_model for encoding/decoding
                resized_latent = self.raw_model.encode_rgb_to_latent(resized["target_rgb"])
                decoded_rgb = self.raw_model.decode_latent_to_rgb(resized_latent).clamp(0.0, 1.0)
            resized["target_latent"] = resized_latent
            resized["target_rgb"] = decoded_rgb
            decoded_hw = decoded_rgb.shape[-2:]
            if decoded_hw != current_hw:
                resized["rgb_render"] = self._resize_tensor_to_hw(resized["rgb_render"], decoded_hw, mode="bilinear")
                if resized.get("support_anchor_rgb") is not None:
                    resized["support_anchor_rgb"] = self._resize_tensor_to_hw(
                        resized["support_anchor_rgb"],
                        decoded_hw,
                        mode="bilinear",
                    )
                resized["depth_render"] = self._resize_tensor_to_hw(
                    resized.get("depth_render"),
                    decoded_hw,
                    mode="nearest",
                )
                if resized.get("geometry_depth_render") is not None:
                    resized["geometry_depth_render"] = self._resize_tensor_to_hw(
                        resized["geometry_depth_render"],
                        decoded_hw,
                        mode="nearest",
                    )
                if resized.get("normal_render") is not None:
                    resized["normal_render"] = self._resize_tensor_to_hw(
                        resized["normal_render"],
                        decoded_hw,
                        mode="bilinear",
                    )
                if resized.get("sparse_images") is not None:
                    resized["sparse_images"] = self._resize_tensor_to_hw(
                        resized["sparse_images"],
                        decoded_hw,
                        mode="bilinear",
                    )
                    resized["sparse_images"] = self._ensure_sparse_images_shape(
                        resized["sparse_images"],
                        batch.get("sparse_poses"),
                    )
                if resized.get("sparse_depth_maps") is not None:
                    resized["sparse_depth_maps"] = self._resize_tensor_to_hw(
                        resized["sparse_depth_maps"],
                        decoded_hw,
                        mode="nearest",
                    )
                if resized.get("reference_depth_map") is not None:
                    resized["reference_depth_map"] = self._resize_tensor_to_hw(
                        resized["reference_depth_map"],
                        decoded_hw,
                        mode="nearest",
                    )
                if resized.get("warp_valid_mask") is not None:
                    resized["warp_valid_mask"] = self._resize_tensor_to_hw(
                        resized["warp_valid_mask"],
                        decoded_hw,
                        mode="nearest",
                    )
                if resized.get("warp_support_confidence") is not None:
                    resized["warp_support_confidence"] = self._resize_tensor_to_hw(
                        resized["warp_support_confidence"],
                        decoded_hw,
                        mode="bilinear",
                    )
                if resized.get("supported_exist_mask") is not None:
                    resized["supported_exist_mask"] = self._resize_tensor_to_hw(
                        resized["supported_exist_mask"],
                        decoded_hw,
                        mode="nearest",
                    )
                if resized.get("target_intrinsics") is not None:
                    resized["target_intrinsics"] = self._scale_intrinsics_tensor(
                        resized["target_intrinsics"],
                        current_hw,
                        decoded_hw,
                    )
                if resized.get("reference_intrinsics") is not None:
                    resized["reference_intrinsics"] = self._scale_intrinsics_tensor(
                        resized["reference_intrinsics"],
                        current_hw,
                        decoded_hw,
                    )
                if resized.get("sparse_intrinsics") is not None:
                    resized["sparse_intrinsics"] = self._scale_intrinsics_tensor(
                        resized["sparse_intrinsics"],
                        current_hw,
                        decoded_hw,
                    )
                current_hw = decoded_hw
        return resized

    def _maybe_downscale_batch(
        self,
        batch: Dict[str, torch.Tensor],
        factor: float,
        *,
        recompute_latent: bool,
    ) -> Tuple[Dict[str, torch.Tensor], Tuple[int, int]]:
        original_hw = batch["target_rgb"].shape[-2:]
        target_hw = self._compute_target_hw(original_hw, factor)
        if target_hw == original_hw:
            normalized = dict(batch)
            if normalized.get("sparse_images") is not None:
                normalized["sparse_images"] = self._ensure_sparse_images_shape(
                    normalized["sparse_images"],
                    normalized.get("sparse_poses"),
                )
            return normalized, original_hw
        resized = self._resize_batch_to_hw(batch, target_hw, recompute_latent=recompute_latent)
        actual_hw = resized["target_rgb"].shape[-2:]
        return resized, actual_hw

    def _detach_parameter_grads(self) -> None:
        for param in self.model.parameters():
            if param.grad is not None and param.grad.requires_grad:
                param.grad = param.grad.detach()

    def _build_target_latent_cache_keys(
        self,
        batch: Dict[str, Any],
    ) -> List[Optional[Tuple[str, ...]]]:
        target_rgb = batch.get("target_rgb")
        if not isinstance(target_rgb, torch.Tensor) or target_rgb.dim() < 4:
            return []
        batch_size = int(target_rgb.shape[0])

        def _extract_batch_values(batch_key: str) -> List[Optional[str]]:
            batch_value = batch.get(batch_key)
            if batch_value is None:
                return [None] * batch_size
            if isinstance(batch_value, torch.Tensor):
                flattened_values = batch_value.detach().view(-1).cpu().tolist()
                if len(flattened_values) == 1 and batch_size > 1:
                    flattened_values = flattened_values * batch_size
                return [
                    None if raw_value is None else str(int(raw_value))
                    for raw_value in flattened_values[:batch_size]
                ] + [None] * max(batch_size - len(flattened_values), 0)
            if isinstance(batch_value, (list, tuple)):
                normalized_values = [None if raw_value is None else str(raw_value) for raw_value in batch_value]
                return normalized_values[:batch_size] + [None] * max(batch_size - len(normalized_values), 0)
            return [str(batch_value)] * batch_size

        scene_name_values = _extract_batch_values("scene_name")
        target_image_index_values = _extract_batch_values("target_image_index")
        scene_index_values = _extract_batch_values("scene_index")
        scene_sample_index_values = _extract_batch_values("scene_sample_index")

        cache_keys: List[Optional[Tuple[str, ...]]] = []
        for sample_index in range(batch_size):
            scene_name_value = scene_name_values[sample_index]
            target_image_index_value = target_image_index_values[sample_index]
            scene_index_value = scene_index_values[sample_index]
            scene_sample_index_value = scene_sample_index_values[sample_index]
            if scene_name_value is not None and target_image_index_value is not None:
                cache_keys.append(("scene_target", scene_name_value, target_image_index_value))
            elif scene_index_value is not None and target_image_index_value is not None:
                cache_keys.append(("scene_index_target", scene_index_value, target_image_index_value))
            elif target_image_index_value is not None:
                cache_keys.append(("target", target_image_index_value))
            elif scene_index_value is not None and scene_sample_index_value is not None:
                cache_keys.append(("scene_sample", scene_index_value, scene_sample_index_value))
            elif scene_sample_index_value is not None:
                cache_keys.append(("sample", scene_sample_index_value))
            else:
                cache_keys.append(None)
        return cache_keys

    def _ensure_batch_target_latents(
        self,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        target_rgb = batch.get("target_rgb")
        if not isinstance(target_rgb, torch.Tensor):
            raise KeyError("batch['target_rgb'] is required to build target latents.")

        target_latent = batch.get("target_latent")
        if not isinstance(target_latent, torch.Tensor):
            latent_height = max(1, int(target_rgb.shape[-2]) // 8)
            latent_width = max(1, int(target_rgb.shape[-1]) // 8)
            target_latent = torch.zeros(
                target_rgb.shape[0],
                4,
                latent_height,
                latent_width,
                device=target_rgb.device,
                dtype=target_rgb.dtype,
            )
            batch["target_latent"] = target_latent

        latent_needs_encode_mask = (
            target_latent.reshape(target_latent.shape[0], -1).abs().sum(dim=1) < 1.0e-6
        )
        if not bool(latent_needs_encode_mask.any().item()):
            return target_latent

        target_latent_cache_keys = self._build_target_latent_cache_keys(batch)
        for sample_index, cache_key in enumerate(target_latent_cache_keys):
            if cache_key is None or not bool(latent_needs_encode_mask[sample_index].item()):
                continue
            cached_target_latent = self._target_latent_cache.get(cache_key)
            if cached_target_latent is None:
                continue
            target_latent[sample_index : sample_index + 1] = cached_target_latent.to(
                device=target_latent.device,
                dtype=target_latent.dtype,
            )
            latent_needs_encode_mask[sample_index] = False

        if bool(latent_needs_encode_mask.any().item()):
            encode_sample_indices = torch.nonzero(
                latent_needs_encode_mask,
                as_tuple=False,
            ).flatten()
            target_rgb_to_encode = target_rgb.index_select(0, encode_sample_indices)
            with torch.no_grad():
                if hasattr(self.raw_model, "encode_rgb_to_latent"):
                    encoded_target_latents = self.raw_model.encode_rgb_to_latent(target_rgb_to_encode)
                elif hasattr(self.raw_model, "_encode_vae"):
                    encoded_target_latents = self.raw_model._encode_vae(target_rgb_to_encode)
                else:
                    raise AttributeError("raw_model 缺少 encode_rgb_to_latent/_encode_vae，无法构造 target_latent。")
            for local_encode_index, sample_index_tensor in enumerate(encode_sample_indices):
                sample_index = int(sample_index_tensor.item())
                encoded_target_latent = encoded_target_latents[
                    local_encode_index : local_encode_index + 1
                ]
                target_latent[sample_index : sample_index + 1] = encoded_target_latent
                cache_key = (
                    target_latent_cache_keys[sample_index]
                    if sample_index < len(target_latent_cache_keys)
                    else None
                )
                if cache_key is not None:
                    self._target_latent_cache[cache_key] = (
                        encoded_target_latent.detach().cpu().float().contiguous()
                    )

        if self._is_main_rank and not self._target_latent_cache_logged:
            print("[Trainer] target_latent 已启用跨 batch 复用缓存。")
            self._target_latent_cache_logged = True
        batch["target_latent"] = target_latent
        return target_latent

    def _update_early_stop_state(
        self,
        *,
        stage_name: str,
        metric_name: str,
        current_metric: Optional[float],
        best_metric: Optional[float],
        plateau_count: int,
        epoch: int,
    ) -> Tuple[Optional[float], int, bool]:
        patience = max(0, int(self.early_stop_patience))
        if patience <= 0 or current_metric is None:
            return best_metric, plateau_count, False

        min_epochs = max(0, int(self.early_stop_min_epochs))
        min_delta = max(0.0, float(self.early_stop_min_delta))
        if best_metric is None or current_metric > (best_metric + min_delta):
            return float(current_metric), 0, False
        if (epoch + 1) < min_epochs:
            return best_metric, plateau_count, False

        updated_plateau_count = int(plateau_count) + 1
        should_stop = updated_plateau_count >= patience
        if self._is_main_rank:
            print(
                f"[{stage_name}] {metric_name} 未提升: current={float(current_metric):.4f}, "
                f"best={float(best_metric):.4f}, plateau={updated_plateau_count}/{patience}"
            )
            if should_stop:
                print(
                    f"[{stage_name}] 触发早停: {metric_name} 已连续 {updated_plateau_count} 次无显著提升。"
                )
        return best_metric, updated_plateau_count, should_stop

    def _sample_diffusion_inputs(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target_latent = self._ensure_batch_target_latents(batch)
        
        # 优先使用noise_scheduler（用于训练，支持任意timestep）
        scheduler = self._resolve_training_noise_scheduler()
        if scheduler is None:
            sched = getattr(self.raw_model, "scheduler", None)
            # EulerDiscreteScheduler不支持任意timestep，需要创建DDPMScheduler
            if sched is not None and "Euler" in type(sched).__name__:
                from diffusers import DDPMScheduler
                scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    clip_sample=False,
                )
            else:
                scheduler = sched
        
        noise = torch.randn_like(target_latent)
        if scheduler is not None and hasattr(scheduler, "add_noise"):
            num_train_steps = int(getattr(scheduler.config, "num_train_timesteps", 1000))
            timesteps = torch.randint(
                0,
                num_train_steps,
                (target_latent.shape[0],),
                device=self.device,
                dtype=torch.long,
            )
            noisy_latents = scheduler.add_noise(target_latent, noise, timesteps)
        else:
            timesteps = torch.randint(
                0,
                1000,
                (target_latent.shape[0],),
                device=self.device,
                dtype=torch.long,
            )
            noisy_latents = self._add_noise(target_latent, noise)
        diffusion_supervision_target = self._compute_diffusion_supervision_target(
            target_latent=target_latent,
            sampled_noise=noise,
            timesteps=timesteps,
            scheduler=scheduler,
        )
        return timesteps, diffusion_supervision_target, noisy_latents

    def _generate_sample(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: Optional[int] = None,
        sampler_type: Optional[str] = None,
        return_diagnostics: bool = False,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """使用当前模型生成目标视角。"""

        previous_training_state = self.model.training
        self.model.eval()
        steps_to_use = steps or self.stage3_diffusion_steps
        sampler = sampler_type or self.sampler_type
        
        # 确保输入张量是FP32（与模型权重匹配）
        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.float16:
                batch[key] = batch[key].float()
        try:
            with torch.no_grad():
                phase1_episode_conditioned = batch.get("phase1_episode_conditioned")
                sample_output_mode = self._resolve_sample_output_mode(batch)
                # Use raw_model for custom method
                rgb = self.raw_model.generate_view_correctly(
                    sparse_images=batch["sparse_images"],
                    sparse_depth_maps=batch.get("sparse_depth_maps"),
                    sparse_poses=batch["sparse_poses"],
                    target_poses=batch["target_pose"],
                    rgb_render=batch["rgb_render"],
                    support_anchor_rgb=self._get_support_anchor_rgb_from_batch(batch),
                    depth_render=batch.get("depth_render", torch.zeros_like(batch["rgb_render"][:, :1])),
                    normal_render=batch.get("normal_render", torch.zeros_like(batch["rgb_render"])),
                    geometry_depth_render=batch.get("geometry_depth_render"),
                    reference_depth_map=batch.get("reference_depth_map"),
                    reference_intrinsics=batch.get("reference_intrinsics"),
                    camera_intrinsics=batch.get("target_intrinsics"),
                    sparse_intrinsics=batch.get("sparse_intrinsics"),
                    support_confidence=batch.get("warp_support_confidence"),
                    warp_valid_mask=batch.get("warp_valid_mask"),
                    frontier_mask=batch.get("frontier_mask"),
                    verification_prior=batch.get("verification_prior"),
                    accum_render=batch.get("accum_render"),
                    transmittance_render=batch.get("transmittance_render"),
                    use_3dgs_residual_refiner=batch.get("use_3dgs_residual_refiner"),
                    episode_conditioned=phase1_episode_conditioned,
                    phase1_condition_source_tag=batch.get("phase1_condition_source_tag"),
                    guidance_scale=self.sample_guidance_scale,
                    diagnostic_disable_latent_blending=self.sample_disable_latent_blending,
                    diagnostic_inpaint_img2img_strength_override=self.sample_inpaint_img2img_strength_override,
                    diagnostic_masked_dual_path_support_step_offset_override=(
                        self.sample_masked_dual_path_support_step_offset_override
                    ),
                    steps=steps_to_use,
                    # 为避免与 VAE 解码尺寸不一致，使用 target_rgb 的空间尺寸作为目标分辨率
                    resolution=batch["target_rgb"].shape[-2:],
                    enable_mc_dropout=False,
                    device=self.device,
                    sampler_type=sampler,
                    scheduler_overrides=self.scheduler_overrides,
                    scheduler_eta=self.scheduler_eta,
                    return_diagnostics=return_diagnostics,
                    output_mode=sample_output_mode,
                )
            return rgb
        finally:
            if previous_training_state:
                self.model.train()
            else:
                self.model.eval()

    def _generate_trainable_sample(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: Optional[int] = None,
        sampler_type: Optional[str] = None,
        return_diagnostics: bool = False,
        enable_mc_dropout: bool = False,
        mc_dropout_p: Optional[float] = None,
    ) -> Optional[Union[torch.Tensor, Dict[str, Any]]]:
        """绕过 no_grad 装饰器，直接对 sample-path 建立可微监督。"""
        previous_training_state = self.model.training
        self.model.eval()
        steps_to_use = steps or self.stage3_diffusion_steps
        sampler = sampler_type or self.sampler_type

        for key in batch:
            if isinstance(batch[key], torch.Tensor) and batch[key].dtype == torch.float16:
                batch[key] = batch[key].float()

        bound_generate_function = getattr(self.raw_model, "generate_view_correctly", None)
        undecorated_generate_function = getattr(bound_generate_function, "__wrapped__", None)
        if undecorated_generate_function is None:
            class_generate_function = getattr(type(self.raw_model), "generate_view_correctly", None)
            undecorated_generate_function = getattr(class_generate_function, "__wrapped__", None)
        if undecorated_generate_function is None:
            if self._is_main_rank and not getattr(self, "_trainable_sample_supervision_warned", False):
                print("[Trainer] 未找到可微 generate_view_correctly 原函数，跳过 sample-path 直接监督。")
                self._trainable_sample_supervision_warned = True
            if previous_training_state:
                self.model.train()
            else:
                self.model.eval()
            return None

        try:
            phase1_episode_conditioned = batch.get("phase1_episode_conditioned")
            sample_output_mode = self._resolve_sample_output_mode(batch)
            return undecorated_generate_function(
                self.raw_model,
                sparse_images=batch["sparse_images"],
                sparse_depth_maps=batch.get("sparse_depth_maps"),
                sparse_poses=batch["sparse_poses"],
                target_poses=batch["target_pose"],
                rgb_render=batch["rgb_render"],
                support_anchor_rgb=self._get_support_anchor_rgb_from_batch(batch),
                depth_render=batch.get("depth_render", torch.zeros_like(batch["rgb_render"][:, :1])),
                normal_render=batch.get("normal_render", torch.zeros_like(batch["rgb_render"])),
                geometry_depth_render=batch.get("geometry_depth_render"),
                reference_depth_map=batch.get("reference_depth_map"),
                reference_intrinsics=batch.get("reference_intrinsics"),
                camera_intrinsics=batch.get("target_intrinsics"),
                sparse_intrinsics=batch.get("sparse_intrinsics"),
                support_confidence=batch.get("warp_support_confidence"),
                warp_valid_mask=batch.get("warp_valid_mask"),
                frontier_mask=batch.get("frontier_mask"),
                verification_prior=batch.get("verification_prior"),
                accum_render=batch.get("accum_render"),
                transmittance_render=batch.get("transmittance_render"),
                use_3dgs_residual_refiner=batch.get("use_3dgs_residual_refiner"),
                episode_conditioned=phase1_episode_conditioned,
                phase1_condition_source_tag=batch.get("phase1_condition_source_tag"),
                guidance_scale=self.sample_guidance_scale,
                diagnostic_disable_latent_blending=self.sample_disable_latent_blending,
                diagnostic_inpaint_img2img_strength_override=self.sample_inpaint_img2img_strength_override,
                diagnostic_masked_dual_path_support_step_offset_override=(
                    self.sample_masked_dual_path_support_step_offset_override
                ),
                steps=steps_to_use,
                resolution=batch["target_rgb"].shape[-2:],
                enable_mc_dropout=bool(enable_mc_dropout),
                mc_dropout_p=float(
                    mc_dropout_p
                    if mc_dropout_p is not None
                    else getattr(self, "proposal_empirical_candidate_mc_dropout_p", 0.15)
                ),
                device=self.device,
                sampler_type=sampler,
                scheduler_overrides=self.scheduler_overrides,
                scheduler_eta=self.scheduler_eta,
                return_diagnostics=return_diagnostics,
                output_mode=sample_output_mode,
                track_grad_decode=True,
            )
        finally:
            if previous_training_state:
                self.model.train()
            else:
                self.model.eval()

    def _extract_proposal_candidate_tensors(
        self,
        generated_output: Optional[Union[torch.Tensor, Dict[str, Any]]],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if generated_output is None:
            return None, None
        if isinstance(generated_output, dict):
            candidate_rgb = generated_output.get("proposal_raw_rgb")
            if candidate_rgb is None:
                candidate_rgb = generated_output.get("rgb", generated_output.get("image"))
            candidate_residual = generated_output.get("proposal_residual_rgb")
            return (
                candidate_rgb if isinstance(candidate_rgb, torch.Tensor) else None,
                candidate_residual if isinstance(candidate_residual, torch.Tensor) else None,
            )
        if isinstance(generated_output, torch.Tensor):
            return generated_output, None
        return None, None

    def _build_empirical_repairability_candidates(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: Optional[int],
        sampler_type: Optional[str] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        candidate_count = max(
            int(getattr(self, "proposal_empirical_candidate_count", 1)),
            1,
        )
        if candidate_count <= 1:
            return {
                "candidate_rgbs": [],
                "candidate_residuals": [],
                "candidate_rgb_sources": [],
                "candidate_residual_sources": [],
            }
        candidate_rgbs: List[torch.Tensor] = []
        candidate_residuals: List[torch.Tensor] = []
        candidate_rgb_sources: List[str] = []
        candidate_residual_sources: List[str] = []
        dropout_probability = max(
            float(getattr(self, "proposal_empirical_candidate_mc_dropout_p", 0.15)),
            0.0,
        )

        def _batch_is_episode_conditioned() -> bool:
            episode_conditioned_flag = batch.get("phase1_episode_conditioned")
            if isinstance(episode_conditioned_flag, torch.Tensor):
                return bool(
                    torch.nan_to_num(
                        episode_conditioned_flag.detach().float(),
                        nan=0.0,
                    ).mean().item()
                    > 0.5
                )
            return bool(episode_conditioned_flag)

        def _reference_hw() -> Optional[Tuple[int, int]]:
            target_rgb_tensor = batch.get("target_rgb")
            if isinstance(target_rgb_tensor, torch.Tensor):
                return tuple(target_rgb_tensor.shape[-2:])
            render_rgb_tensor = batch.get("rgb_render")
            if isinstance(render_rgb_tensor, torch.Tensor):
                return tuple(render_rgb_tensor.shape[-2:])
            return None

        def _prepare_rgb_candidate(candidate_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if not isinstance(candidate_tensor, torch.Tensor):
                return None
            prepared_candidate = candidate_tensor
            if prepared_candidate.dim() == 3:
                prepared_candidate = prepared_candidate.unsqueeze(0)
            if prepared_candidate.shape[1] > 3:
                prepared_candidate = prepared_candidate[:, :3]
            prepared_candidate = torch.nan_to_num(
                prepared_candidate,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            target_hw = _reference_hw()
            if target_hw is not None and prepared_candidate.shape[-2:] != target_hw:
                prepared_candidate = F.interpolate(
                    prepared_candidate,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_candidate

        def _prepare_mask_candidate(mask_tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if not isinstance(mask_tensor, torch.Tensor):
                return None
            prepared_mask = mask_tensor
            if prepared_mask.dim() == 3:
                prepared_mask = prepared_mask.unsqueeze(1)
            if prepared_mask.shape[1] > 1:
                prepared_mask = prepared_mask[:, :1]
            prepared_mask = torch.nan_to_num(
                prepared_mask,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            target_hw = _reference_hw()
            if target_hw is not None and prepared_mask.shape[-2:] != target_hw:
                prepared_mask = F.interpolate(
                    prepared_mask,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_mask

        def _append_rgb_candidate(candidate_tensor: Optional[torch.Tensor], source_name: str) -> None:
            prepared_candidate = _prepare_rgb_candidate(candidate_tensor)
            if not isinstance(prepared_candidate, torch.Tensor):
                return
            candidate_rgbs.append(prepared_candidate.detach())
            candidate_rgb_sources.append(source_name)

        def _append_residual_candidate(candidate_tensor: Optional[torch.Tensor], source_name: str) -> None:
            if not isinstance(candidate_tensor, torch.Tensor):
                return
            candidate_residuals.append(candidate_tensor.detach())
            candidate_residual_sources.append(source_name)

        def _masked_mean_std(
            value_tensor: torch.Tensor,
            mask_tensor: Optional[torch.Tensor],
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            if isinstance(mask_tensor, torch.Tensor) and float(mask_tensor.sum().item()) >= 1.0:
                aligned_mask = mask_tensor.to(device=value_tensor.device, dtype=value_tensor.dtype)
                if aligned_mask.shape[-2:] != value_tensor.shape[-2:]:
                    aligned_mask = F.interpolate(
                        aligned_mask,
                        size=value_tensor.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                if aligned_mask.shape[1] > 1:
                    aligned_mask = aligned_mask[:, :1]
                weight = aligned_mask.clamp_min(1e-6)
                normalizer = weight.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
                mean_tensor = (value_tensor * weight).sum(dim=(2, 3), keepdim=True) / normalizer
                variance_tensor = (
                    (value_tensor - mean_tensor).pow(2) * weight
                ).sum(dim=(2, 3), keepdim=True) / normalizer
            else:
                mean_tensor = value_tensor.mean(dim=(2, 3), keepdim=True)
                variance_tensor = value_tensor.var(dim=(2, 3), keepdim=True, unbiased=False)
            return mean_tensor, variance_tensor.clamp_min(1e-6).sqrt()

        def _append_tone_matched_candidate(
            candidate_tensor: Optional[torch.Tensor],
            reference_tensor: Optional[torch.Tensor],
            mask_tensor: Optional[torch.Tensor],
            source_name: str,
        ) -> None:
            prepared_candidate = _prepare_rgb_candidate(candidate_tensor)
            prepared_reference = _prepare_rgb_candidate(reference_tensor)
            if not isinstance(prepared_candidate, torch.Tensor) or not isinstance(
                prepared_reference,
                torch.Tensor,
            ):
                return
            prepared_candidate = prepared_candidate.to(
                device=prepared_reference.device,
                dtype=prepared_reference.dtype,
            )
            prepared_mask = _prepare_mask_candidate(mask_tensor)
            if isinstance(prepared_mask, torch.Tensor):
                prepared_mask = prepared_mask.to(
                    device=prepared_reference.device,
                    dtype=prepared_reference.dtype,
                )
            candidate_mean, candidate_std = _masked_mean_std(prepared_candidate, prepared_mask)
            reference_mean, reference_std = _masked_mean_std(prepared_reference, prepared_mask)
            tone_matched_candidate = torch.clamp(
                (prepared_candidate - candidate_mean) / candidate_std.clamp_min(1e-4)
                * reference_std
                + reference_mean,
                min=0.0,
                max=1.0,
            )
            _append_rgb_candidate(tone_matched_candidate, source_name)

        def _append_local_repair_candidates() -> None:
            if not bool(getattr(self, "proposal_empirical_include_local_repair_candidates", True)):
                return
            render_rgb = _prepare_rgb_candidate(batch.get("rgb_render"))
            if not isinstance(render_rgb, torch.Tensor):
                return
            support_confidence = _prepare_mask_candidate(batch.get("warp_support_confidence"))
            support_valid_mask = _prepare_mask_candidate(batch.get("warp_valid_mask"))
            if support_confidence is None:
                support_confidence = support_valid_mask
            elif support_valid_mask is not None:
                support_confidence = torch.clamp(
                    support_confidence
                    * support_valid_mask.to(
                        device=support_confidence.device,
                        dtype=support_confidence.dtype,
                    ),
                    min=0.0,
                    max=1.0,
                )
            support_anchor_rgb = None
            if _batch_is_episode_conditioned():
                support_anchor_rgb = _prepare_rgb_candidate(self._get_support_anchor_rgb_from_batch(batch))
            local_source_rgb = support_anchor_rgb if isinstance(support_anchor_rgb, torch.Tensor) else render_rgb
            if isinstance(support_anchor_rgb, torch.Tensor):
                _append_rgb_candidate(support_anchor_rgb, "episode_support_anchor")
                _append_rgb_candidate(
                    torch.clamp(0.75 * render_rgb + 0.25 * support_anchor_rgb, min=0.0, max=1.0),
                    "render_support_anchor_blend025",
                )
                _append_rgb_candidate(
                    torch.clamp(0.50 * render_rgb + 0.50 * support_anchor_rgb, min=0.0, max=1.0),
                    "render_support_anchor_blend050",
                )
            configured_kernel_sizes = getattr(
                self,
                "proposal_empirical_local_repair_kernel_sizes",
                (9, 21, 41),
            )
            if not isinstance(configured_kernel_sizes, (list, tuple)):
                configured_kernel_sizes = (9, 21, 41)
            repair_mask = (
                torch.clamp(1.0 - support_confidence, min=0.0, max=1.0)
                if isinstance(support_confidence, torch.Tensor)
                else torch.ones_like(render_rgb[:, :1])
            )
            for kernel_size_value in configured_kernel_sizes:
                kernel_size = max(int(kernel_size_value), 1)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                padding = kernel_size // 2
                smoothed_render = F.avg_pool2d(
                    render_rgb.float(),
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                ).to(device=render_rgb.device, dtype=render_rgb.dtype)
                _append_rgb_candidate(
                    torch.clamp(0.75 * render_rgb + 0.25 * smoothed_render, min=0.0, max=1.0),
                    f"render_local_smooth_k{kernel_size}",
                )
                if isinstance(support_confidence, torch.Tensor):
                    source_weight = support_confidence.to(
                        device=local_source_rgb.device,
                        dtype=local_source_rgb.dtype,
                    )
                    pooled_weight = F.avg_pool2d(
                        source_weight.float(),
                        kernel_size=kernel_size,
                        stride=1,
                        padding=padding,
                    ).clamp_min(1e-6)
                    pooled_source = (
                        F.avg_pool2d(
                            (local_source_rgb * source_weight).float(),
                            kernel_size=kernel_size,
                            stride=1,
                            padding=padding,
                        )
                        / pooled_weight
                    ).to(device=render_rgb.device, dtype=render_rgb.dtype)
                    aligned_repair_mask = repair_mask.to(
                        device=render_rgb.device,
                        dtype=render_rgb.dtype,
                    )
                    _append_rgb_candidate(
                        torch.clamp(
                            render_rgb * (1.0 - 0.50 * aligned_repair_mask)
                            + pooled_source * (0.50 * aligned_repair_mask),
                            min=0.0,
                            max=1.0,
                        ),
                        f"support_local_fill050_k{kernel_size}",
                    )
                    _append_rgb_candidate(
                        torch.clamp(
                            render_rgb * (1.0 - aligned_repair_mask)
                            + pooled_source * aligned_repair_mask,
                            min=0.0,
                            max=1.0,
                        ),
                        f"support_local_fill100_k{kernel_size}",
                    )

        model_attribute_overrides: Dict[str, Any] = {}
        if bool(getattr(self, "proposal_empirical_disable_hard_gate", True)):
            if hasattr(self.raw_model, "proposal_use_hard_acceptance_gate"):
                model_attribute_overrides["proposal_use_hard_acceptance_gate"] = False
            if hasattr(self.raw_model, "proposal_safe_confidence_threshold"):
                model_attribute_overrides["proposal_safe_confidence_threshold"] = 0.0
        saved_model_attributes: Dict[str, Any] = {}
        try:
            for attribute_name, override_value in model_attribute_overrides.items():
                saved_model_attributes[attribute_name] = getattr(self.raw_model, attribute_name)
                setattr(self.raw_model, attribute_name, override_value)
            for candidate_index in range(candidate_count - 1):
                candidate_dropout_probability = min(
                    dropout_probability * (1.0 + 0.5 * float(candidate_index)),
                    0.45,
                )
                with torch.no_grad():
                    generated_output = self._generate_trainable_sample(
                        batch,
                        steps=steps,
                        sampler_type=sampler_type,
                        return_diagnostics=True,
                        enable_mc_dropout=candidate_dropout_probability > 0.0,
                        mc_dropout_p=candidate_dropout_probability,
                    )
                candidate_rgb, candidate_residual = self._extract_proposal_candidate_tensors(
                    generated_output
                )
                if isinstance(candidate_residual, torch.Tensor):
                    _append_residual_candidate(
                        candidate_residual,
                        f"proposal_residual_mc{candidate_index}",
                    )
                if isinstance(candidate_rgb, torch.Tensor):
                    _append_rgb_candidate(
                        candidate_rgb,
                        f"proposal_rgb_mc{candidate_index}",
                    )
                if isinstance(generated_output, dict):
                    debug_tensors = generated_output.get("debug_tensors")
                    if isinstance(debug_tensors, dict):
                        debug_proposal_residual = debug_tensors.get("proposal_residual_rgb")
                        if isinstance(debug_proposal_residual, torch.Tensor):
                            _append_residual_candidate(
                                debug_proposal_residual,
                                f"debug_residual_mc{candidate_index}",
                            )
                        if bool(getattr(self, "proposal_empirical_include_decoded_teacher", True)):
                            decoded_teacher_rgb = debug_tensors.get("decoded_teacher_rgb")
                            if isinstance(decoded_teacher_rgb, torch.Tensor):
                                _append_rgb_candidate(
                                    decoded_teacher_rgb,
                                    f"decoded_teacher_mc{candidate_index}",
                                )
                                _append_tone_matched_candidate(
                                    decoded_teacher_rgb,
                                    batch.get("rgb_render"),
                                    batch.get("warp_support_confidence"),
                                    f"decoded_teacher_tonematch_render_mc{candidate_index}",
                                )
            _append_local_repair_candidates()
        finally:
            for attribute_name, saved_value in saved_model_attributes.items():
                setattr(self.raw_model, attribute_name, saved_value)
        return {
            "candidate_rgbs": candidate_rgbs,
            "candidate_residuals": candidate_residuals,
            "candidate_rgb_sources": candidate_rgb_sources,
            "candidate_residual_sources": candidate_residual_sources,
        }

    def _add_noise(
        self, latents: torch.Tensor, noise: torch.Tensor, alpha: float = 0.99
    ) -> torch.Tensor:
        sqrt_alpha = torch.sqrt(torch.tensor(alpha, device=latents.device))
        sqrt_one_minus_alpha = torch.sqrt(torch.tensor(1.0 - alpha, device=latents.device))
        return sqrt_alpha * latents + sqrt_one_minus_alpha * noise

    def _decode_predicted_x0(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        alpha_fallback: float = 0.99,
        track_grad: bool = False,
    ) -> torch.Tensor:
        scheduler = self._resolve_training_noise_scheduler()
        target_device = noise_pred.device
        x0 = self._predict_x0_from_model_output(
            noisy_latents=noisy_latents.to(target_device),
            model_output=noise_pred.to(target_device),
            timesteps=timesteps.to(target_device),
            scheduler=scheduler,
            alpha_fallback=alpha_fallback,
        )
        # Use raw_model for decoding
        try:
            rgb = self.raw_model.decode_latent_to_rgb(x0, track_grad=track_grad)
        except TypeError:
            rgb = self.raw_model.decode_latent_to_rgb(x0)
        return rgb

    def _compute_direct_sample_path_reconstruction_loss(
        self,
        *,
        sample_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        base_rgb: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        depth_render: Optional[torch.Tensor],
        recon_alpha: float,
    ) -> torch.Tensor:
        target_hw = target_rgb.shape[-2:]
        if sample_rgb.shape[-2:] != target_hw:
            sample_rgb = F.interpolate(
                sample_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        editable_region_mask = self._build_low_confidence_editable_mask(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        exist_mask, novel_mask = self._build_projection_aligned_exist_novel_masks(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        if novel_mask is None:
            novel_mask = editable_region_mask
        elif editable_region_mask is not None:
            novel_mask = torch.clamp(
                novel_mask.to(device=sample_rgb.device, dtype=sample_rgb.dtype)
                + editable_region_mask.to(device=sample_rgb.device, dtype=sample_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        if novel_mask is not None:
            exist_mask = torch.clamp(
                1.0 - novel_mask.to(device=sample_rgb.device, dtype=sample_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        return self._region_aware_reconstruction_loss(
            sample_rgb,
            target_rgb,
            base_rgb,
            exist_mask=exist_mask,
            novel_mask=novel_mask,
            recon_alpha=recon_alpha,
        )

    def _to_device(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # 获取模型的dtype，用于确保输入数据类型匹配
        model_dtype = getattr(self.raw_model, "_model_dtype", None)

        moved = {}
        for key, tensor in batch.items():
            if isinstance(tensor, torch.Tensor):
                # 移动到目标设备
                tensor = tensor.to(self.device)
                # 如果是浮点型张量且模型指定了dtype，则转换dtype
                if model_dtype is not None and tensor.dtype in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
                    tensor = tensor.to(dtype=model_dtype)
                moved[key] = tensor
            else:
                moved[key] = tensor

        if moved.get("sparse_images") is not None:
            moved["sparse_images"] = self._ensure_sparse_images_shape(
                moved["sparse_images"],
                moved.get("sparse_poses"),
            )
        return moved

    def _clone_batch_for_visualization(
        self,
        batch: Dict[str, Any],
    ) -> Dict[str, Any]:
        cloned_batch: Dict[str, Any] = {}
        for batch_key, batch_value in batch.items():
            if isinstance(batch_value, torch.Tensor):
                cloned_batch[batch_key] = batch_value.detach().cpu().clone()
            else:
                cloned_batch[batch_key] = batch_value
        return cloned_batch

    def _build_visualization_batch_from_sample(
        self,
        sample: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(sample, dict):
            return None
        visualization_batch: Dict[str, Any] = {}
        for sample_key, sample_value in sample.items():
            if isinstance(sample_value, torch.Tensor):
                visualization_batch[sample_key] = sample_value.unsqueeze(0)
            else:
                visualization_batch[sample_key] = sample_value
        return visualization_batch

    def _resolve_fixed_visualization_sample_indices(
        self,
        dataset_length: int,
    ) -> List[int]:
        requested_sample_indices = self.fixed_visualization_sample_indices
        if not requested_sample_indices:
            requested_sample_indices = [self.fixed_visualization_sample_index]
        resolved_sample_indices: List[int] = []
        seen_sample_indices = set()
        max_sample_index = max(int(dataset_length) - 1, 0)
        for requested_sample_index in requested_sample_indices:
            resolved_sample_index = min(max(int(requested_sample_index), 0), max_sample_index)
            if resolved_sample_index in seen_sample_indices:
                continue
            seen_sample_indices.add(resolved_sample_index)
            resolved_sample_indices.append(resolved_sample_index)
        if not resolved_sample_indices:
            resolved_sample_indices = [0]
        return resolved_sample_indices

    def _resolve_dataset_visualization_sample_indices(
        self,
        dataset: Any,
        dataset_length: int,
    ) -> List[int]:
        resolved_sample_indices = self._resolve_fixed_visualization_sample_indices(
            dataset_length
        )
        scene_datasets = getattr(dataset, "scene_datasets", None)
        scene_offsets = getattr(dataset, "scene_offsets", None)
        if (
            not isinstance(scene_datasets, Sequence)
            or not isinstance(scene_offsets, Sequence)
            or len(scene_datasets) <= 1
            or len(scene_offsets) != len(scene_datasets)
        ):
            return resolved_sample_indices
        remapped_sample_indices: List[int] = []
        seen_sample_indices = set()
        scene_count = len(scene_datasets)
        for requested_sample_index in resolved_sample_indices:
            scene_index = int(requested_sample_index) % scene_count
            local_rank = int(requested_sample_index) // scene_count
            scene_dataset = scene_datasets[scene_index]
            scene_length = len(scene_dataset) if hasattr(scene_dataset, "__len__") else 0
            if scene_length <= 0:
                continue
            local_index = min(max(local_rank, 0), scene_length - 1)
            remapped_global_index = int(scene_offsets[scene_index]) + int(local_index)
            if remapped_global_index in seen_sample_indices:
                continue
            seen_sample_indices.add(remapped_global_index)
            remapped_sample_indices.append(remapped_global_index)
        if remapped_sample_indices:
            return remapped_sample_indices
        return resolved_sample_indices

    def _get_visualization_batches_from_loader_dataset(
        self,
        dataloader: Optional[Iterable[Dict[str, torch.Tensor]]],
        *,
        dataset_name: str,
    ) -> List[Tuple[Dict[str, Any], str]]:
        if dataloader is None:
            return []
        dataset = getattr(dataloader, "dataset", None)
        visualization_batches: List[Tuple[Dict[str, Any], str]] = []
        if dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            dataset_length = len(dataset)
            if dataset_length > 0:
                for resolved_sample_index in self._resolve_dataset_visualization_sample_indices(
                    dataset,
                    dataset_length,
                ):
                    dataset_sample = dataset[resolved_sample_index]
                    visualization_batch = self._build_visualization_batch_from_sample(
                        dataset_sample
                    )
                    if visualization_batch is None:
                        continue
                    scene_name = dataset_sample.get("scene_name") if isinstance(dataset_sample, dict) else None
                    proposal_hardness_score = (
                        dataset_sample.get("episode_proposal_hardness_score")
                        if isinstance(dataset_sample, dict)
                        else None
                    )
                    novel_coverage = (
                        dataset_sample.get("episode_novel_coverage")
                        if isinstance(dataset_sample, dict)
                        else None
                    )
                    source_suffix_segments: List[str] = []
                    if isinstance(scene_name, str) and scene_name:
                        source_suffix_segments.append(f"scene={scene_name}")
                    if isinstance(proposal_hardness_score, torch.Tensor) and proposal_hardness_score.numel() > 0:
                        source_suffix_segments.append(
                            f"hardness={float(proposal_hardness_score.reshape(-1)[0].item()):.4f}"
                        )
                    if isinstance(novel_coverage, torch.Tensor) and novel_coverage.numel() > 0:
                        source_suffix_segments.append(
                            f"novel={float(novel_coverage.reshape(-1)[0].item()):.4f}"
                        )
                    source_suffix = ""
                    if source_suffix_segments:
                        source_suffix = "/" + "/".join(source_suffix_segments)
                    visualization_batches.append(
                        (
                            visualization_batch,
                            f"{dataset_name}_dataset[{resolved_sample_index}]{source_suffix}",
                        )
                    )
                if visualization_batches:
                    return visualization_batches
        try:
            batch = next(iter(dataloader))
        except StopIteration:
            return []
        return [(batch, f"{dataset_name}[0]")]

    def _get_fixed_visualization_batches(
        self,
        fallback_batch: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Dict[str, Any], str]]:
        if self._fixed_visualization_batches_cpu is None:
            candidate_batches_with_sources = self._get_visualization_batches_from_loader_dataset(
                self.val_dataloader,
                dataset_name="validation",
            )
            if not candidate_batches_with_sources:
                candidate_batches_with_sources = self._get_visualization_batches_from_loader_dataset(
                    self.train_dataloader,
                    dataset_name="train",
                )
            if not candidate_batches_with_sources and fallback_batch is not None:
                candidate_batches_with_sources = [(fallback_batch, "current_batch_fallback")]
            if candidate_batches_with_sources:
                self._fixed_visualization_batches_cpu = [
                    self._clone_batch_for_visualization(candidate_batch)
                    for candidate_batch, _ in candidate_batches_with_sources
                ]
                self._fixed_visualization_batch_sources = [
                    candidate_source for _, candidate_source in candidate_batches_with_sources
                ]
                self._fixed_visualization_batch_cpu = self._fixed_visualization_batches_cpu[0]
                self._fixed_visualization_batch_source = self._fixed_visualization_batch_sources[0]
                if self._is_main_rank:
                    print(
                        "[Trainer] 已固定可视化样本来源组: "
                        + ", ".join(self._fixed_visualization_batch_sources)
                    )
        if self._fixed_visualization_batches_cpu is None:
            return []
        resolved_sources = self._fixed_visualization_batch_sources or [
            "unknown"
        ] * len(self._fixed_visualization_batches_cpu)
        return [
            (
                self._to_device(self._clone_batch_for_visualization(visualization_batch)),
                visualization_source,
            )
            for visualization_batch, visualization_source in zip(
                self._fixed_visualization_batches_cpu,
                resolved_sources,
            )
        ]

    def _get_fixed_visualization_batch(
        self,
        fallback_batch: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        fixed_visualization_batches = self._get_fixed_visualization_batches(
            fallback_batch=fallback_batch
        )
        if not fixed_visualization_batches:
            return None
        return fixed_visualization_batches[0][0]

    def _ensure_sparse_images_shape(
        self,
        sparse_images: Optional[torch.Tensor],
        sparse_poses: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if sparse_images is None or sparse_poses is None:
            return sparse_images
        if sparse_images.dim() < 3 or sparse_poses.dim() < 3:
            return sparse_images
        expected_batch = int(sparse_poses.shape[0])
        expected_views = int(sparse_poses.shape[1])
        if expected_batch <= 0 or expected_views <= 0:
            return sparse_images

        def _expand_batch(tensor: torch.Tensor) -> torch.Tensor:
            if expected_batch == 1:
                return tensor.unsqueeze(0)
            expanded = tensor.unsqueeze(0).expand(expected_batch, *tensor.shape)
            return expanded.contiguous()

        if sparse_images.dim() == 5:
            if (
                sparse_images.shape[0] == expected_batch
                and sparse_images.shape[1] == expected_views
            ):
                return sparse_images
            leading = sparse_images.shape[0] * sparse_images.shape[1]
            if leading == expected_batch * expected_views:
                return sparse_images.reshape(
                    expected_batch,
                    expected_views,
                    *sparse_images.shape[2:],
                )
            return sparse_images

        if sparse_images.dim() == 4:
            C = sparse_images.shape[-3]
            H, W = sparse_images.shape[-2:]
            total = sparse_images.shape[0]
            if total == expected_views:
                tensor = _expand_batch(sparse_images)
                return tensor
            if total == expected_batch * expected_views:
                return sparse_images.reshape(expected_batch, expected_views, C, H, W)
            if total == expected_batch:
                tensor = sparse_images.unsqueeze(1)
                if expected_views > 1:
                    tensor = tensor.expand(-1, expected_views, -1, -1, -1).contiguous()
                return tensor

        if sparse_images.dim() == 3:
            tensor = sparse_images.unsqueeze(0).unsqueeze(0)
            tensor = tensor.expand(expected_batch, expected_views, -1, -1, -1).contiguous()
            return tensor

        return sparse_images

    def _build_depth_mask(
        self,
        depth: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if depth is None:
            return None
        mask = (depth > 1e-6).to(depth.dtype)
        # 如果 mask 全为 0（depth 是占位符），返回 None 不使用掩码
        if mask.sum() < 1:
            return None
        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask, size=target_hw, mode="nearest")
        return mask

    def _build_support_mask(
        self,
        support_mask: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if support_mask is None:
            return None
        if support_mask.dim() == 3:
            support_mask = support_mask.unsqueeze(1)
        if support_mask.shape[1] > 1:
            support_mask = support_mask[:, :1]
        mask = (support_mask > 0.5).to(support_mask.dtype)
        if mask.sum() < 1:
            return None
        if mask.shape[-2:] != target_hw:
            mask = F.interpolate(mask, size=target_hw, mode="nearest")
        return mask

    def _build_sampler_aligned_support_signals(
        self,
        *,
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Dict[str, Optional[torch.Tensor]]:
        base_support_mask = self._build_support_mask(support_mask, target_hw)
        if base_support_mask is None:
            return {
                "warp_valid_mask": None,
                "support_confidence": None,
                "inpaint_mask": None,
                "support_projection_mask": None,
                "editable_low_confidence_mask": None,
            }

        aligned_support_confidence = support_confidence
        if aligned_support_confidence is None:
            aligned_support_confidence = base_support_mask.float()
        if aligned_support_confidence.dim() == 3:
            aligned_support_confidence = aligned_support_confidence.unsqueeze(1)
        if aligned_support_confidence.shape[1] > 1:
            aligned_support_confidence = aligned_support_confidence[:, :1]
        if aligned_support_confidence.shape[-2:] != target_hw:
            aligned_support_confidence = F.interpolate(
                aligned_support_confidence.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_support_confidence = torch.nan_to_num(
            aligned_support_confidence.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        refine_support_function = getattr(
            self.raw_model,
            "_refine_support_masks_for_sampling",
            None,
        )
        if callable(refine_support_function):
            refined_support_outputs = refine_support_function(
                warp_valid_mask=base_support_mask.float(),
                support_confidence=aligned_support_confidence,
                raw_support_confidence=aligned_support_confidence,
                inpaint_mask=torch.clamp(1.0 - base_support_mask.float(), min=0.0, max=1.0),
            )
            return {
                "warp_valid_mask": refined_support_outputs.get("warp_valid_mask"),
                "support_confidence": refined_support_outputs.get("support_confidence"),
                "inpaint_mask": refined_support_outputs.get("inpaint_mask"),
                "support_projection_mask": refined_support_outputs.get("support_projection_mask"),
                "editable_low_confidence_mask": refined_support_outputs.get("editable_low_confidence_mask"),
            }

        return {
            "warp_valid_mask": base_support_mask.float(),
            "support_confidence": aligned_support_confidence,
            "inpaint_mask": torch.clamp(1.0 - base_support_mask.float(), min=0.0, max=1.0),
            "support_projection_mask": base_support_mask.float() * aligned_support_confidence,
            "editable_low_confidence_mask": None,
        }

    def _build_frontier_proposal_mask(
        self,
        *,
        support_projection_mask: Optional[torch.Tensor],
        inpaint_mask: Optional[torch.Tensor],
        editable_low_confidence_mask: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        if inpaint_mask is None:
            return editable_low_confidence_mask
        frontier_kernel_size = max(
            int(getattr(self, "geometry_aware_frontier_kernel_size", 31)),
            3,
        )
        if frontier_kernel_size % 2 == 0:
            frontier_kernel_size += 1
        support_threshold = min(
            max(float(getattr(self, "geometry_aware_frontier_support_threshold", 0.05)), 0.0),
            1.0,
        )
        resolved_inpaint_mask = torch.nan_to_num(
            inpaint_mask.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if resolved_inpaint_mask.shape[-2:] != target_hw:
            resolved_inpaint_mask = F.interpolate(
                resolved_inpaint_mask,
                size=target_hw,
                mode="nearest",
            )

        support_seed = support_projection_mask
        if support_seed is None:
            support_seed = torch.clamp(1.0 - resolved_inpaint_mask, min=0.0, max=1.0)
        support_seed = torch.nan_to_num(
            support_seed.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if support_seed.shape[-2:] != target_hw:
            support_seed = F.interpolate(
                support_seed,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        support_binary_mask = (support_seed > support_threshold).float()
        dilated_support_mask = F.max_pool2d(
            support_binary_mask,
            kernel_size=frontier_kernel_size,
            stride=1,
            padding=frontier_kernel_size // 2,
        )
        frontier_proposal_mask = resolved_inpaint_mask * dilated_support_mask
        if editable_low_confidence_mask is not None:
            editable_low_confidence_mask = torch.nan_to_num(
                editable_low_confidence_mask.float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if editable_low_confidence_mask.shape[-2:] != target_hw:
                editable_low_confidence_mask = F.interpolate(
                    editable_low_confidence_mask,
                    size=target_hw,
                    mode="nearest",
                )
            frontier_proposal_mask = torch.clamp(
                frontier_proposal_mask + editable_low_confidence_mask,
                min=0.0,
                max=1.0,
            )
        if float(frontier_proposal_mask.sum().item()) < 1.0:
            return resolved_inpaint_mask
        return frontier_proposal_mask

    def _build_projection_aligned_exist_novel_masks(
        self,
        *,
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        refined_support_signals = self._build_sampler_aligned_support_signals(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        refined_support_projection_mask = refined_support_signals.get("support_projection_mask")
        refined_inpaint_mask = refined_support_signals.get("inpaint_mask")
        editable_low_confidence_mask = refined_support_signals.get(
            "editable_low_confidence_mask"
        )
        if bool(getattr(self, "geometry_aware_frontier_proposal_focus", False)):
            refined_inpaint_mask = self._build_frontier_proposal_mask(
                support_projection_mask=refined_support_projection_mask,
                inpaint_mask=refined_inpaint_mask,
                editable_low_confidence_mask=editable_low_confidence_mask,
                target_hw=target_hw,
            )
        if refined_support_projection_mask is None and refined_inpaint_mask is None:
            return None, None
        if refined_inpaint_mask is not None and refined_inpaint_mask.sum() < 1:
            refined_inpaint_mask = None
        return refined_support_projection_mask, refined_inpaint_mask

    def _build_exist_novel_masks(
        self,
        support_mask: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        exist_mask = self._build_support_mask(support_mask, target_hw)
        if exist_mask is None:
            exist_mask = self._build_depth_mask(depth, target_hw)
        if exist_mask is None:
            return None, None
        novel_mask = torch.clamp(1.0 - exist_mask, min=0.0, max=1.0)
        if novel_mask.sum() < 1:
            novel_mask = None
        return exist_mask, novel_mask

    def _build_latent_exist_novel_masks(
        self,
        support_mask: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
        latent_hw: Tuple[int, int],
        support_confidence: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if support_mask is not None and bool(
            getattr(self, "geometry_aware_frontier_proposal_focus", False)
        ):
            projection_exist_mask, projection_novel_mask = (
                self._build_projection_aligned_exist_novel_masks(
                    support_mask=support_mask,
                    support_confidence=support_confidence,
                    target_hw=latent_hw,
                )
            )
            if projection_exist_mask is not None or projection_novel_mask is not None:
                return projection_exist_mask, projection_novel_mask
        exist_mask, novel_mask = self._build_exist_novel_masks(support_mask, depth, latent_hw)
        return exist_mask, novel_mask

    def _masked_diffusion_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        depth_render: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor] = None,
        support_confidence: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        min_snr_gamma: Optional[float] = None,
        exist_diffusion_weight: Optional[float] = None,
        novel_diffusion_weight: Optional[float] = None,
    ) -> torch.Tensor:
        exist_mask, novel_mask = self._build_latent_exist_novel_masks(
            support_mask,
            depth_render,
            prediction.shape[-2:],
            support_confidence=support_confidence,
        )
        squared_error = (prediction - target.to(prediction.device, dtype=prediction.dtype)) ** 2
        if (
            self.enable_min_snr_weighting
            and timesteps is not None
        ):
            resolved_min_snr_gamma = float(
                self.min_snr_gamma if min_snr_gamma is None else min_snr_gamma
            )
            if resolved_min_snr_gamma > 0.0:
                alpha_values = self._get_scheduler_alphas_cumprod(
                    scheduler=self._resolve_training_noise_scheduler(),
                    device=prediction.device,
                    dtype=prediction.dtype,
                )
                if alpha_values is not None:
                    step_indices = timesteps.to(device=prediction.device, dtype=torch.long)
                    if step_indices.ndim == 0:
                        step_indices = step_indices.unsqueeze(0)
                    max_index = alpha_values.shape[0] - 1
                    step_indices = step_indices.clamp_min(0).clamp_max(max_index)
                    alpha_t = alpha_values.index_select(0, step_indices).view(-1, 1, 1, 1)
                    snr = alpha_t / torch.clamp(1.0 - alpha_t, min=1e-8)
                    min_snr_weight = torch.minimum(
                        snr,
                        torch.full_like(snr, resolved_min_snr_gamma),
                    ) / torch.clamp(snr, min=1e-8)
                    squared_error = squared_error * min_snr_weight
        if exist_mask is None and novel_mask is None:
            return squared_error.mean()

        weight_map = torch.zeros_like(squared_error[:, :1])
        resolved_novel_diffusion_weight = max(
            float(
                self.novel_diffusion_weight
                if novel_diffusion_weight is None
                else novel_diffusion_weight
            ),
            0.0,
        )
        resolved_exist_diffusion_weight = max(
            float(
                self.exist_diffusion_weight
                if exist_diffusion_weight is None
                else exist_diffusion_weight
            ),
            0.0,
        )
        if novel_mask is not None and resolved_novel_diffusion_weight > 0.0:
            novel_mask = novel_mask.to(device=prediction.device, dtype=prediction.dtype)
            weight_map = weight_map + novel_mask * resolved_novel_diffusion_weight
        if exist_mask is not None and resolved_exist_diffusion_weight > 0.0:
            exist_mask = exist_mask.to(device=prediction.device, dtype=prediction.dtype)
            weight_map = weight_map + exist_mask * resolved_exist_diffusion_weight
        if float(weight_map.sum().item()) < 1e-6:
            return squared_error.mean()
        expanded_weight_map = weight_map.expand_as(squared_error).clamp_min(1e-6)
        weighted_error = squared_error * expanded_weight_map
        return (
            weighted_error.sum(dim=(1, 2, 3))
            / expanded_weight_map.sum(dim=(1, 2, 3)).clamp_min(1e-6)
        ).mean()

    def _region_aware_reconstruction_loss(
        self,
        prediction_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        base_rgb: Optional[torch.Tensor],
        *,
        exist_mask: Optional[torch.Tensor],
        novel_mask: Optional[torch.Tensor],
        recon_alpha: float,
    ) -> torch.Tensor:
        if exist_mask is None and novel_mask is None:
            l1_loss = F.l1_loss(prediction_rgb, target_rgb)
            l2_loss = F.mse_loss(prediction_rgb, target_rgb)
            return recon_alpha * l1_loss + (1.0 - recon_alpha) * l2_loss

        loss_terms: List[torch.Tensor] = []
        if novel_mask is not None:
            novel_mask = novel_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
            masked_l1 = (
                (prediction_rgb - target_rgb).abs() * novel_mask
            ).sum(dim=(1, 2, 3)) / (
                novel_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * prediction_rgb.shape[1]
            )
            masked_l2 = (
                (prediction_rgb - target_rgb).pow(2) * novel_mask
            ).sum(dim=(1, 2, 3)) / (
                novel_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * prediction_rgb.shape[1]
            )
            novel_loss = (
                recon_alpha * masked_l1.mean() + (1.0 - recon_alpha) * masked_l2.mean()
            ) * self.novel_reconstruction_weight
            loss_terms.append(novel_loss)
        if exist_mask is not None and base_rgb is not None:
            exist_mask = exist_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
            base_rgb = base_rgb.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
            if base_rgb.shape[-2:] != prediction_rgb.shape[-2:]:
                base_rgb = F.interpolate(
                    base_rgb,
                    size=prediction_rgb.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            exist_l1 = (
                (prediction_rgb - base_rgb).abs() * exist_mask
            ).sum(dim=(1, 2, 3)) / (
                exist_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * prediction_rgb.shape[1]
            )
            loss_terms.append(exist_l1.mean() * self.exist_identity_weight)
        if not loss_terms:
            return prediction_rgb.new_zeros(())
        return torch.stack(loss_terms).mean()

    def _build_low_confidence_editable_mask(
        self,
        *,
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        refined_support_signals = self._build_sampler_aligned_support_signals(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        editable_low_confidence_mask = refined_support_signals.get("editable_low_confidence_mask")
        if editable_low_confidence_mask is None or editable_low_confidence_mask.sum() < 1:
            return None
        return editable_low_confidence_mask

    def _generate_final_sample_reference(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int,
    ) -> Optional[torch.Tensor]:
        if steps <= 0:
            return None
        sample_generation = self._generate_sample(
            batch,
            steps=steps,
            return_diagnostics=False,
        )
        if isinstance(sample_generation, dict):
            sample_rgb = sample_generation.get("rgb")
            if sample_rgb is None:
                sample_rgb = sample_generation.get("image")
        else:
            sample_rgb = sample_generation
        if sample_rgb is None:
            return None
        return torch.nan_to_num(sample_rgb.detach(), nan=0.0, posinf=1.0, neginf=0.0)

    def _estimate_batch_support_coverage(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[float]:
        support_mask = batch.get("warp_valid_mask")
        if not isinstance(support_mask, torch.Tensor):
            return None
        refined_support_signals = self._build_sampler_aligned_support_signals(
            support_mask=support_mask,
            support_confidence=batch.get("warp_support_confidence"),
            target_hw=support_mask.shape[-2:],
        )
        refined_support_projection_mask = refined_support_signals.get("support_projection_mask")
        if refined_support_projection_mask is None:
            return None
        return float(torch.nan_to_num(refined_support_projection_mask, nan=0.0).mean().item())

    def _estimate_batch_hole_ratio(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[float]:
        support_mask = batch.get("warp_valid_mask")
        if not isinstance(support_mask, torch.Tensor):
            return None
        refined_support_signals = self._build_sampler_aligned_support_signals(
            support_mask=support_mask,
            support_confidence=batch.get("warp_support_confidence"),
            target_hw=support_mask.shape[-2:],
        )
        refined_inpaint_mask = refined_support_signals.get("inpaint_mask")
        if refined_inpaint_mask is None:
            return None
        return float(torch.nan_to_num(refined_inpaint_mask, nan=0.0).mean().item())

    def _compute_exist_regret_vs_render(
        self,
        *,
        teacher_rgb: torch.Tensor,
        render_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        depth_render: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        effect_targets = self._compute_teacher_effect_targets(
            teacher_rgb=teacher_rgb,
            render_rgb=render_rgb,
            target_rgb=target_rgb,
            depth_render=depth_render,
            support_mask=support_mask,
            support_confidence=support_confidence,
            use_projection_aligned_support=True,
        )
        risk_exist_target = effect_targets.get("risk_exist_target")
        exist_mask = effect_targets.get("exist_mask")
        if risk_exist_target is None or exist_mask is None:
            return None
        exist_mask = exist_mask.to(device=risk_exist_target.device, dtype=risk_exist_target.dtype)
        masked_regret = (risk_exist_target * exist_mask).sum(dim=(1, 2, 3)) / (
            exist_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )
        return masked_regret.mean()

    def _resolve_final_sample_supervision_config(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        base_steps: int,
        base_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
    ) -> Dict[str, Optional[float]]:
        resolved_steps = max(1, int(base_steps))
        resolved_weight = None if base_weight is None else float(max(0.0, base_weight))
        resolved_max_weight = float(
            self.final_sample_supervision_max_weight if max_weight is None else max_weight
        )
        allow_hole_ratio_weight_boost = bool(
            getattr(self, "supports_sample_centric_training", False)
        )
        hole_ratio = self._estimate_batch_hole_ratio(batch)
        if hole_ratio is not None:
            large_hole_ratio_threshold = float(
                max(0.0, min(getattr(self, "final_sample_large_hole_ratio_threshold", 0.55), 0.99))
            )
            if hole_ratio >= large_hole_ratio_threshold:
                resolved_steps = max(
                    resolved_steps,
                    int(base_steps + getattr(self, "final_sample_large_hole_step_boost", 8)),
                )
                if resolved_weight is not None and allow_hole_ratio_weight_boost:
                    hole_severity = (hole_ratio - large_hole_ratio_threshold) / max(
                        1.0 - large_hole_ratio_threshold,
                        1e-6,
                    )
                    resolved_weight = min(
                        resolved_max_weight,
                        resolved_weight + 0.25 * float(max(0.0, hole_severity)),
                    )
            elif hole_ratio >= max(0.35, large_hole_ratio_threshold - 0.15):
                resolved_steps = max(
                    resolved_steps,
                    int(base_steps + getattr(self, "final_sample_medium_hole_step_boost", 4)),
                )
        return {
            "steps": float(resolved_steps),
            "weight": resolved_weight,
            "hole_ratio": hole_ratio,
        }

    def _compute_psnr_from_rgb(
        self,
        predicted_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
    ) -> torch.Tensor:
        mse_value = F.mse_loss(predicted_rgb, target_rgb)
        return 10.0 * torch.log10(1.0 / (mse_value + 1e-8))

    def _compute_psnr_from_error_statistics(
        self,
        *,
        squared_error_sum: float,
        weight_sum: float,
    ) -> Optional[float]:
        resolved_weight_sum = float(weight_sum)
        if resolved_weight_sum <= 1e-6:
            return None
        masked_mse = float(squared_error_sum) / resolved_weight_sum
        return float(10.0 * math.log10(1.0 / (masked_mse + 1e-8)))

    def _compute_rgb_squared_error_statistics(
        self,
        *,
        predicted_rgb: Optional[torch.Tensor],
        target_rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[float, float]:
        if predicted_rgb is None or target_rgb is None:
            return 0.0, 0.0
        target_hw = target_rgb.shape[-2:]
        aligned_predicted_rgb = torch.nan_to_num(
            predicted_rgb.to(device=target_rgb.device, dtype=target_rgb.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        if aligned_predicted_rgb.shape[-2:] != target_hw:
            aligned_predicted_rgb = F.interpolate(
                aligned_predicted_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_target_rgb = torch.nan_to_num(
            target_rgb.to(device=aligned_predicted_rgb.device, dtype=aligned_predicted_rgb.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        squared_error = (aligned_predicted_rgb - aligned_target_rgb).pow(2)
        if mask is None:
            return (
                float(squared_error.sum().item()),
                float(aligned_predicted_rgb.numel()),
            )
        aligned_mask = mask
        if aligned_mask.dim() == 3:
            aligned_mask = aligned_mask.unsqueeze(1)
        if aligned_mask.shape[1] != 1:
            aligned_mask = aligned_mask[:, :1]
        if aligned_mask.shape[-2:] != target_hw:
            aligned_mask = F.interpolate(
                aligned_mask.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask = torch.nan_to_num(
            aligned_mask.to(device=aligned_predicted_rgb.device, dtype=aligned_predicted_rgb.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        masked_weight_sum = float(
            (aligned_mask.sum() * float(aligned_predicted_rgb.shape[1])).item()
        )
        if masked_weight_sum <= 1e-6:
            return 0.0, 0.0
        masked_squared_error_sum = float(
            (squared_error * aligned_mask).sum().item()
        )
        return masked_squared_error_sum, masked_weight_sum

    def _compute_masked_psnr_from_rgb(
        self,
        predicted_rgb: Optional[torch.Tensor],
        target_rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if predicted_rgb is None or target_rgb is None or mask is None:
            return None
        target_hw = target_rgb.shape[-2:]
        if predicted_rgb.shape[-2:] != target_hw:
            predicted_rgb = F.interpolate(
                predicted_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask = mask
        if aligned_mask.dim() == 3:
            aligned_mask = aligned_mask.unsqueeze(1)
        if aligned_mask.shape[1] != 1:
            aligned_mask = aligned_mask[:, :1]
        if aligned_mask.shape[-2:] != target_hw:
            aligned_mask = F.interpolate(
                aligned_mask.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask = torch.nan_to_num(
            aligned_mask.to(device=predicted_rgb.device, dtype=predicted_rgb.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if float(aligned_mask.sum().item()) < 1e-6:
            return None
        aligned_target_rgb = target_rgb.to(device=predicted_rgb.device, dtype=predicted_rgb.dtype)
        squared_error = (predicted_rgb - aligned_target_rgb).pow(2)
        masked_mse = (
            squared_error * aligned_mask
        ).sum(dim=(1, 2, 3)) / (
            aligned_mask.sum(dim=(1, 2, 3)).clamp_min(1.0) * predicted_rgb.shape[1]
        )
        return 10.0 * torch.log10(1.0 / (masked_mse.mean() + 1e-8))

    def _compute_masked_regret_vs_reference(
        self,
        *,
        predicted_rgb: Optional[torch.Tensor],
        reference_rgb: Optional[torch.Tensor],
        target_rgb: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if predicted_rgb is None or reference_rgb is None or target_rgb is None or mask is None:
            return None
        target_hw = target_rgb.shape[-2:]
        if predicted_rgb.shape[-2:] != target_hw:
            predicted_rgb = F.interpolate(
                predicted_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if reference_rgb.shape[-2:] != target_hw:
            reference_rgb = F.interpolate(
                reference_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask = mask
        if aligned_mask.dim() == 3:
            aligned_mask = aligned_mask.unsqueeze(1)
        if aligned_mask.shape[1] != 1:
            aligned_mask = aligned_mask[:, :1]
        if aligned_mask.shape[-2:] != target_hw:
            aligned_mask = F.interpolate(
                aligned_mask.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask = torch.nan_to_num(
            aligned_mask.to(device=predicted_rgb.device, dtype=predicted_rgb.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if float(aligned_mask.sum().item()) < 1e-6:
            return None
        aligned_target_rgb = target_rgb.to(device=predicted_rgb.device, dtype=predicted_rgb.dtype)
        aligned_reference_rgb = reference_rgb.to(device=predicted_rgb.device, dtype=predicted_rgb.dtype)
        predicted_error = (predicted_rgb - aligned_target_rgb).abs().mean(dim=1, keepdim=True)
        reference_error = (aligned_reference_rgb - aligned_target_rgb).abs().mean(dim=1, keepdim=True)
        regret_map = F.relu(predicted_error - reference_error)
        masked_regret = (regret_map * aligned_mask).sum(dim=(1, 2, 3)) / (
            aligned_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        )
        return masked_regret.mean()

    def _tensor_mean_value(
        self,
        tensor: Optional[torch.Tensor],
    ) -> Optional[float]:
        if tensor is None or not isinstance(tensor, torch.Tensor):
            return None
        return float(
            torch.nan_to_num(
                tensor.detach().float(),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).mean().item()
        )

    def _extract_float_sequence(
        self,
        value: Any,
    ) -> List[float]:
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            flattened_tensor = torch.nan_to_num(
                value.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).reshape(-1)
            return [float(sequence_item.item()) for sequence_item in flattened_tensor]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            flattened_values: List[float] = []
            for sequence_item in value:
                flattened_values.extend(self._extract_float_sequence(sequence_item))
            return flattened_values
        try:
            return [float(value)]
        except (TypeError, ValueError):
            return []

    def _to_json_serializable_value(
        self,
        value: Any,
    ) -> Optional[Any]:
        if value is None:
            return None
        if isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            return value
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, torch.Tensor):
            serialized_tensor_values = self._extract_float_sequence(value)
            if not serialized_tensor_values:
                return None
            if len(serialized_tensor_values) == 1:
                return serialized_tensor_values[0]
            return serialized_tensor_values
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            serialized_sequence: List[Any] = []
            for sequence_item in value:
                serialized_item = self._to_json_serializable_value(sequence_item)
                if serialized_item is None:
                    continue
                serialized_sequence.append(serialized_item)
            return serialized_sequence
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _extract_serializable_prefixed_metrics(
        self,
        source_metrics: Dict[str, Any],
        *,
        key_prefix: str,
    ) -> Dict[str, Any]:
        serializable_metrics: Dict[str, Any] = {}
        for metric_name, metric_value in source_metrics.items():
            if not metric_name.startswith(key_prefix):
                continue
            serialized_metric_value = self._to_json_serializable_value(metric_value)
            if serialized_metric_value is None:
                continue
            serializable_metrics[metric_name] = serialized_metric_value
        return dict(sorted(serializable_metrics.items()))

    def _build_compact_fixed_runtime_log_segments(
        self,
        *,
        fixed_runtime_metrics: Dict[str, float],
        fixed_visualization_source: str,
    ) -> List[str]:
        compact_metric_segments: List[str] = []
        if "fixed_runtime_steps" in fixed_runtime_metrics:
            compact_metric_segments.append(
                f"steps={int(fixed_runtime_metrics['fixed_runtime_steps'])}"
            )
        for metric_name, metric_label, metric_format in FIXED_RUNTIME_COMPACT_LOG_METRICS:
            if metric_name not in fixed_runtime_metrics:
                continue
            metric_value = float(fixed_runtime_metrics[metric_name])
            if metric_format == "psnr":
                compact_metric_segments.append(f"{metric_label}={metric_value:.2f}dB")
            else:
                compact_metric_segments.append(f"{metric_label}={metric_value:.4f}")
        compact_metric_segments.append(f"source={fixed_visualization_source or 'unknown'}")
        return compact_metric_segments

    def _build_fixed_runtime_artifact_payload(
        self,
        *,
        stage_tag: str,
        epoch: int,
        sample_index: int,
        fixed_visualization_source: str,
        fixed_runtime_metrics: Dict[str, float],
        fixed_runtime_metric_sequences: Dict[str, List[float]],
        generation_diagnostics: Dict[str, Any],
        sample_debug_tensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        predecode_metrics = self._extract_serializable_prefixed_metrics(
            generation_diagnostics,
            key_prefix="predecode_",
        )
        requested_predecode_metric_names: List[str] = [
            "predecode_controlnet_down_residual_abs_mean_avg",
            "predecode_controlnet_mid_residual_abs_mean_avg",
            "predecode_cfg_noise_delta_abs_mean_avg",
            "predecode_cfg_noise_delta_variance_avg",
            "predecode_scheduler_step_delta_abs_mean_avg",
            "predecode_gcd_delta_abs_mean_avg",
            "predecode_cags_delta_abs_mean_avg",
            "predecode_projection_delta_abs_mean_avg",
        ]
        requested_predecode_metric_names.extend(
            sorted(
                metric_name
                for metric_name in predecode_metrics.keys()
                if metric_name.startswith("predecode_final_latent_")
            )
        )
        requested_predecode_metrics = {
            metric_name: predecode_metrics[metric_name]
            for metric_name in requested_predecode_metric_names
            if metric_name in predecode_metrics
        }
        serialized_metric_sequences = {
            sequence_name: [float(sequence_value) for sequence_value in sequence_values]
            for sequence_name, sequence_values in sorted(fixed_runtime_metric_sequences.items())
            if sequence_values
        }
        return {
            "stage_tag": stage_tag,
            "epoch": int(epoch),
            "sample_index": int(sample_index),
            "source": fixed_visualization_source or "unknown",
            "metrics": {
                metric_name: float(metric_value)
                for metric_name, metric_value in sorted(fixed_runtime_metrics.items())
            },
            "metric_sequences": serialized_metric_sequences,
            "requested_predecode_metrics": requested_predecode_metrics,
            "predecode_metrics": predecode_metrics,
            "available_debug_tensors": sorted(
                tensor_name
                for tensor_name, tensor_value in sample_debug_tensors.items()
                if isinstance(tensor_value, torch.Tensor)
            ),
        }

    def _save_json_payload(
        self,
        *,
        save_path: Path,
        payload: Dict[str, Any],
    ) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("w", encoding="utf-8") as payload_file:
            json.dump(payload, payload_file, indent=2, ensure_ascii=False, sort_keys=True)

    def _augment_fixed_runtime_gate0_debug_tensors(
        self,
        *,
        visualization_batch: Dict[str, Any],
        sample_debug_tensors: Dict[str, Any],
    ) -> Dict[str, Any]:
        augmented_debug_tensors = dict(sample_debug_tensors)

        def add_if_tensor_missing(output_name: str, tensor_value: Any) -> None:
            if output_name in augmented_debug_tensors:
                return
            if isinstance(tensor_value, torch.Tensor):
                augmented_debug_tensors[output_name] = tensor_value

        target_depth_proxy = visualization_batch.get("target_depth")
        if target_depth_proxy is None:
            target_depth_proxy = visualization_batch.get("target_depth_proxy")
        if target_depth_proxy is None:
            target_depth_proxy = visualization_batch.get("geometry_depth_render")
        add_if_tensor_missing("target_depth_proxy", target_depth_proxy)

        warp_depth = visualization_batch.get("warp_depth")
        if warp_depth is None:
            warp_depth = visualization_batch.get("render_depth")
        if warp_depth is None:
            warp_depth = visualization_batch.get("depth_render")
        add_if_tensor_missing("warp_depth", warp_depth)

        warp_alpha = visualization_batch.get("warp_alpha")
        if warp_alpha is None:
            warp_alpha = visualization_batch.get("render_alpha")
        if warp_alpha is None:
            warp_alpha = visualization_batch.get("accum_render")
        add_if_tensor_missing("warp_alpha", warp_alpha)

        add_if_tensor_missing("target_alpha", visualization_batch.get("target_alpha"))
        add_if_tensor_missing("proposal_depth", sample_debug_tensors.get("proposal_depth"))
        add_if_tensor_missing("proposal_alpha", sample_debug_tensors.get("proposal_alpha"))
        return augmented_debug_tensors

    def _prepare_fixed_runtime_gate0_tensor(
        self,
        tensor_value: Any,
    ) -> Optional[torch.Tensor]:
        if not isinstance(tensor_value, torch.Tensor):
            return None
        tensor_to_save = tensor_value.detach().cpu().float()
        if tensor_to_save.dim() == 4:
            tensor_to_save = tensor_to_save[0]
        if tensor_to_save.dim() not in (2, 3):
            return None
        return tensor_to_save

    def _resize_fixed_runtime_gate0_mask_like(
        self,
        mask_value: Optional[torch.Tensor],
        reference_value: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not isinstance(mask_value, torch.Tensor):
            return None
        if mask_value.dim() != 3:
            return None
        if not isinstance(reference_value, torch.Tensor):
            return mask_value.clamp(0.0, 1.0)
        if reference_value.dim() != 3:
            return mask_value.clamp(0.0, 1.0)
        if tuple(mask_value.shape[-2:]) == tuple(reference_value.shape[-2:]):
            return mask_value.clamp(0.0, 1.0)
        resized_mask = F.interpolate(
            mask_value.unsqueeze(0),
            size=tuple(reference_value.shape[-2:]),
            mode="nearest",
        ).squeeze(0)
        return resized_mask.clamp(0.0, 1.0)

    def _save_fixed_runtime_gate0_bundle(
        self,
        *,
        stage_tag: str,
        epoch: int,
        sample_index: int,
        fixed_visualization_source: str,
        target_rgb: Optional[torch.Tensor],
        sample_rgb: torch.Tensor,
        sample_debug_tensors: Dict[str, Any],
    ) -> None:
        if self.stage_samples_dir is None or not self._is_main_rank:
            return
        prepared_target_rgb = self._prepare_fixed_runtime_gate0_tensor(target_rgb)
        if prepared_target_rgb is None:
            return
        bundle: Dict[str, Any] = {
            "record_id": f"{stage_tag}_fixedsample{sample_index:02d}_epoch{epoch:03d}",
            "stage_tag": stage_tag,
            "epoch": int(epoch),
            "sample_index": int(sample_index),
            "source": fixed_visualization_source or "unknown",
            "target_rgb": prepared_target_rgb,
        }
        field_sources: Dict[str, str] = {"target_rgb": "fixed_runtime_target_rgb"}
        tensor_aliases: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
            ("warp_rgb", ("warped_rgb",)),
            ("target_depth_proxy", ("target_depth", "target_depth_proxy", "geometry_depth_render")),
            ("warp_depth", ("warp_depth", "render_depth", "depth_render")),
            ("warp_alpha", ("warp_alpha", "render_alpha", "accum_render")),
            ("target_alpha", ("target_alpha",)),
            ("proposal_depth", ("proposal_depth",)),
            ("proposal_alpha", ("proposal_alpha",)),
            ("support_composed_rgb", ("support_composed_image",)),
            ("raw_x0_rgb", ("decoded_teacher_rgb",)),
            ("proposal_rgb", ("proposal_raw_rgb",)),
            ("evaluation_mask", ("true_novel_inpaint_mask", "inpaint_mask")),
            ("support_projection_mask", ("support_projection_mask",)),
            ("P_rgb_trust", ("proposal_acceptance_confidence",)),
            ("P_depth_trust", ("proposal_verification_confidence",)),
            ("P_alpha_trust", ("proposal_repairability_confidence",)),
        )
        neutral_trust_fields: Dict[str, str] = {}
        for output_name, debug_tensor_names in tensor_aliases:
            tensor_value = None
            for debug_tensor_name in debug_tensor_names:
                tensor_value = sample_debug_tensors.get(debug_tensor_name)
                if tensor_value is not None:
                    break
            prepared_tensor = self._prepare_fixed_runtime_gate0_tensor(tensor_value)
            if prepared_tensor is not None:
                if output_name in {"P_rgb_trust", "P_depth_trust", "P_alpha_trust"}:
                    finite_tensor = torch.nan_to_num(
                        prepared_tensor.detach().float(),
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    )
                    dynamic_range = float((finite_tensor.max() - finite_tensor.min()).item())
                    mean_value = float(finite_tensor.mean().item())
                    if dynamic_range <= 1e-6 and abs(mean_value - 0.5) <= 1e-3:
                        neutral_trust_fields[output_name] = debug_tensor_name
                        continue
                bundle[output_name] = prepared_tensor
                field_sources[output_name] = debug_tensor_name
        bundle["field_sources"] = field_sources
        if neutral_trust_fields:
            bundle["skipped_neutral_trust_fields"] = neutral_trust_fields
            bundle["skipped_neutral_trust_policy"] = (
                "Constant neutral 0.5 trust tensors are non-discriminative diagnostics "
                "and are not exported as calibrated trust evidence."
            )
        bundle["proposal_depth_alpha_audit"] = {
            "proposal_depth_available": "proposal_depth" in bundle,
            "proposal_depth_source": field_sources.get("proposal_depth"),
            "proposal_alpha_available": "proposal_alpha" in bundle,
            "proposal_alpha_source": field_sources.get("proposal_alpha"),
            "scientific_policy": (
                "Proposal depth/alpha must come from explicit model debug tensors. They are not "
                "copied from target or baseline maps; calibration must validate proxy usability."
            ),
        }
        prepared_evaluation_mask = self._prepare_fixed_runtime_gate0_tensor(
            sample_debug_tensors.get("true_novel_inpaint_mask")
            if sample_debug_tensors.get("true_novel_inpaint_mask") is not None
            else sample_debug_tensors.get("inpaint_mask")
        )
        prepared_gate_masks: List[Tuple[str, torch.Tensor]] = []
        for gate_tensor_name in (
            "proposal_acceptance_gate",
            "proposal_component_gate",
            "proposal_focus_mask",
            "proposal_residual_gate",
        ):
            prepared_gate_component = self._prepare_fixed_runtime_gate0_tensor(
                sample_debug_tensors.get(gate_tensor_name)
            )
            if prepared_gate_component is None:
                continue
            bundle[gate_tensor_name] = prepared_gate_component
            resized_gate_component = self._resize_fixed_runtime_gate0_mask_like(
                prepared_gate_component,
                prepared_evaluation_mask,
            )
            if resized_gate_component is not None:
                gate_min = float(resized_gate_component.min().item())
                gate_max = float(resized_gate_component.max().item())
                gate_mean = float(resized_gate_component.mean().item())
                if abs(gate_max - gate_min) <= 1e-6 and 0.49 <= gate_mean <= 0.51:
                    continue
                prepared_gate_masks.append((gate_tensor_name, (resized_gate_component >= 0.5).float()))
        prepared_gate_mask = None
        if prepared_gate_masks:
            prepared_gate_mask = torch.ones_like(prepared_gate_masks[0][1])
            for _, prepared_gate_component in prepared_gate_masks:
                prepared_gate_mask = prepared_gate_mask * prepared_gate_component
            prepared_gate_mask = prepared_gate_mask.clamp(0.0, 1.0)
            bundle["proposal_gate_mask"] = prepared_gate_mask
            bundle["proposal_gate_mask_source"] = ",".join(
                gate_tensor_name for gate_tensor_name, _ in prepared_gate_masks
            )
        if (
            prepared_evaluation_mask is not None
            and prepared_gate_mask is not None
        ):
            bundle["verified_evidence_mask"] = (
                prepared_evaluation_mask * prepared_gate_mask
            ).clamp(0.0, 1.0)
            bundle["verified_evidence_mask_source"] = "evaluation_mask_x_proposal_gate"
        elif prepared_evaluation_mask is not None:
            bundle["verified_evidence_mask"] = prepared_evaluation_mask
            bundle["verified_evidence_mask_source"] = "evaluation_mask"
        elif prepared_gate_mask is not None:
            bundle["verified_evidence_mask"] = prepared_gate_mask
            bundle["verified_evidence_mask_source"] = "proposal_gate"
        else:
            prepared_editable_mask = self._prepare_fixed_runtime_gate0_tensor(
                sample_debug_tensors.get("masked_inpaint_editable_mask")
            )
            if prepared_editable_mask is not None:
                bundle["verified_evidence_mask"] = prepared_editable_mask
                bundle["verified_evidence_mask_source"] = "masked_inpaint_editable_mask"
        verified_evidence_mask = bundle.get("verified_evidence_mask")
        if isinstance(verified_evidence_mask, torch.Tensor):
            zero_mask = torch.zeros_like(verified_evidence_mask)
            bundle["where_wrong_hole"] = verified_evidence_mask
            bundle["where_wrong_geo_misalign"] = zero_mask
            bundle["where_wrong_artifact"] = zero_mask
            bundle["P_repairable_any"] = verified_evidence_mask
            bundle["P_repairable_hole"] = verified_evidence_mask
            bundle["P_repairable_geo_misalign"] = zero_mask
            bundle["P_repairable_artifact"] = zero_mask
        prepared_post_compose_rgb = self._prepare_fixed_runtime_gate0_tensor(sample_rgb)
        if prepared_post_compose_rgb is not None:
            bundle["post_compose_teacher_rgb"] = prepared_post_compose_rgb
        torch.save(
            bundle,
            self.stage_samples_dir
            / f"{stage_tag}_fixedsample{sample_index:02d}_epoch{epoch:03d}_gate0_bundle.pt",
        )

    def _persist_fixed_runtime_artifacts(
        self,
        *,
        stage_tag: str,
        epoch: int,
        sample_index: int,
        fixed_visualization_source: str,
        fixed_runtime_metrics: Dict[str, float],
        fixed_runtime_metric_sequences: Dict[str, List[float]],
        generation_diagnostics: Dict[str, Any],
        sample_rgb: torch.Tensor,
        sample_debug_tensors: Dict[str, Any],
        target_rgb: Optional[torch.Tensor] = None,
    ) -> None:
        if self.stage_samples_dir is None or not self._is_main_rank:
            return
        fixed_sample_stage_tag = f"{stage_tag}_fixedsample{sample_index:02d}"
        self._save_json_payload(
            save_path=self.stage_samples_dir
            / f"{fixed_sample_stage_tag}_epoch{epoch:03d}_metrics.json",
            payload=self._build_fixed_runtime_artifact_payload(
                stage_tag=stage_tag,
                epoch=epoch,
                sample_index=sample_index,
                fixed_visualization_source=fixed_visualization_source,
                fixed_runtime_metrics=fixed_runtime_metrics,
                fixed_runtime_metric_sequences=fixed_runtime_metric_sequences,
                generation_diagnostics=generation_diagnostics,
                sample_debug_tensors=sample_debug_tensors,
            ),
        )
        self._save_fixed_runtime_gate0_bundle(
            stage_tag=stage_tag,
            epoch=epoch,
            sample_index=sample_index,
            fixed_visualization_source=fixed_visualization_source,
            target_rgb=target_rgb,
            sample_rgb=sample_rgb,
            sample_debug_tensors=sample_debug_tensors,
        )

    def _append_fixed_runtime_chain_metrics(
        self,
        *,
        fixed_runtime_metrics: Dict[str, float],
        metric_stem: str,
        predicted_rgb: Optional[torch.Tensor],
        target_rgb: Optional[torch.Tensor],
        warped_rgb: Optional[torch.Tensor],
        proposal_reference_rgb: Optional[torch.Tensor],
        support_composed_image: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> None:
        warp_psnr = self._compute_masked_psnr_from_rgb(
            warped_rgb,
            target_rgb,
            mask,
        )
        if warp_psnr is not None:
            fixed_runtime_metrics[f"fixed_runtime_warp_{metric_stem}_psnr"] = float(
                warp_psnr.item()
            )
        pre_compose_psnr = self._compute_masked_psnr_from_rgb(
            proposal_reference_rgb,
            target_rgb,
            mask,
        )
        if pre_compose_psnr is not None:
            fixed_runtime_metrics[
                f"fixed_runtime_pre_compose_teacher_{metric_stem}_psnr"
            ] = float(pre_compose_psnr.item())
        support_composed_psnr = self._compute_masked_psnr_from_rgb(
            support_composed_image,
            target_rgb,
            mask,
        )
        if support_composed_psnr is not None:
            fixed_runtime_metrics[
                f"fixed_runtime_support_composed_{metric_stem}_psnr"
            ] = float(support_composed_psnr.item())
        post_compose_psnr = self._compute_masked_psnr_from_rgb(
            predicted_rgb,
            target_rgb,
            mask,
        )
        if post_compose_psnr is not None:
            fixed_runtime_metrics[
                f"fixed_runtime_post_compose_teacher_{metric_stem}_psnr"
            ] = float(post_compose_psnr.item())
        if pre_compose_psnr is not None and post_compose_psnr is not None:
            fixed_runtime_metrics[
                f"fixed_runtime_compose_gain_vs_pre_{metric_stem}_psnr"
            ] = float((post_compose_psnr - pre_compose_psnr).item())
        if warp_psnr is not None and post_compose_psnr is not None:
            fixed_runtime_metrics[
                f"fixed_runtime_compose_gain_vs_warp_{metric_stem}_psnr"
            ] = float((post_compose_psnr - warp_psnr).item())

    def _collect_fixed_visualization_group_metrics(
        self,
        *,
        visualization_batch: Dict[str, Any],
        fixed_visualization_source: str,
        base_steps: int,
        base_weight: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        fixed_sample_config = self._resolve_final_sample_supervision_config(
            visualization_batch,
            base_steps=base_steps,
            base_weight=base_weight,
        )
        fixed_sample_steps = max(1, int(fixed_sample_config["steps"] or base_steps))
        sample_generation = self._generate_sample(
            visualization_batch,
            steps=fixed_sample_steps,
            return_diagnostics=True,
        )
        if not isinstance(sample_generation, dict):
            return None
        fixed_sample_rgb = sample_generation.get("rgb")
        if fixed_sample_rgb is None:
            fixed_sample_rgb = sample_generation.get("image")
        if fixed_sample_rgb is None or not isinstance(fixed_sample_rgb, torch.Tensor):
            return None
        fixed_target_rgb = visualization_batch.get("target_rgb")
        if fixed_target_rgb is None or not isinstance(fixed_target_rgb, torch.Tensor):
            return None
        generation_diagnostics = sample_generation.get("diagnostics")
        if not isinstance(generation_diagnostics, dict):
            generation_diagnostics = {}
        sample_debug_tensors = sample_generation.get("debug_tensors")
        if not isinstance(sample_debug_tensors, dict):
            sample_debug_tensors = {}
        sample_debug_tensors = self._augment_fixed_runtime_gate0_debug_tensors(
            visualization_batch=visualization_batch,
            sample_debug_tensors=sample_debug_tensors,
        )
        runtime_support_projection_mask = sample_debug_tensors.get("support_projection_mask")
        runtime_true_novel_mask = sample_debug_tensors.get("true_novel_inpaint_mask")
        runtime_inpaint_mask = (
            runtime_true_novel_mask
            if isinstance(runtime_true_novel_mask, torch.Tensor)
            else sample_debug_tensors.get("inpaint_mask")
        )
        runtime_masked_inpaint_editable_mask = sample_debug_tensors.get(
            "masked_inpaint_editable_mask"
        )
        runtime_warped_rgb = sample_debug_tensors.get("warped_rgb")
        runtime_proposal_reference_rgb = self._resolve_proposal_reference_rgb_from_debug_tensors(
            sample_debug_tensors,
            fallback_rgb=fixed_sample_rgb,
        )
        runtime_support_composed_image = sample_debug_tensors.get("support_composed_image")
        group_metrics: Dict[str, float] = {
            "fixed_sample_psnr": float(
                self._compute_psnr_from_rgb(
                    fixed_sample_rgb,
                    fixed_target_rgb.to(
                        device=fixed_sample_rgb.device,
                        dtype=fixed_sample_rgb.dtype,
                    ),
                ).item()
            ),
            "fixed_runtime_steps": float(fixed_sample_steps),
        }
        for predecode_metric_name in FIXED_RUNTIME_PREDECODE_METRIC_NAMES:
            predecode_metric_value = generation_diagnostics.get(predecode_metric_name)
            if predecode_metric_value is None:
                continue
            group_metrics[f"fixed_runtime_{predecode_metric_name}"] = float(predecode_metric_value)
        fixed_runtime_metric_sequences: Dict[str, List[float]] = {}
        for sequence_name in FIXED_RUNTIME_SEQUENCE_DIAGNOSTIC_NAMES:
            extracted_sequence_values = self._extract_float_sequence(
                generation_diagnostics.get(sequence_name)
            )
            if not extracted_sequence_values:
                continue
            fixed_runtime_metric_sequences[sequence_name] = extracted_sequence_values
        support_projection_coverage = self._tensor_mean_value(runtime_support_projection_mask)
        if support_projection_coverage is not None:
            group_metrics["fixed_runtime_support_projection_coverage"] = float(
                support_projection_coverage
            )
        editable_coverage = self._tensor_mean_value(runtime_masked_inpaint_editable_mask)
        if editable_coverage is not None:
            group_metrics["fixed_runtime_masked_inpaint_editable_coverage"] = float(
                editable_coverage
            )
        novel_coverage = self._tensor_mean_value(runtime_inpaint_mask)
        if novel_coverage is not None:
            group_metrics["fixed_runtime_novel_coverage"] = float(novel_coverage)
        fixed_novel_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_inpaint_mask,
        )
        if fixed_novel_psnr is not None:
            group_metrics["fixed_runtime_novel_psnr"] = float(fixed_novel_psnr.item())
        self._append_fixed_runtime_chain_metrics(
            fixed_runtime_metrics=group_metrics,
            metric_stem="support_projection",
            predicted_rgb=fixed_sample_rgb,
            target_rgb=fixed_target_rgb,
            warped_rgb=runtime_warped_rgb,
            proposal_reference_rgb=runtime_proposal_reference_rgb,
            support_composed_image=runtime_support_composed_image,
            mask=runtime_support_projection_mask,
        )
        self._append_fixed_runtime_chain_metrics(
            fixed_runtime_metrics=group_metrics,
            metric_stem="masked_inpaint_editable",
            predicted_rgb=fixed_sample_rgb,
            target_rgb=fixed_target_rgb,
            warped_rgb=runtime_warped_rgb,
            proposal_reference_rgb=runtime_proposal_reference_rgb,
            support_composed_image=runtime_support_composed_image,
            mask=runtime_masked_inpaint_editable_mask,
        )
        self._append_fixed_runtime_chain_metrics(
            fixed_runtime_metrics=group_metrics,
            metric_stem="novel",
            predicted_rgb=fixed_sample_rgb,
            target_rgb=fixed_target_rgb,
            warped_rgb=runtime_warped_rgb,
            proposal_reference_rgb=runtime_proposal_reference_rgb,
            support_composed_image=runtime_support_composed_image,
            mask=runtime_inpaint_mask,
        )
        return {
            "metrics": group_metrics,
            "metric_sequences": fixed_runtime_metric_sequences,
            "generation_diagnostics": generation_diagnostics,
            "sample_rgb": fixed_sample_rgb,
            "sample_debug_tensors": sample_debug_tensors,
            "source": fixed_visualization_source,
            "visualization_batch": visualization_batch,
        }

    def _log_fixed_visualization_runtime_metrics(
        self,
        *,
        stage_tag: str,
        epoch: int,
        base_steps: int,
        base_weight: Optional[float] = None,
    ) -> Optional[Dict[str, float]]:
        if not self._is_main_rank:
            return None
        fixed_visualization_batches = self._get_fixed_visualization_batches()
        if not fixed_visualization_batches:
            return None
        visualization_batch, fixed_visualization_source = fixed_visualization_batches[0]
        fixed_sample_config = self._resolve_final_sample_supervision_config(
            visualization_batch,
            base_steps=base_steps,
            base_weight=base_weight,
        )
        fixed_sample_steps = max(1, int(fixed_sample_config["steps"] or base_steps))
        sample_generation = self._generate_sample(
            visualization_batch,
            steps=fixed_sample_steps,
            return_diagnostics=True,
        )
        if not isinstance(sample_generation, dict):
            return None
        fixed_sample_rgb = sample_generation.get("rgb")
        if fixed_sample_rgb is None:
            fixed_sample_rgb = sample_generation.get("image")
        if fixed_sample_rgb is None or not isinstance(fixed_sample_rgb, torch.Tensor):
            return None
        fixed_target_rgb = visualization_batch.get("target_rgb")
        if fixed_target_rgb is None or not isinstance(fixed_target_rgb, torch.Tensor):
            return None
        generation_diagnostics = sample_generation.get("diagnostics")
        if not isinstance(generation_diagnostics, dict):
            generation_diagnostics = {}
        sample_debug_tensors = sample_generation.get("debug_tensors")
        if not isinstance(sample_debug_tensors, dict):
            sample_debug_tensors = {}
        runtime_support_projection_mask = sample_debug_tensors.get("support_projection_mask")
        runtime_true_novel_mask = sample_debug_tensors.get("true_novel_inpaint_mask")
        runtime_inpaint_mask = (
            runtime_true_novel_mask
            if isinstance(runtime_true_novel_mask, torch.Tensor)
            else sample_debug_tensors.get("inpaint_mask")
        )
        runtime_masked_inpaint_editable_mask = sample_debug_tensors.get(
            "masked_inpaint_editable_mask"
        )
        runtime_warped_rgb = sample_debug_tensors.get("warped_rgb")
        runtime_proposal_reference_rgb = self._resolve_proposal_reference_rgb_from_debug_tensors(
            sample_debug_tensors,
            fallback_rgb=fixed_sample_rgb,
        )
        runtime_support_composed_image = sample_debug_tensors.get("support_composed_image")
        runtime_counterfactual_support_projection_mask = sample_debug_tensors.get(
            "counterfactual_support_projection_mask"
        )
        runtime_counterfactual_support_projection_delta_mask = sample_debug_tensors.get(
            "counterfactual_support_projection_delta_mask"
        )
        runtime_counterfactual_applied_mask = sample_debug_tensors.get(
            "counterfactual_applied_mask"
        )
        runtime_newly_added_projection_mask = sample_debug_tensors.get(
            "newly_added_projection_mask"
        )
        runtime_newly_added_quality_suspect_mask = sample_debug_tensors.get(
            "newly_added_quality_suspect_mask"
        )
        fixed_runtime_metric_sequences: Dict[str, List[float]] = {}
        runtime_exist_mask = None
        if isinstance(runtime_inpaint_mask, torch.Tensor):
            runtime_exist_mask = torch.clamp(
                1.0 - runtime_inpaint_mask.to(dtype=fixed_sample_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        fixed_runtime_metrics: Dict[str, float] = {
            "fixed_sample_psnr": float(
                self._compute_psnr_from_rgb(
                    fixed_sample_rgb,
                    fixed_target_rgb.to(device=fixed_sample_rgb.device, dtype=fixed_sample_rgb.dtype),
                ).item()
            ),
            "fixed_runtime_steps": float(fixed_sample_steps),
            "fixed_masked_inpaint_support_uses_projection": float(
                bool(generation_diagnostics.get("masked_inpaint_support_uses_projection", False))
            ),
        }
        runtime_support_projection_coverage = generation_diagnostics.get(
            "runtime_support_projection_coverage"
        )
        if runtime_support_projection_coverage is None:
            runtime_support_projection_coverage = self._tensor_mean_value(
                runtime_support_projection_mask
            )
        if runtime_support_projection_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_support_projection_coverage"] = float(
                runtime_support_projection_coverage
            )
        counterfactual_support_projection_strength = generation_diagnostics.get(
            "support_projection_counterfactual_strength"
        )
        if counterfactual_support_projection_strength is not None:
            fixed_runtime_metrics[
                "fixed_runtime_support_projection_counterfactual_strength"
            ] = float(counterfactual_support_projection_strength)
        runtime_counterfactual_support_projection_coverage = generation_diagnostics.get(
            "counterfactual_support_projection_coverage"
        )
        if runtime_counterfactual_support_projection_coverage is None:
            runtime_counterfactual_support_projection_coverage = self._tensor_mean_value(
                runtime_counterfactual_support_projection_mask
            )
        if runtime_counterfactual_support_projection_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_counterfactual_support_projection_coverage"
            ] = float(runtime_counterfactual_support_projection_coverage)
        runtime_counterfactual_support_projection_delta_coverage = (
            generation_diagnostics.get("counterfactual_support_projection_delta_coverage")
        )
        if runtime_counterfactual_support_projection_delta_coverage is None:
            runtime_counterfactual_support_projection_delta_coverage = self._tensor_mean_value(
                runtime_counterfactual_support_projection_delta_mask
            )
        if runtime_counterfactual_support_projection_delta_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_counterfactual_support_projection_delta_coverage"
            ] = float(runtime_counterfactual_support_projection_delta_coverage)
        runtime_counterfactual_applied_mask_coverage = generation_diagnostics.get(
            "counterfactual_applied_mask_coverage"
        )
        if runtime_counterfactual_applied_mask_coverage is None:
            runtime_counterfactual_applied_mask_coverage = self._tensor_mean_value(
                runtime_counterfactual_applied_mask
            )
        if runtime_counterfactual_applied_mask_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_counterfactual_applied_mask_coverage"
            ] = float(runtime_counterfactual_applied_mask_coverage)
        runtime_latent_support_mask_coverage = generation_diagnostics.get(
            "runtime_latent_support_mask_coverage"
        )
        if runtime_latent_support_mask_coverage is None:
            runtime_latent_support_mask_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("masked_inpaint_support_mask")
            )
        if runtime_latent_support_mask_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_latent_support_mask_coverage"] = float(
                runtime_latent_support_mask_coverage
            )
        masked_support_clean_latent_abs_mean = generation_diagnostics.get(
            "masked_support_clean_latent_abs_mean"
        )
        if masked_support_clean_latent_abs_mean is not None:
            fixed_runtime_metrics["fixed_runtime_masked_support_clean_latent_abs_mean"] = float(
                masked_support_clean_latent_abs_mean
            )
        masked_support_clean_latent_variance = generation_diagnostics.get(
            "masked_support_clean_latent_variance"
        )
        if masked_support_clean_latent_variance is not None:
            fixed_runtime_metrics["fixed_runtime_masked_support_clean_latent_variance"] = float(
                masked_support_clean_latent_variance
            )
        masked_support_noised_latent_abs_mean = generation_diagnostics.get(
            "masked_support_noised_latent_abs_mean"
        )
        if masked_support_noised_latent_abs_mean is not None:
            fixed_runtime_metrics["fixed_runtime_masked_support_noised_latent_abs_mean"] = float(
                masked_support_noised_latent_abs_mean
            )
        masked_support_noised_latent_variance = generation_diagnostics.get(
            "masked_support_noised_latent_variance"
        )
        if masked_support_noised_latent_variance is not None:
            fixed_runtime_metrics["fixed_runtime_masked_support_noised_latent_variance"] = float(
                masked_support_noised_latent_variance
            )
        for predecode_metric_name in FIXED_RUNTIME_PREDECODE_METRIC_NAMES:
            predecode_metric_value = generation_diagnostics.get(predecode_metric_name)
            if predecode_metric_value is None:
                continue
            fixed_runtime_metrics[f"fixed_runtime_{predecode_metric_name}"] = float(
                predecode_metric_value
            )
        runtime_union_valid_coverage_before_distance_weight = generation_diagnostics.get(
            "union_valid_coverage_before_distance_weight"
        )
        if runtime_union_valid_coverage_before_distance_weight is None:
            runtime_union_valid_coverage_before_distance_weight = self._tensor_mean_value(
                sample_debug_tensors.get("union_valid_mask_before_distance_weight")
            )
        if runtime_union_valid_coverage_before_distance_weight is not None:
            fixed_runtime_metrics[
                "fixed_runtime_union_valid_coverage_before_distance_weight"
            ] = float(runtime_union_valid_coverage_before_distance_weight)
        runtime_weighted_support_coverage_before_projection_threshold = generation_diagnostics.get(
            "weighted_support_coverage_before_projection_threshold"
        )
        if runtime_weighted_support_coverage_before_projection_threshold is None:
            runtime_weighted_support_coverage_before_projection_threshold = self._tensor_mean_value(
                sample_debug_tensors.get("raw_support_confidence")
            )
        if runtime_weighted_support_coverage_before_projection_threshold is not None:
            fixed_runtime_metrics[
                "fixed_runtime_weighted_support_coverage_before_projection_threshold"
            ] = float(runtime_weighted_support_coverage_before_projection_threshold)
        runtime_joint_valid_but_below_projection_threshold_coverage = generation_diagnostics.get(
            "joint_valid_but_below_projection_threshold_coverage"
        )
        if runtime_joint_valid_but_below_projection_threshold_coverage is None:
            runtime_joint_valid_but_below_projection_threshold_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("joint_valid_but_below_projection_threshold_mask")
            )
        if runtime_joint_valid_but_below_projection_threshold_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_joint_valid_but_below_projection_threshold_coverage"
            ] = float(runtime_joint_valid_but_below_projection_threshold_coverage)
        runtime_legacy_support_projection_coverage = generation_diagnostics.get(
            "legacy_support_projection_coverage"
        )
        if runtime_legacy_support_projection_coverage is None:
            runtime_legacy_support_projection_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("legacy_support_projection_mask")
            )
        if runtime_legacy_support_projection_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_legacy_support_projection_coverage"] = float(
                runtime_legacy_support_projection_coverage
            )
        runtime_newly_added_projection_coverage = generation_diagnostics.get(
            "newly_added_projection_coverage"
        )
        if runtime_newly_added_projection_coverage is None:
            runtime_newly_added_projection_coverage = self._tensor_mean_value(
                runtime_newly_added_projection_mask
            )
        if runtime_newly_added_projection_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_newly_added_projection_coverage"] = float(
                runtime_newly_added_projection_coverage
            )
        runtime_quality_suspect_projection_coverage = generation_diagnostics.get(
            "quality_suspect_projection_coverage"
        )
        if runtime_quality_suspect_projection_coverage is None:
            runtime_quality_suspect_projection_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("quality_suspect_projection_mask")
            )
        if runtime_quality_suspect_projection_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_quality_suspect_projection_coverage"] = float(
                runtime_quality_suspect_projection_coverage
            )
        runtime_newly_added_quality_suspect_coverage = generation_diagnostics.get(
            "newly_added_quality_suspect_coverage"
        )
        if runtime_newly_added_quality_suspect_coverage is None:
            runtime_newly_added_quality_suspect_coverage = self._tensor_mean_value(
                runtime_newly_added_quality_suspect_mask
            )
        if runtime_newly_added_quality_suspect_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_newly_added_quality_suspect_coverage"
            ] = float(runtime_newly_added_quality_suspect_coverage)
        runtime_support_confidence_optimism_gap_coverage = generation_diagnostics.get(
            "support_confidence_optimism_gap_coverage"
        )
        if runtime_support_confidence_optimism_gap_coverage is None:
            runtime_support_confidence_optimism_gap_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("support_confidence_optimism_gap")
            )
        if runtime_support_confidence_optimism_gap_coverage is not None:
            fixed_runtime_metrics[
                "fixed_runtime_support_confidence_optimism_gap_coverage"
            ] = float(runtime_support_confidence_optimism_gap_coverage)
        runtime_support_fusion_rgb_disagreement = generation_diagnostics.get(
            "support_fusion_rgb_disagreement"
        )
        if runtime_support_fusion_rgb_disagreement is None:
            runtime_support_fusion_rgb_disagreement = self._tensor_mean_value(
                sample_debug_tensors.get("support_fusion_rgb_disagreement")
            )
        if runtime_support_fusion_rgb_disagreement is not None:
            fixed_runtime_metrics["fixed_runtime_support_fusion_rgb_disagreement"] = float(
                runtime_support_fusion_rgb_disagreement
            )
        per_view_valid_coverage_values = self._extract_float_sequence(
            generation_diagnostics.get("per_view_valid_coverage")
        )
        if per_view_valid_coverage_values:
            fixed_runtime_metric_sequences["per_view_valid_coverage"] = per_view_valid_coverage_values
        per_view_distance_weight_values = self._extract_float_sequence(
            generation_diagnostics.get("per_view_distance_weight")
        )
        if per_view_distance_weight_values:
            fixed_runtime_metric_sequences["per_view_distance_weight"] = per_view_distance_weight_values
        per_view_raw_inverse_distance_weight_values = self._extract_float_sequence(
            generation_diagnostics.get("per_view_raw_inverse_distance_weight")
        )
        if per_view_raw_inverse_distance_weight_values:
            fixed_runtime_metric_sequences["per_view_raw_inverse_distance_weight"] = (
                per_view_raw_inverse_distance_weight_values
            )
        per_view_coverage_distance_weight_values = self._extract_float_sequence(
            generation_diagnostics.get("per_view_coverage_distance_weight")
        )
        if per_view_coverage_distance_weight_values:
            fixed_runtime_metric_sequences["per_view_coverage_distance_weight"] = (
                per_view_coverage_distance_weight_values
            )
        per_view_fusion_distance_weight_values = self._extract_float_sequence(
            generation_diagnostics.get("per_view_fusion_distance_weight")
        )
        if per_view_fusion_distance_weight_values:
            fixed_runtime_metric_sequences["per_view_fusion_distance_weight"] = (
                per_view_fusion_distance_weight_values
            )
        runtime_masked_support_coverage = generation_diagnostics.get(
            "runtime_masked_inpaint_support_coverage"
        )
        if runtime_masked_support_coverage is None:
            runtime_masked_support_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("masked_inpaint_support_mask")
            )
        if runtime_masked_support_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_masked_inpaint_support_coverage"] = float(
                runtime_masked_support_coverage
            )
        runtime_masked_editable_coverage = generation_diagnostics.get(
            "runtime_masked_inpaint_editable_coverage"
        )
        if runtime_masked_editable_coverage is None:
            runtime_masked_editable_coverage = self._tensor_mean_value(
                sample_debug_tensors.get("masked_inpaint_editable_mask")
            )
        if runtime_masked_editable_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_masked_inpaint_editable_coverage"] = float(
                runtime_masked_editable_coverage
            )
        runtime_novel_coverage = self._tensor_mean_value(runtime_inpaint_mask)
        if runtime_novel_coverage is not None:
            fixed_runtime_metrics["fixed_runtime_novel_coverage"] = float(
                runtime_novel_coverage
            )
        fixed_exist_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_exist_mask,
        )
        if fixed_exist_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_exist_psnr"] = float(fixed_exist_psnr.item())
        fixed_novel_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_inpaint_mask,
        )
        if fixed_novel_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_novel_psnr"] = float(fixed_novel_psnr.item())
        fixed_warp_exist_psnr = self._compute_masked_psnr_from_rgb(
            runtime_warped_rgb,
            fixed_target_rgb,
            runtime_exist_mask,
        )
        if fixed_warp_exist_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_warp_exist_psnr"] = float(
                fixed_warp_exist_psnr.item()
            )
        fixed_warp_support_projection_psnr = self._compute_masked_psnr_from_rgb(
            runtime_warped_rgb,
            fixed_target_rgb,
            runtime_support_projection_mask,
        )
        if fixed_warp_support_projection_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_warp_support_projection_psnr"] = float(
                fixed_warp_support_projection_psnr.item()
            )
        fixed_pre_compose_teacher_support_projection_psnr = self._compute_masked_psnr_from_rgb(
            runtime_proposal_reference_rgb,
            fixed_target_rgb,
            runtime_support_projection_mask,
        )
        if fixed_pre_compose_teacher_support_projection_psnr is not None:
            fixed_runtime_metrics[
                "fixed_runtime_pre_compose_teacher_support_projection_psnr"
            ] = float(fixed_pre_compose_teacher_support_projection_psnr.item())
        fixed_support_composed_projection_psnr = self._compute_masked_psnr_from_rgb(
            runtime_support_composed_image,
            fixed_target_rgb,
            runtime_support_projection_mask,
        )
        if fixed_support_composed_projection_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_support_composed_projection_psnr"] = float(
                fixed_support_composed_projection_psnr.item()
            )
        fixed_post_compose_teacher_support_projection_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_support_projection_mask,
        )
        if fixed_post_compose_teacher_support_projection_psnr is not None:
            fixed_runtime_metrics[
                "fixed_runtime_post_compose_teacher_support_projection_psnr"
            ] = float(fixed_post_compose_teacher_support_projection_psnr.item())
        if (
            fixed_pre_compose_teacher_support_projection_psnr is not None
            and fixed_post_compose_teacher_support_projection_psnr is not None
        ):
            fixed_runtime_metrics[
                "fixed_runtime_compose_gain_vs_pre_support_projection_psnr"
            ] = float(
                (
                    fixed_post_compose_teacher_support_projection_psnr
                    - fixed_pre_compose_teacher_support_projection_psnr
                ).item()
            )
        if (
            fixed_warp_support_projection_psnr is not None
            and fixed_post_compose_teacher_support_projection_psnr is not None
        ):
            fixed_runtime_metrics[
                "fixed_runtime_compose_gain_vs_warp_support_projection_psnr"
            ] = float(
                (
                    fixed_post_compose_teacher_support_projection_psnr
                    - fixed_warp_support_projection_psnr
                ).item()
            )
        self._append_fixed_runtime_chain_metrics(
            fixed_runtime_metrics=fixed_runtime_metrics,
            metric_stem="masked_inpaint_editable",
            predicted_rgb=fixed_sample_rgb,
            target_rgb=fixed_target_rgb,
            warped_rgb=runtime_warped_rgb,
            proposal_reference_rgb=runtime_proposal_reference_rgb,
            support_composed_image=runtime_support_composed_image,
            mask=runtime_masked_inpaint_editable_mask,
        )
        self._append_fixed_runtime_chain_metrics(
            fixed_runtime_metrics=fixed_runtime_metrics,
            metric_stem="novel",
            predicted_rgb=fixed_sample_rgb,
            target_rgb=fixed_target_rgb,
            warped_rgb=runtime_warped_rgb,
            proposal_reference_rgb=runtime_proposal_reference_rgb,
            support_composed_image=runtime_support_composed_image,
            mask=runtime_inpaint_mask,
        )
        fixed_exist_regret_vs_warp = self._compute_masked_regret_vs_reference(
            predicted_rgb=fixed_sample_rgb,
            reference_rgb=runtime_warped_rgb,
            target_rgb=fixed_target_rgb,
            mask=runtime_exist_mask,
        )
        if fixed_exist_regret_vs_warp is not None:
            fixed_runtime_metrics["fixed_runtime_exist_regret_vs_warp"] = float(
                fixed_exist_regret_vs_warp.item()
            )
        fixed_newly_added_projection_sample_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_newly_added_projection_mask,
        )
        if fixed_newly_added_projection_sample_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_newly_added_projection_sample_psnr"] = float(
                fixed_newly_added_projection_sample_psnr.item()
            )
        fixed_newly_added_projection_warp_psnr = self._compute_masked_psnr_from_rgb(
            runtime_warped_rgb,
            fixed_target_rgb,
            runtime_newly_added_projection_mask,
        )
        if fixed_newly_added_projection_warp_psnr is not None:
            fixed_runtime_metrics["fixed_runtime_newly_added_projection_warp_psnr"] = float(
                fixed_newly_added_projection_warp_psnr.item()
            )
        fixed_newly_added_projection_exist_regret_vs_warp = self._compute_masked_regret_vs_reference(
            predicted_rgb=fixed_sample_rgb,
            reference_rgb=runtime_warped_rgb,
            target_rgb=fixed_target_rgb,
            mask=runtime_newly_added_projection_mask,
        )
        if fixed_newly_added_projection_exist_regret_vs_warp is not None:
            fixed_runtime_metrics[
                "fixed_runtime_newly_added_projection_exist_regret_vs_warp"
            ] = float(fixed_newly_added_projection_exist_regret_vs_warp.item())
        fixed_newly_added_quality_suspect_sample_psnr = self._compute_masked_psnr_from_rgb(
            fixed_sample_rgb,
            fixed_target_rgb,
            runtime_newly_added_quality_suspect_mask,
        )
        if fixed_newly_added_quality_suspect_sample_psnr is not None:
            fixed_runtime_metrics[
                "fixed_runtime_newly_added_quality_suspect_sample_psnr"
            ] = float(fixed_newly_added_quality_suspect_sample_psnr.item())
        fixed_newly_added_quality_suspect_warp_psnr = self._compute_masked_psnr_from_rgb(
            runtime_warped_rgb,
            fixed_target_rgb,
            runtime_newly_added_quality_suspect_mask,
        )
        if fixed_newly_added_quality_suspect_warp_psnr is not None:
            fixed_runtime_metrics[
                "fixed_runtime_newly_added_quality_suspect_warp_psnr"
            ] = float(fixed_newly_added_quality_suspect_warp_psnr.item())
        fixed_newly_added_quality_suspect_exist_regret_vs_warp = (
            self._compute_masked_regret_vs_reference(
                predicted_rgb=fixed_sample_rgb,
                reference_rgb=runtime_warped_rgb,
                target_rgb=fixed_target_rgb,
                mask=runtime_newly_added_quality_suspect_mask,
            )
        )
        if fixed_newly_added_quality_suspect_exist_regret_vs_warp is not None:
            fixed_runtime_metrics[
                "fixed_runtime_newly_added_quality_suspect_exist_regret_vs_warp"
            ] = float(fixed_newly_added_quality_suspect_exist_regret_vs_warp.item())
        stage_label = "Stage2" if stage_tag == "stage2b" else "Stage3"
        fixed_metric_segments = self._build_compact_fixed_runtime_log_segments(
            fixed_runtime_metrics=fixed_runtime_metrics,
            fixed_visualization_source=fixed_visualization_source,
        )
        print(
            f"[{stage_label}] Epoch {epoch} fixed-sample runtime metrics: "
            + ", ".join(fixed_metric_segments)
        )
        if self.writer is not None:
            for metric_name, metric_value in fixed_runtime_metrics.items():
                self.writer.add_scalar(f"{stage_tag}/{metric_name}", metric_value, epoch)
            for sequence_name, sequence_values in fixed_runtime_metric_sequences.items():
                for sequence_index, sequence_value in enumerate(sequence_values):
                    self.writer.add_scalar(
                        f"{stage_tag}/fixed_runtime_{sequence_name}_view{sequence_index}",
                        float(sequence_value),
                        epoch,
                    )
        primary_runtime_entry: Dict[str, Any] = {
            "metrics": fixed_runtime_metrics,
            "metric_sequences": fixed_runtime_metric_sequences,
            "generation_diagnostics": generation_diagnostics,
            "sample_rgb": fixed_sample_rgb,
            "sample_debug_tensors": sample_debug_tensors,
            "source": fixed_visualization_source,
            "visualization_batch": visualization_batch,
        }
        self._persist_fixed_runtime_artifacts(
            stage_tag=stage_tag,
            epoch=epoch,
            sample_index=0,
            fixed_visualization_source=fixed_visualization_source,
            fixed_runtime_metrics=fixed_runtime_metrics,
            fixed_runtime_metric_sequences=fixed_runtime_metric_sequences,
            generation_diagnostics=generation_diagnostics,
            sample_rgb=fixed_sample_rgb,
            sample_debug_tensors=sample_debug_tensors,
            target_rgb=fixed_target_rgb,
        )
        if len(fixed_visualization_batches) > 1:
            group_runtime_entries: List[Dict[str, Any]] = [primary_runtime_entry]
            summary_metric_names = [
                "fixed_sample_psnr",
                "fixed_runtime_support_projection_coverage",
                "fixed_runtime_masked_inpaint_editable_coverage",
                "fixed_runtime_novel_coverage",
                "fixed_runtime_warp_support_projection_psnr",
                "fixed_runtime_pre_compose_teacher_support_projection_psnr",
                "fixed_runtime_post_compose_teacher_support_projection_psnr",
                "fixed_runtime_warp_masked_inpaint_editable_psnr",
                "fixed_runtime_pre_compose_teacher_masked_inpaint_editable_psnr",
                "fixed_runtime_post_compose_teacher_masked_inpaint_editable_psnr",
                "fixed_runtime_warp_novel_psnr",
                "fixed_runtime_pre_compose_teacher_novel_psnr",
                "fixed_runtime_post_compose_teacher_novel_psnr",
                "fixed_runtime_compose_gain_vs_pre_support_projection_psnr",
                "fixed_runtime_compose_gain_vs_pre_masked_inpaint_editable_psnr",
                "fixed_runtime_compose_gain_vs_pre_novel_psnr",
            ]
            for sample_group_index, (
                group_visualization_batch,
                group_visualization_source,
            ) in enumerate(fixed_visualization_batches[1:], start=1):
                group_runtime_entry = self._collect_fixed_visualization_group_metrics(
                    visualization_batch=group_visualization_batch,
                    fixed_visualization_source=group_visualization_source,
                    base_steps=base_steps,
                    base_weight=base_weight,
                )
                if group_runtime_entry is None:
                    continue
                group_runtime_entries.append(group_runtime_entry)
                group_metric_segments: List[str] = []
                for metric_name in summary_metric_names:
                    if metric_name not in group_runtime_entry["metrics"]:
                        continue
                    metric_value = float(group_runtime_entry["metrics"][metric_name])
                    metric_label = metric_name.replace("fixed_runtime_", "")
                    if "psnr" in metric_name:
                        group_metric_segments.append(f"{metric_label}={metric_value:.2f}dB")
                    else:
                        group_metric_segments.append(f"{metric_label}={metric_value:.4f}")
                group_metric_segments.append(f"source={group_visualization_source}")
                print(
                    f"[{stage_label}] Epoch {epoch} fixed-sample[{sample_group_index}] runtime metrics: "
                    + ", ".join(group_metric_segments)
                )
                if self.writer is not None:
                    for metric_name, metric_value in group_runtime_entry["metrics"].items():
                        self.writer.add_scalar(
                            f"{stage_tag}/fixed_runtime_sample{sample_group_index}/{metric_name}",
                            float(metric_value),
                            epoch,
                        )
                if self.stage_samples_dir is not None:
                    self._persist_fixed_runtime_artifacts(
                        stage_tag=stage_tag,
                        epoch=epoch,
                        sample_index=sample_group_index,
                        fixed_visualization_source=group_visualization_source,
                        fixed_runtime_metrics=group_runtime_entry["metrics"],
                        fixed_runtime_metric_sequences=group_runtime_entry.get(
                            "metric_sequences",
                            {},
                        ),
                        generation_diagnostics=group_runtime_entry.get(
                            "generation_diagnostics",
                            {},
                        ),
                        sample_rgb=group_runtime_entry["sample_rgb"],
                        sample_debug_tensors=group_runtime_entry["sample_debug_tensors"],
                        target_rgb=group_runtime_entry.get("visualization_batch", {}).get(
                            "target_rgb"
                        ),
                    )
            if len(group_runtime_entries) > 1:
                summary_segments = [f"sample_count={len(group_runtime_entries)}"]
                for metric_name in summary_metric_names:
                    metric_values = [
                        float(group_runtime_entry["metrics"][metric_name])
                        for group_runtime_entry in group_runtime_entries
                        if metric_name in group_runtime_entry["metrics"]
                    ]
                    if not metric_values:
                        continue
                    metric_tensor = torch.tensor(metric_values, dtype=torch.float32)
                    summary_segments.append(
                        f"{metric_name}="
                        f"{metric_tensor.mean().item():.4f}"
                        f"+/-{metric_tensor.std(unbiased=False).item():.4f}"
                    )
                    if self.writer is not None:
                        self.writer.add_scalar(
                            f"{stage_tag}/fixed_runtime_group_mean_{metric_name}",
                            float(metric_tensor.mean().item()),
                            epoch,
                        )
                        self.writer.add_scalar(
                            f"{stage_tag}/fixed_runtime_group_std_{metric_name}",
                            float(metric_tensor.std(unbiased=False).item()),
                            epoch,
                        )
                summary_segments.append(
                    "sources=["
                    + ", ".join(group_runtime_entry["source"] for group_runtime_entry in group_runtime_entries)
                    + "]"
                )
                print(
                    f"[{stage_label}] Epoch {epoch} fixed-sample-group summary: "
                    + ", ".join(summary_segments)
                )
        return fixed_runtime_metrics

    def _compute_final_sample_guided_reconstruction_loss(
        self,
        *,
        prediction_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        base_rgb: Optional[torch.Tensor],
        final_sample_rgb: torch.Tensor,
        support_mask: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        novel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        prediction_device = prediction_rgb.device
        prediction_dtype = prediction_rgb.dtype
        target_rgb = target_rgb.to(device=prediction_device, dtype=prediction_dtype)
        if base_rgb is not None:
            base_rgb = base_rgb.to(device=prediction_device, dtype=prediction_dtype)
        final_sample_rgb = final_sample_rgb.to(device=prediction_device, dtype=prediction_dtype)
        if support_mask is not None:
            support_mask = support_mask.to(device=prediction_device)
        if support_confidence is not None:
            support_confidence = support_confidence.to(device=prediction_device)
        if novel_mask is not None:
            novel_mask = novel_mask.to(device=prediction_device)
        target_hw = target_rgb.shape[-2:]
        if prediction_rgb.shape[-2:] != target_hw:
            prediction_rgb = F.interpolate(
                prediction_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if final_sample_rgb.shape[-2:] != target_hw:
            final_sample_rgb = F.interpolate(
                final_sample_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if base_rgb is not None and base_rgb.shape[-2:] != target_hw:
            base_rgb = F.interpolate(
                base_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        _, projection_aligned_novel_mask = self._build_projection_aligned_exist_novel_masks(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        editable_region_mask = novel_mask
        if editable_region_mask is None:
            editable_region_mask = projection_aligned_novel_mask
        elif projection_aligned_novel_mask is not None:
            editable_region_mask = torch.clamp(
                editable_region_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
                + projection_aligned_novel_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        low_confidence_editable_mask = self._build_low_confidence_editable_mask(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        if editable_region_mask is None:
            editable_region_mask = low_confidence_editable_mask
        elif low_confidence_editable_mask is not None:
            editable_region_mask = torch.clamp(
                editable_region_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
                + low_confidence_editable_mask.to(device=prediction_rgb.device, dtype=prediction_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        if editable_region_mask is None:
            editable_region_mask = prediction_rgb.new_zeros(
                (prediction_rgb.shape[0], 1, target_hw[0], target_hw[1])
            )
        else:
            editable_region_mask = editable_region_mask.to(
                device=prediction_rgb.device,
                dtype=prediction_rgb.dtype,
            )
        final_sample_error_map = (final_sample_rgb - target_rgb).abs().mean(dim=1, keepdim=True)
        focused_l1 = prediction_rgb.new_zeros((prediction_rgb.shape[0],))
        if torch.any(editable_region_mask > 0):
            masked_error_denominator = editable_region_mask.sum(
                dim=(1, 2, 3), keepdim=True
            ).clamp_min(1.0)
            normalized_error_weight = (
                editable_region_mask * final_sample_error_map
            ).sum(dim=(1, 2, 3), keepdim=True) / masked_error_denominator
            normalized_error_weight = (
                1.0 + final_sample_error_map / normalized_error_weight.clamp_min(1e-4)
            )
            focused_weight_map = editable_region_mask * normalized_error_weight
            focused_l1 = (
                (prediction_rgb - target_rgb).abs() * focused_weight_map
            ).sum(dim=(1, 2, 3)) / (
                focused_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
                * prediction_rgb.shape[1]
            )
        support_fidelity_weight = max(
            float(getattr(self, "final_sample_support_fidelity_weight", 0.0)),
            0.0,
        )
        support_consistency_weight = max(
            float(getattr(self, "final_sample_support_consistency_weight", 0.0)),
            0.0,
        )
        support_confidence_boost = max(
            float(getattr(self, "final_sample_support_confidence_boost", 0.0)),
            0.0,
        )
        support_low_confidence_anchor_weight = max(
            float(
                getattr(
                    self,
                    "final_sample_support_low_confidence_anchor_weight",
                    0.0,
                )
            ),
            0.0,
        )
        if (
            support_fidelity_weight > 0.0
            or support_consistency_weight > 0.0
            or support_low_confidence_anchor_weight > 0.0
        ):
            refined_support_signals = self._build_sampler_aligned_support_signals(
                support_mask=support_mask,
                support_confidence=support_confidence,
                target_hw=target_hw,
            )
            sampler_aligned_valid_mask = refined_support_signals.get("warp_valid_mask")
            support_projection_mask = refined_support_signals.get("support_projection_mask")
            refined_support_confidence = refined_support_signals.get("support_confidence")
            editable_low_confidence_mask = refined_support_signals.get(
                "editable_low_confidence_mask"
            )
            if support_projection_mask is not None:
                support_projection_mask = torch.nan_to_num(
                    support_projection_mask.to(
                        device=prediction_rgb.device,
                        dtype=prediction_rgb.dtype,
                    ),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                support_projection_mask = support_projection_mask * torch.clamp(
                    1.0 - editable_region_mask,
                    min=0.0,
                    max=1.0,
                )
                support_error_denominator = support_projection_mask.sum(
                    dim=(1, 2, 3), keepdim=True
                ).clamp_min(1.0)
                normalized_support_error = (
                    (support_projection_mask * final_sample_error_map).sum(
                        dim=(1, 2, 3), keepdim=True
                    )
                    / support_error_denominator
                )
                support_weight_map = support_projection_mask * (
                    1.0
                    + final_sample_error_map
                    / normalized_support_error.clamp_min(1e-4)
                )
                if refined_support_confidence is not None and support_confidence_boost > 0.0:
                    refined_support_confidence = torch.nan_to_num(
                        refined_support_confidence.to(
                            device=prediction_rgb.device,
                            dtype=prediction_rgb.dtype,
                        ),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).clamp(0.0, 1.0)
                    confidence_vulnerability_weight = 1.0 + support_confidence_boost * torch.clamp(
                        1.0 - refined_support_confidence,
                        min=0.0,
                        max=1.0,
                    )
                    support_weight_map = support_weight_map * confidence_vulnerability_weight
                support_fidelity_l1 = (
                    (prediction_rgb - target_rgb).abs() * support_weight_map
                ).sum(dim=(1, 2, 3)) / (
                    support_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
                    * prediction_rgb.shape[1]
                )
                focused_l1 = focused_l1 + support_fidelity_weight * support_fidelity_l1
                if support_consistency_weight > 0.0 and base_rgb is not None:
                    support_reference_weight_map = support_projection_mask
                    if sampler_aligned_valid_mask is not None:
                        sampler_aligned_valid_mask = torch.nan_to_num(
                            sampler_aligned_valid_mask.to(
                                device=prediction_rgb.device,
                                dtype=prediction_rgb.dtype,
                            ),
                            nan=0.0,
                            posinf=1.0,
                            neginf=0.0,
                        ).clamp(0.0, 1.0)
                        support_reference_weight_map = (
                            support_reference_weight_map * sampler_aligned_valid_mask
                        )
                    support_reference_l1 = (
                        (prediction_rgb - base_rgb).abs() * support_reference_weight_map
                    ).sum(dim=(1, 2, 3)) / (
                        support_reference_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
                        * prediction_rgb.shape[1]
                    )
                    focused_l1 = focused_l1 + (
                        support_consistency_weight * support_reference_l1
                    )
            if (
                editable_low_confidence_mask is not None
                and support_low_confidence_anchor_weight > 0.0
            ):
                editable_low_confidence_mask = torch.nan_to_num(
                    editable_low_confidence_mask.to(
                        device=prediction_rgb.device,
                        dtype=prediction_rgb.dtype,
                    ),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                editable_low_confidence_mask = editable_low_confidence_mask * torch.clamp(
                    1.0 - editable_region_mask,
                    min=0.0,
                    max=1.0,
                )
                if torch.any(editable_low_confidence_mask > 0):
                    low_confidence_error_denominator = editable_low_confidence_mask.sum(
                        dim=(1, 2, 3),
                        keepdim=True,
                    ).clamp_min(1.0)
                    normalized_low_confidence_error = (
                        editable_low_confidence_mask * final_sample_error_map
                    ).sum(dim=(1, 2, 3), keepdim=True) / low_confidence_error_denominator
                    low_confidence_weight_map = editable_low_confidence_mask * (
                        1.0
                        + final_sample_error_map
                        / normalized_low_confidence_error.clamp_min(1e-4)
                    )
                    low_confidence_anchor_l1 = (
                        (prediction_rgb - target_rgb).abs() * low_confidence_weight_map
                    ).sum(dim=(1, 2, 3)) / (
                        low_confidence_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0)
                        * prediction_rgb.shape[1]
                    )
                    focused_l1 = focused_l1 + (
                        support_low_confidence_anchor_weight * low_confidence_anchor_l1
                    )
        return focused_l1.mean()

    def _estimate_train_support_coverage(self, max_batches: int = 2) -> Optional[float]:
        coverage_values: List[float] = []
        for batch_index, batch in enumerate(self.train_dataloader):
            if batch_index >= max_batches:
                break
            support_mask = batch.get("warp_valid_mask")
            if not isinstance(support_mask, torch.Tensor):
                continue
            refined_support_signals = self._build_sampler_aligned_support_signals(
                support_mask=support_mask,
                support_confidence=batch.get("warp_support_confidence"),
                target_hw=support_mask.shape[-2:],
            )
            refined_support_projection_mask = refined_support_signals.get("support_projection_mask")
            if refined_support_projection_mask is None:
                continue
            coverage_values.append(
                float(torch.nan_to_num(refined_support_projection_mask, nan=0.0).mean().item())
            )
        if not coverage_values:
            return None
        return float(sum(coverage_values) / len(coverage_values))

    def _compute_teacher_effect_targets(
        self,
        *,
        teacher_rgb: torch.Tensor,
        render_rgb: Optional[torch.Tensor],
        target_rgb: torch.Tensor,
        depth_render: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor] = None,
        support_confidence: Optional[torch.Tensor] = None,
        frontier_mask: Optional[torch.Tensor] = None,
        verification_prior: Optional[torch.Tensor] = None,
        proposal_residual_rgb: Optional[torch.Tensor] = None,
        proposal_confidence: Optional[torch.Tensor] = None,
        proposal_warp_error_logit: Optional[torch.Tensor] = None,
        proposal_repairability_logit: Optional[torch.Tensor] = None,
        proposal_verification_logit: Optional[torch.Tensor] = None,
        proposal_acceptance_logit: Optional[torch.Tensor] = None,
        proposal_warp_error_confidence: Optional[torch.Tensor] = None,
        proposal_repairability_confidence: Optional[torch.Tensor] = None,
        proposal_verification_confidence: Optional[torch.Tensor] = None,
        proposal_acceptance_confidence: Optional[torch.Tensor] = None,
        empirical_candidate_rgbs: Optional[Sequence[torch.Tensor]] = None,
        empirical_candidate_residuals: Optional[Sequence[torch.Tensor]] = None,
        empirical_candidate_rgb_sources: Optional[Sequence[str]] = None,
        empirical_candidate_residual_sources: Optional[Sequence[str]] = None,
        use_projection_aligned_support: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        target_hw = target_rgb.shape[-2:]

        def _prepare_effect_mask(
            mask_tensor: Optional[torch.Tensor],
            *,
            interpolation_mode: str = "bilinear",
            preserve_empty: bool = False,
        ) -> Optional[torch.Tensor]:
            if not isinstance(mask_tensor, torch.Tensor):
                return None
            prepared_mask = mask_tensor
            if prepared_mask.dim() == 3:
                prepared_mask = prepared_mask.unsqueeze(1)
            if prepared_mask.shape[1] > 1:
                prepared_mask = prepared_mask[:, :1]
            prepared_mask = torch.nan_to_num(
                prepared_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if prepared_mask.shape[-2:] != target_hw:
                interpolate_kwargs = {"size": target_hw, "mode": interpolation_mode}
                if interpolation_mode != "nearest":
                    interpolate_kwargs["align_corners"] = False
                prepared_mask = F.interpolate(prepared_mask, **interpolate_kwargs)
            if not preserve_empty and float(prepared_mask.sum().item()) < 1.0:
                return None
            return prepared_mask.clamp(0.0, 1.0)

        def _prepare_logit_tensor(
            logit_tensor: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if not isinstance(logit_tensor, torch.Tensor):
                return None
            prepared_logit = logit_tensor
            if prepared_logit.dim() == 3:
                prepared_logit = prepared_logit.unsqueeze(1)
            if prepared_logit.shape[1] > 1:
                prepared_logit = prepared_logit[:, :1]
            prepared_logit = torch.nan_to_num(
                prepared_logit.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ).clamp(-20.0, 20.0)
            if prepared_logit.shape[-2:] != target_hw:
                prepared_logit = F.interpolate(
                    prepared_logit,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_logit

        def _prepare_rgb_like_tensor(
            rgb_like_tensor: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if not isinstance(rgb_like_tensor, torch.Tensor):
                return None
            prepared_rgb_like_tensor = rgb_like_tensor
            if prepared_rgb_like_tensor.dim() == 3:
                prepared_rgb_like_tensor = prepared_rgb_like_tensor.unsqueeze(0)
            if prepared_rgb_like_tensor.shape[1] > 3:
                prepared_rgb_like_tensor = prepared_rgb_like_tensor[:, :3]
            prepared_rgb_like_tensor = torch.nan_to_num(
                prepared_rgb_like_tensor.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if prepared_rgb_like_tensor.shape[-2:] != target_hw:
                prepared_rgb_like_tensor = F.interpolate(
                    prepared_rgb_like_tensor,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_rgb_like_tensor

        if teacher_rgb.shape[-2:] != target_hw:
            teacher_rgb = F.interpolate(
                teacher_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        if render_rgb is None:
            render_rgb = torch.zeros_like(target_rgb)
        elif render_rgb.shape[-2:] != target_hw:
            render_rgb = F.interpolate(
                render_rgb,
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )

        teacher_error = (teacher_rgb - target_rgb).abs().mean(dim=1, keepdim=True)
        render_error = (render_rgb - target_rgb).abs().mean(dim=1, keepdim=True)
        if use_projection_aligned_support:
            exist_mask, novel_mask = self._build_projection_aligned_exist_novel_masks(
                support_mask=support_mask,
                support_confidence=support_confidence,
                target_hw=target_hw,
            )
        else:
            exist_mask, novel_mask = self._build_exist_novel_masks(
                support_mask,
                depth_render,
                target_hw,
            )
        editable_mask = novel_mask
        low_confidence_editable_mask = self._build_low_confidence_editable_mask(
            support_mask=support_mask,
            support_confidence=support_confidence,
            target_hw=target_hw,
        )
        if editable_mask is None:
            editable_mask = low_confidence_editable_mask
        elif low_confidence_editable_mask is not None:
            editable_mask = torch.clamp(
                editable_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                + low_confidence_editable_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        prepared_frontier_mask = _prepare_effect_mask(
            frontier_mask,
            interpolation_mode="nearest",
        )
        if prepared_frontier_mask is not None:
            if editable_mask is None:
                editable_mask = prepared_frontier_mask
            else:
                editable_mask = torch.clamp(
                    editable_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                    + prepared_frontier_mask,
                    min=0.0,
                    max=1.0,
                )
            if novel_mask is None:
                novel_mask = prepared_frontier_mask
        prepared_verification_prior = _prepare_effect_mask(
            verification_prior,
            interpolation_mode="bilinear",
            preserve_empty=True,
        )
        prepared_proposal_residual_rgb = _prepare_rgb_like_tensor(proposal_residual_rgb)
        prepared_proposal_confidence = _prepare_effect_mask(
            proposal_confidence,
            interpolation_mode="bilinear",
        )
        prepared_proposal_warp_error_logit = _prepare_logit_tensor(
            proposal_warp_error_logit
        )
        prepared_proposal_repairability_logit = _prepare_logit_tensor(
            proposal_repairability_logit
        )
        prepared_proposal_verification_logit = _prepare_logit_tensor(
            proposal_verification_logit
        )
        prepared_proposal_acceptance_logit = _prepare_logit_tensor(
            proposal_acceptance_logit
        )
        prepared_proposal_warp_error_confidence = _prepare_effect_mask(
            proposal_warp_error_confidence,
            interpolation_mode="bilinear",
        )
        prepared_proposal_repairability_confidence = _prepare_effect_mask(
            proposal_repairability_confidence,
            interpolation_mode="bilinear",
        )
        prepared_proposal_verification_confidence = _prepare_effect_mask(
            proposal_verification_confidence,
            interpolation_mode="bilinear",
        )
        prepared_proposal_acceptance_confidence = _prepare_effect_mask(
            proposal_acceptance_confidence,
            interpolation_mode="bilinear",
        )

        proposal_training_mask = editable_mask
        if proposal_training_mask is None:
            proposal_training_mask = novel_mask
        proposal_opportunity_threshold = max(
            float(getattr(self, "proposal_opportunity_error_threshold", 0.0)),
            0.0,
        )
        if proposal_opportunity_threshold > 0.0:
            proposal_opportunity_mask = (
                render_error.detach() > proposal_opportunity_threshold
            ).to(dtype=teacher_rgb.dtype)
        else:
            proposal_opportunity_mask = torch.ones_like(render_error)
        proposal_opportunity_top_fraction = min(
            max(float(getattr(self, "proposal_opportunity_top_fraction", 1.0)), 0.0),
            1.0,
        )
        if proposal_opportunity_top_fraction > 0.0 and proposal_opportunity_top_fraction < 1.0:
            base_opportunity_mask = (
                proposal_training_mask
                if proposal_training_mask is not None
                else torch.ones_like(render_error)
            )
            top_error_mask = torch.zeros_like(render_error)
            detached_render_error = render_error.detach().float()
            detached_base_mask = (
                base_opportunity_mask.detach().to(device=render_error.device).float() > 1e-6
            )
            for batch_index in range(detached_render_error.shape[0]):
                valid_error_values = detached_render_error[batch_index][
                    detached_base_mask[batch_index]
                ]
                if valid_error_values.numel() < 1:
                    continue
                quantile_threshold = torch.quantile(
                    valid_error_values,
                    max(0.0, 1.0 - proposal_opportunity_top_fraction),
                )
                top_error_mask[batch_index] = (
                    detached_render_error[batch_index] >= quantile_threshold
                ).to(dtype=teacher_rgb.dtype)
            top_error_mask = top_error_mask * base_opportunity_mask.to(
                device=teacher_rgb.device,
                dtype=teacher_rgb.dtype,
            )
            combined_opportunity_mask = torch.clamp(
                proposal_opportunity_mask * top_error_mask,
                min=0.0,
                max=1.0,
            )
            if float(combined_opportunity_mask.sum().item()) >= 1.0:
                proposal_opportunity_mask = combined_opportunity_mask
            elif float(top_error_mask.sum().item()) >= 1.0:
                proposal_opportunity_mask = top_error_mask
        if proposal_training_mask is None:
            proposal_training_mask = proposal_opportunity_mask
        else:
            proposal_training_mask = torch.clamp(
                proposal_training_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                * proposal_opportunity_mask,
                min=0.0,
                max=1.0,
            )
        if float(proposal_training_mask.sum().item()) < 1.0:
            proposal_training_mask = None

        proposal_warp_error_target = torch.zeros_like(render_error)
        proposal_warp_error_rank_target = torch.zeros_like(render_error)
        if proposal_training_mask is not None:
            repair_prior_quantile = min(
                max(float(getattr(self, "proposal_repair_prior_quantile", 0.90)), 0.50),
                0.99,
            )
            detached_render_error = render_error.detach().float()
            detached_proposal_mask = (
                proposal_training_mask.detach().to(device=render_error.device).float() > 1e-6
            )
            for batch_index in range(detached_render_error.shape[0]):
                valid_error_values = detached_render_error[batch_index][
                    detached_proposal_mask[batch_index]
                ]
                if valid_error_values.numel() < 1:
                    continue
                robust_error_scale = torch.quantile(
                    valid_error_values,
                    repair_prior_quantile,
                ).clamp_min(1e-4)
                proposal_warp_error_target[batch_index] = torch.clamp(
                    detached_render_error[batch_index] / robust_error_scale,
                    min=0.0,
                    max=1.0,
                ).to(dtype=teacher_rgb.dtype)
            proposal_warp_error_target = torch.clamp(
                proposal_warp_error_target
                * proposal_training_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                min=0.0,
                max=1.0,
            )
            rank_blend_weight = min(
                max(float(getattr(self, "proposal_warp_error_rank_blend", 0.50)), 0.0),
                1.0,
            )
            if rank_blend_weight > 0.0:
                detached_render_error_flat = detached_render_error.view(
                    detached_render_error.shape[0],
                    -1,
                )
                detached_proposal_mask_flat = detached_proposal_mask.view(
                    detached_proposal_mask.shape[0],
                    -1,
                )
                rank_target_flat = torch.zeros_like(detached_render_error_flat)
                for batch_index in range(detached_render_error_flat.shape[0]):
                    valid_error_values = detached_render_error_flat[batch_index][
                        detached_proposal_mask_flat[batch_index]
                    ]
                    if valid_error_values.numel() < 1:
                        continue
                    if valid_error_values.numel() == 1:
                        valid_ranks = torch.ones_like(valid_error_values)
                    else:
                        sorted_errors = torch.sort(valid_error_values).values
                        lower_ranks = torch.searchsorted(
                            sorted_errors, valid_error_values, right=False
                        )
                        upper_ranks = torch.searchsorted(
                            sorted_errors, valid_error_values, right=True
                        )
                        valid_ranks = (lower_ranks + upper_ranks - 1).to(
                            dtype=valid_error_values.dtype
                        ) / (2 * (valid_error_values.numel() - 1))
                    valid_ranks = torch.where(
                        valid_error_values > 0.0,
                        valid_ranks,
                        torch.zeros_like(valid_ranks),
                    )
                    rank_target_flat[batch_index][
                        detached_proposal_mask_flat[batch_index]
                    ] = valid_ranks
                proposal_warp_error_rank_target = rank_target_flat.view_as(
                    detached_render_error
                ).to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                proposal_warp_error_target = torch.clamp(
                    proposal_warp_error_target * (1.0 - rank_blend_weight)
                    + proposal_warp_error_rank_target * rank_blend_weight,
                    min=0.0,
                    max=1.0,
                )
        if isinstance(prepared_verification_prior, torch.Tensor):
            proposal_verification_target = prepared_verification_prior.to(
                device=teacher_rgb.device,
                dtype=teacher_rgb.dtype,
            )
        else:
            verification_signal_terms: List[torch.Tensor] = []
            prepared_support_confidence = _prepare_effect_mask(
                support_confidence,
                interpolation_mode="bilinear",
            )
            prepared_support_mask = _prepare_effect_mask(
                support_mask,
                interpolation_mode="nearest",
            )
            if isinstance(prepared_support_confidence, torch.Tensor):
                verification_signal_terms.append(prepared_support_confidence)
            if isinstance(prepared_support_mask, torch.Tensor):
                verification_signal_terms.append(prepared_support_mask)
            if (
                isinstance(prepared_frontier_mask, torch.Tensor)
                and isinstance(prepared_support_confidence, torch.Tensor)
            ):
                verification_signal_terms.append(
                    prepared_frontier_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                    * prepared_support_confidence.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                )
            if verification_signal_terms:
                proposal_verification_target = torch.stack(
                    [
                        verification_signal.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                        for verification_signal in verification_signal_terms
                    ],
                    dim=0,
                ).amax(dim=0)
            else:
                proposal_verification_target = torch.zeros_like(render_error)
            verification_min_support = min(
                max(float(getattr(self, "proposal_verification_min_support", 0.05)), 0.0),
                1.0,
            )
            if verification_min_support > 0.0:
                proposal_verification_target = torch.where(
                    proposal_verification_target >= verification_min_support,
                    proposal_verification_target,
                    torch.zeros_like(proposal_verification_target),
                )
        proxy_repairability_target = torch.clamp(
            torch.sqrt(
                torch.clamp(proposal_warp_error_target, min=0.0, max=1.0)
                * torch.clamp(proposal_verification_target, min=0.0, max=1.0)
            ),
            min=0.0,
            max=1.0,
        )

        def _patch_average_tensor(
            value_tensor: torch.Tensor,
            mask_tensor: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            empirical_patch_kernel_size = max(
                int(getattr(self, "proposal_empirical_patch_kernel_size", 15)),
                1,
            )
            if empirical_patch_kernel_size % 2 == 0:
                empirical_patch_kernel_size += 1
            if empirical_patch_kernel_size <= 1:
                return value_tensor
            value_float = value_tensor.detach().float()
            if mask_tensor is None:
                return F.avg_pool2d(
                    value_float,
                    kernel_size=empirical_patch_kernel_size,
                    stride=1,
                    padding=empirical_patch_kernel_size // 2,
                ).to(device=value_tensor.device, dtype=value_tensor.dtype)
            aligned_mask = mask_tensor.detach().to(
                device=value_tensor.device,
                dtype=torch.float32,
            )
            if aligned_mask.shape[-2:] != value_tensor.shape[-2:]:
                aligned_mask = F.interpolate(
                    aligned_mask,
                    size=value_tensor.shape[-2:],
                    mode="nearest",
                )
            if aligned_mask.shape[1] > 1:
                aligned_mask = aligned_mask[:, :1]
            pooled_weight = F.avg_pool2d(
                aligned_mask,
                kernel_size=empirical_patch_kernel_size,
                stride=1,
                padding=empirical_patch_kernel_size // 2,
            ).clamp_min(1e-6)
            pooled_value = F.avg_pool2d(
                value_float * aligned_mask,
                kernel_size=empirical_patch_kernel_size,
                stride=1,
                padding=empirical_patch_kernel_size // 2,
            ) / pooled_weight
            return pooled_value.to(device=value_tensor.device, dtype=value_tensor.dtype)

        def _prepare_candidate_rgb(
            candidate_rgb: torch.Tensor,
        ) -> torch.Tensor:
            prepared_candidate_rgb = candidate_rgb
            if prepared_candidate_rgb.dim() == 3:
                prepared_candidate_rgb = prepared_candidate_rgb.unsqueeze(0)
            if prepared_candidate_rgb.shape[1] > 3:
                prepared_candidate_rgb = prepared_candidate_rgb[:, :3]
            prepared_candidate_rgb = torch.nan_to_num(
                prepared_candidate_rgb.to(device=target_rgb.device, dtype=target_rgb.dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            if prepared_candidate_rgb.shape[-2:] != target_hw:
                prepared_candidate_rgb = F.interpolate(
                    prepared_candidate_rgb,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_candidate_rgb

        def _candidate_error_from_prepared_rgb(
            prepared_candidate_rgb: torch.Tensor,
        ) -> torch.Tensor:
            return (prepared_candidate_rgb.detach() - target_rgb.detach()).abs().mean(
                dim=1,
                keepdim=True,
            )

        def _candidate_error_from_rgb(
            candidate_rgb: torch.Tensor,
        ) -> torch.Tensor:
            return _candidate_error_from_prepared_rgb(_prepare_candidate_rgb(candidate_rgb))

        def _prepare_candidate_residual(
            candidate_residual: torch.Tensor,
        ) -> torch.Tensor:
            prepared_candidate_residual = candidate_residual
            if prepared_candidate_residual.dim() == 3:
                prepared_candidate_residual = prepared_candidate_residual.unsqueeze(0)
            if prepared_candidate_residual.shape[1] > 3:
                prepared_candidate_residual = prepared_candidate_residual[:, :3]
            prepared_candidate_residual = torch.nan_to_num(
                prepared_candidate_residual.to(
                    device=render_rgb.device,
                    dtype=render_rgb.dtype,
                ),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if prepared_candidate_residual.shape[-2:] != target_hw:
                prepared_candidate_residual = F.interpolate(
                    prepared_candidate_residual,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return prepared_candidate_residual

        def _candidate_error_from_prepared_residual(
            prepared_candidate_residual: torch.Tensor,
        ) -> torch.Tensor:
            candidate_rgb_from_residual = torch.clamp(
                render_rgb + prepared_candidate_residual,
                min=0.0,
                max=1.0,
            )
            return _candidate_error_from_prepared_rgb(candidate_rgb_from_residual)

        def _candidate_error_from_residual(
            candidate_residual: torch.Tensor,
        ) -> torch.Tensor:
            return _candidate_error_from_prepared_residual(
                _prepare_candidate_residual(candidate_residual)
            )

        def _candidate_residual_from_rgb(
            candidate_rgb: torch.Tensor,
        ) -> torch.Tensor:
            return _prepare_candidate_rgb(candidate_rgb) - render_rgb

        # 准入评价的是实际被消费的RGB；原始残差可能以warp为基底且尚未经过门控。
        current_candidate_residual = _candidate_residual_from_rgb(teacher_rgb)
        proposal_candidate_error = _candidate_error_from_rgb(teacher_rgb)
        patch_render_error = _patch_average_tensor(
            render_error,
            proposal_training_mask,
        )
        patch_current_candidate_error = _patch_average_tensor(
            proposal_candidate_error,
            proposal_training_mask,
        )
        patch_best_candidate_error = patch_current_candidate_error
        best_candidate_residual = current_candidate_residual.detach()
        best_candidate_source_index = torch.zeros_like(patch_best_candidate_error)
        empirical_candidate_count = 1
        if empirical_candidate_residuals:
            for residual_candidate_index, candidate_residual in enumerate(empirical_candidate_residuals):
                if not isinstance(candidate_residual, torch.Tensor):
                    continue
                current_source_index = float(empirical_candidate_count)
                prepared_candidate_residual = _prepare_candidate_residual(
                    candidate_residual
                )
                candidate_patch_error = _patch_average_tensor(
                    _candidate_error_from_prepared_residual(prepared_candidate_residual),
                    proposal_training_mask,
                )
                candidate_better_mask = (
                    candidate_patch_error < patch_best_candidate_error
                ).to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                best_candidate_residual = torch.where(
                    candidate_better_mask.expand_as(best_candidate_residual).bool(),
                    prepared_candidate_residual.detach(),
                    best_candidate_residual,
                )
                best_candidate_source_index = torch.where(
                    candidate_better_mask.bool(),
                    torch.full_like(best_candidate_source_index, current_source_index),
                    best_candidate_source_index,
                )
                patch_best_candidate_error = torch.minimum(
                    patch_best_candidate_error,
                    candidate_patch_error,
                )
                empirical_candidate_count += 1
        if empirical_candidate_rgbs:
            for rgb_candidate_index, candidate_rgb in enumerate(empirical_candidate_rgbs):
                if not isinstance(candidate_rgb, torch.Tensor):
                    continue
                current_source_index = float(empirical_candidate_count)
                prepared_candidate_rgb = _prepare_candidate_rgb(candidate_rgb)
                prepared_candidate_residual = prepared_candidate_rgb - render_rgb
                candidate_patch_error = _patch_average_tensor(
                    _candidate_error_from_prepared_rgb(prepared_candidate_rgb),
                    proposal_training_mask,
                )
                candidate_better_mask = (
                    candidate_patch_error < patch_best_candidate_error
                ).to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
                best_candidate_residual = torch.where(
                    candidate_better_mask.expand_as(best_candidate_residual).bool(),
                    prepared_candidate_residual.detach(),
                    best_candidate_residual,
                )
                best_candidate_source_index = torch.where(
                    candidate_better_mask.bool(),
                    torch.full_like(best_candidate_source_index, current_source_index),
                    best_candidate_source_index,
                )
                patch_best_candidate_error = torch.minimum(
                    patch_best_candidate_error,
                    candidate_patch_error,
                )
                empirical_candidate_count += 1
        best_candidate_rgb = torch.clamp(
            render_rgb + best_candidate_residual,
            min=0.0,
            max=1.0,
        )
        patch_verification_target = _patch_average_tensor(
            proposal_verification_target,
            proposal_training_mask,
        )
        current_improvement_target = torch.relu(
            patch_render_error
            - patch_current_candidate_error
            - max(float(getattr(self, "teacher_warp_margin", 0.0)), 0.0)
        )
        empirical_improvement_target = torch.relu(
            patch_render_error
            - patch_best_candidate_error
            - max(float(getattr(self, "teacher_warp_margin", 0.0)), 0.0)
        )
        empirical_repairability_target = torch.clamp(
            empirical_improvement_target / patch_render_error.detach().clamp_min(1e-4),
            min=0.0,
            max=1.0,
        )
        current_acceptance_target = torch.clamp(
            current_improvement_target / patch_render_error.detach().clamp_min(1e-4),
            min=0.0,
            max=1.0,
        )
        empirical_repairability_target = torch.clamp(
            empirical_repairability_target
            * torch.clamp(patch_verification_target, min=0.0, max=1.0),
            min=0.0,
            max=1.0,
        )
        current_acceptance_target = torch.clamp(
            current_acceptance_target
            * torch.clamp(patch_verification_target, min=0.0, max=1.0),
            min=0.0,
            max=1.0,
        )
        if proposal_training_mask is not None:
            empirical_repairability_target = torch.clamp(
                empirical_repairability_target
                * proposal_training_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                min=0.0,
                max=1.0,
            )
            current_acceptance_target = torch.clamp(
                current_acceptance_target
                * proposal_training_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype),
                min=0.0,
                max=1.0,
            )
        empirical_max_weight = min(
            max(float(getattr(self, "proposal_repairability_empirical_max_weight", 0.0)), 0.0),
            1.0,
        )
        empirical_transition_epochs = max(
            int(getattr(self, "proposal_repairability_empirical_transition_epochs", 4)),
            1,
        )
        current_stage_epoch = int(getattr(self, "_current_proposal_effect_epoch", 0))
        empirical_blend_weight = empirical_max_weight * min(
            max(float(current_stage_epoch + 1) / float(empirical_transition_epochs), 0.0),
            1.0,
        )
        proposal_repairability_target = torch.clamp(
            proxy_repairability_target * (1.0 - empirical_blend_weight)
            + empirical_repairability_target * empirical_blend_weight,
            min=0.0,
            max=1.0,
        )
        proposal_residual_training_target = torch.clamp(
            proposal_repairability_target,
            min=0.0,
            max=1.0,
        )
        proposal_acceptance_target = torch.clamp(
            current_acceptance_target,
            min=0.0,
            max=1.0,
        )
        proposal_repair_prior = proposal_residual_training_target
        proposal_confidence_target = proposal_acceptance_target.detach()

        risk_exist_target = torch.relu(teacher_error - render_error)
        gain_novel_target = torch.relu(render_error - teacher_error)
        novel_underperform_target = torch.relu(teacher_error - render_error)
        if exist_mask is not None:
            risk_exist_target = risk_exist_target * exist_mask
        if novel_mask is not None:
            gain_novel_target = gain_novel_target * novel_mask
            novel_underperform_target = novel_underperform_target * novel_mask

        return {
            "teacher_rgb": teacher_rgb,
            "render_rgb": render_rgb,
            "target_rgb": target_rgb,
            "teacher_error": teacher_error,
            "render_error": render_error,
            "exist_mask": exist_mask,
            "novel_mask": novel_mask,
            "editable_mask": editable_mask,
            "frontier_mask": prepared_frontier_mask,
            "verification_prior": prepared_verification_prior,
            "proposal_training_mask": proposal_training_mask,
            "proposal_opportunity_mask": proposal_opportunity_mask,
            "proposal_repair_prior": torch.nan_to_num(
                proposal_repair_prior,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_candidate_error": torch.nan_to_num(
                proposal_candidate_error,
                nan=0.0,
                posinf=1.0,
                neginf=1.0,
            ),
            "proposal_best_candidate_patch_error": torch.nan_to_num(
                patch_best_candidate_error,
                nan=1.0,
                posinf=1.0,
                neginf=1.0,
            ),
            "proposal_best_candidate_rgb": torch.nan_to_num(
                best_candidate_rgb,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_best_candidate_residual_target": torch.nan_to_num(
                best_candidate_residual,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "proposal_best_candidate_source_index": torch.nan_to_num(
                best_candidate_source_index,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "proposal_empirical_improvement_target": torch.nan_to_num(
                empirical_improvement_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_current_acceptance_target": torch.nan_to_num(
                current_acceptance_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_residual_training_target": torch.nan_to_num(
                proposal_residual_training_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_empirical_candidate_count": torch.tensor(
                float(empirical_candidate_count),
                device=teacher_rgb.device,
                dtype=teacher_rgb.dtype,
            ),
            "proposal_residual_rgb": prepared_proposal_residual_rgb,
            "proposal_confidence": prepared_proposal_confidence,
            "proposal_warp_error_logit": prepared_proposal_warp_error_logit,
            "proposal_repairability_logit": prepared_proposal_repairability_logit,
            "proposal_verification_logit": prepared_proposal_verification_logit,
            "proposal_acceptance_logit": prepared_proposal_acceptance_logit,
            "proposal_warp_error_confidence": prepared_proposal_warp_error_confidence,
            "proposal_repairability_confidence": prepared_proposal_repairability_confidence,
            "proposal_verification_confidence": prepared_proposal_verification_confidence,
            "proposal_acceptance_confidence": prepared_proposal_acceptance_confidence,
            "proposal_warp_error_target": torch.nan_to_num(
                proposal_warp_error_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_warp_error_rank_target": torch.nan_to_num(
                proposal_warp_error_rank_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_oracle_residual_target": torch.nan_to_num(
                best_candidate_residual,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "proposal_true_residual_target": torch.nan_to_num(
                target_rgb - render_rgb,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            "proposal_repairability_target": torch.nan_to_num(
                proposal_repairability_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_verification_target": torch.nan_to_num(
                proposal_verification_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_acceptance_target": torch.nan_to_num(
                proposal_acceptance_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "proposal_confidence_target": torch.nan_to_num(
                proposal_confidence_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0),
            "risk_exist_target": torch.nan_to_num(
                risk_exist_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            "gain_novel_target": torch.nan_to_num(
                gain_novel_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
            "novel_underperform_target": torch.nan_to_num(
                novel_underperform_target,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ),
        }

    def _compute_frontier_residual_proposal_loss(
        self,
        effect_targets: Dict[str, Optional[torch.Tensor]],
        *,
        proposal_rgb: torch.Tensor,
    ) -> torch.Tensor:
        # 标签可来自停止梯度的采样参考；修复预测必须显式保留实际提议计算图。
        teacher_rgb = proposal_rgb
        render_rgb = effect_targets.get("render_rgb")
        target_rgb = effect_targets.get("target_rgb")
        if (
            teacher_rgb is None
            or render_rgb is None
            or target_rgb is None
        ):
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            return torch.zeros((), device=device)
        proposal_mask = effect_targets.get("proposal_training_mask")
        if proposal_mask is None:
            proposal_mask = effect_targets.get("editable_mask")
        if proposal_mask is None:
            proposal_mask = effect_targets.get("novel_mask")
        if proposal_mask is None:
            return teacher_rgb.new_zeros(())
        proposal_mask = proposal_mask.to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
        if float(proposal_mask.sum().item()) < 1.0:
            return teacher_rgb.new_zeros(())
        # 失败候选仍需要独立参考的纠错监督，不能因其当前可修复性低而退出。
        proposal_loss_mask = proposal_mask

        if teacher_rgb.shape[-2:] != target_rgb.shape[-2:]:
            teacher_rgb = F.interpolate(
                teacher_rgb,
                size=target_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if render_rgb.shape[-2:] != target_rgb.shape[-2:]:
            render_rgb = F.interpolate(
                render_rgb,
                size=target_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # raw/gate 的分解不唯一；当前候选 oracle 也可能只是输出自己的副本。
        # 直接监督已组装 RGB，独立训练参考不参与反向传播。
        target_rgb = target_rgb.detach().to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)
        render_rgb = render_rgb.detach().to(device=teacher_rgb.device, dtype=teacher_rgb.dtype)

        def _masked_rgb_reconstruction(
            prediction_rgb: torch.Tensor,
            reference_rgb: torch.Tensor,
            mask_rgb: torch.Tensor,
            recon_alpha: float,
        ) -> torch.Tensor:
            mask_rgb = mask_rgb.detach().to(device=prediction_rgb.device, dtype=prediction_rgb.dtype)
            mask_sum = mask_rgb.sum(dim=(1, 2, 3)).clamp_min(1.0) * prediction_rgb.shape[1]
            masked_l1 = ((prediction_rgb - reference_rgb).abs() * mask_rgb).sum(dim=(1, 2, 3)) / mask_sum
            masked_l2 = ((prediction_rgb - reference_rgb).pow(2) * mask_rgb).sum(dim=(1, 2, 3)) / mask_sum
            return recon_alpha * masked_l1.mean() + (1.0 - recon_alpha) * masked_l2.mean()

        resolved_frontier_residual_recon_alpha = min(
            max(float(getattr(self, "frontier_residual_recon_alpha", 0.75)), 0.0),
            1.0,
        )
        proposal_loss = _masked_rgb_reconstruction(
            teacher_rgb,
            target_rgb,
            proposal_loss_mask,
            resolved_frontier_residual_recon_alpha,
        )

        resolved_frontier_residual_anchor_weight = max(
            float(getattr(self, "frontier_residual_anchor_weight", 0.0)),
            0.0,
        )
        if resolved_frontier_residual_anchor_weight <= 0.0:
            return proposal_loss

        anchor_mask = torch.clamp(1.0 - proposal_mask, min=0.0, max=1.0)
        exist_mask = effect_targets.get("exist_mask")
        if isinstance(exist_mask, torch.Tensor):
            anchor_mask = anchor_mask * exist_mask.to(
                device=teacher_rgb.device, dtype=teacher_rgb.dtype,
            ).clamp(0.0, 1.0)
        if float(anchor_mask.sum().item()) < 1.0:
            return proposal_loss

        anchor_loss = _masked_rgb_reconstruction(
            teacher_rgb,
            render_rgb,
            anchor_mask,
            resolved_frontier_residual_recon_alpha,
        )
        return proposal_loss + anchor_loss * resolved_frontier_residual_anchor_weight

    def _compute_relative_teacher_effect_loss(
        self,
        effect_targets: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        calibration_terms: List[torch.Tensor] = []
        risk_exist_target = effect_targets.get("risk_exist_target")
        exist_mask = effect_targets.get("exist_mask")
        if risk_exist_target is not None and exist_mask is not None:
            calibration_terms.append(
                self._calibration_loss(
                    risk_exist_target,
                    torch.zeros_like(risk_exist_target),
                    exist_mask,
                )
            )
        novel_underperform_target = effect_targets.get("novel_underperform_target")
        novel_mask = effect_targets.get("novel_mask")
        if novel_underperform_target is not None and novel_mask is not None:
            calibration_terms.append(
                self._calibration_loss(
                    novel_underperform_target,
                    torch.zeros_like(novel_underperform_target),
                    novel_mask,
                )
            )
        calibration_loss = None
        if calibration_terms:
            calibration_loss = torch.stack(calibration_terms).mean()

        margin_terms: List[torch.Tensor] = []
        teacher_error = effect_targets.get("teacher_error")
        render_error = effect_targets.get("render_error")
        editable_mask = effect_targets.get("editable_mask")
        proposal_training_mask = effect_targets.get("proposal_training_mask")
        proposal_repair_prior = effect_targets.get("proposal_repair_prior")
        resolved_teacher_warp_margin = max(
            float(getattr(self, "teacher_warp_margin", 0.0)),
            0.0,
        )
        resolved_teacher_warp_margin_weight = max(
            float(getattr(self, "teacher_warp_margin_weight", 0.0)),
            0.0,
        )
        resolved_teacher_warp_editable_weight = max(
            float(getattr(self, "teacher_warp_editable_weight", 1.0)),
            0.0,
        )
        if (
            resolved_teacher_warp_margin_weight > 0.0
            and teacher_error is not None
            and render_error is not None
        ):
            teacher_underperform_margin = torch.relu(
                teacher_error - render_error + resolved_teacher_warp_margin
            )
            if isinstance(proposal_repair_prior, torch.Tensor):
                margin_terms.append(
                    self._calibration_loss(
                        teacher_underperform_margin,
                        torch.zeros_like(teacher_underperform_margin),
                        proposal_repair_prior,
                    )
                )
            elif proposal_training_mask is not None:
                margin_terms.append(
                    self._calibration_loss(
                        teacher_underperform_margin,
                        torch.zeros_like(teacher_underperform_margin),
                        proposal_training_mask,
                    )
                )
            elif novel_mask is not None:
                margin_terms.append(
                    self._calibration_loss(
                        teacher_underperform_margin,
                        torch.zeros_like(teacher_underperform_margin),
                        novel_mask,
                    )
                )
            elif editable_mask is not None and resolved_teacher_warp_editable_weight > 0.0:
                margin_terms.append(
                    self._calibration_loss(
                        teacher_underperform_margin,
                        torch.zeros_like(teacher_underperform_margin),
                        editable_mask,
                    )
                    * resolved_teacher_warp_editable_weight
                )

        selective_risk_loss = None
        resolved_selective_risk_weight = max(
            float(getattr(self, "proposal_selective_risk_weight", 0.0)),
            0.0,
        )
        proposal_acceptance_confidence = effect_targets.get("proposal_acceptance_confidence")
        if (
            resolved_selective_risk_weight > 0.0
            and isinstance(proposal_acceptance_confidence, torch.Tensor)
            and teacher_error is not None
            and render_error is not None
        ):
            aligned_acceptance_confidence = proposal_acceptance_confidence.to(
                device=teacher_error.device,
                dtype=teacher_error.dtype,
            )
            if aligned_acceptance_confidence.dim() == 3:
                aligned_acceptance_confidence = aligned_acceptance_confidence.unsqueeze(1)
            if aligned_acceptance_confidence.shape[1] > 1:
                aligned_acceptance_confidence = aligned_acceptance_confidence[:, :1]
            if aligned_acceptance_confidence.shape[-2:] != teacher_error.shape[-2:]:
                aligned_acceptance_confidence = F.interpolate(
                    aligned_acceptance_confidence,
                    size=teacher_error.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            accepted_underperform_risk = (
                aligned_acceptance_confidence.clamp(0.0, 1.0)
                * torch.relu(teacher_error - render_error + resolved_teacher_warp_margin)
            )
            selective_risk_mask = proposal_training_mask
            if selective_risk_mask is None:
                selective_risk_mask = editable_mask
            if selective_risk_mask is None:
                selective_risk_mask = novel_mask
            if selective_risk_mask is None:
                selective_risk_mask = torch.ones_like(accepted_underperform_risk)
            selective_risk_loss = (
                self._calibration_loss(
                    accepted_underperform_risk,
                    torch.zeros_like(accepted_underperform_risk),
                    selective_risk_mask,
                )
                * resolved_selective_risk_weight
            )

        if calibration_loss is None and not margin_terms and selective_risk_loss is None:
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            return torch.zeros((), device=device)
        total_loss_terms: List[torch.Tensor] = []
        if calibration_loss is not None:
            total_loss_terms.append(calibration_loss)
        if margin_terms:
            total_loss_terms.append(
                torch.stack(margin_terms).mean() * resolved_teacher_warp_margin_weight
            )
        if selective_risk_loss is not None:
            total_loss_terms.append(selective_risk_loss)
        return torch.stack(total_loss_terms).sum()

    def _compute_proposal_confidence_supervision_loss(
        self,
        effect_targets: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        proposal_confidence = effect_targets.get("proposal_confidence")
        proposal_confidence_target = effect_targets.get("proposal_confidence_target")
        if (
            not isinstance(proposal_confidence, torch.Tensor)
            or not isinstance(proposal_confidence_target, torch.Tensor)
        ):
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            return torch.zeros((), device=device)
        proposal_mask = effect_targets.get("proposal_training_mask")
        if proposal_confidence.shape[-2:] != proposal_confidence_target.shape[-2:]:
            proposal_confidence = F.interpolate(
                proposal_confidence,
                size=proposal_confidence_target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if proposal_confidence.shape[1] > 1:
            proposal_confidence = proposal_confidence[:, :1]
        if proposal_confidence_target.shape[1] > 1:
            proposal_confidence_target = proposal_confidence_target[:, :1]
        confidence_probability = proposal_confidence.float().clamp(1e-4, 1.0 - 1e-4)
        confidence_target = (
            proposal_confidence_target.detach().float().clamp(0.0, 1.0)
        )
        confidence_loss = -(
            confidence_target * torch.log(confidence_probability)
            + (1.0 - confidence_target) * torch.log1p(-confidence_probability)
        )
        if isinstance(proposal_mask, torch.Tensor):
            proposal_mask = proposal_mask.to(
                device=proposal_confidence.device,
                dtype=confidence_loss.dtype,
            )
            if proposal_mask.shape[-2:] != confidence_loss.shape[-2:]:
                proposal_mask = F.interpolate(
                    proposal_mask,
                    size=confidence_loss.shape[-2:],
                    mode="nearest",
                )
            if proposal_mask.shape[1] > 1:
                proposal_mask = proposal_mask[:, :1]
            if float(proposal_mask.sum().item()) < 1.0:
                return proposal_confidence.new_zeros(())
            confidence_loss = confidence_loss * proposal_mask
            denominator = proposal_mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            return (
                confidence_loss.sum(dim=(1, 2, 3)) / denominator
            ).mean()
        return confidence_loss.mean()

    def _compute_probability_head_supervision_loss(
        self,
        prediction_tensor: Optional[torch.Tensor],
        target_tensor: Optional[torch.Tensor],
        mask_tensor: Optional[torch.Tensor],
        *,
        prediction_is_logits: bool = False,
        positive_balance: bool = False,
    ) -> torch.Tensor:
        if (
            not isinstance(prediction_tensor, torch.Tensor)
            or not isinstance(target_tensor, torch.Tensor)
        ):
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            return torch.zeros((), device=device)
        if prediction_tensor.shape[-2:] != target_tensor.shape[-2:]:
            prediction_tensor = F.interpolate(
                prediction_tensor,
                size=target_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if prediction_tensor.shape[1] > 1:
            prediction_tensor = prediction_tensor[:, :1]
        if target_tensor.shape[1] > 1:
            target_tensor = target_tensor[:, :1]
        detached_target_tensor = target_tensor.detach().float().clamp(0.0, 1.0)
        positive_weight_tensor = None
        if positive_balance:
            balance_mask = torch.ones_like(detached_target_tensor)
            if isinstance(mask_tensor, torch.Tensor):
                balance_mask = mask_tensor.to(
                    device=prediction_tensor.device,
                    dtype=detached_target_tensor.dtype,
                )
                if balance_mask.shape[-2:] != detached_target_tensor.shape[-2:]:
                    balance_mask = F.interpolate(
                        balance_mask,
                        size=detached_target_tensor.shape[-2:],
                        mode="nearest",
                    )
                if balance_mask.shape[1] > 1:
                    balance_mask = balance_mask[:, :1]
            balance_mask = (balance_mask > 1e-6).to(dtype=detached_target_tensor.dtype)
            positive_mass = (detached_target_tensor * balance_mask).sum()
            negative_mass = ((1.0 - detached_target_tensor) * balance_mask).sum()
            if float(positive_mass.item()) > 1e-6 and float(negative_mass.item()) > 1e-6:
                max_positive_balance = max(
                    float(getattr(self, "proposal_head_positive_balance_max", 8.0)),
                    1.0,
                )
                positive_weight_tensor = torch.clamp(
                    negative_mass / positive_mass.clamp_min(1e-6),
                    min=1.0,
                    max=max_positive_balance,
                ).to(device=prediction_tensor.device, dtype=torch.float32)
        if prediction_is_logits:
            prediction_probability = torch.sigmoid(prediction_tensor.float())
            supervision_loss = F.binary_cross_entropy_with_logits(
                prediction_tensor.float(),
                detached_target_tensor,
                pos_weight=positive_weight_tensor,
                reduction="none",
            )
        else:
            probability_tensor = prediction_tensor.float().clamp(1e-4, 1.0 - 1e-4)
            prediction_probability = probability_tensor
            positive_log_term = detached_target_tensor * torch.log(probability_tensor)
            if positive_weight_tensor is not None:
                positive_log_term = positive_log_term * positive_weight_tensor
            supervision_loss = -(
                positive_log_term
                + (1.0 - detached_target_tensor) * torch.log1p(-probability_tensor)
            )
        base_loss = None
        prepared_mask_tensor_for_calibration = None
        if isinstance(mask_tensor, torch.Tensor):
            prepared_mask_tensor = mask_tensor.to(
                device=prediction_tensor.device,
                dtype=supervision_loss.dtype,
            )
            if prepared_mask_tensor.shape[-2:] != supervision_loss.shape[-2:]:
                prepared_mask_tensor = F.interpolate(
                    prepared_mask_tensor,
                    size=supervision_loss.shape[-2:],
                    mode="nearest",
                )
            if prepared_mask_tensor.shape[1] > 1:
                prepared_mask_tensor = prepared_mask_tensor[:, :1]
            if float(prepared_mask_tensor.sum().item()) < 1.0:
                return prediction_tensor.new_zeros(())
            supervision_loss = supervision_loss * prepared_mask_tensor
            denominator = prepared_mask_tensor.sum(dim=(1, 2, 3)).clamp_min(1.0)
            base_loss = (supervision_loss.sum(dim=(1, 2, 3)) / denominator).mean()
            prepared_mask_tensor_for_calibration = prepared_mask_tensor
        else:
            base_loss = supervision_loss.mean()
        mean_calibration_weight = max(
            float(getattr(self, "proposal_head_mean_calibration_weight", 0.0)),
            0.0,
        )
        if mean_calibration_weight <= 0.0:
            return base_loss
        if prepared_mask_tensor_for_calibration is not None:
            calibration_denominator = prepared_mask_tensor_for_calibration.sum(
                dim=(1, 2, 3)
            ).clamp_min(1.0)
            prediction_mean = (
                prediction_probability * prepared_mask_tensor_for_calibration
            ).sum(dim=(1, 2, 3)) / calibration_denominator
            target_mean = (
                detached_target_tensor * prepared_mask_tensor_for_calibration
            ).sum(dim=(1, 2, 3)) / calibration_denominator
        else:
            prediction_mean = prediction_probability.mean(dim=(1, 2, 3))
            target_mean = detached_target_tensor.mean(dim=(1, 2, 3))
        return base_loss + (prediction_mean - target_mean).pow(2).mean() * mean_calibration_weight

    def _calibration_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        eps = 1e-6
        # 多GPU模式下对齐设备
        target = target.to(prediction.device)
        if mask is not None:
            mask = mask.to(prediction.device)
        # 尺寸对齐，防止由于 VAE 下采样导致的 2px 差异
        if prediction.shape[-2:] != target.shape[-2:]:
            prediction = F.interpolate(
                prediction, size=target.shape[-2:], mode="bilinear", align_corners=False
            )
        pred_sqrt = torch.sqrt(torch.clamp(prediction, min=0.0) + eps)
        target_sqrt = torch.sqrt(torch.clamp(target, min=0.0) + eps)
        diff = (pred_sqrt - target_sqrt) ** 2
        if mask is not None:
            diff = diff * mask
            denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
            diff = diff.sum(dim=(1, 2, 3)) / denom
            return diff.mean()
        return diff.mean()

    def _expected_calibration_error(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor],
        num_bins: int = 15,
    ) -> float:
        # 多GPU模式下对齐设备
        device = prediction.device
        target = target.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        pred = torch.sqrt(torch.clamp(prediction.detach(), min=0.0))
        tgt = torch.sqrt(torch.clamp(target.detach(), min=0.0))
        # 对齐到共同尺寸，避免掩码与张量形状不一致
        if mask is not None:
            mh, mw = mask.shape[-2:]
        else:
            mh = mw = 10**9
        h = min(pred.shape[-2], tgt.shape[-2], mh)
        w = min(pred.shape[-1], tgt.shape[-1], mw)
        if pred.shape[-2:] != (h, w):
            pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
        if tgt.shape[-2:] != (h, w):
            tgt = F.interpolate(tgt, size=(h, w), mode="bilinear", align_corners=False)
        if mask is not None and mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode="nearest")
        if mask is not None:
            # 确保mask是1通道（如果是多通道取第一个或平均）
            if mask.dim() == 4 and mask.shape[1] > 1:
                mask = mask[:, :1, ...]  # 取第一个通道
            mask_flat = mask.detach().reshape(mask.shape[0], -1) > 0.5
            pred = pred.reshape(pred.shape[0], -1)
            tgt = tgt.reshape(tgt.shape[0], -1)
            selected_pred = []
            selected_tgt = []
            for batch_index in range(pred.shape[0]):
                valid = mask_flat[batch_index]
                if valid.any():
                    selected_pred.append(pred[batch_index][valid])
                    selected_tgt.append(tgt[batch_index][valid])
            if selected_pred:
                pred_flat = torch.cat(selected_pred, dim=0)
                tgt_flat = torch.cat(selected_tgt, dim=0)
            else:
                return 0.0
        else:
            pred_flat = pred.reshape(-1)
            tgt_flat = tgt.reshape(-1)
        if pred_flat.numel() == 0:
            return 0.0
        pred_min = float(pred_flat.min().item())
        pred_max = float(pred_flat.max().item())
        if not math.isfinite(pred_min) or not math.isfinite(pred_max):
            return 0.0
        if pred_max - pred_min < 1e-6:
            diff = torch.abs(pred_flat.mean() - tgt_flat.mean())
            return float(diff.item())
        bin_edges = torch.linspace(
            pred_min,
            pred_max + 1e-6,
            num_bins + 1,
            device=pred_flat.device,
        )
        total_count = float(pred_flat.numel())
        ece = torch.tensor(0.0, device=pred_flat.device)
        for bin_index in range(num_bins):
            left = bin_edges[bin_index]
            right = bin_edges[bin_index + 1]
            in_bin = (pred_flat >= left) & (pred_flat < right)
            count = in_bin.sum()
            if count == 0:
                continue
            avg_pred = pred_flat[in_bin].mean()
            avg_true = tgt_flat[in_bin].mean()
            weight = count.float() / total_count
            ece = ece + weight * torch.abs(avg_pred - avg_true)
        return float(ece.item())

    def _flatten_with_mask(
        self,
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 多GPU模式下对齐设备
        device = lhs.device
        rhs = rhs.to(device)
        if mask is not None:
            mask = mask.to(device)
        
        # 对齐到共同尺寸（最小H,W），避免掩码/张量不匹配
        if mask is not None:
            mh, mw = mask.shape[-2:]
        else:
            mh = mw = 10**9
        h = min(lhs.shape[-2], rhs.shape[-2], mh)
        w = min(lhs.shape[-1], rhs.shape[-1], mw)
        if lhs.shape[-2:] != (h, w):
            lhs = F.interpolate(lhs, size=(h, w), mode="bilinear", align_corners=False)
        if rhs.shape[-2:] != (h, w):
            rhs = F.interpolate(rhs, size=(h, w), mode="bilinear", align_corners=False)
        if mask is not None and mask.shape[-2:] != (h, w):
            mask = F.interpolate(mask, size=(h, w), mode="nearest")
        lhs_flat = lhs.detach().reshape(lhs.shape[0], -1)
        rhs_flat = rhs.detach().reshape(rhs.shape[0], -1)
        if mask is not None:
            # 确保mask是1通道
            if mask.dim() == 4 and mask.shape[1] > 1:
                mask = mask[:, :1, ...]
            mask_flat = mask.detach().reshape(mask.shape[0], -1) > 0.5
            selected_lhs = []
            selected_rhs = []
            for batch_index in range(mask_flat.shape[0]):
                valid = mask_flat[batch_index]
                if valid.any():
                    selected_lhs.append(lhs_flat[batch_index][valid])
                    selected_rhs.append(rhs_flat[batch_index][valid])
            if not selected_lhs:
                empty = lhs_flat.new_empty(0)
                return empty, empty
            lhs_flat = torch.cat(selected_lhs, dim=0)
            rhs_flat = torch.cat(selected_rhs, dim=0)
        else:
            lhs_flat = lhs_flat.reshape(-1)
            rhs_flat = rhs_flat.reshape(-1)
        return lhs_flat, rhs_flat

    def _uncertainty_error_correlation(
        self,
        prediction_std: torch.Tensor,
        error_map: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> float:
        pred_flat, err_flat = self._flatten_with_mask(prediction_std, error_map, mask)
        if pred_flat.numel() < 2 or err_flat.numel() < 2:
            return 0.0
        pred_mean = pred_flat.mean()
        err_mean = err_flat.mean()
        cov = ((pred_flat - pred_mean) * (err_flat - err_mean)).mean()
        pred_var = (pred_flat - pred_mean).pow(2).mean()
        err_var = (err_flat - err_mean).pow(2).mean()
        denom = torch.sqrt(pred_var * err_var + 1e-12)
        if denom.item() <= 1e-12:
            return 0.0
        corr = cov / denom
        if not torch.isfinite(corr):
            return 0.0
        return float(corr.item())

    def _log_uncertainty_scatter(
        self,
        stage_tag: str,
        epoch: int,
        prediction_std: torch.Tensor,
        error_map: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> None:
        if self.writer is None:
            return
        pred_flat, err_flat = self._flatten_with_mask(prediction_std, error_map, mask)
        if pred_flat.numel() == 0:
            return
        max_points = 4096
        if pred_flat.numel() > max_points:
            indices = torch.randperm(pred_flat.numel(), device=pred_flat.device)[:max_points]
            pred_flat = pred_flat[indices]
            err_flat = err_flat[indices]
        pred_np = pred_flat.detach().cpu().numpy()
        err_np = err_flat.detach().cpu().numpy()
        try:
            import matplotlib.pyplot as plt  # type: ignore
        except Exception:
            return
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.scatter(pred_np, err_np, s=4, alpha=0.3)
        ax.set_xlabel("Predicted std")
        ax.set_ylabel("Abs error")
        ax.set_title(f"{stage_tag} uncertainty vs error")
        ax.grid(alpha=0.2)
        self.writer.add_figure(f"{stage_tag}/uncert_error_scatter", fig, global_step=epoch)
        plt.close(fig)

    def _save_tensor_as_png(
        self, tensor: torch.Tensor, save_path: Path, normalize: bool = True
    ) -> None:
        """保存torch.Tensor为PNG文件。

        Args:
            tensor: [C, H, W] 格式的张量
            save_path: 保存路径
            normalize: 是否归一化到[0, 1]范围
        """
        if Image is None:
            return
        data = tensor.detach().cpu()
        if normalize:
            data = data - data.min()
            data = data / data.max().clamp_min(1e-6)
        if data.dim() == 3 and data.shape[0] == 1:
            data = data.squeeze(0)
        if data.dim() == 2:
            array = data.mul(255.0).clamp(0, 255).to(torch.uint8).numpy()
        elif data.dim() == 3 and data.shape[0] == 3:
            array = data.mul(255.0).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        else:
            return
        Image.fromarray(array).save(save_path)

    def _log_uncertainty_heatmaps(
        self,
        stage_tag: str,
        epoch: int,
        error_map: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor,
    ) -> None:
        if self.writer is None:
            return

        # 获取目标分辨率（与 error_map 一致，通常为 RGB 分辨率）
        target_hw = error_map.shape[-2:]

        def _prepare(
            tensor: torch.Tensor, *, take_sqrt: bool, upsample_to: Optional[Tuple[int, int]] = None
        ) -> torch.Tensor:
            data = tensor.detach()
            # 上采样到目标分辨率（如果需要）
            if upsample_to is not None and data.shape[-2:] != upsample_to:
                data = F.interpolate(data, size=upsample_to, mode="bilinear", align_corners=False)
            if take_sqrt:
                data = torch.sqrt(torch.clamp(data, min=0.0))
            data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            data = data[:1]
            denom = data.max().clamp_min(1e-6)
            data = data / denom
            return data.squeeze(0).cpu()

        error_prepared = _prepare(error_map, take_sqrt=False)
        # 将 uncertainty 图上采样到与 error_map 相同的分辨率
        epistemic_prepared = _prepare(epistemic, take_sqrt=True, upsample_to=target_hw)
        aleatoric_prepared = _prepare(aleatoric, take_sqrt=True, upsample_to=target_hw)

        # 记录到TensorBoard
        self.writer.add_image(
            f"{stage_tag}/error_map",
            error_prepared,
            epoch,
        )
        self.writer.add_image(
            f"{stage_tag}/epistemic_std",
            epistemic_prepared,
            epoch,
        )
        self.writer.add_image(
            f"{stage_tag}/aleatoric_std",
            aleatoric_prepared,
            epoch,
        )

        # 保存PNG文件
        if self.stage_samples_dir is not None and self._is_main_rank:
            stage_name = stage_tag.replace("/", "_")
            self._save_tensor_as_png(
                error_prepared,
                self.stage_samples_dir / f"{stage_name}_epoch{epoch:03d}_error.png",
                normalize=False,
            )
            self._save_tensor_as_png(
                epistemic_prepared,
                self.stage_samples_dir / f"{stage_name}_epoch{epoch:03d}_epistemic.png",
                normalize=False,
            )
            self._save_tensor_as_png(
                aleatoric_prepared,
                self.stage_samples_dir / f"{stage_name}_epoch{epoch:03d}_aleatoric.png",
                normalize=False,
            )


def _log_stage_progress(stage: str, epoch: int, loss: float) -> None:
    print(f"[{stage}] Epoch {epoch}: loss={loss:.4f}")


# ----------------------------------------------------------------------
# 附加：VGG感知损失实现（若torchvision可用）
# ----------------------------------------------------------------------
class _PerceptualLoss:
    """VGG19 感知损失，支持自定义特征层和权重。

    为避免网络访问失败导致的中断，若权重不可用，`available=False` 并在调用处跳过。
    """

    _LAYER_INDEX = {
        "relu1_1": 1,
        "relu1_2": 3,
        "relu2_1": 6,
        "relu2_2": 8,
        "relu3_1": 11,
        "relu3_2": 13,
        "relu3_3": 15,
        "relu3_4": 17,
        "relu4_1": 20,
        "relu4_2": 22,
        "relu4_3": 24,
        "relu4_4": 26,
        "relu5_1": 29,
        "relu5_2": 31,
        "relu5_3": 33,
        "relu5_4": 35,
    }

    def __init__(
        self,
        device: str,
        *,
        layers: Optional[Sequence[str]] = None,
        weights: Optional[Sequence[float]] = None,
    ) -> None:
        self.available = False
        self.device = device
        self._mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        default_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
        default_weights = [0.5, 0.75, 1.0, 1.5]
        self.layer_names = list(layers) if layers is not None else default_layers
        weight_candidates = list(weights) if weights is not None else default_weights
        if len(weight_candidates) != len(self.layer_names):
            weight_candidates = default_weights
        self.layer_weights = weight_candidates
        self.l1_layers = set(self.layer_names[:2])
        self.layer_configs: Sequence[Tuple[int, float, bool]] = []
        if vgg19 is None:
            return
        try:
            weight_path: Optional[Path] = None
            env_path = os.environ.get("VGG19_WEIGHTS_PATH")
            if env_path:
                candidate = Path(env_path)
                if candidate.is_file():
                    weight_path = candidate
            if weight_path is None:
                repo_root = Path(__file__).resolve().parents[2]
                candidate = repo_root / "models" / "vgg19" / "vgg19-dcbb9e9d.pth"
                if candidate.is_file():
                    weight_path = candidate
            if weight_path is not None:
                state_dict = torch.load(weight_path, map_location=device)
                vgg = vgg19(weights=None)  # type: ignore[arg-type]
                vgg.load_state_dict(state_dict)
            else:
                weights_enum = None
                if VGG19_Weights is not None:
                    weights_enum = VGG19_Weights.IMAGENET1K_FEATURES  # type: ignore[attr-defined]
                vgg = vgg19(weights=weights_enum) if weights_enum is not None else vgg19(pretrained=True)  # type: ignore[arg-type]
            vgg.features.eval()
            for p in vgg.parameters():
                p.requires_grad = False
            self.vgg_features = vgg.features.to(device)
            configs = []
            for name, weight in zip(self.layer_names, self.layer_weights):
                index = self._LAYER_INDEX.get(name)
                if index is None:
                    continue
                configs.append((index, float(weight), name in self.l1_layers))
            if not configs:
                return
            self.layer_configs = sorted(configs, key=lambda item: item[0])
            self.available = True
        except Exception:
            self.available = False

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.available:
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        orig_device = x.device
        orig_dtype = x.dtype
        
        # 将输入移动到VGG设备（多GPU时可能不同）
        vgg_device = next(self.vgg_features.parameters()).device
        x_moved = x.to(device=vgg_device, dtype=torch.float32)
        y_moved = y.to(device=vgg_device, dtype=torch.float32)
        
        x_n = self._norm(x_moved)
        collected_x: Dict[int, torch.Tensor] = {}
        collected_y: Dict[int, torch.Tensor] = {}
        max_index = self.layer_configs[-1][0]
        target_indices = {cfg[0] for cfg in self.layer_configs}
        with torch.no_grad():
            y_n = self._norm(y_moved)
            feat_y = y_n
            for idx, layer in enumerate(self.vgg_features):  # type: ignore[attr-defined]
                feat_y = layer(feat_y)
                if idx in target_indices:
                    collected_y[idx] = feat_y.detach()
                if idx >= max_index:
                    break
        feat_x = x_n
        for idx, layer in enumerate(self.vgg_features):  # type: ignore[attr-defined]
            feat_x = layer(feat_x)
            if idx in target_indices:
                collected_x[idx] = feat_x
            if idx >= max_index:
                break

        loss = torch.tensor(0.0, device=vgg_device, dtype=torch.float32)
        for target_idx, weight, use_l1 in self.layer_configs:
            fx = collected_x.get(target_idx)
            fy = collected_y.get(target_idx)
            if fx is None or fy is None:
                continue
            if use_l1:
                layer_loss = torch.abs(fx - fy).mean()
            else:
                layer_loss = F.mse_loss(fx, fy)
            loss = loss + layer_loss * fx.new_tensor(weight)
        
        # 将结果移动回原始设备和dtype
        return loss.to(device=orig_device, dtype=orig_dtype)

    def _norm(self, img: torch.Tensor) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        mean = self._mean.to(img.device, dtype=img.dtype)
        std = self._std.to(img.device, dtype=img.dtype)
        return (img - mean) / std


# ----------------------------------------------------------------------
# 附加：多视角几何一致性损失（基于反投影与重投影）
# ----------------------------------------------------------------------
def _joint_support_consistency_loss(
    *,
    rgb_gen: torch.Tensor,
    support_base_rgb: Optional[torch.Tensor],
    support_mask: Optional[torch.Tensor],
    support_confidence: Optional[torch.Tensor] = None,
    return_stats: bool = False,
) -> Tuple[torch.Tensor, Optional[Dict[str, float]]]:
    """复用 joint warp 语义的 support 一致性损失。"""

    if support_base_rgb is None or support_mask is None:
        zero = torch.tensor(0.0, device=rgb_gen.device)
        return zero, None

    if support_mask.dim() == 3:
        support_mask = support_mask.unsqueeze(1)
    if support_mask.shape[1] > 1:
        support_mask = support_mask[:, :1]
    if support_mask.shape[-2:] != rgb_gen.shape[-2:]:
        support_mask = F.interpolate(support_mask, size=rgb_gen.shape[-2:], mode="nearest")

    if support_base_rgb.shape[-2:] != rgb_gen.shape[-2:]:
        support_base_rgb = F.interpolate(
            support_base_rgb,
            size=rgb_gen.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    support_weight_map = support_mask.to(device=rgb_gen.device, dtype=rgb_gen.dtype)
    if support_confidence is not None:
        if support_confidence.dim() == 3:
            support_confidence = support_confidence.unsqueeze(1)
        if support_confidence.shape[1] > 1:
            support_confidence = support_confidence[:, :1]
        if support_confidence.shape[-2:] != rgb_gen.shape[-2:]:
            support_confidence = F.interpolate(
                support_confidence,
                size=rgb_gen.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        support_weight_map = support_weight_map * support_confidence.to(
            device=rgb_gen.device,
            dtype=rgb_gen.dtype,
        )

    absolute_difference = (
        rgb_gen - support_base_rgb.to(device=rgb_gen.device, dtype=rgb_gen.dtype)
    ).abs()
    weighted_absolute_difference = absolute_difference * support_weight_map
    valid_weight_sum = support_weight_map.sum(dim=(1, 2, 3)).clamp_min(1.0) * rgb_gen.shape[1]
    loss = weighted_absolute_difference.sum(dim=(1, 2, 3)) / valid_weight_sum

    if not return_stats:
        return loss.mean(), None

    binary_support_mask = (support_mask > 0.5).float()
    confidence_active_mask = (support_weight_map > 0.5).float()
    support_coverage_ratio = float(binary_support_mask.mean().item())
    support_confidence_ratio = float(confidence_active_mask.mean().item())
    stats = {
        "valid_view_ratio": support_coverage_ratio,
        "gate_trigger_ratio": float(max(0.0, 1.0 - support_confidence_ratio)),
        "out_of_frame_ratio": float(max(0.0, 1.0 - support_coverage_ratio)),
        "support_coverage_ratio": support_coverage_ratio,
        "support_confidence_ratio": support_confidence_ratio,
    }
    return loss.mean(), stats
