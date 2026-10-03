"""
GDDN with the paper's SD 2.1 + depth-ControlNet backbone.

关键改进:
- 使用2D扩散模型（非视频模型），匹配单帧合成任务
- ControlNet注入depth + warped image条件
- 精确的相对pose编码
- 内置不确定性预测

本地模型路径:
- SD 2.1: models/stable-diffusion-2-1-base
- ControlNet (SD 2.1): models/ControlNet
- DepthAnything: models/depth_anything_v2
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Union, List, Sequence
from pathlib import Path
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# Huggingface hub兼容性修复
import huggingface_hub
if not hasattr(huggingface_hub, "cached_download"):
    from huggingface_hub import hf_hub_download
    def cached_download(*args, **kwargs):
        return hf_hub_download(*args, **kwargs)
    huggingface_hub.cached_download = cached_download

try:
    from diffusers import (
        UNet2DConditionModel,
        AutoencoderKL,
        DDPMScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        ControlNetModel,
    )
    from transformers import (
        CLIPTextModel,
        CLIPTokenizer,
    )
except ImportError as e:
    raise ImportError(
        "diffusers and transformers are required for GDDN_ControlNet. "
        f"Install via `pip install diffusers transformers`. "
        f"Original error: {e}"
    ) from e

from controlnet_components import (
    PoseEncoder,
    PluckerEmbedder,
    ControlNetConditionPreprocessor,
    UncertaintyHead,
    FailurePriorHead,
)

@dataclass
class GDDNControlNetOutput:
    """论文生成先验的结构化输出。"""
    noise_pred: torch.Tensor
    uncertainty_epistemic: Optional[torch.Tensor] = None
    uncertainty_aleatoric: Optional[torch.Tensor] = None
    predicted_image: Optional[torch.Tensor] = None
    predicted_pure_x0_image: Optional[torch.Tensor] = None
    predicted_residual_refined_image: Optional[torch.Tensor] = None
    predicted_proposal_image: Optional[torch.Tensor] = None
    predicted_proposal_residual: Optional[torch.Tensor] = None
    predicted_proposal_applied_residual: Optional[torch.Tensor] = None
    predicted_proposal_confidence: Optional[torch.Tensor] = None
    predicted_proposal_warp_error: Optional[torch.Tensor] = None
    predicted_proposal_repairability: Optional[torch.Tensor] = None
    predicted_proposal_verification: Optional[torch.Tensor] = None
    predicted_proposal_acceptance: Optional[torch.Tensor] = None
    predicted_proposal_warp_error_logit: Optional[torch.Tensor] = None
    predicted_proposal_repairability_logit: Optional[torch.Tensor] = None
    predicted_proposal_verification_logit: Optional[torch.Tensor] = None
    predicted_proposal_acceptance_logit: Optional[torch.Tensor] = None
    predicted_support_composed_image: Optional[torch.Tensor] = None
    predicted_support_projection_mask: Optional[torch.Tensor] = None
    predicted_inpaint_mask: Optional[torch.Tensor] = None
    effect_risk_exist: Optional[torch.Tensor] = None
    effect_gain_novel: Optional[torch.Tensor] = None
    failure_wrong: Optional[torch.Tensor] = None
    failure_geo_wrong: Optional[torch.Tensor] = None
    failure_repair: Optional[torch.Tensor] = None
    failure_rgb_trust_proxy: Optional[torch.Tensor] = None
    failure_depth_trust_proxy: Optional[torch.Tensor] = None
    failure_repair_is_calibrated: bool = False
    failure_output_validity: Optional[Dict[str, bool]] = None
    failure_teacher_signal_source: Optional[str] = None
    failure_teacher_signal_is_calibrated: bool = False
    failure_online_promotion_allowed: bool = False


class GeometryAuxiliaryRouter(nn.Module):
    """将深度/置信度/可编辑掩码编码为门控式几何 token。"""

    def __init__(
        self,
        input_channels: int = 4,
        hidden_channels: int = 64,
        token_dim: int = 1024,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.token_projection = nn.Linear(hidden_channels, token_dim)
        self.gate_projection = nn.Linear(hidden_channels, 1)
        nn.init.xavier_uniform_(self.token_projection.weight)
        nn.init.zeros_(self.token_projection.bias)
        nn.init.xavier_uniform_(self.gate_projection.weight)
        nn.init.zeros_(self.gate_projection.bias)

    def forward(
        self,
        geometry_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_geometry = self.stem(geometry_maps)
        pooled_geometry = F.adaptive_avg_pool2d(encoded_geometry, output_size=1).flatten(1)
        geometry_token = self.token_projection(pooled_geometry).unsqueeze(1)
        learned_gate = torch.sigmoid(self.gate_projection(pooled_geometry)).view(-1, 1, 1)
        return geometry_token, learned_gate


class FrontierProposalHead(nn.Module):
    """显式 frontier proposal head：输出 residual + warp_error/repairability/verification 多头。"""

    def __init__(
        self,
        input_channels: int = 13,
        hidden_channels: int = 64,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.output_head = nn.Conv2d(hidden_channels, 6, kernel_size=3, padding=1)
        self.acceptance_head = nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.output_head.weight)
        nn.init.zeros_(self.output_head.bias)
        nn.init.zeros_(self.acceptance_head.weight)
        nn.init.zeros_(self.acceptance_head.bias)

    def forward(self, feature_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoded_feature_map = self.stem(feature_map)
        raw_output = self.output_head(encoded_feature_map)
        proposal_residual_rgb = torch.tanh(raw_output[:, :3])
        proposal_warp_error_logit = raw_output[:, 3:4]
        proposal_repairability_logit = raw_output[:, 4:5]
        proposal_verification_logit = raw_output[:, 5:6]
        proposal_acceptance_logit = self.acceptance_head(encoded_feature_map)
        proposal_warp_error_confidence = torch.sigmoid(proposal_warp_error_logit)
        proposal_repairability_confidence = torch.sigmoid(proposal_repairability_logit)
        proposal_verification_confidence = torch.sigmoid(proposal_verification_logit)
        proposal_acceptance_confidence = torch.sigmoid(proposal_acceptance_logit)
        return {
            "proposal_residual_rgb": proposal_residual_rgb,
            "proposal_confidence": proposal_acceptance_confidence,
            "proposal_warp_error_logit": proposal_warp_error_logit,
            "proposal_repairability_logit": proposal_repairability_logit,
            "proposal_verification_logit": proposal_verification_logit,
            "proposal_acceptance_logit": proposal_acceptance_logit,
            "proposal_warp_error_confidence": proposal_warp_error_confidence,
            "proposal_repairability_confidence": proposal_repairability_confidence,
            "proposal_verification_confidence": proposal_verification_confidence,
            "proposal_acceptance_confidence": proposal_acceptance_confidence,
        }


class GDDN_ControlNet(nn.Module):
    """
    GDDN with Stable Diffusion 2.1 + ControlNet backbone.
    
    使用 SD 2.1 和 ControlNet 实现单帧视图合成。
    
    Args:
        pretrained_model_path: SD 2.1模型路径
        controlnet_model_path: ControlNet模型路径
        dropout_rate: MC-Dropout概率
        torch_dtype: 模型精度
    """
    
    # 默认本地路径（相对于项目根目录）
    DEFAULT_SD_PATH = "models/stable-diffusion-2-1-base"
    DEFAULT_CONTROLNET_PATH = "models/ControlNet"
    DEFAULT_DEPTH_MODEL_PATH = "models/depth_anything_v2"
    
    def __init__(
        self,
        pretrained_model_path: Optional[str] = None,
        controlnet_model_path: Optional[str] = None,
        dropout_rate: float = 0.2,
        torch_dtype: Optional[torch.dtype] = None,
        use_depth_estimation: bool = False,
        default_guidance_scale: float = 1.0,
        enable_plucker_conditioning: bool = False,
        controlnet_condition_channels: int = 3,
    ):
        super().__init__()
        
        # 多GPU相关属性默认值
        self._multi_gpu_enabled = False
        self._device_map = None
        self._vae_split_across_devices = False
        self._vae_encode_device: Optional[torch.device] = None
        self._vae_decode_device: Optional[torch.device] = None
        self._auxiliary_loss_device: Optional[torch.device] = None
        self.vae_decode_chunk_size: int = 1
        
        if torch_dtype is None:
            torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model_dtype = torch_dtype
        self.default_guidance_scale = float(default_guidance_scale)
        self.enable_plucker_conditioning = bool(enable_plucker_conditioning)
        self.controlnet_condition_channels = int(controlnet_condition_channels)
        self.primary_controlnet_condition_channels = 3
        self.proposal_residual_source = "head"
        
        # 解析本地路径
        self._sd_path = self._resolve_model_path(
            pretrained_model_path, 
            self.DEFAULT_SD_PATH,
            "GDDN_SD_PATH"
        )
        self._controlnet_path = self._resolve_model_path(
            controlnet_model_path,
            self.DEFAULT_CONTROLNET_PATH,
            "GDDN_CONTROLNET_PATH"
        )
        self.supports_native_inpainting_checkpoint = self._detect_inpainting_checkpoint(
            self._sd_path
        )
        self.enable_true_masked_latent_inpainting = (
            self.supports_native_inpainting_checkpoint
        )
        # 加载预训练模型
        self._load_pretrained_models()
        
        # 获取cross-attention维度（ControlNet的维度为768，SD 2.1为1024）
        # 使用ControlNet的维度
        self._cross_attention_dim = self.controlnet.config.cross_attention_dim
        
        # 自定义组件
        self.pose_encoder = PoseEncoder(
            pose_dim=12,
            hidden_dim=256,
            output_dim=self._cross_attention_dim,
        )
        
        self.condition_preprocessor = ControlNetConditionPreprocessor(
            output_channels=self.controlnet_condition_channels,
            use_depth_model=use_depth_estimation,
        )
        self._expand_controlnet_condition_input_channels(
            self.primary_controlnet_condition_channels
        )
        
        self.uncertainty_head = UncertaintyHead(
            latent_dim=4,
            hidden_dim=64,
        )
        self.enable_failure_prior_head = False
        self.failure_prior_teacher_signal_is_calibrated = False
        self.failure_prior_head = FailurePriorHead(
            hidden_dim=64,
            active_outputs=("wrong", "geo_wrong", "rgb_trust_proxy", "depth_trust_proxy"),
        )
        
        # SD 2.1 text encoder输出1024维，ControlNet需要768维
        # 添加投影层处理维度不匹配
        self._sd_cross_attention_dim = self.unet.config.cross_attention_dim  # 1024
        self._controlnet_cross_attention_dim = self.controlnet.config.cross_attention_dim  # 768
        
        if self._sd_cross_attention_dim != self._controlnet_cross_attention_dim:
            self.encoder_proj = nn.Linear(
                self._sd_cross_attention_dim,
                self._controlnet_cross_attention_dim,
            )
        else:
            self.encoder_proj = None
        
        # Dropout用于MC-Dropout
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate = dropout_rate
        
        # 训练器通过该属性判断是否存在稀疏视图聚合器
        self.view_aggregator = None
        self.geometry_auxiliary_router = None
        
        # Schedulers
        self.noise_scheduler = DDPMScheduler.from_config(self.scheduler.config)
        self.inference_scheduler = DPMSolverMultistepScheduler.from_config(
            self.scheduler.config
        )
        self._scheduler_base_config = self._export_scheduler_config(
            self.scheduler.config
        )
        self._noise_scheduler_base_config = self._export_scheduler_config(
            self.noise_scheduler.config
        )
        self._inference_scheduler_base_config = self._export_scheduler_config(
            self.inference_scheduler.config
        )
        
        # VAE缩放因子
        self.vae_scaling_factor = float(
            getattr(self.vae.config, "scaling_factor", 0.18215)
        )
        
        # 冻结预训练层
        self._freeze_pretrained()
        
        # 将自定义组件转换为目标dtype
        self._convert_custom_modules_dtype()
        
        # 显存优化
        self._enable_memory_optimizations()
        
        # 预初始化pose投影层（768 -> 1024），避免forward中动态创建导致的初始化问题
        # PoseEncoder输出768维，SD 2.1 text encoder输出1024维
        self._pose_proj = nn.Linear(768, self._sd_cross_attention_dim)
        nn.init.xavier_uniform_(self._pose_proj.weight)
        nn.init.zeros_(self._pose_proj.bias)
        self._pose_proj = self._pose_proj.to(dtype=self._model_dtype)
        self.geometry_auxiliary_router = GeometryAuxiliaryRouter(
            input_channels=4,
            hidden_channels=64,
            token_dim=self._sd_cross_attention_dim,
        ).to(dtype=self._model_dtype)
        self.frontier_proposal_head = FrontierProposalHead(
            input_channels=13,
            hidden_channels=64,
        ).to(dtype=self._model_dtype)
        self.plucker_embedder = None
        if self.enable_plucker_conditioning:
            self.plucker_embedder = PluckerEmbedder(
                output_dim=self._sd_cross_attention_dim,
            ).to(dtype=self._model_dtype)
        self.infer_use_pose_condition = True
        self.minimum_reliable_warp_valid_ratio = 0.10
        self.support_projection_confidence_threshold = 0.45
        self.editable_support_confidence_threshold = 0.20
        self.proposal_safe_confidence_threshold = 0.0
        self.proposal_safe_confidence_gate_power = 1.0
        self.proposal_use_hard_acceptance_gate = False
        self.proposal_hard_acceptance_straight_through = True
        self.proposal_warp_error_gate_power = 1.0
        self.proposal_repairability_gate_power = 1.0
        self.proposal_verification_gate_power = 1.0
        self.support_projection_relax_hole_ratio_threshold = 0.60
        self.support_projection_min_scale = 0.55
        self.editable_schedule_hole_ratio_weight = 0.45
        self.editable_schedule_projection_gap_weight = 0.55
        self.editable_schedule_easy_threshold = 0.35
        self.editable_schedule_moderate_threshold = 0.55
        self.editable_schedule_hard_threshold = 0.70
        self.editable_schedule_extreme_threshold = 0.85
        self.support_preserve_projection_gap_weight = 0.75
        self.support_preserve_confidence_gap_weight = 0.25
        self.minimum_masked_dual_path_support_step_offset = 4
        self.maximum_masked_dual_path_support_step_offset = 8
        self.supportswitch_temporal_ramp_half_window = 2
        self.supportswitch_projection_pre_step_weight = 0.55
        self.supportswitch_dual_path_pre_step_weight = 0.75
        self.support_projection_conflict_target_scheduler_delta = 0.030
        self.support_projection_conflict_min_scale = 0.65
        self.support_projection_conflict_gamma = 1.0
        self.enable_multiview_correspondence_tokens = False
        self.support_coverage_distance_weight_floor = 0.25
        self.support_coverage_distance_weight_gamma = 0.50
        self.support_coverage_distance_weight_blend = 0.75
        self.support_fusion_distance_weight_floor = 0.10
        self.support_fusion_distance_weight_gamma = 0.75
        self.support_fusion_topk = 3
        self.support_fusion_temperature = 0.35
        self.support_projection_counterfactual_mode = "none"
        self.support_projection_counterfactual_strength = 1.0
        self.supports_mask_aware_fallback_sampling = True
        self.enforce_stepwise_masked_latent_dual_path = True
        self.masked_latent_dual_path_support_step_ratio = 0.35

        self.multimodal_encoder = None
        self.sparse_view_adapter = None

        
    def _resolve_model_path(
        self,
        provided_path: Optional[str],
        default_relative: str,
        env_var: str,
    ) -> str:
        """解析模型路径，优先级: 参数 > 环境变量 > 默认相对路径"""
        # 1. 如果提供了路径且存在，直接使用
        if provided_path is not None:
            if Path(provided_path).exists():
                return provided_path
        
        # 2. 检查环境变量
        env_path = os.environ.get(env_var)
        if env_path and Path(env_path).exists():
            return env_path
        
        # 3. 使用相对路径（相对于GDDN根目录）
        # 从当前文件位置向上推断根目录
        current_file = Path(__file__).resolve()
        gddn_root_from_file = current_file.parent.parent.parent  # examples/active_sampling -> GDDN
        
        # 尝试多种可能的根目录
        possible_roots = [
            gddn_root_from_file,
            Path.cwd(),
            Path.cwd().parent,
        ]
        
        for root in possible_roots:
            full_path = root / default_relative
            if full_path.exists():
                return str(full_path)
        
        # 4. 如果都找不到，返回提供的路径或默认路径
        if provided_path is not None:
            return provided_path
        return default_relative

    @staticmethod
    def _detect_inpainting_checkpoint(model_path: str) -> bool:
        normalized_model_path = str(model_path).lower()
        return ("inpaint" in normalized_model_path) or (
            "inpainting" in normalized_model_path
        )
    
    def _load_pretrained_models(self):
        """加载SD和ControlNet预训练模型。"""
        # 加载UNet
        self.unet = UNet2DConditionModel.from_pretrained(
            self._sd_path,
            subfolder="unet",
            torch_dtype=self._model_dtype,
        )
        
        # 加载VAE
        self.vae = AutoencoderKL.from_pretrained(
            self._sd_path,
            subfolder="vae",
            torch_dtype=self._model_dtype,
        )
        
        # 加载Scheduler
        self.scheduler = DDIMScheduler.from_pretrained(
            self._sd_path,
            subfolder="scheduler",
        )
        
        # 加载Text Encoder（用于生成空文本embedding）
        self.text_encoder = CLIPTextModel.from_pretrained(
            self._sd_path,
            subfolder="text_encoder",
            torch_dtype=self._model_dtype,
        )
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self._sd_path,
            subfolder="tokenizer",
        )
        
        # 加载ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            self._controlnet_path,
            torch_dtype=self._model_dtype,
        )

    def _export_scheduler_config(self, config_obj: Any) -> Dict[str, Any]:
        if config_obj is None:
            return {}
        if hasattr(config_obj, "to_dict"):
            return dict(config_obj.to_dict())
        if isinstance(config_obj, dict):
            return dict(config_obj)
        exported: Dict[str, Any] = {}
        for key, value in vars(config_obj).items():
            if key.startswith("_"):
                continue
            exported[key] = value
        return exported

    def apply_scheduler_overrides(self, overrides: Optional[Dict[str, Any]]) -> None:
        if not overrides:
            return
        scheduler_config = dict(self._scheduler_base_config)
        scheduler_config.update(overrides)
        noise_scheduler_config = dict(self._noise_scheduler_base_config)
        noise_scheduler_config.update(overrides)
        inference_scheduler_config = dict(self._inference_scheduler_base_config)
        inference_scheduler_config.update(overrides)
        self._scheduler_base_config = scheduler_config
        self._noise_scheduler_base_config = noise_scheduler_config
        self._inference_scheduler_base_config = inference_scheduler_config
        self.scheduler = DDIMScheduler.from_config(self._scheduler_base_config)
        self.noise_scheduler = DDPMScheduler.from_config(
            self._noise_scheduler_base_config
        )
        self.inference_scheduler = DPMSolverMultistepScheduler.from_config(
            self._inference_scheduler_base_config
        )
        
    def _freeze_pretrained(self):
        """冻结预训练层以稳定训练。"""
        # 完全冻结VAE
        self.vae.requires_grad_(False)
        
        # 完全冻结Text Encoder
        self.text_encoder.requires_grad_(False)
        
        # 部分冻结UNet（保留部分层可训练用于适配）
        # 冻结预训练主干，仅训练论文定义的条件与不确定性模块
        total_params = list(self.unet.parameters())
        freeze_until = int(len(total_params) * 0.8)
        for idx, param in enumerate(total_params):
            if idx < freeze_until:
                param.requires_grad = False
        
        # ControlNet保持可训练
        # 但也可以选择冻结以减少训练开销
        self.controlnet.requires_grad_(True)

    def _expand_controlnet_condition_input_channels(
        self,
        target_input_channels: int,
    ) -> None:
        controlnet_condition_embedding = getattr(self.controlnet, "controlnet_cond_embedding", None)
        conv_in = getattr(controlnet_condition_embedding, "conv_in", None)
        if not isinstance(conv_in, nn.Conv2d):
            return
        if conv_in.in_channels == target_input_channels:
            return

        expanded_conv = nn.Conv2d(
            in_channels=target_input_channels,
            out_channels=conv_in.out_channels,
            kernel_size=conv_in.kernel_size,
            stride=conv_in.stride,
            padding=conv_in.padding,
            dilation=conv_in.dilation,
            groups=conv_in.groups,
            bias=conv_in.bias is not None,
            padding_mode=conv_in.padding_mode,
        ).to(device=conv_in.weight.device, dtype=conv_in.weight.dtype)
        with torch.no_grad():
            expanded_conv.weight.zero_()
            copied_channels = min(conv_in.in_channels, target_input_channels)
            expanded_conv.weight[:, :copied_channels].copy_(conv_in.weight[:, :copied_channels])
            if conv_in.bias is not None and expanded_conv.bias is not None:
                expanded_conv.bias.copy_(conv_in.bias)
        self.controlnet.controlnet_cond_embedding.conv_in = expanded_conv

    def _convert_custom_modules_dtype(self):
        """将自定义模块转换为目标dtype"""
        self.pose_encoder = self.pose_encoder.to(dtype=self._model_dtype)
        if self.geometry_auxiliary_router is not None:
            self.geometry_auxiliary_router = self.geometry_auxiliary_router.to(
                dtype=self._model_dtype
            )
        if hasattr(self, "frontier_proposal_head") and self.frontier_proposal_head is not None:
            self.frontier_proposal_head = self.frontier_proposal_head.to(
                dtype=self._model_dtype
            )
        if hasattr(self, "plucker_embedder") and self.plucker_embedder is not None:
            self.plucker_embedder = self.plucker_embedder.to(dtype=self._model_dtype)
        self.condition_preprocessor = self.condition_preprocessor.to(dtype=self._model_dtype)
        self.condition_preprocessor.keep_depth_model_in_float32()
        self.uncertainty_head = self.uncertainty_head.to(dtype=self._model_dtype)
        if hasattr(self, "failure_prior_head") and self.failure_prior_head is not None:
            self.failure_prior_head = self.failure_prior_head.to(dtype=self._model_dtype)
        
        # encoder_proj用于SD 2.1 -> ControlNet维度投影
        if self.encoder_proj is not None:
            self.encoder_proj = self.encoder_proj.to(dtype=self._model_dtype)
        
    def _enable_memory_optimizations(self):
        """启用显存优化"""
        # 启用梯度检查点
        if hasattr(self.vae, "enable_gradient_checkpointing"):
            self.vae.enable_gradient_checkpointing()
        if hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        if hasattr(self.controlnet, "enable_gradient_checkpointing"):
            self.controlnet.enable_gradient_checkpointing()
        
        # 尝试启用xformers
        try:
            if hasattr(self.unet, "enable_xformers_memory_efficient_attention"):
                self.unet.enable_xformers_memory_efficient_attention()
            if hasattr(self.controlnet, "enable_xformers_memory_efficient_attention"):
                self.controlnet.enable_xformers_memory_efficient_attention()
        except Exception:
            pass  # xformers不可用时忽略
        try:
            if hasattr(self.vae, "enable_slicing"):
                self.vae.enable_slicing()
        except Exception:
            pass
    
    # ==================== ThreeStageTrainer 兼容属性 ====================
    
    @property
    def epistemic_head(self):
        """兼容ThreeStageTrainer的epistemic_head属性"""
        if hasattr(self, '_epistemic_head_override') and self._epistemic_head_override is not None:
            return self._epistemic_head_override
        return self.uncertainty_head.epistemic_net if self.uncertainty_head else None
    
    @epistemic_head.setter
    def epistemic_head(self, value):
        """允许ThreeStageTrainer设置epistemic_head为None"""
        self._epistemic_head_override = value
    
    @property
    def aleatoric_head(self):
        """兼容ThreeStageTrainer的aleatoric_head属性"""
        if hasattr(self, '_aleatoric_head_override') and self._aleatoric_head_override is not None:
            return self._aleatoric_head_override
        return self.uncertainty_head.aleatoric_net if self.uncertainty_head else None

    @aleatoric_head.setter
    def aleatoric_head(self, value):
        """允许ThreeStageTrainer设置aleatoric_head为None"""
        self._aleatoric_head_override = value

    # ================================================================
    # Layer 2 截断MC-Dropout 支持
    # ================================================================
    def enable_mc_dropout(self, p: Optional[float] = None) -> None:
        """启用UNet和ControlNet中的所有nn.Dropout层用于MC采样。

        保存原始状态以便后续 disable_mc_dropout() 恢复。

        Args:
            p: 可选的dropout概率覆写。如果为None则使用各层原始概率。
        """
        self._mc_dropout_saved_states: Dict[str, Tuple[bool, float]] = {}
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout):
                self._mc_dropout_saved_states[name] = (module.training, module.p)
                module.train()
                if p is not None:
                    module.p = p

    def disable_mc_dropout(self) -> None:
        """恢复所有nn.Dropout层到 enable_mc_dropout 之前的状态。"""
        saved = getattr(self, "_mc_dropout_saved_states", {})
        for name, module in self.named_modules():
            if isinstance(module, nn.Dropout) and name in saved:
                was_training, original_p = saved[name]
                if not was_training:
                    module.eval()
                module.p = original_p
        self._mc_dropout_saved_states = {}

    def set_infer_use_pose_condition(self, enable: bool) -> None:
        """控制推理路径是否注入PoseEncoder条件，不影响训练forward路径。"""
        self.infer_use_pose_condition = bool(enable)

    def set_enable_plucker_conditioning(self, enable: bool) -> None:
        """控制训练/推理路径是否注入 Plucker token。"""
        self.enable_plucker_conditioning = bool(enable)

    def configure_failure_prior_head(
        self,
        *,
        enable: bool,
        active_outputs: Optional[Sequence[str]] = None,
        teacher_signal_is_calibrated: bool = False,
    ) -> None:
        """Enable/disable the offline where_wrong head output path.

        This only controls whether ``failure_*`` maps are returned from teacher
        forward.  It does not enable online sampling, supervision, densification,
        or repair routing.
        """

        self.enable_failure_prior_head = bool(enable)
        self.failure_prior_teacher_signal_is_calibrated = bool(teacher_signal_is_calibrated)
        if active_outputs is not None and self.failure_prior_head is not None:
            self.failure_prior_head.active_outputs = tuple(active_outputs)
        if self.failure_prior_head is not None:
            self.failure_prior_head = self.failure_prior_head.to(
                device=next(self.parameters()).device,
                dtype=self._model_dtype,
            )
    
    
        
    def _get_text_embeddings(
        self,
        batch_size: int,
        device: torch.device,
        prompt: str = "a high quality photo",
    ) -> torch.Tensor:
        """获取文本embedding
        
        Args:
            batch_size: 批次大小
            device: 设备
            prompt: 文本提示（默认"a high quality photo"提供基本语义引导）
        """
        text_input = self.tokenizer(
            [prompt] * batch_size,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(device)
            )[0]
        
        return text_embeddings.to(dtype=self._model_dtype)
    
    def _select_nearest_view(
        self,
        sparse_poses: torch.Tensor,  # [B, V, 4, 4]
        target_pose: torch.Tensor,   # [B, 4, 4]
    ) -> torch.Tensor:
        """选择与目标pose最近的参考视角。
        
        Args:
            sparse_poses: [B, V, 4, 4] 稀疏视角位姿
            target_pose: [B, 4, 4] 目标位姿
            
        Returns:
            [B] 最近视角的索引
        """
        sparse_rotations = sparse_poses[:, :, :3, :3].float()
        sparse_translations = sparse_poses[:, :, :3, 3].float()
        target_rotations = target_pose[:, :3, :3].float()
        target_translations = target_pose[:, :3, 3].float()

        sparse_centers = -torch.matmul(
            sparse_rotations.transpose(-1, -2),
            sparse_translations.unsqueeze(-1),
        ).squeeze(-1)
        target_centers = -torch.matmul(
            target_rotations.transpose(-1, -2),
            target_translations.unsqueeze(-1),
        ).squeeze(-1)

        distances = torch.norm(
            sparse_centers - target_centers.unsqueeze(1),
            dim=-1,
        )  # [B, V]
        nearest_idx = distances.argmin(dim=1)  # [B]
        return nearest_idx

    def _normalize_sparse_intrinsics(
        self,
        sparse_intrinsics: Optional[torch.Tensor],
        batch_size: int,
        num_views: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """将稀疏视角内参统一整理为 [B, V, 3, 3]。"""
        if sparse_intrinsics is None:
            return None
        normalized_sparse_intrinsics = sparse_intrinsics.to(device=device, dtype=dtype)
        if normalized_sparse_intrinsics.dim() == 3:
            normalized_sparse_intrinsics = normalized_sparse_intrinsics.unsqueeze(0)
        if normalized_sparse_intrinsics.shape[0] == 1 and batch_size > 1:
            normalized_sparse_intrinsics = normalized_sparse_intrinsics.expand(batch_size, -1, -1, -1)
        if normalized_sparse_intrinsics.shape[1] == 1 and num_views > 1:
            normalized_sparse_intrinsics = normalized_sparse_intrinsics.expand(-1, num_views, -1, -1)
        return normalized_sparse_intrinsics

    def _build_control_condition_from_support_base(
        self,
        support_rgb: torch.Tensor,
        support_confidence: torch.Tensor,
        depth_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """基于目标平面对齐后的 support base 构造 ControlNet 条件。"""
        support_rgb = support_rgb.to(dtype=self._model_dtype)
        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        if depth_map.shape[1] != 1:
            depth_map = depth_map[:, :1]
        if depth_map.shape[-2:] != support_rgb.shape[-2:]:
            depth_map = F.interpolate(
                depth_map.float(),
                size=support_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=depth_map.dtype)
        if support_confidence.dim() == 3:
            support_confidence = support_confidence.unsqueeze(1)
        if support_confidence.shape[1] != 1:
            support_confidence = support_confidence[:, :1]
        if support_confidence.shape[-2:] != support_rgb.shape[-2:]:
            support_confidence = F.interpolate(
                support_confidence.float(),
                size=support_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=support_confidence.dtype)

        depth_normalized = depth_map.float()
        depth_min = depth_normalized.amin(dim=(2, 3), keepdim=True)
        depth_max = depth_normalized.amax(dim=(2, 3), keepdim=True)
        depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min + 1e-8)

        support_confidence = torch.nan_to_num(
            support_confidence.to(dtype=self._model_dtype).clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        primary_condition, low_confidence_known_mask = (
            self.condition_preprocessor._build_primary_image_like_condition(
                support_rgb=support_rgb,
                depth_normalized=depth_normalized.to(dtype=self._model_dtype),
                support_confidence=support_confidence,
            )
        )
        raw_geometry_condition = torch.cat(
            [
                support_rgb,
                depth_normalized.to(dtype=self._model_dtype),
                support_confidence,
            ],
            dim=1,
        )
        if self.condition_preprocessor.output_channels == 5:
            controlnet_condition = torch.cat(
                [
                    support_rgb,
                    depth_normalized.to(dtype=self._model_dtype),
                    support_confidence,
                ],
                dim=1,
            )
        elif self.condition_preprocessor.output_channels == 3:
            depth_3ch = depth_normalized.to(dtype=self._model_dtype).repeat(1, 3, 1, 1)
            confidence_3ch = support_confidence.repeat(1, 3, 1, 1)
            alpha_rgb = 0.8
            controlnet_condition = (
                (alpha_rgb * support_rgb + (1.0 - alpha_rgb) * depth_3ch) * confidence_3ch
                + depth_3ch * (1.0 - confidence_3ch)
            )
        else:
            condition_input = torch.cat(
                [
                    support_rgb,
                    depth_normalized.to(dtype=self._model_dtype),
                    support_confidence,
                ],
                dim=1,
            )
            controlnet_condition = self.condition_preprocessor.condition_fusion(condition_input)

        controlnet_condition = torch.nan_to_num(
            controlnet_condition,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        conditioning_details = {
            "warped_image": support_rgb,
            "warp_confidence": support_confidence,
            "depth_normalized": depth_normalized,
            "primary_condition": primary_condition,
            "raw_geometry_condition": raw_geometry_condition,
            "low_confidence_known_mask": low_confidence_known_mask,
            "editable_low_confidence_mask": low_confidence_known_mask,
        }
        return controlnet_condition, support_confidence, conditioning_details

    def _resolve_primary_controlnet_condition(
        self,
        *,
        controlnet_condition: torch.Tensor,
        conditioning_details: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        if conditioning_details is None:
            return controlnet_condition
        primary_condition = conditioning_details.get("primary_condition")
        if primary_condition is None:
            return controlnet_condition
        return primary_condition.to(
            device=controlnet_condition.device,
            dtype=controlnet_condition.dtype,
        )

    def _build_geometry_routing_inputs(
        self,
        *,
        conditioning_details: Optional[Dict[str, torch.Tensor]],
        support_projection_mask: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        if conditioning_details is None:
            return None, None, None

        depth_normalized = conditioning_details.get("depth_normalized")
        warp_confidence = conditioning_details.get("warp_confidence")
        low_confidence_known_mask = conditioning_details.get("low_confidence_known_mask")
        if low_confidence_known_mask is None:
            low_confidence_known_mask = conditioning_details.get("editable_low_confidence_mask")
        if depth_normalized is None or warp_confidence is None:
            return None, None, None

        def _align_single_channel(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if tensor is None:
                return None
            aligned_tensor = tensor.to(device=device, dtype=self._model_dtype)
            if aligned_tensor.dim() == 3:
                aligned_tensor = aligned_tensor.unsqueeze(1)
            if aligned_tensor.shape[1] != 1:
                aligned_tensor = aligned_tensor[:, :1]
            if aligned_tensor.shape[-2:] != target_hw:
                aligned_tensor = F.interpolate(
                    aligned_tensor,
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.nan_to_num(
                aligned_tensor,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        depth_normalized = _align_single_channel(depth_normalized)
        warp_confidence = _align_single_channel(warp_confidence)
        low_confidence_known_mask = _align_single_channel(low_confidence_known_mask)
        if low_confidence_known_mask is None and warp_confidence is not None:
            low_confidence_known_mask = (
                (warp_confidence > 1e-6).float()
                * (warp_confidence < self.editable_support_confidence_threshold).float()
            )
        support_projection_mask = _align_single_channel(support_projection_mask)
        if support_projection_mask is None and warp_confidence is not None:
            support_projection_mask = (
                (warp_confidence - self.support_projection_confidence_threshold)
                / max(1.0 - self.support_projection_confidence_threshold, 1e-6)
            ).clamp(0.0, 1.0)
        if (
            depth_normalized is None
            or warp_confidence is None
            or low_confidence_known_mask is None
            or support_projection_mask is None
        ):
            return None, None, None

        geometry_maps = torch.cat(
            [
                depth_normalized,
                warp_confidence,
                low_confidence_known_mask,
                support_projection_mask,
            ],
            dim=1,
        )
        low_confidence_ratio = low_confidence_known_mask.mean(
            dim=(2, 3),
            keepdim=False,
        ).unsqueeze(-1)
        routing_strength = torch.clamp(0.10 + 0.90 * low_confidence_ratio, min=0.0, max=1.0)
        return geometry_maps, routing_strength, low_confidence_known_mask

    def _append_geometry_condition_tokens(
        self,
        *,
        encoder_hidden_states: torch.Tensor,
        target_pose: torch.Tensor,
        target_intrinsics: Optional[torch.Tensor],
        height: int,
        width: int,
        geometry_maps: Optional[torch.Tensor],
        routing_strength: Optional[torch.Tensor],
        low_confidence_known_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if geometry_maps is not None and routing_strength is not None:
            geometry_token, learned_gate = self.geometry_auxiliary_router(geometry_maps)
            combined_gate = torch.clamp(
                0.5 * learned_gate.to(dtype=self._model_dtype)
                + 0.5 * routing_strength.to(dtype=self._model_dtype),
                min=0.0,
                max=1.0,
            )
            geometry_token = geometry_token.to(dtype=self._model_dtype) * combined_gate
            encoder_hidden_states = torch.cat([encoder_hidden_states, geometry_token], dim=1)
        else:
            combined_gate = None

        if (
            self.enable_plucker_conditioning
            and self.plucker_embedder is not None
            and target_intrinsics is not None
        ):
            plucker_tokens = self.plucker_embedder(
                target_pose=target_pose,
                target_intrinsics=target_intrinsics,
                height=height,
                width=width,
            )
            if low_confidence_known_mask is not None:
                pooled_editable_mask = F.adaptive_avg_pool2d(
                    low_confidence_known_mask.to(dtype=self._model_dtype),
                    output_size=(
                        self.plucker_embedder.token_grid_size,
                        self.plucker_embedder.token_grid_size,
                    ),
                )
                plucker_token_gate = 0.15 + 0.85 * pooled_editable_mask.flatten(2).transpose(1, 2)
                plucker_tokens = plucker_tokens * plucker_token_gate
            if combined_gate is not None:
                plucker_tokens = plucker_tokens * combined_gate
            plucker_tokens = torch.nan_to_num(plucker_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            encoder_hidden_states = torch.cat([encoder_hidden_states, plucker_tokens], dim=1)

        return encoder_hidden_states

    def _build_correspondence_texture_source(
        self,
        *,
        ref_image: torch.Tensor,
        support_rgb: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """构造带 target-plane 对齐 support 提示的纹理输入。"""
        correspondence_texture_source = ref_image.to(dtype=self._model_dtype)
        if support_rgb is None or support_confidence is None:
            return correspondence_texture_source

        aligned_support_rgb = support_rgb.to(dtype=self._model_dtype)
        if aligned_support_rgb.shape[-2:] != correspondence_texture_source.shape[-2:]:
            aligned_support_rgb = F.interpolate(
                aligned_support_rgb,
                size=correspondence_texture_source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        normalized_support_confidence = support_confidence
        if normalized_support_confidence.dim() == 3:
            normalized_support_confidence = normalized_support_confidence.unsqueeze(1)
        if normalized_support_confidence.shape[1] != 1:
            normalized_support_confidence = normalized_support_confidence[:, :1]
        if normalized_support_confidence.shape[-2:] != correspondence_texture_source.shape[-2:]:
            normalized_support_confidence = F.interpolate(
                normalized_support_confidence.float(),
                size=correspondence_texture_source.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).to(dtype=normalized_support_confidence.dtype)

        normalized_support_confidence = torch.nan_to_num(
            normalized_support_confidence.to(dtype=self._model_dtype).clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
        boosted_support_weight = torch.sqrt(normalized_support_confidence.clamp_min(0.0))
        editable_support_confidence_threshold = float(
            max(0.0, min(getattr(self, "editable_support_confidence_threshold", 0.20), 1.0))
        )
        low_confidence_support_mask = (
            (normalized_support_confidence > 0.0)
            * (normalized_support_confidence < editable_support_confidence_threshold).float()
        )
        correspondence_weight = torch.clamp(
            boosted_support_weight + 0.35 * low_confidence_support_mask,
            min=0.0,
            max=1.0,
        )
        correspondence_weight_rgb = correspondence_weight.repeat(1, 3, 1, 1)
        correspondence_texture_source = (
            correspondence_weight_rgb * aligned_support_rgb
            + (1.0 - correspondence_weight_rgb) * correspondence_texture_source
        )
        return torch.nan_to_num(
            correspondence_texture_source.clamp(0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )

    def _encode_texture_condition_tokens(
        self,
        *,
        texture_source: torch.Tensor,
        latent_hw: Tuple[int, int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if self.sparse_view_adapter is None:
            return None

        extracted_texture = self.sparse_view_adapter.texture_extractor(
            texture_source.to(device=device, dtype=self._model_dtype)
        )
        projected_texture = self.sparse_view_adapter.projections[0](extracted_texture)
        projected_texture = F.interpolate(
            projected_texture,
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        texture_sequence = projected_texture.flatten(2).transpose(1, 2)
        if not hasattr(self, "_texture_proj") or self._texture_proj is None:
            self._texture_proj = nn.Linear(320, self._sd_cross_attention_dim).to(
                device=device, dtype=self._model_dtype
            )
            nn.init.xavier_uniform_(self._texture_proj.weight)
            nn.init.zeros_(self._texture_proj.bias)
        texture_proj_dtype = self._texture_proj.weight.dtype
        texture_proj_device = self._texture_proj.weight.device
        texture_sequence = texture_sequence.to(
            device=texture_proj_device,
            dtype=texture_proj_dtype,
        )
        texture_tokens = self._texture_proj(texture_sequence).to(dtype=self._model_dtype)
        texture_tokens = torch.clamp(texture_tokens, min=-65504.0, max=65504.0)
        return torch.nan_to_num(texture_tokens, nan=0.0, posinf=0.0, neginf=0.0)

    def _encode_multiview_correspondence_tokens(
        self,
        *,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        target_pose: torch.Tensor,
        support_rgb: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        latent_hw: Tuple[int, int],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """显式聚合多视图 correspondence 特征，并与 target-plane support 提示融合。"""
        if self.sparse_view_adapter is None:
            return None

        normalized_sparse_images = sparse_images
        normalized_sparse_poses = sparse_poses
        normalized_target_pose = target_pose
        if normalized_sparse_images.dim() == 4:
            normalized_sparse_images = normalized_sparse_images.unsqueeze(0)
        if normalized_sparse_poses.dim() == 3:
            normalized_sparse_poses = normalized_sparse_poses.unsqueeze(0)
        if normalized_target_pose.dim() == 2:
            normalized_target_pose = normalized_target_pose.unsqueeze(0)

        multiview_features = self.sparse_view_adapter.extract_and_warp(
            normalized_sparse_images.to(device=device, dtype=self._model_dtype),
            normalized_sparse_poses[:, :, :3, :4].to(device=device, dtype=self._model_dtype),
            normalized_target_pose[:, :3, :4].to(device=device, dtype=self._model_dtype),
        )
        multiview_features = torch.nan_to_num(
            multiview_features,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        if support_rgb is not None:
            aligned_support_rgb = support_rgb.to(device=device, dtype=self._model_dtype)
            if aligned_support_rgb.dim() == 3:
                aligned_support_rgb = aligned_support_rgb.unsqueeze(0)
            if aligned_support_rgb.shape[-2:] != multiview_features.shape[-2:]:
                aligned_support_rgb = F.interpolate(
                    aligned_support_rgb,
                    size=multiview_features.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            support_features = self.sparse_view_adapter.texture_extractor(aligned_support_rgb)
            support_features = torch.nan_to_num(
                support_features,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if support_confidence is not None:
                normalized_support_confidence = support_confidence.float()
                if normalized_support_confidence.dim() == 3:
                    normalized_support_confidence = normalized_support_confidence.unsqueeze(1)
                if normalized_support_confidence.shape[1] != 1:
                    normalized_support_confidence = normalized_support_confidence[:, :1]
                if normalized_support_confidence.shape[-2:] != multiview_features.shape[-2:]:
                    normalized_support_confidence = F.interpolate(
                        normalized_support_confidence,
                        size=multiview_features.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                normalized_support_confidence = torch.nan_to_num(
                    normalized_support_confidence.to(device=device, dtype=self._model_dtype),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
            else:
                normalized_support_confidence = torch.ones(
                    multiview_features.shape[0],
                    1,
                    multiview_features.shape[-2],
                    multiview_features.shape[-1],
                    device=device,
                    dtype=self._model_dtype,
                )
            multiview_features = (
                (1.0 - normalized_support_confidence) * multiview_features
                + normalized_support_confidence * support_features
            )

        projected_texture = self.sparse_view_adapter.projections[0](multiview_features)
        projected_texture = F.interpolate(
            projected_texture,
            size=latent_hw,
            mode="bilinear",
            align_corners=False,
        )
        texture_sequence = projected_texture.flatten(2).transpose(1, 2)
        if not hasattr(self, "_texture_proj") or self._texture_proj is None:
            self._texture_proj = nn.Linear(320, self._sd_cross_attention_dim).to(
                device=device, dtype=self._model_dtype
            )
            nn.init.xavier_uniform_(self._texture_proj.weight)
            nn.init.zeros_(self._texture_proj.bias)
        texture_proj_dtype = self._texture_proj.weight.dtype
        texture_proj_device = self._texture_proj.weight.device
        texture_sequence = texture_sequence.to(
            device=texture_proj_device,
            dtype=texture_proj_dtype,
        )
        texture_tokens = self._texture_proj(texture_sequence).to(dtype=self._model_dtype)
        texture_tokens = torch.clamp(texture_tokens, min=-65504.0, max=65504.0)
        return torch.nan_to_num(texture_tokens, nan=0.0, posinf=0.0, neginf=0.0)

    def _aggregate_multiview_support_scores(
        self,
        per_view_support_scores: torch.Tensor,
    ) -> torch.Tensor:
        """用有界累积聚合保留互补的多视图中等支持。"""
        support_scores_dtype = per_view_support_scores.dtype
        clamped_support_scores = torch.nan_to_num(
            per_view_support_scores.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        aggregated_support_scores = 1.0 - torch.prod(
            1.0 - clamped_support_scores,
            dim=0,
        )
        return aggregated_support_scores.to(dtype=support_scores_dtype)

    def _normalize_inverse_distance_weights(
        self,
        view_distances: torch.Tensor,
    ) -> torch.Tensor:
        """将视角距离转为[0, 1]归一化逆距离权重。"""
        inverse_distance_weights = 1.0 / torch.clamp(view_distances.float(), min=1e-4)
        inverse_distance_weights = inverse_distance_weights / torch.clamp(
            inverse_distance_weights.max(),
            min=1e-6,
        )
        return inverse_distance_weights.clamp(0.0, 1.0)

    def _apply_distance_weight_curve(
        self,
        normalized_distance_weights: torch.Tensor,
        *,
        weight_floor: float,
        weight_gamma: float,
    ) -> torch.Tensor:
        """使用带下界和幂次的缓和距离权重，避免远视图support被过度压扁。"""
        clamped_distance_weights = torch.nan_to_num(
            normalized_distance_weights.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        effective_weight_floor = float(min(max(weight_floor, 0.0), 1.0))
        effective_weight_gamma = float(max(weight_gamma, 1e-4))
        curved_distance_weights = clamped_distance_weights.pow(effective_weight_gamma)
        softened_distance_weights = effective_weight_floor + (
            1.0 - effective_weight_floor
        ) * curved_distance_weights
        return softened_distance_weights.clamp(0.0, 1.0).to(
            dtype=normalized_distance_weights.dtype
        )

    def _build_joint_support_base(
        self,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        target_pose: torch.Tensor,
        depth_map: torch.Tensor,
        target_intrinsics: Optional[torch.Tensor],
        sparse_intrinsics: Optional[torch.Tensor] = None,
        sparse_depth_maps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """将所有稀疏视角分别 warp 到目标视平面并进行联合融合。"""
        batch_size, num_views, _, image_height, image_width = sparse_images.shape
        device = sparse_images.device

        if depth_map.dim() == 3:
            depth_map = depth_map.unsqueeze(1)
        if depth_map.shape[1] != 1:
            depth_map = depth_map[:, :1]

        if target_intrinsics is None:
            focal_length = float(max(image_height, image_width))
            default_target_intrinsics = torch.tensor(
                [
                    [focal_length, 0.0, image_width / 2.0],
                    [0.0, focal_length, image_height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                device=device,
                dtype=torch.float32,
            )
            target_intrinsics = default_target_intrinsics.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            target_intrinsics = target_intrinsics.to(device=device, dtype=torch.float32)
            if target_intrinsics.dim() == 2:
                target_intrinsics = target_intrinsics.unsqueeze(0)
            if target_intrinsics.shape[0] == 1 and batch_size > 1:
                target_intrinsics = target_intrinsics.expand(batch_size, -1, -1)

        normalized_sparse_intrinsics = self._normalize_sparse_intrinsics(
            sparse_intrinsics=sparse_intrinsics,
            batch_size=batch_size,
            num_views=num_views,
            device=device,
            dtype=torch.float32,
        )
        if normalized_sparse_intrinsics is None:
            normalized_sparse_intrinsics = target_intrinsics.unsqueeze(1).expand(-1, num_views, -1, -1)

        nearest_view_indices = self._select_nearest_view(sparse_poses, target_pose)
        batch_indices = torch.arange(batch_size, device=device)
        nearest_reference_images = sparse_images[batch_indices, nearest_view_indices]
        nearest_reference_poses = sparse_poses[batch_indices, nearest_view_indices]
        nearest_reference_intrinsics = normalized_sparse_intrinsics[batch_indices, nearest_view_indices]

        fused_warped_rgb_list: List[torch.Tensor] = []
        joint_valid_mask_list: List[torch.Tensor] = []
        support_confidence_list: List[torch.Tensor] = []
        prethreshold_valid_mask_list: List[torch.Tensor] = []
        raw_support_confidence_list: List[torch.Tensor] = []
        legacy_raw_support_confidence_list: List[torch.Tensor] = []
        support_quality_confidence_list: List[torch.Tensor] = []
        support_fusion_rgb_disagreement_list: List[torch.Tensor] = []
        union_valid_mask_before_distance_weight_list: List[torch.Tensor] = []
        per_view_valid_coverage_list: List[torch.Tensor] = []
        per_view_distance_weight_list: List[torch.Tensor] = []
        per_view_raw_inverse_distance_weight_list: List[torch.Tensor] = []
        per_view_coverage_distance_weight_list: List[torch.Tensor] = []
        per_view_fusion_distance_weight_list: List[torch.Tensor] = []
        nearest_reference_depth_list: List[torch.Tensor] = []

        sparse_rotations = sparse_poses[:, :, :3, :3].float()
        sparse_translations = sparse_poses[:, :, :3, 3].float()
        target_rotations = target_pose[:, :3, :3].float()
        target_translations = target_pose[:, :3, 3].float()
        sparse_centers = -torch.matmul(
            sparse_rotations.transpose(-1, -2),
            sparse_translations.unsqueeze(-1),
        ).squeeze(-1)
        target_centers = -torch.matmul(
            target_rotations.transpose(-1, -2),
            target_translations.unsqueeze(-1),
        ).squeeze(-1)
        view_distances = torch.norm(
            sparse_centers - target_centers.unsqueeze(1),
            dim=-1,
        )  # [B, V]

        for batch_index in range(batch_size):
            batch_target_depth = depth_map[batch_index:batch_index + 1].to(device=device, dtype=torch.float32)
            batch_target_pose = target_pose[batch_index:batch_index + 1].to(device=device, dtype=torch.float32)
            batch_target_intrinsics = target_intrinsics[batch_index:batch_index + 1]

            warped_images_per_view: List[torch.Tensor] = []
            valid_masks_per_view: List[torch.Tensor] = []
            reference_depth_maps_per_view: List[torch.Tensor] = []

            for view_index in range(num_views):
                reference_image = sparse_images[batch_index:batch_index + 1, view_index].to(
                    device=device,
                    dtype=self._model_dtype,
                )
                reference_pose = sparse_poses[batch_index:batch_index + 1, view_index].to(
                    device=device,
                    dtype=torch.float32,
                )
                reference_intrinsics = normalized_sparse_intrinsics[batch_index:batch_index + 1, view_index]
                if sparse_depth_maps is not None:
                    reference_depth_map = sparse_depth_maps[batch_index:batch_index + 1, view_index].to(
                        device=device,
                        dtype=torch.float32,
                    )
                else:
                    reference_depth_map = self.condition_preprocessor.estimate_depth(
                        reference_image.to(dtype=torch.float32)
                    ).float()
                warped_image, valid_mask = self.condition_preprocessor.perspective_warp(
                    image=reference_image,
                    depth=batch_target_depth,
                    K=batch_target_intrinsics,
                    ref_pose=reference_pose,
                    target_pose=batch_target_pose,
                    ref_K=reference_intrinsics,
                    target_K=batch_target_intrinsics,
                    reference_depth_map=reference_depth_map,
                )
                warped_images_per_view.append(warped_image[0])
                valid_masks_per_view.append(valid_mask[0])
                reference_depth_maps_per_view.append(reference_depth_map[0])

            stacked_warped_images = torch.stack(warped_images_per_view, dim=0)  # [V, 3, H, W]
            stacked_valid_masks = torch.stack(valid_masks_per_view, dim=0)      # [V, 1, H, W]
            nearest_reference_depth_map = reference_depth_maps_per_view[int(nearest_view_indices[batch_index].item())]
            binary_valid_masks = (stacked_valid_masks > 1e-6).float()
            union_valid_mask_before_distance_weight = binary_valid_masks.amax(dim=0)
            per_view_valid_coverage = binary_valid_masks.mean(dim=(1, 2, 3))
            raw_inverse_distance_weights = self._normalize_inverse_distance_weights(
                view_distances[batch_index]
            )
            coverage_distance_weights = self._apply_distance_weight_curve(
                raw_inverse_distance_weights,
                weight_floor=float(
                    getattr(self, "support_coverage_distance_weight_floor", 0.25)
                ),
                weight_gamma=float(
                    getattr(self, "support_coverage_distance_weight_gamma", 0.50)
                ),
            )
            coverage_distance_weight_blend = float(
                min(
                    max(
                        getattr(self, "support_coverage_distance_weight_blend", 0.75),
                        0.0,
                    ),
                    1.0,
                )
            )
            blended_coverage_distance_weights = torch.lerp(
                torch.ones_like(coverage_distance_weights),
                coverage_distance_weights,
                coverage_distance_weight_blend,
            )
            legacy_per_view_support_scores = (
                stacked_valid_masks
                * raw_inverse_distance_weights.view(num_views, 1, 1, 1)
            )
            per_view_support_scores = (
                stacked_valid_masks
                * blended_coverage_distance_weights.view(num_views, 1, 1, 1)
            )
            fusion_distance_weights = self._apply_distance_weight_curve(
                raw_inverse_distance_weights,
                weight_floor=float(
                    getattr(self, "support_fusion_distance_weight_floor", 0.10)
                ),
                weight_gamma=float(
                    getattr(self, "support_fusion_distance_weight_gamma", 0.75)
                ),
            )
            stabilized_support_scores = (
                stacked_valid_masks
                * fusion_distance_weights.view(num_views, 1, 1, 1)
            ).clamp_min(0.0)
            support_fusion_topk = max(1, min(int(getattr(self, "support_fusion_topk", 3)), num_views))
            if support_fusion_topk < num_views:
                kth_support_score = torch.topk(
                    stabilized_support_scores.squeeze(1),
                    k=support_fusion_topk,
                    dim=0,
                ).values[-1:].unsqueeze(1)
                stabilized_support_scores = stabilized_support_scores * (
                    stabilized_support_scores >= kth_support_score
                ).float()
            support_fusion_temperature = float(
                max(getattr(self, "support_fusion_temperature", 0.35), 1e-4)
            )
            stabilized_support_scores = stabilized_support_scores.pow(1.0 / support_fusion_temperature)
            normalized_support_weights = stabilized_support_scores / torch.clamp(
                stabilized_support_scores.sum(dim=0, keepdim=True),
                min=1e-6,
            )
            fused_warped_rgb = (normalized_support_weights * stacked_warped_images).sum(dim=0)
            per_view_rgb_residual = (
                stacked_warped_images.float()
                - fused_warped_rgb.unsqueeze(0).float()
            ).abs().mean(dim=1, keepdim=True)
            fusion_rgb_disagreement = (
                normalized_support_weights.float() * per_view_rgb_residual
            ).sum(dim=0).clamp(0.0, 1.0)

            prethreshold_valid_mask = self._aggregate_multiview_support_scores(
                stacked_valid_masks
            )
            legacy_raw_support_confidence = self._aggregate_multiview_support_scores(
                legacy_per_view_support_scores
            )
            raw_support_confidence = self._aggregate_multiview_support_scores(
                per_view_support_scores
            )
            support_quality_confidence = (
                raw_support_confidence.float()
                * (1.0 - fusion_rgb_disagreement.float())
            ).clamp(0.0, 1.0).to(dtype=raw_support_confidence.dtype)
            # 当前 stacked_valid_masks 已是 [0,1] 软几何置信度；
            # 先聚合多视图的互补中等支持，再要求联合 support 达到中等强度，
            # 避免 `amax` 只保留最强单视图而忽略多视图叠加价值。
            joint_valid_mask = (prethreshold_valid_mask >= 0.25).float()
            support_confidence = raw_support_confidence * joint_valid_mask
            fused_warped_rgb = fused_warped_rgb * joint_valid_mask

            fused_warped_rgb_list.append(fused_warped_rgb)
            joint_valid_mask_list.append(joint_valid_mask)
            support_confidence_list.append(support_confidence)
            prethreshold_valid_mask_list.append(prethreshold_valid_mask)
            raw_support_confidence_list.append(raw_support_confidence)
            legacy_raw_support_confidence_list.append(legacy_raw_support_confidence)
            support_quality_confidence_list.append(support_quality_confidence)
            support_fusion_rgb_disagreement_list.append(fusion_rgb_disagreement)
            union_valid_mask_before_distance_weight_list.append(union_valid_mask_before_distance_weight)
            per_view_valid_coverage_list.append(per_view_valid_coverage)
            per_view_distance_weight_list.append(
                blended_coverage_distance_weights.to(dtype=torch.float32)
            )
            per_view_raw_inverse_distance_weight_list.append(
                raw_inverse_distance_weights.to(dtype=torch.float32)
            )
            per_view_coverage_distance_weight_list.append(
                blended_coverage_distance_weights.to(dtype=torch.float32)
            )
            per_view_fusion_distance_weight_list.append(
                fusion_distance_weights.to(dtype=torch.float32)
            )
            nearest_reference_depth_list.append(nearest_reference_depth_map)

        return {
            "warped_rgb": torch.stack(fused_warped_rgb_list, dim=0),
            "valid_mask": torch.stack(joint_valid_mask_list, dim=0),
            "support_confidence": torch.stack(support_confidence_list, dim=0),
            "prethreshold_valid_mask": torch.stack(prethreshold_valid_mask_list, dim=0),
            "raw_support_confidence": torch.stack(raw_support_confidence_list, dim=0),
            "legacy_raw_support_confidence": torch.stack(
                legacy_raw_support_confidence_list,
                dim=0,
            ),
            "support_quality_confidence": torch.stack(
                support_quality_confidence_list,
                dim=0,
            ),
            "support_fusion_rgb_disagreement": torch.stack(
                support_fusion_rgb_disagreement_list,
                dim=0,
            ),
            "union_valid_mask_before_distance_weight": torch.stack(
                union_valid_mask_before_distance_weight_list,
                dim=0,
            ),
            "per_view_valid_coverage": torch.stack(per_view_valid_coverage_list, dim=0),
            "per_view_distance_weight": torch.stack(per_view_distance_weight_list, dim=0),
            "per_view_raw_inverse_distance_weight": torch.stack(
                per_view_raw_inverse_distance_weight_list,
                dim=0,
            ),
            "per_view_coverage_distance_weight": torch.stack(
                per_view_coverage_distance_weight_list,
                dim=0,
            ),
            "per_view_fusion_distance_weight": torch.stack(
                per_view_fusion_distance_weight_list,
                dim=0,
            ),
            "nearest_reference_image": nearest_reference_images,
            "nearest_reference_pose": nearest_reference_poses,
            "nearest_reference_intrinsics": nearest_reference_intrinsics,
            "nearest_reference_depth_map": torch.stack(nearest_reference_depth_list, dim=0),
        }
    
    def _encode_vae(self, image: torch.Tensor) -> torch.Tensor:
        """使用VAE编码图像到latent。
        
        强制使用FP32: SD 2.1 VAE在FP16下已知产生NaN/灰色输出。
        离线训练(gddn_trainer.py Stage 1)已强制FP32，在线推理同样需要。
        
        Args:
            image: [B, 3, H, W] 图像，范围[0, 1]
            
        Returns:
            [B, 4, H/8, W/8] latent
        """
        encode_device = (
            self._vae_encode_device
            if self._vae_encode_device is not None
            else self.vae.device
        )

        def _set_module_dtype(module: Optional[nn.Module], target_dtype: torch.dtype) -> None:
            if module is not None:
                module.to(dtype=target_dtype)

        # 强制FP32编码: 防止FP16精度不足导致latent退化
        vae_original_dtype = self.vae.dtype
        encoder_original_dtype = None
        quant_conv_original_dtype = None
        if self._vae_split_across_devices:
            if hasattr(self.vae, "encoder") and self.vae.encoder is not None:
                encoder_original_dtype = next(self.vae.encoder.parameters()).dtype
                _set_module_dtype(self.vae.encoder, torch.float32)
            if hasattr(self.vae, "quant_conv") and self.vae.quant_conv is not None:
                quant_conv_original_dtype = next(self.vae.quant_conv.parameters()).dtype
                _set_module_dtype(self.vae.quant_conv, torch.float32)
        else:
            self.vae.to(dtype=torch.float32)
        image = image.to(dtype=torch.float32, device=encode_device)
        
        # 转换到[-1, 1]范围
        image = image * 2.0 - 1.0
        
        with torch.no_grad():
            latent_dist = self.vae.encode(image).latent_dist
            latent = latent_dist.sample() * self.vae_scaling_factor
        
        # 恢复VAE原始dtype
        if self._vae_split_across_devices:
            if hasattr(self.vae, "encoder") and self.vae.encoder is not None and encoder_original_dtype is not None:
                _set_module_dtype(self.vae.encoder, encoder_original_dtype)
            if hasattr(self.vae, "quant_conv") and self.vae.quant_conv is not None and quant_conv_original_dtype is not None:
                _set_module_dtype(self.vae.quant_conv, quant_conv_original_dtype)
        else:
            self.vae.to(dtype=vae_original_dtype)
        return latent

    def _refine_support_masks_for_sampling(
        self,
        *,
        warp_valid_mask: torch.Tensor,
        support_confidence: Optional[torch.Tensor],
        raw_support_confidence: Optional[torch.Tensor],
        legacy_raw_support_confidence: Optional[torch.Tensor] = None,
        support_quality_confidence: Optional[torch.Tensor] = None,
        inpaint_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, float]]:
        """显式分离 true novel 空洞区与低置信已知区，并构造高置信软投影mask。"""
        refined_valid_mask = warp_valid_mask.float()
        if refined_valid_mask.dim() == 3:
            refined_valid_mask = refined_valid_mask.unsqueeze(1)
        if refined_valid_mask.shape[1] != 1:
            refined_valid_mask = refined_valid_mask[:, :1]
        refined_valid_mask = torch.nan_to_num(refined_valid_mask, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        continuous_support_confidence = raw_support_confidence
        if continuous_support_confidence is None:
            continuous_support_confidence = support_confidence
        if continuous_support_confidence is None:
            continuous_support_confidence = refined_valid_mask
        continuous_support_confidence = continuous_support_confidence.float()
        if continuous_support_confidence.dim() == 3:
            continuous_support_confidence = continuous_support_confidence.unsqueeze(1)
        if continuous_support_confidence.shape[1] != 1:
            continuous_support_confidence = continuous_support_confidence[:, :1]
        if continuous_support_confidence.shape[-2:] != refined_valid_mask.shape[-2:]:
            continuous_support_confidence = F.interpolate(
                continuous_support_confidence,
                size=refined_valid_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        continuous_support_confidence = torch.nan_to_num(
            continuous_support_confidence,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        def _align_optional_confidence(
            confidence_tensor: Optional[torch.Tensor],
        ) -> Optional[torch.Tensor]:
            if confidence_tensor is None:
                return None
            aligned_confidence = confidence_tensor.float()
            if aligned_confidence.dim() == 3:
                aligned_confidence = aligned_confidence.unsqueeze(1)
            if aligned_confidence.shape[1] != 1:
                aligned_confidence = aligned_confidence[:, :1]
            if aligned_confidence.shape[-2:] != refined_valid_mask.shape[-2:]:
                aligned_confidence = F.interpolate(
                    aligned_confidence,
                    size=refined_valid_mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.nan_to_num(
                aligned_confidence,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        aligned_legacy_support_confidence = _align_optional_confidence(
            legacy_raw_support_confidence
        )
        aligned_support_quality_confidence = _align_optional_confidence(
            support_quality_confidence
        )

        editable_support_confidence_threshold = float(
            max(
                0.0,
                min(
                    getattr(self, "editable_support_confidence_threshold", 0.20),
                    getattr(self, "support_projection_confidence_threshold", 0.45),
                ),
            )
        )
        support_projection_confidence_threshold = float(
            max(
                editable_support_confidence_threshold,
                min(getattr(self, "support_projection_confidence_threshold", 0.45), 0.95),
            )
        )
        low_confidence_known_mask = (
            (refined_valid_mask > 0.5).float()
            * (continuous_support_confidence < editable_support_confidence_threshold).float()
        )

        if inpaint_mask is None:
            refined_inpaint_mask = torch.clamp(1.0 - refined_valid_mask, min=0.0, max=1.0)
        else:
            refined_inpaint_mask = inpaint_mask.float()
            if refined_inpaint_mask.dim() == 3:
                refined_inpaint_mask = refined_inpaint_mask.unsqueeze(1)
            if refined_inpaint_mask.shape[1] != 1:
                refined_inpaint_mask = refined_inpaint_mask[:, :1]
            if refined_inpaint_mask.shape[-2:] != refined_valid_mask.shape[-2:]:
                refined_inpaint_mask = F.interpolate(
                    refined_inpaint_mask,
                    size=refined_valid_mask.shape[-2:],
                    mode="nearest",
                )
            refined_inpaint_mask = torch.nan_to_num(
                refined_inpaint_mask,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        true_novel_inpaint_mask = refined_inpaint_mask.clamp(0.0, 1.0)

        refined_support_confidence = continuous_support_confidence * refined_valid_mask
        refined_hole_ratio = float(true_novel_inpaint_mask.mean().item())
        projection_strength_scale = 1.0
        effective_support_projection_confidence_threshold = support_projection_confidence_threshold
        relax_hole_ratio_threshold = float(
            min(max(getattr(self, "support_projection_relax_hole_ratio_threshold", 0.60), 0.0), 0.99)
        )
        relax_progress = 0.0
        if refined_hole_ratio >= relax_hole_ratio_threshold:
            relax_progress = min(
                1.0,
                max(
                    0.0,
                    (refined_hole_ratio - relax_hole_ratio_threshold)
                    / max(1.0 - relax_hole_ratio_threshold, 1e-6),
                ),
            )
            effective_support_projection_confidence_threshold = max(
                editable_support_confidence_threshold,
                support_projection_confidence_threshold
                - (support_projection_confidence_threshold - editable_support_confidence_threshold)
                * 0.75
                * relax_progress,
            )

        def _project_support_confidence(
            confidence_tensor: torch.Tensor,
        ) -> torch.Tensor:
            projected_support_mask = (
                (confidence_tensor - effective_support_projection_confidence_threshold)
                / max(1.0 - effective_support_projection_confidence_threshold, 1e-6)
            ).clamp(0.0, 1.0)
            projected_support_mask = projected_support_mask * refined_valid_mask
            if relax_progress > 0.0:
                relaxed_support_projection_mask = (
                    (confidence_tensor - editable_support_confidence_threshold)
                    / max(1.0 - editable_support_confidence_threshold, 1e-6)
                ).clamp(0.0, 1.0)
                relaxed_support_projection_mask = (
                    relaxed_support_projection_mask * refined_valid_mask
                )
                projected_support_mask = torch.maximum(
                    projected_support_mask,
                    relaxed_support_projection_mask * projection_strength_scale,
                )
            return projected_support_mask.clamp(0.0, 1.0)

        if relax_progress > 0.0:
            projection_strength_scale = max(
                float(getattr(self, "support_projection_min_scale", 0.55)),
                1.0 - 0.45 * relax_progress,
            )
        support_projection_mask = _project_support_confidence(continuous_support_confidence)
        legacy_support_projection_mask = (
            _project_support_confidence(aligned_legacy_support_confidence)
            if aligned_legacy_support_confidence is not None
            else None
        )
        quality_gated_support_projection_mask = (
            _project_support_confidence(aligned_support_quality_confidence)
            if aligned_support_quality_confidence is not None
            else None
        )
        support_confidence_optimism_gap = (
            (continuous_support_confidence - aligned_support_quality_confidence).clamp(0.0, 1.0)
            * refined_valid_mask
            if aligned_support_quality_confidence is not None
            else None
        )
        newly_added_projection_mask = (
            (support_projection_mask - legacy_support_projection_mask).clamp(0.0, 1.0)
            * refined_valid_mask
            if legacy_support_projection_mask is not None
            else None
        )
        quality_suspect_projection_mask = (
            (support_projection_mask - quality_gated_support_projection_mask).clamp(0.0, 1.0)
            * refined_valid_mask
            if quality_gated_support_projection_mask is not None
            else None
        )
        newly_added_quality_suspect_mask = (
            torch.minimum(newly_added_projection_mask, quality_suspect_projection_mask)
            if newly_added_projection_mask is not None
            and quality_suspect_projection_mask is not None
            else None
        )
        pre_counterfactual_support_projection_mask = support_projection_mask.clamp(0.0, 1.0)
        counterfactual_support_projection_mask = None
        counterfactual_support_projection_delta_mask = None
        counterfactual_applied_mask = None
        counterfactual_mode = str(
            getattr(self, "support_projection_counterfactual_mode", "none")
        ).strip().lower()
        counterfactual_strength = float(
            getattr(self, "support_projection_counterfactual_strength", 1.0)
        )
        counterfactual_strength = min(max(counterfactual_strength, 0.0), 1.0)
        counterfactual_effective_strength = (
            1.0 if counterfactual_mode == "rollback" else counterfactual_strength
        )
        if (
            counterfactual_mode in ("rollback", "attenuate")
            and newly_added_quality_suspect_mask is not None
        ):
            counterfactual_target_support_projection_mask = (
                quality_gated_support_projection_mask
                if quality_gated_support_projection_mask is not None
                else legacy_support_projection_mask
            )
            if (
                counterfactual_target_support_projection_mask is not None
                and counterfactual_effective_strength > 0.0
            ):
                counterfactual_applied_mask = (
                    newly_added_quality_suspect_mask.clamp(0.0, 1.0)
                    * counterfactual_effective_strength
                ).clamp(0.0, 1.0)
                counterfactual_support_projection_mask = torch.lerp(
                    pre_counterfactual_support_projection_mask,
                    counterfactual_target_support_projection_mask.clamp(0.0, 1.0),
                    counterfactual_applied_mask,
                ).clamp(0.0, 1.0)
                counterfactual_support_projection_delta_mask = (
                    pre_counterfactual_support_projection_mask
                    - counterfactual_support_projection_mask
                ).clamp(0.0, 1.0)
                support_projection_mask = counterfactual_support_projection_mask
        if counterfactual_support_projection_mask is None:
            counterfactual_mode = "none"
            counterfactual_effective_strength = 0.0

        joint_valid_but_below_projection_threshold_mask = (
            (refined_valid_mask > 0.5).float()
            * (support_projection_mask <= 1e-6).float()
        )
        projection_coverage_ratio = float(support_projection_mask.mean().item())
        return {
            "warp_valid_mask": refined_valid_mask,
            "support_confidence": refined_support_confidence,
            "inpaint_mask": true_novel_inpaint_mask,
            "true_novel_inpaint_mask": true_novel_inpaint_mask,
            "support_projection_mask": support_projection_mask.clamp(0.0, 1.0),
            "pre_counterfactual_support_projection_mask": (
                pre_counterfactual_support_projection_mask.clamp(0.0, 1.0)
            ),
            "legacy_support_projection_mask": (
                legacy_support_projection_mask.clamp(0.0, 1.0)
                if legacy_support_projection_mask is not None
                else None
            ),
            "quality_gated_support_projection_mask": (
                quality_gated_support_projection_mask.clamp(0.0, 1.0)
                if quality_gated_support_projection_mask is not None
                else None
            ),
            "newly_added_projection_mask": (
                newly_added_projection_mask.clamp(0.0, 1.0)
                if newly_added_projection_mask is not None
                else None
            ),
            "quality_suspect_projection_mask": (
                quality_suspect_projection_mask.clamp(0.0, 1.0)
                if quality_suspect_projection_mask is not None
                else None
            ),
            "newly_added_quality_suspect_mask": (
                newly_added_quality_suspect_mask.clamp(0.0, 1.0)
                if newly_added_quality_suspect_mask is not None
                else None
            ),
            "counterfactual_support_projection_mask": (
                counterfactual_support_projection_mask.clamp(0.0, 1.0)
                if counterfactual_support_projection_mask is not None
                else None
            ),
            "counterfactual_support_projection_delta_mask": (
                counterfactual_support_projection_delta_mask.clamp(0.0, 1.0)
                if counterfactual_support_projection_delta_mask is not None
                else None
            ),
            "counterfactual_applied_mask": (
                counterfactual_applied_mask.clamp(0.0, 1.0)
                if counterfactual_applied_mask is not None
                else None
            ),
            "support_quality_confidence": aligned_support_quality_confidence,
            "support_confidence_optimism_gap": support_confidence_optimism_gap,
            "joint_valid_but_below_projection_threshold_mask": (
                joint_valid_but_below_projection_threshold_mask
            ),
            "joint_valid_but_below_projection_threshold_coverage": float(
                joint_valid_but_below_projection_threshold_mask.mean().item()
            ),
            "legacy_support_projection_coverage": float(
                legacy_support_projection_mask.mean().item()
            ) if legacy_support_projection_mask is not None else None,
            "quality_gated_support_projection_coverage": float(
                quality_gated_support_projection_mask.mean().item()
            ) if quality_gated_support_projection_mask is not None else None,
            "newly_added_projection_coverage": float(
                newly_added_projection_mask.mean().item()
            ) if newly_added_projection_mask is not None else None,
            "quality_suspect_projection_coverage": float(
                quality_suspect_projection_mask.mean().item()
            ) if quality_suspect_projection_mask is not None else None,
            "newly_added_quality_suspect_coverage": float(
                newly_added_quality_suspect_mask.mean().item()
            ) if newly_added_quality_suspect_mask is not None else None,
            "support_confidence_optimism_gap_coverage": float(
                support_confidence_optimism_gap.mean().item()
            ) if support_confidence_optimism_gap is not None else None,
            "support_projection_counterfactual_mode": counterfactual_mode,
            "support_projection_counterfactual_strength": float(
                counterfactual_effective_strength
            ),
            "counterfactual_support_projection_coverage": float(
                counterfactual_support_projection_mask.mean().item()
            ) if counterfactual_support_projection_mask is not None else None,
            "counterfactual_support_projection_delta_coverage": float(
                counterfactual_support_projection_delta_mask.mean().item()
            ) if counterfactual_support_projection_delta_mask is not None else None,
            "counterfactual_applied_mask_coverage": float(
                counterfactual_applied_mask.mean().item()
            ) if counterfactual_applied_mask is not None else None,
            "low_confidence_known_mask": low_confidence_known_mask,
            "editable_low_confidence_mask": low_confidence_known_mask,
            "projection_coverage_ratio": projection_coverage_ratio,
            "hole_ratio": refined_hole_ratio,
            "projection_strength_scale": float(projection_strength_scale),
            "effective_support_projection_confidence_threshold": float(
                effective_support_projection_confidence_threshold
            ),
        }

    def _compute_dual_budget_inpaint_schedule(
        self,
        *,
        true_novel_hole_ratio: float,
        support_projection_ratio: float,
        support_confidence_ratio: Optional[float] = None,
    ) -> Dict[str, float]:
        """基于可编辑预算与support保真预算，分离计算采样schedule。"""
        clamped_true_novel_hole_ratio = float(min(max(true_novel_hole_ratio, 0.0), 1.0))
        clamped_support_projection_ratio = float(min(max(support_projection_ratio, 0.0), 1.0))
        if support_confidence_ratio is None:
            clamped_support_confidence_ratio = clamped_support_projection_ratio
        else:
            clamped_support_confidence_ratio = float(min(max(support_confidence_ratio, 0.0), 1.0))

        editable_difficulty_signal = float(
            min(
                max(
                    getattr(self, "editable_schedule_hole_ratio_weight", 0.45) * clamped_true_novel_hole_ratio
                    + getattr(self, "editable_schedule_projection_gap_weight", 0.55)
                    * (1.0 - clamped_support_projection_ratio),
                    0.0,
                ),
                1.0,
            )
        )

        easy_threshold = float(getattr(self, "editable_schedule_easy_threshold", 0.35))
        moderate_threshold = float(getattr(self, "editable_schedule_moderate_threshold", 0.55))
        hard_threshold = float(getattr(self, "editable_schedule_hard_threshold", 0.70))
        extreme_threshold = float(getattr(self, "editable_schedule_extreme_threshold", 0.85))

        if editable_difficulty_signal < easy_threshold:
            scheduled_img2img_strength = 0.25
        elif editable_difficulty_signal < moderate_threshold:
            moderate_progress = (
                (editable_difficulty_signal - easy_threshold)
                / max(moderate_threshold - easy_threshold, 1e-6)
            )
            scheduled_img2img_strength = 0.25 + moderate_progress * (0.40 - 0.25)
        elif editable_difficulty_signal < hard_threshold:
            hard_progress = (
                (editable_difficulty_signal - moderate_threshold)
                / max(hard_threshold - moderate_threshold, 1e-6)
            )
            scheduled_img2img_strength = 0.40 + hard_progress * (0.82 - 0.40)
        elif editable_difficulty_signal < extreme_threshold:
            scheduled_img2img_strength = 0.82
        else:
            extreme_progress = (
                (editable_difficulty_signal - extreme_threshold)
                / max(1.0 - extreme_threshold, 1e-6)
            )
            scheduled_img2img_strength = 0.82 + extreme_progress * (0.90 - 0.82)

        support_preserve_budget_signal = float(
            min(
                max(
                    getattr(self, "support_preserve_projection_gap_weight", 0.75)
                    * (1.0 - clamped_support_projection_ratio)
                    + getattr(self, "support_preserve_confidence_gap_weight", 0.25)
                    * (1.0 - clamped_support_confidence_ratio),
                    0.0,
                ),
                1.0,
            )
        )

        minimum_support_step_offset = int(
            max(1, getattr(self, "minimum_masked_dual_path_support_step_offset", 4))
        )
        maximum_support_step_offset = int(
            max(minimum_support_step_offset, getattr(self, "maximum_masked_dual_path_support_step_offset", 8))
        )
        support_step_progress = 0.0
        if support_preserve_budget_signal < 0.55:
            support_step_progress = 0.0
        elif support_preserve_budget_signal < 0.75:
            support_step_progress = 0.5 * (
                (support_preserve_budget_signal - 0.55)
                / max(0.75 - 0.55, 1e-6)
            )
        elif support_preserve_budget_signal < 0.90:
            support_step_progress = 0.5 + 0.5 * (
                (support_preserve_budget_signal - 0.75)
                / max(0.90 - 0.75, 1e-6)
            )
        else:
            support_step_progress = 1.0
        support_step_progress = float(min(max(support_step_progress, 0.0), 1.0))

        planned_support_step_offset = int(
            round(
                minimum_support_step_offset
                + support_step_progress
                * max(maximum_support_step_offset - minimum_support_step_offset, 0)
            )
        )
        planned_support_step_offset = int(
            min(max(planned_support_step_offset, minimum_support_step_offset), maximum_support_step_offset)
        )

        return {
            "editable_difficulty_signal": editable_difficulty_signal,
            "support_preserve_budget_signal": support_preserve_budget_signal,
            "img2img_strength": float(min(max(scheduled_img2img_strength, 0.0), 1.0)),
            "support_step_progress": support_step_progress,
            "support_step_offset": planned_support_step_offset,
            "support_projection_ratio": clamped_support_projection_ratio,
            "support_confidence_ratio": clamped_support_confidence_ratio,
        }
    
    def _decode_vae(self, latent: torch.Tensor, *, track_grad: bool = False) -> torch.Tensor:
        """使用VAE解码latent到图像。
        
        强制使用FP32: SD 2.1 VAE在FP16下已知产生灰色输出。
        这是灰色伪视图的最可能直接原因。
        
        Args:
            latent: [B, 4, H/8, W/8] latent
            
        Returns:
            [B, 3, H, W] 图像，范围[0, 1]
        """
        decode_device = (
            self._vae_decode_device
            if self._vae_decode_device is not None
            else self.vae.device
        )

        def _set_module_dtype(module: Optional[nn.Module], target_dtype: torch.dtype) -> None:
            if module is not None:
                module.to(dtype=target_dtype)

        # 强制FP32解码: 防止FP16精度不足导致灰色输出
        vae_original_dtype = self.vae.dtype
        decoder_original_dtype = None
        post_quant_conv_original_dtype = None
        if self._vae_split_across_devices:
            if hasattr(self.vae, "decoder") and self.vae.decoder is not None:
                decoder_original_dtype = next(self.vae.decoder.parameters()).dtype
                _set_module_dtype(self.vae.decoder, torch.float32)
            if hasattr(self.vae, "post_quant_conv") and self.vae.post_quant_conv is not None:
                post_quant_conv_original_dtype = next(self.vae.post_quant_conv.parameters()).dtype
                _set_module_dtype(self.vae.post_quant_conv, torch.float32)
        elif vae_original_dtype != torch.float32:
            self.vae.to(dtype=torch.float32)
        latent = latent.to(dtype=torch.float32, device=decode_device)
        latent = latent / self.vae_scaling_factor
        decode_chunk_size = int(
            getattr(
                self,
                "vae_decode_chunk_size",
                1 if latent.shape[0] > 1 else latent.shape[0],
            )
        )
        decode_chunk_size = max(1, decode_chunk_size)
        
        def _decode_latent_chunks(latent_tensor: torch.Tensor) -> torch.Tensor:
            if latent_tensor.shape[0] <= decode_chunk_size:
                return self.vae.decode(latent_tensor).sample
            decoded_chunks = []
            for chunk_start in range(0, latent_tensor.shape[0], decode_chunk_size):
                latent_chunk = latent_tensor[chunk_start:chunk_start + decode_chunk_size]
                decoded_chunk = self.vae.decode(latent_chunk).sample
                decoded_chunks.append(decoded_chunk)
            return torch.cat(decoded_chunks, dim=0)

        decode_context = torch.enable_grad() if track_grad else torch.no_grad()
        try:
            with decode_context:
                image = _decode_latent_chunks(latent)
        except torch.OutOfMemoryError:
            if track_grad:
                if self._vae_split_across_devices:
                    if hasattr(self.vae, "decoder") and self.vae.decoder is not None and decoder_original_dtype is not None:
                        _set_module_dtype(self.vae.decoder, decoder_original_dtype)
                    if hasattr(self.vae, "post_quant_conv") and self.vae.post_quant_conv is not None and post_quant_conv_original_dtype is not None:
                        _set_module_dtype(self.vae.post_quant_conv, post_quant_conv_original_dtype)
                elif vae_original_dtype != torch.float32:
                    self.vae.to(dtype=vae_original_dtype)
                raise
            if latent.is_cuda:
                torch.cuda.empty_cache()
            if hasattr(self.vae, "enable_tiling"):
                try:
                    self.vae.enable_tiling()
                except Exception:
                    pass
            if hasattr(self.vae, "enable_slicing"):
                try:
                    self.vae.enable_slicing()
                except Exception:
                    pass
            if not getattr(self, "_vae_decode_tiling_retry_logged", False):
                print("[GDDN] VAE FP32解码触发OOM，已自动启用tiling/slicing后重试")
                self._vae_decode_tiling_retry_logged = True
            with decode_context:
                image = _decode_latent_chunks(latent)
        finally:
            if self._vae_split_across_devices:
                if hasattr(self.vae, "decoder") and self.vae.decoder is not None and decoder_original_dtype is not None:
                    _set_module_dtype(self.vae.decoder, decoder_original_dtype)
                if hasattr(self.vae, "post_quant_conv") and self.vae.post_quant_conv is not None and post_quant_conv_original_dtype is not None:
                    _set_module_dtype(self.vae.post_quant_conv, post_quant_conv_original_dtype)
            elif vae_original_dtype != torch.float32:
                self.vae.to(dtype=vae_original_dtype)

        # 转换到[0, 1]范围
        image = (image + 1.0) / 2.0
        return image.clamp(0.0, 1.0)

    def decode_latent_to_rgb(self, latent: torch.Tensor, *, track_grad: bool = False) -> torch.Tensor:
        """兼容接口: 解码latent到RGB。"""
        return self._decode_vae(latent, track_grad=track_grad)
    
    def encode_rgb_to_latent(self, rgb: torch.Tensor) -> torch.Tensor:
        """兼容接口: 编码RGB到latent。"""
        return self._encode_vae(rgb)

    def _predict_x0_latent(
        self,
        noisy_latents: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
        alpha_fallback: float = 0.99,
    ) -> torch.Tensor:
        """根据当前扩散步的噪声预测恢复 x0 latent，使训练输入更接近推理时的最终 latent。"""
        scheduler = getattr(self, "noise_scheduler", None)
        if scheduler is not None and hasattr(scheduler, "alphas_cumprod"):
            alphas_cumprod = scheduler.alphas_cumprod
            if not torch.is_tensor(alphas_cumprod):
                alpha_values = torch.tensor(
                    alphas_cumprod,
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype,
                )
            else:
                alpha_values = alphas_cumprod.to(
                    device=noisy_latents.device,
                    dtype=noisy_latents.dtype,
                )
            step_indices = timesteps.to(device=noisy_latents.device)
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
        sqrt_alpha = torch.sqrt(torch.clamp(alpha_t, min=1e-12))
        sqrt_one_minus_alpha = torch.sqrt(torch.clamp(1.0 - alpha_t, min=1e-12))
        aligned_noise_pred = noise_pred.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
        prediction_type = "epsilon"
        scheduler_config = getattr(scheduler, "config", None)
        if scheduler_config is not None:
            prediction_type = getattr(scheduler_config, "prediction_type", prediction_type)
            if isinstance(scheduler_config, dict):
                prediction_type = scheduler_config.get("prediction_type", prediction_type)
        prediction_type = str(prediction_type)
        if prediction_type == "v_prediction":
            x0_latent = sqrt_alpha * noisy_latents - sqrt_one_minus_alpha * aligned_noise_pred
        elif prediction_type == "sample":
            x0_latent = aligned_noise_pred
        else:
            x0_latent = (noisy_latents - sqrt_one_minus_alpha * aligned_noise_pred) / sqrt_alpha
        return torch.nan_to_num(x0_latent.clamp(-4.0, 4.0))

    def _compute_effect_maps(
        self,
        *,
        teacher_rgb: torch.Tensor,
        render_rgb: Optional[torch.Tensor],
        warp_rgb: Optional[torch.Tensor],
        warp_confidence: Optional[torch.Tensor],
        depth_map: Optional[torch.Tensor],
        exist_mask: Optional[torch.Tensor] = None,
        novel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if exist_mask is None:
            if depth_map is not None:
                effect_depth_map = depth_map
                if effect_depth_map.dim() == 3:
                    effect_depth_map = effect_depth_map.unsqueeze(1)
                if effect_depth_map.shape[1] > 1:
                    effect_depth_map = effect_depth_map[:, :1]
                exist_mask = (effect_depth_map > 1e-6).float()
            elif warp_confidence is not None:
                exist_mask = (warp_confidence > 0.05).float()
        if novel_mask is None and exist_mask is not None:
            novel_mask = torch.clamp(1.0 - exist_mask, min=0.0, max=1.0)
        return self.uncertainty_head(
            teacher_rgb=teacher_rgb,
            render_rgb=render_rgb,
            warp_rgb=warp_rgb,
            warp_confidence=warp_confidence,
            exist_mask=exist_mask,
            novel_mask=novel_mask,
        )

    def _build_exist_novel_masks_from_signals(
        self,
        *,
        depth_map: Optional[torch.Tensor],
        warp_confidence: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        exist_mask = None
        if depth_map is not None:
            exist_mask = depth_map
            if exist_mask.dim() == 3:
                exist_mask = exist_mask.unsqueeze(1)
            if exist_mask.shape[1] > 1:
                exist_mask = exist_mask[:, :1]
            exist_mask = (exist_mask > 1e-6).float()
        elif warp_confidence is not None:
            exist_mask = (warp_confidence > 0.05).float()

        if exist_mask is not None and exist_mask.shape[-2:] != target_hw:
            exist_mask = F.interpolate(exist_mask, size=target_hw, mode="nearest")

        novel_mask = None
        if exist_mask is not None:
            novel_mask = torch.clamp(1.0 - exist_mask, min=0.0, max=1.0)
            if novel_mask.sum() < 1:
                novel_mask = None
        return exist_mask, novel_mask

    def _build_refined_support_signals_from_confidence(
        self,
        *,
        warp_confidence: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Dict[str, Optional[torch.Tensor]]:
        if warp_confidence is None:
            return {
                "warp_valid_mask": None,
                "support_confidence": None,
                "inpaint_mask": None,
                "true_novel_inpaint_mask": None,
                "support_projection_mask": None,
                "low_confidence_known_mask": None,
                "editable_low_confidence_mask": None,
            }

        normalized_support_confidence = warp_confidence
        if normalized_support_confidence.dim() == 3:
            normalized_support_confidence = normalized_support_confidence.unsqueeze(1)
        if normalized_support_confidence.shape[1] != 1:
            normalized_support_confidence = normalized_support_confidence[:, :1]
        if normalized_support_confidence.shape[-2:] != target_hw:
            normalized_support_confidence = F.interpolate(
                normalized_support_confidence.float(),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )
        normalized_support_confidence = torch.nan_to_num(
            normalized_support_confidence.float(),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

        warp_valid_mask = (normalized_support_confidence > 1e-6).float()
        initial_inpaint_mask = torch.clamp(1.0 - warp_valid_mask, min=0.0, max=1.0)
        refined_support_outputs = self._refine_support_masks_for_sampling(
            warp_valid_mask=warp_valid_mask,
            support_confidence=normalized_support_confidence,
            raw_support_confidence=normalized_support_confidence,
            inpaint_mask=initial_inpaint_mask,
        )
        return {
            "warp_valid_mask": refined_support_outputs.get("warp_valid_mask"),
            "support_confidence": refined_support_outputs.get("support_confidence"),
            "inpaint_mask": refined_support_outputs.get("inpaint_mask"),
            "true_novel_inpaint_mask": refined_support_outputs.get("true_novel_inpaint_mask"),
            "support_projection_mask": refined_support_outputs.get("support_projection_mask"),
            "low_confidence_known_mask": refined_support_outputs.get("low_confidence_known_mask"),
            "editable_low_confidence_mask": refined_support_outputs.get("editable_low_confidence_mask"),
        }

    def _compose_support_preserving_image(
        self,
        *,
        teacher_rgb: torch.Tensor,
        support_rgb: Optional[torch.Tensor],
        exist_mask: Optional[torch.Tensor],
        novel_mask: Optional[torch.Tensor],
        support_blend_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_support_mask = support_blend_mask
        if effective_support_mask is None:
            effective_support_mask = exist_mask
        if support_rgb is None or effective_support_mask is None:
            return teacher_rgb
        composed_support_rgb = support_rgb
        if composed_support_rgb.dim() == 3:
            composed_support_rgb = composed_support_rgb.unsqueeze(0)
        if composed_support_rgb.shape[-2:] != teacher_rgb.shape[-2:]:
            composed_support_rgb = F.interpolate(
                composed_support_rgb,
                size=teacher_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        composed_exist_mask = effective_support_mask
        if composed_exist_mask.shape[-2:] != teacher_rgb.shape[-2:]:
            composed_exist_mask = F.interpolate(
                composed_exist_mask,
                size=teacher_rgb.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        composed_exist_mask = torch.nan_to_num(
            composed_exist_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if support_blend_mask is not None:
            return (
                teacher_rgb * (1.0 - composed_exist_mask)
                + composed_support_rgb * composed_exist_mask
            )
        composed_novel_mask = novel_mask
        if composed_novel_mask is None:
            composed_novel_mask = torch.clamp(1.0 - composed_exist_mask, min=0.0, max=1.0)
        elif composed_novel_mask.shape[-2:] != teacher_rgb.shape[-2:]:
            composed_novel_mask = F.interpolate(
                composed_novel_mask,
                size=teacher_rgb.shape[-2:],
                mode="nearest",
        )
        return composed_novel_mask * teacher_rgb + composed_exist_mask * composed_support_rgb

    def _build_explicit_frontier_proposal(
        self,
        *,
        teacher_rgb: torch.Tensor,
        base_rgb: Optional[torch.Tensor],
        support_confidence: Optional[torch.Tensor],
        support_projection_mask: Optional[torch.Tensor],
        inpaint_mask: Optional[torch.Tensor],
        frontier_mask: Optional[torch.Tensor],
        verification_prior: Optional[torch.Tensor],
        accum_render: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        target_hw = teacher_rgb.shape[-2:]
        batch_size = teacher_rgb.shape[0]
        target_device = teacher_rgb.device
        target_dtype = teacher_rgb.dtype

        def _align_rgb_tensor(
            rgb_tensor: Optional[torch.Tensor],
            fallback_rgb: torch.Tensor,
        ) -> torch.Tensor:
            if rgb_tensor is None:
                return fallback_rgb
            aligned_rgb_tensor = rgb_tensor
            if aligned_rgb_tensor.dim() == 3:
                aligned_rgb_tensor = aligned_rgb_tensor.unsqueeze(0)
            if aligned_rgb_tensor.shape[1] > 3:
                aligned_rgb_tensor = aligned_rgb_tensor[:, :3]
            if aligned_rgb_tensor.shape[-2:] != target_hw:
                aligned_rgb_tensor = F.interpolate(
                    aligned_rgb_tensor.float(),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.nan_to_num(
                aligned_rgb_tensor.to(device=target_device, dtype=target_dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        def _align_mask_tensor(
            mask_tensor: Optional[torch.Tensor],
        ) -> torch.Tensor:
            if mask_tensor is None:
                return teacher_rgb.new_zeros((batch_size, 1, target_hw[0], target_hw[1]))
            aligned_mask_tensor = mask_tensor
            if aligned_mask_tensor.dim() == 3:
                aligned_mask_tensor = aligned_mask_tensor.unsqueeze(1)
            if aligned_mask_tensor.shape[1] > 1:
                aligned_mask_tensor = aligned_mask_tensor[:, :1]
            if aligned_mask_tensor.shape[-2:] != target_hw:
                aligned_mask_tensor = F.interpolate(
                    aligned_mask_tensor.float(),
                    size=target_hw,
                    mode="bilinear",
                    align_corners=False,
                )
            return torch.nan_to_num(
                aligned_mask_tensor.to(device=target_device, dtype=target_dtype),
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

        aligned_base_rgb = _align_rgb_tensor(base_rgb, teacher_rgb)
        aligned_support_confidence = _align_mask_tensor(
            support_confidence if support_confidence is not None else accum_render
        )
        aligned_support_projection_mask = _align_mask_tensor(support_projection_mask)
        aligned_inpaint_mask = _align_mask_tensor(inpaint_mask)
        aligned_frontier_mask = _align_mask_tensor(frontier_mask)
        aligned_verification_prior = _align_mask_tensor(verification_prior)
        aligned_accum_render = _align_mask_tensor(accum_render)
        teacher_base_abs_delta = (teacher_rgb - aligned_base_rgb).abs().mean(dim=1, keepdim=True)

        proposal_focus_mask = torch.maximum(aligned_inpaint_mask, aligned_frontier_mask)
        proposal_focus_mask = torch.maximum(proposal_focus_mask, aligned_verification_prior)
        if float(proposal_focus_mask.sum().item()) <= 1e-6:
            proposal_focus_mask = aligned_inpaint_mask

        proposal_feature_map = torch.cat(
            [
                teacher_rgb,
                aligned_base_rgb,
                teacher_base_abs_delta,
                aligned_support_confidence,
                aligned_support_projection_mask,
                aligned_inpaint_mask,
                proposal_focus_mask,
                aligned_verification_prior,
                aligned_accum_render,
            ],
            dim=1,
        )
        proposal_outputs = self.frontier_proposal_head(proposal_feature_map)
        proposal_residual_rgb = proposal_outputs["proposal_residual_rgb"]
        if str(getattr(self, "proposal_residual_source", "head")).strip().lower() == "teacher_delta":
            proposal_residual_rgb = teacher_rgb - aligned_base_rgb
        proposal_warp_error_logit = proposal_outputs.get("proposal_warp_error_logit")
        proposal_repairability_logit = proposal_outputs.get("proposal_repairability_logit")
        proposal_verification_logit = proposal_outputs.get("proposal_verification_logit")
        proposal_acceptance_logit = proposal_outputs.get("proposal_acceptance_logit")
        proposal_warp_error_confidence = proposal_outputs["proposal_warp_error_confidence"]
        proposal_repairability_confidence = proposal_outputs["proposal_repairability_confidence"]
        proposal_verification_confidence = proposal_outputs["proposal_verification_confidence"]
        proposal_acceptance_confidence = proposal_outputs["proposal_acceptance_confidence"]
        proposal_confidence = proposal_acceptance_confidence
        proposal_warp_error_gate_power = max(
            float(getattr(self, "proposal_warp_error_gate_power", 1.0)),
            0.1,
        )
        proposal_repairability_gate_power = max(
            float(getattr(self, "proposal_repairability_gate_power", 1.0)),
            0.1,
        )
        proposal_verification_gate_power = max(
            float(getattr(self, "proposal_verification_gate_power", 1.0)),
            0.1,
        )
        proposal_gate_components = [
            proposal_warp_error_confidence.pow(proposal_warp_error_gate_power),
            proposal_repairability_confidence.pow(proposal_repairability_gate_power),
            proposal_verification_confidence.pow(proposal_verification_gate_power),
        ]
        proposal_component_gate = torch.clamp(
            (
                proposal_gate_components[0]
                * proposal_gate_components[1]
                * proposal_gate_components[2]
            ).pow(1.0 / 3.0),
            min=0.0,
            max=1.0,
        )
        proposal_acceptance_gate = torch.clamp(
            proposal_acceptance_confidence,
            min=0.0,
            max=1.0,
        )
        safe_confidence_threshold = min(
            max(float(getattr(self, "proposal_safe_confidence_threshold", 0.0)), 0.0),
            0.99,
        )
        if safe_confidence_threshold > 0.0:
            proposal_residual_gate = torch.clamp(
                (proposal_acceptance_gate - safe_confidence_threshold)
                / max(1.0 - safe_confidence_threshold, 1e-6),
                min=0.0,
                max=1.0,
            )
        else:
            proposal_residual_gate = proposal_acceptance_gate
        safe_confidence_gate_power = max(
            float(getattr(self, "proposal_safe_confidence_gate_power", 1.0)),
            0.1,
        )
        if abs(safe_confidence_gate_power - 1.0) > 1e-6:
            proposal_residual_gate = proposal_residual_gate.pow(
                safe_confidence_gate_power
            )
        proposal_soft_residual_gate = proposal_residual_gate
        if bool(getattr(self, "proposal_use_hard_acceptance_gate", False)):
            proposal_hard_residual_gate = (
                proposal_acceptance_gate >= safe_confidence_threshold
            ).to(dtype=proposal_soft_residual_gate.dtype)
            if bool(getattr(self, "proposal_hard_acceptance_straight_through", True)):
                proposal_residual_gate = (
                    proposal_hard_residual_gate.detach()
                    + proposal_soft_residual_gate
                    - proposal_soft_residual_gate.detach()
                )
            else:
                proposal_residual_gate = proposal_hard_residual_gate
        proposal_applied_residual_rgb = proposal_residual_rgb * proposal_residual_gate
        proposal_raw_rgb = torch.clamp(
            aligned_base_rgb + proposal_applied_residual_rgb,
            min=0.0,
            max=1.0,
        )
        return {
            "proposal_raw_rgb": proposal_raw_rgb,
            "proposal_residual_rgb": proposal_residual_rgb,
            "proposal_applied_residual_rgb": proposal_applied_residual_rgb,
            "proposal_confidence": proposal_confidence,
            "proposal_warp_error_logit": proposal_warp_error_logit,
            "proposal_repairability_logit": proposal_repairability_logit,
            "proposal_verification_logit": proposal_verification_logit,
            "proposal_acceptance_logit": proposal_acceptance_logit,
            "proposal_warp_error_confidence": proposal_warp_error_confidence,
            "proposal_repairability_confidence": proposal_repairability_confidence,
            "proposal_verification_confidence": proposal_verification_confidence,
            "proposal_acceptance_confidence": proposal_acceptance_confidence,
            "proposal_acceptance_gate": proposal_residual_gate,
            "proposal_soft_acceptance_gate": proposal_acceptance_gate,
            "proposal_component_gate": proposal_component_gate,
            "proposal_residual_gate": proposal_residual_gate,
            "proposal_focus_mask": proposal_focus_mask,
        }

    def _compute_masked_tensor_abs_mean_and_variance(
        self,
        *,
        value_tensor: Optional[torch.Tensor],
        mask_tensor: Optional[torch.Tensor],
    ) -> Tuple[Optional[float], Optional[float]]:
        if value_tensor is None or mask_tensor is None:
            return None, None
        if value_tensor.dim() != 4:
            return None, None
        aligned_mask_tensor = mask_tensor.to(
            device=value_tensor.device,
            dtype=value_tensor.dtype,
        )
        if aligned_mask_tensor.dim() == 3:
            aligned_mask_tensor = aligned_mask_tensor.unsqueeze(1)
        if aligned_mask_tensor.shape[1] != 1:
            aligned_mask_tensor = aligned_mask_tensor[:, :1]
        if aligned_mask_tensor.shape[-2:] != value_tensor.shape[-2:]:
            aligned_mask_tensor = F.interpolate(
                aligned_mask_tensor,
                size=value_tensor.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        aligned_mask_tensor = torch.nan_to_num(
            aligned_mask_tensor,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        if float(aligned_mask_tensor.sum().item()) <= 1e-6:
            return None, None
        expanded_mask_tensor = aligned_mask_tensor.expand_as(value_tensor)
        masked_value_tensor = torch.nan_to_num(
            value_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ) * expanded_mask_tensor
        valid_value_count = expanded_mask_tensor.sum().clamp_min(1.0)
        masked_abs_mean = float(masked_value_tensor.abs().sum().item() / valid_value_count.item())
        masked_mean = masked_value_tensor.sum() / valid_value_count
        masked_variance = float(
            (
                ((torch.nan_to_num(value_tensor, nan=0.0, posinf=0.0, neginf=0.0) - masked_mean) ** 2)
                * expanded_mask_tensor
            ).sum().item()
            / valid_value_count.item()
        )
        return masked_abs_mean, masked_variance

    def _compute_tensor_abs_mean_and_variance(
        self,
        *,
        value_tensor: Optional[torch.Tensor],
    ) -> Tuple[Optional[float], Optional[float]]:
        if value_tensor is None or value_tensor.dim() != 4:
            return None, None
        sanitized_value_tensor = torch.nan_to_num(
            value_tensor,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        return (
            float(sanitized_value_tensor.abs().mean().item()),
            float(sanitized_value_tensor.var(unbiased=False).item()),
        )

    def _collect_latent_region_statistics(
        self,
        *,
        metric_prefix: str,
        latent_tensor: Optional[torch.Tensor],
        support_mask: Optional[torch.Tensor],
        editable_mask: Optional[torch.Tensor],
        novel_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        latent_statistics: Dict[str, float] = {}
        overall_abs_mean, overall_variance = self._compute_tensor_abs_mean_and_variance(
            value_tensor=latent_tensor,
        )
        if overall_abs_mean is not None:
            latent_statistics[f"{metric_prefix}_overall_abs_mean"] = float(overall_abs_mean)
        if overall_variance is not None:
            latent_statistics[f"{metric_prefix}_overall_variance"] = float(overall_variance)
        for region_name, region_mask in (
            ("support", support_mask),
            ("editable", editable_mask),
            ("novel", novel_mask),
        ):
            region_abs_mean, region_variance = self._compute_masked_tensor_abs_mean_and_variance(
                value_tensor=latent_tensor,
                mask_tensor=region_mask,
            )
            if region_abs_mean is not None:
                latent_statistics[f"{metric_prefix}_{region_name}_abs_mean"] = float(region_abs_mean)
            if region_variance is not None:
                latent_statistics[f"{metric_prefix}_{region_name}_variance"] = float(region_variance)
        return latent_statistics

    def _summarize_scalar_sequence(
        self,
        *,
        metric_prefix: str,
        scalar_values: List[float],
    ) -> Dict[str, float]:
        if not scalar_values:
            return {}
        scalar_tensor = torch.tensor(scalar_values, dtype=torch.float32)
        return {
            f"{metric_prefix}_avg": float(scalar_tensor.mean().item()),
            f"{metric_prefix}_max": float(scalar_tensor.max().item()),
        }

    def _align_masked_latent_region_mask(
        self,
        *,
        region_mask: Optional[torch.Tensor],
        latent_hw: Tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if region_mask is None:
            return None
        aligned_region_mask = region_mask.to(device=device, dtype=dtype)
        if aligned_region_mask.dim() == 3:
            aligned_region_mask = aligned_region_mask.unsqueeze(1)
        if aligned_region_mask.shape[1] != 1:
            aligned_region_mask = aligned_region_mask[:, :1]
        if aligned_region_mask.shape[-2:] != latent_hw:
            aligned_region_mask = F.interpolate(
                aligned_region_mask,
                size=latent_hw,
                mode="bilinear",
                align_corners=False,
            )
        return torch.nan_to_num(
            aligned_region_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)

    def _compute_projection_conflict_gate(
        self,
        *,
        scheduler_step_delta_abs_mean: Optional[float],
    ) -> float:
        if scheduler_step_delta_abs_mean is None:
            return 1.0
        target_scheduler_delta = float(
            max(
                getattr(
                    self,
                    "support_projection_conflict_target_scheduler_delta",
                    0.030,
                ),
                1e-6,
            )
        )
        minimum_conflict_gate = float(
            min(
                max(
                    getattr(self, "support_projection_conflict_min_scale", 0.65),
                    0.0,
                ),
                1.0,
            )
        )
        conflict_gamma = float(
            max(
                getattr(self, "support_projection_conflict_gamma", 1.0),
                1e-6,
            )
        )
        current_scheduler_delta = max(float(scheduler_step_delta_abs_mean), 1e-6)
        if current_scheduler_delta <= target_scheduler_delta:
            return 1.0
        conflict_gate = target_scheduler_delta / current_scheduler_delta
        if abs(conflict_gamma - 1.0) > 1e-6:
            conflict_gate = conflict_gate ** conflict_gamma
        return float(min(max(conflict_gate, minimum_conflict_gate), 1.0))

    def _compose_masked_latent_dual_path(
        self,
        *,
        editable_region_latent: torch.Tensor,
        clean_support_latent: torch.Tensor,
        support_region_mask: Optional[torch.Tensor],
        editable_region_mask: Optional[torch.Tensor],
        support_region_noise: Optional[torch.Tensor],
        support_timestep: torch.Tensor,
    ) -> torch.Tensor:
        if support_region_mask is None or support_region_noise is None:
            return editable_region_latent

        latent_hw = editable_region_latent.shape[-2:]
        aligned_support_region_mask = self._align_masked_latent_region_mask(
            region_mask=support_region_mask,
            latent_hw=latent_hw,
            device=editable_region_latent.device,
            dtype=editable_region_latent.dtype,
        )
        if aligned_support_region_mask is None:
            return editable_region_latent
        aligned_editable_region_mask = self._align_masked_latent_region_mask(
            region_mask=editable_region_mask,
            latent_hw=latent_hw,
            device=editable_region_latent.device,
            dtype=editable_region_latent.dtype,
        )
        if aligned_editable_region_mask is None:
            aligned_editable_region_mask = torch.clamp(
                1.0 - aligned_support_region_mask,
                min=0.0,
                max=1.0,
            )
        else:
            aligned_editable_region_mask = aligned_editable_region_mask * (
                1.0 - aligned_support_region_mask
            )

        aligned_clean_support_latent = clean_support_latent.to(
            device=editable_region_latent.device,
            dtype=editable_region_latent.dtype,
        )
        if aligned_clean_support_latent.shape[-2:] != latent_hw:
            aligned_clean_support_latent = F.interpolate(
                aligned_clean_support_latent,
                size=latent_hw,
                mode="bilinear",
                align_corners=False,
            )
        aligned_support_region_noise = support_region_noise.to(
            device=editable_region_latent.device,
            dtype=editable_region_latent.dtype,
        )
        if aligned_support_region_noise.shape[-2:] != latent_hw:
            aligned_support_region_noise = F.interpolate(
                aligned_support_region_noise,
                size=latent_hw,
                mode="bilinear",
                align_corners=False,
            )
        support_timestep = support_timestep.to(device=editable_region_latent.device)
        if support_timestep.dim() == 0:
            support_timestep = support_timestep.unsqueeze(0)
        masked_support_latent = self.inference_scheduler.add_noise(
            aligned_clean_support_latent,
            aligned_support_region_noise,
            support_timestep,
        )
        remaining_editable_region_mask = torch.clamp(
            1.0 - aligned_support_region_mask - aligned_editable_region_mask,
            min=0.0,
            max=1.0,
        )
        effective_editable_region_mask = (
            aligned_editable_region_mask + remaining_editable_region_mask
        )
        return (
            aligned_support_region_mask * masked_support_latent
            + effective_editable_region_mask * editable_region_latent
        )
    
    def forward(
        self,
        sparse_images: torch.Tensor,     # [B, V, 3, H, W]
        sparse_poses: torch.Tensor,      # [B, V, 4, 4]
        target_pose: torch.Tensor,       # [B, 4, 4]
        timesteps: Optional[torch.Tensor] = None,
        noisy_latents: Optional[torch.Tensor] = None,
        depth_map: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        # 以下参数用于兼容GDDN接口
        rgb_render: Optional[torch.Tensor] = None,
        depth_render: Optional[torch.Tensor] = None,
        normal_render: Optional[torch.Tensor] = None,
        reference_depth_map: Optional[torch.Tensor] = None,
        reference_intrinsics: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        sparse_intrinsics: Optional[torch.Tensor] = None,
        support_confidence: Optional[torch.Tensor] = None,
        accum_render: Optional[torch.Tensor] = None,
        transmittance_render: Optional[torch.Tensor] = None,
        use_3dgs_residual_refiner: Optional[torch.Tensor] = None,
        # Phase 2A: 参考图像特征注入 (IP-Adapter style)
        ref_image_tokens: Optional[torch.Tensor] = None,  # [B, N_tokens, cross_attn_dim]
    ) -> GDDNControlNetOutput:
        """
        GDDN兼容的forward接口。
        
        Args:
            sparse_images: 稀疏视角图像 [B, V, 3, H, W]
            sparse_poses: 对应位姿 [B, V, 4, 4]
            target_pose: 目标位姿 [B, 4, 4]
            timesteps: 扩散时间步 [B]
            noisy_latents: 带噪声的latent [B, 4, h, w]
            depth_map: 可选深度图 [B, 1, H, W]
            K: 可选相机内参 [B, 3, 3]
            return_uncertainty: 是否返回不确定性
            rgb_render: GDDN兼容参数（未使用）
            depth_render: GDDN兼容参数（可用作depth_map）
            normal_render: GDDN兼容参数（未使用）
            accum_render: 真实3DGS累积透明度/coverage [B, 1, H, W]
            transmittance_render: 真实3DGS透射率 [B, 1, H, W]
            use_3dgs_residual_refiner: 是否启用以3DGS render为base的残差修复输出
            
        Returns:
            GDDNControlNetOutput: 噪声预测和不确定性
        """
        B, V, C, H, W = sparse_images.shape
        device = sparse_images.device
        
        # 转换输入到模型dtype
        sparse_images = sparse_images.to(dtype=self._model_dtype)
        sparse_poses = sparse_poses.to(dtype=self._model_dtype)
        target_pose = target_pose.to(dtype=self._model_dtype)
        if depth_map is not None:
            depth_map = depth_map.to(device=device, dtype=self._model_dtype)
        if K is not None:
            K = K.to(device=device, dtype=self._model_dtype)
        if target_intrinsics is not None:
            target_intrinsics = target_intrinsics.to(device=device, dtype=self._model_dtype)
            if target_intrinsics.dim() == 2:
                target_intrinsics = target_intrinsics.unsqueeze(0)
            if target_intrinsics.shape[0] == 1 and B > 1:
                target_intrinsics = target_intrinsics.expand(B, -1, -1)
        if reference_intrinsics is not None:
            reference_intrinsics = reference_intrinsics.to(device=device, dtype=self._model_dtype)
            if reference_intrinsics.dim() == 2:
                reference_intrinsics = reference_intrinsics.unsqueeze(0)
            if reference_intrinsics.shape[0] == 1 and B > 1:
                reference_intrinsics = reference_intrinsics.expand(B, -1, -1)
        sparse_intrinsics = self._normalize_sparse_intrinsics(
            sparse_intrinsics=sparse_intrinsics,
            batch_size=B,
            num_views=V,
            device=device,
            dtype=self._model_dtype,
        )
        if support_confidence is not None:
            support_confidence = support_confidence.to(device=device, dtype=self._model_dtype)
            if support_confidence.dim() == 3:
                support_confidence = support_confidence.unsqueeze(1)
        if accum_render is not None:
            accum_render = accum_render.to(device=device, dtype=self._model_dtype)
            if accum_render.dim() == 3:
                accum_render = accum_render.unsqueeze(1)
            if accum_render.shape[1] > 1:
                accum_render = accum_render[:, :1]
        if transmittance_render is not None:
            transmittance_render = transmittance_render.to(device=device, dtype=self._model_dtype)
            if transmittance_render.dim() == 3:
                transmittance_render = transmittance_render.unsqueeze(1)
            if transmittance_render.shape[1] > 1:
                transmittance_render = transmittance_render[:, :1]
        if accum_render is None and transmittance_render is not None:
            accum_render = torch.clamp(1.0 - transmittance_render, min=0.0, max=1.0)
        residual_refiner_enabled = False
        if isinstance(use_3dgs_residual_refiner, torch.Tensor):
            residual_refiner_enabled = bool((use_3dgs_residual_refiner > 0.5).any().item())
        elif isinstance(use_3dgs_residual_refiner, bool):
            residual_refiner_enabled = use_3dgs_residual_refiner

        # 1. 选择最近参考视角
        nearest_idx = self._select_nearest_view(sparse_poses, target_pose)
        batch_indices = torch.arange(B, device=device)
        ref_image = sparse_images[batch_indices, nearest_idx]  # [B, 3, H, W]
        ref_pose = sparse_poses[batch_indices, nearest_idx]    # [B, 4, 4]
        
        # 2. 使用depth_render作为depth_map（如果提供）
        if depth_map is None and depth_render is not None:
            depth_map = depth_render
        
        # 3. 准备ControlNet条件
        if rgb_render is not None and support_confidence is not None and depth_map is not None:
            controlnet_cond, warp_confidence, conditioning_details = self._build_control_condition_from_support_base(
                support_rgb=rgb_render.to(device=device, dtype=self._model_dtype),
                support_confidence=support_confidence,
                depth_map=depth_map,
            )
        else:
            controlnet_cond, warp_confidence, conditioning_details = self.condition_preprocessor(
                ref_image,
                ref_pose,
                target_pose,
                depth_map,
                K,
                reference_depth_map=reference_depth_map,
                reference_intrinsics=reference_intrinsics,
                target_intrinsics=target_intrinsics if target_intrinsics is not None else K,
                return_details=True,
            )
        controlnet_cond = self._resolve_primary_controlnet_condition(
            controlnet_condition=controlnet_cond,
            conditioning_details=conditioning_details,
        )
        
        # ========== 方案A: 多模态条件增强 ==========
        # Phase 2: Stage 1 训练后方案A有正确权重，eval模式也启用
        if self.multimodal_encoder is not None:
            if controlnet_cond.shape[1] >= 5:
                depth_1ch = controlnet_cond[:, 3:4]
                confidence_1ch = controlnet_cond[:, 4:5]
                multimodal_rgb = self.multimodal_encoder(depth_1ch, ref_image)
                controlnet_cond = torch.cat([multimodal_rgb, depth_1ch, confidence_1ch], dim=1)
            else:
                depth_1ch = controlnet_cond.mean(dim=1, keepdim=True)
                # 多模态编码: [depth + texture + edge]
                controlnet_cond = self.multimodal_encoder(depth_1ch, ref_image)
        
        # ========== Fix 3A v3: 深度条件对比度增强 ==========
        # 不做全范围0-1归一化(会OOD) — 改为围绕均值做5倍对比度增强
        # 这在保持训练时分布特征的同时增加空间深度变化
        if controlnet_cond.shape[1] >= 5:
            enhanced_condition_channels = controlnet_cond[:, :4]
            confidence_channel = controlnet_cond[:, 4:5].clamp(0.0, 1.0)
            cond_mean = enhanced_condition_channels.mean(dim=(2, 3), keepdim=True)
            enhanced_condition_channels = cond_mean + (enhanced_condition_channels - cond_mean) * 5.0
            controlnet_cond = torch.cat(
                [enhanced_condition_channels.clamp(0, 1), confidence_channel],
                dim=1,
            )
        else:
            cond_mean = controlnet_cond.mean(dim=(2, 3), keepdim=True)
            controlnet_cond = cond_mean + (controlnet_cond - cond_mean) * 5.0
            controlnet_cond = controlnet_cond.clamp(0, 1)
        
        # 4. 获取text embeddings（使用空文本）
        encoder_hidden_states = self._get_text_embeddings(B, device)
        
        # 5. 添加pose embedding到hidden states
        pose_emb = self.pose_encoder(ref_pose, target_pose)  # [B, 1, dim]
        
        # 将pose embedding与text embedding拼接
        # encoder_hidden_states: [B, 77, 1024] (SD 2.1) 或 [B, 77, 768] (SD 1.5)
        # 使用预初始化的_pose_proj进行维度投影
        if pose_emb.shape[-1] != encoder_hidden_states.shape[-1]:
            # 确保_pose_proj在正确的设备上，使用float32进行投影
            self._pose_proj = self._pose_proj.to(device=device)
            pose_emb_f32 = pose_emb.float()
            pose_emb = self._pose_proj(pose_emb_f32).to(dtype=self._model_dtype)
        
        # 数值稳定性：clamp极端值并处理NaN
        pose_emb = torch.clamp(pose_emb, min=-65504.0, max=65504.0)
        pose_emb = torch.nan_to_num(pose_emb, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 拼接pose信息
        encoder_hidden_states = torch.cat([encoder_hidden_states, pose_emb], dim=1)

        geometry_maps, routing_strength, low_confidence_known_mask = (
            self._build_geometry_routing_inputs(
                conditioning_details=conditioning_details,
                support_projection_mask=None,
                target_hw=(H, W),
                device=device,
            )
        )
        plucker_intrinsics = target_intrinsics if target_intrinsics is not None else K
        encoder_hidden_states = self._append_geometry_condition_tokens(
            encoder_hidden_states=encoder_hidden_states,
            target_pose=target_pose,
            target_intrinsics=plucker_intrinsics,
            height=H,
            width=W,
            geometry_maps=geometry_maps,
            routing_strength=routing_strength,
            low_confidence_known_mask=low_confidence_known_mask,
        )

        # ========== Phase 2A: 参考图像latent注入 ==========
        # 将ref_image_tokens concat到encoder_hidden_states
        # 让UNet cross-attention在每步denoising时能"看到"参考图信息
        if ref_image_tokens is not None:
            ref_tokens = ref_image_tokens.to(dtype=self._model_dtype, device=device)
            # 数值稳定性
            ref_tokens = torch.clamp(ref_tokens, min=-65504.0, max=65504.0)
            ref_tokens = torch.nan_to_num(ref_tokens, nan=0.0, posinf=0.0, neginf=0.0)
            encoder_hidden_states = torch.cat([encoder_hidden_states, ref_tokens], dim=1)

        # ========== 方案B: 稀疏视图纹理特征注入 ==========
        # Phase 2: Stage 1 训练后方案B有正确权重，eval模式也启用
        if self.sparse_view_adapter is not None:
            latent_h, latent_w = H // 8, W // 8
            texture_emb = None
            texture_source_name = "target_aligned_support"
            if getattr(self, "enable_multiview_correspondence_tokens", False):
                texture_emb = self._encode_multiview_correspondence_tokens(
                    sparse_images=sparse_images,
                    sparse_poses=sparse_poses,
                    target_pose=target_pose,
                    support_rgb=rgb_render,
                    support_confidence=support_confidence,
                    latent_hw=(latent_h, latent_w),
                    device=device,
                )
                texture_source_name = "multiview_target_aligned_correspondence"
            if texture_emb is None:
                correspondence_texture_source = self._build_correspondence_texture_source(
                    ref_image=ref_image,
                    support_rgb=rgb_render,
                    support_confidence=support_confidence,
                )
                texture_emb = self._encode_texture_condition_tokens(
                    texture_source=correspondence_texture_source,
                    latent_hw=(latent_h, latent_w),
                    device=device,
                )
            if texture_emb is not None:
                encoder_hidden_states = torch.cat([encoder_hidden_states, texture_emb], dim=1)
                if not hasattr(self, '_scheme_b_diag_printed'):
                    self._scheme_b_diag_printed = True
                    print(
                        f"[方案B] 稀疏视图纹理特征已注入, "
                        f"texture_emb shape: {texture_emb.shape}, "
                        f"source={texture_source_name}"
                    )

        
        # 6. 如果没有提供noisy_latents，从参考图像生成
        if noisy_latents is None:
            ref_latent = self._encode_vae(ref_image)
            noise = torch.randn_like(ref_latent)
            if timesteps is None:
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                )
            noisy_latents = self.noise_scheduler.add_noise(ref_latent, noise, timesteps)
        
        noisy_latents = noisy_latents.to(dtype=self._model_dtype)
        
        # 7. 调整ControlNet条件尺寸
        # ControlNet期望条件图像与输入图像尺寸相同（在latent空间之前）
        latent_h, latent_w = noisy_latents.shape[-2:]
        target_h, target_w = latent_h * 8, latent_w * 8
        
        if controlnet_cond.shape[-2:] != (target_h, target_w):
            controlnet_cond = F.interpolate(
                controlnet_cond,
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=True
            )
        
        # 8. ControlNet前向 (多GPU时需要移动数据)
        # ControlNet需要768维，SD 2.1是1024维，使用投影层
        if self.encoder_proj is not None:
            # 使用float32进行投影以提高数值稳定性
            encoder_hidden_states_f32 = encoder_hidden_states.float()
            cn_encoder_states = self.encoder_proj(encoder_hidden_states_f32)
        else:
            cn_encoder_states = encoder_hidden_states
        
        # 数值稳定性处理：clamp极端值防止float16溢出
        cn_encoder_states = torch.clamp(cn_encoder_states, min=-65504.0, max=65504.0)
        cn_encoder_states = torch.nan_to_num(cn_encoder_states, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 获取ControlNet所在的设备和dtype
        controlnet_device = next(self.controlnet.parameters()).device
        controlnet_dtype = next(self.controlnet.parameters()).dtype
        
        # 强制移动所有输入到ControlNet设备并保持dtype一致
        cn_noisy_latents = noisy_latents.to(device=controlnet_device, dtype=controlnet_dtype)
        cn_timesteps = timesteps.to(device=controlnet_device)  # timesteps保持long类型
        cn_encoder_states = cn_encoder_states.to(device=controlnet_device, dtype=controlnet_dtype)
        cn_cond = controlnet_cond.to(device=controlnet_device, dtype=controlnet_dtype)
        
        # 数值安全：将ControlNet输入中的非有限值替换为零。
        if torch.isnan(cn_noisy_latents).any():
            cn_noisy_latents = torch.nan_to_num(cn_noisy_latents)
        if torch.isnan(cn_encoder_states).any():
            cn_encoder_states = torch.nan_to_num(cn_encoder_states)
        if torch.isnan(cn_cond).any():
            cn_cond = torch.nan_to_num(cn_cond)

        
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            cn_noisy_latents,
            cn_timesteps,
            encoder_hidden_states=cn_encoder_states,
            controlnet_cond=cn_cond,
            return_dict=False,
        )
        
        # 检查ControlNet输出是否有NaN
        if torch.isnan(mid_block_res_sample).any():
            print("[WARNING] NaN detected in ControlNet mid_block output, replacing with zeros")
            mid_block_res_sample = torch.nan_to_num(mid_block_res_sample)
            down_block_res_samples = [torch.nan_to_num(d) for d in down_block_res_samples]
        
        # 9. UNet前向（带ControlNet残差，多GPU时需要移动数据）
        # 获取UNet所在的设备和dtype
        unet_device = next(self.unet.parameters()).device
        unet_dtype = next(self.unet.parameters()).dtype
        
        # 强制移动所有输入到UNet设备并保持dtype一致
        unet_noisy_latents = noisy_latents.to(device=unet_device, dtype=unet_dtype)
        unet_timesteps = timesteps.to(device=unet_device)  # timesteps保持long类型
        unet_encoder_states = encoder_hidden_states.to(device=unet_device, dtype=unet_dtype)
        
        # 移动ControlNet残差到UNet设备
        down_residuals = [d.to(device=unet_device, dtype=unet_dtype) for d in down_block_res_samples]
        mid_residual = mid_block_res_sample.to(device=unet_device, dtype=unet_dtype)
        
        unet_output = self.unet(
            unet_noisy_latents,
            unet_timesteps,
            encoder_hidden_states=unet_encoder_states,
            down_block_additional_residuals=down_residuals,
            mid_block_additional_residual=mid_residual,
            return_dict=True,
        )
        noise_pred = unet_output.sample
        
        # 检查UNet输出是否有NaN
        if torch.isnan(noise_pred).any():
            print("[WARNING] NaN detected in UNet noise_pred, replacing with zeros")
            noise_pred = torch.nan_to_num(noise_pred)
        
        # 10. 不确定性预测
        effect_risk_exist = None
        effect_gain_novel = None
        predicted_image = None
        predicted_residual_refined_image = None
        predicted_proposal_image = None
        predicted_proposal_residual = None
        predicted_proposal_applied_residual = None
        predicted_proposal_confidence = None
        predicted_proposal_warp_error = None
        predicted_proposal_repairability = None
        predicted_proposal_verification = None
        predicted_proposal_acceptance = None
        predicted_proposal_warp_error_logit = None
        predicted_proposal_repairability_logit = None
        predicted_proposal_verification_logit = None
        predicted_proposal_acceptance_logit = None
        
        if return_uncertainty:
            uncertainty_latent = noisy_latents
            if timesteps is not None:
                uncertainty_latent = self._predict_x0_latent(
                    noisy_latents=noisy_latents,
                    noise_pred=noise_pred,
                    timesteps=timesteps,
                )
            predicted_pure_x0_image = self.decode_latent_to_rgb(uncertainty_latent)
            if predicted_pure_x0_image.shape[-2:] != (H, W):
                predicted_pure_x0_image = F.interpolate(
                    predicted_pure_x0_image,
                    size=(H, W),
                    mode="bilinear",
                    align_corners=False,
                )
            predicted_image = predicted_pure_x0_image
            support_rgb = None
            if conditioning_details is not None:
                support_rgb = conditioning_details.get("warped_image")
            if support_rgb is None:
                support_rgb = rgb_render
            refined_support_outputs = self._build_refined_support_signals_from_confidence(
                warp_confidence=warp_confidence,
                target_hw=(H, W),
            )
            support_projection_mask = refined_support_outputs.get("support_projection_mask")
            refined_inpaint_mask = refined_support_outputs.get("inpaint_mask")
            refined_exist_mask = None
            if refined_inpaint_mask is not None:
                refined_exist_mask = torch.clamp(1.0 - refined_inpaint_mask, min=0.0, max=1.0)
            if refined_exist_mask is None or refined_inpaint_mask is None:
                refined_exist_mask, refined_inpaint_mask = self._build_exist_novel_masks_from_signals(
                    depth_map=depth_map,
                    warp_confidence=warp_confidence,
                    target_hw=(H, W),
                )
            support_composed_image = self._compose_support_preserving_image(
                teacher_rgb=predicted_pure_x0_image,
                support_rgb=support_rgb,
                exist_mask=refined_exist_mask,
                novel_mask=refined_inpaint_mask,
                support_blend_mask=support_projection_mask,
            )
            proposal_outputs = self._build_explicit_frontier_proposal(
                teacher_rgb=predicted_pure_x0_image,
                base_rgb=support_rgb if support_rgb is not None else rgb_render,
                support_confidence=warp_confidence,
                support_projection_mask=support_projection_mask,
                inpaint_mask=refined_inpaint_mask,
                frontier_mask=None,
                verification_prior=None,
                accum_render=accum_render,
            )
            predicted_proposal_image = proposal_outputs.get("proposal_raw_rgb")
            predicted_proposal_residual = proposal_outputs.get("proposal_residual_rgb")
            predicted_proposal_applied_residual = proposal_outputs.get(
                "proposal_applied_residual_rgb"
            )
            predicted_proposal_confidence = proposal_outputs.get("proposal_confidence")
            predicted_proposal_warp_error = proposal_outputs.get(
                "proposal_warp_error_confidence"
            )
            predicted_proposal_warp_error_logit = proposal_outputs.get(
                "proposal_warp_error_logit"
            )
            predicted_proposal_repairability = proposal_outputs.get(
                "proposal_repairability_confidence"
            )
            predicted_proposal_repairability_logit = proposal_outputs.get(
                "proposal_repairability_logit"
            )
            predicted_proposal_verification = proposal_outputs.get(
                "proposal_verification_confidence"
            )
            predicted_proposal_verification_logit = proposal_outputs.get(
                "proposal_verification_logit"
            )
            predicted_proposal_acceptance = proposal_outputs.get(
                "proposal_acceptance_confidence"
            )
            predicted_proposal_acceptance_logit = proposal_outputs.get(
                "proposal_acceptance_logit"
            )
            if predicted_proposal_image is not None:
                predicted_image = predicted_proposal_image
                support_composed_image = self._compose_support_preserving_image(
                    teacher_rgb=predicted_proposal_image,
                    support_rgb=support_rgb,
                    exist_mask=refined_exist_mask,
                    novel_mask=refined_inpaint_mask,
                    support_blend_mask=support_projection_mask,
                )
            if residual_refiner_enabled and accum_render is not None and rgb_render is not None:
                residual_refiner_confidence = accum_render
                if residual_refiner_confidence.shape[-2:] != (H, W):
                    residual_refiner_confidence = F.interpolate(
                        residual_refiner_confidence,
                        size=(H, W),
                        mode="bilinear",
                        align_corners=False,
                    )
                residual_refiner_confidence = torch.nan_to_num(
                    residual_refiner_confidence,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                residual_refiner_exist_mask = (residual_refiner_confidence > 1e-6).float()
                residual_refiner_novel_mask = torch.clamp(
                    1.0 - residual_refiner_exist_mask,
                    min=0.0,
                    max=1.0,
                )
                predicted_residual_refined_image = self._compose_support_preserving_image(
                    teacher_rgb=(
                        predicted_proposal_image
                        if predicted_proposal_image is not None
                        else predicted_pure_x0_image
                    ),
                    support_rgb=rgb_render,
                    exist_mask=residual_refiner_exist_mask,
                    novel_mask=residual_refiner_novel_mask,
                    support_blend_mask=residual_refiner_confidence,
                )
                predicted_image = predicted_residual_refined_image
            effect_exist_mask = refined_exist_mask
            effect_novel_mask = refined_inpaint_mask
            effective_support_confidence = support_projection_mask
            if residual_refiner_enabled and accum_render is not None:
                effective_support_confidence = residual_refiner_confidence
                effect_exist_mask = residual_refiner_exist_mask
                effect_novel_mask = residual_refiner_novel_mask
            if effective_support_confidence is None:
                effective_support_confidence = warp_confidence
            effect_risk_exist, effect_gain_novel = self._compute_effect_maps(
                teacher_rgb=predicted_image,
                render_rgb=rgb_render,
                warp_rgb=conditioning_details.get("warped_image") if conditioning_details is not None else None,
                warp_confidence=effective_support_confidence,
                depth_map=depth_map,
                exist_mask=effect_exist_mask,
                novel_mask=effect_novel_mask,
            )
        else:
            predicted_pure_x0_image = None
            support_composed_image = None
            support_projection_mask = None
            refined_inpaint_mask = None

        failure_prior_outputs: Dict[str, Any] = {}
        if (
            bool(getattr(self, "enable_failure_prior_head", False))
            and getattr(self, "failure_prior_head", None) is not None
            and predicted_image is not None
        ):
            conditioning_details_for_failure = (
                conditioning_details if isinstance(conditioning_details, dict) else {}
            )
            failure_prior_outputs = self.failure_prior_head(
                teacher_rgb=predicted_image,
                render_rgb=rgb_render,
                warp_rgb=conditioning_details_for_failure.get("warped_image"),
                warp_confidence=(
                    support_projection_mask
                    if support_projection_mask is not None
                    else warp_confidence
                ),
                canonical_support=support_projection_mask,
                canonical_novel=refined_inpaint_mask,
                relative_depth_consistency_error=conditioning_details_for_failure.get(
                    "relative_depth_consistency_error"
                ),
                forward_backward_reprojection_error=conditioning_details_for_failure.get(
                    "forward_backward_reprojection_error"
                ),
                fusion_rgb_disagreement=conditioning_details_for_failure.get(
                    "fusion_rgb_disagreement"
                ),
                support_confidence_gap=conditioning_details_for_failure.get(
                    "support_confidence_gap"
                ),
                teacher_signal_source="diffusion_teacher_predicted_image",
                teacher_signal_is_calibrated=bool(
                    getattr(self, "failure_prior_teacher_signal_is_calibrated", False)
                ),
            )
        
        return GDDNControlNetOutput(
            noise_pred=noise_pred,
            uncertainty_epistemic=effect_risk_exist,
            uncertainty_aleatoric=effect_gain_novel,
            predicted_image=predicted_image,
            predicted_pure_x0_image=predicted_pure_x0_image,
            predicted_residual_refined_image=predicted_residual_refined_image,
            predicted_proposal_image=predicted_proposal_image,
            predicted_proposal_residual=predicted_proposal_residual,
            predicted_proposal_applied_residual=predicted_proposal_applied_residual,
            predicted_proposal_confidence=predicted_proposal_confidence,
            predicted_proposal_warp_error=predicted_proposal_warp_error,
            predicted_proposal_repairability=predicted_proposal_repairability,
            predicted_proposal_verification=predicted_proposal_verification,
            predicted_proposal_acceptance=predicted_proposal_acceptance,
            predicted_proposal_warp_error_logit=predicted_proposal_warp_error_logit,
            predicted_proposal_repairability_logit=predicted_proposal_repairability_logit,
            predicted_proposal_verification_logit=predicted_proposal_verification_logit,
            predicted_proposal_acceptance_logit=predicted_proposal_acceptance_logit,
            predicted_support_composed_image=support_composed_image,
            predicted_support_projection_mask=support_projection_mask,
            predicted_inpaint_mask=refined_inpaint_mask,
            effect_risk_exist=effect_risk_exist,
            effect_gain_novel=effect_gain_novel,
            failure_wrong=failure_prior_outputs.get("failure_wrong"),
            failure_geo_wrong=failure_prior_outputs.get("failure_geo_wrong"),
            failure_repair=failure_prior_outputs.get("failure_repair"),
            failure_rgb_trust_proxy=failure_prior_outputs.get("failure_rgb_trust_proxy"),
            failure_depth_trust_proxy=failure_prior_outputs.get("failure_depth_trust_proxy"),
            failure_repair_is_calibrated=bool(
                failure_prior_outputs.get("failure_repair_is_calibrated", False)
            ),
            failure_output_validity=failure_prior_outputs.get("failure_output_validity"),
            failure_teacher_signal_source=failure_prior_outputs.get("failure_teacher_signal_source"),
            failure_teacher_signal_is_calibrated=bool(
                failure_prior_outputs.get("failure_teacher_signal_is_calibrated", False)
            ),
            failure_online_promotion_allowed=bool(
                failure_prior_outputs.get("failure_online_promotion_allowed", False)
            ),
        )
    
    @torch.no_grad()
    def generate_view(
        self,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        target_pose: torch.Tensor,
        num_steps: int = 25,
        guidance_scale: Optional[float] = None,
        depth_map: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        reference_depth_map: Optional[torch.Tensor] = None,
        reference_intrinsics: Optional[torch.Tensor] = None,
        sparse_intrinsics: Optional[torch.Tensor] = None,
        mc_noise_scale: float = 0.0,  # MC采样噪声强度，>0时在推理中添加随机扰动
        truncation_k: int = 0,  # Layer 2: 截断步数（0=不截断，>0=前k步Dropout ON后关闭）
        fixed_latent: Optional[torch.Tensor] = None,  # Layer 2: 固定初始噪声
        rendered_rgb: Optional[torch.Tensor] = None,  # img2img: 3DGS渲染的RGB图像
        img2img_strength: float = 0.5,  # img2img: 自适应强度（0.3=轻度修复, 0.8=重度修复）
        warped_rgb: Optional[torch.Tensor] = None,    # [Warp-Inpaint] 真实图像warp结果 [B,3,H,W]
        support_confidence: Optional[torch.Tensor] = None,  # [Warp-Inpaint] 联合support置信度 [B,1,H,W]
        accum_render: Optional[torch.Tensor] = None,  # [Residual-Refine] 3DGS累积透明度/coverage [B,1,H,W]
        transmittance_render: Optional[torch.Tensor] = None,  # [Residual-Refine] 3DGS透射率 [B,1,H,W]
        use_3dgs_residual_refiner: Union[bool, torch.Tensor] = False,
        support_projection_mask: Optional[torch.Tensor] = None,  # [Warp-Inpaint] 高置信support软投影mask [B,1,H,W]
        inpaint_mask: Optional[torch.Tensor] = None,  # [Warp-Inpaint] 空洞mask [B,1,H,W]，1=空洞
        frontier_mask: Optional[torch.Tensor] = None,
        verification_prior: Optional[torch.Tensor] = None,
        enable_gcd_guidance: bool = True,
        enable_cags_guidance: bool = True,
        enable_latent_blending: bool = True,
        support_projection_is_reliable: bool = True,
        diagnostic_inpaint_img2img_strength_override: Optional[float] = None,
        diagnostic_masked_dual_path_support_step_offset_override: Optional[int] = None,
        return_diagnostics: bool = False,
        track_grad_decode: bool = False,
        output_mode: str = "composed",
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        生成目标视角的图像。
        
        Args:
            sparse_images: [B, V, 3, H, W]
            sparse_poses: [B, V, 4, 4]
            target_pose: [B, 4, 4]
            num_steps: 去噪步数
            guidance_scale: CFG引导强度；None 时使用实例默认配置
            depth_map: 可选深度图
            K: 可选相机内参
            reference_depth_map: 可选参考视角深度图
            reference_intrinsics: 可选参考视角内参
            
        Returns:
            image: [B, 3, H, W] 生成的图像
            epistemic: [B, 1, H, W] 认知不确定性
            aleatoric: [B, 1, H, W] 随机不确定性
        """
        # 维度适配: 调用方可能传入4D sparse_images [K,3,H,W]，需要扩展为5D [B,V,3,H,W]
        if sparse_images.dim() == 4:
            _batch_size = target_pose.shape[0]
            sparse_images = sparse_images.unsqueeze(0).expand(
                _batch_size, *sparse_images.shape
            )
        if sparse_poses.dim() == 2:
            _batch_size = target_pose.shape[0]
            sparse_poses = sparse_poses.unsqueeze(0).expand(
                _batch_size, *sparse_poses.shape
            )
        elif sparse_poses.dim() == 3 and sparse_poses.shape[0] != target_pose.shape[0]:
            _batch_size = target_pose.shape[0]
            sparse_poses = sparse_poses.unsqueeze(0).expand(
                _batch_size, *sparse_poses.shape
            )

        B, V, C, H, W = sparse_images.shape
        device = sparse_images.device
        guidance_scale = float(
            self.default_guidance_scale if guidance_scale is None else guidance_scale
        )
        
        # 转换输入到模型dtype
        sparse_images = sparse_images.to(dtype=self._model_dtype)
        sparse_poses = sparse_poses.to(dtype=self._model_dtype)
        target_pose = target_pose.to(dtype=self._model_dtype)
        if depth_map is not None:
            depth_map = depth_map.to(device=device, dtype=self._model_dtype)
            # depth_map可能是3通道（DA-V2格式[B,3,H,W]），需要转为单通道[B,1,H,W]
            if depth_map.dim() == 4 and depth_map.shape[1] == 3:
                depth_map = depth_map[:, :1, :, :]
        if K is not None:
            K = K.to(device=device, dtype=self._model_dtype)
        sparse_intrinsics = self._normalize_sparse_intrinsics(
            sparse_intrinsics=sparse_intrinsics,
            batch_size=B,
            num_views=V,
            device=device,
            dtype=self._model_dtype,
        )
        if reference_depth_map is not None:
            reference_depth_map = reference_depth_map.to(device=device, dtype=self._model_dtype)
            if reference_depth_map.dim() == 4 and reference_depth_map.shape[1] == 3:
                reference_depth_map = reference_depth_map[:, :1, :, :]
            if reference_depth_map.shape[0] == 1 and B > 1:
                reference_depth_map = reference_depth_map.expand(B, -1, -1, -1)
        if reference_intrinsics is not None:
            reference_intrinsics = reference_intrinsics.to(device=device, dtype=self._model_dtype)
            if reference_intrinsics.dim() == 2:
                reference_intrinsics = reference_intrinsics.unsqueeze(0)
            if reference_intrinsics.shape[0] == 1 and B > 1:
                reference_intrinsics = reference_intrinsics.expand(B, -1, -1)
        if support_confidence is not None:
            support_confidence = support_confidence.to(device=device, dtype=self._model_dtype)
            if support_confidence.dim() == 3:
                support_confidence = support_confidence.unsqueeze(1)
        if accum_render is not None:
            accum_render = accum_render.to(device=device, dtype=self._model_dtype)
            if accum_render.dim() == 3:
                accum_render = accum_render.unsqueeze(1)
            if accum_render.shape[1] > 1:
                accum_render = accum_render[:, :1]
        if transmittance_render is not None:
            transmittance_render = transmittance_render.to(device=device, dtype=self._model_dtype)
            if transmittance_render.dim() == 3:
                transmittance_render = transmittance_render.unsqueeze(1)
            if transmittance_render.shape[1] > 1:
                transmittance_render = transmittance_render[:, :1]
        if accum_render is None and transmittance_render is not None:
            accum_render = torch.clamp(1.0 - transmittance_render, min=0.0, max=1.0)
        residual_refiner_enabled = False
        if isinstance(use_3dgs_residual_refiner, torch.Tensor):
            residual_refiner_enabled = bool((use_3dgs_residual_refiner > 0.5).any().item())
        elif isinstance(use_3dgs_residual_refiner, bool):
            residual_refiner_enabled = use_3dgs_residual_refiner
        if support_projection_mask is not None:
            support_projection_mask = support_projection_mask.to(device=device, dtype=self._model_dtype)
            if support_projection_mask.dim() == 3:
                support_projection_mask = support_projection_mask.unsqueeze(1)
        
        # 1. 选择最近参考视角
        nearest_idx = self._select_nearest_view(sparse_poses, target_pose)
        batch_indices = torch.arange(B, device=device)
        ref_image = sparse_images[batch_indices, nearest_idx]
        ref_pose = sparse_poses[batch_indices, nearest_idx]
        
        # 2. 准备条件
        if warped_rgb is not None and support_confidence is not None and depth_map is not None:
            controlnet_cond, warp_confidence, conditioning_details = self._build_control_condition_from_support_base(
                support_rgb=warped_rgb.to(device=device, dtype=self._model_dtype),
                support_confidence=support_confidence,
                depth_map=depth_map,
            )
        else:
            controlnet_cond, warp_confidence, conditioning_details = self.condition_preprocessor(
                ref_image,
                ref_pose,
                target_pose,
                depth_map,
                K,
                reference_depth_map=reference_depth_map,
                reference_intrinsics=reference_intrinsics,
                target_intrinsics=K,
                return_details=True,
            )
        controlnet_cond = self._resolve_primary_controlnet_condition(
            controlnet_condition=controlnet_cond,
            conditioning_details=conditioning_details,
        )
        
        # 3. 编码参考图像获取latent尺寸
        # OOM修复：将ref_image resize到VAE安全分辨率（SD2.1训练在512x512）
        # 避免原始高分辨率图像导致VAE编码OOM
        _vae_max_res = 512
        _ref_h, _ref_w = ref_image.shape[-2:]
        if _ref_h > _vae_max_res or _ref_w > _vae_max_res:
            _scale = _vae_max_res / max(_ref_h, _ref_w)
            _new_h = int(_ref_h * _scale) // 8 * 8  # 确保是8的倍数（VAE下采样8x）
            _new_w = int(_ref_w * _scale) // 8 * 8
            ref_image_vae = F.interpolate(ref_image, size=(_new_h, _new_w), mode='bilinear', align_corners=False)
        else:
            # 确保是8的倍数
            _new_h = _ref_h // 8 * 8
            _new_w = _ref_w // 8 * 8
            if _new_h != _ref_h or _new_w != _ref_w:
                ref_image_vae = F.interpolate(ref_image, size=(_new_h, _new_w), mode='bilinear', align_corners=False)
            else:
                ref_image_vae = ref_image
        ref_latent = self._encode_vae(ref_image_vae)
        latent_h, latent_w = ref_latent.shape[-2:]
        
        # 4. 获取text embeddings
        encoder_hidden_states = self._get_text_embeddings(B, device)

        # 训练forward路径显式注入了PoseEncoder；诊断时允许仅在推理路径关闭，
        # 用于隔离pose条件对单图生成与候选排序的影响。
        if getattr(self, "infer_use_pose_condition", True):
            pose_emb = self.pose_encoder(ref_pose, target_pose)  # [B, 1, dim]
            if pose_emb.shape[-1] != encoder_hidden_states.shape[-1]:
                self._pose_proj = self._pose_proj.to(device=device)
                pose_emb_f32 = pose_emb.float()
                pose_emb = self._pose_proj(pose_emb_f32).to(dtype=self._model_dtype)

        pose_emb = torch.clamp(pose_emb, min=-65504.0, max=65504.0)
        pose_emb = torch.nan_to_num(pose_emb, nan=0.0, posinf=0.0, neginf=0.0)
        encoder_hidden_states = torch.cat([encoder_hidden_states, pose_emb], dim=1)

        geometry_maps, routing_strength, low_confidence_known_mask = (
            self._build_geometry_routing_inputs(
                conditioning_details=conditioning_details,
                support_projection_mask=support_projection_mask,
                target_hw=(H, W),
                device=device,
            )
        )
        encoder_hidden_states = self._append_geometry_condition_tokens(
            encoder_hidden_states=encoder_hidden_states,
            target_pose=target_pose,
            target_intrinsics=K.to(device=device, dtype=self._model_dtype) if K is not None else None,
            height=H,
            width=W,
            geometry_maps=geometry_maps,
            routing_strength=routing_strength,
            low_confidence_known_mask=low_confidence_known_mask,
        )
        if self.sparse_view_adapter is not None:
            texture_emb = None
            if getattr(self, "enable_multiview_correspondence_tokens", False):
                texture_emb = self._encode_multiview_correspondence_tokens(
                    sparse_images=sparse_images,
                    sparse_poses=sparse_poses,
                    target_pose=target_pose,
                    support_rgb=warped_rgb,
                    support_confidence=support_confidence,
                    latent_hw=(latent_h, latent_w),
                    device=device,
                )
            if texture_emb is None:
                correspondence_texture_source = self._build_correspondence_texture_source(
                    ref_image=ref_image,
                    support_rgb=warped_rgb,
                    support_confidence=support_confidence,
                )
                texture_emb = self._encode_texture_condition_tokens(
                    texture_source=correspondence_texture_source,
                    latent_hw=(latent_h, latent_w),
                    device=device,
                )
            if texture_emb is not None:
                encoder_hidden_states = torch.cat([encoder_hidden_states, texture_emb], dim=1)

        # 5. 调整条件尺寸
        target_h, target_w = latent_h * 8, latent_w * 8
        if controlnet_cond.shape[-2:] != (target_h, target_w):
            controlnet_cond = F.interpolate(
                controlnet_cond, size=(target_h, target_w),
                mode='bilinear', align_corners=True
            )
        
        # 6. 初始化噪声 — 支持 Warp-Inpaint 模式、img2img 模式和固定噪声
        # ===== [Warp-Inpaint] 优先级最高 =====
        _inpaint_mode = warped_rgb is not None and inpaint_mask is not None
        _inpaint_latent = None   # warp图的latent（有效区域保真引用）
        _mask_latent = None      # 空洞mask在latent空间
        diagnostic_payload: Optional[Dict[str, Any]] = {} if return_diagnostics else None
        predecode_debug_tensors: Optional[Dict[str, torch.Tensor]] = {} if return_diagnostics else None
        predecode_selected_step_tags: Dict[int, str] = {}
        controlnet_down_residual_abs_mean_values: List[float] = []
        controlnet_mid_residual_abs_mean_values: List[float] = []
        cfg_noise_delta_abs_mean_values: List[float] = []
        cfg_noise_delta_variance_values: List[float] = []
        scheduler_step_delta_abs_mean_values: List[float] = []
        gcd_delta_abs_mean_values: List[float] = []
        gcd_delta_novel_abs_mean_values: List[float] = []
        cags_delta_abs_mean_values: List[float] = []
        cags_delta_exist_abs_mean_values: List[float] = []
        dual_path_temporal_weight_values: List[float] = []
        projection_temporal_weight_values: List[float] = []
        projection_conflict_gate_values: List[float] = []
        projection_delta_abs_mean_values: List[float] = []
        projection_delta_support_abs_mean_values: List[float] = []
        inpaint_hole_ratio: Optional[float] = None
        applied_gcd_steps = 0
        applied_cags_steps = 0
        applied_latent_blend_steps = 0
        _support_projection_clean_latent = None
        _support_projection_mask = None
        _support_projection_noise = None
        _masked_inpaint_support_mask = None
        _masked_inpaint_editable_mask = None
        _masked_inpaint_support_noise = None
        _masked_inpaint_support_uses_projection = False
        _stepwise_masked_dual_path_enabled = False
        _masked_dual_path_support_step_offset = 0
        _masked_dual_path_support_step_index = 0
        _scheduled_support_step_offset = None
        _scheduled_support_step_progress = None
        _editable_difficulty_signal = None
        _support_preserve_budget_signal = None
        _support_projection_ratio_for_schedule = None
        _support_confidence_ratio_for_schedule = None
        _latent_support_mask_coverage = None
        _supportswitch_projection_pre_step_weight = float(
            min(
                max(
                    getattr(self, "supportswitch_projection_pre_step_weight", 0.55),
                    0.0,
                ),
                1.0,
            )
        )
        _supportswitch_dual_path_pre_step_weight = float(
            min(
                max(
                    getattr(self, "supportswitch_dual_path_pre_step_weight", 0.75),
                    0.0,
                ),
                1.0,
            )
        )
        _support_projection_conflict_target_scheduler_delta = float(
            max(
                getattr(
                    self,
                    "support_projection_conflict_target_scheduler_delta",
                    0.030,
                ),
                1e-6,
            )
        )
        _support_projection_conflict_min_scale = float(
            min(
                max(
                    getattr(self, "support_projection_conflict_min_scale", 0.65),
                    0.0,
                ),
                1.0,
            )
        )
        _support_projection_conflict_gamma = float(
            max(
                getattr(self, "support_projection_conflict_gamma", 1.0),
                1e-6,
            )
        )
        _masked_support_clean_latent_abs_mean = None
        _masked_support_clean_latent_variance = None
        _masked_support_noised_latent_abs_mean = None
        _masked_support_noised_latent_variance = None

        def _update_predecode_latent_statistics(
            metric_prefix: str,
            *,
            latent_tensor: Optional[torch.Tensor],
        ) -> None:
            if diagnostic_payload is None:
                return
            diagnostic_payload.update(
                self._collect_latent_region_statistics(
                    metric_prefix=metric_prefix,
                    latent_tensor=latent_tensor.detach() if latent_tensor is not None else None,
                    support_mask=_masked_inpaint_support_mask,
                    editable_mask=_masked_inpaint_editable_mask,
                    novel_mask=_mask_latent,
                )
            )

        def _store_predecode_preview(
            tensor_name: str,
            *,
            latent_tensor: Optional[torch.Tensor],
        ) -> None:
            if predecode_debug_tensors is None or latent_tensor is None:
                return
            predecode_preview_rgb = self._decode_vae(
                latent_tensor.detach(),
                track_grad=False,
            )
            predecode_debug_tensors[tensor_name] = predecode_preview_rgb.detach().cpu()

        if _inpaint_mode:
            # 编码 warp 图为 latent
            _warp = warped_rgb.to(dtype=self._model_dtype)
            if _warp.dim() == 3:
                _warp = _warp.unsqueeze(0)
            if _warp.shape[-2:] != (_new_h, _new_w):
                _warp = F.interpolate(_warp, size=(_new_h, _new_w), mode='bilinear', align_corners=False)
            _inpaint_latent = self._encode_vae(_warp)
            # 空洞mask降采样到latent空间（8倍下采样）
            _mask_latent = F.interpolate(
                inpaint_mask.float().to(device=_inpaint_latent.device),
                size=(_inpaint_latent.shape[-2], _inpaint_latent.shape[-1]),
                mode='nearest'
            )
            true_novel_latent_mask = torch.nan_to_num(
                _mask_latent,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            binary_support_latent_mask = torch.clamp(
                1.0 - true_novel_latent_mask,
                min=0.0,
                max=1.0,
            )
            _masked_inpaint_support_mask = binary_support_latent_mask
            if support_projection_is_reliable and support_projection_mask is not None:
                latent_support_projection_mask = F.interpolate(
                    support_projection_mask.float().to(device=_inpaint_latent.device),
                    size=(_inpaint_latent.shape[-2], _inpaint_latent.shape[-1]),
                    mode='bilinear',
                    align_corners=False,
                )
                latent_support_projection_mask = torch.nan_to_num(
                    latent_support_projection_mask,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                latent_support_projection_mask = torch.minimum(
                    latent_support_projection_mask,
                    binary_support_latent_mask,
                )
                if float(latent_support_projection_mask.mean().item()) > 1e-6:
                    _masked_inpaint_support_mask = latent_support_projection_mask
                    _masked_inpaint_support_uses_projection = True
            if _masked_inpaint_support_mask is not None:
                _latent_support_mask_coverage = float(
                    _masked_inpaint_support_mask.mean().item()
                )
                (
                    _masked_support_clean_latent_abs_mean,
                    _masked_support_clean_latent_variance,
                ) = self._compute_masked_tensor_abs_mean_and_variance(
                    value_tensor=_inpaint_latent.detach(),
                    mask_tensor=_masked_inpaint_support_mask.detach(),
                )
            _masked_inpaint_editable_mask = torch.clamp(
                true_novel_latent_mask,
                min=0.0,
                max=1.0,
            )
            _masked_inpaint_support_noise = torch.randn_like(_inpaint_latent)
            inpaint_hole_ratio = float(_mask_latent.mean().item())
            _support_projection_ratio_for_schedule = (
                float(
                    torch.nan_to_num(
                        support_projection_mask.float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).mean().item()
                )
                if support_projection_mask is not None
                else 0.0
            )
            _support_confidence_ratio_for_schedule = (
                float(
                    torch.nan_to_num(
                        support_confidence.float(),
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    ).mean().item()
                )
                if support_confidence is not None
                else _support_projection_ratio_for_schedule
            )
            scheduled_inpaint_budget = self._compute_dual_budget_inpaint_schedule(
                true_novel_hole_ratio=inpaint_hole_ratio,
                support_projection_ratio=_support_projection_ratio_for_schedule,
                support_confidence_ratio=_support_confidence_ratio_for_schedule,
            )
            _editable_difficulty_signal = float(
                scheduled_inpaint_budget["editable_difficulty_signal"]
            )
            _support_preserve_budget_signal = float(
                scheduled_inpaint_budget["support_preserve_budget_signal"]
            )
            _scheduled_support_step_progress = float(
                scheduled_inpaint_budget.get("support_step_progress", 0.0)
            )
            _scheduled_support_step_offset = int(scheduled_inpaint_budget["support_step_offset"])
            img2img_strength = float(scheduled_inpaint_budget["img2img_strength"])
            if diagnostic_inpaint_img2img_strength_override is not None:
                img2img_strength = float(
                    max(0.0, min(1.0, diagnostic_inpaint_img2img_strength_override))
                )

        _img2img_mode = rendered_rgb is not None and not _inpaint_mode
        _img2img_start_step = 0  # 去噪起始步索引

        if _img2img_mode:
            # ========== img2img模式: 从3DGS渲染开始部分去噪 ==========
            _render = rendered_rgb.to(dtype=self._model_dtype)
            if _render.dim() == 3:
                _render = _render.unsqueeze(0)
            if _render.shape[-2:] != (_new_h, _new_w):
                _render = F.interpolate(_render, size=(_new_h, _new_w), mode='bilinear', align_corners=False)
            rendered_latent = self._encode_vae(_render)

        if fixed_latent is not None:
            latent = fixed_latent.to(device=ref_latent.device, dtype=ref_latent.dtype)
        elif _inpaint_mode:
            # Warp-Inpaint: 起点是warp图的latent（有效区域保真）
            latent = _inpaint_latent
        elif _img2img_mode:
            latent = rendered_latent
        else:
            latent = torch.randn_like(ref_latent)

        if latent.shape[0] != B:
            if latent.shape[0] == 1 and B > 1:
                latent = latent.expand(B, -1, -1, -1).contiguous()
            else:
                raise RuntimeError(
                    f"Latent batch mismatch: latent_B={latent.shape[0]} vs expected_B={B}"
                )
        # fixed_latent/initial_latent 可能改变latent空间分辨率，需按最终latent重新对齐ControlNet条件尺寸
        _expected_cond_h = int(latent.shape[-2] * 8)
        _expected_cond_w = int(latent.shape[-1] * 8)
        if controlnet_cond.shape[-2:] != (_expected_cond_h, _expected_cond_w):
            controlnet_cond = F.interpolate(
                controlnet_cond,
                size=(_expected_cond_h, _expected_cond_w),
                mode='bilinear',
                align_corners=True,
            )
        if diagnostic_payload is not None:
            controlnet_condition_abs_mean, controlnet_condition_variance = (
                self._compute_tensor_abs_mean_and_variance(
                    value_tensor=controlnet_cond.detach(),
                )
            )
            if controlnet_condition_abs_mean is not None:
                diagnostic_payload["predecode_controlnet_condition_abs_mean"] = float(
                    controlnet_condition_abs_mean
                )
            if controlnet_condition_variance is not None:
                diagnostic_payload["predecode_controlnet_condition_variance"] = float(
                    controlnet_condition_variance
                )
        if predecode_debug_tensors is not None:
            predecode_debug_tensors["predecode_controlnet_condition"] = (
                controlnet_cond.detach().cpu()
            )
        _update_predecode_latent_statistics(
            "predecode_initial_latent",
            latent_tensor=latent,
        )

        # MC采样扰动
        if mc_noise_scale > 0:
            encoder_hidden_states = encoder_hidden_states + mc_noise_scale * torch.randn_like(encoder_hidden_states)
            latent = latent + mc_noise_scale * 0.5 * torch.randn_like(latent)

        # 7. 设置scheduler
        self.inference_scheduler.set_timesteps(num_steps, device=device)

        # 计算起始步并添加噪声（img2img 和 inpaint 共用）
        _use_partial_denoise = (_inpaint_mode or _img2img_mode) and fixed_latent is None
        if _use_partial_denoise:
            _total_steps = len(self.inference_scheduler.timesteps)
            _img2img_start_step = max(0, int(_total_steps * (1.0 - img2img_strength)))
            if _img2img_start_step < _total_steps:
                t_start = self.inference_scheduler.timesteps[_img2img_start_step]
                if (
                    getattr(self, "enable_true_masked_latent_inpainting", False)
                    and
                    _inpaint_mode
                    and _inpaint_latent is not None
                    and _masked_inpaint_support_mask is not None
                    and _masked_inpaint_editable_mask is not None
                    and _masked_inpaint_support_noise is not None
                ):
                    preserved_region_latent = self.inference_scheduler.add_noise(
                        _inpaint_latent,
                        _masked_inpaint_support_noise,
                        t_start.unsqueeze(0),
                    )
                    editable_region_noise = torch.randn_like(_inpaint_latent)
                    latent = (
                        _masked_inpaint_support_mask * preserved_region_latent
                        + _masked_inpaint_editable_mask * editable_region_noise
                    )
                    (
                        _masked_support_noised_latent_abs_mean,
                        _masked_support_noised_latent_variance,
                    ) = self._compute_masked_tensor_abs_mean_and_variance(
                        value_tensor=preserved_region_latent.detach(),
                        mask_tensor=_masked_inpaint_support_mask.detach(),
                    )
                elif (
                    getattr(self, "supports_mask_aware_fallback_sampling", False)
                    and _inpaint_mode
                    and _inpaint_latent is not None
                    and _masked_inpaint_support_mask is not None
                    and _masked_inpaint_editable_mask is not None
                ):
                    editable_region_noise = torch.randn_like(_inpaint_latent)
                    support_region_noise = torch.randn_like(_inpaint_latent)
                    available_support_step_window = max(
                        _total_steps - _img2img_start_step - 1,
                        0,
                    )
                    configured_min_support_step_offset = int(
                        max(0, getattr(self, "minimum_masked_dual_path_support_step_offset", 4))
                    )
                    configured_max_support_step_offset = int(
                        max(
                            configured_min_support_step_offset,
                            getattr(self, "maximum_masked_dual_path_support_step_offset", 8),
                        )
                    )
                    effective_max_support_step_offset = min(
                        configured_max_support_step_offset,
                        available_support_step_window,
                    )
                    effective_min_support_step_offset = min(
                        configured_min_support_step_offset,
                        effective_max_support_step_offset,
                    )
                    if diagnostic_masked_dual_path_support_step_offset_override is not None:
                        support_step_offset = max(
                            0,
                            int(diagnostic_masked_dual_path_support_step_offset_override),
                        )
                        support_step_offset = min(
                            support_step_offset,
                            effective_max_support_step_offset,
                        )
                    elif _scheduled_support_step_progress is not None:
                        support_step_offset = effective_min_support_step_offset + int(
                            round(
                                float(_scheduled_support_step_progress)
                                * max(
                                    effective_max_support_step_offset - effective_min_support_step_offset,
                                    0,
                                )
                            )
                        )
                    elif _scheduled_support_step_offset is not None:
                        support_step_offset = min(
                            max(0, int(_scheduled_support_step_offset)),
                            effective_max_support_step_offset,
                        )
                    else:
                        support_step_offset = min(
                            max(
                                0,
                                int(
                                    getattr(
                                        self,
                                        "masked_latent_dual_path_support_step_ratio",
                                        0.35,
                                    )
                                    * max(available_support_step_window, 0)
                                ),
                            ),
                            effective_max_support_step_offset,
                        )
                    if available_support_step_window <= 0:
                        support_step_offset = 0
                    elif support_step_offset < effective_min_support_step_offset:
                        support_step_offset = effective_min_support_step_offset
                    support_step_offset = max(
                        0,
                        min(support_step_offset, effective_max_support_step_offset),
                    )
                    _masked_dual_path_support_step_offset = int(support_step_offset)
                    support_step_index = min(
                        _total_steps - 1,
                        _img2img_start_step + support_step_offset,
                    )
                    _masked_dual_path_support_step_index = support_step_index
                    support_timestep = self.inference_scheduler.timesteps[support_step_index]
                    support_region_latent = self.inference_scheduler.add_noise(
                        _inpaint_latent,
                        support_region_noise,
                        support_timestep.unsqueeze(0),
                    )
                    editable_region_latent = self.inference_scheduler.add_noise(
                        _inpaint_latent,
                        editable_region_noise,
                        t_start.unsqueeze(0),
                    )
                    latent = (
                        _masked_inpaint_support_mask * support_region_latent
                        + _masked_inpaint_editable_mask * editable_region_latent
                    )
                    (
                        _masked_support_noised_latent_abs_mean,
                        _masked_support_noised_latent_variance,
                    ) = self._compute_masked_tensor_abs_mean_and_variance(
                        value_tensor=support_region_latent.detach(),
                        mask_tensor=_masked_inpaint_support_mask.detach(),
                    )
                else:
                    noise = torch.randn_like(latent)
                    latent = self.inference_scheduler.add_noise(latent, noise, t_start.unsqueeze(0))
        _update_predecode_latent_statistics(
            "predecode_start_latent",
            latent_tensor=latent,
        )
        _store_predecode_preview(
            "predecode_start_latent_rgb",
            latent_tensor=latent,
        )
        _stepwise_masked_dual_path_enabled = bool(
            getattr(self, "enforce_stepwise_masked_latent_dual_path", False)
            and _inpaint_mode
            and _inpaint_latent is not None
            and _masked_inpaint_support_mask is not None
            and _masked_inpaint_support_noise is not None
        )
        
        # 8. 迭代去噪（含CFG支持）
        # ControlNet需要投影后的encoder states
        if self.encoder_proj is not None:
            encoder_hidden_states_f32 = encoder_hidden_states.float()
            cn_encoder_hidden_states = self.encoder_proj(encoder_hidden_states_f32)
            cn_encoder_hidden_states = torch.clamp(cn_encoder_hidden_states, min=-65504.0, max=65504.0)
            cn_encoder_hidden_states = torch.nan_to_num(cn_encoder_hidden_states, nan=0.0)
        else:
            cn_encoder_hidden_states = encoder_hidden_states
        
        # 获取设备和dtype（与forward保持一致）
        controlnet_device = next(self.controlnet.parameters()).device
        controlnet_dtype = next(self.controlnet.parameters()).dtype
        unet_device = next(self.unet.parameters()).device
        unet_dtype = next(self.unet.parameters()).dtype
        
        # CFG预计算: 无条件embedding（空文本，与cond维度相同无需padding）
        _use_cfg = guidance_scale > 1.0
        if _use_cfg:
            null_embedding = self._get_text_embeddings(B, device, prompt="")
            null_embedding_unet = null_embedding.to(device=unet_device, dtype=unet_dtype)
        
        # Layer 2: 截断MC-Dropout — 前k步Dropout ON，后T-k步关闭
        _truncation_active = truncation_k > 0
        _total_steps = len(self.inference_scheduler.timesteps)

        # ===== [CAGS] 预计算 Exist 域约束的 R_gs 参考（latent 空间，避免每步 VAE decode） =====
        # CAGS 理论：最小化 E_consist = ||M_E ⊙ (x_t - R_gs_latent)||² 减少 I_conflict
        # 与 GCD（Novel 域）互补：GCD 约束 Novel 域，CAGS 约束 Exist 域
        _cags_render_latent = None
        if rendered_rgb is not None:
            try:
                _cags_render = rendered_rgb.to(device=ref_latent.device, dtype=self._model_dtype)
                if _cags_render.dim() == 3:
                    _cags_render = _cags_render.unsqueeze(0)
                if _cags_render.shape[-2:] != (_new_h, _new_w):
                    _cags_render = F.interpolate(_cags_render, size=(_new_h, _new_w),
                                                 mode='bilinear', align_corners=False)
                _cags_render_latent = self._encode_vae(_cags_render).detach()
            except Exception as _cags_init_e:
                _cags_render_latent = None

        if enable_latent_blending:
            if _inpaint_mode and _inpaint_latent is not None and _mask_latent is not None:
                _support_projection_clean_latent = _inpaint_latent.detach()
                if support_projection_mask is not None:
                    _support_projection_mask = support_projection_mask.detach()
                elif support_confidence is not None:
                    _support_projection_mask = support_confidence.detach()
                else:
                    _support_projection_mask = (1.0 - _mask_latent).detach()
            elif _cags_render_latent is not None:
                _support_projection_clean_latent = _cags_render_latent.detach()
                if support_projection_mask is not None:
                    _support_projection_mask = support_projection_mask.detach()
                elif support_confidence is not None:
                    _support_projection_mask = support_confidence.detach()
                else:
                    _exist_projection_mask, _ = self._build_exist_novel_masks_from_signals(
                        depth_map=depth_map,
                        warp_confidence=None,
                        target_hw=_support_projection_clean_latent.shape[-2:],
                    )
                    _support_projection_mask = _exist_projection_mask
            if _support_projection_clean_latent is not None and _support_projection_mask is not None:
                if _masked_inpaint_support_noise is not None:
                    _support_projection_noise = _masked_inpaint_support_noise
                else:
                    _support_projection_noise = torch.randn_like(_support_projection_clean_latent)

        if return_diagnostics:
            total_inference_steps = len(self.inference_scheduler.timesteps)
            active_step_indices = list(range(_img2img_start_step, total_inference_steps))
            if active_step_indices:
                predecode_selected_step_tags[active_step_indices[0]] = "start"
                predecode_selected_step_tags[active_step_indices[len(active_step_indices) // 2]] = "mid"
                if (
                    _stepwise_masked_dual_path_enabled
                    and active_step_indices[0] <= _masked_dual_path_support_step_index < total_inference_steps
                ):
                    predecode_selected_step_tags[_masked_dual_path_support_step_index] = "supportswitch"

        for _step_idx, t in enumerate(self.inference_scheduler.timesteps):
            # img2img: 跳过起始步之前的步骤
            if _step_idx < _img2img_start_step:
                continue
            dual_path_temporal_weight = 1.0
            projection_temporal_weight = 1.0
            projection_conflict_gate = 1.0
            # 截断逻辑：到达第k步时关闭Dropout
            if _truncation_active and _step_idx == truncation_k:
                self.disable_mc_dropout()
            if _stepwise_masked_dual_path_enabled:
                dual_path_temporal_weight_values.append(
                    float(dual_path_temporal_weight)
                )
                masked_support_timestep = t
                if not getattr(self, "enable_true_masked_latent_inpainting", False):
                    masked_support_step_index = min(
                        len(self.inference_scheduler.timesteps) - 1,
                        _step_idx + _masked_dual_path_support_step_offset,
                    )
                    masked_support_timestep = self.inference_scheduler.timesteps[
                        masked_support_step_index
                    ]
                if (
                    _masked_support_noised_latent_abs_mean is None
                    and _inpaint_latent is not None
                    and _masked_inpaint_support_noise is not None
                ):
                    masked_support_latent_preview = self.inference_scheduler.add_noise(
                        _inpaint_latent,
                        _masked_inpaint_support_noise,
                        masked_support_timestep.unsqueeze(0),
                    )
                    (
                        _masked_support_noised_latent_abs_mean,
                        _masked_support_noised_latent_variance,
                    ) = self._compute_masked_tensor_abs_mean_and_variance(
                        value_tensor=masked_support_latent_preview.detach(),
                        mask_tensor=_masked_inpaint_support_mask.detach(),
                    )
                composed_dual_path_latent = self._compose_masked_latent_dual_path(
                    editable_region_latent=latent,
                    clean_support_latent=_inpaint_latent,
                    support_region_mask=_masked_inpaint_support_mask,
                    editable_region_mask=_masked_inpaint_editable_mask,
                    support_region_noise=_masked_inpaint_support_noise,
                    support_timestep=masked_support_timestep,
                )
                latent = composed_dual_path_latent
            t_batch = t.expand(B)
            
            # ControlNet (强制移动到正确设备和dtype)
            cn_latent = latent.to(device=controlnet_device, dtype=controlnet_dtype)
            cn_t = t_batch.to(device=controlnet_device)
            cn_states = cn_encoder_hidden_states.to(device=controlnet_device, dtype=controlnet_dtype)
            cn_cond = controlnet_cond.to(device=controlnet_device, dtype=controlnet_dtype)
            
            down_samples, mid_sample = self.controlnet(
                cn_latent,
                cn_t,
                encoder_hidden_states=cn_states,
                controlnet_cond=cn_cond,
                return_dict=False,
            )
            
            # UNet (强制移动到正确设备和dtype)
            unet_latent = latent.to(device=unet_device, dtype=unet_dtype)
            unet_t = t_batch.to(device=unet_device)
            unet_states = encoder_hidden_states.to(device=unet_device, dtype=unet_dtype)
            down_residuals = [d.to(device=unet_device, dtype=unet_dtype) for d in down_samples]
            mid_residual = mid_sample.to(device=unet_device, dtype=unet_dtype)
            step_controlnet_down_residual_abs_mean = None
            step_controlnet_mid_residual_abs_mean = None
            if down_residuals:
                step_controlnet_down_residual_abs_mean = float(
                    torch.stack(
                        [
                            residual_tensor.detach().float().abs().mean()
                            for residual_tensor in down_residuals
                        ]
                    ).mean().item()
                )
                controlnet_down_residual_abs_mean_values.append(
                    step_controlnet_down_residual_abs_mean
                )
            if mid_residual is not None:
                step_controlnet_mid_residual_abs_mean = float(
                    mid_residual.detach().float().abs().mean().item()
                )
                controlnet_mid_residual_abs_mean_values.append(
                    step_controlnet_mid_residual_abs_mean
                )
            
            # 注意：不要对latent和encoder_states应用dropout！
            # MC-Dropout应该作用于模型内部的权重/激活，而不是输入信号。
            # 对latent应用dropout会破坏扩散去噪过程。
            # UNet内部的dropout层通过enable_mc_dropout_for_unet()启用。
            step_cfg_noise_delta_abs_mean = None
            step_cfg_noise_delta_variance = None
            if _use_cfg:
                # --- Classifier-Free Guidance ---
                # 无条件预测: 不注入ControlNet残差，使用null embedding
                noise_pred_uncond = self.unet(
                    unet_latent,
                    unet_t,
                    encoder_hidden_states=null_embedding_unet,
                ).sample
                
                # 有条件预测: 注入ControlNet残差，使用条件embedding
                noise_pred_cond = self.unet(
                    unet_latent,
                    unet_t,
                    encoder_hidden_states=unet_states,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample
                
                # CFG混合: noise_pred = uncond + scale * (cond - uncond)
                cfg_noise_delta = (noise_pred_cond - noise_pred_uncond).detach()
                step_cfg_noise_delta_abs_mean, step_cfg_noise_delta_variance = (
                    self._compute_tensor_abs_mean_and_variance(
                        value_tensor=cfg_noise_delta,
                    )
                )
                if step_cfg_noise_delta_abs_mean is not None:
                    cfg_noise_delta_abs_mean_values.append(step_cfg_noise_delta_abs_mean)
                if step_cfg_noise_delta_variance is not None:
                    cfg_noise_delta_variance_values.append(step_cfg_noise_delta_variance)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 无CFG: 单次有条件前向
                noise_pred = self.unet(
                    unet_latent,
                    unet_t,
                    encoder_hidden_states=unet_states,
                    down_block_additional_residuals=down_residuals,
                    mid_block_additional_residual=mid_residual,
                ).sample
            
            # MC采样：在每一步去噪时对noise_pred添加扰动
            # 这模拟了DDIM采样中eta>0的效果，增加采样多样性
            if mc_noise_scale > 0:
                # 扰动强度随去噪进度衰减（早期步骤扰动更大）
                step_ratio = float(t) / 1000.0  # t越大表示越早期
                noise_strength = mc_noise_scale * step_ratio * 0.5
                noise_pred = noise_pred + noise_strength * torch.randn_like(noise_pred)
            
            # Scheduler step (确保在同一设备)
            latent_before_scheduler_step = latent.detach()
            noise_pred_cpu = noise_pred.to(latent.device)
            latent = self.inference_scheduler.step(
                noise_pred_cpu, t, latent
            ).prev_sample
            scheduler_step_delta = latent.detach() - latent_before_scheduler_step
            step_scheduler_step_delta_abs_mean, _ = self._compute_tensor_abs_mean_and_variance(
                value_tensor=scheduler_step_delta,
            )
            if step_scheduler_step_delta_abs_mean is not None:
                scheduler_step_delta_abs_mean_values.append(
                    step_scheduler_step_delta_abs_mean
                )

            # ===== [GCD] 生成中几何一致约束 (DPS框架) =====
            # 在 Novel 区域（空洞区域）对 latent 注入 warp 梯度约束
            # 原理: x_{t-1} ← x_{t-1} - λ(t)·∇_x L(D(x), W(I_ref))
            # 仅在 Warp 成功（有 _warped_rgb）且处于 inpaint 模式时激活
            _gcd_enabled = (
                enable_gcd_guidance
                and
                _inpaint_mode
                and _cags_render_latent is not None
                and _mask_latent is not None
                and _step_idx >= 1  # 跳过第一步（latent 还很嘈杂）
                and _step_idx <= _total_steps - 2  # 跳过最后一步（保留细节）
            )
            if _gcd_enabled:
                # 自适应 GCD 权重：早期步（t 大）权重高，后期衰减
                _gcd_step_ratio = 1.0 - float(_step_idx) / max(_total_steps, 1)
                _gcd_lambda = 0.05 * _gcd_step_ratio  # 基础权重 0.05，早期最高

                try:
                    # Novel区域正是warp无效区域，不能再用warped_rgb作为参考；
                    # 否则会把空洞区域往错误/零值纹理上拉，直接制造鬼影和黑洞。
                    _novel_reference_latent = _cags_render_latent.to(latent.device)
                    if _novel_reference_latent.shape[-2:] != latent.shape[-2:]:
                        _novel_reference_latent = F.interpolate(
                            _novel_reference_latent,
                            size=latent.shape[-2:],
                            mode='bilinear',
                            align_corners=False,
                        )
                    # 在 latent 空间计算 Novel 区域的 L1 距离梯度
                    with torch.enable_grad():
                        _latent_gcd = latent.detach().clone().requires_grad_(True)
                        # Novel 区域（mask=1）的 L1 距离
                        _novel_mask = _mask_latent.to(latent.device)  # 1=空洞(novel), 0=有效(exist)
                        _gcd_loss = (
                            _novel_mask
                            * (_latent_gcd - _novel_reference_latent.detach()).abs()
                        ).mean()
                        _gcd_grad = torch.autograd.grad(_gcd_loss, _latent_gcd)[0]
                    # 只在 Novel 区域注入梯度修正
                    latent_before_gcd = latent.detach()
                    latent = latent - _gcd_lambda * _novel_mask * _gcd_grad.detach()
                    applied_gcd_steps += 1
                    gcd_delta = latent.detach() - latent_before_gcd
                    step_gcd_delta_abs_mean, _ = self._compute_tensor_abs_mean_and_variance(
                        value_tensor=gcd_delta,
                    )
                    if step_gcd_delta_abs_mean is not None:
                        gcd_delta_abs_mean_values.append(step_gcd_delta_abs_mean)
                    step_gcd_delta_novel_abs_mean, _ = self._compute_masked_tensor_abs_mean_and_variance(
                        value_tensor=gcd_delta,
                        mask_tensor=_novel_mask,
                    )
                    if step_gcd_delta_novel_abs_mean is not None:
                        gcd_delta_novel_abs_mean_values.append(
                            step_gcd_delta_novel_abs_mean
                        )
                except Exception as _gcd_e:
                    if not getattr(self, '_gcd_fail_logged', False):
                        print(f"[GCD] 梯度约束失败（已跳过）: {_gcd_e}")
                        self._gcd_fail_logged = True

            # ===== [CAGS] Exist 域一致性约束 (互补 GCD，解决 weighted 策略 I_conflict 问题) =====
            # 理论（定理2）: ΔL ~ -β₁·I_novel + β₂·I_conflict，最小化 E_consist = ||M_E ⊙ (x - R_gs)||²
            # GCD 约束 Novel 域 → CAGS 约束 Exist 域，两者结合覆盖全图
            # img2img 降级模式（Warp 失败）下 GCD 无效，CAGS 是唯一 Exist 域约束
            _cags_step_enabled = (
                enable_cags_guidance
                and
                _cags_render_latent is not None
                and _step_idx >= 1
                and _step_idx <= _total_steps - 2
            )
            if _cags_step_enabled:
                _cags_step_ratio = 1.0 - float(_step_idx) / max(_total_steps, 1)
                _cags_lambda = 0.04 * _cags_step_ratio  # 自适应权重：早期步更强
                try:
                    # Exist 域掩码（1=Exist，与 _mask_latent 相反）
                    if _inpaint_mode and _mask_latent is not None:
                        _exist_mask = (1.0 - _mask_latent.to(latent.device))
                    else:
                        # img2img 模式：用渲染 latent 方差估计 Exist 域（细节丰富区=高斯覆盖充分）
                        _render_var = _cags_render_latent.var(dim=1, keepdim=True)
                        _exist_mask = (_render_var > _render_var.mean()).float().to(latent.device)
                    # Exist 域 L2 梯度（latent 空间直接计算，无 VAE decode 开销）
                    with torch.enable_grad():
                        _latent_cags = latent.detach().clone().requires_grad_(True)
                        _R_gs = _cags_render_latent.to(latent.device)
                        if _R_gs.shape[-2:] != _latent_cags.shape[-2:]:
                            _R_gs = F.interpolate(_R_gs, size=_latent_cags.shape[-2:],
                                                  mode='bilinear', align_corners=False)
                        if _exist_mask.shape[-2:] != _latent_cags.shape[-2:]:
                            _exist_mask = F.interpolate(_exist_mask, size=_latent_cags.shape[-2:],
                                                        mode='bilinear', align_corners=False)
                        _cags_loss = (_exist_mask * (_latent_cags - _R_gs.detach()).pow(2)).mean()
                        _cags_grad = torch.autograd.grad(_cags_loss, _latent_cags)[0]
                    # 只在 Exist 区域注入梯度（最小化 E_consist）
                    latent_before_cags = latent.detach()
                    latent = latent - _cags_lambda * _exist_mask * _cags_grad.detach()
                    applied_cags_steps += 1
                    cags_delta = latent.detach() - latent_before_cags
                    step_cags_delta_abs_mean, _ = self._compute_tensor_abs_mean_and_variance(
                        value_tensor=cags_delta,
                    )
                    if step_cags_delta_abs_mean is not None:
                        cags_delta_abs_mean_values.append(step_cags_delta_abs_mean)
                    step_cags_delta_exist_abs_mean, _ = self._compute_masked_tensor_abs_mean_and_variance(
                        value_tensor=cags_delta,
                        mask_tensor=_exist_mask,
                    )
                    if step_cags_delta_exist_abs_mean is not None:
                        cags_delta_exist_abs_mean_values.append(
                            step_cags_delta_exist_abs_mean
                        )
                except Exception as _cags_e:
                    if not getattr(self, '_cags_fail_logged', False):
                        print(f"[CAGS] 约束失败（已跳过）: {_cags_e}")
                        self._cags_fail_logged = True

            # ===== [Support-Preserving Projection] =====
            # 对 Exist/support 区域施加自适应投影。
            # 当前步的 scheduler 更新幅度越大，越降低 projection 强度，
            # 以减少 support 保真约束对 editable/novel 域去噪轨迹的过度拉拽。
            if (
                enable_latent_blending
                and _support_projection_clean_latent is not None
                and _support_projection_mask is not None
            ):
                projection_temporal_weight_values.append(
                    float(projection_temporal_weight)
                )
                projection_conflict_gate = self._compute_projection_conflict_gate(
                    scheduler_step_delta_abs_mean=step_scheduler_step_delta_abs_mean,
                )
                projection_conflict_gate_values.append(
                    float(projection_conflict_gate)
                )
                _exist_projection_mask = _support_projection_mask.to(latent.device)
                if _exist_projection_mask.shape[-2:] != latent.shape[-2:]:
                    _exist_projection_mask = F.interpolate(
                        _exist_projection_mask,
                        size=latent.shape[-2:],
                        mode='nearest',
                    )
                if projection_conflict_gate < 1.0:
                    _exist_projection_mask = (
                        _exist_projection_mask * projection_conflict_gate
                    )
                if _step_idx + 1 < len(self.inference_scheduler.timesteps):
                    _projection_timestep = self.inference_scheduler.timesteps[_step_idx + 1]
                    _projected_support_latent = self.inference_scheduler.add_noise(
                        _support_projection_clean_latent.to(latent.device),
                        _support_projection_noise.to(latent.device),
                        _projection_timestep.unsqueeze(0).to(latent.device),
                    )
                else:
                    _projected_support_latent = _support_projection_clean_latent.to(latent.device)
                if _projected_support_latent.shape[-2:] != latent.shape[-2:]:
                    _projected_support_latent = F.interpolate(
                        _projected_support_latent,
                        size=latent.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                latent_before_projection = latent.detach()
                latent = (1.0 - _exist_projection_mask) * latent + _exist_projection_mask * _projected_support_latent
                applied_latent_blend_steps += 1
                projection_delta = latent.detach() - latent_before_projection
                step_projection_delta_abs_mean, _ = self._compute_tensor_abs_mean_and_variance(
                    value_tensor=projection_delta,
                )
                if step_projection_delta_abs_mean is not None:
                    projection_delta_abs_mean_values.append(
                        step_projection_delta_abs_mean
                    )
                step_projection_delta_support_abs_mean, _ = self._compute_masked_tensor_abs_mean_and_variance(
                    value_tensor=projection_delta,
                    mask_tensor=_exist_projection_mask,
                )
                if step_projection_delta_support_abs_mean is not None:
                    projection_delta_support_abs_mean_values.append(
                        step_projection_delta_support_abs_mean
                    )
            selected_step_tag = predecode_selected_step_tags.get(_step_idx)
            if selected_step_tag is not None:
                _update_predecode_latent_statistics(
                    f"predecode_step_{selected_step_tag}_latent",
                    latent_tensor=latent,
                )
                _store_predecode_preview(
                    f"predecode_step_{selected_step_tag}_rgb",
                    latent_tensor=latent,
                )
                if diagnostic_payload is not None:
                    if step_controlnet_down_residual_abs_mean is not None:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_controlnet_down_residual_abs_mean"
                        ] = float(step_controlnet_down_residual_abs_mean)
                    if step_controlnet_mid_residual_abs_mean is not None:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_controlnet_mid_residual_abs_mean"
                        ] = float(step_controlnet_mid_residual_abs_mean)
                    if step_cfg_noise_delta_abs_mean is not None:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_cfg_noise_delta_abs_mean"
                        ] = float(step_cfg_noise_delta_abs_mean)
                    if step_cfg_noise_delta_variance is not None:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_cfg_noise_delta_variance"
                        ] = float(step_cfg_noise_delta_variance)
                    if step_scheduler_step_delta_abs_mean is not None:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_scheduler_step_delta_abs_mean"
                        ] = float(step_scheduler_step_delta_abs_mean)
                    if (
                        enable_latent_blending
                        and _support_projection_clean_latent is not None
                        and _support_projection_mask is not None
                    ):
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_projection_conflict_gate"
                        ] = float(projection_conflict_gate)
                    if _stepwise_masked_dual_path_enabled:
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_dual_path_temporal_weight"
                        ] = float(dual_path_temporal_weight)
                        diagnostic_payload[
                            f"predecode_step_{selected_step_tag}_projection_temporal_weight"
                        ] = float(projection_temporal_weight)

        # 9. 解码
        _update_predecode_latent_statistics(
            "predecode_final_latent",
            latent_tensor=latent,
        )
        decoded_teacher_rgb = self._decode_vae(latent, track_grad=track_grad_decode)
        decoded_teacher_rgb_for_diagnostics = decoded_teacher_rgb.detach().cpu()
        image = decoded_teacher_rgb
        resolved_output_mode = str(output_mode).strip().lower()
        if resolved_output_mode not in {"composed", "proposal_raw", "raw_teacher"}:
            resolved_output_mode = "composed"
        
        # 10. 计算 teacher intervention effect maps
        effect_warp_rgb = warped_rgb
        if effect_warp_rgb is None and conditioning_details is not None:
            effect_warp_rgb = conditioning_details.get("warped_image")
        effect_exist_mask, effect_novel_mask = self._build_exist_novel_masks_from_signals(
            depth_map=depth_map,
            warp_confidence=warp_confidence,
            target_hw=image.shape[-2:],
        )
        if inpaint_mask is not None:
            effect_novel_mask = inpaint_mask.float()
            effect_exist_mask = torch.clamp(1.0 - effect_novel_mask, min=0.0, max=1.0)
        support_base_rgb = effect_warp_rgb if effect_warp_rgb is not None else rendered_rgb
        effective_effect_confidence = _support_projection_mask if _support_projection_mask is not None else warp_confidence
        proposal_outputs = self._build_explicit_frontier_proposal(
            teacher_rgb=decoded_teacher_rgb,
            base_rgb=support_base_rgb,
            support_confidence=effective_effect_confidence,
            support_projection_mask=_support_projection_mask,
            inpaint_mask=effect_novel_mask,
            frontier_mask=frontier_mask,
            verification_prior=verification_prior,
            accum_render=accum_render,
        )
        proposal_raw_image = proposal_outputs.get("proposal_raw_rgb", decoded_teacher_rgb)
        proposal_residual_rgb = proposal_outputs.get("proposal_residual_rgb")
        proposal_applied_residual_rgb = proposal_outputs.get("proposal_applied_residual_rgb")
        proposal_confidence = proposal_outputs.get("proposal_confidence")
        proposal_warp_error_logit = proposal_outputs.get("proposal_warp_error_logit")
        proposal_repairability_logit = proposal_outputs.get("proposal_repairability_logit")
        proposal_verification_logit = proposal_outputs.get("proposal_verification_logit")
        proposal_acceptance_logit = proposal_outputs.get("proposal_acceptance_logit")
        proposal_warp_error_confidence = proposal_outputs.get(
            "proposal_warp_error_confidence"
        )
        proposal_repairability_confidence = proposal_outputs.get(
            "proposal_repairability_confidence"
        )
        proposal_verification_confidence = proposal_outputs.get(
            "proposal_verification_confidence"
        )
        proposal_acceptance_confidence = proposal_outputs.get(
            "proposal_acceptance_confidence"
        )
        proposal_acceptance_gate = proposal_outputs.get("proposal_acceptance_gate")
        proposal_component_gate = proposal_outputs.get("proposal_component_gate")
        proposal_residual_gate = proposal_outputs.get("proposal_residual_gate")
        proposal_focus_mask = proposal_outputs.get("proposal_focus_mask")
        support_composed_image = proposal_raw_image
        if residual_refiner_enabled and accum_render is not None and rendered_rgb is not None:
            residual_refiner_confidence = accum_render
            if residual_refiner_confidence.shape[-2:] != proposal_raw_image.shape[-2:]:
                residual_refiner_confidence = F.interpolate(
                    residual_refiner_confidence,
                    size=proposal_raw_image.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            residual_refiner_confidence = torch.nan_to_num(
                residual_refiner_confidence,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            effect_exist_mask = (residual_refiner_confidence > 1e-6).float()
            effect_novel_mask = torch.clamp(1.0 - effect_exist_mask, min=0.0, max=1.0)
            support_composed_image = self._compose_support_preserving_image(
                teacher_rgb=proposal_raw_image,
                support_rgb=rendered_rgb,
                exist_mask=effect_exist_mask,
                novel_mask=effect_novel_mask,
                support_blend_mask=residual_refiner_confidence,
            )
            effective_effect_confidence = residual_refiner_confidence
        else:
            support_composed_image = self._compose_support_preserving_image(
                teacher_rgb=proposal_raw_image,
                support_rgb=support_base_rgb,
                exist_mask=effect_exist_mask,
                novel_mask=effect_novel_mask,
                support_blend_mask=_support_projection_mask,
            )
        if resolved_output_mode == "composed":
            image = support_composed_image
        elif resolved_output_mode == "proposal_raw":
            image = proposal_raw_image
        else:
            image = decoded_teacher_rgb
        proposal_raw_image_for_diagnostics = proposal_raw_image.detach().cpu()
        proposal_residual_rgb_for_diagnostics = (
            proposal_residual_rgb.detach().cpu()
            if proposal_residual_rgb is not None
            else None
        )
        proposal_applied_residual_rgb_for_diagnostics = (
            proposal_applied_residual_rgb.detach().cpu()
            if proposal_applied_residual_rgb is not None
            else None
        )
        proposal_confidence_for_diagnostics = (
            proposal_confidence.detach().cpu()
            if proposal_confidence is not None
            else None
        )
        proposal_warp_error_logit_for_diagnostics = (
            proposal_warp_error_logit.detach().cpu()
            if proposal_warp_error_logit is not None
            else None
        )
        proposal_repairability_logit_for_diagnostics = (
            proposal_repairability_logit.detach().cpu()
            if proposal_repairability_logit is not None
            else None
        )
        proposal_verification_logit_for_diagnostics = (
            proposal_verification_logit.detach().cpu()
            if proposal_verification_logit is not None
            else None
        )
        proposal_acceptance_logit_for_diagnostics = (
            proposal_acceptance_logit.detach().cpu()
            if proposal_acceptance_logit is not None
            else None
        )
        proposal_warp_error_confidence_for_diagnostics = (
            proposal_warp_error_confidence.detach().cpu()
            if proposal_warp_error_confidence is not None
            else None
        )
        proposal_repairability_confidence_for_diagnostics = (
            proposal_repairability_confidence.detach().cpu()
            if proposal_repairability_confidence is not None
            else None
        )
        proposal_verification_confidence_for_diagnostics = (
            proposal_verification_confidence.detach().cpu()
            if proposal_verification_confidence is not None
            else None
        )
        proposal_acceptance_confidence_for_diagnostics = (
            proposal_acceptance_confidence.detach().cpu()
            if proposal_acceptance_confidence is not None
            else None
        )
        proposal_acceptance_gate_for_diagnostics = (
            proposal_acceptance_gate.detach().cpu()
            if proposal_acceptance_gate is not None
            else None
        )
        proposal_component_gate_for_diagnostics = (
            proposal_component_gate.detach().cpu()
            if proposal_component_gate is not None
            else None
        )
        proposal_residual_gate_for_diagnostics = (
            proposal_residual_gate.detach().cpu()
            if proposal_residual_gate is not None
            else None
        )
        proposal_focus_mask_for_diagnostics = (
            proposal_focus_mask.detach().cpu()
            if proposal_focus_mask is not None
            else None
        )
        support_composed_image_for_diagnostics = support_composed_image.detach().cpu()
        proposal_depth: Optional[torch.Tensor] = None
        proposal_depth_status = "not_requested"
        proposal_alpha: Optional[torch.Tensor] = None
        proposal_alpha_status = "not_requested"
        if return_diagnostics:
            try:
                proposal_depth = self.condition_preprocessor.estimate_depth(
                    proposal_raw_image.detach().to(dtype=torch.float32)
                ).float()
                if proposal_depth.shape[-2:] != proposal_raw_image.shape[-2:]:
                    proposal_depth = F.interpolate(
                        proposal_depth,
                        size=proposal_raw_image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                proposal_depth = torch.nan_to_num(
                    proposal_depth,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).clamp_min(0.0)
                proposal_depth_status = "proposal_rgb_monocular_depth_proxy"
            except Exception as exc:  # noqa: BLE001
                proposal_depth = None
                proposal_depth_status = f"depth_proxy_failed:{exc}"
            alpha_source_tensor = proposal_acceptance_confidence
            alpha_source_name = "proposal_acceptance_confidence"
            if alpha_source_tensor is None:
                alpha_source_tensor = proposal_component_gate
                alpha_source_name = "proposal_component_gate"
            if alpha_source_tensor is None:
                alpha_source_tensor = proposal_focus_mask
                alpha_source_name = "proposal_focus_mask"
            if alpha_source_tensor is not None:
                proposal_alpha = alpha_source_tensor.float()
                if proposal_alpha.shape[-2:] != proposal_raw_image.shape[-2:]:
                    proposal_alpha = F.interpolate(
                        proposal_alpha,
                        size=proposal_raw_image.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                proposal_alpha = torch.nan_to_num(
                    proposal_alpha,
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                ).clamp(0.0, 1.0)
                proposal_alpha_status = f"model_opacity_proxy_from_{alpha_source_name}"
            else:
                proposal_alpha_status = "missing_model_opacity_proxy_source"
        proposal_depth_for_diagnostics = (
            proposal_depth.detach().cpu() if proposal_depth is not None else None
        )
        proposal_alpha_for_diagnostics = (
            proposal_alpha.detach().cpu() if proposal_alpha is not None else None
        )
        risk_exist, gain_novel = self._compute_effect_maps(
            teacher_rgb=image,
            render_rgb=rendered_rgb,
            warp_rgb=effect_warp_rgb,
            warp_confidence=effective_effect_confidence,
            depth_map=depth_map,
            exist_mask=effect_exist_mask,
            novel_mask=effect_novel_mask,
        )

        if return_diagnostics and diagnostic_payload is not None:
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_controlnet_down_residual_abs_mean",
                    scalar_values=controlnet_down_residual_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_controlnet_mid_residual_abs_mean",
                    scalar_values=controlnet_mid_residual_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_cfg_noise_delta_abs_mean",
                    scalar_values=cfg_noise_delta_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_cfg_noise_delta_variance",
                    scalar_values=cfg_noise_delta_variance_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_scheduler_step_delta_abs_mean",
                    scalar_values=scheduler_step_delta_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_gcd_delta_abs_mean",
                    scalar_values=gcd_delta_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_gcd_delta_novel_abs_mean",
                    scalar_values=gcd_delta_novel_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_cags_delta_abs_mean",
                    scalar_values=cags_delta_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_cags_delta_exist_abs_mean",
                    scalar_values=cags_delta_exist_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_dual_path_temporal_weight",
                    scalar_values=dual_path_temporal_weight_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_projection_temporal_weight",
                    scalar_values=projection_temporal_weight_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_projection_conflict_gate",
                    scalar_values=projection_conflict_gate_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_projection_delta_abs_mean",
                    scalar_values=projection_delta_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                self._summarize_scalar_sequence(
                    metric_prefix="predecode_projection_delta_support_abs_mean",
                    scalar_values=projection_delta_support_abs_mean_values,
                )
            )
            diagnostic_payload.update(
                {
                    "generation_mode": "inpaint" if _inpaint_mode else ("img2img" if _img2img_mode else "text2img"),
                    "output_mode": resolved_output_mode,
                    "inpaint_mode": bool(_inpaint_mode),
                    "img2img_mode": bool(_img2img_mode),
                    "stepwise_masked_dual_path_enabled": bool(_stepwise_masked_dual_path_enabled),
                    "native_masked_latent_inpainting_enabled": bool(
                        getattr(self, "enable_true_masked_latent_inpainting", False)
                    ),
                    "masked_dual_path_support_step_offset": int(_masked_dual_path_support_step_offset),
                    "masked_dual_path_support_step_index": int(_masked_dual_path_support_step_index),
                    "supportswitch_temporal_ramp_half_window": int(
                        max(
                            0,
                            getattr(self, "supportswitch_temporal_ramp_half_window", 2),
                        )
                    ),
                    "supportswitch_projection_pre_step_weight": float(
                        _supportswitch_projection_pre_step_weight
                    ),
                    "supportswitch_dual_path_pre_step_weight": float(
                        _supportswitch_dual_path_pre_step_weight
                    ),
                    "support_projection_conflict_target_scheduler_delta": float(
                        _support_projection_conflict_target_scheduler_delta
                    ),
                    "support_projection_conflict_min_scale": float(
                        _support_projection_conflict_min_scale
                    ),
                    "support_projection_conflict_gamma": float(
                        _support_projection_conflict_gamma
                    ),
                    "effective_img2img_strength": float(img2img_strength),
                    "img2img_start_step": int(_img2img_start_step),
                    "editable_difficulty_signal": (
                        float(_editable_difficulty_signal)
                        if _editable_difficulty_signal is not None
                        else None
                    ),
                    "support_preserve_budget_signal": (
                        float(_support_preserve_budget_signal)
                        if _support_preserve_budget_signal is not None
                        else None
                    ),
                    "planned_support_step_progress": (
                        float(_scheduled_support_step_progress)
                        if _scheduled_support_step_progress is not None
                        else None
                    ),
                    "planned_support_step_offset": (
                        int(_scheduled_support_step_offset)
                        if _scheduled_support_step_offset is not None
                        else None
                    ),
                    "support_projection_ratio_for_schedule": (
                        float(_support_projection_ratio_for_schedule)
                        if _support_projection_ratio_for_schedule is not None
                        else None
                    ),
                    "support_confidence_ratio_for_schedule": (
                        float(_support_confidence_ratio_for_schedule)
                        if _support_confidence_ratio_for_schedule is not None
                        else None
                    ),
                    "runtime_support_projection_coverage": (
                        float(_support_projection_mask.mean().item())
                        if _support_projection_mask is not None
                        else None
                    ),
                    "runtime_masked_inpaint_support_coverage": (
                        float(_masked_inpaint_support_mask.mean().item())
                        if _masked_inpaint_support_mask is not None
                        else None
                    ),
                    "runtime_latent_support_mask_coverage": (
                        float(_latent_support_mask_coverage)
                        if _latent_support_mask_coverage is not None
                        else None
                    ),
                    "runtime_masked_inpaint_editable_coverage": (
                        float(_masked_inpaint_editable_mask.mean().item())
                        if _masked_inpaint_editable_mask is not None
                        else None
                    ),
                    "proposal_confidence_coverage": (
                        float(proposal_confidence.mean().item())
                        if proposal_confidence is not None
                        else None
                    ),
                    "proposal_warp_error_coverage": (
                        float(proposal_warp_error_confidence.mean().item())
                        if proposal_warp_error_confidence is not None
                        else None
                    ),
                    "proposal_repairability_coverage": (
                        float(proposal_repairability_confidence.mean().item())
                        if proposal_repairability_confidence is not None
                        else None
                    ),
                    "proposal_verification_coverage": (
                        float(proposal_verification_confidence.mean().item())
                        if proposal_verification_confidence is not None
                        else None
                    ),
                    "proposal_acceptance_coverage": (
                        float(proposal_acceptance_confidence.mean().item())
                        if proposal_acceptance_confidence is not None
                        else None
                    ),
                    "proposal_acceptance_gate_coverage": (
                        float(proposal_acceptance_gate.mean().item())
                        if proposal_acceptance_gate is not None
                        else None
                    ),
                    "proposal_residual_gate_coverage": (
                        float(proposal_residual_gate.mean().item())
                        if proposal_residual_gate is not None
                        else None
                    ),
                    "proposal_focus_coverage": (
                        float(proposal_focus_mask.mean().item())
                        if proposal_focus_mask is not None
                        else None
                    ),
                    "proposal_depth_status": proposal_depth_status,
                    "proposal_alpha_status": proposal_alpha_status,
                    "proposal_depth_alpha_policy": (
                        "proposal_depth is estimated from proposal RGB by the model depth preprocessor; "
                        "proposal_alpha is a model confidence-derived opacity proxy. Neither is copied "
                        "from target or baseline maps, and strict calibration must decide whether they are usable."
                    ),
                    "masked_support_clean_latent_abs_mean": (
                        float(_masked_support_clean_latent_abs_mean)
                        if _masked_support_clean_latent_abs_mean is not None
                        else None
                    ),
                    "masked_support_clean_latent_variance": (
                        float(_masked_support_clean_latent_variance)
                        if _masked_support_clean_latent_variance is not None
                        else None
                    ),
                    "masked_support_noised_latent_abs_mean": (
                        float(_masked_support_noised_latent_abs_mean)
                        if _masked_support_noised_latent_abs_mean is not None
                        else None
                    ),
                    "masked_support_noised_latent_variance": (
                        float(_masked_support_noised_latent_variance)
                        if _masked_support_noised_latent_variance is not None
                        else None
                    ),
                    "masked_inpaint_support_uses_projection": bool(
                        _masked_inpaint_support_uses_projection
                    ),
                    "num_steps": int(num_steps),
                    "guidance_scale": float(guidance_scale),
                    "inpaint_hole_ratio": float(inpaint_hole_ratio) if inpaint_hole_ratio is not None else None,
                    "enable_gcd_guidance": bool(enable_gcd_guidance),
                    "enable_cags_guidance": bool(enable_cags_guidance),
                    "enable_latent_blending": bool(enable_latent_blending),
                    "applied_gcd_steps": int(applied_gcd_steps),
                    "applied_cags_steps": int(applied_cags_steps),
                    "applied_latent_blend_steps": int(applied_latent_blend_steps),
                    "diagnostic_inpaint_img2img_strength_override": (
                        float(diagnostic_inpaint_img2img_strength_override)
                        if diagnostic_inpaint_img2img_strength_override is not None
                        else None
                    ),
                    "diagnostic_masked_dual_path_support_step_offset_override": (
                        int(diagnostic_masked_dual_path_support_step_offset_override)
                        if diagnostic_masked_dual_path_support_step_offset_override is not None
                        else None
                    ),
                }
            )
            return {
                "rgb": image,
                "image": image,
                "epistemic": risk_exist,
                "aleatoric": gain_novel,
                "risk_exist": risk_exist,
                "gain_novel": gain_novel,
                "proposal_raw_rgb": proposal_raw_image,
                "proposal_residual_rgb": proposal_residual_rgb,
                "proposal_applied_residual_rgb": proposal_applied_residual_rgb,
                "proposal_confidence": proposal_confidence,
                "proposal_warp_error_logit": proposal_warp_error_logit,
                "proposal_repairability_logit": proposal_repairability_logit,
                "proposal_verification_logit": proposal_verification_logit,
                "proposal_acceptance_logit": proposal_acceptance_logit,
                "proposal_warp_error_confidence": proposal_warp_error_confidence,
                "proposal_repairability_confidence": proposal_repairability_confidence,
                "proposal_verification_confidence": proposal_verification_confidence,
                "proposal_acceptance_confidence": proposal_acceptance_confidence,
                "proposal_acceptance_gate": proposal_acceptance_gate,
                "proposal_component_gate": proposal_component_gate,
                "proposal_residual_gate": proposal_residual_gate,
                "proposal_depth": proposal_depth,
                "proposal_alpha": proposal_alpha,
                "diagnostics": diagnostic_payload,
                "debug_tensors": {
                    "masked_inpaint_support_mask": (
                        _masked_inpaint_support_mask.detach().cpu()
                        if _masked_inpaint_support_mask is not None
                        else None
                    ),
                    "masked_inpaint_editable_mask": (
                        _masked_inpaint_editable_mask.detach().cpu()
                        if _masked_inpaint_editable_mask is not None
                        else None
                    ),
                    "proposal_raw_rgb": proposal_raw_image_for_diagnostics,
                    "proposal_residual_rgb": proposal_residual_rgb_for_diagnostics,
                    "proposal_applied_residual_rgb": (
                        proposal_applied_residual_rgb_for_diagnostics
                    ),
                    "proposal_confidence": proposal_confidence_for_diagnostics,
                    "proposal_warp_error_logit": proposal_warp_error_logit_for_diagnostics,
                    "proposal_repairability_logit": proposal_repairability_logit_for_diagnostics,
                    "proposal_verification_logit": proposal_verification_logit_for_diagnostics,
                    "proposal_acceptance_logit": proposal_acceptance_logit_for_diagnostics,
                    "proposal_warp_error_confidence": (
                        proposal_warp_error_confidence_for_diagnostics
                    ),
                    "proposal_repairability_confidence": (
                        proposal_repairability_confidence_for_diagnostics
                    ),
                    "proposal_verification_confidence": (
                        proposal_verification_confidence_for_diagnostics
                    ),
                    "proposal_acceptance_confidence": (
                        proposal_acceptance_confidence_for_diagnostics
                    ),
                    "proposal_acceptance_gate": proposal_acceptance_gate_for_diagnostics,
                    "proposal_component_gate": proposal_component_gate_for_diagnostics,
                    "proposal_residual_gate": proposal_residual_gate_for_diagnostics,
                    "proposal_focus_mask": proposal_focus_mask_for_diagnostics,
                    "proposal_depth": proposal_depth_for_diagnostics,
                    "proposal_alpha": proposal_alpha_for_diagnostics,
                    "decoded_teacher_rgb": decoded_teacher_rgb_for_diagnostics,
                    "support_composed_image": support_composed_image_for_diagnostics,
                    **(predecode_debug_tensors or {}),
                },
            }

        return image, risk_exist, gain_novel
    
    @torch.no_grad()
    def generate_view_correctly(
        self,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        target_poses: torch.Tensor,
        rgb_render: Optional[torch.Tensor] = None,
        depth_render: Optional[torch.Tensor] = None,
        normal_render: Optional[torch.Tensor] = None,
        *,
        steps: int = 25,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        enable_mc_dropout: bool = False,
        mc_dropout_p: float = 0.15,  # MC-Dropout概率（推荐0.15-0.2）
        device: Optional[torch.device] = None,
        truncation_k: int = 0,  # Layer 2: 截断步数
        fixed_latent: Optional[torch.Tensor] = None,  # Layer 2: 固定初始噪声
        **kwargs,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        兼容GDDN的generate_view_correctly接口。
        
        与ThreeStageTrainer兼容的生成接口。
        
        Args:
            enable_mc_dropout: 是否启用MC-Dropout采样
            mc_dropout_p: MC-Dropout概率（默认0.15）
            truncation_k: Layer 2截断步数。>0时仅前k步启用Dropout，后续步确定性精化。
            fixed_latent: Layer 2固定初始噪声z_T。确保MC样本间方差仅来自Dropout。
        """
        if fixed_latent is None and kwargs.get("initial_latent") is not None:
            fixed_latent = kwargs["initial_latent"]

        if enable_mc_dropout:
            # 使用新的enable/disable_mc_dropout方法管理Dropout状态
            self.enable_mc_dropout(p=mc_dropout_p)
        
        # 计算MC采样噪声强度
        mc_noise = mc_dropout_p if enable_mc_dropout else 0.0
        return_diagnostics = bool(kwargs.get("return_diagnostics", False))
        return_uncertainty_maps = bool(kwargs.get("return_uncertainty_maps", False))
        diagnostic_disable_gcd = bool(kwargs.get("diagnostic_disable_gcd", False))
        diagnostic_disable_cags = bool(kwargs.get("diagnostic_disable_cags", False))
        diagnostic_disable_latent_blending = bool(kwargs.get("diagnostic_disable_latent_blending", False))
        track_grad_decode = bool(kwargs.get("track_grad_decode", False))
        raw_episode_conditioned_flag = kwargs.get("episode_conditioned", False)
        if isinstance(raw_episode_conditioned_flag, torch.Tensor):
            episode_conditioned = bool((raw_episode_conditioned_flag > 0.5).any().item())
        else:
            episode_conditioned = bool(raw_episode_conditioned_flag)
        provided_support_confidence = kwargs.get("support_confidence")
        provided_warp_valid_mask = kwargs.get("warp_valid_mask")
        provided_support_anchor_rgb = kwargs.get("support_anchor_rgb")
        provided_frontier_mask = kwargs.get("frontier_mask")
        provided_verification_prior = kwargs.get("verification_prior")
        diagnostic_inpaint_img2img_strength_override = kwargs.get(
            "diagnostic_inpaint_img2img_strength_override", None
        )
        if diagnostic_inpaint_img2img_strength_override is not None:
            diagnostic_inpaint_img2img_strength_override = float(diagnostic_inpaint_img2img_strength_override)
        diagnostic_masked_dual_path_support_step_offset_override = kwargs.get(
            "diagnostic_masked_dual_path_support_step_offset_override", None
        )
        if diagnostic_masked_dual_path_support_step_offset_override is not None:
            diagnostic_masked_dual_path_support_step_offset_override = int(
                diagnostic_masked_dual_path_support_step_offset_override
            )
        diagnostic_tensors: Dict[str, torch.Tensor] = {}
        reference_depth_source = "unavailable"
        geometry_depth_render = kwargs.get("geometry_depth_render")
        sparse_depth_maps = kwargs.get("sparse_depth_maps")
        if geometry_depth_render is not None:
            geometry_depth_render = geometry_depth_render.to(
                device=sparse_images.device,
                dtype=torch.float32,
            )
        if sparse_depth_maps is not None:
            sparse_depth_maps = sparse_depth_maps.to(
                device=sparse_images.device,
                dtype=torch.float32,
            )
            if geometry_depth_render.dim() == 3:
                geometry_depth_render = geometry_depth_render.unsqueeze(1)
            if geometry_depth_render.shape[1] > 1:
                geometry_depth_render = geometry_depth_render[:, :1]
            if geometry_depth_render.shape[0] == 1 and target_poses.shape[0] > 1:
                geometry_depth_render = geometry_depth_render.expand(
                    target_poses.shape[0], -1, -1, -1
                )

        target_intrinsics = kwargs.get("camera_intrinsics")
        if target_intrinsics is not None:
            target_intrinsics = target_intrinsics.to(device=sparse_images.device, dtype=torch.float32)
            if target_intrinsics.dim() == 2:
                target_intrinsics = target_intrinsics.unsqueeze(0)
            if target_intrinsics.shape[0] == 1 and target_poses.shape[0] > 1:
                target_intrinsics = target_intrinsics.expand(target_poses.shape[0], -1, -1)
        sparse_intrinsics = self._normalize_sparse_intrinsics(
            sparse_intrinsics=kwargs.get("sparse_intrinsics"),
            batch_size=target_poses.shape[0],
            num_views=sparse_images.shape[1] if sparse_images.dim() == 5 else sparse_images.shape[0],
            device=sparse_images.device,
            dtype=torch.float32,
        )
        reference_intrinsics = kwargs.get("reference_intrinsics")
        if reference_intrinsics is not None:
            reference_intrinsics = reference_intrinsics.to(device=sparse_images.device, dtype=torch.float32)
            if reference_intrinsics.dim() == 2:
                reference_intrinsics = reference_intrinsics.unsqueeze(0)
            if reference_intrinsics.shape[0] == 1 and target_poses.shape[0] > 1:
                reference_intrinsics = reference_intrinsics.expand(target_poses.shape[0], -1, -1)
        reference_depth_map = kwargs.get("reference_depth_map")
        strict_reference_depth = bool(getattr(self, "require_reference_depth_map", False))
        _reference_depth_for_condition = None
        try:
            if reference_depth_map is not None:
                reference_depth_source = "provided_reference_depth"
                _reference_depth_for_condition = reference_depth_map.to(
                    device=target_poses.device if target_poses is not None else reference_depth_map.device,
                    dtype=torch.float32,
                )
                if _reference_depth_for_condition.dim() == 3:
                    _reference_depth_for_condition = _reference_depth_for_condition.unsqueeze(1)
                if _reference_depth_for_condition.shape[1] > 1:
                    _reference_depth_for_condition = _reference_depth_for_condition[:, :1]
                if _reference_depth_for_condition.shape[0] == 1 and target_poses.shape[0] > 1:
                    _reference_depth_for_condition = _reference_depth_for_condition.expand(
                        target_poses.shape[0], -1, -1, -1
                    )
            elif sparse_images is not None and sparse_poses is not None and target_poses is not None:
                if strict_reference_depth:
                    raise ValueError(
                        "reference_depth_map is required when require_reference_depth_map=True"
                    )
                reference_depth_source = "estimated_reference_depth"
                _ref_batch_size = int(target_poses.shape[0])
                _sp_imgs_for_depth = sparse_images
                _sp_poses_for_depth = sparse_poses
                if _sp_imgs_for_depth.dim() == 4:
                    _sp_imgs_for_depth = _sp_imgs_for_depth.unsqueeze(0).expand(
                        _ref_batch_size, -1, -1, -1, -1
                    )
                if _sp_poses_for_depth.dim() == 3:
                    _sp_poses_for_depth = _sp_poses_for_depth.unsqueeze(0).expand(
                        _ref_batch_size, -1, -1, -1
                    )
                _nearest_idx_for_depth = self._select_nearest_view(_sp_poses_for_depth, target_poses)
                _batch_indices_for_depth = torch.arange(_ref_batch_size, device=target_poses.device)
                _ref_imgs_for_depth = _sp_imgs_for_depth[_batch_indices_for_depth, _nearest_idx_for_depth]
                _reference_depth_for_condition = self.condition_preprocessor.estimate_depth(
                    _ref_imgs_for_depth.to(dtype=torch.float32)
                ).float()
                if not getattr(self, "_metric_reference_depth_missing_logged", False):
                    print("[GDDN] 未提供reference_depth_map，当前回退到单目参考深度，几何Warp可能退化")
                    self._metric_reference_depth_missing_logged = True
            if _reference_depth_for_condition is not None and not getattr(self, "_condition_depth_logged", False):
                self._condition_depth_logged = True
        except Exception as _reference_depth_error:
            if strict_reference_depth:
                raise
            reference_depth_source = "failed"
            if not getattr(self, "_condition_depth_fail_logged", False):
                print(f"[GDDN] ControlNet参考深度准备失败，回退内部估计: {_reference_depth_error}")
                self._condition_depth_fail_logged = True
            _reference_depth_for_condition = None

        # 空白渲染检测（img2img fallback）
        _use_img2img_rgb = rgb_render
        render_rgb_std = None
        render_rgb_mean = None
        if rgb_render is not None:
            render_rgb_std = float(rgb_render.float().std().item())
            render_rgb_mean = float(rgb_render.float().mean().item())
            if (not episode_conditioned) and (render_rgb_std < 0.05 or render_rgb_mean < 0.02):
                _use_img2img_rgb = None
                self._blank_render_skip_count = getattr(self, '_blank_render_skip_count', 0) + 1
                if self._blank_render_skip_count <= 2:
                    print(f"[GDDN] 跳过img2img: 渲染质量过低(std={render_rgb_std:.4f}, mean={render_rgb_mean:.4f})")

        # ===== [Warp-Inpaint] 改动2: 深度感知 Warp → 生成 warped_rgb + inpaint_mask =====
        _warped_rgb = None
        _inpaint_mask = None
        _warp_valid_mask = None
        _warp_support_confidence = None
        _prethreshold_valid_mask = None
        _raw_support_confidence = None
        _legacy_raw_support_confidence = None
        _support_quality_confidence = None
        _support_fusion_rgb_disagreement = None
        _union_valid_mask_before_distance_weight = None
        _per_view_valid_coverage = None
        _per_view_distance_weight = None
        _per_view_raw_inverse_distance_weight = None
        _per_view_coverage_distance_weight = None
        _per_view_fusion_distance_weight = None
        _support_projection_mask = None
        _pre_counterfactual_support_projection_mask = None
        _legacy_support_projection_mask = None
        _quality_gated_support_projection_mask = None
        _newly_added_projection_mask = None
        _quality_suspect_projection_mask = None
        _newly_added_quality_suspect_mask = None
        _counterfactual_support_projection_mask = None
        _counterfactual_support_projection_delta_mask = None
        _counterfactual_applied_mask = None
        _support_confidence_optimism_gap = None
        _joint_valid_but_below_projection_threshold_mask = None
        _editable_low_confidence_mask = None
        _low_confidence_known_mask = None
        _true_novel_inpaint_mask = None
        _reference_image = None
        _adaptive_img2img_strength = 0.5  # 默认：Warp成功时由 inpaint模式 接管，Warp失败时在except覆盖
        _minimum_reliable_warp_valid_ratio = float(
            getattr(self, "minimum_reliable_warp_valid_ratio", 0.10)
        )
        _minimum_support_projection_activation_ratio = float(
            getattr(self, "minimum_support_projection_activation_ratio", 0.01)
        )
        _refined_support_outputs: Dict[str, Any] = {}
        _support_projection_is_reliable = True
        _target_batch_size = int(target_poses.shape[0]) if target_poses is not None else 1
        _enable_warp_inpaint = _target_batch_size == 1
        if (not _enable_warp_inpaint) and (not getattr(self, '_warp_batch_skip_logged', False)):
            print(f"[GDDN] batch={_target_batch_size}，跳过Warp-Inpaint并回退到img2img/text2img路径")
            self._warp_batch_skip_logged = True
        if episode_conditioned:
            if rgb_render is not None and provided_support_confidence is not None:
                episode_support_anchor_rgb = provided_support_anchor_rgb
                if episode_support_anchor_rgb is None:
                    episode_support_anchor_rgb = rgb_render
                _warped_rgb = episode_support_anchor_rgb.to(
                    device=sparse_images.device,
                    dtype=self._model_dtype,
                )
                _reference_image = _warped_rgb.detach().clone()
                normalized_episode_support_confidence = provided_support_confidence.to(
                    device=sparse_images.device,
                    dtype=torch.float32,
                )
                if normalized_episode_support_confidence.dim() == 3:
                    normalized_episode_support_confidence = normalized_episode_support_confidence.unsqueeze(1)
                if normalized_episode_support_confidence.shape[1] > 1:
                    normalized_episode_support_confidence = normalized_episode_support_confidence[:, :1]
                if normalized_episode_support_confidence.shape[0] == 1 and _target_batch_size > 1:
                    normalized_episode_support_confidence = normalized_episode_support_confidence.expand(
                        _target_batch_size, -1, -1, -1
                    )
                refined_support_outputs = self._build_refined_support_signals_from_confidence(
                    warp_confidence=normalized_episode_support_confidence,
                    target_hw=tuple(int(dim) for dim in _warped_rgb.shape[-2:]),
                )
                _refined_support_outputs = refined_support_outputs
                _warp_support_confidence = refined_support_outputs.get("support_confidence")
                _warp_valid_mask = refined_support_outputs.get("warp_valid_mask")
                _inpaint_mask = refined_support_outputs.get("inpaint_mask")
                _true_novel_inpaint_mask = refined_support_outputs.get("true_novel_inpaint_mask")
                _support_projection_mask = refined_support_outputs.get("support_projection_mask")
                _low_confidence_known_mask = refined_support_outputs.get("low_confidence_known_mask")
                _editable_low_confidence_mask = refined_support_outputs.get("editable_low_confidence_mask")
                if provided_warp_valid_mask is not None:
                    normalized_episode_warp_valid_mask = provided_warp_valid_mask.to(
                        device=sparse_images.device,
                        dtype=torch.float32,
                    )
                    if normalized_episode_warp_valid_mask.dim() == 3:
                        normalized_episode_warp_valid_mask = normalized_episode_warp_valid_mask.unsqueeze(1)
                    if normalized_episode_warp_valid_mask.shape[1] > 1:
                        normalized_episode_warp_valid_mask = normalized_episode_warp_valid_mask[:, :1]
                    if normalized_episode_warp_valid_mask.shape[0] == 1 and _target_batch_size > 1:
                        normalized_episode_warp_valid_mask = normalized_episode_warp_valid_mask.expand(
                            _target_batch_size, -1, -1, -1
                        )
                    _warp_valid_mask = normalized_episode_warp_valid_mask
                    if _inpaint_mask is None:
                        _inpaint_mask = torch.clamp(1.0 - _warp_valid_mask, min=0.0, max=1.0)
                if _warp_support_confidence is None:
                    _warp_support_confidence = normalized_episode_support_confidence
                if _warp_valid_mask is None:
                    _warp_valid_mask = (_warp_support_confidence > 1.0e-6).float()
                if _inpaint_mask is None:
                    _inpaint_mask = torch.clamp(1.0 - _warp_valid_mask, min=0.0, max=1.0)
                if _true_novel_inpaint_mask is None:
                    _true_novel_inpaint_mask = _inpaint_mask
                _support_projection_is_reliable = bool(
                    _support_projection_mask is not None
                    and float(_support_projection_mask.float().mean().item()) >= _minimum_support_projection_activation_ratio
                )
                if not getattr(self, "_episode_conditioned_support_logged", False):
                    print(
                        "[GDDN] episode-conditioned support path: "
                        "直接复用离线 episode support 信号，跳过 runtime warp 重算"
                    )
                    self._episode_conditioned_support_logged = True
            else:
                if not getattr(self, "_episode_conditioned_missing_signal_logged", False):
                    print(
                        "[GDDN] episode-conditioned 模式缺少 rgb_render/support_confidence，"
                        "当前回退到 runtime warp 路径。"
                    )
                    self._episode_conditioned_missing_signal_logged = True
                episode_conditioned = False
        if (not episode_conditioned) and _enable_warp_inpaint and sparse_images is not None and sparse_poses is not None and target_poses is not None:

            try:
                _B_sp = target_poses.shape[0]
                _sp_imgs = sparse_images
                _sp_poses = sparse_poses
                # 适配维度：确保 [B, V, 3, H, W] 和 [B, V, 4, 4]
                if _sp_imgs.dim() == 4:
                    _sp_imgs = _sp_imgs.unsqueeze(0).expand(_B_sp, -1, -1, -1, -1)
                if _sp_poses.dim() == 3:
                    _sp_poses = _sp_poses.unsqueeze(0).expand(_B_sp, -1, -1, -1)

                support_geometry_depth = geometry_depth_render
                if support_geometry_depth is None:
                    support_geometry_depth = depth_render
                if support_geometry_depth is None:
                    raise ValueError("target depth_render is required for target-driven backward warp")
                _target_depth = support_geometry_depth[:1].to(
                    device=sparse_images.device,
                    dtype=torch.float32,
                )
                if _target_depth.dim() == 3:
                    _target_depth = _target_depth.unsqueeze(1)
                if _target_depth.shape[1] > 1:
                    _target_depth = _target_depth[:, :1]

                joint_support_outputs = self._build_joint_support_base(
                    sparse_images=_sp_imgs[:1].to(dtype=self._model_dtype),
                    sparse_poses=_sp_poses[:1].to(dtype=self._model_dtype),
                    target_pose=target_poses[:1].to(dtype=self._model_dtype),
                    depth_map=_target_depth.to(dtype=self._model_dtype),
                    target_intrinsics=target_intrinsics[:1] if target_intrinsics is not None else None,
                    sparse_intrinsics=sparse_intrinsics[:1] if sparse_intrinsics is not None else None,
                    sparse_depth_maps=sparse_depth_maps[:1] if sparse_depth_maps is not None else None,
                )
                _reference_image = joint_support_outputs["nearest_reference_image"].to(dtype=self._model_dtype)
                _warped_rgb = joint_support_outputs["warped_rgb"].to(dtype=self._model_dtype)
                _valid_mask = joint_support_outputs["valid_mask"].to(dtype=torch.float32)
                _warp_support_confidence = joint_support_outputs["support_confidence"].to(dtype=torch.float32)
                _prethreshold_valid_mask = joint_support_outputs["prethreshold_valid_mask"].to(dtype=torch.float32)
                _raw_support_confidence = joint_support_outputs["raw_support_confidence"].to(dtype=torch.float32)
                _legacy_raw_support_confidence = joint_support_outputs[
                    "legacy_raw_support_confidence"
                ].to(dtype=torch.float32)
                _support_quality_confidence = joint_support_outputs[
                    "support_quality_confidence"
                ].to(dtype=torch.float32)
                _support_fusion_rgb_disagreement = joint_support_outputs[
                    "support_fusion_rgb_disagreement"
                ].to(dtype=torch.float32)
                _union_valid_mask_before_distance_weight = joint_support_outputs[
                    "union_valid_mask_before_distance_weight"
                ].to(dtype=torch.float32)
                _per_view_valid_coverage = joint_support_outputs["per_view_valid_coverage"].to(
                    dtype=torch.float32
                )
                _per_view_distance_weight = joint_support_outputs["per_view_distance_weight"].to(
                    dtype=torch.float32
                )
                _per_view_raw_inverse_distance_weight = joint_support_outputs[
                    "per_view_raw_inverse_distance_weight"
                ].to(dtype=torch.float32)
                _per_view_coverage_distance_weight = joint_support_outputs[
                    "per_view_coverage_distance_weight"
                ].to(dtype=torch.float32)
                _per_view_fusion_distance_weight = joint_support_outputs[
                    "per_view_fusion_distance_weight"
                ].to(dtype=torch.float32)
                reference_intrinsics = joint_support_outputs["nearest_reference_intrinsics"].to(dtype=torch.float32)
                if _reference_depth_for_condition is None:
                    _reference_depth_for_condition = joint_support_outputs["nearest_reference_depth_map"].to(
                        dtype=torch.float32
                    )

                # 空洞 mask: valid=0 → 需要 SD 生成; valid=1 → 保留 warp
                _inpaint_mask = (~_valid_mask.bool()).float()  # [1, 1, H, W]
                _warp_valid_mask = _valid_mask.float()

                refined_support_outputs = self._refine_support_masks_for_sampling(
                    warp_valid_mask=_warp_valid_mask,
                    support_confidence=_warp_support_confidence,
                    raw_support_confidence=_raw_support_confidence,
                    legacy_raw_support_confidence=_legacy_raw_support_confidence,
                    support_quality_confidence=_support_quality_confidence,
                    inpaint_mask=_inpaint_mask,
                )
                _refined_support_outputs = refined_support_outputs
                _warp_valid_mask = refined_support_outputs["warp_valid_mask"].to(dtype=torch.float32)
                _warp_support_confidence = refined_support_outputs["support_confidence"].to(dtype=torch.float32)
                _inpaint_mask = refined_support_outputs["inpaint_mask"].to(dtype=torch.float32)
                _true_novel_inpaint_mask = refined_support_outputs["true_novel_inpaint_mask"].to(dtype=torch.float32)
                _support_projection_mask = refined_support_outputs["support_projection_mask"].to(dtype=torch.float32)
                _pre_counterfactual_support_projection_mask = refined_support_outputs[
                    "pre_counterfactual_support_projection_mask"
                ].to(dtype=torch.float32)
                if refined_support_outputs.get("legacy_support_projection_mask") is not None:
                    _legacy_support_projection_mask = refined_support_outputs[
                        "legacy_support_projection_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("quality_gated_support_projection_mask") is not None:
                    _quality_gated_support_projection_mask = refined_support_outputs[
                        "quality_gated_support_projection_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("newly_added_projection_mask") is not None:
                    _newly_added_projection_mask = refined_support_outputs[
                        "newly_added_projection_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("quality_suspect_projection_mask") is not None:
                    _quality_suspect_projection_mask = refined_support_outputs[
                        "quality_suspect_projection_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("newly_added_quality_suspect_mask") is not None:
                    _newly_added_quality_suspect_mask = refined_support_outputs[
                        "newly_added_quality_suspect_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("counterfactual_support_projection_mask") is not None:
                    _counterfactual_support_projection_mask = refined_support_outputs[
                        "counterfactual_support_projection_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("counterfactual_support_projection_delta_mask") is not None:
                    _counterfactual_support_projection_delta_mask = refined_support_outputs[
                        "counterfactual_support_projection_delta_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("counterfactual_applied_mask") is not None:
                    _counterfactual_applied_mask = refined_support_outputs[
                        "counterfactual_applied_mask"
                    ].to(dtype=torch.float32)
                if refined_support_outputs.get("support_confidence_optimism_gap") is not None:
                    _support_confidence_optimism_gap = refined_support_outputs[
                        "support_confidence_optimism_gap"
                    ].to(dtype=torch.float32)
                _joint_valid_but_below_projection_threshold_mask = refined_support_outputs[
                    "joint_valid_but_below_projection_threshold_mask"
                ].to(dtype=torch.float32)
                _low_confidence_known_mask = refined_support_outputs["low_confidence_known_mask"].to(dtype=torch.float32)
                _editable_low_confidence_mask = refined_support_outputs["editable_low_confidence_mask"].to(dtype=torch.float32)

                # 日志（首次）
                _valid_ratio = float(_warp_valid_mask.float().mean().item())
                _projection_ratio = float(_support_projection_mask.float().mean().item()) if _support_projection_mask is not None else 0.0
                _hole_ratio = float(_inpaint_mask.float().mean().item())
                _low_confidence_known_ratio = float(_low_confidence_known_mask.float().mean().item()) if _low_confidence_known_mask is not None else 0.0
                _projection_strength_scale = float(refined_support_outputs["projection_strength_scale"])
                _support_projection_is_reliable = _projection_ratio >= _minimum_support_projection_activation_ratio
                if not getattr(self, '_warp_logged', False):
                    print(
                        f"[GDDN] Warp完成: valid_ratio={_valid_ratio:.2f}, "
                        f"projection_ratio={_projection_ratio:.2f}, hole_ratio={_hole_ratio:.2f}, "
                        f"low_confidence_known_ratio={_low_confidence_known_ratio:.2f}"
                    )
                    self._warp_logged = True
                if (
                    _projection_strength_scale < 0.999
                    and not getattr(self, "_support_projection_relax_logged", False)
                ):
                    print(
                        f"[GDDN] hole_ratio较高，已自适应放松support-preserving: "
                        f"projection_scale={_projection_strength_scale:.2f}"
                    )
                    self._support_projection_relax_logged = True
                if (
                    not _support_projection_is_reliable
                    and not getattr(self, "_support_projection_skip_logged", False)
                ):
                    print("[GDDN] 高置信support几乎为空，无法执行masked-support保真")
                    self._support_projection_skip_logged = True
                elif (
                    _projection_ratio < _minimum_reliable_warp_valid_ratio
                    and not getattr(self, "_support_projection_sparse_logged", False)
                ):
                    print("[GDDN] 高置信support覆盖较低，将仅保留少量可靠support并扩大可编辑区")
                    self._support_projection_sparse_logged = True
            except Exception as _warp_e:
                # Warp 失败时降级到 img2img/txt2img
                if not getattr(self, '_warp_fail_logged', False):
                    print(f"[GDDN] Warp失败，降级: {_warp_e}")
                    self._warp_fail_logged = True
                _warped_rgb = None
                _inpaint_mask = None
                _support_projection_is_reliable = False
                # Warp 失败时自适应 img2img strength
                # 3DGS 早期渲染质量很差，固定 0.05 几乎无修复效果，需根据渲染质量动态调整
                if _use_img2img_rgb is not None:
                    _r_std = float(_use_img2img_rgb.float().std().item())
                    if _r_std > 0.1:
                        _adaptive_img2img_strength = 0.5   # 渲染质量尚可：中度修复
                    elif _r_std > 0.05:
                        _adaptive_img2img_strength = 0.65  # 渲染质量较差：重度修复
                    else:
                        _adaptive_img2img_strength = 0.85  # 渲染几乎空白：接近txt2img
                    if not getattr(self, '_warp_fallback_logged', False):
                        print(f"[GDDN] Warp降级 img2img_strength={_adaptive_img2img_strength:.2f} "
                              f"(render_std={_r_std:.3f})")
                        self._warp_fallback_logged = True
                else:
                    _adaptive_img2img_strength = 0.85  # 无渲染图：接近txt2img

        if return_diagnostics:
            if _reference_image is not None:
                diagnostic_tensors["reference_image"] = _reference_image.detach().cpu()
            if rgb_render is not None:
                diagnostic_tensors["render_rgb_input"] = rgb_render[:1].detach().cpu()
            if provided_support_anchor_rgb is not None:
                diagnostic_tensors["support_anchor_rgb_input"] = (
                    provided_support_anchor_rgb[:1].detach().cpu()
                )
            if provided_frontier_mask is not None:
                diagnostic_tensors["frontier_mask"] = provided_frontier_mask[:1].detach().cpu()
            if provided_verification_prior is not None:
                diagnostic_tensors["verification_prior"] = provided_verification_prior[:1].detach().cpu()
            if _warped_rgb is not None:
                diagnostic_tensors["warped_rgb"] = _warped_rgb.detach().cpu()
            if _warp_valid_mask is not None:
                diagnostic_tensors["warp_valid_mask"] = _warp_valid_mask.detach().cpu()
            if _warp_support_confidence is not None:
                diagnostic_tensors["warp_support_confidence"] = _warp_support_confidence.detach().cpu()
            if _inpaint_mask is not None:
                diagnostic_tensors["inpaint_mask"] = _inpaint_mask.detach().cpu()
            if _true_novel_inpaint_mask is not None:
                diagnostic_tensors["true_novel_inpaint_mask"] = _true_novel_inpaint_mask.detach().cpu()
            if _support_projection_mask is not None:
                diagnostic_tensors["support_projection_mask"] = _support_projection_mask.detach().cpu()
            if _low_confidence_known_mask is not None:
                diagnostic_tensors["low_confidence_known_mask"] = _low_confidence_known_mask.detach().cpu()
            if _editable_low_confidence_mask is not None:
                diagnostic_tensors["editable_low_confidence_mask"] = _editable_low_confidence_mask.detach().cpu()

        # 使用 generate_view 生成图像
        generate_view_kwargs = {
            "sparse_images": sparse_images,
            "sparse_poses": sparse_poses,
            "target_pose": target_poses,
            "num_steps": steps,
            "guidance_scale": kwargs.get("guidance_scale"),
            "depth_map": depth_render,
            "K": target_intrinsics,
            "reference_depth_map": _reference_depth_for_condition,
            "reference_intrinsics": reference_intrinsics,
            "sparse_intrinsics": sparse_intrinsics,
            "mc_noise_scale": mc_noise,
            "truncation_k": truncation_k,
            "fixed_latent": fixed_latent,
            "rendered_rgb": _use_img2img_rgb,
            "img2img_strength": _adaptive_img2img_strength,
            "warped_rgb": _warped_rgb,
            "support_confidence": _warp_support_confidence,
            "accum_render": kwargs.get("accum_render"),
            "transmittance_render": kwargs.get("transmittance_render"),
            "use_3dgs_residual_refiner": kwargs.get("use_3dgs_residual_refiner", False),
            "support_projection_mask": _support_projection_mask,
            "inpaint_mask": _inpaint_mask,
            "frontier_mask": provided_frontier_mask,
            "verification_prior": provided_verification_prior,
            "enable_gcd_guidance": not diagnostic_disable_gcd,
            "enable_cags_guidance": not diagnostic_disable_cags,
            "enable_latent_blending": not diagnostic_disable_latent_blending,
            "support_projection_is_reliable": _support_projection_is_reliable,
            "diagnostic_inpaint_img2img_strength_override": diagnostic_inpaint_img2img_strength_override,
            "diagnostic_masked_dual_path_support_step_offset_override": (
                diagnostic_masked_dual_path_support_step_offset_override
            ),
            "return_diagnostics": return_diagnostics,
            "track_grad_decode": track_grad_decode,
            "output_mode": kwargs.get("output_mode", "composed"),
        }
        if track_grad_decode:
            undecorated_generate_view = getattr(self.generate_view, "__wrapped__", None)
            if undecorated_generate_view is None:
                undecorated_generate_view = getattr(type(self).generate_view, "__wrapped__", None)
            if undecorated_generate_view is None:
                raise RuntimeError("generate_view 原始可微函数不可用，无法执行 sample-path 直接监督。")
            generate_view_output = undecorated_generate_view(self, **generate_view_kwargs)
        else:
            generate_view_output = self.generate_view(**generate_view_kwargs)

        # 恢复 Dropout 状态
        if enable_mc_dropout:
            self.disable_mc_dropout()

        if return_diagnostics:
            if isinstance(generate_view_output, dict):
                diagnostic_result = dict(generate_view_output)
            else:
                image, epistemic, aleatoric = generate_view_output
                diagnostic_result = {
                    "rgb": image,
                    "image": image,
                    "epistemic": epistemic,
                    "aleatoric": aleatoric,
                    "risk_exist": epistemic,
                    "gain_novel": aleatoric,
                    "diagnostics": {},
                }
            generation_diagnostics = diagnostic_result.get("diagnostics")
            if not isinstance(generation_diagnostics, dict):
                generation_diagnostics = {}
            model_debug_tensors = diagnostic_result.get("debug_tensors")
            if not isinstance(model_debug_tensors, dict):
                model_debug_tensors = {}
            generation_diagnostics.update(
                {
                    "reference_depth_source": reference_depth_source,
                    "render_rgb_mean": render_rgb_mean,
                    "render_rgb_std": render_rgb_std,
                    "used_rendered_rgb": bool(_use_img2img_rgb is not None),
                    "warp_inpaint_enabled": bool(_enable_warp_inpaint),
                    "warp_inpaint_succeeded": bool(_warped_rgb is not None and _inpaint_mask is not None),
                    "warp_valid_ratio": float(_warp_valid_mask.mean().item()) if _warp_valid_mask is not None else None,
                    "union_valid_coverage_before_distance_weight": (
                        float(_union_valid_mask_before_distance_weight.mean().item())
                        if _union_valid_mask_before_distance_weight is not None
                        else None
                    ),
                    "weighted_support_coverage_before_projection_threshold": (
                        float(_raw_support_confidence.mean().item())
                        if _raw_support_confidence is not None
                        else None
                    ),
                    "legacy_weighted_support_coverage_before_projection_threshold": (
                        float(_legacy_raw_support_confidence.mean().item())
                        if _legacy_raw_support_confidence is not None
                        else None
                    ),
                    "support_quality_confidence_coverage": (
                        float(_support_quality_confidence.mean().item())
                        if _support_quality_confidence is not None
                        else None
                    ),
                    "support_fusion_rgb_disagreement": (
                        float(_support_fusion_rgb_disagreement.mean().item())
                        if _support_fusion_rgb_disagreement is not None
                        else None
                    ),
                    "support_confidence_optimism_gap_coverage": (
                        float(_support_confidence_optimism_gap.mean().item())
                        if _support_confidence_optimism_gap is not None
                        else None
                    ),
                    "joint_valid_but_below_projection_threshold_coverage": (
                        float(
                            _joint_valid_but_below_projection_threshold_mask.mean().item()
                        )
                        if _joint_valid_but_below_projection_threshold_mask is not None
                        else None
                    ),
                    "legacy_support_projection_coverage": (
                        float(_legacy_support_projection_mask.mean().item())
                        if _legacy_support_projection_mask is not None
                        else None
                    ),
                    "quality_gated_support_projection_coverage": (
                        float(_quality_gated_support_projection_mask.mean().item())
                        if _quality_gated_support_projection_mask is not None
                        else None
                    ),
                    "newly_added_projection_coverage": (
                        float(_newly_added_projection_mask.mean().item())
                        if _newly_added_projection_mask is not None
                        else None
                    ),
                    "quality_suspect_projection_coverage": (
                        float(_quality_suspect_projection_mask.mean().item())
                        if _quality_suspect_projection_mask is not None
                        else None
                    ),
                    "newly_added_quality_suspect_coverage": (
                        float(_newly_added_quality_suspect_mask.mean().item())
                        if _newly_added_quality_suspect_mask is not None
                        else None
                    ),
                    "support_projection_counterfactual_mode": _refined_support_outputs.get(
                        "support_projection_counterfactual_mode"
                    ),
                    "support_projection_counterfactual_strength": (
                        float(
                            _refined_support_outputs.get(
                                "support_projection_counterfactual_strength", 0.0
                            )
                        )
                        if _refined_support_outputs is not None
                        else 0.0
                    ),
                    "counterfactual_support_projection_coverage": (
                        float(_counterfactual_support_projection_mask.mean().item())
                        if _counterfactual_support_projection_mask is not None
                        else None
                    ),
                    "counterfactual_support_projection_delta_coverage": (
                        float(_counterfactual_support_projection_delta_mask.mean().item())
                        if _counterfactual_support_projection_delta_mask is not None
                        else None
                    ),
                    "counterfactual_applied_mask_coverage": (
                        float(_counterfactual_applied_mask.mean().item())
                        if _counterfactual_applied_mask is not None
                        else None
                    ),
                    "per_view_valid_coverage": (
                        _per_view_valid_coverage[0].detach().cpu().tolist()
                        if _per_view_valid_coverage is not None
                        and _per_view_valid_coverage.dim() == 2
                        and _per_view_valid_coverage.shape[0] == 1
                        else (
                            _per_view_valid_coverage.detach().cpu().tolist()
                            if _per_view_valid_coverage is not None
                            else None
                        )
                    ),
                    "per_view_distance_weight": (
                        _per_view_distance_weight[0].detach().cpu().tolist()
                        if _per_view_distance_weight is not None
                        and _per_view_distance_weight.dim() == 2
                        and _per_view_distance_weight.shape[0] == 1
                        else (
                            _per_view_distance_weight.detach().cpu().tolist()
                            if _per_view_distance_weight is not None
                            else None
                        )
                    ),
                    "per_view_raw_inverse_distance_weight": (
                        _per_view_raw_inverse_distance_weight[0].detach().cpu().tolist()
                        if _per_view_raw_inverse_distance_weight is not None
                        and _per_view_raw_inverse_distance_weight.dim() == 2
                        and _per_view_raw_inverse_distance_weight.shape[0] == 1
                        else (
                            _per_view_raw_inverse_distance_weight.detach().cpu().tolist()
                            if _per_view_raw_inverse_distance_weight is not None
                            else None
                        )
                    ),
                    "per_view_coverage_distance_weight": (
                        _per_view_coverage_distance_weight[0].detach().cpu().tolist()
                        if _per_view_coverage_distance_weight is not None
                        and _per_view_coverage_distance_weight.dim() == 2
                        and _per_view_coverage_distance_weight.shape[0] == 1
                        else (
                            _per_view_coverage_distance_weight.detach().cpu().tolist()
                            if _per_view_coverage_distance_weight is not None
                            else None
                        )
                    ),
                    "per_view_fusion_distance_weight": (
                        _per_view_fusion_distance_weight[0].detach().cpu().tolist()
                        if _per_view_fusion_distance_weight is not None
                        and _per_view_fusion_distance_weight.dim() == 2
                        and _per_view_fusion_distance_weight.shape[0] == 1
                        else (
                            _per_view_fusion_distance_weight.detach().cpu().tolist()
                            if _per_view_fusion_distance_weight is not None
                            else None
                        )
                    ),
                    "diagnostic_disable_gcd": bool(diagnostic_disable_gcd),
                    "diagnostic_disable_cags": bool(diagnostic_disable_cags),
                    "diagnostic_disable_latent_blending": bool(diagnostic_disable_latent_blending),
                    "diagnostic_inpaint_img2img_strength_override": (
                        float(diagnostic_inpaint_img2img_strength_override)
                        if diagnostic_inpaint_img2img_strength_override is not None
                        else None
                    ),
                    "diagnostic_masked_dual_path_support_step_offset_override": (
                        int(diagnostic_masked_dual_path_support_step_offset_override)
                        if diagnostic_masked_dual_path_support_step_offset_override is not None
                        else None
                    ),
                }
            )
            diagnostic_result["diagnostics"] = generation_diagnostics
            merged_debug_tensors = dict(model_debug_tensors)
            merged_debug_tensors.update(diagnostic_tensors)
            diagnostic_result["debug_tensors"] = merged_debug_tensors
            return diagnostic_result

        if return_uncertainty_maps:
            if isinstance(generate_view_output, dict):
                return generate_view_output
            image, epistemic, aleatoric = generate_view_output
            return {
                "rgb": image,
                "image": image,
                "epistemic": epistemic,
                "aleatoric": aleatoric,
                "risk_exist": epistemic,
                "gain_novel": aleatoric,
            }

        image, _, _ = generate_view_output
        return image

    
    def add_uncertainty_heads(self) -> None:
        """添加不确定性预测头（兼容接口）。
        
        GDDN_ControlNet默认已包含不确定性头，此方法用于兼容。
        """
        pass  # 已在__init__中初始化

    @torch.no_grad()
    def generate_with_uncertainty(
        self,
        sparse_images: torch.Tensor,
        sparse_poses: torch.Tensor,
        target_poses: torch.Tensor,
        rgb_render: torch.Tensor,
        depth_render: torch.Tensor,
        normal_render: torch.Tensor,
        reference_depth_map: Optional[torch.Tensor] = None,
        reference_intrinsics: Optional[torch.Tensor] = None,
        *,
        steps: int = 25,
        resolution: Optional[Union[int, Tuple[int, int]]] = None,
        sampler_type: str = "ddim",
    ) -> Dict[str, torch.Tensor]:
        """生成视图并同时输出不确定性估计。
        
        与GDDN.generate_with_uncertainty接口兼容。
        使用MC-Dropout多次采样计算认知/数据不确定性。
        支持多GPU并行MC采样（自动检测可用GPU数量）。
        
        Returns:
            dict: {"rgb": [B,3,H,W], "epistemic": [B,1,H,W], "aleatoric": [B,1,H,W]}
        """
        mc_samples = 12  # MC-Dropout采样次数（12次使方差估计标准误差降至43%）

        # 维度适配: UncertaintyEstimator传入sparse_images=[K,3,H,W]，
        # 但generate_view期望[B,V,3,H,W]
        batch_size = target_poses.shape[0]
        if sparse_images.dim() == 4:
            sparse_images = sparse_images.unsqueeze(0).expand(
                batch_size, *sparse_images.shape
            )
        if sparse_poses.dim() == 2:
            sparse_poses = sparse_poses.unsqueeze(0).expand(
                batch_size, *sparse_poses.shape
            )
        elif sparse_poses.dim() == 3 and sparse_poses.shape[0] != batch_size:
            sparse_poses = sparse_poses.unsqueeze(0).expand(
                batch_size, *sparse_poses.shape
            )

        main_device = target_poses.device
        if torch.cuda.device_count() > 1 and mc_samples > 1 and not getattr(self, "_mc_uncertainty_serial_logged", False):
            print("[GDDN] generate_with_uncertainty 为保持 uncertainty_head 语义一致，当前使用串行 MC 采样")
            self._mc_uncertainty_serial_logged = True

        mc_rgb_results: List[torch.Tensor] = []
        head_risk_exist_results: List[torch.Tensor] = []
        head_gain_novel_results: List[torch.Tensor] = []
        torch.cuda.empty_cache()
        for _mc_idx in range(mc_samples):
            sample_output = self.generate_view_correctly(
                sparse_images=sparse_images,
                sparse_poses=sparse_poses,
                target_poses=target_poses,
                rgb_render=rgb_render,
                depth_render=depth_render,
                normal_render=normal_render,
                reference_depth_map=reference_depth_map,
                reference_intrinsics=reference_intrinsics,
                steps=steps,
                resolution=resolution,
                enable_mc_dropout=True,
                mc_dropout_p=0.15,
                return_uncertainty_maps=True,
            )
            rgb = sample_output["rgb"]
            epistemic_map = sample_output.get("risk_exist", sample_output.get("epistemic"))
            aleatoric_map = sample_output.get("gain_novel", sample_output.get("aleatoric"))
            mc_rgb_results.append(rgb.detach().cpu())
            if epistemic_map is not None:
                head_risk_exist_results.append(epistemic_map.detach().cpu())
            if aleatoric_map is not None:
                head_gain_novel_results.append(aleatoric_map.detach().cpu())
            del rgb
            torch.cuda.empty_cache()

        stacked_rgb = torch.stack(mc_rgb_results, dim=0)
        mean_rgb = stacked_rgb.mean(dim=0)
        mc_epistemic = stacked_rgb.var(dim=0, unbiased=False).mean(dim=1, keepdim=True)

        if head_risk_exist_results:
            head_epistemic = torch.stack(head_risk_exist_results, dim=0).mean(dim=0)
        else:
            head_epistemic = mc_epistemic.clone()

        if head_gain_novel_results:
            head_aleatoric = torch.stack(head_gain_novel_results, dim=0).mean(dim=0)
        else:
            head_aleatoric = torch.zeros_like(head_epistemic)

        del stacked_rgb, mc_rgb_results, head_risk_exist_results, head_gain_novel_results
        return {
            "rgb": mean_rgb.to(main_device),
            "epistemic": mc_epistemic.to(main_device),
            "aleatoric": head_aleatoric.to(main_device),
            "mc_epistemic": mc_epistemic.to(main_device),
            "head_epistemic": head_epistemic.to(main_device),
            "head_aleatoric": head_aleatoric.to(main_device),
            "risk_exist": head_epistemic.to(main_device),
            "gain_novel": head_aleatoric.to(main_device),
            "head_risk_exist": head_epistemic.to(main_device),
            "head_gain_novel": head_aleatoric.to(main_device),
        }

    def cleanup_mc_replicas(self) -> None:
        """释放多GPU MC采样的模型副本，回收显存。"""
        if hasattr(self, '_mc_gpu_replicas') and self._mc_gpu_replicas is not None:
            main_gpu_id = None
            for gpu_id, model in self._mc_gpu_replicas.items():
                if model is self:
                    main_gpu_id = gpu_id
            for gpu_id, model in list(self._mc_gpu_replicas.items()):
                if gpu_id != main_gpu_id:
                    del model
            self._mc_gpu_replicas = None
            torch.cuda.empty_cache()
            print("[MC-Parallel] 模型副本已清理")
    
    def enable_multi_gpu(self, device_ids: Optional[list] = None) -> None:
        """启用多GPU模型并行。
        
        将模型组件分配到多个GPU上：
        - GPU 0: VAE Encoder, Text Encoder, Condition Preprocessor
        - GPU 1: ControlNet, UNet, VAE Decoder
        
        如果只有1个GPU可用，所有组件保持在同一GPU上。
        
        Args:
            device_ids: 可选的GPU设备ID列表，如[0, 1]。如果不提供，自动检测可用GPU。
        """
        if not torch.cuda.is_available():
            print("[GDDN_ControlNet] CUDA not available, skipping multi-GPU setup")
            return
        
        # 禁用TF32以防止多GPU训练时产生NaN
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        print("[GDDN_ControlNet] Disabled TF32 for numerical stability")
        
        # 确定可用的GPU
        if device_ids is None:
            num_gpus = torch.cuda.device_count()
            device_ids = list(range(num_gpus))
        
        if len(device_ids) < 2:
            print(f"[GDDN_ControlNet] Only {len(device_ids)} GPU(s) available, using single GPU mode")
            self._multi_gpu_enabled = False
            self._device_map = None
            return
        
        print(f"[GDDN_ControlNet] Enabling multi-GPU with devices: {device_ids}")
        
        # 创建设备映射
        self._multi_gpu_enabled = True
        decode_device_override = os.environ.get(
            "GDDN_MULTI_GPU_DECODE_DEVICE_INDEX",
            "",
        ).strip()
        auxiliary_device_override = os.environ.get(
            "GDDN_MULTI_GPU_AUX_DEVICE_INDEX",
            "",
        ).strip()
        if decode_device_override:
            try:
                resolved_decode_device_index = int(decode_device_override)
            except ValueError:
                resolved_decode_device_index = device_ids[0] if len(device_ids) >= 3 else device_ids[-1]
            if resolved_decode_device_index not in device_ids:
                resolved_decode_device_index = device_ids[0] if len(device_ids) >= 3 else device_ids[-1]
        else:
            resolved_decode_device_index = (
                device_ids[0]
                if len(device_ids) >= 3
                else device_ids[-1]
            )
        if auxiliary_device_override:
            try:
                resolved_auxiliary_device_index = int(auxiliary_device_override)
            except ValueError:
                resolved_auxiliary_device_index = (
                    device_ids[0]
                    if len(device_ids) >= 3
                    else resolved_decode_device_index
                )
            if resolved_auxiliary_device_index not in device_ids:
                resolved_auxiliary_device_index = (
                    device_ids[0]
                    if len(device_ids) >= 3
                    else resolved_decode_device_index
                )
        else:
            resolved_auxiliary_device_index = (
                device_ids[0]
                if len(device_ids) >= 3
                else resolved_decode_device_index
            )
        decode_device = torch.device(f"cuda:{resolved_decode_device_index}")
        auxiliary_loss_device = torch.device(f"cuda:{resolved_auxiliary_device_index}")
        self._device_map = {
            'encode': torch.device(f'cuda:{device_ids[0]}'),   # VAE编码、文本编码
            'decode': decode_device,                            # VAE解码
            'controlnet': torch.device(f'cuda:{device_ids[1 % len(device_ids)]}'),  # ControlNet
            'unet': torch.device(f'cuda:{device_ids[-1]}'),     # UNet
        }
        self._vae_encode_device = self._device_map['encode']
        self._vae_decode_device = self._device_map['decode']
        self._auxiliary_loss_device = auxiliary_loss_device
        
        # 分配组件到对应GPU
        self.vae.to(self._device_map['encode'])
        self.text_encoder.to(self._device_map['encode'])
        self.condition_preprocessor.to(self._device_map['encode'])
        self.pose_encoder.to(self._device_map['encode'])
        
        # encoder_proj用于将text encoder输出投影到ControlNet维度
        if self.encoder_proj is not None:
            self.encoder_proj.to(self._device_map['encode'])
        
        self.controlnet.to(self._device_map['controlnet'])
        self.unet.to(self._device_map['unet'])
        self.uncertainty_head.to(self._device_map['unet'])
        if hasattr(self, "failure_prior_head") and self.failure_prior_head is not None:
            self.failure_prior_head.to(self._device_map['unet'])
        if hasattr(self, "frontier_proposal_head") and self.frontier_proposal_head is not None:
            self.frontier_proposal_head.to(self._device_map['decode'])

        self._vae_split_across_devices = False
        if (
            self._vae_encode_device != self._vae_decode_device
            and hasattr(self.vae, "encoder")
            and hasattr(self.vae, "decoder")
        ):
            try:
                self.vae.encoder.to(self._vae_encode_device)
                if hasattr(self.vae, "quant_conv") and self.vae.quant_conv is not None:
                    self.vae.quant_conv.to(self._vae_encode_device)
                self.vae.decoder.to(self._vae_decode_device)
                if hasattr(self.vae, "post_quant_conv") and self.vae.post_quant_conv is not None:
                    self.vae.post_quant_conv.to(self._vae_decode_device)
                self._vae_split_across_devices = True
            except Exception as split_exception:
                self._vae_split_across_devices = False
                print(f"[GDDN_ControlNet] WARNING: VAE split failed, fallback to single-device VAE: {split_exception}")
        
        print(f"[GDDN_ControlNet] Multi-GPU device map:")
        for name, device in self._device_map.items():
            print(f"  - {name}: {device}")
        if self._vae_split_across_devices:
            print(
                "[GDDN_ControlNet] VAE split devices: "
                f"encode={self._vae_encode_device}, decode={self._vae_decode_device}"
            )
        print(
            "[GDDN_ControlNet] Auxiliary loss device: "
            f"{self._auxiliary_loss_device}"
        )

    def get_device_info(self) -> Dict[str, Any]:
        """获取当前设备配置信息。"""
        info = {
            'multi_gpu_enabled': getattr(self, '_multi_gpu_enabled', False),
            'device_map': getattr(self, '_device_map', None),
            'cuda_available': torch.cuda.is_available(),
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
        
        if info['cuda_available']:
            info['gpu_names'] = [
                torch.cuda.get_device_name(i) 
                for i in range(info['gpu_count'])
            ]
        
        return info

    def evaluate_health(
        self,
        rgb_batch: torch.Tensor,
        *,
        min_mean: float,
        max_mean: float,
        min_std: float,
    ) -> Tuple[bool, Optional[str], Dict[str, float]]:
        """
        Evaluate statistical health of generated RGB samples.

        Returns:
            (healthy flag, reason if unhealthy, metric summary)
        """
        if rgb_batch.ndim != 4:
            raise ValueError("RGB batch must have shape [B,3,H,W].")
        batch = rgb_batch.detach()
        metrics: Dict[str, float] = {
            "mean": float(batch.mean().item()),
            "std": float(batch.std(unbiased=False).item()),
            "min": float(batch.min().item()),
            "max": float(batch.max().item()),
        }
        if not torch.isfinite(batch).all():
            return False, "non_finite_output", metrics
        per_image_mean = batch.mean(dim=(1, 2, 3))
        per_image_std = batch.std(dim=(1, 2, 3), unbiased=False)
        if torch.isnan(per_image_mean).any() or torch.isnan(per_image_std).any():
            return False, "nan_detected", metrics
        average_mean = float(per_image_mean.mean().item())
        average_std = float(per_image_std.mean().item())
        metrics["per_image_mean"] = average_mean
        metrics["per_image_std"] = average_std
        if average_mean < min_mean or average_mean > max_mean:
            return False, "mean_out_of_range", metrics
        if average_std < min_std:
            return False, "std_below_threshold", metrics
        return True, None, metrics

