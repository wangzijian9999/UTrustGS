"""
ControlNet辅助组件模块。

包含GDDN_ControlNet所需的辅助类:
- PoseEncoder: 相对位姿编码
- ControlNetConditionPreprocessor: ControlNet条件预处理
- UncertaintyHead: 不确定性预测头

设计参考:
- GS-Diff: EscherNet相机位置编码
- Deceptive-3DGS: depth+RGB+uncertainty条件
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== Depth Anything V2 配置 ====================
# 模型路径配置（按优先级查找）
def _optional_env_path(env_name: str) -> Optional[Path]:
    """读取可选环境变量路径，显式过滤空字符串，避免 Path('') 退化成当前目录。"""
    env_value = os.environ.get(env_name)
    if env_value is None:
        return None
    env_value = env_value.strip()
    if not env_value:
        return None
    return Path(env_value)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DA_MODEL_PATHS = [
    _optional_env_path("DEPTH_ANYTHING_V2_PATH"),
    _PROJECT_ROOT / "models/depth_anything_v2/depth_anything_v2_vitg.pth",
]

_DA_MODEL_FILENAMES_BY_TYPE = {
    "vitg": "depth_anything_v2_vitg.pth",
    "vitl": "depth_anything_v2_vitl.pth",
    "vitb": "depth_anything_v2_vitb.pth",
    "vits": "depth_anything_v2_vits.pth",
}

_DA_MODEL_SEARCH_DIRS = [
    _optional_env_path("DEPTH_ANYTHING_V2_DIR"),
    _PROJECT_ROOT / "models/depth_anything_v2",
]

_DA_REPO_PATHS = [
    _PROJECT_ROOT / "models/Depth-Anything-V2",
]

# 添加DA V2仓库到系统路径
for repo_path in _DA_REPO_PATHS:
    if repo_path.exists() and str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
        break

# 惰性加载DA V2
_DepthAnythingV2 = None
_DA_MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}

def _get_depth_anything_v2_class():
    """惰性加载DepthAnythingV2类"""
    global _DepthAnythingV2
    if _DepthAnythingV2 is None:
        try:
            from depth_anything_v2.dpt import DepthAnythingV2
            _DepthAnythingV2 = DepthAnythingV2
        except ImportError as e:
            print(f"[ControlNetComponents] Warning: 无法加载Depth Anything V2: {e}")
            _DepthAnythingV2 = None
    return _DepthAnythingV2

def _infer_da_model_type_from_path(model_path: Path) -> Optional[str]:
    """根据权重文件名推断 DA-V2 backbone 规格。"""
    model_name = model_path.name.lower()
    for model_type, model_filename in _DA_MODEL_FILENAMES_BY_TYPE.items():
        if model_filename.lower() == model_name:
            return model_type
    return None


def _find_da_model_spec(
    preferred_model_types: Tuple[str, ...] = ("vitg", "vitl", "vitb", "vits"),
) -> Optional[Tuple[Path, str]]:
    """查找可用的 DA-V2 权重与对应规格，优先返回更高容量模型。"""
    direct_path_candidates = [
        _optional_env_path("DEPTH_ANYTHING_V2_PATH"),
        _optional_env_path("DEPTH_ANYTHING_V2_CHECKPOINT"),
    ]
    for direct_path_candidate in direct_path_candidates:
        if direct_path_candidate is None:
            continue
        if direct_path_candidate.is_file():
            inferred_model_type = _infer_da_model_type_from_path(direct_path_candidate)
            if inferred_model_type is not None:
                return direct_path_candidate, inferred_model_type
        elif direct_path_candidate.is_dir():
            for preferred_model_type in preferred_model_types:
                model_filename = _DA_MODEL_FILENAMES_BY_TYPE.get(preferred_model_type)
                if model_filename is None:
                    continue
                candidate_checkpoint_path = direct_path_candidate / model_filename
                if candidate_checkpoint_path.is_file():
                    return candidate_checkpoint_path, preferred_model_type

    for explicit_model_path in _DA_MODEL_PATHS:
        if explicit_model_path is None or not explicit_model_path.is_file():
            continue
        inferred_model_type = _infer_da_model_type_from_path(explicit_model_path)
        if inferred_model_type is not None:
            return explicit_model_path, inferred_model_type

    for search_dir in _DA_MODEL_SEARCH_DIRS:
        if search_dir is None or not search_dir.exists():
            continue
        for preferred_model_type in preferred_model_types:
            model_filename = _DA_MODEL_FILENAMES_BY_TYPE.get(preferred_model_type)
            if model_filename is None:
                continue
            candidate_checkpoint_path = search_dir / model_filename
            if candidate_checkpoint_path.is_file():
                return candidate_checkpoint_path, preferred_model_type

    return None


class PoseEncoder(nn.Module):
    """将相对camera pose编码为cross-attention兼容的embedding。
    
    设计参考: GS-Diff的EscherNet相机位置编码
    
    Args:
        pose_dim: 输入pose维度 (12 = 3x4 RT矩阵展平)
        hidden_dim: 隐藏层维度
        output_dim: 输出维度 (匹配SD cross-attention = 1024/768)
        
    输入:
        ref_pose: [B, 4, 4] 参考相机位姿
        target_pose: [B, 4, 4] 目标相机位姿
        
    输出:
        pose_embedding: [B, 1, output_dim] 位姿embedding
    """
    
    def __init__(
        self, 
        pose_dim: int = 12, 
        hidden_dim: int = 256, 
        output_dim: int = 768,  # SD 1.5 ControlNet默认768
    ):
        super().__init__()
        self.output_dim = output_dim
        
        # 相对pose计算 + MLP编码
        self.relative_pose_mlp = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        
    def compute_relative_pose(
        self,
        ref_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        """计算相对位姿: target * inv(ref)
        
        Args:
            ref_pose: [B, 4, 4] 参考位姿
            target_pose: [B, 4, 4] 目标位姿
            
        Returns:
            [B, 12] 展平的相对位姿
        """
        # 确保float32以提高矩阵求逆的数值稳定性
        ref_pose_f32 = ref_pose.float()
        target_pose_f32 = target_pose.float()
        
        ref_inv = torch.inverse(ref_pose_f32)
        relative = target_pose_f32 @ ref_inv
        
        # 提取3x4部分并展平
        relative_3x4 = relative[:, :3, :]  # [B, 3, 4]
        return relative_3x4.reshape(-1, 12)
    
    def forward(
        self,
        ref_pose: torch.Tensor,
        target_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            ref_pose: [B, 4, 4]
            target_pose: [B, 4, 4]
            
        Returns:
            [B, 1, output_dim] pose embedding
        """
        relative_flat = self.compute_relative_pose(ref_pose, target_pose)
        relative_flat = relative_flat.to(self.relative_pose_mlp[0].weight.dtype)
        embedding = self.relative_pose_mlp(relative_flat)
        return embedding.unsqueeze(1)  # [B, 1, output_dim]


class PluckerEmbedder(nn.Module):
    """将目标视角逐像素射线编码为 Plucker token。"""

    def __init__(
        self,
        output_dim: int = 1024,
        hidden_channels: int = 64,
        token_grid_size: int = 8,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_channels = hidden_channels
        self.token_grid_size = token_grid_size
        self.input_projection = nn.Conv2d(6, hidden_channels, kernel_size=1, bias=False)
        self.output_projection = nn.Linear(hidden_channels, output_dim)
        nn.init.orthogonal_(self.input_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        target_pose: torch.Tensor,
        target_intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        batch_size = target_pose.shape[0]
        compute_device = target_pose.device
        compute_dtype = target_pose.dtype

        y_coords, x_coords = torch.meshgrid(
            torch.arange(height, device=compute_device, dtype=compute_dtype),
            torch.arange(width, device=compute_device, dtype=compute_dtype),
            indexing="ij",
        )
        homogeneous_pixels = torch.stack(
            [
                x_coords + 0.5,
                y_coords + 0.5,
                torch.ones_like(x_coords),
            ],
            dim=0,
        ).view(1, 3, -1).expand(batch_size, -1, -1)

        target_intrinsics_inv = torch.linalg.inv(target_intrinsics.float()).to(dtype=compute_dtype)
        ray_directions_camera = target_intrinsics_inv @ homogeneous_pixels
        camera_to_world = torch.linalg.inv(target_pose.float()).to(dtype=compute_dtype)
        rotation_camera_to_world = camera_to_world[:, :3, :3]
        camera_origins_world = camera_to_world[:, :3, 3:4]
        ray_directions_world = rotation_camera_to_world @ ray_directions_camera
        ray_directions_world = F.normalize(ray_directions_world, dim=1)
        expanded_camera_origins = camera_origins_world.expand_as(ray_directions_world)
        plucker_moments = torch.cross(expanded_camera_origins, ray_directions_world, dim=1)
        plucker_map = torch.cat([plucker_moments, ray_directions_world], dim=1)
        plucker_map = plucker_map.view(batch_size, 6, height, width)

        pooled_plucker_map = F.adaptive_avg_pool2d(
            plucker_map,
            output_size=(self.token_grid_size, self.token_grid_size),
        )
        projected_features = self.input_projection(pooled_plucker_map)
        plucker_tokens = projected_features.flatten(2).transpose(1, 2)
        return self.output_projection(plucker_tokens)


class ControlNetConditionPreprocessor(nn.Module):
    """准备ControlNet的多通道条件输入。
    
    设计参考: Deceptive-3DGS的depth+RGB+uncertainty条件
    
    核心功能:
    - 透视变换 perspective_warp
    - 条件融合
    
    Args:
        output_channels: 输出条件通道数（默认3，匹配ControlNet输入）
        use_depth_model: 是否使用内置深度估计模型
    """
    
    def __init__(
        self,
        output_channels: int = 3,
        use_depth_model: bool = True,  # 默认启用DA V2
        depth_model_type: str = "vitg",  # vitg精度更高
        depth_model_path: Optional[str] = None,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.use_depth_model = use_depth_model
        self._depth_model_type = depth_model_type
        
        # 条件融合层
        # 输入: warped_rgb(3) + depth(1) + confidence(1) = 5通道
        input_channels = 5
        self.condition_fusion = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(32, output_channels, 3, padding=1),
        )
        
        # Depth Anything V2 模型（惰性加载）
        self._depth_model = None
        self._depth_model_path = depth_model_path
        self._depth_model_loaded = False
        self._depth_cache = {}  # 缓存深度估计结果
        self.relative_depth_consistency_tolerance = 0.12
        self.forward_backward_pixel_tolerance = 2.5
        self.relative_depth_consistency_relax_factor = 1.5
        self.forward_backward_pixel_relax_factor = 1.5
        
        # 预加载DA V2（如果启用）
        if use_depth_model:
            self._init_depth_model()

    def _build_primary_image_like_condition(
        self,
        *,
        support_rgb: torch.Tensor,
        depth_normalized: torch.Tensor,
        support_confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """构造稳定的 3 通道主条件，并返回低置信已知区域掩码。"""
        depth_3ch = depth_normalized.repeat(1, 3, 1, 1)
        confidence_3ch = support_confidence.repeat(1, 3, 1, 1).clamp(0.0, 1.0)
        primary_condition = (
            (0.8 * support_rgb + 0.2 * depth_3ch) * confidence_3ch
            + depth_3ch * (1.0 - confidence_3ch)
        )
        low_confidence_known_mask = (
            (support_confidence > 1e-6).float()
            * (support_confidence < 0.35).float()
        )
        primary_condition = torch.nan_to_num(
            primary_condition,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        low_confidence_known_mask = torch.nan_to_num(
            low_confidence_known_mask,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp(0.0, 1.0)
        return primary_condition, low_confidence_known_mask
        
    def _init_depth_model(self):
        """初始化Depth Anything V2模型"""
        if self._depth_model_loaded:
            return
        
        DepthAnythingV2 = _get_depth_anything_v2_class()
        if DepthAnythingV2 is None:
            print("[ControlNetConditionPreprocessor] DA V2不可用，将使用亮度fallback")
            self._depth_model_loaded = True
            return
        
        # 查找模型路径
        model_path = None
        resolved_depth_model_type = self._depth_model_type
        if self._depth_model_path:
            explicit_model_path = Path(self._depth_model_path)
            if explicit_model_path.is_dir():
                model_spec = _find_da_model_spec(
                    preferred_model_types=(self._depth_model_type, "vitg", "vitl", "vitb", "vits")
                )
                if model_spec is not None:
                    model_path, resolved_depth_model_type = model_spec
            else:
                model_path = explicit_model_path
                inferred_model_type = _infer_da_model_type_from_path(explicit_model_path)
                if inferred_model_type is not None:
                    resolved_depth_model_type = inferred_model_type
        else:
            model_spec = _find_da_model_spec(
                preferred_model_types=(self._depth_model_type, "vitg", "vitl", "vitb", "vits")
            )
            if model_spec is not None:
                model_path, resolved_depth_model_type = model_spec
        
        if model_path is None or not model_path.exists():
            print(f"[ControlNetConditionPreprocessor] DA V2权重未找到，使用亮度fallback")
            self._depth_model_loaded = True
            return
        
        try:
            # 创建模型
            config = _DA_MODEL_CONFIGS.get(resolved_depth_model_type, _DA_MODEL_CONFIGS['vitb'])
            self._depth_model = DepthAnythingV2(**config)
            self._depth_model_type = resolved_depth_model_type
            
            # 加载权重
            state_dict = torch.load(str(model_path), map_location='cpu')
            self._depth_model.load_state_dict(state_dict)
            self._depth_model.eval()
            
            # 冻结参数
            for param in self._depth_model.parameters():
                param.requires_grad = False
            
            self._depth_model_loaded = True
            
        except Exception as e:
            print(f"[ControlNetConditionPreprocessor] DA V2加载失败: {e}")
            self._depth_model = None
            self._depth_model_loaded = True
    
    def set_depth_model_path(self, path: str):
        """设置深度估计模型路径"""
        self._depth_model_path = path
        self._depth_model_loaded = False
        self._init_depth_model()
    
    def _move_depth_model_to_device(self, device: torch.device):
        """将深度模型移动到指定设备"""
        if self._depth_model is not None:
            # DA-V2 对精度更敏感，且 estimate_depth() 显式使用 float32 输入。
            # 保持该子模块为 float32，避免在混合精度路径下出现
            # "Input type (float) and bias type (c10::Half)" 之类的卷积 dtype 冲突。
            self._depth_model = self._depth_model.to(device=device, dtype=torch.float32)

    def keep_depth_model_in_float32(self) -> None:
        """确保内置深度模型保持 float32。"""
        if self._depth_model is not None:
            self._depth_model = self._depth_model.to(dtype=torch.float32)
    
    @torch.no_grad()
    def estimate_depth(self, image: torch.Tensor) -> torch.Tensor:
        """使用Depth Anything V2估计深度
        
        Args:
            image: [B, 3, H, W] RGB图像，范围[0, 1]
            
        Returns:
            [B, 1, H, W] 归一化深度图，范围[0, 1]
        """
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        
        # 确保模型已初始化
        if not self._depth_model_loaded:
            self._init_depth_model()
        
        # 如果DA V2不可用，使用亮度fallback
        if self._depth_model is None:
            depth = image.mean(dim=1, keepdim=True)
            depth_min = depth.amin(dim=(2, 3), keepdim=True)
            depth_max = depth.amax(dim=(2, 3), keepdim=True)
            depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
            return depth
        
        # 将模型移动到正确设备
        self._move_depth_model_to_device(device)
        
        # DA V2期望输入: [B, 3, H, W]，需要特定的归一化
        # ImageNet归一化: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=dtype).view(1, 3, 1, 1)
        
        # 图像已经是[0,1]范围，直接归一化
        image_normalized = (image - mean) / std
        
        # DA V2推理
        # 输入尺寸需要是14的倍数（DINOv2要求）
        target_h = (H // 14) * 14
        target_w = (W // 14) * 14
        if target_h == 0:
            target_h = 14
        if target_w == 0:
            target_w = 14

        # 调整尺寸
        if (H, W) != (target_h, target_w):
            image_resized = F.interpolate(image_normalized, size=(target_h, target_w), mode='bilinear', align_corners=True)
        else:
            image_resized = image_normalized

        # 使用float32进行推理以保证精度
        image_f32 = image_resized.float()

        # DA-V2 当前路径固定使用 float32。
        # 对新架构 GPU，xFormers 的 memory_efficient_attention 常因
        # capability / dtype 不兼容而失败；这里只在深度估计前临时禁用。
        previous_xformers_disabled = os.environ.get("XFORMERS_DISABLED")
        os.environ["XFORMERS_DISABLED"] = "1"
        try:
            # DA V2 forward返回 [B, H, W] 深度图
            depth = self._depth_model(image_f32)  # [B, H, W]
        finally:
            if previous_xformers_disabled is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = previous_xformers_disabled
        depth = depth.unsqueeze(1)  # [B, 1, H, W]
        
        # 恢复到原始尺寸
        if (H, W) != (target_h, target_w):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)
        
        # 归一化到[0, 1]（逐样本归一化）
        depth_min = depth.amin(dim=(2, 3), keepdim=True)
        depth_max = depth.amax(dim=(2, 3), keepdim=True)
        depth = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        
        return depth.to(dtype)
    
    def perspective_warp(
        self,
        image: torch.Tensor,       # [B, C, H, W]
        depth: torch.Tensor,       # [B, 1, H, W]，target-driven backward warp 使用目标视角深度
        K: torch.Tensor,           # [B, 3, 3] 或 [3, 3]；兼容旧接口，默认同时作为 ref/target 内参
        ref_pose: torch.Tensor,    # [B, 4, 4]
        target_pose: torch.Tensor, # [B, 4, 4]
        ref_K: Optional[torch.Tensor] = None,
        target_K: Optional[torch.Tensor] = None,
        reference_depth_map: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """使用目标深度图将参考图像 backward warp 到目标视角。
        
        `grid_sample()` 的 grid 表示输出像素应从输入图像的哪里采样，因此必须
        按 target pixel -> world -> reference pixel 的方向构造采样网格。
        
        Args:
            image: 参考图像 [B, C, H, W]
            depth: 目标视角深度图 [B, 1, H, W]
            K: 兼容旧接口的单一内参 [B, 3, 3] 或 [3, 3]
            ref_pose: 参考位姿 [B, 4, 4]
            target_pose: 目标位姿 [B, 4, 4]
            ref_K: 参考相机内参 [B, 3, 3] 或 [3, 3]
            target_K: 目标相机内参 [B, 3, 3] 或 [3, 3]
            reference_depth_map: 参考视角深度图 [B, 1, H, W]，用于source-side可见性校验
            return_diagnostics: 是否额外返回几何诊断图。
            
        Returns:
            warped: 变换后图像 [B, C, H, W]
            valid_mask: 有效像素掩码 [B, 1, H, W]
            diagnostics: 可选诊断图，均为 [B, 1, H, W]
        """
        B, C, H, W = image.shape
        device = image.device
        dtype = image.dtype
        warp_diagnostics: Dict[str, torch.Tensor] = {}

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if depth.shape[1] != 1:
            depth = depth[:, :1]
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(
                depth.float(),
                size=(H, W),
                mode='bilinear',
                align_corners=False,
            ).to(dtype=depth.dtype)
        if reference_depth_map is not None:
            if reference_depth_map.dim() == 3:
                reference_depth_map = reference_depth_map.unsqueeze(1)
            if reference_depth_map.shape[1] != 1:
                reference_depth_map = reference_depth_map[:, :1]
            if reference_depth_map.shape[-2:] != (H, W):
                reference_depth_map = F.interpolate(
                    reference_depth_map.float(),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).to(dtype=reference_depth_map.dtype)

        if ref_K is None:
            ref_K = K
        if target_K is None:
            target_K = K

        if ref_K.dim() == 2:
            ref_K = ref_K.unsqueeze(0).expand(B, -1, -1)
        if target_K.dim() == 2:
            target_K = target_K.unsqueeze(0).expand(B, -1, -1)

        ref_K_f32 = ref_K.float()
        target_K_f32 = target_K.float()
        ref_pose_f32 = ref_pose.float()
        target_pose_f32 = target_pose.float()
        depth_f32 = depth.float()

        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        pixels = torch.stack(
            [x_coords, y_coords, torch.ones_like(x_coords)],
            dim=-1,
        )
        pixels_flat = pixels.reshape(-1, 3)

        try:
            target_K_inv = torch.linalg.pinv(target_K_f32)
        except:
            target_K_inv = torch.inverse(target_K_f32)
        target_rays = torch.einsum('bij,hj->bhi', target_K_inv, pixels_flat)

        depth_flat = depth_f32.reshape(B, -1)
        depth_valid_mask = depth_f32.reshape(B, H, W) > 1e-4
        depth_flat = depth_flat.clamp(min=1e-4)
        depth_flat = torch.nan_to_num(depth_flat, nan=0.0, posinf=0.0, neginf=0.0)
        target_points = target_rays * depth_flat.unsqueeze(-1)

        homogeneous_ones = torch.ones(
            B,
            target_points.shape[1],
            1,
            device=device,
            dtype=torch.float32,
        )
        target_points_h = torch.cat([target_points, homogeneous_ones], dim=-1)

        try:
            target_to_world = torch.linalg.pinv(target_pose_f32)
        except:
            target_to_world = torch.inverse(target_pose_f32)
        world_points = torch.einsum('bij,bhj->bhi', target_to_world, target_points_h)
        ref_points = torch.einsum('bij,bhj->bhi', ref_pose_f32, world_points)[..., :3]

        projected_ref = torch.einsum('bij,bhj->bhi', ref_K_f32, ref_points)
        projected_ref_z = projected_ref[..., 2:].clamp(min=1e-4)
        projected_ref_uv = projected_ref[..., :2] / projected_ref_z

        grid_x = 2.0 * projected_ref_uv[..., 0] / max(W - 1, 1) - 1.0
        grid_y = 2.0 * projected_ref_uv[..., 1] / max(H - 1, 1) - 1.0

        grid_x = torch.clamp(grid_x, min=-10.0, max=10.0)
        grid_y = torch.clamp(grid_y, min=-10.0, max=10.0)
        grid_x = torch.nan_to_num(grid_x, nan=0.0, posinf=10.0, neginf=-10.0)
        grid_y = torch.nan_to_num(grid_y, nan=0.0, posinf=10.0, neginf=-10.0)

        grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, H, W, 2)

        warped = F.grid_sample(
            image,
            grid.to(dtype),
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True,
        )

        valid_x = (grid[..., 0] >= -1.0) & (grid[..., 0] <= 1.0)
        valid_y = (grid[..., 1] >= -1.0) & (grid[..., 1] <= 1.0)
        valid_z = ref_points[..., 2].reshape(B, H, W) > 1e-4
        valid_mask = valid_x & valid_y & valid_z & depth_valid_mask
        if reference_depth_map is not None:
            reference_depth_map = reference_depth_map.to(device=device, dtype=torch.float32)
            sampled_reference_depth = F.grid_sample(
                reference_depth_map,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True,
            ).squeeze(1)
            projected_reference_depth = ref_points[..., 2].reshape(B, H, W)
            reference_depth_valid_mask = sampled_reference_depth > 1e-4
            projected_reference_depth = torch.clamp(projected_reference_depth, min=1e-4)
            sampled_reference_depth = torch.clamp(sampled_reference_depth, min=1e-4)
            projected_depth_min = projected_reference_depth.amin(dim=(1, 2), keepdim=True)
            projected_depth_max = projected_reference_depth.amax(dim=(1, 2), keepdim=True)
            sampled_depth_min = sampled_reference_depth.amin(dim=(1, 2), keepdim=True)
            sampled_depth_max = sampled_reference_depth.amax(dim=(1, 2), keepdim=True)
            sampled_reference_depth_aligned = (
                (sampled_reference_depth - sampled_depth_min)
                / (sampled_depth_max - sampled_depth_min + 1e-8)
            ) * (projected_depth_max - projected_depth_min) + projected_depth_min

            relative_depth_consistency_error = (
                sampled_reference_depth_aligned - projected_reference_depth
            ).abs() / torch.maximum(
                torch.maximum(sampled_reference_depth_aligned, projected_reference_depth),
                torch.full_like(projected_reference_depth, 1e-4),
            )
            relative_depth_consistency_relax_factor = float(
                max(getattr(self, "relative_depth_consistency_relax_factor", 1.0), 1.0)
            )
            relative_depth_consistency_hard_mask = (
                relative_depth_consistency_error <= self.relative_depth_consistency_tolerance
            )
            relative_depth_consistency_relaxed_mask = (
                relative_depth_consistency_error
                <= self.relative_depth_consistency_tolerance * relative_depth_consistency_relax_factor
            )
            relative_depth_consistency_confidence = torch.exp(
                -relative_depth_consistency_error / max(self.relative_depth_consistency_tolerance, 1e-4)
            )
            relative_depth_consistency_confidence = torch.nan_to_num(
                relative_depth_consistency_confidence,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)

            reference_pixels = torch.cat(
                [
                    projected_ref_uv,
                    torch.ones(
                        B,
                        projected_ref_uv.shape[1],
                        1,
                        device=device,
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )
            try:
                reference_intrinsics_inverse = torch.linalg.pinv(ref_K_f32)
            except Exception:
                reference_intrinsics_inverse = torch.inverse(ref_K_f32)
            reference_rays = torch.einsum("bij,bhj->bhi", reference_intrinsics_inverse, reference_pixels)
            reference_points_from_sampled_depth = reference_rays * sampled_reference_depth_aligned.reshape(B, -1, 1)
            reference_points_from_sampled_depth_h = torch.cat(
                [
                    reference_points_from_sampled_depth,
                    torch.ones(
                        B,
                        reference_points_from_sampled_depth.shape[1],
                        1,
                        device=device,
                        dtype=torch.float32,
                    ),
                ],
                dim=-1,
            )
            try:
                reference_to_world = torch.linalg.pinv(ref_pose_f32)
            except Exception:
                reference_to_world = torch.inverse(ref_pose_f32)
            backward_world_points = torch.einsum(
                "bij,bhj->bhi",
                reference_to_world,
                reference_points_from_sampled_depth_h,
            )
            backward_target_points = torch.einsum(
                "bij,bhj->bhi",
                target_pose_f32,
                backward_world_points,
            )[..., :3]
            backward_target_projection = torch.einsum(
                "bij,bhj->bhi",
                target_K_f32,
                backward_target_points,
            )
            backward_target_depth = backward_target_projection[..., 2:].clamp(min=1e-4)
            backward_target_uv = backward_target_projection[..., :2] / backward_target_depth
            original_target_uv = pixels_flat[:, :2].unsqueeze(0).expand(B, -1, -1)
            forward_backward_reprojection_error = torch.linalg.norm(
                backward_target_uv - original_target_uv,
                dim=-1,
            ).reshape(B, H, W)
            forward_backward_pixel_relax_factor = float(
                max(getattr(self, "forward_backward_pixel_relax_factor", 1.0), 1.0)
            )
            forward_backward_hard_mask = (
                forward_backward_reprojection_error <= self.forward_backward_pixel_tolerance
            )
            forward_backward_relaxed_mask = (
                forward_backward_reprojection_error
                <= self.forward_backward_pixel_tolerance * forward_backward_pixel_relax_factor
            )
            forward_backward_confidence = torch.exp(
                -forward_backward_reprojection_error / max(self.forward_backward_pixel_tolerance, 1e-4)
            )
            forward_backward_confidence = torch.nan_to_num(
                forward_backward_confidence,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp(0.0, 1.0)
            warp_diagnostics = {
                "relative_depth_consistency_error": torch.nan_to_num(
                    relative_depth_consistency_error.unsqueeze(1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "relative_depth_consistency_confidence": relative_depth_consistency_confidence.unsqueeze(1),
                "forward_backward_reprojection_error": torch.nan_to_num(
                    forward_backward_reprojection_error.unsqueeze(1),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                "forward_backward_confidence": forward_backward_confidence.unsqueeze(1),
            }
            valid_mask = (
                valid_mask.float()
                * reference_depth_valid_mask.float()
                # 保留 coarse geometry safety，但允许略超原始硬阈值的像素
                # 以低置信方式继续参与后续多视图联合聚合。
                * torch.where(
                    relative_depth_consistency_hard_mask,
                    torch.ones_like(relative_depth_consistency_confidence),
                    relative_depth_consistency_relaxed_mask.float(),
                )
                * torch.where(
                    forward_backward_hard_mask,
                    torch.ones_like(forward_backward_confidence),
                    forward_backward_relaxed_mask.float(),
                )
                * relative_depth_consistency_confidence
                * forward_backward_confidence
            )
        else:
            valid_mask = valid_mask.float()
        valid_mask = valid_mask.unsqueeze(1)

        if return_diagnostics:
            return warped, valid_mask, warp_diagnostics
        return warped, valid_mask

    
    def forward(
        self,
        ref_image: torch.Tensor,       # [B, 3, H, W]
        ref_pose: torch.Tensor,        # [B, 4, 4]
        target_pose: torch.Tensor,     # [B, 4, 4]
        depth_map: Optional[torch.Tensor] = None,  # [B, 1, H, W]，目标视角深度
        K: Optional[torch.Tensor] = None,          # [B, 3, 3]，兼容旧接口，默认作为目标视角内参
        reference_depth_map: Optional[torch.Tensor] = None,
        reference_intrinsics: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]],
    ]:
        """
        准备ControlNet条件。
        
        Args:
            ref_image: 参考图像 [B, 3, H, W]
            ref_pose: 参考位姿 [B, 4, 4]
            target_pose: 目标位姿 [B, 4, 4]
            depth_map: 可选目标深度图 [B, 1, H, W]
            K: 兼容旧接口的目标相机内参 [B, 3, 3]
            reference_depth_map: 可选参考视角深度图 [B, 1, H, W]
            reference_intrinsics: 可选参考相机内参 [B, 3, 3]
            target_intrinsics: 可选目标相机内参 [B, 3, 3]
            
        Returns:
            condition: ControlNet条件 [B, output_channels, H, W]
            warp_confidence: 变换置信度 [B, 1, H, W]
        """
        B, C, H, W = ref_image.shape
        device = ref_image.device
        dtype = ref_image.dtype
        
        # 1. 获取或估计深度
        if depth_map is None:
            depth_map = self.estimate_depth(ref_image)
        
        # 2. 默认内参（如果未提供）
        if target_intrinsics is None:
            target_intrinsics = K
        if target_intrinsics is None:
            fx = fy = max(H, W)
            cx, cy = W / 2.0, H / 2.0
            target_intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], device=device, dtype=dtype)
            target_intrinsics = target_intrinsics.unsqueeze(0).expand(B, -1, -1)
        if reference_intrinsics is None:
            reference_intrinsics = target_intrinsics
        
        # 3. 透视变换
        perspective_warp_outputs = self.perspective_warp(
            ref_image,
            depth_map,
            target_intrinsics,
            ref_pose,
            target_pose,
            ref_K=reference_intrinsics,
            target_K=target_intrinsics,
            reference_depth_map=reference_depth_map,
            return_diagnostics=return_details,
        )
        if return_details:
            warped_image, warp_confidence, warp_diagnostics = perspective_warp_outputs
        else:
            warped_image, warp_confidence = perspective_warp_outputs
            warp_diagnostics = {}
        
        # 4. 组合条件输入
        # 将深度归一化到[0,1]
        depth_normalized = depth_map.clone()
        depth_min = depth_normalized.amin(dim=(2, 3), keepdim=True)
        depth_max = depth_normalized.amax(dim=(2, 3), keepdim=True)
        depth_normalized = (depth_normalized - depth_min) / (depth_max - depth_min + 1e-8)
        
        primary_condition, low_confidence_known_mask = self._build_primary_image_like_condition(
            support_rgb=warped_image,
            depth_normalized=depth_normalized,
            support_confidence=warp_confidence,
        )
        raw_geometry_condition = torch.cat(
            [
                warped_image,
                depth_normalized,
                warp_confidence.clamp(0.0, 1.0),
            ],
            dim=1,
        )
        if self.output_channels == 5:
            condition = raw_geometry_condition
        elif self.output_channels == 3:
            condition = primary_condition
        else:
            condition_input = torch.cat([
                warped_image,
                depth_normalized,
                warp_confidence,
            ], dim=1)
            condition = self.condition_fusion(condition_input)
        
        # 最终NaN保护：确保输出没有NaN
        condition = torch.nan_to_num(condition, nan=0.0, posinf=1.0, neginf=0.0)
        warp_confidence = torch.nan_to_num(warp_confidence, nan=0.0, posinf=1.0, neginf=0.0)
        
        if not return_details:
            return condition, warp_confidence

        conditioning_details: Dict[str, torch.Tensor] = {
            "warped_image": warped_image,
            "depth_normalized": depth_normalized,
            "depth_valid_mask": (depth_map > 1e-4).float(),
            "warp_confidence": warp_confidence,
            "primary_condition": primary_condition,
            "raw_geometry_condition": raw_geometry_condition,
            "low_confidence_known_mask": low_confidence_known_mask,
            "editable_low_confidence_mask": low_confidence_known_mask,
        }
        conditioning_details.update(warp_diagnostics)
        return condition, warp_confidence, conditioning_details



class UncertaintyHead(nn.Module):
    """面向 intervention effect 的像素级风险/收益预测头。

    兼容旧接口名称 `UncertaintyHead`，但语义已从
    `epistemic / aleatoric uncertainty`
    转为
    `risk_exist / gain_novel effect maps`。
    """

    def __init__(
        self,
        latent_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        effect_input_channels = 3 + 3 + 3 + 3 + 3 + 3 + 1 + 1 + 1

        self.effect_stem = nn.Sequential(
            nn.Conv2d(effect_input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.risk_exist_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Softplus(),
        )
        self.gain_novel_net = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Softplus(),
        )

        # 兼容旧代码对 epistemic_net / aleatoric_net 的访问。
        self.epistemic_net = self.risk_exist_net
        self.aleatoric_net = self.gain_novel_net
        self._init_weights()

    def _init_weights(self):
        """初始化为输出合理的初始值。"""
        for net in [self.effect_stem, self.risk_exist_net, self.gain_novel_net]:
            for layer in net:
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def _align_optional_map(
        self,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
        default_value: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.full(
                (batch_size, 1, height, width),
                default_value,
                device=device,
                dtype=dtype,
            )
        aligned_tensor = tensor.to(device=device, dtype=dtype)
        if aligned_tensor.dim() == 3:
            aligned_tensor = aligned_tensor.unsqueeze(1)
        if aligned_tensor.shape[-2:] != (height, width):
            aligned_tensor = F.interpolate(
                aligned_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
        if aligned_tensor.shape[1] != 1:
            aligned_tensor = aligned_tensor.mean(dim=1, keepdim=True)
        return aligned_tensor

    def _align_rgb_map(
        self,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)
        aligned_tensor = tensor.to(device=device, dtype=dtype)
        if aligned_tensor.dim() == 3:
            aligned_tensor = aligned_tensor.unsqueeze(0)
        if aligned_tensor.shape[-2:] != (height, width):
            aligned_tensor = F.interpolate(
                aligned_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )
        if aligned_tensor.shape[1] == 1:
            aligned_tensor = aligned_tensor.repeat(1, 3, 1, 1)
        elif aligned_tensor.shape[1] > 3:
            aligned_tensor = aligned_tensor[:, :3]
        return aligned_tensor

    def forward(
        self,
        teacher_rgb: torch.Tensor,
        render_rgb: Optional[torch.Tensor] = None,
        warp_rgb: Optional[torch.Tensor] = None,
        warp_confidence: Optional[torch.Tensor] = None,
        exist_mask: Optional[torch.Tensor] = None,
        novel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_device = next(self.effect_stem.parameters()).device
        target_dtype = next(self.effect_stem.parameters()).dtype

        teacher_rgb = self._align_rgb_map(
            teacher_rgb,
            batch_size=teacher_rgb.shape[0] if teacher_rgb.dim() == 4 else 1,
            height=teacher_rgb.shape[-2],
            width=teacher_rgb.shape[-1],
            device=target_device,
            dtype=target_dtype,
        )
        batch_size, _, image_height, image_width = teacher_rgb.shape
        render_rgb = self._align_rgb_map(
            render_rgb,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            device=target_device,
            dtype=target_dtype,
        )
        warp_rgb = self._align_rgb_map(
            warp_rgb,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            device=target_device,
            dtype=target_dtype,
        )
        warp_confidence = self._align_optional_map(
            warp_confidence,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=1.0,
            device=target_device,
            dtype=target_dtype,
        ).clamp(0.0, 1.0)
        exist_mask = self._align_optional_map(
            exist_mask,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
        ).clamp(0.0, 1.0)
        if novel_mask is None:
            novel_mask = 1.0 - exist_mask
        novel_mask = self._align_optional_map(
            novel_mask,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
        ).clamp(0.0, 1.0)

        teacher_render_delta = (teacher_rgb - render_rgb).abs()
        teacher_warp_delta = (teacher_rgb - warp_rgb).abs()
        render_warp_delta = (render_rgb - warp_rgb).abs()
        effect_input = torch.cat(
            [
                teacher_rgb,
                render_rgb,
                warp_rgb,
                teacher_render_delta,
                teacher_warp_delta,
                render_warp_delta,
                warp_confidence,
                exist_mask,
                novel_mask,
            ],
            dim=1,
        )
        shared_features = self.effect_stem(effect_input)
        risk_exist = self.risk_exist_net(shared_features) * exist_mask
        gain_novel = self.gain_novel_net(shared_features) * novel_mask
        return risk_exist, gain_novel


class FailurePriorHead(nn.Module):
    """Stage-A where_wrong failure prior head.

    MVP语义只校准 wrong / geo_wrong / rgb_trust_proxy / depth_trust_proxy。
    repair 在 Stage A 中保持 uncalibrated，不能进入任何动作路由。
    """

    DEFAULT_ACTIVE_OUTPUTS: Tuple[str, ...] = (
        "wrong",
        "geo_wrong",
        "rgb_trust_proxy",
        "depth_trust_proxy",
    )
    ALL_OUTPUTS: Tuple[str, ...] = (
        "wrong",
        "geo_wrong",
        "repair",
        "rgb_trust_proxy",
        "depth_trust_proxy",
    )

    def __init__(
        self,
        hidden_dim: int = 64,
        active_outputs: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.active_outputs = tuple(active_outputs or self.DEFAULT_ACTIVE_OUTPUTS)
        unknown_outputs = set(self.active_outputs) - set(self.ALL_OUTPUTS)
        if unknown_outputs:
            raise ValueError(f"Unknown FailurePriorHead outputs: {sorted(unknown_outputs)}")
        input_channels = 3 + 3 + 3 + 3 + 3 + 3 + 8
        self.failure_stem = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
        )
        self.output_heads = nn.ModuleDict(
            {
                output_name: nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
                    nn.GroupNorm(4, hidden_dim // 2),
                    nn.SiLU(),
                    nn.Conv2d(hidden_dim // 2, 1, 1),
                    nn.Sigmoid(),
                )
                for output_name in self.ALL_OUTPUTS
            }
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in [self.failure_stem, self.output_heads]:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def _align_optional_map(
        self,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
        default_value: float,
        device: torch.device,
        dtype: torch.dtype,
        normalize: bool = False,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.full(
                (batch_size, 1, height, width),
                default_value,
                device=device,
                dtype=dtype,
            )
        aligned_tensor = tensor.to(device=device, dtype=dtype)
        if aligned_tensor.dim() == 3:
            aligned_tensor = aligned_tensor.unsqueeze(1)
        if aligned_tensor.shape[-2:] != (height, width):
            aligned_tensor = F.interpolate(
                aligned_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        if aligned_tensor.shape[1] != 1:
            aligned_tensor = aligned_tensor.mean(dim=1, keepdim=True)
        aligned_tensor = torch.nan_to_num(aligned_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        if normalize:
            aligned_tensor = robust_normalize_map(
                aligned_tensor,
                valid_mask=valid_mask,
            )
        return aligned_tensor.clamp(0.0, 1.0)

    def _align_rgb_map(
        self,
        tensor: Optional[torch.Tensor],
        *,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if tensor is None:
            return torch.zeros(batch_size, 3, height, width, device=device, dtype=dtype)
        aligned_tensor = tensor.to(device=device, dtype=dtype)
        if aligned_tensor.dim() == 3:
            aligned_tensor = aligned_tensor.unsqueeze(0)
        if aligned_tensor.shape[-2:] != (height, width):
            aligned_tensor = F.interpolate(
                aligned_tensor,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        if aligned_tensor.shape[1] == 1:
            aligned_tensor = aligned_tensor.repeat(1, 3, 1, 1)
        elif aligned_tensor.shape[1] > 3:
            aligned_tensor = aligned_tensor[:, :3]
        return torch.nan_to_num(aligned_tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def forward(
        self,
        teacher_rgb: torch.Tensor,
        render_rgb: Optional[torch.Tensor] = None,
        warp_rgb: Optional[torch.Tensor] = None,
        warp_confidence: Optional[torch.Tensor] = None,
        canonical_support: Optional[torch.Tensor] = None,
        canonical_novel: Optional[torch.Tensor] = None,
        relative_depth_consistency_error: Optional[torch.Tensor] = None,
        forward_backward_reprojection_error: Optional[torch.Tensor] = None,
        fusion_rgb_disagreement: Optional[torch.Tensor] = None,
        support_confidence_gap: Optional[torch.Tensor] = None,
        active_outputs: Optional[Sequence[str]] = None,
        teacher_signal_source: str = "unspecified",
        teacher_signal_is_calibrated: bool = False,
    ) -> Dict[str, Any]:
        target_device = next(self.failure_stem.parameters()).device
        target_dtype = next(self.failure_stem.parameters()).dtype
        if teacher_rgb.dim() == 3:
            teacher_batch_size = 1
        else:
            teacher_batch_size = teacher_rgb.shape[0]
        teacher_rgb = self._align_rgb_map(
            teacher_rgb,
            batch_size=teacher_batch_size,
            height=teacher_rgb.shape[-2],
            width=teacher_rgb.shape[-1],
            device=target_device,
            dtype=target_dtype,
        )
        batch_size, _, image_height, image_width = teacher_rgb.shape
        render_rgb = self._align_rgb_map(
            render_rgb,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            device=target_device,
            dtype=target_dtype,
        )
        warp_rgb = self._align_rgb_map(
            warp_rgb,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            device=target_device,
            dtype=target_dtype,
        )
        warp_confidence = self._align_optional_map(
            warp_confidence,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=1.0,
            device=target_device,
            dtype=target_dtype,
        )
        canonical_support = self._align_optional_map(
            canonical_support,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=warp_confidence.mean().item(),
            device=target_device,
            dtype=target_dtype,
        )
        canonical_novel = self._align_optional_map(
            canonical_novel,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
        )
        valid_mask = (warp_confidence > 1e-6).float()
        relative_depth_consistency_error = self._align_optional_map(
            relative_depth_consistency_error,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
            normalize=True,
            valid_mask=valid_mask,
        )
        forward_backward_reprojection_error = self._align_optional_map(
            forward_backward_reprojection_error,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
            normalize=True,
            valid_mask=valid_mask,
        )
        if fusion_rgb_disagreement is None:
            fusion_rgb_disagreement = (teacher_rgb - warp_rgb).abs().mean(dim=1, keepdim=True)
        fusion_rgb_disagreement = self._align_optional_map(
            fusion_rgb_disagreement,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
            normalize=True,
            valid_mask=valid_mask,
        )
        support_confidence_gap = self._align_optional_map(
            support_confidence_gap,
            batch_size=batch_size,
            height=image_height,
            width=image_width,
            default_value=0.0,
            device=target_device,
            dtype=target_dtype,
        )

        teacher_render_delta = (teacher_rgb - render_rgb).abs()
        teacher_warp_delta = (teacher_rgb - warp_rgb).abs()
        render_warp_delta = (render_rgb - warp_rgb).abs()
        scalar_evidence = torch.cat(
            [
                warp_confidence,
                canonical_support,
                canonical_novel,
                1.0 - warp_confidence,
                relative_depth_consistency_error,
                forward_backward_reprojection_error,
                fusion_rgb_disagreement,
                support_confidence_gap,
            ],
            dim=1,
        )
        failure_input = torch.cat(
            [
                teacher_rgb,
                render_rgb,
                warp_rgb,
                teacher_render_delta,
                teacher_warp_delta,
                render_warp_delta,
                scalar_evidence,
            ],
            dim=1,
        )
        shared_features = self.failure_stem(failure_input)
        resolved_active_outputs = tuple(active_outputs or self.active_outputs)
        output_validity = {
            output_name: output_name in resolved_active_outputs
            for output_name in self.ALL_OUTPUTS
        }
        outputs: Dict[str, Any] = {
            "failure_output_validity": output_validity,
            "failure_active_outputs": resolved_active_outputs,
            "failure_repair_is_calibrated": bool(output_validity.get("repair", False)),
            "failure_teacher_signal_source": str(teacher_signal_source),
            "failure_teacher_signal_is_calibrated": bool(teacher_signal_is_calibrated),
            "failure_online_promotion_allowed": _where_wrong_teacher_signal_is_promotable(
                teacher_signal_source=str(teacher_signal_source),
                teacher_signal_is_calibrated=bool(teacher_signal_is_calibrated),
            ),
        }
        for output_name in self.ALL_OUTPUTS:
            output_key = f"failure_{output_name}"
            if output_validity[output_name]:
                outputs[output_key] = self.output_heads[output_name](shared_features)
            elif output_name == "repair":
                outputs[output_key] = torch.full(
                    (batch_size, 1, image_height, image_width),
                    0.5,
                    device=target_device,
                    dtype=target_dtype,
                )
            else:
                outputs[output_key] = None
        return outputs


def robust_normalize_map(
    tensor: torch.Tensor,
    *,
    valid_mask: Optional[torch.Tensor] = None,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> torch.Tensor:
    """Normalize raw error maps by valid-mask quantiles and clamp to [0, 1]."""

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    normalized = torch.zeros_like(tensor.float())
    tensor_float = torch.nan_to_num(tensor.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        valid_mask = valid_mask.to(device=tensor.device, dtype=torch.bool)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
    for batch_index in range(tensor_float.shape[0]):
        batch_values = tensor_float[batch_index:batch_index + 1]
        if valid_mask is None:
            selected_values = batch_values.reshape(-1)
        else:
            selected_values = batch_values[valid_mask[batch_index:batch_index + 1]]
        selected_values = selected_values[torch.isfinite(selected_values)]
        if selected_values.numel() < 2:
            normalized[batch_index:batch_index + 1] = batch_values.clamp(0.0, 1.0)
            continue
        low_value = torch.quantile(selected_values, lower_quantile)
        high_value = torch.quantile(selected_values, upper_quantile)
        scale = torch.clamp(high_value - low_value, min=1e-6)
        normalized[batch_index:batch_index + 1] = (batch_values - low_value) / scale
    return normalized.clamp(0.0, 1.0).to(dtype=tensor.dtype)


def _where_wrong_teacher_signal_is_promotable(
    *,
    teacher_signal_source: str,
    teacher_signal_is_calibrated: bool,
) -> bool:
    teacher_independent_sources = {
        "teacher_independent_proxy",
        "teacher_independent_weighted_warp_proxy",
        "online_proxy_only",
    }
    if str(teacher_signal_source) in teacher_independent_sources:
        return True
    return bool(teacher_signal_is_calibrated)

