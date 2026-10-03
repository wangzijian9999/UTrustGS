"""
软门控机制（Soft Gating Mechanism）- GDDN 12.0 模块三

本模块实现将不确定性估计转化为引导权重的核心逻辑。

理论背景：
-----------
引导损失的通用形式：
    L_guide = Σ_(v∈V_pseudo) (1/HW) Σ_(x,y) w(x,y) · ||R_v(G)(x,y) - I'_v(x,y)||^p

其中 w(x,y) 是软门控权重，基于不确定性计算：
    - w_aleatoric(x,y) = 1 / (1 + γ·U_aleatoric(x,y))
    - w_epistemic(x,y) = σ(-β(U_epistemic(x,y) - τ))
    - w(x,y) = f(w_aleatoric, w_epistemic)

组合策略：
    a) 相乘：w = w_aleatoric * w_epistemic
    b) 相加归一化：w = normalize(w_aleatoric + w_epistemic)
    c) 加权几何平均：w = w_aleatoric^α * w_epistemic^(1-α)

核心思想：
    - 低不确定性区域 → 高权重（信任先验）
    - 高不确定性区域 → 低权重（拒绝不可靠先验）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .configs.active_sampling_config import ActiveSamplingConfig
from gsplat.utils import compute_gddn_gate


class SoftGatedGuidance(nn.Module):
    """
    软门控引导机制

    根据认知不确定性（epistemic）和数据不确定性（aleatoric）
    计算像素级的引导权重 w(x,y) ∈ [0, 1]。

    Args:
        config: 包含门控超参数的配置对象

    Examples:
        >>> config = ActiveSamplingConfig(
        ...     gate_gamma=1.0,
        ...     gate_beta=10.0,
        ...     gate_tau=0.5,
        ...     gate_strategy="multiply"
        ... )
        >>> gating = SoftGatedGuidance(config)
        >>> u_epistemic = torch.rand(256, 256)  # 认知不确定性
        >>> u_aleatoric = torch.rand(256, 256)  # 数据不确定性
        >>> weights = gating.compute_weights(u_epistemic, u_aleatoric)
        >>> assert 0 <= weights.min() <= weights.max() <= 1
    """

    def __init__(self, config: ActiveSamplingConfig) -> None:
        super().__init__()

        self.epsilon = float(config.gate_epsilon)
        self.gamma = float(config.gate_gamma)
        self.beta = float(config.gate_beta)
        self.tau = float(config.gate_tau)
        strategy = config.gate_strategy.lower()
        if strategy in {"multiply", "product"}:
            self.strategy = "product"
        elif strategy == "geometric":
            self.strategy = "geometric"
        elif strategy == "add":
            self.strategy = "add"
        else:
            raise ValueError(
                f"不支持的组合策略: {config.gate_strategy}. "
                f"可选: 'product', 'geometric', 'add'"
            )
        self.alpha = float(config.gate_alpha)

        # 参数合法性检查
        if self.epsilon <= 0:
            raise ValueError(f"epsilon必须为正数，当前值: {self.epsilon}")
        if self.gamma < 0:
            raise ValueError(f"gamma必须非负，当前值: {self.gamma}")
        if self.beta <= 0:
            raise ValueError(f"beta必须为正数，当前值: {self.beta}")
        if self.tau < 0:
            raise ValueError(f"tau必须非负，当前值: {self.tau}")
        if not 0 <= self.alpha <= 1:
            raise ValueError(f"alpha必须在[0,1]区间，当前值: {self.alpha}")
        self.variance_floor = float(getattr(config, "gate_variance_floor", 0.0))

    def compute_weights(
        self,
        u_epistemic: torch.Tensor,
        u_aleatoric: torch.Tensor
    ) -> torch.Tensor:
        """
        计算软门控权重

        Args:
            u_epistemic: 认知不确定性图 [H, W] 或 [B, H, W] 或 [B, 1, H, W]
            u_aleatoric: 数据不确定性图 [H, W] 或 [B, H, W] 或 [B, 1, H, W]

        Returns:
            weights: 软门控权重图，形状与输入相同，值域 [0, 1]
        """
        # 处理输入形状
        u_epi = self._squeeze_to_2d_or_3d(u_epistemic)
        u_ale = self._squeeze_to_2d_or_3d(u_aleatoric)
        if self.variance_floor > 0.0:
            u_epi = torch.clamp(u_epi, min=self.variance_floor)
            u_ale = torch.clamp(u_ale, min=self.variance_floor)

        # 形状检查
        if u_epi.shape != u_ale.shape:
            raise ValueError(
                f"不确定性形状不匹配: epistemic={u_epi.shape}, "
                f"aleatoric={u_ale.shape}"
            )

        if self.strategy == "add":
            w_aleatoric = self._compute_aleatoric_weight(u_ale)
            w_epistemic = self._compute_epistemic_weight(u_epi)
            weights = self._combine_add_normalize(w_aleatoric, w_epistemic)
        else:
            weights = compute_gddn_gate(
                u_epi,
                u_ale,
                strategy=self.strategy,
                gamma=self.gamma,
                beta=self.beta,
                tau=self.tau,
                alpha=self.alpha,
                eps=self.epsilon,
            )

        # 最终裁剪确保 [0, 1]
        weights = torch.clamp(weights, min=0.0, max=1.0)

        return weights

    def _compute_aleatoric_weight(self, u_aleatoric: torch.Tensor) -> torch.Tensor:
        """
        基于数据不确定性计算权重

        公式: w_ale(x,y) = 1 / (1 + γ·U_aleatoric(x,y))

        理论：数据不确定性反映固有噪声，逆方差加权降低噪声像素的影响。
        """
        # 数值稳定性：确保非负
        u_ale_safe = torch.clamp(u_aleatoric, min=0.0)

        # 逆方差加权
        w_ale = 1.0 / (1.0 + self.gamma * u_ale_safe + self.epsilon)

        return w_ale

    def _compute_epistemic_weight(self, u_epistemic: torch.Tensor) -> torch.Tensor:
        """
        基于认知不确定性计算权重

        公式: w_epi(x,y) = σ(-β(U_epistemic(x,y) - τ))

        理论：认知不确定性反映模型知识缺乏，使用sigmoid配合阈值进行软门控。
        当 U_epi < τ 时权重接近1；当 U_epi > τ 时权重快速下降到0。
        """
        # 数值稳定性：确保非负
        u_epi_safe = torch.clamp(u_epistemic, min=0.0)

        # Sigmoid门控
        z = -self.beta * (u_epi_safe - self.tau)
        w_epi = torch.sigmoid(z)

        return w_epi

    def _combine_add_normalize(
        self,
        w_aleatoric: torch.Tensor,
        w_epistemic: torch.Tensor
    ) -> torch.Tensor:
        """
        组合策略b：相加后归一化

        公式: w = (w_aleatoric + w_epistemic) / 2

        理论：两种不确定性的平均影响（折中策略）。
        """
        w_sum = w_aleatoric + w_epistemic
        # 归一化到 [0, 1]，除以2因为两项相加最大为2
        w_normalized = w_sum / 2.0
        return w_normalized

    def _squeeze_to_2d_or_3d(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        将输入统一为2D [H, W] 或 3D [B, H, W] 形状

        支持输入形状：
        - [H, W] → [H, W]
        - [B, H, W] → [B, H, W]
        - [B, 1, H, W] → [B, H, W]
        - [1, H, W] → [H, W]
        """
        if tensor.dim() == 2:
            # [H, W] → 保持
            return tensor
        elif tensor.dim() == 3:
            # [B, H, W] → 保持
            # [1, H, W] → [H, W]
            if tensor.shape[0] == 1:
                return tensor.squeeze(0)
            return tensor
        elif tensor.dim() == 4 and tensor.shape[1] == 1:
            # [B, 1, H, W] → [B, H, W]
            return tensor.squeeze(1)
        else:
            raise ValueError(
                f"不支持的不确定性张量形状: {tensor.shape}. "
                f"期望: [H,W], [B,H,W], 或 [B,1,H,W]"
            )

class RobustHeteroscedasticLoss(nn.Module):
    """鲁棒异方差NLL损失，用于BayesGS-Diff模块II。"""

    def __init__(self, epsilon: float = 1e-6, huber_delta: float = 0.1) -> None:
        super().__init__()
        self.epsilon = float(max(epsilon, 1e-9))
        self.huber = nn.HuberLoss(reduction="none", delta=float(huber_delta))

    def forward(
        self,
        pred_rgb: torch.Tensor,
        target_rgb: torch.Tensor,
        aleatoric_var: torch.Tensor,
        epistemic_var: torch.Tensor,
        ood_threshold: float = 1.0,
    ) -> torch.Tensor:
        """计算鲁棒NLL损失。"""

        def _ensure_batch(tensor: torch.Tensor, channels: int = 3) -> torch.Tensor:
            if tensor.dim() == 3:
                return tensor.unsqueeze(0)
            if tensor.dim() == 4:
                return tensor
            if tensor.dim() == 2 and channels == 1:
                return tensor.unsqueeze(0).unsqueeze(0)
            raise ValueError(f"Unsupported tensor shape: {tensor.shape}")

        pred = _ensure_batch(pred_rgb)
        target = _ensure_batch(target_rgb)
        aleatoric = _ensure_batch(aleatoric_var, channels=1)
        epistemic = _ensure_batch(epistemic_var, channels=1)

        target = target.to(device=pred.device, dtype=pred.dtype)
        aleatoric = aleatoric.to(device=pred.device, dtype=pred.dtype)
        epistemic = epistemic.to(device=pred.device, dtype=pred.dtype)

        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")

        safe_var = torch.clamp(aleatoric, min=self.epsilon)
        valid_mask = (epistemic < float(ood_threshold)).float()
        valid_pixels = valid_mask.sum().clamp_min(1.0)

        huber = self.huber(pred, target).mean(dim=1, keepdim=True)
        data_term = huber / (2.0 * safe_var)
        reg_term = 0.5 * torch.log1p(safe_var)
        nll = (data_term + reg_term) * valid_mask
        return nll.sum() / valid_pixels


__all__ = ["SoftGatedGuidance", "RobustHeteroscedasticLoss"]
