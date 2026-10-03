"""
MC-Dropout based uncertainty estimator for candidate viewpoints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

from .configs.active_sampling_config import ActiveSamplingConfig
from .utils import ViewCandidate

try:  # pragma: no cover - optional dependency
    from gsplat.rendering import rasterization as default_rasterization  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    default_rasterization = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DepthAnythingV2 = None  # type: ignore

ConditionDict = Dict[str, torch.Tensor]
RasterizeFn = Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


DEPTH_ANYTHING_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


class _DepthAnythingAdapter:
    def __init__(self, encoder: str, checkpoint: str, device: str) -> None:
        if DepthAnythingV2 is None:
            raise RuntimeError("depth_anything_v2 未安装，无法启用DACD。")
        encoder_key = encoder.lower()
        if encoder_key not in DEPTH_ANYTHING_CONFIGS:
            raise ValueError(f"不支持的Depth Anything编码器: {encoder}")
        if not checkpoint:
            raise ValueError("需要提供Depth Anything权重路径以初始化模型。")
        config = DEPTH_ANYTHING_CONFIGS[encoder_key]
        model = DepthAnythingV2(**config)
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state)
        
        # 禁用xformers，使用PyTorch原生的scaled_dot_product_attention
        import os
        os.environ["XFORMERS_DISABLED"] = "1"  # 禁用xformers
        os.environ["XFORMERS_FORCE_DISABLE_TRITON"] = "1"  # 禁用triton
        
        # 使用FP32精度
        self.model = model.to(device).float().eval()
        self.device = torch.device(device)
        print(f"[DepthAnything] 模型加载完成，设备: {device}, 精度: FP32, xformers: 已禁用")

    @torch.no_grad()
    def infer_batch(self, rgb_batch: torch.Tensor) -> List[torch.Tensor]:
        predictions: List[torch.Tensor] = []
        for image in rgb_batch:
            predictions.append(self._infer_single(image))
        return predictions

    @torch.no_grad()
    def _infer_single(self, image: torch.Tensor) -> torch.Tensor:
        cpu_image = (
            image.detach()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        bgr_image = cpu_image[..., ::-1].copy()
        
        # 使用纯FP32推理（禁用FP16 autocast）
        depth = self.model.infer_image(bgr_image)
        
        depth_tensor = torch.from_numpy(depth).float()
        return depth_tensor.to(image.device)


@dataclass
class SparseViewBundle:
    """Pack sparse views used to condition the generator."""

    images: torch.Tensor  # [K, 3, H, W]
    poses: torch.Tensor  # [K, 4, 4]


@dataclass
class UncertaintyResult:
    """MC-Dropout statistics for a candidate view."""

    candidate: ViewCandidate
    mean_rgb: torch.Tensor  # [3, H, W]
    variance_map: torch.Tensor  # [H, W]
    score: float  # masked score when可见区域存在，否则为全图均值
    unmasked_score: float = 0.0  # 原始全图均值，便于对比
    metrics: Optional[Dict[str, float]] = None
    epistemic_map: Optional[torch.Tensor] = None
    aleatoric_map: Optional[torch.Tensor] = None
    risk_exist_map: Optional[torch.Tensor] = None
    gain_novel_map: Optional[torch.Tensor] = None
    exist_mask: Optional[torch.Tensor] = None
    novel_mask: Optional[torch.Tensor] = None
    calibrated_depth: Optional[torch.Tensor] = None
    mono_depth: Optional[torch.Tensor] = None


Q_GAIN_POSITIVE_FEATURE_NAMES: Tuple[str, ...] = (
    "gain_novel_mean",
    "gain_novel_mass",
    "risk_exist_mean",
    "supported_exist_coverage",
    "unsupported_render_coverage",
    "health_mean",
    "health_std",
)

Q_GAIN_RISK_FEATURE_NAMES: Tuple[str, ...] = (
    "risk_exist_mean",
    "risk_exist_mass",
    "supported_exist_coverage",
    "exist_l1_render_only",
    "warp_vs_render_l1_on_supported_exist",
    "gain_novel_mean",
    "health_mean",
    "health_std",
)

Q_GAIN_GATE_FEATURE_NAMES: Tuple[str, ...] = (
    "risk_exist_mean",
    "risk_exist_mass",
    "gain_novel_mean",
    "supported_exist_coverage",
    "unsupported_render_coverage",
    "exist_l1_render_only",
    "warp_vs_render_l1_on_supported_exist",
    "health_mean",
    "health_std",
)

Q_GAIN_RANKING_FEATURE_NAMES: Tuple[str, ...] = Q_GAIN_POSITIVE_FEATURE_NAMES
Q_GAIN_FEATURE_NAMES: Tuple[str, ...] = Q_GAIN_POSITIVE_FEATURE_NAMES


def _to_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = float(value.item())
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(float_value):
        return None
    return float_value


def flatten_q_gain_record(record: Dict[str, Any]) -> Dict[str, Optional[float]]:
    flat_record: Dict[str, Optional[float]] = {}
    metrics = record.get("metrics")
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            flat_record[key] = _to_optional_float(value)
    for key, value in record.items():
        if key == "metrics":
            continue
        if key == "health" and isinstance(value, dict):
            flat_record["health_mean"] = _to_optional_float(value.get("mean"))
            flat_record["health_std"] = _to_optional_float(value.get("std"))
            continue
        flat_record[key] = _to_optional_float(value)
    return flat_record


def _stable_sigmoid(logit_values: np.ndarray) -> np.ndarray:
    clipped_logits = np.clip(logit_values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped_logits))


def _fit_regularized_logistic_classifier(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_lambda: float,
    max_iterations: int = 64,
    tolerance: float = 1e-6,
) -> Tuple[np.ndarray, float]:
    if feature_matrix.ndim != 2 or labels.ndim != 1:
        raise ValueError("logistic classifier 输入维度不合法")

    num_samples, num_features = feature_matrix.shape
    positive_rate = float(np.mean(labels > 0.5)) if num_samples > 0 else 0.5
    positive_rate = min(max(positive_rate, 1e-4), 1.0 - 1e-4)
    initial_intercept = float(math.log(positive_rate / (1.0 - positive_rate)))
    weights = np.zeros(num_features, dtype=np.float64)
    intercept = initial_intercept

    has_positive = bool(np.any(labels > 0.5))
    has_negative = bool(np.any(labels <= 0.5))
    if not has_positive or not has_negative or num_samples == 0 or num_features == 0:
        return weights, intercept

    positive_mask = labels > 0.5
    negative_mask = ~positive_mask
    sample_weights = np.ones(num_samples, dtype=np.float64)
    if np.any(positive_mask) and np.any(negative_mask):
        sample_weights[positive_mask] = 0.5 / max(float(np.sum(positive_mask)), 1.0)
        sample_weights[negative_mask] = 0.5 / max(float(np.sum(negative_mask)), 1.0)
        sample_weights = sample_weights * float(num_samples)

    regularizer = float(max(ridge_lambda, 1e-8))
    identity_matrix = np.eye(num_features, dtype=np.float64)

    for _ in range(max_iterations):
        logits = feature_matrix @ weights + intercept
        probabilities = _stable_sigmoid(logits)
        probability_error = probabilities - labels
        weighted_error = sample_weights * probability_error

        gradient_weights = feature_matrix.T @ weighted_error + regularizer * weights
        gradient_intercept = float(np.sum(weighted_error))

        curvature = sample_weights * probabilities * (1.0 - probabilities)
        hessian_weights = feature_matrix.T @ (curvature[:, None] * feature_matrix)
        hessian_weights = hessian_weights + regularizer * identity_matrix
        hessian_cross = feature_matrix.T @ curvature
        hessian_intercept = float(np.sum(curvature) + 1e-8)

        full_hessian = np.zeros((num_features + 1, num_features + 1), dtype=np.float64)
        full_hessian[:num_features, :num_features] = hessian_weights
        full_hessian[:num_features, -1] = hessian_cross
        full_hessian[-1, :num_features] = hessian_cross
        full_hessian[-1, -1] = hessian_intercept
        full_gradient = np.concatenate(
            [gradient_weights, np.asarray([gradient_intercept], dtype=np.float64)],
            axis=0,
        )

        try:
            update_step = np.linalg.solve(full_hessian, full_gradient)
        except np.linalg.LinAlgError:
            update_step = np.linalg.pinv(full_hessian) @ full_gradient

        weights = weights - update_step[:num_features]
        intercept = float(intercept - update_step[-1])
        if float(np.linalg.norm(update_step)) <= tolerance:
            break

    return weights, intercept


def _prepare_standardized_feature_values(
    *,
    flat_record: Dict[str, Optional[float]],
    feature_names: Sequence[str],
    feature_means: Sequence[float],
    feature_stds: Sequence[float],
    feature_clip_values: Sequence[float],
) -> Tuple[List[float], int]:
    standardized_values: List[float] = []
    observed_feature_count = 0
    for feature_name, feature_mean, feature_std, feature_clip_value in zip(
        feature_names,
        feature_means,
        feature_stds,
        feature_clip_values,
    ):
        feature_value = _to_optional_float(flat_record.get(feature_name))
        if feature_value is None:
            feature_value = float(feature_mean)
        else:
            observed_feature_count += 1
        standardized_value = (feature_value - float(feature_mean)) / max(float(feature_std), 1e-6)
        if math.isfinite(float(feature_clip_value)):
            standardized_value = max(
                min(standardized_value, float(feature_clip_value)),
                -float(feature_clip_value),
            )
        standardized_values.append(float(standardized_value))
    return standardized_values, observed_feature_count


def _fit_feature_statistics(
    feature_rows: Sequence[Sequence[float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_matrix = np.asarray(feature_rows, dtype=np.float64)
    if feature_matrix.ndim != 2:
        raise ValueError("feature_rows 必须是二维矩阵")
    feature_means = np.zeros(feature_matrix.shape[1], dtype=np.float64)
    for feature_index in range(feature_matrix.shape[1]):
        finite_mask = np.isfinite(feature_matrix[:, feature_index])
        if np.any(finite_mask):
            feature_means[feature_index] = float(np.mean(feature_matrix[finite_mask, feature_index]))
    feature_matrix = np.where(np.isfinite(feature_matrix), feature_matrix, feature_means)
    feature_stds = np.std(feature_matrix, axis=0)
    feature_stds = np.where(feature_stds > 1e-6, feature_stds, 1.0)
    standardized_features = (feature_matrix - feature_means) / feature_stds
    feature_clip_values = np.quantile(np.abs(standardized_features), 0.95, axis=0)
    feature_clip_values = np.maximum(feature_clip_values, 1.0)
    return feature_matrix, feature_means, feature_stds, feature_clip_values


@dataclass
class QGainCalibrator:
    ranking_feature_names: Tuple[str, ...]
    ranking_feature_means: Tuple[float, ...]
    ranking_feature_stds: Tuple[float, ...]
    ranking_feature_clip_values: Tuple[float, ...]
    ranking_coefficients: Tuple[float, ...]
    ranking_intercept: float
    ranking_gain_scale: float
    ranking_gain_clip: float
    risk_feature_names: Tuple[str, ...]
    risk_feature_means: Tuple[float, ...]
    risk_feature_stds: Tuple[float, ...]
    risk_feature_clip_values: Tuple[float, ...]
    risk_coefficients: Tuple[float, ...]
    risk_intercept: float
    risk_scale: float
    risk_clip: float
    gate_feature_names: Tuple[str, ...]
    gate_feature_means: Tuple[float, ...]
    gate_feature_stds: Tuple[float, ...]
    gate_feature_clip_values: Tuple[float, ...]
    gate_coefficients: Tuple[float, ...]
    gate_intercept: float
    baseline_u_geo_slope: float
    baseline_u_geo_intercept: float
    residual_scale: float
    residual_clip: float
    train_sample_count: int
    ridge_lambda: float
    format_version: int = 5

    @classmethod
    def fit_from_records(
        cls,
        records: Sequence[Dict[str, Any]],
        *,
        ranking_feature_names: Sequence[str] = Q_GAIN_POSITIVE_FEATURE_NAMES,
        risk_feature_names: Sequence[str] = Q_GAIN_RISK_FEATURE_NAMES,
        gate_feature_names: Sequence[str] = Q_GAIN_GATE_FEATURE_NAMES,
        ridge_lambda: float = 1e-3,
    ) -> Optional["QGainCalibrator"]:
        selected_ranking_feature_names = tuple(ranking_feature_names)
        selected_risk_feature_names = tuple(risk_feature_names)
        selected_gate_feature_names = tuple(gate_feature_names)
        if len(selected_ranking_feature_names) == 0:
            return None

        ranking_feature_rows: List[List[float]] = []
        risk_feature_rows: List[List[float]] = []
        gate_feature_rows: List[List[float]] = []
        u_geo_values: List[float] = []
        delta_values: List[float] = []

        for record in records:
            flat_record = flatten_q_gain_record(record)
            u_geo_value = _to_optional_float(flat_record.get("u_geo"))
            delta_value = _to_optional_float(
                flat_record.get("leave_one_holdout_delta_psnr", flat_record.get("delta_psnr"))
            )
            if u_geo_value is None or delta_value is None:
                continue
            ranking_feature_rows.append(
                [
                    np.nan
                    if (feature_value := _to_optional_float(flat_record.get(feature_name))) is None
                    else feature_value
                    for feature_name in selected_ranking_feature_names
                ]
            )
            gate_feature_rows.append(
                [
                    np.nan
                    if (feature_value := _to_optional_float(flat_record.get(feature_name))) is None
                    else feature_value
                    for feature_name in selected_gate_feature_names
                ]
            )
            risk_feature_rows.append(
                [
                    np.nan
                    if (feature_value := _to_optional_float(flat_record.get(feature_name))) is None
                    else feature_value
                    for feature_name in selected_risk_feature_names
                ]
            )
            u_geo_values.append(u_geo_value)
            delta_values.append(delta_value)

        if len(delta_values) < max(4, len(selected_ranking_feature_names) // 2):
            return None

        u_geo_array = np.asarray(u_geo_values, dtype=np.float64)
        delta_array = np.asarray(delta_values, dtype=np.float64)

        (
            ranking_feature_matrix,
            ranking_feature_means,
            ranking_feature_stds,
            ranking_feature_clip_values,
        ) = _fit_feature_statistics(ranking_feature_rows)
        (
            gate_feature_matrix,
            gate_feature_means,
            gate_feature_stds,
            gate_feature_clip_values,
        ) = _fit_feature_statistics(gate_feature_rows)
        (
            risk_feature_matrix,
            risk_feature_means,
            risk_feature_stds,
            risk_feature_clip_values,
        ) = _fit_feature_statistics(risk_feature_rows)

        baseline_design = np.stack([u_geo_array, np.ones_like(u_geo_array)], axis=1)
        try:
            baseline_solution, *_ = np.linalg.lstsq(baseline_design, delta_array, rcond=None)
        except np.linalg.LinAlgError:
            return None
        baseline_u_geo_slope = float(baseline_solution[0])
        baseline_u_geo_intercept = float(baseline_solution[1])
        baseline_delta = baseline_u_geo_slope * u_geo_array + baseline_u_geo_intercept
        residual_delta = delta_array - baseline_delta

        ranking_standardized_features = (ranking_feature_matrix - ranking_feature_means) / ranking_feature_stds
        positive_gain_targets = np.maximum(residual_delta, 0.0)
        ranking_intercept = float(np.mean(positive_gain_targets))
        centered_positive_gain = positive_gain_targets - ranking_intercept
        regularizer = float(max(ridge_lambda, 1e-8))
        ridge_matrix = ranking_standardized_features.T @ ranking_standardized_features
        ridge_matrix = ridge_matrix + np.eye(ridge_matrix.shape[0], dtype=np.float64) * regularizer
        ridge_rhs = ranking_standardized_features.T @ centered_positive_gain
        try:
            ranking_coefficients = np.linalg.solve(ridge_matrix, ridge_rhs)
        except np.linalg.LinAlgError:
            ranking_coefficients = np.linalg.pinv(ridge_matrix) @ ridge_rhs
        ranking_gain_scale = float(np.std(positive_gain_targets))
        if ranking_gain_scale < 1e-6:
            ranking_gain_scale = max(float(np.mean(positive_gain_targets)), 1.0)
        ranking_gain_clip = float(np.max(positive_gain_targets))
        ranking_gain_clip = max(ranking_gain_clip, ranking_gain_scale, 1e-6)

        gate_standardized_features = (gate_feature_matrix - gate_feature_means) / gate_feature_stds
        benefit_labels = (residual_delta > 0.0).astype(np.float64)
        gate_coefficients, gate_intercept = _fit_regularized_logistic_classifier(
            gate_standardized_features,
            benefit_labels,
            ridge_lambda=regularizer,
        )

        risk_standardized_features = (risk_feature_matrix - risk_feature_means) / risk_feature_stds
        harmful_risk_targets = np.maximum(-residual_delta, 0.0)
        risk_intercept = float(np.mean(harmful_risk_targets))
        centered_harmful_risk = harmful_risk_targets - risk_intercept
        risk_ridge_matrix = risk_standardized_features.T @ risk_standardized_features
        risk_ridge_matrix = risk_ridge_matrix + np.eye(risk_ridge_matrix.shape[0], dtype=np.float64) * regularizer
        risk_ridge_rhs = risk_standardized_features.T @ centered_harmful_risk
        try:
            risk_coefficients = np.linalg.solve(risk_ridge_matrix, risk_ridge_rhs)
        except np.linalg.LinAlgError:
            risk_coefficients = np.linalg.pinv(risk_ridge_matrix) @ risk_ridge_rhs
        risk_scale = float(np.std(harmful_risk_targets))
        if risk_scale < 1e-6:
            risk_scale = max(float(np.mean(harmful_risk_targets)), 1.0)
        risk_clip = float(np.max(harmful_risk_targets))
        risk_clip = max(risk_clip, risk_scale, 1e-6)

        residual_scale = float(np.std(residual_delta))
        if residual_scale < 1e-6:
            residual_scale = 1.0
        residual_clip = float(np.max(np.abs(residual_delta)))
        residual_clip = max(residual_clip, residual_scale, 1e-6)

        return cls(
            ranking_feature_names=selected_ranking_feature_names,
            ranking_feature_means=tuple(float(value) for value in ranking_feature_means.tolist()),
            ranking_feature_stds=tuple(float(value) for value in ranking_feature_stds.tolist()),
            ranking_feature_clip_values=tuple(
                float(value) for value in ranking_feature_clip_values.tolist()
            ),
            ranking_coefficients=tuple(float(value) for value in ranking_coefficients.tolist()),
            ranking_intercept=ranking_intercept,
            ranking_gain_scale=float(ranking_gain_scale),
            ranking_gain_clip=float(ranking_gain_clip),
            risk_feature_names=selected_risk_feature_names,
            risk_feature_means=tuple(float(value) for value in risk_feature_means.tolist()),
            risk_feature_stds=tuple(float(value) for value in risk_feature_stds.tolist()),
            risk_feature_clip_values=tuple(float(value) for value in risk_feature_clip_values.tolist()),
            risk_coefficients=tuple(float(value) for value in risk_coefficients.tolist()),
            risk_intercept=float(risk_intercept),
            risk_scale=float(risk_scale),
            risk_clip=float(risk_clip),
            gate_feature_names=selected_gate_feature_names,
            gate_feature_means=tuple(float(value) for value in gate_feature_means.tolist()),
            gate_feature_stds=tuple(float(value) for value in gate_feature_stds.tolist()),
            gate_feature_clip_values=tuple(float(value) for value in gate_feature_clip_values.tolist()),
            gate_coefficients=tuple(float(value) for value in gate_coefficients.tolist()),
            gate_intercept=float(gate_intercept),
            baseline_u_geo_slope=baseline_u_geo_slope,
            baseline_u_geo_intercept=baseline_u_geo_intercept,
            residual_scale=residual_scale,
            residual_clip=residual_clip,
            train_sample_count=len(delta_values),
            ridge_lambda=regularizer,
        )

    def predict_from_record(self, record: Dict[str, Any]) -> Dict[str, Optional[float]]:
        if self.format_version < 5:
            return self._predict_from_legacy_v4_record(record)

        flat_record = flatten_q_gain_record(record)
        ranking_clip_values = self.ranking_feature_clip_values
        if len(ranking_clip_values) != len(self.ranking_feature_names):
            ranking_clip_values = tuple(float("inf") for _ in self.ranking_feature_names)
        standardized_ranking_values, observed_ranking_feature_count = _prepare_standardized_feature_values(
            flat_record=flat_record,
            feature_names=self.ranking_feature_names,
            feature_means=self.ranking_feature_means,
            feature_stds=self.ranking_feature_stds,
            feature_clip_values=ranking_clip_values,
        )
        raw_positive_gain_prediction = self.ranking_intercept + float(
            np.dot(
                np.asarray(standardized_ranking_values, dtype=np.float64),
                np.asarray(self.ranking_coefficients, dtype=np.float64),
            )
        )
        ranking_feature_coverage = (
            1.0
            if len(self.ranking_feature_names) == 0
            else float(observed_ranking_feature_count) / float(len(self.ranking_feature_names))
        )
        coverage_adjusted_positive_gain = max(raw_positive_gain_prediction, 0.0) * ranking_feature_coverage
        ranking_gain_clip = max(float(self.ranking_gain_clip), 1e-6)
        positive_gain_prediction = ranking_gain_clip * math.tanh(
            coverage_adjusted_positive_gain / ranking_gain_clip
        )

        gate_feature_coverage = None
        observed_gate_feature_count = None
        q_gain_for_gate: Optional[float] = None
        gate_expected_residual_prediction: Optional[float] = None
        if len(self.gate_feature_names) > 0 and len(self.gate_coefficients) == len(self.gate_feature_names):
            gate_clip_values = self.gate_feature_clip_values
            if len(gate_clip_values) != len(self.gate_feature_names):
                gate_clip_values = tuple(float("inf") for _ in self.gate_feature_names)
            standardized_gate_values, observed_gate_feature_count = _prepare_standardized_feature_values(
                flat_record=flat_record,
                feature_names=self.gate_feature_names,
                feature_means=self.gate_feature_means,
                feature_stds=self.gate_feature_stds,
                feature_clip_values=gate_clip_values,
            )
            gate_logits = self.gate_intercept + float(
                np.dot(
                    np.asarray(standardized_gate_values, dtype=np.float64),
                    np.asarray(self.gate_coefficients, dtype=np.float64),
                )
            )
            q_gain_for_gate = float(_stable_sigmoid(np.asarray([gate_logits], dtype=np.float64))[0])
            gate_feature_coverage = (
                1.0
                if len(self.gate_feature_names) == 0
                else float(observed_gate_feature_count) / float(len(self.gate_feature_names))
            )
        else:
            residual_scale = max(self.residual_scale, 1e-6)
            scaled_gate_residual = positive_gain_prediction / residual_scale
            scaled_gate_residual = max(min(float(scaled_gate_residual), 60.0), -60.0)
            q_gain_for_gate = float(1.0 / (1.0 + math.exp(-scaled_gate_residual)))
            gate_feature_coverage = ranking_feature_coverage
            observed_gate_feature_count = observed_ranking_feature_count

        risk_clip_values = self.risk_feature_clip_values
        if len(risk_clip_values) != len(self.risk_feature_names):
            risk_clip_values = tuple(float("inf") for _ in self.risk_feature_names)
        standardized_risk_values, observed_risk_feature_count = _prepare_standardized_feature_values(
            flat_record=flat_record,
            feature_names=self.risk_feature_names,
            feature_means=self.risk_feature_means,
            feature_stds=self.risk_feature_stds,
            feature_clip_values=risk_clip_values,
        )
        raw_harmful_risk_prediction = self.risk_intercept + float(
            np.dot(
                np.asarray(standardized_risk_values, dtype=np.float64),
                np.asarray(self.risk_coefficients, dtype=np.float64),
            )
        )
        risk_feature_coverage = (
            1.0
            if len(self.risk_feature_names) == 0
            else float(observed_risk_feature_count) / float(len(self.risk_feature_names))
        )
        coverage_adjusted_harmful_risk = max(raw_harmful_risk_prediction, 0.0) * risk_feature_coverage
        risk_clip = max(float(self.risk_clip), 1e-6)
        harmful_risk_prediction = risk_clip * math.tanh(
            coverage_adjusted_harmful_risk / risk_clip
        )

        expected_residual_prediction = float(
            float(q_gain_for_gate) * positive_gain_prediction
            - (1.0 - float(q_gain_for_gate)) * harmful_risk_prediction
        )
        gate_expected_residual_prediction = expected_residual_prediction

        u_geo_value = _to_optional_float(flat_record.get("u_geo"))
        baseline_delta_prediction = (
            None
            if u_geo_value is None
            else self.baseline_u_geo_slope * u_geo_value + self.baseline_u_geo_intercept
        )
        predicted_delta_psnr = (
            None
            if baseline_delta_prediction is None
            else float(baseline_delta_prediction + expected_residual_prediction)
        )
        ranking_gain_scale = max(self.ranking_gain_scale, 1e-6)
        q_gain = float(
            positive_gain_prediction / (positive_gain_prediction + ranking_gain_scale)
        )
        return {
            "q_gain": float(q_gain),
            "q_gain_for_gate": None if q_gain_for_gate is None else float(q_gain_for_gate),
            "predicted_residual_delta_psnr": float(expected_residual_prediction),
            "gate_predicted_residual_delta_psnr": (
                None
                if gate_expected_residual_prediction is None
                else float(gate_expected_residual_prediction)
            ),
            "raw_predicted_residual_delta_psnr": float(
                float(q_gain_for_gate) * max(raw_positive_gain_prediction, 0.0)
                - (1.0 - float(q_gain_for_gate)) * max(raw_harmful_risk_prediction, 0.0)
            ),
            "predicted_positive_gain_delta_psnr": float(positive_gain_prediction),
            "predicted_harmful_risk_delta_psnr": float(harmful_risk_prediction),
            "predicted_delta_psnr": predicted_delta_psnr,
            "baseline_delta_psnr_from_u_geo": None
            if baseline_delta_prediction is None
            else float(baseline_delta_prediction),
            "feature_coverage": float(ranking_feature_coverage),
            "missing_feature_count": int(
                max(len(self.ranking_feature_names) - observed_ranking_feature_count, 0)
            ),
            "gate_feature_coverage": None
            if gate_feature_coverage is None
            else float(gate_feature_coverage),
            "gate_missing_feature_count": None
            if observed_gate_feature_count is None
            else int(max(len(self.gate_feature_names) - observed_gate_feature_count, 0)),
            "risk_feature_coverage": float(risk_feature_coverage),
            "risk_missing_feature_count": int(max(len(self.risk_feature_names) - observed_risk_feature_count, 0)),
            "residual_clip": float(self.residual_clip),
        }

    def _predict_from_legacy_v4_record(self, record: Dict[str, Any]) -> Dict[str, Optional[float]]:
        flat_record = flatten_q_gain_record(record)
        ranking_clip_values = self.ranking_feature_clip_values
        if len(ranking_clip_values) != len(self.ranking_feature_names):
            ranking_clip_values = tuple(float("inf") for _ in self.ranking_feature_names)
        standardized_ranking_values, observed_ranking_feature_count = _prepare_standardized_feature_values(
            flat_record=flat_record,
            feature_names=self.ranking_feature_names,
            feature_means=self.ranking_feature_means,
            feature_stds=self.ranking_feature_stds,
            feature_clip_values=ranking_clip_values,
        )
        raw_residual_prediction = self.ranking_intercept + float(
            np.dot(
                np.asarray(standardized_ranking_values, dtype=np.float64),
                np.asarray(self.ranking_coefficients, dtype=np.float64),
            )
        )
        ranking_feature_coverage = (
            1.0
            if len(self.ranking_feature_names) == 0
            else float(observed_ranking_feature_count) / float(len(self.ranking_feature_names))
        )
        coverage_adjusted_residual = raw_residual_prediction * ranking_feature_coverage
        ranking_residual_clip = max(float(self.residual_clip), 1e-6)
        ranking_residual_prediction = ranking_residual_clip * math.tanh(
            coverage_adjusted_residual / ranking_residual_clip
        )

        gate_feature_coverage = ranking_feature_coverage
        observed_gate_feature_count = observed_ranking_feature_count
        if len(self.gate_feature_names) > 0 and len(self.gate_coefficients) == len(self.gate_feature_names):
            gate_clip_values = self.gate_feature_clip_values
            if len(gate_clip_values) != len(self.gate_feature_names):
                gate_clip_values = tuple(float("inf") for _ in self.gate_feature_names)
            standardized_gate_values, observed_gate_feature_count = _prepare_standardized_feature_values(
                flat_record=flat_record,
                feature_names=self.gate_feature_names,
                feature_means=self.gate_feature_means,
                feature_stds=self.gate_feature_stds,
                feature_clip_values=gate_clip_values,
            )
            gate_logits = self.gate_intercept + float(
                np.dot(
                    np.asarray(standardized_gate_values, dtype=np.float64),
                    np.asarray(self.gate_coefficients, dtype=np.float64),
                )
            )
            q_gain_for_gate = float(_stable_sigmoid(np.asarray([gate_logits], dtype=np.float64))[0])
            gate_feature_coverage = (
                1.0
                if len(self.gate_feature_names) == 0
                else float(observed_gate_feature_count) / float(len(self.gate_feature_names))
            )
            gate_expected_residual_prediction = float(
                ((2.0 * q_gain_for_gate) - 1.0) * abs(ranking_residual_prediction)
            )
        else:
            residual_scale = max(self.residual_scale, 1e-6)
            scaled_gate_residual = ranking_residual_prediction / residual_scale
            scaled_gate_residual = max(min(float(scaled_gate_residual), 60.0), -60.0)
            q_gain_for_gate = float(1.0 / (1.0 + math.exp(-scaled_gate_residual)))
            gate_expected_residual_prediction = float(ranking_residual_prediction)

        u_geo_value = _to_optional_float(flat_record.get("u_geo"))
        baseline_delta_prediction = (
            None
            if u_geo_value is None
            else self.baseline_u_geo_slope * u_geo_value + self.baseline_u_geo_intercept
        )
        predicted_delta_psnr = (
            None
            if baseline_delta_prediction is None
            else float(baseline_delta_prediction + ranking_residual_prediction)
        )
        residual_scale = max(self.residual_scale, 1e-6)
        scaled_ranking_residual = ranking_residual_prediction / residual_scale
        scaled_ranking_residual = max(min(float(scaled_ranking_residual), 60.0), -60.0)
        q_gain = 1.0 / (1.0 + math.exp(-scaled_ranking_residual))
        return {
            "q_gain": float(q_gain),
            "q_gain_for_gate": float(q_gain_for_gate),
            "predicted_residual_delta_psnr": float(ranking_residual_prediction),
            "gate_predicted_residual_delta_psnr": float(gate_expected_residual_prediction),
            "raw_predicted_residual_delta_psnr": float(raw_residual_prediction),
            "predicted_delta_psnr": predicted_delta_psnr,
            "baseline_delta_psnr_from_u_geo": None
            if baseline_delta_prediction is None
            else float(baseline_delta_prediction),
            "feature_coverage": float(ranking_feature_coverage),
            "missing_feature_count": int(
                max(len(self.ranking_feature_names) - observed_ranking_feature_count, 0)
            ),
            "gate_feature_coverage": float(gate_feature_coverage),
            "gate_missing_feature_count": int(
                max(len(self.gate_feature_names) - observed_gate_feature_count, 0)
            ),
            "residual_clip": float(ranking_residual_clip),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": 5,
            "ranking_feature_names": list(self.ranking_feature_names),
            "ranking_feature_means": list(self.ranking_feature_means),
            "ranking_feature_stds": list(self.ranking_feature_stds),
            "ranking_feature_clip_values": list(self.ranking_feature_clip_values),
            "ranking_coefficients": list(self.ranking_coefficients),
            "ranking_intercept": float(self.ranking_intercept),
            "ranking_gain_scale": float(self.ranking_gain_scale),
            "ranking_gain_clip": float(self.ranking_gain_clip),
            "ranking_target_mode": "positive_gain_magnitude",
            "risk_feature_names": list(self.risk_feature_names),
            "risk_feature_means": list(self.risk_feature_means),
            "risk_feature_stds": list(self.risk_feature_stds),
            "risk_feature_clip_values": list(self.risk_feature_clip_values),
            "risk_coefficients": list(self.risk_coefficients),
            "risk_intercept": float(self.risk_intercept),
            "risk_scale": float(self.risk_scale),
            "risk_clip": float(self.risk_clip),
            "risk_target_mode": "harmful_risk_magnitude",
            "gate_feature_names": list(self.gate_feature_names),
            "gate_feature_means": list(self.gate_feature_means),
            "gate_feature_stds": list(self.gate_feature_stds),
            "gate_feature_clip_values": list(self.gate_feature_clip_values),
            "gate_coefficients": list(self.gate_coefficients),
            "gate_intercept": float(self.gate_intercept),
            "baseline_u_geo_slope": float(self.baseline_u_geo_slope),
            "baseline_u_geo_intercept": float(self.baseline_u_geo_intercept),
            "residual_scale": float(self.residual_scale),
            "residual_clip": float(self.residual_clip),
            "train_sample_count": int(self.train_sample_count),
            "ridge_lambda": float(self.ridge_lambda),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "QGainCalibrator":
        format_version = int(payload.get("version", 0))
        legacy_feature_names = payload.get("feature_names", [])
        legacy_feature_means = payload.get("feature_means", [])
        legacy_feature_stds = payload.get("feature_stds", [])
        legacy_feature_clip_values = payload.get(
            "feature_clip_values",
            [60.0] * len(legacy_feature_names),
        )
        legacy_coefficients = payload.get("coefficients", [])
        return cls(
            ranking_feature_names=tuple(
                str(value)
                for value in payload.get("ranking_feature_names", legacy_feature_names)
            ),
            ranking_feature_means=tuple(
                float(value)
                for value in payload.get("ranking_feature_means", legacy_feature_means)
            ),
            ranking_feature_stds=tuple(
                float(value)
                for value in payload.get("ranking_feature_stds", legacy_feature_stds)
            ),
            ranking_feature_clip_values=tuple(
                float(value)
                for value in payload.get("ranking_feature_clip_values", legacy_feature_clip_values)
            ),
            ranking_coefficients=tuple(
                float(value)
                for value in payload.get("ranking_coefficients", legacy_coefficients)
            ),
            ranking_intercept=float(
                payload.get("ranking_intercept", payload.get("residual_intercept", 0.0))
            ),
            ranking_gain_scale=float(
                payload.get("ranking_gain_scale", payload.get("residual_scale", 1.0))
            ),
            ranking_gain_clip=float(
                payload.get(
                    "ranking_gain_clip",
                    payload.get(
                        "residual_clip",
                        max(float(payload.get("residual_scale", 1.0)), 1e-6),
                    ),
                )
            ),
            risk_feature_names=tuple(
                str(value) for value in payload.get("risk_feature_names", [])
            ),
            risk_feature_means=tuple(
                float(value) for value in payload.get("risk_feature_means", [])
            ),
            risk_feature_stds=tuple(
                float(value) for value in payload.get("risk_feature_stds", [])
            ),
            risk_feature_clip_values=tuple(
                float(value) for value in payload.get("risk_feature_clip_values", [])
            ),
            risk_coefficients=tuple(
                float(value) for value in payload.get("risk_coefficients", [])
            ),
            risk_intercept=float(payload.get("risk_intercept", 0.0)),
            risk_scale=float(payload.get("risk_scale", payload.get("residual_scale", 1.0))),
            risk_clip=float(
                payload.get(
                    "risk_clip",
                    payload.get(
                        "residual_clip",
                        max(float(payload.get("residual_scale", 1.0)), 1e-6),
                    ),
                )
            ),
            gate_feature_names=tuple(
                str(value) for value in payload.get("gate_feature_names", [])
            ),
            gate_feature_means=tuple(
                float(value) for value in payload.get("gate_feature_means", [])
            ),
            gate_feature_stds=tuple(
                float(value) for value in payload.get("gate_feature_stds", [])
            ),
            gate_feature_clip_values=tuple(
                float(value) for value in payload.get("gate_feature_clip_values", [])
            ),
            gate_coefficients=tuple(
                float(value) for value in payload.get("gate_coefficients", [])
            ),
            gate_intercept=float(payload.get("gate_intercept", 0.0)),
            baseline_u_geo_slope=float(payload.get("baseline_u_geo_slope", 0.0)),
            baseline_u_geo_intercept=float(payload.get("baseline_u_geo_intercept", 0.0)),
            residual_scale=float(payload.get("residual_scale", 1.0)),
            residual_clip=float(
                payload.get(
                    "residual_clip",
                    max(float(payload.get("residual_scale", 1.0)), 1e-6),
                )
            ),
            train_sample_count=int(payload.get("train_sample_count", 0)),
            ridge_lambda=float(payload.get("ridge_lambda", 1e-3)),
            format_version=format_version,
        )

    @classmethod
    def load(cls, path: str) -> "QGainCalibrator":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)


ConditionProvider = Callable[[Sequence[ViewCandidate]], ConditionDict]


class UncertaintyEstimator:
    """
    Estimate epistemic uncertainty for candidate viewpoints using MC-Dropout.

    Args:
        model: Trained GDDN generator.
        config: Active sampling configuration.
        device: Torch device used for inference.
        condition_provider: Callable that supplies rendered conditioning
            tensors for a list of candidates. Must return a dictionary with
            keys {"rgb_render", "depth_render", "normal_render", "target_latent"}.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        config: ActiveSamplingConfig,
        device: torch.device,
        condition_provider: ConditionProvider,
        *,
        writer: Optional["SummaryWriter"] = None,
        mode: Optional[str] = None,
        gaussians: Optional[Any] = None,
        rasterizer: Optional[RasterizeFn] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.device = device
        self.condition_provider = condition_provider
        self.writer = writer
        self.mode = (mode or config.estimator_mode).lower()
        self.gaussians = gaussians
        self.rasterizer = rasterizer or default_rasterization
        if self.mode not in {"gs_dropout", "gddn"}:
            raise ValueError(f"Unsupported estimator mode: {self.mode}")
        if self.mode == "gddn" and self.model is None:
            raise ValueError("GDDN mode requires a generator model.")
        if self.mode == "gs_dropout" and self.rasterizer is None:
            raise ValueError(
                "GS-Dropout mode requires gsplat.rendering.rasterization to be available."
            )
        if self.mode == "gs_dropout" and self.gaussians is None:
            raise ValueError("GS-Dropout mode requires a Gaussian scene reference.")
        self._depth_adapter = self._build_depth_adapter()

    def _align_depth_mask(
        self,
        source_depth: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if source_depth is None:
            return None
        depth_tensor = source_depth.to(self.device)
        if depth_tensor.dim() == 3:
            depth_tensor = depth_tensor.unsqueeze(1)
        if depth_tensor.dim() == 4 and depth_tensor.shape[1] > 1:
            depth_tensor = depth_tensor[:, :1, ...]
        if depth_tensor.shape[-2:] != target_hw:
            depth_tensor = F.interpolate(depth_tensor.float(), size=target_hw, mode="nearest")
        mask_tensor = (depth_tensor > 1e-6).to(dtype).squeeze(1)
        return torch.nan_to_num(mask_tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _normalize_region_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        normalized_mask = mask.to(device=self.device, dtype=torch.float32)
        if normalized_mask.dim() == 4:
            normalized_mask = normalized_mask.squeeze(0).squeeze(0)
        elif normalized_mask.dim() == 3:
            normalized_mask = normalized_mask.squeeze(0)
        if normalized_mask.dim() != 2:
            return None
        normalized_mask = torch.nan_to_num(normalized_mask, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized_mask.clamp_(0.0, 1.0)

    def _compute_frontier_region_masks(
        self,
        exist_mask: Optional[torch.Tensor],
        novel_mask: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        normalized_exist_mask = self._normalize_region_mask(exist_mask)
        normalized_novel_mask = self._normalize_region_mask(novel_mask)
        if normalized_exist_mask is None:
            return None, None
        if normalized_novel_mask is None:
            normalized_novel_mask = torch.clamp(1.0 - normalized_exist_mask, min=0.0, max=1.0)
        frontier_radius = max(int(getattr(self.config, "phase2_frontier_band_radius", 3)), 1)
        dilation_kernel = frontier_radius * 2 + 1
        dilated_exist_mask = F.max_pool2d(
            normalized_exist_mask.unsqueeze(0).unsqueeze(0),
            kernel_size=dilation_kernel,
            stride=1,
            padding=frontier_radius,
        ).squeeze(0).squeeze(0)
        frontier_mask = torch.clamp(normalized_novel_mask * dilated_exist_mask, min=0.0, max=1.0)
        unsupported_novel_mask = torch.clamp(normalized_novel_mask - frontier_mask, min=0.0, max=1.0)
        return frontier_mask, unsupported_novel_mask

    def _attach_frontier_metrics(self, results: Sequence[UncertaintyResult]) -> None:
        for result in results:
            frontier_mask, unsupported_novel_mask = self._compute_frontier_region_masks(
                result.exist_mask,
                result.novel_mask,
            )
            if result.metrics is None:
                result.metrics = {}
            if frontier_mask is None:
                result.metrics["frontier_area_ratio"] = 0.0
                result.metrics["frontier_coverage"] = 0.0
                result.metrics["unsupported_novel_ratio"] = 0.0
                continue
            novel_mask = self._normalize_region_mask(result.novel_mask)
            if novel_mask is None:
                novel_mask = torch.clamp(1.0 - frontier_mask, min=0.0, max=1.0)
            novel_area = float(novel_mask.sum().item())
            frontier_area = float(frontier_mask.sum().item())
            unsupported_area = (
                float(unsupported_novel_mask.sum().item())
                if unsupported_novel_mask is not None
                else max(novel_area - frontier_area, 0.0)
            )
            frontier_area_ratio = frontier_area / max(novel_area, 1e-6)
            unsupported_novel_ratio = unsupported_area / max(novel_area, 1e-6)
            result.metrics["frontier_area_ratio"] = float(frontier_area_ratio)
            result.metrics["frontier_coverage"] = float(frontier_mask.mean().item())
            result.metrics["unsupported_novel_ratio"] = float(unsupported_novel_ratio)
            result.metrics["frontier_weighted_coverage"] = float(frontier_area / max(frontier_mask.numel(), 1))

    @staticmethod
    def _extract_generated_rgb_tensor(gen_result: Any) -> Optional[torch.Tensor]:
        if isinstance(gen_result, dict):
            generated_rgb = gen_result.get("rgb", gen_result.get("image"))
        else:
            generated_rgb = gen_result
        if not isinstance(generated_rgb, torch.Tensor):
            return None
        if generated_rgb.dim() == 4:
            generated_rgb = generated_rgb[0]
        return generated_rgb

    @staticmethod
    def _extract_generation_diagnostics(gen_result: Any) -> Dict[str, Any]:
        if not isinstance(gen_result, dict):
            return {}
        diagnostics = gen_result.get("diagnostics", {})
        return diagnostics if isinstance(diagnostics, dict) else {}

    @staticmethod
    def _masked_l1_from_mask(
        lhs: torch.Tensor,
        rhs: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Optional[float]:
        if mask is None:
            return None
        lhs_tensor = lhs
        rhs_tensor = rhs
        mask_tensor = mask
        if lhs_tensor.dim() == 3:
            lhs_tensor = lhs_tensor.unsqueeze(0)
        if rhs_tensor.dim() == 3:
            rhs_tensor = rhs_tensor.unsqueeze(0)
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
        elif mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.to(device=lhs_tensor.device, dtype=lhs_tensor.dtype)
        if mask_tensor.shape[-2:] != lhs_tensor.shape[-2:]:
            mask_tensor = F.interpolate(mask_tensor, size=lhs_tensor.shape[-2:], mode="nearest")
        if rhs_tensor.shape[-2:] != lhs_tensor.shape[-2:]:
            rhs_tensor = F.interpolate(rhs_tensor, size=lhs_tensor.shape[-2:], mode="bilinear", align_corners=False)
        masked_denominator = float(mask_tensor.sum().item()) * float(lhs_tensor.shape[1])
        if masked_denominator <= 1e-6:
            return None
        masked_l1 = (lhs_tensor - rhs_tensor).abs() * mask_tensor
        return float(masked_l1.sum().item() / masked_denominator)

    @staticmethod
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

    @staticmethod
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

    @classmethod
    def _resize_single_channel_mask(
        cls,
        mask: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        normalized_mask = cls._normalize_single_channel_mask(mask)
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

    @classmethod
    def _resize_rgb_tensor(
        cls,
        image: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        normalized_image = cls._normalize_rgb_tensor(image)
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

    @classmethod
    def _compute_single_channel_l1_map(
        cls,
        lhs: Optional[torch.Tensor],
        rhs: Optional[torch.Tensor],
        target_hw: Tuple[int, int],
    ) -> Optional[torch.Tensor]:
        lhs_image = cls._resize_rgb_tensor(lhs, target_hw)
        rhs_image = cls._resize_rgb_tensor(rhs, target_hw)
        if lhs_image is None or rhs_image is None:
            return None
        return (lhs_image - rhs_image).abs().mean(dim=0, keepdim=True)

    @classmethod
    def _apply_local_mask_consistency_filter(
        cls,
        mask: Optional[torch.Tensor],
        *,
        kernel_size: int,
        min_ratio: float,
    ) -> Optional[torch.Tensor]:
        normalized_mask = cls._normalize_single_channel_mask(mask)
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

    @classmethod
    def _mask_coverage(cls, mask: Optional[torch.Tensor]) -> float:
        normalized_mask = cls._normalize_single_channel_mask(mask)
        if normalized_mask is None:
            return 0.0
        return float(normalized_mask.mean().item())

    @staticmethod
    def _extract_support_projection_ratio(generation_diagnostics: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(generation_diagnostics, dict):
            return None
        for field_name in (
            "support_projection_ratio",
            "support_projection_ratio_for_schedule",
            "quality_gated_support_projection_coverage",
            "legacy_support_projection_coverage",
        ):
            field_value = _to_optional_float(generation_diagnostics.get(field_name))
            if field_value is not None:
                return float(field_value)
        return None

    def _build_probe_consistency_debug_bundle(
        self,
        sparse_bundle: SparseViewBundle,
        candidate: ViewCandidate,
        probe_conditions: ConditionDict,
    ) -> Dict[str, torch.Tensor]:
        debug_bundle: Dict[str, torch.Tensor] = {}
        rgb_render = probe_conditions.get("rgb_render")
        if not isinstance(rgb_render, torch.Tensor):
            return debug_bundle
        render_rgb = rgb_render[:1].float().to(self.device)
        depth_render = probe_conditions.get("depth_render", torch.zeros_like(render_rgb[:, :1]))
        if not isinstance(depth_render, torch.Tensor):
            depth_render = torch.zeros_like(render_rgb[:, :1])
        render_depth = depth_render[:1].float().to(self.device)
        if render_depth.dim() == 4 and render_depth.shape[1] > 1:
            render_depth = render_depth[:, :1]
        render_mask = (render_depth > 1e-6).float()
        debug_bundle["render_rgb"] = render_rgb
        debug_bundle["render_depth"] = render_depth
        debug_bundle["render_mask"] = render_mask

        if self.model is None:
            return debug_bundle
        condition_preprocessor = getattr(self.model, "condition_preprocessor", None)
        nearest_view_fn = getattr(self.model, "_select_nearest_view", None)
        if condition_preprocessor is None or nearest_view_fn is None:
            return debug_bundle

        sparse_images = sparse_bundle.images.to(self.device)
        sparse_poses = sparse_bundle.poses.to(self.device)
        target_pose = candidate.viewmat.unsqueeze(0).to(self.device)
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

        target_intrinsics = probe_conditions.get("camera_intrinsics")
        if isinstance(target_intrinsics, torch.Tensor):
            target_intrinsics = target_intrinsics[:1].to(device=self.device, dtype=torch.float32)
        else:
            target_intrinsics = candidate.intrinsics.unsqueeze(0).to(device=self.device, dtype=torch.float32)

        reference_intrinsics = probe_conditions.get("reference_intrinsics")
        if isinstance(reference_intrinsics, torch.Tensor):
            reference_intrinsics = reference_intrinsics[:1].to(device=self.device, dtype=torch.float32)
        else:
            reference_intrinsics = target_intrinsics

        reference_depth_map = probe_conditions.get("reference_depth_map")
        if isinstance(reference_depth_map, torch.Tensor):
            reference_depth_map = reference_depth_map[:1].to(device=self.device, dtype=torch.float32)
        else:
            reference_depth_map = None

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

    def _build_probe_verified_evidence_mask(
        self,
        probe_rgb: Optional[torch.Tensor],
        debug_bundle: Dict[str, torch.Tensor],
        generation_diagnostics: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        verified_metrics: Dict[str, float] = {}
        supported_exist_mask = self._normalize_single_channel_mask(debug_bundle.get("supported_exist_mask"))
        render_rgb = self._normalize_rgb_tensor(debug_bundle.get("render_rgb"))
        if supported_exist_mask is None or render_rgb is None:
            return None, verified_metrics

        target_hw = supported_exist_mask.shape[-2:]
        probe_rgb_resized = self._resize_rgb_tensor(probe_rgb, target_hw)
        render_rgb_resized = self._resize_rgb_tensor(render_rgb, target_hw)
        warped_rgb_resized = self._resize_rgb_tensor(debug_bundle.get("warped_rgb"), target_hw)
        if probe_rgb_resized is None or render_rgb_resized is None:
            return None, verified_metrics

        warp_valid_ratio = _to_optional_float(
            generation_diagnostics.get("warp_valid_ratio") if isinstance(generation_diagnostics, dict) else None
        )
        support_projection_ratio = self._extract_support_projection_ratio(generation_diagnostics)
        if warp_valid_ratio is not None:
            verified_metrics["probe_verified_warp_valid_ratio"] = float(warp_valid_ratio)
            if warp_valid_ratio < float(getattr(self.config, "pseudo_verified_min_warp_valid_ratio", 0.0)):
                return None, verified_metrics
        if support_projection_ratio is not None:
            verified_metrics["probe_verified_support_projection_ratio"] = float(support_projection_ratio)
            if support_projection_ratio < float(
                getattr(self.config, "pseudo_verified_min_support_projection_ratio", 0.0)
            ):
                return None, verified_metrics

        verified_mask = supported_exist_mask.clone()
        verified_metrics["probe_supported_input_coverage"] = float(verified_mask.mean().item())

        render_l1_map = self._compute_single_channel_l1_map(probe_rgb_resized, render_rgb_resized, target_hw)
        if render_l1_map is not None:
            render_l1_threshold = float(getattr(self.config, "pseudo_verified_pixel_l1_max", 0.18))
            verified_mask = verified_mask * (render_l1_map <= render_l1_threshold).float()

        if warped_rgb_resized is not None:
            pseudo_warp_l1_map = self._compute_single_channel_l1_map(probe_rgb_resized, warped_rgb_resized, target_hw)
            warp_render_l1_map = self._compute_single_channel_l1_map(warped_rgb_resized, render_rgb_resized, target_hw)
            if pseudo_warp_l1_map is not None:
                pseudo_warp_l1_threshold = float(getattr(self.config, "pseudo_verified_warp_l1_max", 0.18))
                verified_mask = verified_mask * (pseudo_warp_l1_map <= pseudo_warp_l1_threshold).float()
            if warp_render_l1_map is not None:
                warp_render_l1_threshold = float(getattr(self.config, "pseudo_verified_warp_render_l1_max", 0.12))
                verified_mask = verified_mask * (warp_render_l1_map <= warp_render_l1_threshold).float()

        verified_mask = self._apply_local_mask_consistency_filter(
            verified_mask,
            kernel_size=int(getattr(self.config, "pseudo_verified_neighborhood_kernel", 1)),
            min_ratio=float(getattr(self.config, "pseudo_verified_neighborhood_ratio", 0.0)),
        )
        if verified_mask is None:
            return None, verified_metrics

        verified_coverage = float(verified_mask.mean().item())
        verified_metrics["probe_verified_evidence_coverage"] = verified_coverage
        masked_render_l1 = self._masked_l1_from_mask(probe_rgb_resized, render_rgb_resized, verified_mask)
        if masked_render_l1 is not None:
            verified_metrics["probe_verified_masked_render_l1"] = float(masked_render_l1)

        masked_pseudo_warp_l1 = None
        if warped_rgb_resized is not None:
            masked_pseudo_warp_l1 = self._masked_l1_from_mask(probe_rgb_resized, warped_rgb_resized, verified_mask)
            if masked_pseudo_warp_l1 is not None:
                verified_metrics["probe_verified_masked_pseudo_warp_l1"] = float(masked_pseudo_warp_l1)

        agreement_score = 1.0
        render_l1_threshold = float(max(getattr(self.config, "pseudo_verified_pixel_l1_max", 0.18), 1e-6))
        if masked_render_l1 is not None:
            agreement_score *= max(0.0, 1.0 - float(masked_render_l1) / render_l1_threshold)
        pseudo_warp_l1_threshold = float(max(getattr(self.config, "pseudo_verified_warp_l1_max", 0.18), 1e-6))
        if masked_pseudo_warp_l1 is not None:
            agreement_score *= max(0.0, 1.0 - float(masked_pseudo_warp_l1) / pseudo_warp_l1_threshold)
        verified_metrics["probe_verified_evidence_score"] = float(verified_coverage * agreement_score)
        return verified_mask, verified_metrics

    def _run_probe_rerank(
        self,
        sparse_bundle: SparseViewBundle,
        results: List[UncertaintyResult],
    ) -> None:
        if (
            self.model is None
            or not bool(getattr(self.config, "phase2_probe_rerank_enable", False))
            or len(results) == 0
        ):
            return
        probe_topk = max(1, min(int(getattr(self.config, "phase2_probe_topk", 8)), len(results)))
        probe_strength = float(getattr(self.config, "phase2_probe_rerank_strength", 1.0))
        if probe_strength <= 0.0:
            return
        gate_threshold = float(getattr(self.config, "pseudo_exist_region_l1_max", 0.20))
        degrade_threshold = float(
            getattr(
                self.config,
                "pseudo_degraded_accept_l1_max",
                gate_threshold * 1.5,
            )
        )
        ranked_indices = sorted(range(len(results)), key=lambda idx: results[idx].score, reverse=True)[:probe_topk]
        for rank_1based, result_index in enumerate(ranked_indices, start=1):
            current_result = results[result_index]
            try:
                probe_conditions = self._prepare_conditions([current_result.candidate])
                probe_conditions = {
                    key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                    for key, value in probe_conditions.items()
                }
                rgb_render = probe_conditions["rgb_render"].to(self.device)
                depth_render = probe_conditions.get(
                    "depth_render",
                    torch.zeros_like(rgb_render[:, :1]),
                ).to(self.device)
                normal_render = probe_conditions.get(
                    "normal_render",
                    torch.zeros_like(rgb_render),
                ).to(self.device)
                with torch.no_grad():
                    probe_output = self.model.generate_view_correctly(
                        sparse_images=sparse_bundle.images.to(self.device),
                        sparse_poses=sparse_bundle.poses.to(self.device),
                        target_poses=current_result.candidate.viewmat.unsqueeze(0).to(self.device),
                        rgb_render=rgb_render,
                        depth_render=depth_render,
                        normal_render=normal_render,
                        reference_depth_map=probe_conditions.get("reference_depth_map"),
                        reference_intrinsics=probe_conditions.get("reference_intrinsics"),
                        camera_intrinsics=probe_conditions.get("camera_intrinsics"),
                        steps=int(self.config.fine_steps),
                        resolution=int(self.config.fine_resolution),
                        return_diagnostics=True,
                    )
                probe_rgb = self._extract_generated_rgb_tensor(probe_output)
                if probe_rgb is None:
                    continue
                probe_diagnostics = self._extract_generation_diagnostics(probe_output)
                probe_debug_bundle = self._build_probe_consistency_debug_bundle(
                    sparse_bundle,
                    current_result.candidate,
                    probe_conditions,
                )
                render_mask = self._normalize_single_channel_mask(probe_debug_bundle.get("render_mask"))
                supported_exist_mask = self._normalize_single_channel_mask(
                    probe_debug_bundle.get("supported_exist_mask")
                )
                exist_l1_render_only = self._masked_l1_from_mask(probe_rgb, rgb_render, render_mask)
                supported_exist_l1 = self._masked_l1_from_mask(
                    probe_rgb,
                    rgb_render,
                    supported_exist_mask,
                )
                warp_vs_render_l1 = self._masked_l1_from_mask(
                    probe_debug_bundle.get("warped_rgb"),
                    rgb_render,
                    supported_exist_mask,
                )
                frontier_mask, _ = self._compute_frontier_region_masks(
                    current_result.exist_mask,
                    current_result.novel_mask,
                )
                frontier_l1 = None
                if frontier_mask is not None:
                    frontier_l1 = self._masked_l1_from_mask(
                        probe_rgb,
                        rgb_render,
                        frontier_mask,
                    )
                warp_valid_ratio = probe_diagnostics.get("warp_valid_ratio")
                support_projection_ratio = self._extract_support_projection_ratio(probe_diagnostics)
                warp_valid_ratio = float(warp_valid_ratio) if warp_valid_ratio is not None else 0.0
                support_projection_ratio = float(support_projection_ratio) if support_projection_ratio is not None else 0.0
                supported_coverage = self._mask_coverage(supported_exist_mask)
                unsupported_coverage = 0.0
                if render_mask is not None and supported_exist_mask is not None:
                    unsupported_render_mask = (render_mask * (1.0 - supported_exist_mask)).clamp(0.0, 1.0)
                    unsupported_coverage = self._mask_coverage(unsupported_render_mask)
                probe_verified_mask, probe_verified_metrics = self._build_probe_verified_evidence_mask(
                    probe_rgb,
                    probe_debug_bundle,
                    probe_diagnostics,
                )
                verified_coverage = float(probe_verified_metrics.get("probe_verified_evidence_coverage", 0.0))
                verified_score = float(probe_verified_metrics.get("probe_verified_evidence_score", 0.0))
                consistency_term = 0.0
                reference_consistency_l1 = supported_exist_l1
                if reference_consistency_l1 is None:
                    reference_consistency_l1 = exist_l1_render_only
                if reference_consistency_l1 is not None:
                    consistency_term = max(
                        0.0,
                        1.0 - float(reference_consistency_l1) / max(degrade_threshold, 1e-6),
                    )
                geometry_agreement_term = 0.0
                if warp_vs_render_l1 is not None:
                    geometry_agreement_term = max(
                        0.0,
                        1.0
                        - float(warp_vs_render_l1)
                        / max(float(getattr(self.config, "pseudo_verified_warp_render_l1_max", 0.12)), 1e-6),
                    )
                frontier_prior = float((current_result.metrics or {}).get("frontier_area_ratio", 0.0))
                min_verified_coverage = float(getattr(self.config, "phase2_probe_min_verified_coverage", 0.10))
                expected_verified_utility = min(
                    max(verified_score / max(min_verified_coverage, 1e-6), 0.0),
                    1.0,
                )
                generation_mode_name = str(probe_diagnostics.get("generation_mode", "") or "").lower()
                generation_mode_penalty = 1.0
                if generation_mode_name == "img2img":
                    generation_mode_penalty *= 0.7
                elif generation_mode_name == "text2img":
                    generation_mode_penalty *= 0.5
                warp_inpaint_succeeded = probe_diagnostics.get("warp_inpaint_succeeded")
                if warp_inpaint_succeeded is False:
                    generation_mode_penalty *= 0.75
                low_verified_penalty = 1.0
                if verified_coverage < min_verified_coverage:
                    low_verified_penalty = max(verified_coverage / max(min_verified_coverage, 1e-6), 0.05)
                recoverability_score = (
                    0.40 * expected_verified_utility
                    + 0.20 * min(max(supported_coverage, 0.0), 1.0)
                    + 0.15 * min(max(warp_valid_ratio, 0.0), 1.0)
                    + 0.10 * min(max(support_projection_ratio, 0.0), 1.0)
                    + 0.10 * consistency_term
                    + 0.05 * geometry_agreement_term
                    + 0.05 * min(max(frontier_prior, 0.0), 1.0)
                    - 0.15 * min(max(unsupported_coverage, 0.0), 1.0)
                )
                recoverability_score *= generation_mode_penalty * low_verified_penalty
                recoverability_score = float(min(max(recoverability_score, 0.0), 1.0))
                selection_priority = (
                    0.70 * expected_verified_utility
                    + 0.15 * consistency_term
                    + 0.10 * geometry_agreement_term
                    + 0.05 * min(max(supported_coverage, 0.0), 1.0)
                )
                selection_priority *= generation_mode_penalty * low_verified_penalty
                selection_priority = float(min(max(selection_priority, 0.0), 1.0))
                original_score = float(current_result.score)
                probe_multiplier = (1.0 - probe_strength) + probe_strength * recoverability_score
                current_result.score = original_score * max(probe_multiplier, 1e-3)
                if current_result.metrics is None:
                    current_result.metrics = {}
                current_result.metrics["probe_rank_1based"] = float(rank_1based)
                current_result.metrics["probe_warp_valid_ratio"] = float(warp_valid_ratio)
                current_result.metrics["probe_support_projection_ratio"] = float(support_projection_ratio)
                current_result.metrics["probe_exist_l1_render_only"] = (
                    float(exist_l1_render_only) if exist_l1_render_only is not None else -1.0
                )
                current_result.metrics["probe_supported_exist_l1"] = (
                    float(supported_exist_l1) if supported_exist_l1 is not None else -1.0
                )
                current_result.metrics["probe_warp_vs_render_l1_on_supported_exist"] = (
                    float(warp_vs_render_l1) if warp_vs_render_l1 is not None else -1.0
                )
                current_result.metrics["probe_frontier_l1"] = (
                    float(frontier_l1) if frontier_l1 is not None else -1.0
                )
                current_result.metrics["probe_supported_exist_coverage"] = float(supported_coverage)
                current_result.metrics["probe_unsupported_render_coverage"] = float(unsupported_coverage)
                current_result.metrics["probe_verified_evidence_coverage"] = float(verified_coverage)
                current_result.metrics["probe_verified_evidence_score"] = float(verified_score)
                current_result.metrics["probe_expected_verified_utility"] = float(expected_verified_utility)
                current_result.metrics["probe_geometry_agreement_term"] = float(geometry_agreement_term)
                current_result.metrics["probe_generation_mode_penalty"] = float(generation_mode_penalty)
                current_result.metrics["probe_low_verified_penalty"] = float(low_verified_penalty)
                current_result.metrics["probe_inpaint_mode"] = 1.0 if generation_mode_name == "inpaint" else 0.0
                current_result.metrics["probe_warp_inpaint_succeeded"] = (
                    1.0 if bool(warp_inpaint_succeeded) else 0.0
                )
                current_result.metrics["probe_consistency_term"] = float(consistency_term)
                current_result.metrics["probe_recoverability_score"] = float(recoverability_score)
                current_result.metrics["probe_selection_priority"] = float(selection_priority)
                current_result.metrics["probe_base_score"] = float(original_score)
                current_result.metrics["probe_multiplier"] = float(probe_multiplier)
                current_result.metrics["probe_reranked_score"] = float(current_result.score)
            except Exception as probe_exception:
                if current_result.metrics is None:
                    current_result.metrics = {}
                current_result.metrics["probe_failed"] = 1.0
                if not getattr(self, "_probe_rerank_fail_logged", False):
                    print(f"[Phase2Probe] recoverability重排失败，已退回原排序: {probe_exception}")
                    self._probe_rerank_fail_logged = True

    def _reduce_score_maps(
        self,
        variance_map: torch.Tensor,
        score_map: torch.Tensor,
        mask_tensor: Optional[torch.Tensor],
        *,
        use_mask_scores: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        def _mean_with_mask(data: torch.Tensor, current_mask: Optional[torch.Tensor]) -> torch.Tensor:
            if current_mask is None:
                return data.mean(dim=(1, 2))
            masked_sum = (data * current_mask).sum(dim=(1, 2))
            denom = current_mask.sum(dim=(1, 2)).clamp_min(1.0)
            return masked_sum / denom

        variance_unmasked = variance_map.mean(dim=(1, 2))
        score_unmasked = score_map.mean(dim=(1, 2))
        if mask_tensor is None:
            coverage = torch.zeros_like(score_unmasked)
            fallback = torch.ones_like(score_unmasked, dtype=torch.bool) if use_mask_scores else torch.zeros_like(score_unmasked, dtype=torch.bool)
            return variance_unmasked, variance_unmasked, score_unmasked, score_unmasked, score_unmasked, coverage.to(variance_map.dtype), fallback

        coverage = mask_tensor.mean(dim=(1, 2))
        variance_masked_raw = _mean_with_mask(variance_map, mask_tensor)
        score_masked_raw = _mean_with_mask(score_map, mask_tensor)

        min_coverage = float(getattr(self.config, "mask_score_min_coverage", 0.05))
        tiny_eps = float(getattr(self.config, "score_fallback_variance_eps", 1e-8))
        fallback = coverage < min_coverage
        fallback = fallback | ~torch.isfinite(score_masked_raw)
        fallback = fallback | ~torch.isfinite(variance_masked_raw)
        fallback = fallback | ((score_masked_raw.abs() <= tiny_eps) & (score_unmasked.abs() > tiny_eps))

        variance_masked = torch.where(fallback, variance_unmasked, variance_masked_raw)
        score_masked = torch.where(fallback, score_unmasked, score_masked_raw)
        scores = score_masked if use_mask_scores else score_unmasked
        return variance_masked, variance_unmasked, score_masked, score_unmasked, scores, coverage, fallback

    def estimate(
        self,
        sparse_bundle: SparseViewBundle,
        candidates: Sequence[ViewCandidate],
        *,
        global_step: Optional[int] = None,
        num_samples: Optional[int] = None,
        label: str = "fine",
    ) -> List[UncertaintyResult]:
        results: List[UncertaintyResult] = []
        force_single_candidate_generation = bool(
            getattr(self.config, "estimator_force_single_candidate_generation", False)
        )
        if force_single_candidate_generation and self.mode == "gddn" and self.model is not None:
            batch_size = 1
        else:
            batch_size = max(1, self.config.batch_size)
        sample_count = num_samples or self.config.mc_dropout_samples_fine

        use_uncertainty_heads = (
            self.mode == "gddn"
            and self.model is not None
            and self.model.epistemic_head is not None
            and self.model.aleatoric_head is not None
        )

        for start in range(0, len(candidates), batch_size):
            batch_candidates = candidates[start : start + batch_size]
            conditions = self._prepare_conditions(batch_candidates)
            if use_uncertainty_heads:
                batch_results = self._head_uncertainty_pass(
                    sparse_bundle=sparse_bundle,
                    candidates=batch_candidates,
                    conditions=conditions,
                    batch_offset=start,
                    global_step=global_step,
                    label=label,
                )
            elif self.mode == "gddn" and bool(getattr(self.config, "mc_two_layer_enable", False)):
                # Layer 2 算法2：两层MC采样（精确epistemic/aleatoric分离）
                batch_results = self._two_layer_mc_pass(
                    sparse_bundle,
                    batch_candidates,
                    conditions,
                    batch_offset=start,
                    global_step=global_step,
                    label=label,
                )
            else:
                batch_results = self._mc_dropout_pass(
                    sparse_bundle,
                    batch_candidates,
                    conditions,
                    batch_offset=start,
                    global_step=global_step,
                    num_samples=sample_count,
                    label=label,
                )
            results.extend(batch_results)

        self._attach_frontier_metrics(results)

        # ========== DDUD: 双维度不确定性解耦（Dual-Dimension Uncertainty Decoupling）==========
        # 维度1: U_geo（3DGS-native几何不确定性）
        #   公式: U_geo(v*) = 1 / mean(count_accum[visible_gaussians(v*)])
        #   含义: 可见高斯体被观测次数少 → 几何约束弱 → 信息增益大
        # 维度2: Q_gen（MC-Dropout生成可信度）
        #   公式: Q_gen = 1/(1+σ²_MC) ∈(0,1]
        #   含义: MC方差越高 → SD生成越不可信 → SoftGating降权
        # 联合决策: Priority(v*) = U_geo(v*) × Q_gen(v*)
        _ddud_enable = getattr(self.config, "ddud_enable", True)
        if _ddud_enable and results:
            try:
                n_results = len(results)
                _rank_denom = max(n_results - 1, 1)

                # --- 计算精确 U_geo: frustum culling + count_accum查找 ---
                u_geo_list: List[float] = []
                _has_count_accum = (
                    self.gaussians is not None
                    and hasattr(self.gaussians, "_ddud_count_accum")
                    and self.gaussians._ddud_count_accum is not None
                )
                for result_i, result in enumerate(results):
                    if _has_count_accum:
                        # 精确公式: U_geo = 1 / mean(count_accum[visible_gaussians(v*)])
                        candidate = result.candidate
                        # 从平均引数推断渲染分辨率
                        if hasattr(self.config, 'fine_resolution') and self.config.fine_resolution:
                            _render_hw = (int(self.config.fine_resolution), int(self.config.fine_resolution))
                        else:
                            _render_hw = (256, 256)
                        u_geo_val = self._compute_u_geo_frustum(
                            candidate=candidate,
                            count_accum=self.gaussians._ddud_count_accum,
                            render_hw=_render_hw,
                        )
                    elif result.metrics and "mask_coverage" in result.metrics:
                        # 降级代理: mask_coverage倒数
                        coverage = max(float(result.metrics["mask_coverage"]), 1e-4)
                        u_geo_val = 1.0 / coverage
                    else:
                        # 最后降级: variance_map均値
                        u_geo_val = float(result.variance_map.mean().item()) + 1e-6
                    u_geo_list.append(u_geo_val)

                # --- 计算 Q_gen: σ²_MC → 生成可信度 ---
                q_gen_list: List[float] = []
                for result in results:
                    sigma2_mc = float(result.variance_map.mean().item())
                    q_gen_val = 1.0 / (1.0 + sigma2_mc)  # ∈(0,1]，方差越低越可信
                    q_gen_list.append(q_gen_val)

                # --- U_geo rank归一化 ---
                geo_sorted_indices = sorted(range(n_results), key=lambda i: u_geo_list[i])
                u_geo_ranks = [0.0] * n_results
                for rank_pos, idx in enumerate(geo_sorted_indices):
                    u_geo_ranks[idx] = rank_pos / _rank_denom  # 高U_geo→高rank→高优先级

                # --- 联合优先级 Priority = U_geo_rank × D_baseline × recoverability_boost ---
                _ddud_geo_weight = float(getattr(self.config, "ddud_geo_weight", 0.5))
                _ddud_hole_weight = float(getattr(self.config, "ddud_hole_weight", 0.0))
                _ddud_frontier_weight = float(getattr(self.config, "ddud_frontier_weight", 0.0))
                _ddud_unsupported_penalty = float(getattr(self.config, "ddud_unsupported_penalty", 0.0))
                # --- [改动4] 预计算 D_baseline: 候选与真实相机的偏轴角正弦 ---
                # D_baseline(v*) = min_{v_i∈existing} sin(angle(v*-C_i, d_i))
                # 偏轴越大 → 深度约束互补性越强 → 解决 F1 平面退化
                _ref_cams = (
                    [c for c in self._reference_cameras]
                    if hasattr(self, '_reference_cameras') and self._reference_cameras
                    else []
                )
                d_baseline_list: list = []
                for res in results:
                    _d_base = 0.5  # 默认中性值（无法计算时）
                    if _ref_cams and hasattr(res, 'candidate') and res.candidate is not None:
                        try:
                            _cand_vm = res.candidate.viewmat  # W2C [4,4]
                            _cand_pos = -_cand_vm[:3, :3].T @ _cand_vm[:3, 3]
                            _min_sin = 1.0
                            for _rc in _ref_cams:
                                _rc_vm = _rc.viewmat
                                _rc_pos = -_rc_vm[:3, :3].T @ _rc_vm[:3, 3]
                                _rc_fwd = _rc_vm[2, :3]  # W2C Z行 = 前方方向
                                _delta = _cand_pos - _rc_pos
                                _delta_norm = _delta / (_delta.norm() + 1e-8)
                                _cos_a = _rc_fwd.dot(_delta_norm).abs()
                                _sin_a = float((1.0 - _cos_a ** 2).clamp(min=0).sqrt().item())
                                _min_sin = min(_min_sin, _sin_a)
                            _d_base = _min_sin
                        except Exception:
                            _d_base = 0.5
                    d_baseline_list.append(_d_base)

                for idx, result in enumerate(results):
                    _u_geo_rank = u_geo_ranks[idx]
                    _q_gen = q_gen_list[idx]
                    _d_base = d_baseline_list[idx]
                    _coverage = 1.0
                    if result.metrics is not None:
                        _coverage = float(result.metrics.get("mask_coverage", 1.0))
                    _coverage = min(max(_coverage, 0.0), 1.0)
                    _coverage_gap = 1.0 - _coverage
                    _frontier_ratio = float((result.metrics or {}).get("frontier_area_ratio", 0.0))
                    _unsupported_novel_ratio = float((result.metrics or {}).get("unsupported_novel_ratio", _coverage_gap))
                    _recoverability_boost = (
                        1.0
                        + _ddud_hole_weight * _coverage_gap
                        + _ddud_frontier_weight * _frontier_ratio
                        - _ddud_unsupported_penalty * _unsupported_novel_ratio
                    )
                    _recoverability_boost = max(_recoverability_boost, 0.1)
                    # U_geo_rank：几何信息增益；MC_score：原始不确定性分；D_baseline：深度补偿系数
                    _priority_before_qgen = (
                        _ddud_geo_weight * _u_geo_rank
                        + (1.0 - _ddud_geo_weight) * result.score
                    )
                    # q_gen 当前未校准，仅保留诊断记录，不再参与候选排序。
                    _priority = _priority_before_qgen * (0.5 + 0.5 * _d_base) * _recoverability_boost

                    if result.metrics is None:
                        result.metrics = {}
                    result.metrics["u_geo"] = float(u_geo_list[idx])
                    result.metrics["u_geo_rank"] = _u_geo_rank
                    result.metrics["q_gen"] = _q_gen
                    result.metrics["d_baseline"] = _d_base
                    result.metrics["coverage_gap"] = _coverage_gap
                    result.metrics["coverage_boost"] = 1.0 + _ddud_hole_weight * _coverage_gap
                    result.metrics["recoverability_boost"] = _recoverability_boost
                    result.metrics["frontier_ratio_used"] = _frontier_ratio
                    result.metrics["unsupported_novel_ratio_used"] = _unsupported_novel_ratio
                    result.metrics["sigma2_mc"] = float(result.variance_map.mean().item())
                    result.metrics["ddud_priority_before_q_gen"] = _priority_before_qgen
                    result.metrics["ddud_priority"] = _priority
                    result.score = _priority  # 替换为联合优先级分数

                if global_step is not None and global_step % 200 == 0 and results:
                    _top = max(results, key=lambda r: r.score)
                    _top_idx = next(
                        (i for i, r in enumerate(results) if r is _top), 0
                    )
                    _mode_str = "count_accum" if _has_count_accum else "mask_proxy"
                    print(f"[DDUD/{_mode_str}] step={global_step}: top候选#{_top_idx} "
                          f"u_geo={u_geo_list[_top_idx]:.4f}(rank={u_geo_ranks[_top_idx]:.2f}) "
                          f"d_base={d_baseline_list[_top_idx]:.3f} "
                          f"q_gen={q_gen_list[_top_idx]:.4f} priority={_top.score:.4f}")

            except Exception as _ddud_e:
                import traceback
                print(f"[DDUD] 双维度评分计算失败: {_ddud_e}")
                traceback.print_exc()

        if label == "fine":
            self._run_probe_rerank(sparse_bundle, results)

        return results

    def _compute_u_geo_frustum(
        self,
        candidate: "ViewCandidate",
        count_accum: torch.Tensor,
        render_hw: Tuple[int, int],
    ) -> float:
        """精确计算 3DGS-native 几何不确定性

        公式: U_geo(v*) = 1 / mean(count_accum[visible_gaussians(v*)])

        方法: Frustum culling判断可见高斯
        1. 将所有高斯体中心投影到候选视角图像平面
        2. 保留在相机前方且在图像范围内的高斯 (frustum culling)
        3. 查询这些高斯的 count_accum 平均值
        4. U_geo = 1/mean_count —— 观测少则不确定性高

        Returns:
            float: U_geo 分数，越高表示几何信息缺失越严重
        """
        if self.gaussians is None:
            return 1.0
        gauss_means = self.gaussians.means.detach()  # [N, 3]
        N = gauss_means.shape[0]
        if N == 0:
            return 10.0  # 无高斯→极高不确定性
        H, W = render_hw
        # 获取候选视角的相机内参数
        viewmat = candidate.viewmat.to(gauss_means.device)   # [4, 4]
        K = candidate.intrinsics.to(gauss_means.device)       # [3, 3]
        fx = K[0, 0].item()
        fy = K[1, 1].item()
        cx = K[0, 2].item()
        cy = K[1, 2].item()
        # 将高斯中心变换到相机坐标系
        ones = torch.ones(N, 1, device=gauss_means.device, dtype=gauss_means.dtype)
        means_h = torch.cat([gauss_means, ones], dim=1)  # [N, 4]
        # viewmat: world-to-camera [4,4]
        cam_coords = (viewmat @ means_h.T).T  # [N, 4]
        z = cam_coords[:, 2]  # 深度分量 [N]
        # 居中过滤: z > 0.01（相机前方）
        valid_z = z > 0.01
        if valid_z.sum() == 0:
            return 10.0  # 所有高斯均居相机后方
        # 投影到图像平面
        z_safe = z.clamp(min=1e-6)
        u_proj = cam_coords[:, 0] / z_safe * fx + cx
        v_proj = cam_coords[:, 1] / z_safe * fy + cy
        # Frustum culling: u∈[0,W), v∈[0,H)
        valid_uv = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H)
        visible_mask = valid_z & valid_uv
        if visible_mask.sum() == 0:
            return 10.0  # 无可见高斯 → 极高几何不确定性
        # 查询可见高斯的 count_accum
        _count_dev = count_accum.to(gauss_means.device)
        if _count_dev.shape[0] != N:
            # 如果 count_accum 大小不匹配（densification后），按实际大小截断
            _min_n = min(_count_dev.shape[0], N)
            visible_mask = visible_mask[:_min_n]
            _count_dev = _count_dev[:_min_n]
        visible_counts = _count_dev[visible_mask].float()
        mean_count = visible_counts.mean().clamp(min=1e-4).item()
        return 1.0 / mean_count

    def _prepare_conditions(
        self, candidates: Sequence[ViewCandidate]
    ) -> ConditionDict:
        conditions = self.condition_provider(candidates)
        if self.mode == "gs_dropout":
            required = {"rgb_render"}
        else:
            required = {"rgb_render", "depth_render", "normal_render", "target_latent"}
        missing = required.difference(conditions.keys())
        if missing:
            raise ValueError(f"Condition provider missing keys: {missing}")
        return {key: value.to(self.device) for key, value in conditions.items()}

    def _build_depth_adapter(self) -> Optional["_DepthAnythingAdapter"]:
        if not bool(getattr(self.config, "dacd_enable", False)):
            return None
        if not bool(getattr(self.config, "depth_anything_enable", False)):
            return None
        if DepthAnythingV2 is None:
            return None
        checkpoint = getattr(self.config, "depth_anything_checkpoint", None)
        encoder = getattr(self.config, "depth_anything_encoder", "vitg")
        checkpoint_path = Path(checkpoint) if checkpoint else (
            Path(__file__).resolve().parents[2]
            / "models"
            / "depth_anything_v2"
            / f"depth_anything_v2_{encoder}.pth"
        )
        if not checkpoint_path.is_file():
            return None
        device = getattr(self.config, "depth_anything_device", str(self.device))
        try:
            return _DepthAnythingAdapter(
                encoder=encoder,
                checkpoint=str(checkpoint_path),
                device=device,
            )
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[DepthAnything] 初始化失败: {exc}")
            return None

    def _predict_mono_depths(
        self, rgb: torch.Tensor
    ) -> Optional[List[torch.Tensor]]:
        if self._depth_adapter is None:
            return None
        try:
            result = self._depth_adapter.infer_batch(rgb)
            return result
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[DepthAnything] 推理失败: {exc}")
            return None

    def _compute_calibrated_depths(
        self, depth_render: torch.Tensor, mono_depths: Optional[Sequence[torch.Tensor]]
    ) -> List[Tuple[Optional[torch.Tensor], float, float, float, float]]:
        batch_size = depth_render.shape[0]
        fallback: List[Tuple[Optional[torch.Tensor], float, float, float, float]] = [
            (None, 0.0, 0.0, 0.0, 0.0) for _ in range(batch_size)
        ]
        if mono_depths is None:
            return fallback
        alignments: List[Tuple[Optional[torch.Tensor], float, float, float, float]] = []
        for index in range(batch_size):
            calibrated, anchor_ratio, r2, scale, bias = self._align_single_depth(
                depth_render[index], mono_depths[index]
            )
            alignments.append((calibrated, anchor_ratio, r2, scale, bias))
        return alignments

    def _align_single_depth(
        self, depth_map: torch.Tensor, mono_depth: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], float, float, float, float]:
        if depth_map.dim() == 3 and depth_map.shape[0] == 3:
            depth_tensor = depth_map[0].detach().squeeze().float()
        else:
            depth_tensor = depth_map.detach().squeeze().float()
            
        mono_tensor = mono_depth.detach().float()
        # 确保mono_tensor是2D [H, W]
        if mono_tensor.dim() == 3:
            mono_tensor = mono_tensor.squeeze()
            
        if depth_tensor.numel() == 0:
            return None, 0.0, 0.0, 0.0, 0.0
            
        if mono_tensor.shape != depth_tensor.shape:
            # depth_tensor应该是[H, W]，size也应该是(H, W)
            mono_tensor = F.interpolate(
                mono_tensor.unsqueeze(0).unsqueeze(0),
                size=depth_tensor.shape,
                mode="bilinear",
                align_corners=False,
            ).squeeze()
        valid = depth_tensor > 1e-4
        anchor_ratio = float(valid.float().mean().item()) if valid.numel() else 0.0
        min_ratio = float(getattr(self.config, "dacd_min_anchor_ratio", 0.0))
        if anchor_ratio < min_ratio or valid.sum() < 10:
            return None, anchor_ratio, 0.0, 0.0, 0.0
        depth_valid = torch.clamp(depth_tensor[valid], min=1e-3)
        mono_valid = torch.clamp(mono_tensor[valid], min=1e-4)
        target_disp = 1.0 / depth_valid
        A = torch.stack([mono_valid, torch.ones_like(mono_valid)], dim=1)
        try:
            solution = torch.linalg.lstsq(A, target_disp).solution
        except RuntimeError:
            return None, anchor_ratio, 0.0, 0.0, 0.0
        s, t = solution[0], solution[1]
        scale = float(s.item())
        bias = float(t.item())
        coeffs = torch.stack([s, t])
        pred_anchor = A @ coeffs
        target_mean = target_disp.mean()
        ss_tot = torch.sum((target_disp - target_mean) ** 2).clamp_min(1e-6)
        ss_res = torch.sum((target_disp - pred_anchor) ** 2)
        r2 = float(1.0 - (ss_res / ss_tot).item())
        min_r2 = float(getattr(self.config, "dacd_min_r2", 0.0))
        if not math.isfinite(r2) or r2 < min_r2:
            return None, anchor_ratio, r2, scale, bias
        disp_full = torch.clamp(s * mono_tensor + t, min=1e-4)
        calibrated = 1.0 / disp_full
        return calibrated, anchor_ratio, r2, scale, bias

    def _head_uncertainty_pass(
        self,
        *,
        sparse_bundle: SparseViewBundle,
        candidates: Sequence[ViewCandidate],
        conditions: ConditionDict,
        batch_offset: int,
        global_step: Optional[int],
        label: str,
    ) -> List[UncertaintyResult]:
        if self.model is None:
            raise RuntimeError("GDDN model not initialised.")
        if self.model.epistemic_head is None or self.model.aleatoric_head is None:
            raise RuntimeError("Uncertainty heads are not available for head-based inference.")

        if label == "coarse":
            resolution = int(self.config.coarse_resolution)
            steps = int(self.config.coarse_steps)
        else:
            resolution = int(self.config.fine_resolution)
            steps = int(self.config.fine_steps)

        candidate_viewmats = torch.stack([c.viewmat for c in candidates]).to(self.device)
        rgb_render = conditions["rgb_render"].to(self.device)
        depth_render = conditions.get("depth_render", torch.zeros_like(rgb_render[:, :1])).to(self.device)
        normal_render = conditions.get("normal_render", torch.zeros_like(rgb_render)).to(self.device)

        autocast_enabled = bool(self.config.use_mixed_precision)
        with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled):
            outputs = self.model.generate_with_uncertainty(
                sparse_images=sparse_bundle.images,
                sparse_poses=sparse_bundle.poses,
                target_poses=candidate_viewmats,
                rgb_render=rgb_render,
                depth_render=depth_render,
                normal_render=normal_render,
                reference_depth_map=conditions.get("reference_depth_map"),
                reference_intrinsics=conditions.get("reference_intrinsics"),
                steps=steps,
                resolution=resolution,
                sampler_type=self.config.mc_sampler_type,
            )
        rgb = outputs["rgb"]
        mc_epistemic = outputs.get("mc_epistemic", outputs["epistemic"])
        head_risk_exist = outputs.get(
            "head_risk_exist",
            outputs.get("risk_exist", outputs.get("head_epistemic", outputs["epistemic"])),
        )
        head_gain_novel = outputs.get(
            "head_gain_novel",
            outputs.get("gain_novel", outputs.get("head_aleatoric", outputs["aleatoric"])),
        )

        mono_depths = self._predict_mono_depths(rgb)
        depth_alignments = self._compute_calibrated_depths(depth_render, mono_depths)

        variance_map = torch.nan_to_num(mc_epistemic.squeeze(1))
        risk_exist_map = torch.nan_to_num(head_risk_exist.squeeze(1))
        gain_novel_map = torch.nan_to_num(head_gain_novel.squeeze(1))
        epistemic_map = risk_exist_map
        aleatoric_map = gain_novel_map
        predictive_var = torch.clamp(variance_map + aleatoric_map, min=1e-6)

        eps = 1e-6
        score_type = (self.config.score_type or "variance").lower()
        use_mask_scores = bool(self.config.use_mask_for_scores)
        score_map = variance_map.clone()
        mi_map: Optional[torch.Tensor] = None
        if score_type == "mi":
            conditional_floor = max(float(self.config.aleatoric_variance_floor), eps)
            conditional = torch.clamp(aleatoric_map, min=conditional_floor)
            mi_map = 0.5 * torch.log((predictive_var + eps) / (conditional + eps))
            mi_map = torch.clamp(mi_map, min=0.0)
            score_map = mi_map
        score_map = torch.nan_to_num(score_map)
        mask_tensor = self._align_depth_mask(
            conditions.get("depth_render"),
            variance_map.shape[-2:],
            variance_map.dtype,
        )
        novel_mask_tensor = None if mask_tensor is None else torch.clamp(1.0 - mask_tensor, min=0.0, max=1.0)
        (
            variance_masked,
            variance_unmasked,
            score_masked,
            score_unmasked,
            scores,
            mask_coverage,
            score_fallback,
        ) = self._reduce_score_maps(
            variance_map,
            score_map,
            mask_tensor,
            use_mask_scores=use_mask_scores,
        )

        results: List[UncertaintyResult] = []
        for index, candidate in enumerate(candidates):
            metrics: Dict[str, float] = {
                "variance_masked": float(variance_masked[index].item()),
                "variance_unmasked": float(variance_unmasked[index].item()),
                "mask_coverage": float(mask_coverage[index].item()),
                "score_fallback_to_unmasked": float(score_fallback[index].item()),
            }
            if mi_map is not None:
                metrics["mi_masked"] = float(score_masked[index].item())
                metrics["mi_unmasked"] = float(score_unmasked[index].item())
            metrics["epistemic_head_mean"] = float(epistemic_map[index].mean().item())
            metrics["aleatoric_mean"] = float(aleatoric_map[index].mean().item())
            if mask_tensor is not None:
                current_exist_mask = mask_tensor[index]
                exist_denom = current_exist_mask.sum().clamp_min(1.0)
                metrics["risk_exist_mean"] = float(
                    ((risk_exist_map[index] * current_exist_mask).sum() / exist_denom).item()
                )
            else:
                metrics["risk_exist_mean"] = float(risk_exist_map[index].mean().item())
            if novel_mask_tensor is not None:
                current_novel_mask = novel_mask_tensor[index]
                novel_denom = current_novel_mask.sum().clamp_min(1.0)
                metrics["gain_novel_mean"] = float(
                    ((gain_novel_map[index] * current_novel_mask).sum() / novel_denom).item()
                )
            else:
                metrics["gain_novel_mean"] = float(gain_novel_map[index].mean().item())
            metrics["risk_exist_mass"] = float(risk_exist_map[index].mean().item())
            metrics["gain_novel_mass"] = float(gain_novel_map[index].mean().item())
            (
                depth_calibrated,
                anchor_ratio,
                r2_value,
                disp_scale,
                disp_bias,
            ) = depth_alignments[index]
            if anchor_ratio > 0.0:
                metrics["depth_anchor_ratio"] = float(anchor_ratio)
            if r2_value != 0.0:
                metrics["depth_r2"] = float(r2_value)
            if disp_scale != 0.0 or disp_bias != 0.0:
                metrics["depth_disp_scale"] = float(disp_scale)
                metrics["depth_disp_bias"] = float(disp_bias)
            results.append(
                UncertaintyResult(
                    candidate=candidate,
                    mean_rgb=rgb[index],
                    variance_map=variance_map[index],
                    score=float(scores[index].item()),
                    unmasked_score=float(score_unmasked[index].item()),
                    metrics=metrics,
                    epistemic_map=epistemic_map[index],
                    aleatoric_map=aleatoric_map[index],
                    risk_exist_map=risk_exist_map[index],
                    gain_novel_map=gain_novel_map[index],
                    exist_mask=None if mask_tensor is None else mask_tensor[index],
                    novel_mask=None if novel_mask_tensor is None else novel_mask_tensor[index],
                    calibrated_depth=depth_calibrated,
                    mono_depth=None if mono_depths is None else mono_depths[index],
                )
            )
        self._log_results(results, batch_offset, global_step, label, log_mean_rgb=True)
        return results

    def _mc_dropout_pass(
        self,
        sparse_bundle: SparseViewBundle,
        candidates: Sequence[ViewCandidate],
        conditions: ConditionDict,
        batch_offset: int,
        global_step: Optional[int],
        num_samples: int,
        label: str,
    ) -> List[UncertaintyResult]:
        if self.mode == "gs_dropout":
            return self._primitive_dropout_pass(
                candidates,
                conditions,
                batch_offset=batch_offset,
                global_step=global_step,
                num_samples=num_samples,
                label=label,
            )
        if self.model is None:
            raise RuntimeError("GDDN model not initialised.")

        # Two-stage schedule
        if label == "coarse":
            resolution = int(self.config.coarse_resolution)
            steps = int(self.config.coarse_steps)
        else:
            resolution = int(self.config.fine_resolution)
            steps = int(self.config.fine_steps)

        sample_count = max(1, num_samples)
        mc_samples: List[torch.Tensor] = []

        # Stack candidate viewmats for this batch
        candidate_viewmats = torch.stack([c.viewmat for c in candidates]).to(self.device)
        rgb_render = conditions["rgb_render"].to(self.device)
        depth_render = conditions.get("depth_render", torch.zeros_like(rgb_render[:, :1]))
        normal_render = conditions.get("normal_render", torch.zeros_like(rgb_render))
        camera_intrinsics = conditions.get("camera_intrinsics")

        # Ensure conditioning matches the number of candidates in this batch
        assert rgb_render.shape[0] == candidate_viewmats.shape[0], "Condition/candidate batch mismatch"

        # ============================================================
        # Layer 2: 截断MC-Dropout参数
        # ============================================================
        truncation_enable = bool(getattr(self.config, "mc_truncation_enable", False))
        truncation_k = int(getattr(self.config, "mc_truncation_k", 0)) if truncation_enable else 0
        fix_noise = bool(getattr(self.config, "mc_fix_initial_noise", True))

        # 固定初始噪声z_T：确保MC样本间方差仅来自Dropout（认知不确定性）
        # 而非扩散过程的采样噪声，这是Pearson验证有效的数学前提
        fixed_latent: Optional[torch.Tensor] = None
        latent_h = max(1, rgb_render.shape[-2] // 8)
        latent_w = max(1, rgb_render.shape[-1] // 8)
        fixed_latent = torch.randn(
            candidate_viewmats.shape[0], 4, latent_h, latent_w,
            device=self.device, dtype=torch.float32,
        )

        if truncation_enable and global_step is not None and global_step % 100 == 0:
            print(f"[Layer2] 截断MC-Dropout: k={truncation_k}/{steps}, "
                  f"fix_noise={fix_noise}, N={sample_count}")

        for _ in range(sample_count):
            with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
                rgb = self.model.generate_view_correctly(
                    sparse_images=sparse_bundle.images,
                    sparse_poses=sparse_bundle.poses,
                    target_poses=candidate_viewmats,
                    rgb_render=rgb_render,
                    depth_render=depth_render,
                    normal_render=normal_render,
                    reference_depth_map=conditions.get("reference_depth_map"),
                    reference_intrinsics=conditions.get("reference_intrinsics"),
                    camera_intrinsics=camera_intrinsics,
                    steps=steps,
                    resolution=resolution,
                    enable_mc_dropout=True,
                    device=self.device,
                    sampler_type=self.config.mc_sampler_type,
                    per_step_noise_scale=0.0,
                    mc_dropout2d_p=self.config.mc_dropout2d_p,
                    mc_token_dropout_p=self.config.mc_token_dropout_p,
                    mc_cond_noise_sigma=self.config.mc_cond_noise_sigma,
                    mc_condition_alpha=self.config.mc_condition_alpha,
                    mc_latent_noise_std=self.config.mc_latent_noise_std,
                    # Layer 2 参数: 固定z_T确保方差来自Dropout
                    initial_latent=fixed_latent,
                )  # [B,3,res,res]
            mc_samples.append(rgb)

        stacked = torch.stack(mc_samples, dim=0)  # [T,B,3,H,W]
        mean_rgb = stacked.mean(dim=0)
        # Channel-wise variance aggregated over RGB
        variance_map = torch.nan_to_num(stacked.var(dim=0, unbiased=False).mean(dim=1))

        score_type = (self.config.score_type or "variance").lower()
        use_mask_scores = bool(self.config.use_mask_for_scores)
        eps = 1e-6

        predictive_var = torch.clamp(variance_map, min=eps)
        score_map = predictive_var
        mi_map: Optional[torch.Tensor] = None
        if score_type == "mi":
            conditional_floor = max(float(self.config.aleatoric_variance_floor), eps)
            conditional_map = conditions.get("ensemble_variance_map")
            if conditional_map is not None:
                if conditional_map.dim() == 4 and conditional_map.shape[1] == 1:
                    conditional_map = conditional_map.squeeze(1)
                if conditional_map.shape[-2:] != predictive_var.shape[-2:]:
                    conditional_map = F.interpolate(
                        conditional_map.unsqueeze(1),
                        size=predictive_var.shape[-2:],
                        mode="nearest",
                    ).squeeze(1)
                conditional_map = torch.nan_to_num(conditional_map, nan=conditional_floor).clamp_min(eps)
            else:
                conditional_map = torch.full_like(predictive_var, conditional_floor)
            mi_map = 0.5 * torch.log((predictive_var + eps) / (conditional_map + eps))
            mi_map = torch.clamp(mi_map, min=0.0)
            score_map = mi_map

        score_map = torch.nan_to_num(score_map)
        mask_tensor = self._align_depth_mask(
            conditions.get("depth_render"),
            variance_map.shape[-2:],
            variance_map.dtype,
        )
        novel_mask_tensor = None if mask_tensor is None else torch.clamp(1.0 - mask_tensor, min=0.0, max=1.0)
        (
            variance_masked,
            variance_unmasked,
            score_masked,
            score_unmasked,
            scores,
            mask_coverage,
            score_fallback,
        ) = self._reduce_score_maps(
            variance_map,
            score_map,
            mask_tensor,
            use_mask_scores=use_mask_scores,
        )
        results: List[UncertaintyResult] = []
        for index, candidate in enumerate(candidates):
            metrics: Dict[str, float] = {
                "variance_masked": float(variance_masked[index].item()),
                "variance_unmasked": float(variance_unmasked[index].item()),
                "mask_coverage": float(mask_coverage[index].item()),
                "score_fallback_to_unmasked": float(score_fallback[index].item()),
            }
            if mi_map is not None:
                metrics["mi_masked"] = float(score_masked[index].item())
                metrics["mi_unmasked"] = float(score_unmasked[index].item())

            # 添加数据不确定性估计（用于软门控机制）
            # MC-Dropout主要捕获认知不确定性（variance_map），
            # 数据不确定性通过方差的一个比例来近似估计
            aleatoric_estimate = float(variance_map[index].mean().item()) * 0.3
            metrics["aleatoric_mean"] = aleatoric_estimate

            results.append(
                UncertaintyResult(
                    candidate=candidate,
                    mean_rgb=mean_rgb[index],
                    variance_map=variance_map[index],
                    score=float(scores[index].item()),
                    unmasked_score=float(score_unmasked[index].item()),
                    metrics=metrics,
                    epistemic_map=variance_map[index],
                    aleatoric_map=None,
                    exist_mask=None if mask_tensor is None else mask_tensor[index],
                    novel_mask=None if novel_mask_tensor is None else novel_mask_tensor[index],
                    calibrated_depth=None,
                    mono_depth=None,
                )
            )
        self._log_results(results, batch_offset, global_step, label, log_mean_rgb=True)
        return results

    # ================================================================
    # Layer 2 算法2：两层MC采样（Law of Total Variance）
    # ================================================================
    def _two_layer_mc_pass(
        self,
        sparse_bundle: SparseViewBundle,
        candidates: Sequence[ViewCandidate],
        conditions: ConditionDict,
        batch_offset: int,
        global_step: Optional[int],
        label: str,
    ) -> List[UncertaintyResult]:
        """使用两层MC采样精确分离epistemic和aleatoric方差。

        外循环（T_outer次）：不同Dropout mask → 捕获epistemic方差
        内循环（S_inner次）：不同噪声种子z_T → 捕获aleatoric方差

        Law of Total Variance:
            Var_total = E[Var_inner] + Var[E_inner]
            Var_aleatoric = E[Var_inner]    （噪声种子间的方差均值）
            Var_epistemic = Var[E_inner]    （Dropout mask间的均值方差）
        """
        if self.model is None:
            raise RuntimeError("GDDN model not initialised.")

        if label == "coarse":
            resolution = int(self.config.coarse_resolution)
            steps = int(self.config.coarse_steps)
        else:
            resolution = int(self.config.fine_resolution)
            steps = int(self.config.fine_steps)

        t_outer = int(getattr(self.config, "mc_two_layer_outer", 4))
        s_inner = int(getattr(self.config, "mc_two_layer_inner", 2))
        truncation_enable = bool(getattr(self.config, "mc_truncation_enable", False))
        truncation_k = int(getattr(self.config, "mc_truncation_k", 0)) if truncation_enable else 0

        candidate_viewmats = torch.stack([c.viewmat for c in candidates]).to(self.device)
        rgb_render = conditions["rgb_render"].to(self.device)
        depth_render = conditions.get("depth_render", torch.zeros_like(rgb_render[:, :1]))
        normal_render = conditions.get("normal_render", torch.zeros_like(rgb_render))
        camera_intrinsics = conditions.get("camera_intrinsics")
        B = candidate_viewmats.shape[0]

        latent_h = max(1, rgb_render.shape[-2] // 8)
        latent_w = max(1, rgb_render.shape[-1] // 8)

        if global_step is not None and global_step % 100 == 0:
            print(f"[Layer2] 两层MC采样: T_outer={t_outer}, S_inner={s_inner}, "
                  f"truncation_k={truncation_k}/{steps}, total_inferences={t_outer * s_inner}")

        outer_means: List[torch.Tensor] = []   # 每个Dropout mask下的均值
        inner_vars: List[torch.Tensor] = []    # 每个Dropout mask下的aleatoric方差

        for _t in range(t_outer):
            inner_samples: List[torch.Tensor] = []
            for _s in range(s_inner):
                # 每次内循环使用不同噪声种子
                z_T_s = torch.randn(B, 4, latent_h, latent_w,
                                    device=self.device, dtype=torch.float32)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.config.use_mixed_precision):
                    rgb = self.model.generate_view_correctly(
                        sparse_images=sparse_bundle.images,
                        sparse_poses=sparse_bundle.poses,
                        target_poses=candidate_viewmats,
                        rgb_render=rgb_render,
                        depth_render=depth_render,
                        normal_render=normal_render,
                        reference_depth_map=conditions.get("reference_depth_map"),
                        reference_intrinsics=conditions.get("reference_intrinsics"),
                        camera_intrinsics=camera_intrinsics,
                        steps=steps,
                        resolution=resolution,
                        enable_mc_dropout=True,
                        device=self.device,
                        truncation_k=truncation_k,
                        fixed_latent=z_T_s,
                    )
                inner_samples.append(rgb)

            stacked_inner = torch.stack(inner_samples, dim=0)  # [S, B, 3, H, W]
            f_bar_t = stacked_inner.mean(dim=0)                # [B, 3, H, W]
            # 该mask下的aleatoric方差: channel-wise var → mean over channels
            sigma_a_t = stacked_inner.var(dim=0, unbiased=False).mean(dim=1)  # [B, H, W]
            outer_means.append(f_bar_t)
            inner_vars.append(sigma_a_t)

        # Law of Total Variance
        stacked_outer = torch.stack(outer_means, dim=0)     # [T, B, 3, H, W]
        mean_rgb = stacked_outer.mean(dim=0)                # [B, 3, H, W]
        # Var_epistemic = Dropout mask间的方差
        var_epistemic = torch.nan_to_num(
            stacked_outer.var(dim=0, unbiased=False).mean(dim=1)  # [B, H, W]
        )
        # Var_aleatoric = 噪声种子间的方差均值
        var_aleatoric = torch.nan_to_num(
            torch.stack(inner_vars, dim=0).mean(dim=0)   # [B, H, W]
        )

        # 使用epistemic方差作为score（几何缺失代理）
        variance_map = var_epistemic

        use_mask_scores = bool(self.config.use_mask_for_scores)
        score_map = torch.nan_to_num(variance_map)
        mask_tensor = self._align_depth_mask(
            conditions.get("depth_render"),
            variance_map.shape[-2:],
            variance_map.dtype,
        )
        novel_mask_tensor = None if mask_tensor is None else torch.clamp(1.0 - mask_tensor, min=0.0, max=1.0)
        (
            variance_masked,
            variance_unmasked,
            score_masked,
            score_unmasked,
            scores,
            mask_coverage,
            score_fallback,
        ) = self._reduce_score_maps(
            variance_map,
            score_map,
            mask_tensor,
            use_mask_scores=use_mask_scores,
        )

        results: List[UncertaintyResult] = []
        for index, candidate in enumerate(candidates):
            metrics: Dict[str, float] = {
                "variance_masked": float(variance_masked[index].item()),
                "variance_unmasked": float(variance_unmasked[index].item()),
                "epistemic_mean": float(var_epistemic[index].mean().item()),
                "aleatoric_mean": float(var_aleatoric[index].mean().item()),
                "two_layer_t_outer": float(t_outer),
                "two_layer_s_inner": float(s_inner),
                "mask_coverage": float(mask_coverage[index].item()),
                "score_fallback_to_unmasked": float(score_fallback[index].item()),
            }
            results.append(
                UncertaintyResult(
                    candidate=candidate,
                    mean_rgb=mean_rgb[index],
                    variance_map=variance_map[index],
                    score=float(scores[index].item()),
                    unmasked_score=float(score_unmasked[index].item()),
                    metrics=metrics,
                    epistemic_map=var_epistemic[index],
                    aleatoric_map=var_aleatoric[index],
                    exist_mask=None if mask_tensor is None else mask_tensor[index],
                    novel_mask=None if novel_mask_tensor is None else novel_mask_tensor[index],
                    calibrated_depth=None,
                    mono_depth=None,
                )
            )
        self._log_results(results, batch_offset, global_step, label, log_mean_rgb=True)
        return results

    def _primitive_dropout_pass(
        self,
        candidates: Sequence[ViewCandidate],
        conditions: ConditionDict,
        *,
        batch_offset: int,
        global_step: Optional[int],
        num_samples: int,
        label: str,
    ) -> List[UncertaintyResult]:
        if self.gaussians is None or self.rasterizer is None:
            raise RuntimeError("GS-Dropout requires Gaussian scene and rasterizer.")
        rgb_render = conditions["rgb_render"]
        height, width = rgb_render.shape[-2:]
        sample_count = max(1, num_samples)
        with torch.no_grad():
            base_means = self.gaussians.means.detach()
            base_quats = self.gaussians.quats.detach()
            base_scales = self.gaussians.scales.detach()
            base_opacities = self.gaussians.opacities.detach()
            base_colors = self.gaussians.colors.detach()

        dropout_prob = float(self.config.primitive_dropout_p)
        dropout_prob = min(max(dropout_prob, 0.0), 1.0)
        jitter_scale = float(self.config.opacity_scale_jitter)

        results: List[UncertaintyResult] = []
        for candidate in candidates:
            samples: List[torch.Tensor] = []
            for _ in range(sample_count):
                with torch.no_grad():
                    keep_mask = torch.bernoulli(
                        torch.full_like(base_opacities, 1.0 - dropout_prob)
                    )
                    sampled_opacity = base_opacities * keep_mask
                    if jitter_scale > 0.0:
                        jitter = torch.randn_like(sampled_opacity) * jitter_scale
                        sampled_opacity = torch.clamp(
                            sampled_opacity * (1.0 + jitter), min=0.0, max=1.0
                        )
                    # alpha=0 已剔除该基元，不能把物理尺度变成零。
                    sampled_scales = base_scales
                    render_colors, _, _ = self.rasterizer(
                        means=base_means,
                        quats=base_quats,
                        scales=sampled_scales,
                        opacities=sampled_opacity,
                        colors=base_colors,
                        viewmats=candidate.viewmat.unsqueeze(0),
                        Ks=candidate.intrinsics.unsqueeze(0),
                        width=width,
                        height=height,
                    )
                rgb = render_colors[0, ..., :3].permute(2, 0, 1).contiguous()
                samples.append(rgb)
            stacked = torch.stack(samples, dim=0)
            mean_rgb = stacked.mean(dim=0)

            if stacked.shape[0] == 1:
                variance_map = torch.zeros(
                    (height, width), device=mean_rgb.device, dtype=mean_rgb.dtype
                )
            else:
                variance_map = stacked.var(dim=0, unbiased=False).mean(dim=0)

            score = float(variance_map.mean().item())
            results.append(
                UncertaintyResult(
                    candidate=candidate,
                    mean_rgb=mean_rgb,
                    variance_map=variance_map,
                    score=score,
                )
            )
        self._log_results(
            results,
            batch_offset,
            global_step,
            label,
            log_mean_rgb=False,
        )
        return results

    def _log_results(
        self,
        results: Sequence[UncertaintyResult],
        batch_offset: int,
        global_step: Optional[int],
        label: str,
        log_mean_rgb: bool,
    ) -> None:
        if self.writer is None or global_step is None:
            return
        for local_index, result in enumerate(results):
            candidate_index = batch_offset + local_index
            variance_map = torch.nan_to_num(result.variance_map.detach())
            variance_min = torch.min(variance_map)
            variance_max = torch.max(variance_map)
            denom = torch.clamp(variance_max - variance_min, min=1e-6)
            normalized_variance = ((variance_map - variance_min) / denom).unsqueeze(0)
            cpu_variance = normalized_variance.cpu()
            self.writer.add_image(
                f"uncertainty/variance_map_{label}/candidate_{candidate_index}",
                cpu_variance,
                global_step,
            )
            self.writer.add_histogram(
                f"uncertainty/variance_hist_{label}/candidate_{candidate_index}",
                variance_map.flatten().cpu(),
                global_step,
            )
            # 记录 masked 与 unmasked 分数，方便对比
            self.writer.add_scalar(
                f"uncertainty/score_{label}/candidate_{candidate_index}",
                result.score,
                global_step,
            )
            self.writer.add_scalar(
                f"uncertainty/score_unmasked_{label}/candidate_{candidate_index}",
                result.unmasked_score,
                global_step,
            )
            if result.metrics:
                for metric_name, metric_value in result.metrics.items():
                    self.writer.add_scalar(
                        f"uncertainty/{metric_name}_{label}/candidate_{candidate_index}",
                        metric_value,
                        global_step,
                    )
            if log_mean_rgb:
                mean_rgb = result.mean_rgb.detach().clamp(0.0, 1.0).cpu()
                self.writer.add_image(
                    f"uncertainty/mean_rgb_{label}/candidate_{candidate_index}",
                    mean_rgb,
                    global_step,
                )
