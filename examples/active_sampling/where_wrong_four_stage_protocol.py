#!/usr/bin/env python
"""Admissibility protocol for uncertainty-guided sparse-view 3DGS repair.

This module implements the executable logic chain:

WhereWrong -> WhereRepairable -> HowRepairable -> WhenTrust.

Trust outputs are blocked from online Gaussian refinement unless the loaded
generative prior is explicitly marked as calibrated.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


TensorDict = Dict[str, Any]


@dataclass
class FourStageConfig:
    """Conservative defaults for evidence admission."""

    wrong_threshold: float = 0.35
    repairable_threshold: float = 0.25
    trust_threshold: float = 0.50
    min_patch_pixels: int = 16
    max_patches: int = 8
    patch_padding: int = 0
    min_trust_coverage: float = 0.01
    fusion_mode: str = "max"
    trust_is_calibrated: bool = False
    rgb_trust_is_calibrated: Optional[bool] = None
    depth_trust_is_calibrated: Optional[bool] = None
    alpha_trust_is_calibrated: Optional[bool] = None
    gate0_status: Optional[str] = None
    current_iter: Optional[int] = None
    valid_until_iter: Optional[int] = None


@dataclass
class FourStagePatch:
    bbox_xyxy: Tuple[int, int, int, int]
    image: torch.Tensor
    mask: torch.Tensor
    score: float
    coverage: float
    source: str


@dataclass
class WhereWrongOutput:
    p_hole: torch.Tensor
    p_geo_misalign: torch.Tensor
    p_artifact: torch.Tensor
    p_wrong: torch.Tensor
    confirmed_type_map: torch.Tensor
    uncertain_wrong_map: torch.Tensor
    excluded_mask: torch.Tensor
    source: str
    fusion_mode: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WhereRepairableOutput:
    p_repairable_any: torch.Tensor
    p_repairable_hole: torch.Tensor
    p_repairable_geo_misalign: torch.Tensor
    p_repairable_artifact: torch.Tensor
    repair_evidence_mask: torch.Tensor
    repair_ambiguous_mask: torch.Tensor
    repair_excluded_mask: torch.Tensor
    source: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HowRepairableOutput:
    proposal_rgb: torch.Tensor
    proposal_mask: torch.Tensor
    proposal_source: str
    proposal_patches: List[FourStagePatch]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class WhenTrustOutput:
    p_rgb_trust: torch.Tensor
    p_depth_trust: torch.Tensor
    p_alpha_trust: torch.Tensor
    joint_trust_mask: torch.Tensor
    trust_ambiguous_mask: torch.Tensor
    trust_reject_mask: torch.Tensor
    phase3_rgb_weight: torch.Tensor
    phase3_depth_weight: torch.Tensor
    phase3_alpha_weight: torch.Tensor
    online_admission_allowed: bool
    calibration_state: str
    gate_reasons: List[str]
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class FourStageResult:
    where_wrong: WhereWrongOutput
    where_repairable: WhereRepairableOutput
    how_repairable: HowRepairableOutput
    when_trust: WhenTrustOutput


TARGET_RGB_ALIASES = (
    "render_rgb",
    "features.render_rgb",
    "debug_tensors.render_rgb",
)
WARP_RGB_ALIASES = (
    "warp_rgb",
    "warped_rgb",
    "features.warp_rgb",
    "debug_tensors.warped_rgb",
)
SUPPORT_RGB_ALIASES = (
    "support_composed_rgb",
    "support_composed_image",
    "predicted_support_composed_image",
    "features.support_composed_rgb",
    "debug_tensors.support_composed_image",
)
PROPOSAL_RGB_ALIASES = (
    "proposal_rgb",
    "proposal_raw_rgb",
    "predicted_proposal_image",
    "features.proposal_rgb",
    "debug_tensors.proposal_raw_rgb",
)


def _get_nested(payload: TensorDict, key_path: str) -> Any:
    value: Any = payload
    for key_name in key_path.split("."):
        if not isinstance(value, dict) or key_name not in value:
            return None
        value = value[key_name]
    return value


def _find_first(payload: TensorDict, aliases: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    for alias in aliases:
        value = _get_nested(payload, alias)
        if value is not None:
            return value, alias
    return None, None


def _as_tensor(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().float()
    if isinstance(value, (list, tuple)):
        return torch.as_tensor(value).detach().cpu().float()
    return None


def _as_rgb(value: Any) -> Optional[torch.Tensor]:
    tensor = _as_tensor(value)
    if tensor is None:
        return None
    while tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    if tensor.dim() != 3:
        return None
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] > 3:
        tensor = tensor[:3]
    elif tensor.shape[0] != 3:
        return None
    return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _as_map(
    value: Any,
    target_hw: Tuple[int, int],
    default_value: float = 0.0,
    *,
    clamp_unit_range: bool = True,
) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor is None:
        return torch.full((1, target_hw[0], target_hw[1]), float(default_value))
    while tensor.dim() > 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.dim() != 3:
        return torch.full((1, target_hw[0], target_hw[1]), float(default_value))
    if tensor.shape[0] != 1:
        tensor = tensor[:1]
    tensor = torch.nan_to_num(tensor.float(), nan=0.0, posinf=1.0, neginf=0.0)
    if tuple(tensor.shape[-2:]) != tuple(target_hw):
        tensor = F.interpolate(tensor.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)
    if clamp_unit_range:
        tensor = tensor.clamp(0.0, 1.0)
    return tensor


def _resize_rgb(rgb: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    if tuple(rgb.shape[-2:]) == tuple(target_hw):
        return rgb
    return F.interpolate(rgb.unsqueeze(0), size=target_hw, mode="bilinear", align_corners=False).squeeze(0)


def _normalise_map(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    value = torch.nan_to_num(value.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if mask is not None and float(mask.max().item()) > 1e-6:
        selected = value[mask > 1e-6]
        if selected.numel() > 0:
            low = torch.quantile(selected, 0.05)
            high = torch.quantile(selected, 0.95)
        else:
            low = value.min()
            high = value.max()
    else:
        low = value.min()
        high = value.max()
    scale = torch.clamp(high - low, min=1e-6)
    return ((value - low) / scale).clamp(0.0, 1.0)


def _binary_mask(value: torch.Tensor, threshold: float) -> torch.Tensor:
    return (value >= float(threshold)).float()


def _coverage(mask: torch.Tensor) -> float:
    if mask.numel() == 0:
        return 0.0
    return float(mask.float().mean().item())


def _first_rgb(payload: TensorDict, aliases: Sequence[str], target_hw: Optional[Tuple[int, int]] = None) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    value, source = _find_first(payload, aliases)
    rgb = _as_rgb(value)
    if rgb is not None and target_hw is not None:
        rgb = _resize_rgb(rgb, target_hw)
    return rgb, source


def _first_map(
    payload: TensorDict,
    aliases: Sequence[str],
    target_hw: Tuple[int, int],
    default_value: float = 0.0,
) -> torch.Tensor:
    value, _source = _find_first(payload, aliases)
    return _as_map(value, target_hw, default_value=default_value)


def _optional_map(
    payload: TensorDict,
    aliases: Sequence[str],
    target_hw: Tuple[int, int],
    *,
    clamp_unit_range: bool = True,
) -> Optional[torch.Tensor]:
    value, _source = _find_first(payload, aliases)
    if value is None:
        return None
    return _as_map(value, target_hw, default_value=0.0, clamp_unit_range=clamp_unit_range)


def _masked_map_stats(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[float, float, float]:
    value = torch.nan_to_num(value.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    selected = value
    if mask is not None and float(mask.detach().float().max().item()) > 1e-6:
        active = mask.detach().float() > 1e-6
        if bool(active.any()):
            selected = value[active]
    return (
        float(selected.min().item()),
        float(selected.mean().item()),
        float(selected.max().item()),
    )


def _is_neutral_constant_trust_map(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> bool:
    min_value, mean_value, max_value = _masked_map_stats(value, mask)
    dynamic_range = float(max_value - min_value)
    return dynamic_range <= 1e-6 and abs(mean_value - 0.5) <= 1e-3


def _is_constant_map(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> bool:
    min_value, _mean_value, max_value = _masked_map_stats(value, mask)
    return float(max_value - min_value) <= 1e-6


def _explicit_non_degenerate_trust_map(
    payload: TensorDict,
    aliases: Sequence[str],
    target_hw: Tuple[int, int],
    repair_mask: torch.Tensor,
) -> Tuple[Optional[torch.Tensor], Optional[str], bool]:
    value, source = _find_first(payload, aliases)
    if value is None:
        return None, source, False
    trust_map = _as_map(value, target_hw, default_value=0.0)
    if _is_neutral_constant_trust_map(trust_map, repair_mask):
        return None, source, True
    return trust_map * repair_mask, source, False


def _online_rgb_trust_fallback(
    *,
    proposal_rgb: torch.Tensor,
    warp_rgb: Optional[torch.Tensor],
    rgb_agreement: torch.Tensor,
    proposal_confidence: torch.Tensor,
    uncertainty: torch.Tensor,
    proposal_depth: Optional[torch.Tensor],
    warp_depth: Optional[torch.Tensor],
    repair_mask: torch.Tensor,
) -> torch.Tensor:
    if warp_rgb is None:
        fallback_score = rgb_agreement
    else:
        proposal_warp_delta = (proposal_rgb - warp_rgb).abs().mean(dim=0, keepdim=True)
        fallback_score = _normalise_map(proposal_warp_delta, mask=repair_mask)
        if proposal_depth is not None and warp_depth is not None:
            proposal_warp_depth_delta = (proposal_depth - warp_depth).abs()
            depth_consistency = (
                1.0 - _normalise_map(proposal_warp_depth_delta, mask=repair_mask)
            ).clamp(0.0, 1.0)
            fallback_score = (fallback_score * depth_consistency).clamp(0.0, 1.0)
    return (
        repair_mask
        * fallback_score
        * proposal_confidence
        * (1.0 - uncertainty).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)


def _online_depth_trust_fallback(
    *,
    proposal_depth: Optional[torch.Tensor],
    warp_depth: Optional[torch.Tensor],
    repair_mask: torch.Tensor,
) -> torch.Tensor:
    if warp_depth is None:
        return torch.zeros_like(repair_mask)
    if proposal_depth is None:
        return torch.zeros_like(repair_mask)
    proposal_warp_depth_delta = (proposal_depth - warp_depth).abs()
    return (
        repair_mask
        * (1.0 - _normalise_map(proposal_warp_depth_delta, mask=repair_mask)).clamp(0.0, 1.0)
    ).clamp(0.0, 1.0)


def _online_alpha_trust_fallback(
    *,
    proposal_alpha: Optional[torch.Tensor],
    repair_mask: torch.Tensor,
) -> torch.Tensor:
    if proposal_alpha is None or _is_constant_map(proposal_alpha, repair_mask):
        return torch.zeros_like(repair_mask)
    return proposal_alpha.clamp(0.0, 1.0) * repair_mask


def _extract_component_boxes(
    mask: torch.Tensor,
    *,
    min_pixels: int,
    max_components: int,
    padding: int,
) -> List[Tuple[int, int, int, int, int]]:
    mask_2d = (mask.detach().cpu().float()[0] > 1e-6)
    height, width = int(mask_2d.shape[0]), int(mask_2d.shape[1])
    visited = torch.zeros_like(mask_2d, dtype=torch.bool)
    boxes: List[Tuple[int, int, int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or not bool(mask_2d[y, x]):
                continue
            queue: deque[Tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            xs: List[int] = []
            ys: List[int] = []
            while queue:
                cy, cx = queue.popleft()
                ys.append(cy)
                xs.append(cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= height or nx < 0 or nx >= width:
                        continue
                    if visited[ny, nx] or not bool(mask_2d[ny, nx]):
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            area = len(xs)
            if area < int(min_pixels):
                continue
            x0 = max(min(xs) - int(padding), 0)
            y0 = max(min(ys) - int(padding), 0)
            x1 = min(max(xs) + 1 + int(padding), width)
            y1 = min(max(ys) + 1 + int(padding), height)
            boxes.append((x0, y0, x1, y1, area))
    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[: int(max_components)]


def _resolve_target_hw(payload: TensorDict) -> Tuple[int, int]:
    for aliases in (TARGET_RGB_ALIASES, WARP_RGB_ALIASES, SUPPORT_RGB_ALIASES, PROPOSAL_RGB_ALIASES):
        rgb, _source = _first_rgb(payload, aliases)
        if rgb is not None:
            return int(rgb.shape[-2]), int(rgb.shape[-1])
    raise ValueError("At least one RGB tensor is required to resolve protocol resolution.")


def run_where_wrong_stage(payload: TensorDict, config: Optional[FourStageConfig] = None) -> WhereWrongOutput:
    config = config or FourStageConfig()
    target_hw = _resolve_target_hw(payload)
    render_rgb, render_source = _first_rgb(payload, TARGET_RGB_ALIASES, target_hw=target_hw)
    warp_rgb, _warp_source = _first_rgb(payload, WARP_RGB_ALIASES, target_hw=target_hw)
    if render_rgb is None:
        if warp_rgb is None:
            raise ValueError("WhereWrong requires render_rgb or warped_rgb.")
        render_rgb = warp_rgb
        render_source = "warped_rgb_as_render_fallback"
    if warp_rgb is None:
        warp_rgb = render_rgb

    existing_hole = _optional_map(payload, ("where_wrong_hole", "P_hole", "where_wrong.p_hole"), target_hw)
    existing_geo = _optional_map(
        payload,
        ("where_wrong_geo_misalign", "P_geo_misalign", "where_wrong.p_geo_misalign"),
        target_hw,
    )
    existing_artifact = _optional_map(
        payload,
        ("where_wrong_artifact", "P_artifact", "where_wrong.p_artifact"),
        target_hw,
    )
    has_existing_typed = existing_hole is not None and existing_geo is not None and existing_artifact is not None

    if has_existing_typed:
        p_hole = existing_hole.clamp(0.0, 1.0)
        p_geo = existing_geo.clamp(0.0, 1.0)
        p_artifact = existing_artifact.clamp(0.0, 1.0)
        source = "provided_typed_where_wrong_maps"
    else:
        render_alpha = _first_map(payload, ("render_alpha", "alpha", "features.render_alpha", "debug_tensors.render_alpha"), target_hw, default_value=1.0)
        warp_confidence = _first_map(
            payload,
            ("warp_confidence", "warp_valid_mask", "supported_exist_mask", "features.warp_confidence", "debug_tensors.warp_valid_mask"),
            target_hw,
            default_value=0.0,
        )
        support_evidence = _first_map(
            payload,
            ("supported_exist_mask", "surface_evidence", "canonical_support", "support_mask", "features.surface_evidence", "features.canonical_support"),
            target_hw,
            default_value=0.0,
        )
        frontier = _first_map(payload, ("frontier_mask", "frontier", "features.frontier"), target_hw, default_value=0.0)
        depth_error = _first_map(
            payload,
            ("depth_consistency_error", "forward_backward_reprojection_error", "depth_mode_disagreement", "features.depth_consistency_error", "features.forward_backward_reprojection_error", "features.depth_mode_disagreement"),
            target_hw,
            default_value=0.0,
        )
        render_warp_l1 = (render_rgb - warp_rgb).abs().mean(dim=0, keepdim=True)
        appearance_evidence = _normalise_map(render_warp_l1, mask=torch.maximum(warp_confidence, support_evidence))
        geometry_evidence = _normalise_map(depth_error, mask=torch.maximum(warp_confidence, support_evidence))
        surface_or_support = torch.maximum(torch.maximum(support_evidence, warp_confidence), frontier)
        p_hole = ((1.0 - render_alpha).clamp(0.0, 1.0) * surface_or_support).clamp(0.0, 1.0)
        p_geo = (render_alpha * geometry_evidence * (0.5 + 0.5 * surface_or_support)).clamp(0.0, 1.0)
        p_artifact = (
            render_alpha
            * appearance_evidence
            * surface_or_support
            * (1.0 - 0.5 * geometry_evidence).clamp(0.0, 1.0)
        ).clamp(0.0, 1.0)
        source = f"observable_proxy_from_{render_source or 'render_rgb'}"

    if config.fusion_mode == "noisy_or_heuristic":
        p_wrong = (1.0 - (1.0 - p_hole) * (1.0 - p_geo) * (1.0 - p_artifact)).clamp(0.0, 1.0)
    elif config.fusion_mode == "max":
        p_wrong = torch.maximum(torch.maximum(p_hole, p_geo), p_artifact)
    else:
        raise ValueError(f"Unsupported where_wrong fusion_mode: {config.fusion_mode}")

    typed_stack = torch.cat([p_artifact, p_geo, p_hole], dim=0)
    max_typed, typed_index = typed_stack.max(dim=0, keepdim=True)
    confirmed = _binary_mask(max_typed, config.wrong_threshold)
    confirmed_type_map = typed_index.float() * confirmed - (1.0 - confirmed)
    uncertain_wrong_map = _binary_mask(p_wrong, config.wrong_threshold) * (1.0 - confirmed)
    excluded_mask = (p_wrong < float(config.wrong_threshold)).float()
    return WhereWrongOutput(
        p_hole=p_hole,
        p_geo_misalign=p_geo,
        p_artifact=p_artifact,
        p_wrong=p_wrong,
        confirmed_type_map=confirmed_type_map,
        uncertain_wrong_map=uncertain_wrong_map,
        excluded_mask=excluded_mask,
        source=source,
        fusion_mode=config.fusion_mode,
        metrics={
            "wrong_coverage": _coverage(_binary_mask(p_wrong, config.wrong_threshold)),
            "artifact_mean": float(p_artifact.mean().item()),
            "geo_misalign_mean": float(p_geo.mean().item()),
            "hole_mean": float(p_hole.mean().item()),
        },
    )


def run_where_repairable_stage(
    payload: TensorDict,
    where_wrong: WhereWrongOutput,
    config: Optional[FourStageConfig] = None,
) -> WhereRepairableOutput:
    config = config or FourStageConfig()
    target_hw = tuple(int(dim) for dim in where_wrong.p_wrong.shape[-2:])
    explicit_any = _optional_map(
        payload,
        ("P_repairable_any", "repairability_map", "where_repairable.p_repairable_any"),
        target_hw,
    )
    explicit_hole = _optional_map(
        payload,
        ("P_repairable_hole", "where_repairable.p_repairable_hole"),
        target_hw,
    )
    explicit_geo = _optional_map(
        payload,
        ("P_repairable_geo_misalign", "P_repairable_geo", "where_repairable.p_repairable_geo_misalign"),
        target_hw,
    )
    explicit_artifact = _optional_map(
        payload,
        ("P_repairable_artifact", "where_repairable.p_repairable_artifact"),
        target_hw,
    )
    has_explicit_typed = explicit_hole is not None and explicit_geo is not None and explicit_artifact is not None
    if explicit_any is not None or has_explicit_typed:
        if has_explicit_typed:
            p_repairable_hole = explicit_hole.clamp(0.0, 1.0)
            p_repairable_geo = explicit_geo.clamp(0.0, 1.0)
            p_repairable_artifact = explicit_artifact.clamp(0.0, 1.0)
            typed_any = torch.maximum(
                torch.maximum(p_repairable_hole, p_repairable_geo),
                p_repairable_artifact,
            )
            p_repairable_any = typed_any if explicit_any is None else torch.maximum(explicit_any.clamp(0.0, 1.0), typed_any)
        else:
            p_repairable_any = explicit_any.clamp(0.0, 1.0)
            p_repairable_hole = (p_repairable_any * where_wrong.p_hole).clamp(0.0, 1.0)
            p_repairable_geo = (p_repairable_any * where_wrong.p_geo_misalign).clamp(0.0, 1.0)
            p_repairable_artifact = (p_repairable_any * where_wrong.p_artifact).clamp(0.0, 1.0)
        repair_evidence_mask = _binary_mask(p_repairable_any, config.repairable_threshold)
        wrong_mask = _binary_mask(where_wrong.p_wrong, config.wrong_threshold)
        repair_ambiguous_mask = wrong_mask * (1.0 - repair_evidence_mask).clamp(0.0, 1.0) * (p_repairable_any > 0.0).float()
        repair_excluded_mask = (1.0 - repair_evidence_mask).clamp(0.0, 1.0) * (1.0 - repair_ambiguous_mask).clamp(0.0, 1.0)
        return WhereRepairableOutput(
            p_repairable_any=p_repairable_any,
            p_repairable_hole=p_repairable_hole,
            p_repairable_geo_misalign=p_repairable_geo,
            p_repairable_artifact=p_repairable_artifact,
            repair_evidence_mask=repair_evidence_mask,
            repair_ambiguous_mask=repair_ambiguous_mask,
            repair_excluded_mask=repair_excluded_mask,
            source="provided_repairability_maps",
            metrics={
                "repairable_coverage": _coverage(repair_evidence_mask),
                "repair_ambiguous_coverage": _coverage(repair_ambiguous_mask),
                "repairable_mean": float(p_repairable_any.mean().item()),
            },
        )
    support_evidence = _first_map(
        payload,
        ("supported_exist_mask", "surface_evidence", "canonical_support", "warp_confidence", "warp_valid_mask", "features.surface_evidence", "features.canonical_support", "features.warp_confidence"),
        target_hw,
        default_value=0.0,
    )
    frontier = _first_map(payload, ("frontier_mask", "frontier", "features.frontier"), target_hw, default_value=0.0)
    support_evidence = torch.maximum(support_evidence, frontier)
    depth_conflict = _first_map(
        payload,
        ("support_conflict", "fusion_rgb_disagreement", "depth_mode_disagreement", "depth_consistency_error", "features.fusion_rgb_disagreement", "features.depth_mode_disagreement", "features.depth_consistency_error"),
        target_hw,
        default_value=0.0,
    )
    depth_conflict = _normalise_map(depth_conflict, mask=support_evidence)
    evidence_strength = (support_evidence * (1.0 - 0.75 * depth_conflict).clamp(0.0, 1.0)).clamp(0.0, 1.0)

    p_repairable_hole = (where_wrong.p_hole * evidence_strength).clamp(0.0, 1.0)
    p_repairable_geo = (where_wrong.p_geo_misalign * evidence_strength).clamp(0.0, 1.0)
    p_repairable_artifact = (where_wrong.p_artifact * evidence_strength).clamp(0.0, 1.0)
    p_repairable_any = torch.maximum(
        torch.maximum(p_repairable_hole, p_repairable_geo),
        p_repairable_artifact,
    )
    repair_evidence_mask = _binary_mask(p_repairable_any, config.repairable_threshold)
    wrong_mask = _binary_mask(where_wrong.p_wrong, config.wrong_threshold)
    low_evidence = (evidence_strength < float(config.repairable_threshold)).float()
    high_conflict = (depth_conflict > 0.75).float()
    repair_ambiguous_mask = wrong_mask * torch.maximum(low_evidence, high_conflict) * (1.0 - repair_evidence_mask)
    repair_excluded_mask = (1.0 - repair_evidence_mask).clamp(0.0, 1.0) * (1.0 - repair_ambiguous_mask).clamp(0.0, 1.0)
    return WhereRepairableOutput(
        p_repairable_any=p_repairable_any,
        p_repairable_hole=p_repairable_hole,
        p_repairable_geo_misalign=p_repairable_geo,
        p_repairable_artifact=p_repairable_artifact,
        repair_evidence_mask=repair_evidence_mask,
        repair_ambiguous_mask=repair_ambiguous_mask,
        repair_excluded_mask=repair_excluded_mask,
        source="round0_non_cyclic_evidence_proxy",
        metrics={
            "repairable_coverage": _coverage(repair_evidence_mask),
            "repair_ambiguous_coverage": _coverage(repair_ambiguous_mask),
            "repairable_mean": float(p_repairable_any.mean().item()),
        },
    )


def run_how_repairable_stage(
    payload: TensorDict,
    where_repairable: WhereRepairableOutput,
    config: Optional[FourStageConfig] = None,
) -> HowRepairableOutput:
    config = config or FourStageConfig()
    target_hw = tuple(int(dim) for dim in where_repairable.repair_evidence_mask.shape[-2:])
    proposal_rgb, proposal_source = _first_rgb(payload, PROPOSAL_RGB_ALIASES, target_hw=target_hw)
    if proposal_rgb is None:
        proposal_rgb, proposal_source = _first_rgb(payload, SUPPORT_RGB_ALIASES, target_hw=target_hw)
    if proposal_rgb is None:
        proposal_rgb, proposal_source = _first_rgb(payload, WARP_RGB_ALIASES, target_hw=target_hw)
    if proposal_rgb is None:
        proposal_rgb, proposal_source = _first_rgb(payload, TARGET_RGB_ALIASES, target_hw=target_hw)
    if proposal_rgb is None:
        raise ValueError("HowRepairable requires proposal, support-composed, warp, or render RGB.")
    proposal_source = proposal_source or "unknown_rgb_source"

    proposal_mask = where_repairable.repair_evidence_mask.detach().cpu().float().clamp(0.0, 1.0)
    boxes = _extract_component_boxes(
        proposal_mask,
        min_pixels=config.min_patch_pixels,
        max_components=config.max_patches,
        padding=config.patch_padding,
    )
    patches: List[FourStagePatch] = []
    total_pixels = float(target_hw[0] * target_hw[1])
    for x0, y0, x1, y1, area_pixels in boxes:
        patch_mask = proposal_mask[:, y0:y1, x0:x1]
        patch_image = proposal_rgb[:, y0:y1, x0:x1]
        patch_score = float(area_pixels) / max(total_pixels, 1.0)
        patches.append(
            FourStagePatch(
                bbox_xyxy=(x0, y0, x1, y1),
                image=patch_image.detach().cpu(),
                mask=patch_mask.detach().cpu(),
                score=patch_score,
                coverage=_coverage(patch_mask),
                source=proposal_source,
            )
        )
    return HowRepairableOutput(
        proposal_rgb=proposal_rgb.detach().cpu(),
        proposal_mask=proposal_mask,
        proposal_source=proposal_source,
        proposal_patches=patches,
        metrics={
            "proposal_patch_count": float(len(patches)),
            "proposal_mask_coverage": _coverage(proposal_mask),
            "proposal_patch_area_ratio": float(sum(patch.score for patch in patches)),
        },
    )


def _cache_is_valid(payload: TensorDict, config: FourStageConfig) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    current_iter = config.current_iter
    valid_until_iter = config.valid_until_iter
    payload_current = _get_nested(payload, "current_iter")
    payload_valid_until = _get_nested(payload, "valid_until_iter")
    if current_iter is None and payload_current is not None:
        current_iter = int(payload_current)
    if valid_until_iter is None and payload_valid_until is not None:
        valid_until_iter = int(payload_valid_until)
    if current_iter is not None and valid_until_iter is not None and current_iter > valid_until_iter:
        reasons.append(f"stale cache: current_iter={current_iter} > valid_until_iter={valid_until_iter}")
        return False, reasons
    cache_valid_map = _get_nested(payload, "cache_valid")
    if isinstance(cache_valid_map, bool) and not cache_valid_map:
        reasons.append("cache_valid=False")
        return False, reasons
    return True, reasons


def run_when_trust_stage(
    payload: TensorDict,
    where_repairable: WhereRepairableOutput,
    how_repairable: HowRepairableOutput,
    config: Optional[FourStageConfig] = None,
) -> WhenTrustOutput:
    config = config or FourStageConfig()
    target_hw = tuple(int(dim) for dim in how_repairable.proposal_mask.shape[-2:])
    render_rgb, _render_source = _first_rgb(payload, TARGET_RGB_ALIASES, target_hw=target_hw)
    support_rgb, _support_source = _first_rgb(payload, SUPPORT_RGB_ALIASES, target_hw=target_hw)
    warp_rgb, _warp_source = _first_rgb(payload, WARP_RGB_ALIASES, target_hw=target_hw)
    reference_rgb = support_rgb if support_rgb is not None else warp_rgb

    repair_mask = where_repairable.repair_evidence_mask.float()
    p_rgb_trust, explicit_rgb_trust_source, rgb_neutral_constant_ignored = _explicit_non_degenerate_trust_map(
        payload,
        ("P_rgb_trust", "rgb_trust_map", "failure_rgb_trust_proxy"),
        target_hw,
        repair_mask,
    )
    if reference_rgb is not None:
        proposal_reference_l1 = (how_repairable.proposal_rgb - reference_rgb).abs().mean(dim=0, keepdim=True)
        rgb_agreement = (1.0 - _normalise_map(proposal_reference_l1, mask=repair_mask)).clamp(0.0, 1.0)
    elif render_rgb is not None:
        proposal_render_l1 = (how_repairable.proposal_rgb - render_rgb).abs().mean(dim=0, keepdim=True)
        rgb_agreement = (1.0 - _normalise_map(proposal_render_l1, mask=repair_mask)).clamp(0.0, 1.0)
    else:
        rgb_agreement = torch.zeros_like(repair_mask)

    proposal_confidence = _first_map(
        payload,
        (
            "proposal_acceptance_confidence",
            "proposal_verification_confidence",
            "proposal_confidence",
            "debug_tensors.proposal_acceptance_confidence",
        ),
        target_hw,
        default_value=1.0,
    )
    uncertainty = _first_map(
        payload,
        ("proposal_uncertainty", "epistemic_map", "aleatoric_map", "uncertainty"),
        target_hw,
        default_value=0.0,
    )
    proposal_depth = _optional_map(
        payload,
        ("proposal_depth", "features.proposal_depth", "debug_tensors.proposal_depth"),
        target_hw,
        clamp_unit_range=False,
    )
    warp_depth = _optional_map(
        payload,
        ("warp_depth", "render_depth", "features.render_depth", "debug_tensors.render_depth"),
        target_hw,
        clamp_unit_range=False,
    )
    proposal_alpha = _optional_map(
        payload,
        ("proposal_alpha", "features.proposal_alpha", "debug_tensors.proposal_alpha"),
        target_hw,
    )
    if p_rgb_trust is None:
        p_rgb_trust = _online_rgb_trust_fallback(
            proposal_rgb=how_repairable.proposal_rgb,
            warp_rgb=warp_rgb,
            rgb_agreement=rgb_agreement,
            proposal_confidence=proposal_confidence,
            uncertainty=uncertainty,
            proposal_depth=proposal_depth,
            warp_depth=warp_depth,
            repair_mask=repair_mask,
        )
        rgb_trust_source = "online_rgb_delta_depth_consistency_fallback"
    else:
        rgb_trust_source = f"explicit:{explicit_rgb_trust_source or 'unknown'}"

    p_depth_trust, explicit_depth_trust_source, depth_neutral_constant_ignored = _explicit_non_degenerate_trust_map(
        payload,
        ("P_depth_trust", "depth_trust_map", "failure_depth_trust_proxy"),
        target_hw,
        repair_mask,
    )
    if p_depth_trust is None:
        p_depth_trust = _online_depth_trust_fallback(
            proposal_depth=proposal_depth,
            warp_depth=warp_depth,
            repair_mask=repair_mask,
        )
        depth_trust_source = "online_depth_proposal_warp_consistency_fallback"
    else:
        depth_trust_source = f"explicit:{explicit_depth_trust_source or 'unknown'}"

    p_alpha_trust, explicit_alpha_trust_source, alpha_neutral_constant_ignored = _explicit_non_degenerate_trust_map(
        payload,
        ("P_alpha_trust", "alpha_trust_map", "alpha_confidence"),
        target_hw,
        repair_mask,
    )
    if p_alpha_trust is None:
        p_alpha_trust = _online_alpha_trust_fallback(
            proposal_alpha=proposal_alpha,
            repair_mask=repair_mask,
        )
        alpha_trust_source = "online_alpha_proposal_fallback"
    else:
        alpha_trust_source = f"explicit:{explicit_alpha_trust_source or 'unknown'}"

    joint_trust_mask = _binary_mask(p_rgb_trust, config.trust_threshold) * repair_mask
    trust_ambiguous_mask = repair_mask * (1.0 - joint_trust_mask) * (p_rgb_trust > 0.0).float()
    trust_reject_mask = (1.0 - joint_trust_mask).clamp(0.0, 1.0) * (1.0 - trust_ambiguous_mask).clamp(0.0, 1.0)

    gate_reasons: List[str] = []
    gate0_status = config.gate0_status or _get_nested(payload, "gate0_status")
    if gate0_status not in ("go", "conditional_go"):
        gate_reasons.append(f"Gate0 status is not admissible: {gate0_status}")
    rgb_trust_is_calibrated = (
        bool(config.trust_is_calibrated)
        if config.rgb_trust_is_calibrated is None
        else bool(config.rgb_trust_is_calibrated)
    )
    depth_trust_is_calibrated = (
        bool(config.trust_is_calibrated)
        if config.depth_trust_is_calibrated is None
        else bool(config.depth_trust_is_calibrated)
    )
    alpha_trust_is_calibrated = (
        bool(config.trust_is_calibrated)
        if config.alpha_trust_is_calibrated is None
        else bool(config.alpha_trust_is_calibrated)
    )
    missing_calibrations = [
        trust_name
        for trust_name, trust_is_calibrated in (
            ("rgb", rgb_trust_is_calibrated),
            ("depth", depth_trust_is_calibrated),
            ("alpha", alpha_trust_is_calibrated),
        )
        if not trust_is_calibrated
    ]
    if missing_calibrations:
        gate_reasons.append(
            "trust maps are proxy-only or not calibrated: "
            + ",".join(missing_calibrations)
        )
    cache_valid, cache_reasons = _cache_is_valid(payload, config)
    gate_reasons.extend(cache_reasons)
    trust_coverage = _coverage(joint_trust_mask)
    if trust_coverage < float(config.min_trust_coverage):
        gate_reasons.append(
            f"trust coverage below threshold: {trust_coverage:.6f} < {config.min_trust_coverage:.6f}"
        )
    all_trust_is_calibrated = (
        rgb_trust_is_calibrated
        and depth_trust_is_calibrated
        and alpha_trust_is_calibrated
    )
    online_allowed = len(gate_reasons) == 0 and cache_valid and all_trust_is_calibrated
    if online_allowed:
        phase3_rgb_weight = p_rgb_trust * joint_trust_mask
        phase3_depth_weight = p_depth_trust * _binary_mask(p_depth_trust, config.trust_threshold)
        phase3_alpha_weight = p_alpha_trust * _binary_mask(p_alpha_trust, config.trust_threshold)
    else:
        phase3_rgb_weight = torch.zeros_like(p_rgb_trust)
        phase3_depth_weight = torch.zeros_like(p_depth_trust)
        phase3_alpha_weight = torch.zeros_like(p_alpha_trust)

    return WhenTrustOutput(
        p_rgb_trust=p_rgb_trust,
        p_depth_trust=p_depth_trust,
        p_alpha_trust=p_alpha_trust,
        joint_trust_mask=joint_trust_mask,
        trust_ambiguous_mask=trust_ambiguous_mask,
        trust_reject_mask=trust_reject_mask,
        phase3_rgb_weight=phase3_rgb_weight,
        phase3_depth_weight=phase3_depth_weight,
        phase3_alpha_weight=phase3_alpha_weight,
        online_admission_allowed=bool(online_allowed),
        calibration_state="calibrated" if all_trust_is_calibrated else "proxy_only",
        gate_reasons=gate_reasons,
        metrics={
            "rgb_trust_mean": float(p_rgb_trust.mean().item()),
            "depth_trust_mean": float(p_depth_trust.mean().item()),
            "alpha_trust_mean": float(p_alpha_trust.mean().item()),
            "rgb_trust_source": rgb_trust_source,
            "depth_trust_source": depth_trust_source,
            "alpha_trust_source": alpha_trust_source,
            "rgb_neutral_constant_ignored": float(rgb_neutral_constant_ignored),
            "depth_neutral_constant_ignored": float(depth_neutral_constant_ignored),
            "alpha_neutral_constant_ignored": float(alpha_neutral_constant_ignored),
            "rgb_trust_dynamic_range": float(_masked_map_stats(p_rgb_trust, repair_mask)[2] - _masked_map_stats(p_rgb_trust, repair_mask)[0]),
            "depth_trust_dynamic_range": float(_masked_map_stats(p_depth_trust, repair_mask)[2] - _masked_map_stats(p_depth_trust, repair_mask)[0]),
            "alpha_trust_dynamic_range": float(_masked_map_stats(p_alpha_trust, repair_mask)[2] - _masked_map_stats(p_alpha_trust, repair_mask)[0]),
            "rgb_trust_is_calibrated": float(rgb_trust_is_calibrated),
            "depth_trust_is_calibrated": float(depth_trust_is_calibrated),
            "alpha_trust_is_calibrated": float(alpha_trust_is_calibrated),
            "joint_trust_coverage": trust_coverage,
        },
    )


def run_four_stage_protocol(payload: TensorDict, config: Optional[FourStageConfig] = None) -> FourStageResult:
    """Run PLAN6 four-stage logic on a tensor/debug bundle."""

    config = config or FourStageConfig()
    where_wrong = run_where_wrong_stage(payload, config)
    where_repairable = run_where_repairable_stage(payload, where_wrong, config)
    how_repairable = run_how_repairable_stage(payload, where_repairable, config)
    when_trust = run_when_trust_stage(payload, where_repairable, how_repairable, config)
    return FourStageResult(
        where_wrong=where_wrong,
        where_repairable=where_repairable,
        how_repairable=how_repairable,
        when_trust=when_trust,
    )
