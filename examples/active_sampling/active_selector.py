"""
Greedy active selection with diversity regularisation and quality gating.
"""

from __future__ import annotations

import math
import hashlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import torch

from .configs.active_sampling_config import ActiveSamplingConfig
from .utils import ViewCandidate, cosine_distance, normalize

if TYPE_CHECKING:
    from .uncertainty_estimator import UncertaintyResult
else:  # pragma: no cover - fallback for optional dependency
    UncertaintyResult = Any  # type: ignore


class ActiveSelector:
    """Select top-K candidate views based on uncertainty scores."""

    def __init__(self, config: ActiveSamplingConfig) -> None:
        self.config = config

    def select(
        self, results: Sequence[UncertaintyResult], num_select: int
    ) -> List[UncertaintyResult]:
        """
        Greedy selection with diversity constraint.
        """
        sorted_results = self._rank_results(results)
        selected: List[UncertaintyResult] = []
        for result in sorted_results:
            if len(selected) >= num_select:
                break
            if not self._passes_quality(result):
                continue
            if self._is_diverse(result.candidate, [sel.candidate for sel in selected]):
                selected.append(result)
        return selected[:num_select]

    def preselect(
        self, results: Sequence[UncertaintyResult], limit: int
    ) -> List[UncertaintyResult]:
        """Select top results after quality filtering for the fine stage."""
        if limit <= 0:
            return []
        filtered = [res for res in results if self._passes_quality(res)]
        if not filtered:
            filtered = list(results)
        ranked = self._rank_results(filtered)
        chosen = ranked[: min(limit, len(ranked))]
        return chosen

    def _rank_results(
        self, results: Sequence[UncertaintyResult]
    ) -> List[UncertaintyResult]:
        if bool(getattr(self.config, "ablation_random_selection", False)):
            return sorted(results, key=self._get_ablation_random_key)
        return sorted(results, key=self._get_selection_score, reverse=True)

    def _get_ablation_random_key(self, result: "UncertaintyResult") -> str:
        seed = int(getattr(self.config, "ablation_random_selection_seed", 0))
        position = getattr(result.candidate, "position", None)
        if isinstance(position, torch.Tensor):
            position_values = [
                f"{float(value):.6f}"
                for value in position.detach().cpu().reshape(-1).tolist()
            ]
        else:
            position_values = [str(position)]
        payload = f"{seed}|{'|'.join(position_values)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_selection_score(self, result: "UncertaintyResult") -> float:
        metrics = result.metrics or {}
        if bool(getattr(self.config, "selection_sort_by_verified_utility", False)):
            prioritized_score = metrics.get("probe_selection_priority")
            if prioritized_score is None:
                prioritized_score = metrics.get("probe_expected_verified_utility")
            if prioritized_score is not None:
                try:
                    prioritized_score = float(prioritized_score)
                except (TypeError, ValueError):
                    prioritized_score = None
            if prioritized_score is not None and math.isfinite(prioritized_score):
                return prioritized_score
        return float(result.score)

    def _get_effective_diversity_cosine_margin(self) -> float:
        return float(self.config.diversity_cosine_margin)

    def _get_diversity_threshold_metadata(self) -> Dict[str, Any]:
        effective_margin = self._get_effective_diversity_cosine_margin()
        return {
            "cosine_margin_raw_config": float(self.config.diversity_cosine_margin),
            "cosine_margin_effective": float(effective_margin),
            "min_angle_deg": float(self.config.min_selection_angle_deg),
            "cosine_margin_source": "config",
        }

    def _is_diverse(
        self, candidate: ViewCandidate, selected_candidates: Sequence[ViewCandidate]
    ) -> bool:
        diversity_audit = self._evaluate_diversity(
            candidate,
            [{"candidate": selected_candidate} for selected_candidate in selected_candidates],
        )
        return bool(diversity_audit["pass"])

    def _passes_quality(self, result: "UncertaintyResult") -> bool:
        quality_audit = self._evaluate_quality(result)
        reject_reason = quality_audit.get("reject_reason")
        if reject_reason == "quality_mean_variance":
            print(
                f"[QualityCheck] FAILED: mean_variance={quality_audit['mean_variance']:.6e} <= "
                f"{quality_audit['threshold_mean_variance_min']:.6e}"
            )
            return False
        if reject_reason == "quality_variation_coeff":
            print(
                f"[QualityCheck] FAILED: variation_coeff={quality_audit['variation_coeff']:.6f} < "
                f"threshold={quality_audit['threshold_variation_coeff_min']}"
            )
            return False
        if reject_reason == "quality_brightness":
            print(
                f"[QualityCheck] FAILED: brightness={quality_audit['brightness']:.4f} < "
                f"threshold={quality_audit['threshold_brightness_min']}"
            )
            return False
        if reject_reason == "quality_contrast":
            print(
                f"[QualityCheck] FAILED: contrast={quality_audit['contrast']:.6f} < "
                f"threshold={quality_audit['threshold_contrast_min']}"
            )
            return False
        if reject_reason == "quality_q_gen":
            print(
                f"[QualityCheck] FAILED: q_gen={quality_audit['q_gen']:.4f} < "
                f"threshold={quality_audit['threshold_q_gen_min']}"
            )
            return False
        if reject_reason == "quality_q_gain":
            print(
                f"[QualityCheck] FAILED: q_gain_for_gate={quality_audit['q_gain_for_gate']:.4f} < "
                f"threshold={quality_audit['threshold_q_gain_min']}"
            )
            return False
        if reject_reason == "quality_probe_verified_utility":
            print(
                "[QualityCheck] FAILED: probe_expected_verified_utility="
                f"{quality_audit['probe_expected_verified_utility']:.4f} < "
                f"threshold={quality_audit['threshold_verified_utility_min']}"
            )
            return False
        if reject_reason == "quality_probe_verified_coverage":
            print(
                "[QualityCheck] FAILED: probe_verified_evidence_coverage="
                f"{quality_audit['probe_verified_evidence_coverage']:.4f} < "
                f"threshold={quality_audit['threshold_probe_verified_coverage_min']}"
            )
            return False
        if reject_reason == "quality_mask_coverage_non_finite":
            print("[QualityCheck] FAILED: mask_coverage is not finite")
            return False
        return True

    def _evaluate_quality(self, result: "UncertaintyResult") -> Dict[str, Any]:
        metrics = result.metrics or {}
        variance_map = torch.nan_to_num(result.variance_map.detach())
        mean_variance = float(variance_map.mean().item())
        std_variance = float(variance_map.std(unbiased=False).item())
        variation_coeff = std_variance / max(mean_variance, 1e-6)
        mean_rgb = result.mean_rgb.detach().clamp(0.0, 1.0)
        brightness = float(mean_rgb.mean().item())
        contrast = float(mean_rgb.std(unbiased=False).item())
        q_gen = float(metrics.get("q_gen", 1.0))
        q_gain = metrics.get("q_gain")
        q_gain = 0.5 if q_gain is None else float(q_gain)
        q_gain_for_gate = metrics.get("q_gain_for_gate", q_gain)
        q_gain_for_gate = 0.5 if q_gain_for_gate is None else float(q_gain_for_gate)
        probe_expected_verified_utility = metrics.get("probe_expected_verified_utility")
        probe_expected_verified_utility = (
            None if probe_expected_verified_utility is None else float(probe_expected_verified_utility)
        )
        probe_verified_evidence_coverage = metrics.get("probe_verified_evidence_coverage")
        probe_verified_evidence_coverage = (
            None if probe_verified_evidence_coverage is None else float(probe_verified_evidence_coverage)
        )
        mask_coverage = float(metrics.get("mask_coverage", 1.0))
        reject_reason: Optional[str] = None
        if mean_variance <= 1e-6:
            reject_reason = "quality_mean_variance"
        elif variation_coeff < self.config.quality_variation_threshold:
            reject_reason = "quality_variation_coeff"
        elif brightness < self.config.quality_min_brightness:
            reject_reason = "quality_brightness"
        elif contrast < self.config.quality_min_contrast:
            reject_reason = "quality_contrast"
        elif (
            bool(getattr(self.config, "selection_use_q_gen_gate", False))
            and q_gen < self.config.selection_min_q_gen
        ):
            reject_reason = "quality_q_gen"
        elif (
            bool(getattr(self.config, "selection_use_q_gain_gate", False))
            and q_gain_for_gate < self.config.selection_min_q_gain
        ):
            reject_reason = "quality_q_gain"
        elif (
            bool(getattr(self.config, "selection_use_verified_utility_gate", False))
            and probe_expected_verified_utility is not None
            and probe_expected_verified_utility < float(getattr(self.config, "selection_min_verified_utility", 0.0))
        ):
            reject_reason = "quality_probe_verified_utility"
        elif (
            bool(getattr(self.config, "selection_use_verified_utility_gate", False))
            and probe_verified_evidence_coverage is not None
            and probe_verified_evidence_coverage < float(getattr(self.config, "selection_min_probe_verified_coverage", 0.0))
        ):
            reject_reason = "quality_probe_verified_coverage"
        elif not math.isfinite(mask_coverage):
            reject_reason = "quality_mask_coverage_non_finite"
        return {
            "pass": reject_reason is None,
            "reject_reason": reject_reason,
            "mean_variance": float(mean_variance),
            "std_variance": float(std_variance),
            "variation_coeff": float(variation_coeff),
            "brightness": float(brightness),
            "contrast": float(contrast),
            "q_gen": float(q_gen),
            "q_gain": float(q_gain),
            "q_gain_for_gate": float(q_gain_for_gate),
            "probe_expected_verified_utility": (
                -1.0 if probe_expected_verified_utility is None else float(probe_expected_verified_utility)
            ),
            "probe_verified_evidence_coverage": (
                -1.0 if probe_verified_evidence_coverage is None else float(probe_verified_evidence_coverage)
            ),
            "selection_score": float(self._get_selection_score(result)),
            "mask_coverage": float(mask_coverage),
            "threshold_mean_variance_min": 1e-6,
            "threshold_variation_coeff_min": float(self.config.quality_variation_threshold),
            "threshold_brightness_min": float(self.config.quality_min_brightness),
            "threshold_contrast_min": float(self.config.quality_min_contrast),
            "threshold_q_gen_min": float(self.config.selection_min_q_gen),
            "threshold_q_gain_min": float(self.config.selection_min_q_gain),
            "threshold_verified_utility_min": float(
                getattr(self.config, "selection_min_verified_utility", 0.0)
            ),
            "threshold_probe_verified_coverage_min": float(
                getattr(self.config, "selection_min_probe_verified_coverage", 0.0)
            ),
        }

    def _evaluate_diversity(
        self,
        candidate: ViewCandidate,
        selected_candidates: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        threshold_metadata = self._get_diversity_threshold_metadata()
        effective_cosine_margin = float(threshold_metadata["cosine_margin_effective"])
        if not selected_candidates:
            return {
                "pass": True,
                "reject_reason": None,
                "nearest_selected_candidate_index": None,
                "nearest_selected_rank_1based": None,
                "nearest_cosine_distance": None,
                "nearest_angle_deg": None,
                "threshold_cosine_margin_min": float(effective_cosine_margin),
                "threshold_min_angle_deg": float(self.config.min_selection_angle_deg),
                "threshold_cosine_margin_raw_config": float(threshold_metadata["cosine_margin_raw_config"]),
                "threshold_cosine_margin_source": str(threshold_metadata["cosine_margin_source"]),
            }

        candidate_direction = normalize(candidate.position)
        nearest_distance: Optional[float] = None
        nearest_angle_deg: Optional[float] = None
        nearest_selected_candidate_index: Optional[int] = None
        nearest_selected_rank: Optional[int] = None
        reject_reason: Optional[str] = None
        reject_selected_candidate_index: Optional[int] = None
        reject_selected_rank: Optional[int] = None
        reject_cosine_distance: Optional[float] = None
        reject_angle_deg: Optional[float] = None

        for selected_item in selected_candidates:
            selected_candidate = selected_item["candidate"]
            selected_direction = normalize(selected_candidate.position)
            cosine_distance_value = float(
                cosine_distance(candidate_direction, selected_direction).item()
            )
            similarity = 1.0 - cosine_distance_value
            similarity = max(-1.0, min(1.0, similarity))
            angle_deg = float(math.degrees(math.acos(similarity)))

            if nearest_distance is None or cosine_distance_value < nearest_distance:
                nearest_distance = cosine_distance_value
                nearest_angle_deg = angle_deg
                nearest_selected_candidate_index = int(
                    selected_item.get("candidate_index", -1)
                )
                nearest_selected_rank = int(selected_item.get("rank_1based", -1))

            if cosine_distance_value < effective_cosine_margin:
                reject_reason = "diversity_cosine_margin"
            elif angle_deg < self.config.min_selection_angle_deg:
                reject_reason = "diversity_min_angle_deg"

            if reject_reason is not None:
                reject_selected_candidate_index = int(
                    selected_item.get("candidate_index", -1)
                )
                reject_selected_rank = int(selected_item.get("rank_1based", -1))
                reject_cosine_distance = cosine_distance_value
                reject_angle_deg = angle_deg
                break

        return {
            "pass": reject_reason is None,
            "reject_reason": reject_reason,
            "nearest_selected_candidate_index": nearest_selected_candidate_index,
            "nearest_selected_rank_1based": nearest_selected_rank,
            "nearest_cosine_distance": nearest_distance,
            "nearest_angle_deg": nearest_angle_deg,
            "reject_selected_candidate_index": reject_selected_candidate_index,
            "reject_selected_rank_1based": reject_selected_rank,
            "reject_cosine_distance": reject_cosine_distance,
            "reject_angle_deg": reject_angle_deg,
            "threshold_cosine_margin_min": float(effective_cosine_margin),
            "threshold_min_angle_deg": float(self.config.min_selection_angle_deg),
            "threshold_cosine_margin_raw_config": float(threshold_metadata["cosine_margin_raw_config"]),
            "threshold_cosine_margin_source": str(threshold_metadata["cosine_margin_source"]),
        }
