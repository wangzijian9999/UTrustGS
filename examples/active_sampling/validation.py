"""
Validation utilities for MC-Dropout correlation.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import ViewCandidate

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .uncertainty_estimator import SparseViewBundle, UncertaintyResult

ConditionProvider = Callable[[Iterable[ViewCandidate]], Dict[str, torch.Tensor]]


def pearson_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Pearson correlation coefficient."""
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    numerator = (x_centered * y_centered).sum()
    denominator = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum()))
    if denominator.abs() < 1e-8:
        return 0.0
    return float((numerator / denominator).item())


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman rank correlation coefficient."""
    if x.numel() == 0 or y.numel() == 0:
        return 0.0
    x_ranks = torch.argsort(torch.argsort(x))
    y_ranks = torch.argsort(torch.argsort(y))
    return pearson_correlation(x_ranks.to(torch.float32), y_ranks.to(torch.float32))


def bootstrap_ensemble_scores(
    *,
    gddn_model: nn.Module,
    sparse_bundle: "SparseViewBundle",
    results_batch: Sequence["UncertaintyResult"],
    repeats: int,
    condition_provider: ConditionProvider,
    steps: int,
    resolution: int,
    device: torch.device,
    score_type: str = "variance",
    use_mask_for_scores: bool = True,
    aleatoric_variance_floor: float = 1e-4,
    sampler_type: str = "ddim",
    per_step_noise_scale: float = 0.0,
    mc_dropout2d_p: float = 0.1,
    mc_token_dropout_p: float = 0.08,
    mc_cond_noise_sigma: float = 0.02,
    mc_condition_alpha: float = 0.5,
    mc_latent_noise_std: float = 0.005,
    candidate_batch_size: int = 1,
) -> "tuple[List[float], float]":
    """
    Split-Half Reliability方差估计。

    做 2*repeats 次 MC-Dropout 采样（固定同一个z_T），
    分成两半各自计算per-candidate方差，
    返回 (full_scores, split_half_pearson)。

    split_half_pearson 是两半方差排名的相关系数，
    衡量MC-Dropout方差估计的内部一致性（信度）。
    """
    total_samples = max(4, 2 * repeats)  # 至少4次（每半2次）
    print(f"[Phase 2] Bootstrap Split-Half: {total_samples}次采样 × {len(results_batch)}候选")
    # 退化保护: 候选数<3或采样不足时无法计算有意义的相关性
    if total_samples < 4 or len(results_batch) < 3:
        for result in results_batch:
            if result.metrics is None:
                result.metrics = {}
            result.metrics.setdefault("ensemble_variance_masked", 0.0)
            result.metrics.setdefault("ensemble_variance_unmasked", 0.0)
        if len(results_batch) < 3:
            print(f"[Phase 2] Bootstrap跳过: 候选数={len(results_batch)}<3")
        return [0.0 for _ in results_batch], 0.0

    candidates = [result.candidate for result in results_batch]
    conditions = condition_provider(candidates)
    rgb_render = conditions["rgb_render"].to(device)
    depth_render = conditions.get("depth_render", torch.zeros_like(rgb_render[:, :1])).to(device)
    normal_render = conditions.get("normal_render", torch.zeros_like(rgb_render)).to(device)
    camera_intrinsics = conditions.get("camera_intrinsics")
    reference_depth_map = conditions.get("reference_depth_map")
    sparse_images = sparse_bundle.images.to(device)
    sparse_poses = sparse_bundle.poses.to(device)
    target_poses = torch.stack([candidate.viewmat for candidate in candidates]).to(device)

    # 固定z_T：所有2N次采样共享同一个初始噪声
    B = len(candidates)
    latent_h = max(1, rgb_render.shape[-2] // 8)
    latent_w = max(1, rgb_render.shape[-1] // 8)
    fixed_latent = torch.randn((B, 4, latent_h, latent_w), device=device)
    effective_candidate_batch_size = max(1, min(int(candidate_batch_size), B))
    print(f"[Phase 2] Bootstrap候选分块大小: {effective_candidate_batch_size}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target_height, target_width = rgb_render.shape[-2:]
    sum_full = torch.zeros((B, 3, target_height, target_width), dtype=torch.float32)
    sqsum_full = torch.zeros_like(sum_full)
    half = total_samples // 2
    sum_half_a = torch.zeros_like(sum_full)
    sqsum_half_a = torch.zeros_like(sum_full)
    sum_half_b = torch.zeros_like(sum_full)
    sqsum_half_b = torch.zeros_like(sum_full)

    def _slice_optional_tensor(optional_tensor: Optional[torch.Tensor], start: int, stop: int) -> Optional[torch.Tensor]:
        if optional_tensor is None:
            return None
        return optional_tensor[start:stop]

    def _accumulate_statistics(
        destination_sum: torch.Tensor,
        destination_sqsum: torch.Tensor,
        start: int,
        stop: int,
        sample_images: torch.Tensor,
    ) -> None:
        cpu_images = sample_images.detach().to(device="cpu", dtype=torch.float32)
        destination_sum[start:stop].add_(cpu_images)
        destination_sqsum[start:stop].add_(cpu_images.square())

    try:
        for _sample_idx in range(total_samples):
            # 简洁进度输出（每5次或最后一次）
            if (_sample_idx + 1) % 5 == 0 or _sample_idx == total_samples - 1:
                print(f"  [Bootstrap] 采样进度: {_sample_idx+1}/{total_samples}", flush=True)
            candidate_start_index = 0
            while candidate_start_index < B:
                current_chunk_size = min(effective_candidate_batch_size, B - candidate_start_index)
                while True:
                    candidate_stop_index = candidate_start_index + current_chunk_size
                    try:
                        chunk_images = gddn_model.generate_view_correctly(
                            sparse_images=sparse_images,
                            sparse_poses=sparse_poses,
                            target_poses=target_poses[candidate_start_index:candidate_stop_index],
                            rgb_render=rgb_render[candidate_start_index:candidate_stop_index],
                            depth_render=depth_render[candidate_start_index:candidate_stop_index],
                            normal_render=normal_render[candidate_start_index:candidate_stop_index],
                            reference_depth_map=_slice_optional_tensor(
                                reference_depth_map,
                                candidate_start_index,
                                candidate_stop_index,
                            ),
                            camera_intrinsics=_slice_optional_tensor(
                                camera_intrinsics,
                                candidate_start_index,
                                candidate_stop_index,
                            ),
                            steps=steps,
                            resolution=resolution,
                            enable_mc_dropout=True,
                            device=device,
                            sampler_type=sampler_type,
                            per_step_noise_scale=float(per_step_noise_scale),
                            mc_dropout2d_p=mc_dropout2d_p,
                            mc_token_dropout_p=mc_token_dropout_p,
                            mc_cond_noise_sigma=mc_cond_noise_sigma,
                            mc_condition_alpha=mc_condition_alpha,
                            mc_latent_noise_std=mc_latent_noise_std,
                            initial_latent=fixed_latent[candidate_start_index:candidate_stop_index],
                        )
                        break
                    except torch.OutOfMemoryError:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if current_chunk_size <= 1:
                            raise
                        reduced_chunk_size = max(1, current_chunk_size // 2)
                        print(
                            f"  [Bootstrap] CUDA OOM，候选分块从 {current_chunk_size} 降到 {reduced_chunk_size} 后重试"
                        )
                        current_chunk_size = reduced_chunk_size
                        effective_candidate_batch_size = min(effective_candidate_batch_size, current_chunk_size)

                if chunk_images.dim() == 3:
                    chunk_images = chunk_images.unsqueeze(0)
                if chunk_images.shape[-2:] != (target_height, target_width):
                    chunk_images = F.interpolate(
                        chunk_images,
                        size=(target_height, target_width),
                        mode="bilinear",
                        align_corners=False,
                    )

                _accumulate_statistics(
                    sum_full,
                    sqsum_full,
                    candidate_start_index,
                    candidate_stop_index,
                    chunk_images,
                )
                if _sample_idx < half:
                    _accumulate_statistics(
                        sum_half_a,
                        sqsum_half_a,
                        candidate_start_index,
                        candidate_stop_index,
                        chunk_images,
                    )
                else:
                    _accumulate_statistics(
                        sum_half_b,
                        sqsum_half_b,
                        candidate_start_index,
                        candidate_stop_index,
                        chunk_images,
                    )

                del chunk_images
                candidate_start_index = candidate_stop_index
    except torch.OutOfMemoryError as oom_exception:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[Phase 2] Bootstrap因CUDA OOM降级跳过: {oom_exception}")
        for result in results_batch:
            if result.metrics is None:
                result.metrics = {}
            result.metrics["ensemble_variance_masked"] = 0.0
            result.metrics["ensemble_variance_unmasked"] = 0.0
            result.metrics["split_half_pearson"] = 0.0
            result.metrics["split_half_spearman"] = 0.0
            result.metrics["bootstrap_oom_fallback"] = 1.0
        return [0.0 for _ in results_batch], 0.0

    variance_map_A = torch.nan_to_num(
        (sqsum_half_a / float(half)) - (sum_half_a / float(half)).pow(2)
    ).mean(dim=1)
    variance_map_B = torch.nan_to_num(
        (sqsum_half_b / float(half)) - (sum_half_b / float(half)).pow(2)
    ).mean(dim=1)
    variance_map = torch.nan_to_num(
        (sqsum_full / float(total_samples)) - (sum_full / float(total_samples)).pow(2)
    ).mean(dim=1)

    def _prepare_mask(depth: torch.Tensor, target_hw: torch.Size) -> torch.Tensor:
        # 处理多通道深度图
        if depth.dim() == 4 and depth.shape[1] > 1:
            depth = depth[:, :1, ...]
            
        mask = (depth > 1e-6).to(variance_map.dtype)
        if mask.shape[-2:] != tuple(target_hw):
            mask = F.interpolate(mask, size=target_hw, mode="nearest")
        return mask.squeeze(1)

    def _mean_with_mask(tensor: torch.Tensor, mask_tensor: Optional[torch.Tensor]) -> torch.Tensor:
        if mask_tensor is None:
            return tensor.mean(dim=(1, 2))
        masked_sum = (tensor * mask_tensor).sum(dim=(1, 2))
        denom = mask_tensor.sum(dim=(1, 2)).clamp_min(1.0)
        return masked_sum / denom

    mask_tensor: Optional[torch.Tensor]
    if depth_render is not None:
        mask_tensor = _prepare_mask(depth_render, variance_map.shape[-2:]).cpu()
    else:
        mask_tensor = None

    # P1修复: 冷启动死锁防护 — 当mask覆盖率过低时回退到unmasked
    min_mask_coverage = 0.05
    if mask_tensor is not None:
        coverage = mask_tensor.mean().item()
        if coverage < min_mask_coverage:
            mask_tensor = None  # 静默回退到unmasked

    # Per-candidate scores from each half and full
    scores_A = torch.nan_to_num(_mean_with_mask(variance_map_A, mask_tensor))
    scores_B = torch.nan_to_num(_mean_with_mask(variance_map_B, mask_tensor))
    variance_masked = _mean_with_mask(variance_map, mask_tensor)
    variance_unmasked = variance_map.mean(dim=(1, 2))

    # Split-Half退化检测: 如果A/B分数无差异(std≈0)，表示MC采样无区分力
    _scores_a_std = float(scores_A.std().item()) if scores_A.numel() > 1 else 0.0
    _scores_b_std = float(scores_B.std().item()) if scores_B.numel() > 1 else 0.0
    if _scores_a_std < 1e-8 and _scores_b_std < 1e-8:
        # 所有候选分数几乎相同 → MC采样无法区分 → reliability=0
        split_half_pearson_value = 0.0
        split_half_spearman_value = 0.0
        best_split_half = 0.0
    else:
        # 正常计算Split-Half Pearson相关
        split_half_pearson_value = pearson_correlation(scores_A, scores_B)
        split_half_spearman_value = spearman_correlation(scores_A, scores_B)
        # NaN保护: 如果任何一个correlation返回NaN，替换为0.0
        import math
        if math.isnan(split_half_pearson_value):
            split_half_pearson_value = 0.0
        if math.isnan(split_half_spearman_value):
            split_half_spearman_value = 0.0
        best_split_half = max(split_half_pearson_value, split_half_spearman_value)

    eps = 1e-6
    score_map = torch.clamp(variance_map, min=eps)
    score_type_lower = score_type.lower()
    if score_type_lower == "mi":
        conditional = torch.clamp(variance_map, min=eps)
        predictive_proxy = conditional + max(aleatoric_variance_floor, eps)
        score_map = 0.5 * torch.log((predictive_proxy + eps) / (conditional + eps))
        score_map = torch.clamp(score_map, min=0.0)

    score_map = torch.nan_to_num(score_map)
    score_masked = _mean_with_mask(score_map, mask_tensor)
    final_scores_tensor = score_masked if use_mask_for_scores else score_map.mean(dim=(1, 2))
    final_scores = [float(item.item()) for item in final_scores_tensor]

    for index, result in enumerate(results_batch):
        metrics = dict(result.metrics) if result.metrics else {}
        metrics["ensemble_variance_masked"] = float(variance_masked[index].item())
        metrics["ensemble_variance_unmasked"] = float(variance_unmasked[index].item())
        metrics["split_half_pearson"] = split_half_pearson_value
        metrics["split_half_spearman"] = split_half_spearman_value
        if mask_tensor is not None:
            denom = max(1.0, float(mask_tensor[index].numel()))
            coverage_val = mask_tensor[index].sum().item() / denom
            metrics["ensemble_mask_coverage"] = float(coverage_val)
        result.metrics = metrics

    print(f"  [Bootstrap] 完成: reliability={best_split_half:.4f} (pearson={split_half_pearson_value:.4f}, spearman={split_half_spearman_value:.4f})")
    return final_scores, best_split_half
