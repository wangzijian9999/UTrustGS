#!/usr/bin/env python
"""Aggregate UTrustGS ablation metrics into CSV, LaTeX, and JSON tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ABLATION_ROOT = REPO_ROOT / "ablation"
DEFAULT_VARIANTS = (
    "full",
    "no_active_selection",
    "no_verified_evidence",
    "no_trust_weights",
    "rgb_only",
    "full_image_pseudo",
    "pseudo_densify_stats",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _split_csv(raw_value: str) -> List[str]:
    return [item.strip() for item in str(raw_value).split(",") if item.strip()]


def _split_csv_set(raw_value: Optional[str]) -> Optional[Set[str]]:
    values = set(_split_csv(raw_value or ""))
    return values or None


def _format_float(value: Optional[float], digits: int = 3) -> str:
    if value is None or not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def _mean_std(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None, None
    standard_deviation = float(pstdev(finite_values)) if len(finite_values) > 1 else 0.0
    return float(mean(finite_values)), standard_deviation


def _collect_failed_counts(
    ablation_root: Path,
    scene_filter: Optional[Set[str]] = None,
    seed_filter: Optional[Set[str]] = None,
) -> Dict[str, int]:
    failure_counts: Dict[str, int] = {}
    counted_runs: Set[str] = set()
    for failed_path in sorted((ablation_root / "reports").glob("failed_runs*.json")):
        try:
            failures = _read_json(failed_path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(failures, list):
            continue
        for failure in failures:
            if not isinstance(failure, dict):
                continue
            run_dir = failure.get("run_dir")
            if run_dir is not None and Path(str(run_dir), "eval", "metrics.json").exists():
                continue
            scene = str(failure.get("scene", ""))
            seed = str(failure.get("seed", ""))
            if scene_filter is not None and scene not in scene_filter:
                continue
            if seed_filter is not None and seed not in seed_filter:
                continue
            variant = str(failure.get("variant", "unknown"))
            run_key = str(run_dir) if run_dir is not None else f"{scene}/{variant}/{seed}"
            if run_key in counted_runs:
                continue
            counted_runs.add(run_key)
            failure_counts[variant] = failure_counts.get(variant, 0) + 1
    return failure_counts


def _iter_metrics(
    ablation_root: Path,
    variant: str,
    scene_filter: Optional[Set[str]] = None,
    seed_filter: Optional[Set[str]] = None,
) -> Iterable[Dict[str, Any]]:
    runs_root = ablation_root / "runs" / "llff"
    if not runs_root.exists():
        return
    for metrics_path in sorted(runs_root.glob(f"*/{variant}/seed_*/eval/metrics.json")):
        scene_name = metrics_path.parents[3].name
        seed_name = metrics_path.parents[1].name.removeprefix("seed_")
        if scene_filter is not None and scene_name not in scene_filter:
            continue
        if seed_filter is not None and seed_name not in seed_filter:
            continue
        try:
            payload = _read_json(metrics_path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("mean"), dict):
            payload["_metrics_path"] = str(metrics_path)
            yield payload


def _accepted_patch_count(metrics_payload: Dict[str, Any]) -> float:
    value = metrics_payload.get("ablation", {}).get("accepted_patch_count")
    try:
        if value is not None and math.isfinite(float(value)):
            return float(value)
    except (TypeError, ValueError):
        pass

    metrics_path = metrics_payload.get("_metrics_path")
    if metrics_path is None:
        return float("nan")
    diagnostic_dir = (
        Path(str(metrics_path)).parents[1]
        / "logs"
        / "phase2_pseudo_views"
        / "generation_diagnostics"
    )
    accepted_patch_count = 0
    accepted_candidate_found = False
    for diagnostic_path in sorted(diagnostic_dir.glob("round_*.json")):
        try:
            diagnostic_payload = _read_json(diagnostic_path)
        except (OSError, json.JSONDecodeError):
            continue
        for candidate in diagnostic_payload.get("candidates", []):
            if not bool(candidate.get("accepted", False)):
                continue
            accepted_candidate_found = True
            try:
                accepted_patch_count += int(candidate.get("proposal_patch_count", 0) or 0)
            except (TypeError, ValueError):
                continue
    return float(accepted_patch_count) if accepted_candidate_found else float("nan")


def _build_component_rows(
    ablation_root: Path,
    variants: Sequence[str],
    scene_filter: Optional[Set[str]] = None,
    seed_filter: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    failure_counts = _collect_failed_counts(
        ablation_root,
        scene_filter=scene_filter,
        seed_filter=seed_filter,
    )
    rows: List[Dict[str, Any]] = []
    for variant in variants:
        metrics_list = list(
            _iter_metrics(
                ablation_root,
                variant,
                scene_filter=scene_filter,
                seed_filter=seed_filter,
            )
        )
        metric_specs = {
            "PSNR": [item["mean"].get("psnr", float("nan")) for item in metrics_list],
            "SSIM": [item["mean"].get("ssim", float("nan")) for item in metrics_list],
            "LPIPS": [item["mean"].get("lpips", float("nan")) for item in metrics_list],
            "Verified_Cov": [
                item.get("ablation", {}).get("verified_evidence_coverage", float("nan"))
                for item in metrics_list
            ],
            "Pseudo": [
                item.get("ablation", {}).get("pseudo_view_count", float("nan"))
                for item in metrics_list
            ],
            "Accepted_Patch": [_accepted_patch_count(item) for item in metrics_list],
        }
        row: Dict[str, Any] = {"Variant": variant, "N": len(metrics_list)}
        for metric_name, metric_values in metric_specs.items():
            metric_mean, metric_std = _mean_std(metric_values)
            row[f"{metric_name}_mean"] = metric_mean
            row[f"{metric_name}_std"] = metric_std

        failure_count = int(failure_counts.get(variant, 0))
        if metrics_list and failure_count == 0:
            row["Failure Note"] = "-"
        elif metrics_list:
            row["Failure Note"] = f"{failure_count} failed"
        else:
            row["Failure Note"] = (
                "no completed runs" if failure_count == 0 else f"{failure_count} failed"
            )
        rows.append(row)
    return rows


def _write_component_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Variant",
        "PSNR_mean",
        "PSNR_std",
        "SSIM_mean",
        "SSIM_std",
        "LPIPS_mean",
        "LPIPS_std",
        "Verified_Cov_mean",
        "Verified_Cov_std",
        "Pseudo_mean",
        "Pseudo_std",
        "Accepted_Patch_mean",
        "Accepted_Patch_std",
        "N",
        "Failure Note",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def _write_component_tex(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\\begin{tabular}{lcccccccl}",
        "\\toprule",
        "Variant & PSNR$\\uparrow$ & SSIM$\\uparrow$ & LPIPS$\\downarrow$ & Verified Cov.$\\uparrow$ & \\#Pseudo & \\#Patch & N & Failure Note \\\\",
        "\\midrule",
    ]
    for row in rows:
        psnr = f"{_format_float(row['PSNR_mean'])}" + "$\\pm$" + f"{_format_float(row['PSNR_std'])}"
        ssim = f"{_format_float(row['SSIM_mean'], 4)}" + "$\\pm$" + f"{_format_float(row['SSIM_std'], 4)}"
        lpips = f"{_format_float(row['LPIPS_mean'], 4)}" + "$\\pm$" + f"{_format_float(row['LPIPS_std'], 4)}"
        coverage = f"{_format_float(row['Verified_Cov_mean'])}" + "$\\pm$" + f"{_format_float(row['Verified_Cov_std'])}"
        pseudo = f"{_format_float(row['Pseudo_mean'], 2)}" + "$\\pm$" + f"{_format_float(row['Pseudo_std'], 2)}"
        patch = f"{_format_float(row['Accepted_Patch_mean'], 2)}" + "$\\pm$" + f"{_format_float(row['Accepted_Patch_std'], 2)}"
        lines.append(
            f"{_latex_escape(str(row['Variant']))} & {psnr} & {ssim} & {lpips} & "
            f"{coverage} & {pseudo} & {patch} & {row['N']} & "
            f"{_latex_escape(str(row['Failure Note']))} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize UTrustGS ablation metrics.")
    parser.add_argument(
        "--protocol",
        choices=["single_scene_mechanism", "full_llff"],
        default="single_scene_mechanism",
    )
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    parser.add_argument("--scenes", default="", help="Optional comma-separated scene filter.")
    parser.add_argument("--seeds", default="", help="Optional comma-separated seed filter.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ablation_root = args.ablation_root.expanduser().resolve()
    variants = _split_csv(args.variants)
    scene_filter = _split_csv_set(args.scenes)
    seed_filter = _split_csv_set(args.seeds)
    component_rows = _build_component_rows(
        ablation_root,
        variants,
        scene_filter=scene_filter,
        seed_filter=seed_filter,
    )

    tables_dir = ablation_root / "tables"
    _write_component_csv(tables_dir / "component_ablation.csv", component_rows)
    _write_component_tex(tables_dir / "component_ablation.tex", component_rows)
    _write_json(
        ablation_root / "reports" / "summary.json",
        {
            "protocol": str(args.protocol),
            "generalization_claim_allowed": bool(args.protocol == "full_llff"),
            "component_rows": component_rows,
            "component_filter": {
                "scenes": sorted(scene_filter) if scene_filter is not None else None,
                "seeds": sorted(seed_filter) if seed_filter is not None else None,
            },
        },
    )


if __name__ == "__main__":
    main()
