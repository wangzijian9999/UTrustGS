#!/usr/bin/env python
"""修复可靠性的离线诊断；不训练模型，也不读取正式测试集。

用法：python -B evaluate_repair_reliability.py manifest.json --output report.json
输入 JSON 顶层字段：schema_version=1、data_role（development 或
synthetic_diagnostic）、dataset、parent_sha256、generator_sha256、
population="all_candidates_before_admission"、records（非空列表）。
每条记录：scene、seed（非负整数）、view、channel（rgb/depth/coverage）、
arrays（相对 manifest 的 NPZ 路径）、arrays_sha256、reference_kind、
error_metric、risk_error_threshold、correct_error_threshold、help_margin、
trust_semantics（correct_probability/helpful_probability/score）。
reference_kind 分别为 independent_rgb/measured_depth/silhouette；合成诊断
只能使用 synthetic。错误必须由调用方用独立参考、同一误差定义预先计算。

NPZ 包含同形一维 current_error/proposal_error/risk_score/trust_score，以及
布尔 valid/support/admitted，可选 uncertainty_score。错误为有限非负值，
所有分数有限，admitted ⊆ support ⊆ valid。必须保留拒绝项；全准入也合法。
risk_score 越大表示当前错误风险越高，trust_score 越大表示修复越可信；
可选 uncertainty_score 越小越可信，评价时用其相反数排序。
population 声明和参考来源无法仅凭数组核验，调用方须提供外部来源审计。
不支持仅导出准入项后声称完整候选，也不推断缺失参考或校准概率语义。

标签：risk=current_error>risk_error_threshold；
correct=proposal_error<=correct_error_threshold；
helpful=proposal_error+help_margin<current_error（严格改善）。
AP 按同分组计算非插值 average precision；无正例返回 null。
选择性曲线按同分整组纳入，报告实际覆盖。概率 ECE 使用 10 个等宽箱，
最后一箱包含 1；score 语义不计算 ECE/Brier。只逐记录报告，不跨通道、
误差尺度或场景汇总，也不提供把像素当独立样本的置信区间。
输出全为 DIAGNOSTIC_ONLY；输入哈希不等于参考独立性或来源真实性证明。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


STATUS = "DIAGNOSTIC_ONLY"
REQUIRED_ARRAYS = ("current_error", "proposal_error", "risk_score", "trust_score",
                   "valid", "support", "admitted")
MASK_NAMES = ("valid", "support", "admitted")
COVERAGES = (0.1, 0.25, 0.5, 1.0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} 必须是非空字符串")
    return value


def _sha256(value, name: str) -> str:
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"{name} 必须为 64 位小写十六进制 SHA256")
    return value


def _threshold(value, name: str) -> float:
    _require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
             f"{name} 必须为有限非负数")
    return float(value)


def average_precision(labels: np.ndarray, scores: np.ndarray):
    """同分整组 AP；同分组内部置换不影响结果。"""
    positive_count = int(labels.sum())
    if not positive_count:
        return None
    order = np.argsort(-scores, kind="stable")
    ordered_scores, ordered_labels = scores[order], labels[order]
    group_ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]),
                       len(scores) - 1]
    true_positives = np.cumsum(ordered_labels, dtype=np.int64)[group_ends]
    precision = true_positives / (group_ends + 1)
    recall_increments = np.diff(np.r_[0, true_positives]) / positive_count
    return float(np.sum(precision * recall_increments))


def calibration(labels: np.ndarray, probabilities: np.ndarray, event: str):
    if not labels.size:
        return None
    assignments = np.minimum((probabilities * 10).astype(np.int64), 9)
    bins, expected_error = [], 0.0
    for index in range(10):
        selected = assignments == index
        count = int(selected.sum())
        accuracy = float(labels[selected].mean()) if count else None
        confidence = float(probabilities[selected].mean()) if count else None
        if count:
            expected_error += count / labels.size * abs(accuracy - confidence)
        bins.append({"lower": index / 10, "upper": (index + 1) / 10,
                     "count": count, "event_rate": accuracy, "mean_probability": confidence})
    return {"event": event, "method": "10_equal_width_bins_last_includes_1",
            "ece": expected_error, "brier": float(np.mean((probabilities - labels) ** 2)),
            "bins": bins}


def _rates(mask: np.ndarray, arrays: dict, labels: dict) -> dict:
    count = int(mask.sum())
    return {"count": count,
            "correct_rate": float(labels["correct"][mask].mean()) if count else None,
            "helpful_rate": float(labels["helpful"][mask].mean()) if count else None,
            "mean_signed_gain": float(np.mean(arrays["current_error"][mask]
                                              - arrays["proposal_error"][mask])) if count else None}


def _domain(mask: np.ndarray, arrays: dict, labels: dict, semantics: str) -> dict:
    result = _rates(mask, arrays, labels)
    count, valid_count = result["count"], int(arrays["valid"].sum())
    result["fraction_of_valid"] = count / valid_count if valid_count else None
    result["admission_coverage"] = int((mask & arrays["admitted"]).sum()) / count if count else None
    result["risk_positive_count"] = int(labels["risk"][mask].sum())
    result["risk_detection_ap"] = average_precision(labels["risk"][mask], arrays["risk_score"][mask])
    for event in ("correct", "helpful"):
        event_labels = labels[event][mask]
        result[event + "_ranking"] = {
            "positive_count": int(event_labels.sum()),
            "trust_ap": average_precision(event_labels, arrays["trust_score"][mask]),
            "risk_ap": average_precision(event_labels, arrays["risk_score"][mask]),
            "negative_uncertainty_ap": average_precision(event_labels, -arrays["uncertainty_score"][mask])
            if "uncertainty_score" in arrays else None}
    event = {"correct_probability": "correct", "helpful_probability": "helpful"}.get(semantics)
    result["calibration"] = calibration(labels[event][mask], arrays["trust_score"][mask], event) if event else None
    return result


def _selective_curve(mask: np.ndarray, arrays: dict, labels: dict) -> list:
    count = int(mask.sum())
    ordered_scores = np.sort(arrays["trust_score"][mask])[::-1]
    rows = []
    for requested_coverage in COVERAGES:
        threshold = float(ordered_scores[math.ceil(requested_coverage * count) - 1]) if count else None
        selected = mask & (arrays["trust_score"] >= threshold) if count else mask.copy()
        row = _rates(selected, arrays, labels)
        row.update({"requested_coverage": requested_coverage,
                    "actual_coverage": row["count"] / count if count else None,
                    "score_threshold": threshold})
        rows.append(row)
    return rows


def _load_record(record: dict, directory: Path, data_role: str) -> tuple:
    _require(isinstance(record, dict), "每个 record 必须为对象")
    metadata = {key: _text(record.get(key), key) for key in
                ("scene", "view", "channel", "reference_kind", "error_metric", "trust_semantics", "arrays")}
    seed = record.get("seed")
    _require(type(seed) is int and seed >= 0, "seed 必须为非负整数")
    metadata["seed"] = seed
    channel = metadata["channel"]
    _require(channel in ("rgb", "depth", "coverage"), "channel 必须为 rgb/depth/coverage")
    reference = "synthetic" if data_role == "synthetic_diagnostic" else {
        "rgb": "independent_rgb", "depth": "measured_depth", "coverage": "silhouette"}[channel]
    _require(metadata["reference_kind"] == reference, "reference_kind 与 channel/data_role 不一致")
    _require(metadata["trust_semantics"] in ("correct_probability", "helpful_probability", "score"),
             "trust_semantics 无效")
    for key in ("risk_error_threshold", "correct_error_threshold", "help_margin"):
        metadata[key] = _threshold(record.get(key), key)
    metadata["arrays_sha256"] = _sha256(record.get("arrays_sha256"), "arrays_sha256")
    relative_path = Path(metadata["arrays"])
    _require(not relative_path.is_absolute(), "arrays 必须使用相对路径")
    arrays_path = (directory / relative_path).resolve()
    _require(arrays_path.is_relative_to(directory), "arrays 不能越出 manifest 所在目录")
    _require(sha256_file(arrays_path) == metadata["arrays_sha256"], "arrays_sha256 不匹配")
    with np.load(arrays_path, allow_pickle=False) as archive:
        _require(all(name in archive.files for name in REQUIRED_ARRAYS), "NPZ 缺少必需数组")
        names = REQUIRED_ARRAYS + (("uncertainty_score",) if "uncertainty_score" in archive.files else ())
        arrays = {name: archive[name].copy() for name in names}
    shape = arrays["current_error"].shape
    _require(len(shape) == 1 and all(values.shape == shape for values in arrays.values()), "数组必须同形且一维")
    for name, values in arrays.items():
        if name in MASK_NAMES:
            _require(values.dtype == np.bool_, f"{name} 必须为 bool 数组")
        else:
            _require(values.dtype.kind in "iuf", f"{name} 必须为实数数组")
            arrays[name] = values.astype(np.float64)
            _require(bool(np.isfinite(arrays[name]).all()), f"{name} 含非有限值")
    for name in ("current_error", "proposal_error"):
        _require(bool((arrays[name] >= 0).all()), f"{name} 不能为负")
    _require(not bool((arrays["support"] & ~arrays["valid"]).any()), "support 必须属于 valid")
    _require(not bool((arrays["admitted"] & ~arrays["support"]).any()), "admitted 必须属于 support")
    if metadata["trust_semantics"] != "score":
        _require(bool(((arrays["trust_score"] >= 0) & (arrays["trust_score"] <= 1)).all()), "概率必须在 [0, 1]")
    return metadata, arrays


def evaluate_manifest(manifest_path: Path) -> dict:
    manifest_path = manifest_path.resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    _require(isinstance(manifest, dict), "manifest 必须为对象")
    _require(type(manifest.get("schema_version")) is int and manifest["schema_version"] == 1, "schema_version 必须为 1")
    data_role = manifest.get("data_role")
    _require(data_role in ("development", "synthetic_diagnostic"), "仅接受 development/synthetic_diagnostic，拒绝 formal_test")
    _require(manifest.get("population") == "all_candidates_before_admission", "必须声明准入前完整候选集合")
    result = {"schema_version": 1, "status": STATUS, "aggregation": "per_record_only",
              "data_role": data_role, "dataset": _text(manifest.get("dataset"), "dataset"),
              "population": manifest["population"],
              "parent_sha256": _sha256(manifest.get("parent_sha256"), "parent_sha256"),
              "generator_sha256": _sha256(manifest.get("generator_sha256"), "generator_sha256"),
              "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
              "evaluator_sha256": sha256_file(Path(__file__)), "records": []}
    records = manifest.get("records")
    _require(isinstance(records, list) and bool(records), "records 必须为非空列表")
    seen = set()
    for record in records:
        metadata, arrays = _load_record(record, manifest_path.parent, data_role)
        key = tuple(metadata[name] for name in ("scene", "seed", "view", "channel"))
        _require(key not in seen, f"重复 record：{key}")
        seen.add(key)
        # 非负误差与 margin 之和若上溢为 +inf，严格改善条件自然为 False。
        with np.errstate(over="ignore"):
            labels = {"risk": arrays["current_error"] > metadata["risk_error_threshold"],
                      "correct": arrays["proposal_error"] <= metadata["correct_error_threshold"],
                      "helpful": arrays["proposal_error"] + metadata["help_margin"] < arrays["current_error"]}
        masks = {"all_valid": arrays["valid"], "support": arrays["support"],
                 "accepted": arrays["admitted"], "rejected": arrays["valid"] & ~arrays["admitted"]}
        metadata.update({"status": STATUS, "input_count": len(arrays["valid"]),
                         "uncertainty_score_available": "uncertainty_score" in arrays,
                         "domains": {name: _domain(mask, arrays, labels, metadata["trust_semantics"])
                                     for name, mask in masks.items()},
                         "selective_curves": {name: _selective_curve(masks[name], arrays, labels)
                                              for name in ("all_valid", "support")}})
        result["records"].append(metadata)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_manifest(args.manifest)
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as output:
            output.write(serialized)
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.exit(2, f"诊断失败：{error}\n")


if __name__ == "__main__":
    main()
