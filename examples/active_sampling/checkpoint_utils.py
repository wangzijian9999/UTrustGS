"""Checkpoint loading helpers for active-sampling teacher models.

The where_wrong path adds optional heads, so bare ``strict=False`` loading can
silently hide real compatibility errors.  This module keeps backward
compatibility explicit by allowing only pre-registered prefixes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch


DEFAULT_ALLOWED_MISSING_PREFIXES: Tuple[str, ...] = (
    "failure_prior_head.",
    "frontier_proposal_head.",
    "where_wrong_head.",
    "condition_preprocessor._depth_model.",
)
DEFAULT_ALLOWED_UNEXPECTED_PREFIXES: Tuple[str, ...] = ()
FAILURE_PRIOR_PREFIXES: Tuple[str, ...] = (
    "failure_prior_head.",
    "where_wrong_head.",
)
RUNTIME_DEPTH_PREFIX = "condition_preprocessor._depth_model."


@dataclass
class CheckpointLoadReport:
    """Structured report returned by whitelist checkpoint loading."""

    missing_keys: List[str]
    unexpected_keys: List[str]
    disallowed_missing_keys: List[str]
    disallowed_unexpected_keys: List[str]

    @property
    def ok(self) -> bool:
        return not self.disallowed_missing_keys and not self.disallowed_unexpected_keys


def extract_model_state_dict(checkpoint_obj: Any) -> Dict[str, Any]:
    """Extract a model state_dict from common checkpoint containers."""

    if isinstance(checkpoint_obj, dict):
        for key_name in ("state_dict", "model_state_dict", "model", "module"):
            candidate = checkpoint_obj.get(key_name)
            if isinstance(candidate, dict):
                return candidate
        return checkpoint_obj
    raise TypeError("checkpoint object must be a state_dict-like dictionary")


def extract_checkpoint_metadata(checkpoint_obj: Any) -> Dict[str, Any]:
    """Return optional checkpoint metadata without inventing missing fields."""

    if not isinstance(checkpoint_obj, dict):
        return {}
    metadata = checkpoint_obj.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    checkpoint_metadata = checkpoint_obj.get("checkpoint_metadata")
    if isinstance(checkpoint_metadata, dict):
        return checkpoint_metadata
    return {}


def strip_module_prefix(state_dict: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip a leading DDP ``module.`` prefix where present."""

    stripped_state_dict: Dict[str, Any] = {}
    for key_name, value in state_dict.items():
        if key_name.startswith("module."):
            stripped_state_dict[key_name[len("module."):]] = value
        else:
            stripped_state_dict[key_name] = value
    return stripped_state_dict


def _materialize_known_lazy_modules(
    module: torch.nn.Module,
    state_dict: Mapping[str, Any],
) -> None:
    """Create small lazy compatibility modules before ``load_state_dict``.

    Older ControlNet teacher checkpoints may contain ``_texture_proj`` even
    though the current model creates it lazily on first texture-token use.  If
    we do not materialize it before loading, PyTorch reports the checkpoint
    weights as unexpected and the trained projection is silently unusable.
    """

    texture_proj_weight = state_dict.get("_texture_proj.weight")
    texture_proj_bias = state_dict.get("_texture_proj.bias")
    if not isinstance(texture_proj_weight, torch.Tensor):
        return
    if texture_proj_weight.dim() != 2:
        return
    existing_texture_proj = getattr(module, "_texture_proj", None)
    if existing_texture_proj is not None:
        return
    output_features = int(texture_proj_weight.shape[0])
    input_features = int(texture_proj_weight.shape[1])
    try:
        reference_parameter = next(module.parameters())
        target_device = reference_parameter.device
    except StopIteration:
        target_device = texture_proj_weight.device
    target_dtype = texture_proj_weight.dtype
    if isinstance(texture_proj_bias, torch.Tensor):
        target_dtype = texture_proj_bias.dtype
    module._texture_proj = torch.nn.Linear(  # type: ignore[attr-defined]
        input_features,
        output_features,
        bias=isinstance(texture_proj_bias, torch.Tensor),
    ).to(device=target_device, dtype=target_dtype)


def count_key_prefixes(keys: Sequence[str], depth: int = 2) -> Dict[str, int]:
    prefix_counts: Dict[str, int] = {}
    for key_name in keys:
        key_parts = key_name.split(".")
        prefix = ".".join(key_parts[: min(depth, len(key_parts))])
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    return dict(sorted(prefix_counts.items(), key=lambda item: (-item[1], item[0])))


def _is_allowed_key(key_name: str, allowed_prefixes: Iterable[str]) -> bool:
    return any(key_name.startswith(prefix) for prefix in allowed_prefixes)


def _print_key_report(
    *,
    report_prefix: str,
    key_label: str,
    keys: Sequence[str],
) -> None:
    sorted_keys = sorted(keys)
    if not sorted_keys:
        return
    print(f"{report_prefix} {key_label} keys 前缀统计:")
    for prefix_name, prefix_count in count_key_prefixes(sorted_keys).items():
        print(f"{report_prefix}   {prefix_name}: {prefix_count}")
    print(f"{report_prefix} {key_label} keys 完整列表:")
    for key_index, key_name in enumerate(sorted_keys, start=1):
        print(f"{report_prefix}   [{key_index:04d}] {key_name}")


def _warn_if_depth_metadata_missing(
    *,
    report_prefix: str,
    checkpoint_obj: Any,
    missing_keys: Sequence[str],
) -> None:
    depth_missing_count = sum(
        1 for key_name in missing_keys if key_name.startswith(RUNTIME_DEPTH_PREFIX)
    )
    if depth_missing_count <= 0:
        return
    metadata = extract_checkpoint_metadata(checkpoint_obj)
    depth_metadata = metadata.get("depth_anything_v2") or metadata.get("depth_model")
    if isinstance(depth_metadata, dict):
        print(f"{report_prefix} DepthAnything metadata: {depth_metadata}")
    else:
        print(
            f"{report_prefix} WARNING: {depth_missing_count} 个 DepthAnything 运行时参数缺失，"
            "且 checkpoint metadata 中未找到 depth model 类型/路径/哈希；将依赖外部初始化。"
        )


def load_state_dict_with_whitelist(
    module: torch.nn.Module,
    checkpoint_obj: Any,
    *,
    report_prefix: str,
    allowed_missing_prefixes: Optional[Sequence[str]] = None,
    allowed_unexpected_prefixes: Optional[Sequence[str]] = None,
    allow_unused_failure_prior_keys: bool = False,
    strip_ddp_module_prefix: bool = True,
) -> CheckpointLoadReport:
    """Load a checkpoint with explicit missing/unexpected key policy.

    The actual PyTorch load still uses ``strict=False`` so that we can inspect
    missing/unexpected keys, but disallowed mismatches raise immediately after
    reporting.  This preserves old-checkpoint compatibility without silent
    failure.
    """

    state_dict = extract_model_state_dict(checkpoint_obj)
    if strip_ddp_module_prefix and any(key.startswith("module.") for key in state_dict):
        state_dict = strip_module_prefix(state_dict)
    _materialize_known_lazy_modules(module, state_dict)

    missing_keys, unexpected_keys = module.load_state_dict(state_dict, strict=False)
    sorted_missing_keys = sorted(str(key) for key in missing_keys)
    sorted_unexpected_keys = sorted(str(key) for key in unexpected_keys)

    resolved_allowed_missing = tuple(
        DEFAULT_ALLOWED_MISSING_PREFIXES
        if allowed_missing_prefixes is None
        else allowed_missing_prefixes
    )
    resolved_allowed_unexpected = tuple(
        DEFAULT_ALLOWED_UNEXPECTED_PREFIXES
        if allowed_unexpected_prefixes is None
        else allowed_unexpected_prefixes
    )
    if allow_unused_failure_prior_keys:
        resolved_allowed_unexpected = resolved_allowed_unexpected + FAILURE_PRIOR_PREFIXES

    disallowed_missing = [
        key_name
        for key_name in sorted_missing_keys
        if not _is_allowed_key(key_name, resolved_allowed_missing)
    ]
    disallowed_unexpected = [
        key_name
        for key_name in sorted_unexpected_keys
        if not _is_allowed_key(key_name, resolved_allowed_unexpected)
    ]

    print(
        f"{report_prefix} checkpoint加载完成: "
        f"missing={len(sorted_missing_keys)}, unexpected={len(sorted_unexpected_keys)}, "
        f"disallowed_missing={len(disallowed_missing)}, "
        f"disallowed_unexpected={len(disallowed_unexpected)}"
    )
    _warn_if_depth_metadata_missing(
        report_prefix=report_prefix,
        checkpoint_obj=checkpoint_obj,
        missing_keys=sorted_missing_keys,
    )
    _print_key_report(
        report_prefix=report_prefix,
        key_label="missing",
        keys=sorted_missing_keys,
    )
    _print_key_report(
        report_prefix=report_prefix,
        key_label="unexpected",
        keys=sorted_unexpected_keys,
    )

    report = CheckpointLoadReport(
        missing_keys=sorted_missing_keys,
        unexpected_keys=sorted_unexpected_keys,
        disallowed_missing_keys=disallowed_missing,
        disallowed_unexpected_keys=disallowed_unexpected,
    )
    if not report.ok:
        raise RuntimeError(
            f"{report_prefix} checkpoint key mismatch not allowed: "
            f"disallowed_missing={disallowed_missing[:10]}, "
            f"disallowed_unexpected={disallowed_unexpected[:10]}"
        )
    return report
