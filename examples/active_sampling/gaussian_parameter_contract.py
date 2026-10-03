"""GaussianPointField 的物理参数契约；优化采用投影 Adam，PLY 单独转换。

scales 是世界坐标下的正标准差，opacities 是 alpha，quats 为 wxyz。
本模块不把无标记的历史物理参数解释为 log-scale 或 logit。
"""
import copy
import random

import numpy as np
import torch


FORMAT = "gaussian_point_field_physical_v1"
PARAMETERS = ("means", "quats", "scales", "opacities", "_color_logits")
EPSILON = 1e-6


@torch.no_grad()
def accumulate_density_gradients(info, grad2d_accum, count_accum, width, height):
    """按渲染器全局高斯ID累计；packed行号不是高斯ID。"""
    means2d = info.get("means2d")
    if means2d is None:
        return
    grads = getattr(means2d, "absgrad", None)
    if grads is None:
        grads = means2d.grad
    if grads is None:
        return
    grads = grads.detach().clone()
    radii = info.get("radii")
    if radii is None or grads.shape[-1:] != (2,):
        raise ValueError("增密统计要求渲染器radii及二维梯度")
    if grad2d_accum.ndim != 1 or count_accum.shape != grad2d_accum.shape:
        raise ValueError("增密统计数组形状不一致")
    grads[..., 0] *= info.get("width", width) / 2.0 * info.get("n_cameras", 1)
    grads[..., 1] *= info.get("height", height) / 2.0 * info.get("n_cameras", 1)
    if grads.ndim == 2:
        ids = info.get("gaussian_ids")
        if (ids is None or ids.shape != grads.shape[:1] or ids.dtype != torch.int64
                or radii.shape != grads.shape):
            raise ValueError("packed增密统计要求[nnz]全局ID和[nnz,2]半径")
        visible = (radii > 0).all(-1)
        ids, values = ids[visible], grads[visible]
    elif grads.ndim == 3:
        if radii.shape != grads.shape or grads.shape[1] != len(grad2d_accum):
            raise ValueError("dense增密统计要求[C,N,2]半径和梯度")
        visible = (radii > 0).all(-1)
        ids, values = torch.where(visible)[1], grads[visible]
    else:
        raise ValueError("不支持的增密梯度形状")
    if ((ids < 0) | (ids >= len(grad2d_accum))).any() or not torch.isfinite(values).all():
        raise ValueError("增密全局ID越界或梯度非有限")
    grad2d_accum.index_add_(0, ids, values.norm(dim=-1))
    count_accum.index_add_(0, ids, torch.ones_like(ids, dtype=count_accum.dtype))


def validate_field(gaussians):
    count = len(gaussians.means)
    shapes = ((count, 3), (count, 4), (count, 3), (count,), (count, 3))
    if count == 0:
        raise ValueError("高斯场不能为空")
    for name, shape in zip(PARAMETERS, shapes):
        value = getattr(gaussians, name)
        if value.shape != shape or not torch.isfinite(value).all():
            raise ValueError(f"{name} 的形状或有限性不符合物理参数契约")
    if (gaussians.scales <= 0).any():
        raise ValueError("scales 必须是正的物理标准差，不能是 log-scale")
    if ((gaussians.opacities < 0) | (gaussians.opacities > 1)).any():
        raise ValueError("opacities 必须是 [0,1] 内的物理 alpha")
    if (gaussians.quats.norm(dim=-1) <= EPSILON).any():
        raise ValueError("四元数不能为零")


@torch.no_grad()
def project_field(gaussians):
    # 非有限状态不能靠裁剪掩盖；仅投影有限的越界优化结果。
    for name in PARAMETERS:
        if not torch.isfinite(getattr(gaussians, name)).all():
            raise ValueError(f"优化后 {name} 非有限")
    norms = gaussians.quats.norm(dim=-1, keepdim=True)
    if (norms <= EPSILON).any():
        raise ValueError("优化产生零四元数")
    gaussians.scales.clamp_(min=EPSILON)
    gaussians.opacities.clamp_(EPSILON, 1 - EPSILON)
    gaussians.quats.div_(norms)


def field_state(gaussians):
    validate_field(gaussians)
    result = {name: getattr(gaussians, name).detach().cpu().clone() for name in PARAMETERS}
    result["parameter_format"] = FORMAT
    result["colors"] = gaussians.colors.detach().cpu().clone()
    return result


def load_field(gaussians, state, device):
    if state.get("parameter_format", FORMAT) != FORMAT:
        raise ValueError("不支持的高斯参数格式；禁止隐式转换")
    values = dict(state)
    if "_color_logits" not in values:
        values["_color_logits"] = values.get("color_logits", values.get("logits"))
        if values["_color_logits"] is None:
            colors = values["colors"]
            if not torch.isfinite(colors).all() or ((colors < 0) | (colors > 1)).any():
                raise ValueError("历史颜色不在 [0,1]")
            values["_color_logits"] = torch.logit(colors.clamp(1e-4, 1 - 1e-4))
    # 在更改目标前验证，旧的非法失败快照不能被静默修复。
    from types import SimpleNamespace
    candidate = SimpleNamespace(**{name: values[name].to(device).clone() for name in PARAMETERS})
    validate_field(candidate)
    with torch.no_grad():
        for name in PARAMETERS:
            old = getattr(gaussians, name, None)
            value = getattr(candidate, name)
            if old is not None and old.shape == value.shape:
                old.copy_(value)
            else:
                setattr(gaussians, name, torch.nn.Parameter(value))


@torch.no_grad()
def replace_rows(gaussians, optimizer, values, old_rows):
    """old_rows=-1 表示新基元；存活行继承动量，新行动量为零。

Adam 的标量 step 沿用参数张量的时间，新增行不声称具有独立 Adam 时钟。
"""
    from types import SimpleNamespace
    validate_field(SimpleNamespace(**values))
    if old_rows.shape != (len(values["means"]),):
        raise ValueError("行映射形状不匹配")
    if ((old_rows < -1) | (old_rows >= len(gaussians.means))).any():
        raise ValueError("无效的旧行索引")
    for name in PARAMETERS:
        old = getattr(gaussians, name)
        new = torch.nn.Parameter(values[name].detach().clone())
        state = optimizer.state.pop(old, {})
        migrated = {}
        for key, value in state.items():
            if torch.is_tensor(value) and value.shape == old.shape:
                rows = old_rows.to(value.device)
                keep = rows >= 0
                target = value.new_zeros(new.shape)
                target[keep] = value[rows[keep]]
                migrated[key] = target
            else:
                migrated[key] = copy.deepcopy(value)
        for group in optimizer.param_groups:
            group["params"] = [new if parameter is old else parameter for parameter in group["params"]]
        if migrated:
            optimizer.state[new] = migrated
        setattr(gaussians, name, new)


@torch.no_grad()
def refine_field(gaussians, optimizer, gradients, scale_threshold, gradient_threshold, prune_opacity):
    validate_field(gaussians)
    rows = torch.arange(len(gaussians.means), device=gaussians.means.device)
    prune = gaussians.opacities.detach() < prune_opacity
    if prune.all():
        prune[gaussians.opacities.argmax()] = False
    high = (gradients > gradient_threshold) & ~prune
    small = gaussians.scales.detach().amax(dim=-1) <= scale_threshold
    duplicate = rows[high & small]
    split = rows[high & ~small]
    keep = rows[~prune & ~(high & ~small)]
    source = torch.cat((keep, duplicate, split, split))
    values = {name: getattr(gaussians, name).detach()[source].clone() for name in PARAMETERS}
    if len(split):
        offset = torch.randn_like(gaussians.means[split]) * gaussians.scales[split].mean(-1, keepdim=True) / 1.6
        start = len(keep) + len(duplicate)
        values["means"][start:start + len(split)] += offset
        values["means"][start + len(split):] -= offset
        values["scales"][start:] = (values["scales"][start:] / 1.6).clamp_min(EPSILON)
    counts = (len(duplicate), len(split), int(prune.sum()))
    if any(counts):
        old_rows = torch.cat((keep, rows.new_full((len(source) - len(keep),), -1)))
        replace_rows(gaussians, optimizer, values, old_rows)
    return counts


def capture_training_state(gaussians, optimizer, **loop_state):
    numpy_state = np.random.get_state()
    optimizer_state = optimizer.state_dict()
    optimizer_state = {
        "state": {index: {name: value.detach().cpu().clone() if torch.is_tensor(value)
                          else copy.deepcopy(value) for name, value in state.items()}
                  for index, state in optimizer_state["state"].items()},
        "param_groups": copy.deepcopy(optimizer_state["param_groups"]),
    }
    return {"parameter_format": FORMAT, "gaussians": field_state(gaussians),
            "optimizer": optimizer_state, "loop": copy.deepcopy(loop_state),
            # 只保存张量/基础类型，保持 torch.load(weights_only=True) 可读。
            "rng": {"python": random.getstate(),
                    "numpy": (numpy_state[0], torch.from_numpy(numpy_state[1].copy()), *numpy_state[2:]),
                    "torch": torch.get_rng_state().clone(),
                    "cuda": torch.cuda.get_rng_state_all() if gaussians.means.is_cuda else None}}


def restore_training_state(gaussians, optimizer, state):
    if state.get("parameter_format") != FORMAT:
        raise ValueError("完整续训要求显式参数格式")
    old_parameters = list(gaussians.parameters())
    load_field(gaussians, state["gaussians"], gaussians.means.device)
    mapping = dict(zip(old_parameters, gaussians.parameters()))
    for group in optimizer.param_groups:
        group["params"] = [mapping[parameter] for parameter in group["params"]]
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    random.setstate(state["rng"]["python"])
    numpy_state = state["rng"]["numpy"]
    np.random.set_state((numpy_state[0], numpy_state[1].cpu().numpy(), *numpy_state[2:]))
    torch.set_rng_state(state["rng"]["torch"].cpu())
    if state["rng"]["cuda"] is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["rng"]["cuda"]])
    return copy.deepcopy(state["loop"])


def ply_parameters(gaussians):
    validate_field(gaussians)
    # PLY 查看器读取的是 log-scale / opacity logit / SH DC，并非物理 RGB。
    return {"means": gaussians.means.detach(), "scales": gaussians.scales.detach().log(),
            "quats": torch.nn.functional.normalize(gaussians.quats.detach(), dim=-1),
            "opacities": torch.logit(gaussians.opacities.detach().clamp(EPSILON, 1 - EPSILON)),
            "sh0": ((gaussians.colors.detach() - 0.5) / 0.28209479177387814).unsqueeze(1),
            "shN": gaussians.means.new_zeros((len(gaussians.means), 0, 3))}
