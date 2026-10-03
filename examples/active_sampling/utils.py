"""
Utility functions for the validation-active sampling module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import torch


@dataclass
class ViewCandidate:
    """Container storing candidate camera parameters."""

    position: torch.Tensor  # [3]
    viewmat: torch.Tensor  # [4, 4]
    intrinsics: torch.Tensor  # [3, 3]


def fibonacci_sphere(
    num_samples: int, start_index: int = 0, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """
    Generate approximately uniform directions on a sphere using Fibonacci sampling.

    Returns:
        torch.Tensor: [num_samples, 3] unit vectors.
    """
    assert num_samples > 0
    offset = 2.0 / num_samples
    increment = math.pi * (3.0 - math.sqrt(5.0))
    samples = []
    for index in range(num_samples):
        y = ((index + start_index) * offset) - 1.0 + (offset / 2.0)
        radius = math.sqrt(max(0.0, 1.0 - y * y))
        phi = ((index + start_index) % num_samples) * increment
        x = math.cos(phi) * radius
        z = math.sin(phi) * radius
        samples.append([x, y, z])
    return torch.tensor(samples, dtype=dtype)


def normalize(vector: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return vector / (vector.norm(dim=-1, keepdim=True) + eps)


def look_at_matrix(
    eye: torch.Tensor,
    center: torch.Tensor,
    up: torch.Tensor = torch.tensor([0.0, 0.0, 1.0]),
) -> torch.Tensor:
    """
    Construct a look-at camera view matrix (OpenCV/COLMAP convention).

    坐标约定（与gsplat/COLMAP一致）:
      X = right, Y = down, Z = forward（进入场景）

    Args:
        eye: Camera position [3]
        center: Target position [3]
        up: World up vector [3]
    Returns:
        torch.Tensor: [4, 4] view matrix (W2C).
    """
    device = eye.device
    dtype = eye.dtype
    up = up.to(device=device, dtype=dtype)
    forward = normalize(center - eye)
    right = normalize(torch.cross(forward, up, dim=-1))
    true_up = torch.cross(right, forward, dim=-1)
    view = torch.eye(4, dtype=eye.dtype, device=eye.device)
    # OpenCV约定: X=right, Y=-up(down), Z=+forward
    # 与COLMAP的build_view_matrix和gsplat的rasterization一致
    view[:3, :3] = torch.stack([right, -true_up, forward], dim=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine distance (1 - cosine similarity) between vectors.
    """
    a_norm = normalize(a)
    b_norm = normalize(b)
    similarity = (a_norm * b_norm).sum(dim=-1)
    return 1.0 - similarity
