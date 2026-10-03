"""
Candidate viewpoint generation for validation-active sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import math
import torch

from .configs.active_sampling_config import ActiveSamplingConfig
from .utils import (
    ViewCandidate,
    cosine_distance,
    fibonacci_sphere,
    look_at_matrix,
    normalize,
)


@dataclass
class ExistingCamera:
    """Minimal camera description used for filtering."""

    position: torch.Tensor  # [3]
    viewmat: torch.Tensor  # [4, 4]
    intrinsics: torch.Tensor  # [3, 3]


class ViewpointSampler:
    """Generate candidate viewpoints on a sphere around the scene."""

    def __init__(self, config: ActiveSamplingConfig) -> None:
        self.config = config

    def generate_candidates(
        self,
        scene_center: torch.Tensor,
        intrinsics: torch.Tensor,
        existing_cameras: Sequence[ExistingCamera],
        device: torch.device,
        reference_cameras: Sequence[ExistingCamera] = None,
        look_at_target: torch.Tensor = None,
        world_up: torch.Tensor = None,
    ) -> List[ViewCandidate]:
        """
        Generate candidate viewpoints with coverage-cone constrained sampling.

        采用覆盖锥约束 (CACG) 替代全球面均匀采样。

        Args:
            scene_center:  [3] 球心坐标（候选相机围绕此点的球面上分布）。
                           对于 forward-facing 场景应使用相机群均值（Z≈0），
                           不是场景内容中心（Z≈80）！
            intrinsics:    [3, 3] camera intrinsic matrix.
            existing_cameras: 已有相机列表（用于近邻过滤）.
            device: target device.
            reference_cameras: CACG 基准相机（仅真实相机）。
            look_at_target: [3] 候选相机的朝向目标点（look-at target）。
                           若为 None，则退化为使用 scene_center（原行为）。
                           对于 forward-facing 场景应设为高斯体均值（Z≈80），
                           使生成的伪视角朝向场景内容而非朝向相机群。
        """
        scene_center = scene_center.detach().to(device)
        # look_at_target: 朝向目标。forward-facing 场景与球心是两个不同的点
        if look_at_target is not None:
            look_at_target = look_at_target.detach().to(device)
        else:
            look_at_target = scene_center  # 退化为原行为
        # world_up: 世界坐标系"向上"方向，用于 look_at_matrix 确定相机 Y 轴方向
        # 对于 OpenCV/COLMAP 数据集: camera-Y=向下, world-up=-Y=[0,-1,0]
        # 若未提供则尝试从 reference_cameras 推导，否则使用 [0,-1,0] 默认值
        if world_up is None:
            _ref = reference_cameras if reference_cameras is not None else existing_cameras
            if _ref:
                # camera Y轴 = viewmat 第1行（即 W2C R[1,:]），对应「向下」方向
                # world up = 负的相机Y轴均值
                _cam_Y_dirs = []
                for _c in _ref:
                    _cam_Y = _c.viewmat[:3, 1].detach().to(device)
                    _cam_Y = _cam_Y / (_cam_Y.norm() + 1e-8)
                    _cam_Y_dirs.append(_cam_Y)
                _avg_cam_Y = torch.stack(_cam_Y_dirs).mean(dim=0)
                world_up = -_avg_cam_Y  # world-up = 负的相机Y（因相机Y朝下）
                world_up = world_up / (world_up.norm() + 1e-8)
            else:
                world_up = torch.tensor([0.0, -1.0, 0.0], device=device)
        else:
            world_up = world_up.detach().to(device)

        num_candidates = self.config.candidate_pool_size * 6
        directions = fibonacci_sphere(
            num_candidates, start_index=self.config.fibonacci_start_index
        ).to(device)
        radius = self._compute_radius(scene_center, existing_cameras)
        filtered: List[ViewCandidate] = []

        # CACG: 使用参考相机（真实相机）的实际观测方向
        cacg_ref = reference_cameras if reference_cameras is not None else existing_cameras
        ref_forward_dirs: List[torch.Tensor] = []
        for cam in cacg_ref:
            cam_forward = cam.viewmat[:3, 2].detach().to(device)
            cam_forward = normalize(cam_forward)
            ref_forward_dirs.append(cam_forward)

        max_deviation_rad = math.radians(self.config.max_view_deviation_angle)
        _cacg_rejected = 0

        # 自动检测场景类型: forward-facing vs object-centric
        _is_forward_facing = False
        _avg_forward = None
        if len(ref_forward_dirs) >= 2:
            stacked = torch.stack(ref_forward_dirs, dim=0)
            _avg_forward = normalize(stacked.mean(dim=0))
            consistencies = [torch.dot(f, _avg_forward).item() for f in ref_forward_dirs]
            min_consistency = min(consistencies)
            if min_consistency > 0.9:
                _is_forward_facing = True
                print(f"[CACG] 检测到forward-facing场景 (consistency={min_consistency:.3f}), "
                      f"球心Z={scene_center[2].item():.2f}, look-at-target Z={look_at_target[2].item():.2f}")

        for direction in directions:
            if torch.abs(direction[2]) > 0.999:
                continue
            elevation = torch.acos(torch.clamp(direction[2], -1.0, 1.0)) * (180.0 / torch.pi)
            azimuth = torch.atan2(direction[1], direction[0]) * (180.0 / torch.pi)
            azimuth = (azimuth + 360.0) % 360.0
            if not (
                self.config.elevation_range[0]
                <= elevation
                <= self.config.elevation_range[1]
            ):
                continue
            if not (
                self.config.azimuth_range[0] <= azimuth <= self.config.azimuth_range[1]
            ):
                continue

            position = scene_center + radius * normalize(direction)

            # CACG覆盖锥过滤：候选观测方向与参考相机前方夹角 <= max_view_deviation_angle
            if ref_forward_dirs:
                if _is_forward_facing:
                    # forward-facing: 候选前方 = look_at_target 方向
                    candidate_forward = normalize(look_at_target - position)
                else:
                    # object-centric: 候选前方 = 朝向 look_at_target
                    candidate_forward = normalize(look_at_target - position)
                min_angle = float('inf')
                for ref_dir in ref_forward_dirs:
                    cos_sim = torch.dot(candidate_forward, ref_dir).clamp(-1.0, 1.0)
                    angle = torch.acos(cos_sim).item()
                    min_angle = min(min_angle, angle)
                if min_angle > max_deviation_rad:
                    _cacg_rejected += 1
                    continue

            # look_at_matrix: 候选相机从 position 朝向 look_at_target
            viewmat = look_at_matrix(position, look_at_target, up=world_up)
            if self._is_too_close(position, scene_center, existing_cameras):
                continue
            candidate = ViewCandidate(
                position=position, viewmat=viewmat, intrinsics=intrinsics
            )
            filtered.append(candidate)
            if len(filtered) >= self.config.candidate_pool_size:
                break

        if _cacg_rejected > 0:
            print(f"[CACG] 覆盖锥过滤: 拒绝{_cacg_rejected}个背面候选 "
                  f"(阈值={self.config.max_view_deviation_angle}°, "
                  f"保留={len(filtered)})")
        return filtered


    def _is_too_close(
        self,
        position: torch.Tensor,
        scene_center: torch.Tensor,
        existing_cameras: Sequence[ExistingCamera],
    ) -> bool:
        if not existing_cameras:
            return False
        candidate_direction = normalize(scene_center - position)
        for camera in existing_cameras:
            camera_direction = normalize(scene_center - camera.position)
            distance = cosine_distance(candidate_direction, camera_direction).abs()
            if distance < self.config.min_view_cosine_distance:
                return True
            similarity = 1.0 - float(distance.item())
            similarity = max(-1.0, min(1.0, similarity))
            angle_deg = math.degrees(math.acos(similarity))
            if angle_deg < self.config.min_selection_angle_deg:
                return True
        return False

    def _compute_radius(
        self,
        scene_center: torch.Tensor,
        existing_cameras: Sequence[ExistingCamera],
    ) -> float:
        radius = float(self.config.sphere_radius)
        if radius > 0.0:
            return radius
        if not existing_cameras:
            return 1.0
        positions = torch.stack(
            [camera.position.detach() for camera in existing_cameras]
        )
        min_corner = positions.min(dim=0).values
        max_corner = positions.max(dim=0).values
        bbox_diag = torch.norm(max_corner - min_corner).item()
        distances = torch.norm(positions - scene_center.unsqueeze(0), dim=-1)
        max_distance = distances.max().item()
        auto_radius = max(bbox_diag, max_distance)
        return max(auto_radius, 1.0)
