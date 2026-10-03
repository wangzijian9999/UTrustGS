"""从显式物理参数检查点提取期望深度 TSDF 表面；不读取扫描或自动配准。

输入世界单位为米；相机为 world-to-camera，gsplat 像素中心为 j+0.5。
Open3D 整数像素中心所用主点减0.5。期望深度可能跨层混合，非真实表面保证。
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from gaussian_parameter_contract import FORMAT, load_field


def validate_camera(view, intrinsic):
    view, intrinsic = np.asarray(view, dtype=np.float64), np.asarray(intrinsic, dtype=np.float64)
    if (view.shape != (4, 4) or intrinsic.shape != (3, 3)
            or not np.isfinite(view).all() or not np.isfinite(intrinsic).all()):
        raise ValueError('相机必须是有限的4×4外参和3×3内参')
    if (not np.allclose(view[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(view[:3, :3].T @ view[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(view[:3, :3]), 1., atol=1e-5)):
        raise ValueError('外参必须为刚体world-to-camera变换')
    if (not np.allclose(intrinsic[2], [0, 0, 1], atol=1e-7)
            or not np.allclose([intrinsic[0, 1], intrinsic[1, 0]], 0, atol=1e-7)
            or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0):
        raise ValueError('只接受无skew的正焦距针孔内参')
    return view, intrinsic


def fusion_intrinsic(intrinsic, width, height):
    import open3d as o3d
    _, intrinsic = validate_camera(np.eye(4), intrinsic)
    return o3d.camera.PinholeCameraIntrinsic(width, height, intrinsic[0, 0], intrinsic[1, 1],
                                            intrinsic[0, 2] - .5, intrinsic[1, 2] - .5)


def fuse_depths(depths, alphas, views, intrinsics, *, voxel_m, sdf_trunc_m,
                depth_max_m, alpha_min):
    """输入相机z深度（米）；保留所有提取组件，不用GT选择网格。"""
    import open3d as o3d
    if (not all(np.isfinite(x) and x > 0 for x in [voxel_m, sdf_trunc_m, depth_max_m])
            or sdf_trunc_m < voxel_m or not np.isfinite(alpha_min) or not 0 < alpha_min <= 1):
        raise ValueError('非法融合阈值')
    if not (len(depths) == len(alphas) == len(views) == len(intrinsics) > 0):
        raise ValueError('深度/alpha/相机数量必须一致且非空')
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_m, sdf_trunc=sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor,
        depth_sampling_stride=1)
    counts = []
    for depth, alpha, view, intrinsic in zip(depths, alphas, views, intrinsics):
        depth, alpha = np.asarray(depth), np.asarray(alpha)
        if (depth.ndim != 2 or alpha.shape != depth.shape or not np.isfinite(depth).all()
                or not np.isfinite(alpha).all() or (alpha < 0).any() or (alpha > 1 + 1e-6).any()):
            raise ValueError('深度和alpha须为同形有限H×W数组，alpha在[0,1]')
        view, intrinsic = validate_camera(view, intrinsic)
        valid = (alpha >= alpha_min) & (depth > 0) & (depth < depth_max_m)
        counts.append(int(valid.sum()))
        if not valid.any():
            continue
        height, width = depth.shape
        depth_image = o3d.geometry.Image(np.ascontiguousarray(np.where(valid, depth, 0), dtype=np.float32))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)), depth_image,
            depth_scale=1., depth_trunc=depth_max_m, convert_rgb_to_intensity=False)
        camera = fusion_intrinsic(intrinsic, width, height)
        volume.integrate(rgbd, camera, view)
    mesh = volume.extract_triangle_mesh()
    if len(mesh.triangles) == 0 or len(mesh.vertices) == 0:
        raise ValueError('提取为空；不得当作零几何误差')
    if not np.isfinite(np.asarray(mesh.vertices)).all():
        raise ValueError('提取网格含非有限坐标')
    return mesh, counts


@torch.no_grad()
def render_checkpoint_depths(checkpoint_path, device):
    from gsplat.rendering import rasterization
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    if checkpoint.get('parameter_format') != FORMAT or 'loop' not in checkpoint:
        raise ValueError('要求显式物理参数格式和含源相机的完整训练检查点')
    field = SimpleNamespace()
    load_field(field, checkpoint['gaussians'], device)
    loop = checkpoint['loop']
    images = loop['real_images']
    if images.ndim != 4 or images.shape[1] != 3 or len(loop['cameras']) != len(images):
        raise ValueError('检查点源图尺寸或相机数无效')
    height, width = images.shape[-2:]
    depths, alphas, views, intrinsics = [], [], [], []
    for view, intrinsic in loop['cameras']:
        validate_camera(view.numpy(), intrinsic.numpy())
        output, alpha, _ = rasterization(
            means=field.means, quats=field.quats, scales=field.scales, opacities=field.opacities,
            colors=field._color_logits.sigmoid(), viewmats=view.to(device)[None],
            Ks=intrinsic.to(device)[None], width=width, height=height, render_mode='ED')
        depths.append(output[0, ..., 0].cpu().numpy())
        alphas.append(alpha[0, ..., 0].cpu().numpy())
        views.append(view.numpy())
        intrinsics.append(intrinsic.numpy())
    return depths, alphas, views, intrinsics


def extract_checkpoint(checkpoint_path, output_path, settings, device):
    """调用者负责声明/审核输入为米坐标；本函数只做固定单位转换。"""
    import open3d as o3d
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError('禁止覆盖已有网格')
    depths, alphas, views, intrinsics = render_checkpoint_depths(checkpoint_path, device)
    mesh, counts = fuse_depths(depths, alphas, views, intrinsics, **settings)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * 1000.)
    if not o3d.io.write_triangle_mesh(str(output_path), mesh, write_vertex_normals=False):
        raise RuntimeError('网格写出失败')
    return {'vertices': len(mesh.vertices), 'triangles': len(mesh.triangles), 'valid_depth_pixels': counts,
            'source_image_size_wh': [depths[0].shape[1], depths[0].shape[0]],
            'world_input_unit': 'm', 'mesh_output_unit': 'mm', 'depth': 'expected_camera_z',
            'component_filter': 'NONE', 'open3d_version': o3d.__version__}
