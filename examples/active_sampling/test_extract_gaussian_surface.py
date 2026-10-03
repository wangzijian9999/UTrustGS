"""冻结提取器的解析表面与非法输入检查，不用真实扫描。"""
import unittest
import numpy as np

from extract_gaussian_surface import fuse_depths, validate_camera, fusion_intrinsic


SETTINGS = dict(voxel_m=.002, sdf_trunc_m=.008, depth_max_m=3., alpha_min=.5)


class SurfaceExtractionTests(unittest.TestCase):
    def setUp(self):
        self.intrinsic = np.array([[160., 0, 32.], [0, 160., 32.], [0, 0, 1.]])
        yy, xx = np.mgrid[:64, :64]
        self.rays = np.stack([(xx + .5 - 32) / 160, (yy + .5 - 32) / 160, np.ones_like(xx)], -1)

    def test_tilted_plane_world_transform(self):
        # world平面 z=0.6+0.2x；非零相机中心检查外参方向。
        depths, views = [], []
        for center in [np.array([0., 0., 0.]), np.array([.02, 0., .03])]:
            view = np.eye(4)
            view[:3, 3] = -center
            depths.append((.6 + .2 * center[0] - center[2]) / (1 - .2 * self.rays[..., 0]))
            views.append(view)
        mesh, counts = fuse_depths(depths, [np.ones((64, 64))] * 2, views,
                                  [self.intrinsic] * 2, **SETTINGS)
        vertices = np.asarray(mesh.vertices)
        errors = np.abs(vertices[:, 2] - .6 - .2 * vertices[:, 0]) / np.sqrt(1.04)
        self.assertLess(errors.max(), .002)
        self.assertEqual(counts, [4096, 4096])

    def test_sphere_visible_surface(self):
        center = np.array([0., 0., .6])
        radius = .08
        a = np.square(self.rays).sum(-1)
        b = -2 * (self.rays @ center)
        c = center @ center - radius ** 2
        discriminant = b*b - 4*a*c
        valid = discriminant > 0
        depth = np.where(valid, (-b - np.sqrt(np.maximum(discriminant, 0))) / (2*a), 0)
        mesh, _ = fuse_depths([depth], [valid.astype(float)], [np.eye(4)], [self.intrinsic], **SETTINGS)
        errors = np.abs(np.linalg.norm(np.asarray(mesh.vertices) - center, axis=1) - radius)
        self.assertLess(np.quantile(errors, .95), .004)

    def test_empty_and_invalid_input_rejected(self):
        with self.assertRaisesRegex(ValueError, '为空'):
            fuse_depths([np.ones((64, 64))], [np.zeros((64, 64))], [np.eye(4)], [self.intrinsic], **SETTINGS)
        invalid = np.ones((64, 64)); invalid[0, 0] = np.nan
        with self.assertRaises(ValueError):
            fuse_depths([invalid], [np.ones((64, 64))], [np.eye(4)], [self.intrinsic], **SETTINGS)
        with self.assertRaises(ValueError):
            validate_camera(np.eye(4) * 1000, self.intrinsic)
        bad = self.intrinsic.copy(); bad[0, 0] = -1
        with self.assertRaises(ValueError):
            validate_camera(np.eye(4), bad)

    def test_native_backprojection_with_rotation_resolves_half_pixel(self):
        import open3d as o3d
        angle = .3
        rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0],
                             [-np.sin(angle), 0, np.cos(angle)]])
        view = np.eye(4); view[:3, :3] = rotation; view[:3, 3] = [.1, -.05, .02]
        depth = o3d.geometry.Image(np.ones((64, 64), dtype=np.float32) * .6)
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            depth, fusion_intrinsic(self.intrinsic, 64, 64), view, depth_scale=1., depth_trunc=3.)
        expected = (self.rays.reshape(-1, 3) * np.float32(.6) - view[:3, 3]) @ rotation
        np.testing.assert_allclose(np.asarray(cloud.points), expected, atol=1e-8, rtol=0)
        wrong = o3d.camera.PinholeCameraIntrinsic(64, 64, 160., 160., 32., 32.)
        shifted = o3d.geometry.PointCloud.create_from_depth_image(depth, wrong, view, depth_scale=1., depth_trunc=3.)
        self.assertGreater(np.linalg.norm(np.asarray(shifted.points) - expected, axis=1).min(), .002)


if __name__ == '__main__':
    unittest.main()
