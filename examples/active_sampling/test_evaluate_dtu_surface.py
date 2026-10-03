"""表面评价的解析几何、DTU边界语义与来源约束检查。"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.io import savemat

from evaluate_dtu_surface import (bounded_distances, distance_statistics, evaluate_surface,
                                  observation_membership, reduce_surface_samples, run_manifest,
                                  sample_triangle_surface, sha256_file)


class DTUSurfaceTests(unittest.TestCase):
    def setUp(self):
        self.vertices = np.array([[0., 0., 1.], [1., 0., 1.], [0., 1., 1.]])
        self.faces = np.array([[0, 1, 2]])
        self.bounding_box = np.array([[-5., -5., -5.], [5., 5., 5.]])

    def test_triangle_sampling_stays_on_surface_and_is_reproducible(self):
        points = sample_triangle_surface(self.vertices, self.faces, 10000, 7)
        np.testing.assert_array_equal(points, sample_triangle_surface(self.vertices, self.faces, 10000, 7))
        np.testing.assert_allclose(points[:, 2], 1, atol=1e-15)
        self.assertTrue(np.all(points[:, :2] >= 0))
        self.assertTrue(np.all(points[:, 0] + points[:, 1] <= 1 + 1e-15))
        np.testing.assert_allclose(points.mean(0), [1/3, 1/3, 1], atol=0.015)

    def test_sampling_rejects_centers_degenerate_and_invalid_mesh(self):
        for faces in (np.empty((0, 3), dtype=int), np.array([[0, 0, 0]]), np.array([[0, 1, 3]])):
            with self.subTest(faces=faces.tolist()), self.assertRaises(ValueError):
                sample_triangle_surface(self.vertices, faces, 10, 0)

    def test_density_reduction_preserves_separation(self):
        points = np.array([[0., 0., 0.], [0.1, 0., 0.], [1., 0., 0.]])
        reduced = reduce_surface_samples(points, 0.2, 7)
        self.assertEqual(len(reduced), 2)
        self.assertGreater(np.linalg.norm(reduced[0] - reduced[1]), 0.2)
        np.testing.assert_array_equal(reduced, reduce_surface_samples(points, 0.2, 7))

    def test_nearest_distance_and_outside_cell_default(self):
        source = np.array([[0., 0., 0.], [1., 0., 0.], [-6., 0., 0.]])
        result = bounded_distances(np.array([[0., 0., 0.]]), source, self.bounding_box)
        np.testing.assert_allclose(result, [0, 1, 60])

    def test_official_last_cell_extends_past_bounding_box(self):
        points = np.array([[50., 0., 0.]])
        np.testing.assert_array_equal(bounded_distances(points, points, self.bounding_box), [0])

    def test_existing_neighbor_distance_is_not_clamped_to_sixty(self):
        bounding_box = np.array([[0., 0., 0.], [1., 1., 1.]])
        result = bounded_distances(np.array([[119., 119., 119.]]), np.zeros((1, 3)), bounding_box)
        self.assertAlmostEqual(result[0], np.sqrt(3) * 119)
        empty = bounded_distances(np.empty((0, 3)), np.zeros((1, 3)), bounding_box)
        np.testing.assert_array_equal(empty, [60])

    def test_matlab_rounding_one_based_axes_and_mask_bounds(self):
        mask = np.zeros((3, 2, 2), dtype=bool)
        mask[0, 0, 0] = True
        mask[2, 0, 0] = True
        points = np.array([[-0.5, 0, 0], [-0.5001, 0, 0], [0.5, 0, 0],
                           [1.5, 0, 0], [2.5, 0, 0], [0, 1, 0]])
        actual = observation_membership(points, np.array([[0, 0, 0], [2, 1, 1]]), 1., mask)
        np.testing.assert_array_equal(actual, [True, False, False, True, False, False])

    def test_statistics_keep_outliers_and_strict_twenty_boundary(self):
        result = distance_statistics(np.array([1., 19., 20., 60., 0.]),
                                     np.array([True, True, True, True, False]))
        self.assertEqual(result["trimmed_mean_mm"], 10)
        self.assertEqual(result["untrimmed_mean_mm"], 25)
        self.assertEqual(result["outlier_fraction"], 0.5)
        self.assertEqual(result["outside_domain_count"], 1)
        self.assertEqual(result["trimmed_sample_variance_mm2"], 162)

    def test_empty_domain_is_null_not_zero_error(self):
        result = distance_statistics(np.array([1.]), np.array([False]))
        self.assertIsNone(result["trimmed_mean_mm"])
        self.assertIsNone(result["outlier_fraction"])

    def test_surface_identity_and_strict_plane_domain(self):
        samples = reduce_surface_samples(sample_triangle_surface(self.vertices, self.faces, 100, 7), 0.2, 7)
        reference = np.vstack([samples, [0., 0., 0.]])
        result = evaluate_surface(self.vertices, self.faces, reference, self.bounding_box, 1.,
                                  np.ones((11, 11, 11)), [0, 0, 1, 0], 100, 7)
        self.assertEqual(result["accuracy"]["trimmed_mean_mm"], 0)
        self.assertEqual(result["completeness"]["trimmed_mean_mm"], 0)
        self.assertEqual(result["completeness"]["outside_domain_count"], 1)

    def test_parallel_plane_offset_has_known_bidirectional_error(self):
        reference = reduce_surface_samples(sample_triangle_surface(self.vertices, self.faces, 100, 7), 0.2, 7)
        shifted = self.vertices + np.array([0., 0., 2.])
        result = evaluate_surface(shifted, self.faces, reference, self.bounding_box, 1.,
                                  np.ones((11, 11, 11)), [0, 0, 1, 0], 100, 7)
        self.assertAlmostEqual(result["accuracy"]["trimmed_mean_mm"], 2.)
        self.assertAlmostEqual(result["completeness"]["trimmed_mean_mm"], 2.)

    def fixture(self, directory):
        vertices = np.array([tuple(point) for point in self.vertices], dtype=[("x", "f8"), ("y", "f8"), ("z", "f8")])
        faces = np.array([(self.faces[0],)], dtype=[("vertex_indices", "i4", (3,))])
        PlyData([PlyElement.describe(vertices, "vertex"), PlyElement.describe(faces, "face")]).write(str(directory / "mesh.ply"))
        PlyData([PlyElement.describe(vertices, "vertex")]).write(str(directory / "reference.ply"))
        savemat(directory / "mask.mat", {"BB": self.bounding_box, "Res": 1., "ObsMask": np.ones((11, 11, 11))})
        savemat(directory / "plane.mat", {"P": np.array([0, 0, 1, 0])[:, None]})
        manifest = {"schema_version": 1, "data_role": "development", "scan": 1,
                    "representation": "triangle_surface", "coordinate_frame": "dtu_cal18_reference_mm",
                    "extraction_provenance": "仅用于接口测试的合成三角形，未使用DTU真实数据",
                    "coordinate_provenance": "合成毫米坐标，不构成实际DTU配准证据",
                    "sample_count": 100, "seed": 7,
                    "inputs": {name: {"path": filename, "sha256": sha256_file(directory / filename)}
                               for name, filename in (("mesh", "mesh.ply"), ("reference", "reference.ply"),
                                                      ("observation_mask", "mask.mat"), ("ground_plane", "plane.mat"))}}
        path = directory / "manifest.json"
        path.write_text(json.dumps(manifest))
        return path, manifest

    def test_manifest_rejects_test_role_hash_mismatch_and_point_cloud(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, manifest = self.fixture(directory)
            for field, invalid in (("data_role", "test"), ("scan", 2), ("representation", "gaussian_centers")):
                altered = dict(manifest, **{field: invalid})
                path.write_text(json.dumps(altered))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    run_manifest(path)
            manifest["inputs"]["mesh"]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "哈希不匹配"):
                run_manifest(path)
            manifest["inputs"]["mesh"] = manifest["inputs"]["reference"]
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "三角面"):
                run_manifest(path)

    def test_cli_records_inputs_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path, manifest = self.fixture(directory)
            output = directory / "result.json"
            command = [sys.executable, "-B", str(Path(__file__).with_name("evaluate_dtu_surface.py")), str(path), "--output", str(output)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(output.read_text())
            self.assertEqual(result["inputs"], manifest["inputs"])
            self.assertEqual(result["status"], "DIAGNOSTIC_ONLY_NOT_OFFICIAL_EQUIVALENCE")
            digest = sha256_file(output)
            repeated = subprocess.run(command, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual(sha256_file(output), digest)


if __name__ == "__main__":
    unittest.main()
