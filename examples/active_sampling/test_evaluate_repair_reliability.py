"""合成输入只验证评价口径和输入约束，不代表 UTrustGS 的实验结果。"""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import evaluate_repair_reliability as evaluator


class RepairReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.directory = Path(self.temporary_directory.name)
        self.manifest_path = self.directory / "manifest.json"
        self.arrays = {
            "current_error": np.array([10.0, 10.0, 1.0, 1.0]),
            "proposal_error": np.array([20.0, 20.0, 2.0, 0.5]),
            "risk_score": np.array([0.9, 0.8, 0.2, 0.1]),
            "trust_score": np.array([0.9, 0.8, 0.2, 0.1]),
            "valid": np.ones(4, dtype=bool),
            "support": np.ones(4, dtype=bool),
            "admitted": np.array([True, True, False, False]),
        }
        self.manifest = {
            "schema_version": 1, "data_role": "synthetic_diagnostic", "dataset": "synthetic",
            "parent_sha256": "a" * 64, "generator_sha256": "b" * 64,
            "population": "all_candidates_before_admission",
            "records": [{"scene": "counterexample", "seed": 0, "view": "0", "channel": "rgb",
                         "arrays": "arrays.npz", "arrays_sha256": "0" * 64,
                         "reference_kind": "synthetic", "error_metric": "synthetic_absolute_error",
                         "risk_error_threshold": 5.0, "correct_error_threshold": 0.75,
                         "help_margin": 0.0, "trust_semantics": "helpful_probability"}],
        }

    def write_inputs(self):
        arrays_path = self.directory / "arrays.npz"
        np.savez(arrays_path, **self.arrays)
        self.manifest["records"][0]["arrays_sha256"] = evaluator.sha256_file(arrays_path)
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def evaluate(self):
        self.write_inputs()
        return evaluator.evaluate_manifest(self.manifest_path)

    def test_perfect_risk_detection_does_not_imply_reliable_repairs(self):
        record = self.evaluate()["records"][0]
        all_valid = record["domains"]["all_valid"]
        self.assertEqual(all_valid["risk_detection_ap"], 1.0)
        self.assertEqual(all_valid["helpful_ranking"]["trust_ap"], 0.25)
        self.assertEqual(all_valid["helpful_ranking"]["risk_ap"], 0.25)
        self.assertEqual(all_valid["correct_ranking"]["trust_ap"], 0.25)
        self.assertEqual(record["domains"]["accepted"]["helpful_rate"], 0.0)
        self.assertEqual(record["domains"]["accepted"]["mean_signed_gain"], -10.0)
        self.assertEqual(record["status"], "DIAGNOSTIC_ONLY")

    def test_correct_repair_can_fail_to_improve_current_render(self):
        self.arrays["current_error"][:] = 0.1
        self.arrays["proposal_error"][:] = 0.2
        domain = self.evaluate()["records"][0]["domains"]["all_valid"]
        self.assertEqual(domain["correct_rate"], 1.0)
        self.assertEqual(domain["helpful_rate"], 0.0)
        self.assertIsNone(domain["helpful_ranking"]["trust_ap"])
        self.assertIsNone(domain["risk_detection_ap"])

    def test_threshold_boundaries_and_help_margin_are_explicit(self):
        self.arrays["current_error"][:] = 1.0
        self.arrays["proposal_error"][:] = 0.75
        self.manifest["records"][0].update(risk_error_threshold=1.0, help_margin=0.25)
        domain = self.evaluate()["records"][0]["domains"]["all_valid"]
        self.assertEqual(domain["correct_rate"], 1.0)
        self.assertEqual(domain["helpful_rate"], 0.0)
        self.assertIsNone(domain["risk_detection_ap"])

    def test_helpful_uses_declared_addition_before_strict_comparison(self):
        self.arrays["current_error"][:] = 0.4
        self.arrays["proposal_error"][:] = 0.3
        self.manifest["records"][0]["help_margin"] = 0.1
        domain = self.evaluate()["records"][0]["domains"]["all_valid"]
        self.assertEqual(domain["helpful_rate"], 0.0)

    def test_tied_average_precision_is_grouped_and_permutation_invariant(self):
        labels = np.array([True, False, False, True])
        scores = np.array([0.8, 0.8, 0.2, 0.2])
        self.assertEqual(evaluator.average_precision(labels, scores), 0.5)
        for order in ([1, 0, 3, 2], [3, 1, 2, 0], [2, 3, 0, 1]):
            self.assertEqual(evaluator.average_precision(labels[order], scores[order]), 0.5)

    def test_selective_curve_includes_whole_ties(self):
        self.arrays["trust_score"][:] = 0.5
        first = self.evaluate()["records"][0]["selective_curves"]
        for name in first:
            self.assertEqual([row["actual_coverage"] for row in first[name]], [1.0] * 4)
        permutation = [3, 0, 2, 1]
        self.arrays = {name: values[permutation] for name, values in self.arrays.items()}
        self.assertEqual(first, self.evaluate()["records"][0]["selective_curves"])

    def test_calibration_matches_probability_event_and_includes_one(self):
        labels = np.array([False, True])
        metrics = evaluator.calibration(labels, np.array([0.2, 1.0]), "helpful")
        self.assertAlmostEqual(metrics["ece"], 0.1)
        self.assertAlmostEqual(metrics["brier"], 0.02)
        self.assertEqual(metrics["bins"][-1]["count"], 1)
        self.manifest["records"][0]["trust_semantics"] = "correct_probability"
        record = self.evaluate()["records"][0]
        self.assertEqual(record["domains"]["all_valid"]["calibration"]["event"], "correct")

    def test_empty_valid_masks_and_empty_arrays_return_null(self):
        for empty_arrays in (False, True):
            with self.subTest(empty_arrays=empty_arrays):
                if empty_arrays:
                    self.arrays = {name: values[:0] for name, values in self.arrays.items()}
                else:
                    for name in evaluator.MASK_NAMES:
                        self.arrays[name][:] = False
                record = self.evaluate()["records"][0]
                for domain in record["domains"].values():
                    self.assertEqual(domain["count"], 0)
                    for metric in ("correct_rate", "helpful_rate", "admission_coverage",
                                   "mean_signed_gain", "risk_detection_ap", "calibration"):
                        self.assertIsNone(domain[metric])
                self.assertIsNone(record["selective_curves"]["all_valid"][0]["actual_coverage"])

    def test_zero_and_full_admission_are_both_legal(self):
        for admit_all in (False, True):
            with self.subTest(admit_all=admit_all):
                self.arrays["admitted"][:] = admit_all
                record = self.evaluate()["records"][0]
                self.assertEqual(record["domains"]["all_valid"]["admission_coverage"], float(admit_all))
                empty_domain = "rejected" if admit_all else "accepted"
                self.assertEqual(record["domains"][empty_domain]["count"], 0)
                self.assertIsNone(record["domains"][empty_domain]["helpful_rate"])

    def test_score_semantics_skip_calibration_and_allow_nonprobability_values(self):
        self.arrays["trust_score"] = np.array([-5.0, 2.0, 9.0, 7.0])
        self.arrays["uncertainty_score"] = np.array([3.0, 2.0, 1.0, 0.0])
        self.manifest["records"][0]["trust_semantics"] = "score"
        record = self.evaluate()["records"][0]
        self.assertIsNone(record["domains"]["all_valid"]["calibration"])
        self.assertEqual(record["domains"]["all_valid"]["helpful_ranking"]["negative_uncertainty_ap"], 1.0)

    def test_mask_membership_boolean_shape_and_finite_checks(self):
        invalid_cases = [
            ("support", np.array([False, True, True, True])),
            ("valid", np.array([True, True, False, True])),
            ("valid", np.ones(4, dtype=int)),
            ("proposal_error", np.array([1.0, 2.0])),
            ("risk_score", np.array([0.9, np.nan, 0.2, 0.1])),
            ("proposal_error", np.array([1.0, 2.0, -1.0, 2.0])),
            ("trust_score", np.ones((2, 2))),
        ]
        for name, values in invalid_cases:
            with self.subTest(name=name, values=values):
                original = self.arrays[name]
                self.arrays[name] = values
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.arrays[name] = original

    def test_probability_outside_unit_interval_is_rejected(self):
        for value in (-0.01, 1.01):
            with self.subTest(value=value):
                self.arrays["trust_score"][0] = value
                with self.assertRaisesRegex(ValueError, "概率"):
                    self.evaluate()

    def test_input_hash_mismatch_is_rejected(self):
        self.write_inputs()
        self.manifest["records"][0]["arrays_sha256"] = "f" * 64
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256|sha256"):
            evaluator.evaluate_manifest(self.manifest_path)

    def test_duplicate_record_is_rejected(self):
        self.write_inputs()
        self.manifest["records"].append(copy.deepcopy(self.manifest["records"][0]))
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "重复"):
            evaluator.evaluate_manifest(self.manifest_path)

    def test_formal_data_and_undeclared_population_are_rejected(self):
        for field, value in (("data_role", "formal_test"), ("population", "accepted_only"),
                             ("parent_sha256", "not-a-hash"), ("schema_version", True)):
            with self.subTest(field=field):
                original = self.manifest[field]
                self.manifest[field] = value
                with self.assertRaises(ValueError):
                    self.evaluate()
                self.manifest[field] = original

    def test_reference_kind_must_match_channel_and_data_role(self):
        self.manifest["data_role"] = "development"
        with self.assertRaisesRegex(ValueError, "reference_kind"):
            self.evaluate()
        self.manifest["records"][0]["reference_kind"] = "independent_rgb"
        self.assertEqual(self.evaluate()["data_role"], "development")
        self.manifest["records"][0]["channel"] = "depth"
        with self.assertRaisesRegex(ValueError, "reference_kind"):
            self.evaluate()

    def test_cli_report_has_provenance_and_never_overwrites_files(self):
        self.write_inputs()
        output = self.directory / "report.json"
        command = [sys.executable, "-B", evaluator.__file__, str(self.manifest_path), "--output", str(output)]
        completed = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report_bytes = output.read_bytes()
        report = json.loads(report_bytes)
        self.assertEqual(report["aggregation"], "per_record_only")
        self.assertEqual(report["manifest_sha256"], evaluator.sha256_file(self.manifest_path))
        self.assertEqual(report["evaluator_sha256"], evaluator.sha256_file(Path(evaluator.__file__)))
        self.assertNotIn("NaN", report_bytes.decode("utf-8"))
        for destination in (output, self.manifest_path, self.directory / "arrays.npz"):
            with self.subTest(destination=destination):
                before = destination.read_bytes()
                completed = subprocess.run(command[:-1] + [str(destination)], capture_output=True, text=True)
                self.assertEqual(completed.returncode, 2)
                self.assertEqual(destination.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
