"""核验显式证据值与缺失字段在原生监督标签中保持不同语义。"""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")

import torch

from gddn_trainer import ThreeStageTrainer


class TeacherVerificationPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def targets(self, prior, *, height=32, width=32, batch_size=1,
                teacher_value=0.5, geometry_aware=True):
        # 不构造模型或优化器；直接测试完整生产类的原生标签方法。
        trainer = ThreeStageTrainer.__new__(ThreeStageTrainer)
        trainer.raw_model = SimpleNamespace()
        if geometry_aware:
            trainer.teacher_warp_margin = 0.01
            trainer.proposal_opportunity_top_fraction = 0.30
            trainer.proposal_repairability_empirical_max_weight = 1.0
            trainer.proposal_repairability_empirical_transition_epochs = 1
        rgb_shape = (batch_size, 3, height, width)
        mask = torch.ones(batch_size, 1, height, width)
        return trainer._compute_teacher_effect_targets(
            teacher_rgb=torch.full(rgb_shape, teacher_value),
            render_rgb=torch.full(rgb_shape, 0.1),
            target_rgb=torch.full(rgb_shape, 0.5),
            depth_render=mask,
            support_mask=mask,
            support_confidence=mask,
            frontier_mask=mask,
            verification_prior=prior,
        )

    def test_explicit_zero_blocks_verification_and_admission(self):
        for geometry_aware in (False, True):
            for shape in ((1, 32, 32), (1, 1, 32, 32), (1, 1, 4, 5), (1, 2, 7, 9)):
                with self.subTest(geometry_aware=geometry_aware, shape=shape):
                    output = self.targets(torch.zeros(shape), geometry_aware=geometry_aware)
                    for key in ("proposal_verification_target", "proposal_repairability_target",
                                "proposal_acceptance_target", "proposal_current_acceptance_target"):
                        self.assertEqual(torch.count_nonzero(output[key]).item(), 0, key)

    def test_missing_prior_preserves_support_fallback(self):
        for geometry_aware in (False, True):
            with self.subTest(geometry_aware=geometry_aware):
                output = self.targets(None, geometry_aware=geometry_aware)
                self.assertTrue(torch.equal(output["proposal_verification_target"], torch.ones(1, 1, 32, 32)))
                self.assertGreater(output["proposal_acceptance_target"].mean().item(), 0.9)

    def test_positive_prior_retains_positive_admission(self):
        output = self.targets(torch.ones(1, 1, 4, 5), height=12, width=16)
        torch.testing.assert_close(output["proposal_verification_target"], torch.ones(1, 1, 12, 16), rtol=0, atol=1e-6)
        self.assertAlmostEqual(output["proposal_acceptance_target"].mean().item(), 0.975, places=5)

    def test_subunit_total_evidence_is_not_replaced_with_full_support(self):
        output = self.targets(torch.full((1, 1, 4, 5), 1e-4), height=12, width=16)
        torch.testing.assert_close(output["proposal_verification_target"], torch.full((1, 1, 12, 16), 1e-4))
        self.assertGreater(output["proposal_acceptance_target"].mean().item(), 0.0)
        self.assertLess(output["proposal_acceptance_target"].max().item(), 1e-3)

    def test_batch_keeps_zero_evidence_separate_from_positive_evidence(self):
        prior = torch.cat((torch.zeros(1, 1, 32, 32), torch.ones(1, 1, 32, 32)))
        output = self.targets(prior, batch_size=2)
        self.assertEqual(output["proposal_acceptance_target"][0].max().item(), 0.0)
        self.assertGreater(output["proposal_acceptance_target"][1].mean().item(), 0.9)

    def test_invalid_or_negative_evidence_does_not_create_positive_support(self):
        for value in (float("nan"), float("-inf"), -1.0):
            with self.subTest(value=value):
                output = self.targets(torch.full((1, 1, 32, 32), value))
                self.assertTrue(torch.isfinite(output["proposal_verification_target"]).all())
                self.assertEqual(output["proposal_verification_target"].max().item(), 0.0)
                self.assertEqual(output["proposal_acceptance_target"].max().item(), 0.0)

    def test_positive_support_does_not_admit_harmful_teacher(self):
        output = self.targets(torch.ones(1, 1, 32, 32), teacher_value=1.0)
        self.assertEqual(output["proposal_acceptance_target"].max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
