"""缺陷监督应保留零误差、同误差同标签及样本打包不变性。"""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")

import torch

from gddn_trainer import ThreeStageTrainer


class ProposalDefectTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def targets(self, error, *, geometry_aware=False, rank_blend=0.5,
                all_pixels=False):
        trainer = ThreeStageTrainer.__new__(ThreeStageTrainer)
        trainer.raw_model = SimpleNamespace()
        trainer.proposal_warp_error_rank_blend = rank_blend
        if geometry_aware:
            trainer.teacher_warp_margin = 0.01
            trainer.proposal_opportunity_top_fraction = 0.30
            trainer.proposal_repairability_empirical_max_weight = 1.0
            trainer.proposal_repairability_empirical_transition_epochs = 1
        if all_pixels:
            trainer.proposal_opportunity_top_fraction = 1.0
        target = torch.full((error.shape[0], 3, *error.shape[-2:]), 0.5)
        mask = torch.ones_like(error)
        return trainer._compute_teacher_effect_targets(
            teacher_rgb=target,
            render_rgb=target + error,
            target_rgb=target,
            depth_render=mask,
            support_mask=mask,
            support_confidence=mask,
            frontier_mask=mask,
            verification_prior=mask,
        )

    def test_zero_error_has_zero_defect_for_each_rank_blend(self):
        for profile in (False, True):
            for blend in (0.0, 0.5, 1.0):
                for size in (1, 16):
                    with self.subTest(profile=profile, blend=blend, size=size):
                        output = self.targets(torch.zeros(1, 1, size, size),
                                              geometry_aware=profile, rank_blend=blend)
                        for key in ("proposal_warp_error_target", "proposal_warp_error_rank_target"):
                            self.assertEqual(torch.count_nonzero(output[key]).item(), 0, key)

    def test_equal_positive_errors_have_equal_labels(self):
        for profile in (False, True):
            with self.subTest(profile=profile):
                output = self.targets(torch.full((1, 1, 16, 16), 0.125), geometry_aware=profile)
                torch.testing.assert_close(output["proposal_warp_error_target"],
                                           torch.full((1, 1, 16, 16), 0.75), rtol=0, atol=1e-6)

    def test_zero_sample_does_not_depend_on_batch_companion(self):
        for profile in (False, True):
            with self.subTest(profile=profile):
                zero = torch.zeros(1, 1, 16, 16)
                positive = torch.full_like(zero, 0.125)
                alone = self.targets(zero, geometry_aware=profile)
                packed = self.targets(torch.cat((zero, positive)), geometry_aware=profile)
                for key in ("proposal_warp_error_target", "proposal_repairability_target"):
                    torch.testing.assert_close(alone[key], packed[key][:1], rtol=0, atol=0)

    def test_tied_label_map_is_equivariant_to_horizontal_flip(self):
        error = torch.tensor([0.0, 0.125, 0.125, 0.5]).reshape(1, 1, 1, 4).repeat(1, 1, 4, 1)
        output = self.targets(error, all_pixels=True)
        flipped = self.targets(error.flip(-1), all_pixels=True)
        for key in ("proposal_warp_error_target", "proposal_warp_error_rank_target"):
            torch.testing.assert_close(output[key], flipped[key].flip(-1), rtol=0, atol=0)

    def test_zero_error_pixels_remain_zero_among_nonzero_errors(self):
        error = torch.tensor([0.0, 0.0, 0.125, 0.5]).reshape(1, 1, 1, 4)
        output = self.targets(error, all_pixels=True)
        for key in ("proposal_warp_error_target", "proposal_warp_error_rank_target"):
            self.assertEqual(torch.count_nonzero(output[key][error == 0]).item(), 0, key)

    def test_distinct_positive_errors_preserve_existing_relative_scale(self):
        error = torch.tensor([0.125, 0.25, 0.375, 0.5]).reshape(1, 1, 1, 4)
        output = self.targets(error, all_pixels=True)
        expected = torch.tensor([0.5 * 0.125 / 0.4625,
                                 0.5 * 0.25 / 0.4625 + 1 / 6,
                                 0.5 * 0.375 / 0.4625 + 1 / 3,
                                 1.0]).reshape_as(error)
        torch.testing.assert_close(output["proposal_warp_error_target"], expected, rtol=0, atol=1e-6)

    def test_single_positive_error_keeps_full_relative_priority(self):
        output = self.targets(torch.full((1, 1, 1, 1), 0.125))
        self.assertEqual(output["proposal_warp_error_target"].item(), 1.0)

    def test_rank_disabled_preserves_positive_magnitude_label(self):
        output = self.targets(torch.full((1, 1, 16, 16), 0.125), rank_blend=0.0)
        self.assertTrue(torch.equal(output["proposal_warp_error_target"], torch.ones(1, 1, 16, 16)))
        self.assertEqual(torch.count_nonzero(output["proposal_warp_error_rank_target"]).item(), 0)


if __name__ == "__main__":
    unittest.main()
