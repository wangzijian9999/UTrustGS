"""准入监督必须评价实际提议，而非假定基底和未应用残差组成的另一幅图。"""
import os
from types import SimpleNamespace
import unittest

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")

import torch

from gddn_trainer import ThreeStageTrainer


class ProposalCandidateIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def targets(self, proposal, residual=None, empirical=None):
        trainer = ThreeStageTrainer.__new__(ThreeStageTrainer)
        trainer.raw_model = SimpleNamespace()
        trainer.teacher_warp_margin = 0.01
        trainer.proposal_opportunity_top_fraction = 1.0
        trainer.proposal_empirical_patch_kernel_size = 1
        trainer.proposal_repairability_empirical_max_weight = 1.0
        trainer.proposal_repairability_empirical_transition_epochs = 1
        image = torch.full((1, 3, 16, 16), proposal, requires_grad=True)
        mask = torch.ones(1, 1, 16, 16)
        output = trainer._compute_teacher_effect_targets(
            teacher_rgb=image,
            render_rgb=torch.full_like(image, 0.6),
            target_rgb=torch.full_like(image, 0.5),
            depth_render=mask,
            support_mask=mask,
            support_confidence=mask,
            frontier_mask=mask,
            verification_prior=mask,
            proposal_residual_rgb=None if residual is None else torch.full_like(image, residual),
            empirical_candidate_rgbs=None if empirical is None else [torch.full_like(image, empirical)],
        )
        return output

    def test_harmful_actual_proposal_is_not_accepted_for_good_unapplied_residual(self):
        # 实际warp=.85、gate=.5、raw residual=-.1，得到proposal=.8。
        # 错误地用render=.6加raw residual会得到完全正确但并未消费的.5。
        output = self.targets(0.8, -0.1)
        self.assertEqual(torch.count_nonzero(output["proposal_acceptance_target"]).item(), 0)
        torch.testing.assert_close(output["proposal_candidate_error"],
                                   torch.full((1, 1, 16, 16), 0.3), rtol=0, atol=1e-6)

    def test_good_actual_proposal_is_not_rejected_for_wrong_residual_base(self):
        # 实际warp=.4、gate=.5、raw residual=.2，得到proposal=.5。
        output = self.targets(0.5, 0.2)
        self.assertGreater(float(output["proposal_acceptance_target"].mean()), 0.89)
        self.assertEqual(torch.count_nonzero(output["proposal_candidate_error"]).item(), 0)

    def test_current_acceptance_depends_on_actual_rgb_not_raw_residual(self):
        reference = self.targets(0.55)
        for residual in (-0.4, 0.0, 0.4):
            with self.subTest(residual=residual):
                output = self.targets(0.55, residual)
                for key in ("proposal_candidate_error", "proposal_acceptance_target"):
                    torch.testing.assert_close(output[key], reference[key], rtol=0, atol=0)

    def test_zero_applied_update_is_not_a_successful_repair(self):
        self.assertEqual(torch.count_nonzero(
            self.targets(0.6, -0.1)["proposal_acceptance_target"]).item(), 0)

    def test_better_alternative_does_not_make_current_harmful_proposal_acceptable(self):
        output = self.targets(0.8, -0.1, empirical=0.5)
        self.assertGreater(float(output["proposal_repairability_target"].mean()), 0.89)
        self.assertEqual(torch.count_nonzero(output["proposal_acceptance_target"]).item(), 0)

    def test_supervision_targets_do_not_backpropagate_into_proposal(self):
        output = self.targets(0.55, 0.2)
        self.assertFalse(output["proposal_acceptance_target"].requires_grad)
        self.assertFalse(output["proposal_candidate_error"].requires_grad)


if __name__ == "__main__":
    unittest.main()
