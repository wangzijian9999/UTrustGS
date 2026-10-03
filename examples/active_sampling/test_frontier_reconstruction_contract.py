"""实际提议、独立参考与停止梯度标签之间的修复监督契约。"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")

import torch
from gddn_trainer import ThreeStageTrainer


class FrontierReconstructionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.trainer = ThreeStageTrainer.__new__(ThreeStageTrainer)
        self.trainer.raw_model = SimpleNamespace()
        self.trainer.device = torch.device("cpu")
        self.trainer.frontier_residual_anchor_weight = 0.0
        self.trainer.frontier_residual_recon_alpha = 0.75
        self.trainer.proposal_opportunity_top_fraction = 1.0
        self.trainer.proposal_empirical_patch_kernel_size = 1
        self.trainer.proposal_repairability_empirical_max_weight = 1.0
        self.trainer.proposal_repairability_empirical_transition_epochs = 1

    def targets(self, actual, raw=None):
        mask = torch.ones(actual.shape[0], 1, *actual.shape[-2:])
        return self.trainer._compute_teacher_effect_targets(
            teacher_rgb=actual.detach(),
            render_rgb=torch.full_like(actual, 0.625),
            target_rgb=torch.full_like(actual, 0.5),
            depth_render=mask, support_mask=mask, support_confidence=mask,
            frontier_mask=mask, verification_prior=mask,
            proposal_residual_rgb=raw,
        )

    def loss(self, actual, targets=None):
        return self.trainer._compute_frontier_residual_proposal_loss(
            self.targets(actual) if targets is None else targets,
            proposal_rgb=actual,
        )

    def test_perfect_actual_rgb_has_zero_loss_for_different_raw_gate(self):
        for raw_value, gate_value in ((0.25, 0.5), (0.5, 0.25)):
            with self.subTest(raw=raw_value):
                raw = torch.tensor(raw_value, requires_grad=True)
                gate = torch.tensor(gate_value, requires_grad=True)
                actual = (0.375 + raw * gate) * torch.ones(1, 3, 16, 16)
                loss = self.loss(actual, self.targets(actual, raw.expand_as(actual)))
                self.assertEqual(float(loss.detach()), 0.0)
                self.assertEqual(tuple(float(v) for v in torch.autograd.grad(loss, (raw, gate))), (0.0, 0.0))

    def test_wrong_actual_rgb_learns_despite_detached_self_oracle(self):
        raw = torch.tensor(0.125, requires_grad=True)
        gate = torch.tensor(0.5, requires_grad=True)
        actual = (0.375 + raw * gate) * torch.ones(1, 3, 16, 16)
        targets = self.targets(actual, raw.expand_as(actual))
        torch.testing.assert_close(targets["proposal_oracle_residual_target"], actual.detach() - 0.625)
        loss = self.loss(actual, targets)
        self.assertAlmostEqual(float(loss.detach()), 0.0478515625, places=6)
        raw_gradient, gate_gradient = torch.autograd.grad(loss, (raw, gate))
        self.assertAlmostEqual(float(raw_gradient), -0.390625, places=6)
        self.assertAlmostEqual(float(gate_gradient), -0.09765625, places=6)

    def test_raw_and_oracle_values_do_not_change_actual_reconstruction_loss(self):
        actual = torch.full((1, 3, 16, 16), 0.4375, requires_grad=True)
        targets = self.targets(actual)
        reference = self.loss(actual, targets)
        targets["proposal_residual_rgb"] = torch.full_like(actual, 4.0)
        targets["proposal_oracle_residual_target"] = torch.full_like(actual, -3.0)
        torch.testing.assert_close(self.loss(actual, targets), reference, rtol=0, atol=0)

    def test_labels_and_loss_weights_are_not_trainable_shortcuts(self):
        actual = torch.full((1, 3, 16, 16), 0.4375, requires_grad=True)
        targets = self.targets(actual)
        leaves = []
        for key in ("target_rgb", "render_rgb", "proposal_training_mask", "proposal_residual_training_target"):
            targets[key] = targets[key].detach().clone().requires_grad_()
            leaves.append(targets[key])
        self.loss(actual, targets).backward()
        self.assertGreater(float(actual.grad.abs().sum()), 0.0)
        self.assertTrue(all(value.grad is None for value in leaves))

    def test_anchor_preserves_actual_render_not_zero_raw(self):
        actual = torch.full((1, 3, 16, 16), 0.5, requires_grad=True)
        targets = self.targets(actual)
        targets["proposal_training_mask"][:, :, :, 8:] = 0
        targets["proposal_residual_training_target"] = torch.zeros(1, 1, 16, 16)
        self.trainer.frontier_residual_anchor_weight = 2.0
        self.assertAlmostEqual(float(self.loss(actual, targets).detach()), 2 * (0.75 * 0.125 + 0.25 * 0.125 ** 2), places=6)

    def test_empty_proposal_mask_does_not_invent_supervision(self):
        actual = torch.full((1, 3, 16, 16), 0.4, requires_grad=True)
        targets = self.targets(actual)
        targets["proposal_training_mask"] = torch.zeros(1, 1, 16, 16)
        self.assertEqual(float(self.loss(actual, targets).detach()), 0.0)

    def test_masked_out_pixels_receive_no_reconstruction_gradient(self):
        actual = torch.full((1, 3, 16, 16), 0.4, requires_grad=True)
        targets = self.targets(actual)
        targets["proposal_training_mask"][:, :, :, 8:] = 0
        targets["proposal_residual_training_target"] = torch.ones(1, 1, 16, 16)
        self.loss(actual, targets).backward()
        self.assertEqual(float(actual.grad[:, :, :, 8:].abs().sum()), 0.0)
        self.assertGreater(float(actual.grad[:, :, :, :8].abs().sum()), 0.0)

    def test_unrepaired_sample_cannot_drop_out_when_other_sample_improves(self):
        actual = torch.full((2, 3, 16, 16), 0.75, requires_grad=True)
        targets = self.targets(actual)
        repairability = torch.ones(2, 1, 16, 16)
        repairability[1] = 0
        targets["proposal_residual_training_target"] = repairability
        self.loss(actual, targets).backward()
        self.assertGreater(float(actual.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(actual.grad[1].abs().sum()), 0.0)

    def test_resizing_keeps_live_prediction_gradient(self):
        small = torch.full((1, 3, 8, 8), 0.4375, requires_grad=True)
        targets = self.targets(torch.full((1, 3, 16, 16), 0.9))
        loss = self.loss(small, targets)
        self.assertAlmostEqual(float(loss.detach()), 0.0478515625, places=6)
        loss.backward()
        self.assertTrue(bool((small.grad < 0).all()))

    def test_clamp_saturation_remains_an_explicit_boundary(self):
        raw = torch.tensor(0.5, requires_grad=True)
        actual = (0.875 + raw).clamp(0, 1) * torch.ones(1, 3, 16, 16)
        loss = self.loss(actual)
        self.assertGreater(float(loss.detach()), 0.0)
        self.assertEqual(float(torch.autograd.grad(loss, raw)[0]), 0.0)

    def test_closed_gate_blocks_raw_but_not_nonzero_residual_gate_gradient(self):
        raw = torch.tensor(0.125, requires_grad=True)
        gate = torch.tensor(0.0, requires_grad=True)
        actual = (0.375 + raw * gate) * torch.ones(1, 3, 16, 16)
        raw_gradient, gate_gradient = torch.autograd.grad(self.loss(actual), (raw, gate))
        self.assertEqual(float(raw_gradient), 0.0)
        self.assertLess(float(gate_gradient), 0.0)

    def test_three_training_call_sites_supply_live_rgb_explicitly(self):
        tree = ast.parse(Path(__file__).with_name("gddn_trainer.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "_compute_frontier_residual_proposal_loss"]
        supplied = [next(keyword.value for keyword in call.keywords if keyword.arg == "proposal_rgb") for call in calls]
        self.assertEqual(len(supplied), 3)
        self.assertTrue(all(isinstance(value, ast.Name) for value in supplied))
        self.assertEqual(sorted(value.id for value in supplied), ["mean_rgb", "mean_rgb", "rgb_prediction_chunk"])

    def test_stage3_ibc_and_joint_entry_update_native_head(self):
        # 用轻量骨干夹具隔离接线；原生训练入口、提议组装、损失及AdamW实际执行。
        from gddn_controlnet import FrontierProposalHead, GDDN_ControlNet

        class EntryFixture(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.frontier_proposal_head = FrontierProposalHead()
                self.noise_level = torch.nn.Parameter(torch.tensor(0.0))
                self.epistemic_head = torch.nn.Identity()
                self.aleatoric_head = torch.nn.Identity()

            def forward(self, **inputs):
                render = inputs["rgb_render"]
                mask = torch.ones_like(render[:, :1])
                proposal = GDDN_ControlNet._build_explicit_frontier_proposal(
                    self, teacher_rgb=render, base_rgb=render,
                    support_confidence=mask, support_projection_mask=mask,
                    inpaint_mask=mask, frontier_mask=mask,
                    verification_prior=mask, accum_render=mask,
                )
                return SimpleNamespace(
                    noise_pred=self.noise_level.expand_as(inputs["noisy_latents"]),
                    predicted_proposal_image=proposal["proposal_raw_rgb"],
                    predicted_proposal_residual=proposal["proposal_residual_rgb"],
                    uncertainty_epistemic=mask * 0.1,
                    uncertainty_aleatoric=mask * 0.1,
                )

        class FinishedOneUpdate(Exception):
            pass

        for ibc_enabled in (False, True):
            with self.subTest(ibc=ibc_enabled):
                torch.manual_seed(20260930)
                model = EntryFixture()
                rgb = torch.full((1, 3, 16, 16), 0.625)
                mask = torch.ones(1, 1, 16, 16)
                batch = dict(rgb_render=rgb, target_rgb=torch.full_like(rgb, 0.5),
                             depth_render=mask, normal_render=rgb,
                             warp_valid_mask=mask, warp_support_confidence=mask,
                             frontier_mask=mask, verification_prior=mask,
                             sparse_images=rgb.unsqueeze(1), sparse_poses=torch.eye(4)[None, None],
                             target_pose=torch.eye(4)[None], phase1_episode_conditioned=torch.ones(1))
                trainer = ThreeStageTrainer(model, [batch], device="cpu")
                trainer.ibc_enabled = ibc_enabled
                trainer.stage3_frontier_residual_weight = 1.0
                trainer.frontier_residual_anchor_weight = 0.0
                trainer.stage3_primary_metric_use_final_sample = False
                trainer.final_sample_supervision_weight = 0.0
                trainer.trainable_sample_supervision_weight = 0.0
                trainer.lambda_vpdt = trainer.lambda_calib = trainer.lambda_var_preserve = 0.0
                trainer.relative_teacher_effect_weight = 0.0
                trainer.stage3_variance_weight = 0.0
                trainer.proposal_empirical_candidate_count = 1
                before = model.frontier_proposal_head.output_head.weight.detach().clone()
                original_step = torch.optim.AdamW.step

                def step_and_stop(optimizer, *args, **kwargs):
                    original_step(optimizer, *args, **kwargs)
                    raise FinishedOneUpdate()

                diffusion_inputs = (torch.zeros(1, dtype=torch.long),
                                    torch.zeros(1, 4, 2, 2), torch.zeros(1, 4, 2, 2))
                with mock.patch.object(trainer, "_sample_diffusion_inputs", return_value=diffusion_inputs), \
                     mock.patch.object(torch.optim.AdamW, "step", step_and_stop):
                    with self.assertRaises(FinishedOneUpdate):
                        trainer.stage3_finetune(num_epochs=1, lr=0.001, lambda_distill=0.0,
                                               recon_weight=0.0, perceptual_weight=0.0)
                after = model.frontier_proposal_head.output_head.weight.detach()
                self.assertGreater(float((after - before).abs().max()), 0.0)
                self.assertTrue(bool(torch.isfinite(after).all()))


if __name__ == "__main__":
    unittest.main()
