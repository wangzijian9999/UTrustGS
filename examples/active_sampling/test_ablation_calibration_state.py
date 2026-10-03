"""组件预设不得把未校准状态或未通过的准入状态改写为通过。"""
import json
import os
import unittest

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")

import torch

from examples.active_sampling.configs.active_sampling_config import ActiveSamplingConfig
from examples.active_sampling.train_active_sampling import _apply_ablation_preset
from examples.active_sampling.simple_trainer_with_active_sampling import (
    _build_plan6_protocol_config,
    _compute_plan6_depth_alpha_supervision_losses,
)
from examples.active_sampling.where_wrong_four_stage_protocol import (
    HowRepairableOutput,
    WhereRepairableOutput,
    run_when_trust_stage,
)


PRESETS = ("full", "no_active_selection", "no_verified_evidence", "no_trust_weights",
           "rgb_only", "full_image_pseudo", "pseudo_densify_stats")
STATE_FIELDS = ("plan6_gate0_status", "plan6_trust_is_calibrated",
                "plan6_rgb_trust_is_calibrated", "plan6_depth_trust_is_calibrated",
                "plan6_alpha_trust_is_calibrated")


def apply_preset(config, preset="full"):
    _apply_ablation_preset(config, preset=preset, seed=7, world_rank=1)


def declared_calibration_config():
    # 仅为测试输入的布尔声明，不代表任何模型实际已校准。
    config = ActiveSamplingConfig()
    config.plan6_gate0_status = "go"
    config.plan6_trust_is_calibrated = True
    config.plan6_rgb_trust_is_calibrated = True
    config.plan6_depth_trust_is_calibrated = True
    config.plan6_alpha_trust_is_calibrated = True
    return config


def controlled_trust_result(config, proposal_alpha=None):
    mask = torch.ones(1, 4, 4)
    zero = torch.zeros_like(mask)
    repairable = WhereRepairableOutput(mask, mask, zero, zero, mask, zero, zero, "synthetic")
    how = HowRepairableOutput(torch.full((3, 4, 4), 0.5), mask, "synthetic", [])
    payload = {"render_rgb": torch.full((3, 4, 4), 0.4),
               "P_rgb_trust": torch.full_like(mask, 0.8), "proposal_alpha": proposal_alpha}
    return run_when_trust_stage(payload, repairable, how, _build_plan6_protocol_config(config))


class AblationCalibrationStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_presets_preserve_default_uncalibrated_state(self):
        for preset in PRESETS:
            with self.subTest(preset=preset):
                config = ActiveSamplingConfig()
                expected = tuple(getattr(config, name) for name in STATE_FIELDS)
                apply_preset(config, preset)
                self.assertEqual(tuple(getattr(config, name) for name in STATE_FIELDS), expected)

    def test_presets_preserve_explicit_mixed_state_and_rejection(self):
        for preset in PRESETS:
            with self.subTest(preset=preset):
                config = declared_calibration_config()
                config.plan6_gate0_status = "stop"
                config.plan6_depth_trust_is_calibrated = False
                config.plan6_alpha_trust_is_calibrated = None
                expected = tuple(getattr(config, name) for name in STATE_FIELDS)
                apply_preset(config, preset)
                self.assertEqual(tuple(getattr(config, name) for name in STATE_FIELDS), expected)

    def test_presets_preserve_existing_positive_declarations(self):
        for preset in PRESETS:
            with self.subTest(preset=preset):
                config = declared_calibration_config()
                expected = tuple(getattr(config, name) for name in STATE_FIELDS)
                apply_preset(config, preset)
                self.assertEqual(tuple(getattr(config, name) for name in STATE_FIELDS), expected)

    def test_uncalibrated_full_preset_cannot_enable_online_weights(self):
        config = ActiveSamplingConfig()
        apply_preset(config)
        result = controlled_trust_result(config, torch.linspace(0.2, 0.8, 16).reshape(1, 4, 4))
        self.assertFalse(result.online_admission_allowed)
        for weight in (result.phase3_rgb_weight, result.phase3_depth_weight, result.phase3_alpha_weight):
            self.assertEqual(torch.count_nonzero(weight).item(), 0)

    def test_one_uncalibrated_channel_remains_blocking(self):
        for name in STATE_FIELDS[2:]:
            with self.subTest(channel=name):
                config = declared_calibration_config()
                setattr(config, name, False)
                apply_preset(config)
                self.assertFalse(controlled_trust_result(config).online_admission_allowed)

    def test_positive_declarations_keep_existing_online_path(self):
        config = declared_calibration_config()
        apply_preset(config)
        result = controlled_trust_result(config)
        self.assertTrue(result.online_admission_allowed)
        self.assertGreater(result.phase3_rgb_weight.min().item(), 0.7)

    def test_component_ablation_switches_remain_effective(self):
        for preset, name, expected in (
            ("no_active_selection", "ablation_random_selection", True),
            ("rgb_only", "phase3_use_plan6_depth_alpha_losses", False),
            ("pseudo_densify_stats", "allow_pseudo_densification_stats", True),
            ("full", "phase3_use_plan6_depth_alpha_losses", True),
        ):
            with self.subTest(preset=preset):
                config = ActiveSamplingConfig()
                apply_preset(config, preset)
                self.assertEqual(getattr(config, name), expected)

    def test_none_preset_preserves_state(self):
        config = ActiveSamplingConfig()
        expected = dict(vars(config))
        apply_preset(config, "none")
        self.assertEqual(vars(config), expected)

    def test_controlled_alpha_proxy_consumption_boundary(self):
        records = []
        for declared in (False, True):
            for kind in ("missing", "constant", "varying"):
                config = declared_calibration_config() if declared else ActiveSamplingConfig()
                config.phase3_use_plan6_depth_alpha_losses = True
                alpha = None if kind == "missing" else (
                    torch.full((1, 4, 4), 0.5) if kind == "constant" else
                    torch.linspace(0.2, 0.8, 16).reshape(1, 4, 4))
                result = controlled_trust_result(config, alpha)
                render_alpha = torch.full((1, 4, 4, 1), 0.9, requires_grad=True)
                _, loss = _compute_plan6_depth_alpha_supervision_losses(
                    torch.zeros(1, 4, 4, 3), render_alpha, None,
                    proposal_depth=None, proposal_alpha=alpha, depth_weight_map=None,
                    alpha_weight_map=result.phase3_alpha_weight, config=config)
                gradient = None
                if loss is not None:
                    loss.backward()
                    gradient = float(render_alpha.grad.sum())
                records.append({"declared_calibration_only": declared, "alpha_kind": kind,
                                "online_allowed": result.online_admission_allowed,
                                "alpha_weight_sum": float(result.phase3_alpha_weight.sum()),
                                "loss": None if loss is None else float(loss.detach()),
                                "render_alpha_gradient_sum": gradient})
                if declared and kind == "varying":
                    self.assertIsNotNone(loss)
                    self.assertGreater(gradient, 0.0)
                else:
                    self.assertIsNone(loss)
        print("ALPHA_CONSUMER_PROBE=" + json.dumps(records, ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
