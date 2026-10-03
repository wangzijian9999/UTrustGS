"""物理参数、拓扑状态迁移、原生渲染及完整 Phase 1 续训的确定性验证。"""
import copy
import io
import os
import random
import unittest

os.environ.setdefault("GDDN_SKIP_HEAVY_TRAINER_OPTIONAL_IMPORTS", "1")
import numpy as np
import torch
from plyfile import PlyData

from examples.active_sampling.gaussian_parameter_contract import (
    PARAMETERS, capture_training_state, field_state, load_field, ply_parameters,
    project_field, refine_field, replace_rows, restore_training_state, validate_field,
)
from examples.active_sampling.train_active_sampling import GaussianPointField, build_condition_provider
from examples.active_sampling.simple_trainer_with_active_sampling import (
    _apply_gaussian_state, _serialize_gaussian_state, train_3dgs_with_active_sampling,
)
from examples.active_sampling.configs.active_sampling_config import ActiveSamplingConfig
from examples.active_sampling.uncertainty_estimator import SparseViewBundle
from examples.active_sampling.viewpoint_sampler import ExistingCamera
from gsplat.rendering import rasterization
from gsplat.exporter import export_splats


def make_field(device="cpu"):
    return GaussianPointField(
        torch.tensor([[-0.2, 0., 2.], [0.2, 0., 2.], [0., 0.2, 2.]]),
        torch.tensor([[.7, .2, .1], [.1, .7, .2], [.2, .1, .7]]),
        torch.device(device), scale_init=.1, opacity_init=.4,
    )


class GaussianContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def assert_tree(self, left, right, atol=0):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=atol, atol=atol)
        elif isinstance(left, np.ndarray):
            np.testing.assert_array_equal(left, right)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_tree(left[key], right[key], atol)
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_tree(first, second, atol)
        else:
            self.assertEqual(left, right)

    def populate_adam(self, field):
        optimizer = torch.optim.Adam(field.parameters(), lr=.001, amsgrad=True)
        sum(parameter.square().sum() for parameter in field.parameters()).backward()
        optimizer.step()
        project_field(field)
        return optimizer

    def test_projection_preserves_interior_and_rejects_nonfinite(self):
        field = make_field()
        previous = field_state(field)
        project_field(field)
        self.assert_tree(previous, field_state(field))
        with torch.no_grad():
            field.scales[0] = -.5
            field.opacities[0], field.opacities[1] = -1., 2.
        project_field(field)
        validate_field(field)
        with torch.no_grad():
            field.means[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            project_field(field)

    def test_serialization_roundtrip_and_legacy_physical(self):
        field, restored = make_field(), make_field()
        state = _serialize_gaussian_state(field)
        _apply_gaussian_state(restored, state, torch.device("cpu"))
        self.assert_tree(field_state(field), field_state(restored))
        state.pop("parameter_format")
        state["color_logits"] = state.pop("_color_logits")
        load_field(restored, state, torch.device("cpu"))
        self.assert_tree(field_state(field), field_state(restored))
        for name, value in (("parameter_format", "log_scale_v0"), ("scales", -field.scales),
                            ("opacities", field.opacities + 2)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_field(restored, dict(state, **{name: value}), torch.device("cpu"))
        self.assert_tree(field_state(field), field_state(restored))

    def test_ply_actual_bytes_decode_to_physical_parameters(self):
        field = make_field()
        data = PlyData.read(io.BytesIO(export_splats(**ply_parameters(field))))["vertex"]
        scale = np.stack([data[f"scale_{axis}"] for axis in range(3)], -1)
        color = np.stack([data[f"f_dc_{axis}"] for axis in range(3)], -1)
        np.testing.assert_allclose(np.exp(scale), field.scales.detach(), atol=1e-6)
        np.testing.assert_allclose(1 / (1 + np.exp(-data["opacity"])), field.opacities.detach(), atol=1e-6)
        np.testing.assert_allclose(color * .28209479177387814 + .5, field.colors.detach(), atol=1e-6)

    def test_append_and_prune_preserve_adam_including_amsgrad(self):
        field = make_field()
        optimizer = self.populate_adam(field)
        previous = {name: copy.deepcopy(optimizer.state[getattr(field, name)]) for name in PARAMETERS}
        rows = torch.tensor([2, 0, -1])
        values = {name: getattr(field, name).detach()[[2, 0, 1]] for name in PARAMETERS}
        replace_rows(field, optimizer, values, rows)
        self.assertEqual(len(optimizer.state), 5)
        for name in PARAMETERS:
            parameter = getattr(field, name)
            self.assertTrue(any(parameter is p for group in optimizer.param_groups for p in group["params"]))
            for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                torch.testing.assert_close(optimizer.state[parameter][key][:2], previous[name][key][[2, 0]], rtol=0, atol=0)
                self.assertEqual(int(torch.count_nonzero(optimizer.state[parameter][key][2])), 0)
            self.assertEqual(optimizer.state[parameter]["step"], previous[name]["step"])

    def test_clone_split_and_independent_prune_use_physical_units(self):
        field = make_field()
        optimizer = self.populate_adam(field)
        with torch.no_grad():
            field.scales[0] = .01
            field.scales[1] = .2
            field.opacities[2] = .001
        center = field.means[1].detach().clone()
        torch.manual_seed(17)
        self.assertEqual(refine_field(field, optimizer, torch.ones(3), .05, .5, .005), (1, 1, 1))
        self.assertEqual(len(field.means), 4)
        torch.testing.assert_close(field.scales[-2:], torch.full((2, 3), .125))
        torch.testing.assert_close(field.means[-2:].mean(0), center)
        validate_field(field)
        with torch.no_grad():
            field.opacities[0] = .001
        self.assertEqual(refine_field(field, optimizer, torch.zeros(4), .05, .5, .005), (0, 0, 1))
        with torch.no_grad():
            field.opacities.fill_(.001)
        self.assertEqual(refine_field(field, optimizer, torch.zeros(3), .05, .5, .005), (0, 0, 2))
        self.assertEqual(len(field.means), 1)

    def test_noop_refinement_preserves_parameter_identity(self):
        field = make_field()
        optimizer = self.populate_adam(field)
        ids = [id(parameter) for parameter in field.parameters()]
        self.assertEqual(refine_field(field, optimizer, torch.zeros(3), .05, .5, .005), (0, 0, 0))
        self.assertEqual(ids, [id(parameter) for parameter in field.parameters()])

    def test_cpu_save_resume_reproduces_updates_rng_and_topology_exactly(self):
        def advance(field, optimizer, start, stop):
            for index in range(start, stop):
                optimizer.zero_grad()
                loss = sum((parameter - torch.rand_like(parameter) * .01).square().mean()
                           for parameter in field.parameters()) * (1 + random.random() + np.random.rand())
                loss.backward()
                optimizer.step()
                project_field(field)
                if index in (2, 4):
                    refine_field(field, optimizer, torch.ones(len(field.means)), .05, .5, .005)
        torch.manual_seed(23)
        random.seed(23)
        np.random.seed(23)
        field = make_field()
        optimizer = torch.optim.Adam(field.parameters(), lr=.001)
        advance(field, optimizer, 0, 3)
        saved = capture_training_state(field, optimizer, completed_iterations=3)
        stream = io.BytesIO()
        torch.save(saved, stream)
        advance(field, optimizer, 3, 6)
        expected = capture_training_state(field, optimizer, completed_iterations=6)
        restored = make_field()
        resumed_optimizer = torch.optim.Adam(restored.parameters(), lr=.001)
        stream.seek(0)
        loop = restore_training_state(restored, resumed_optimizer, torch.load(stream, weights_only=True))
        advance(restored, resumed_optimizer, loop["completed_iterations"], 6)
        self.assert_tree(expected, capture_training_state(restored, resumed_optimizer, completed_iterations=6))

    @unittest.skipUnless(torch.cuda.is_available(), "真实 gsplat CUDA 核需 GPU")
    def test_native_renderer_condition_provider_and_phase1_resume(self):
        device = torch.device("cuda")
        field = make_field(device)
        # 使用旋转可辨识的各向异性夹具。首次球形夹具的 CUDA 误差
        # 单独保留在审计报告；不据本测试推断所有配置都确定。
        with torch.no_grad():
            field.scales[:] = torch.tensor([.08, .11, .15], device=device)
            field.quats[:] = torch.tensor([1., .2, .3, .4], device=device)
        project_field(field)
        view = torch.eye(4, device=device)
        intrinsics = torch.tensor([[24., 0., 16.], [0., 24., 16.], [0., 0., 1.]], device=device)
        camera = ExistingCamera(position=torch.zeros(3, device=device), viewmat=view, intrinsics=intrinsics)
        def render(current):
            return rasterization(means=current.means, quats=current.quats, scales=current.scales,
                                 opacities=current.opacities, colors=current.colors,
                                 viewmats=view[None], Ks=intrinsics[None], width=32, height=32)[0]
        with torch.no_grad():
            target = render(field).permute(0, 3, 1, 2).contiguous()
        provider = build_condition_provider(field, 32, 32, device, "gs_dropout")
        torch.testing.assert_close(provider([camera])["rgb_render"], target, rtol=0, atol=0)
        with torch.no_grad():
            field._color_logits.add_(.3)
        start = field_state(field)
        config = ActiveSamplingConfig()
        config.estimator_mode = "gs_dropout"
        config.gate_enable = False
        config.phase1_convergence_min_iter = 10000
        config.refine_start_iter, config.refine_every, config.refine_stop_iter = 2, 2, 8
        config.grow_grad2d = -1.
        config.output_dir = None
        bundle = SparseViewBundle(images=target, poses=view[None])
        def train(current, steps, resume=None):
            train_3dgs_with_active_sampling(current, [camera], target, None, steps, bundle, config,
                                           current_phase=1, resume_training_state=resume)
        torch.manual_seed(31)
        train(field, 6)
        expected = field._training_state
        restored = make_field(device)
        load_field(restored, start, device)
        torch.manual_seed(31)
        train(restored, 3)
        self.assertEqual(restored._training_state["loop"]["completed_iterations"], 3)
        stream = io.BytesIO()
        torch.save(restored._training_state, stream)
        stream.seek(0)
        checkpoint = torch.load(stream, weights_only=True)
        resumed = make_field(device)
        train(resumed, 6, checkpoint)
        for key in ("gaussians", "optimizer", "loop"):
            self.assert_tree(expected[key], resumed._training_state[key], atol=1e-6)
        self.assertGreater(len(resumed.means), 3)
        torch.testing.assert_close(render(field), render(resumed), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
