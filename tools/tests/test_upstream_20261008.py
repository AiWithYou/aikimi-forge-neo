"""CPU regressions for the selected October 8 Forge Neo updates."""

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backend import attention
from backend.sampling import sampling_function
from modules import shared
from modules_forge.anima_lora import ANIMA_29B_TO_38B, ANIMA_BASE_TO_29B
from tools.tests import test_controllllite_tiling as control_fixture
from tools.tests import test_sampling_composition_batches as composition
from tools.tests import test_sampling_model_retention as retention
from tools.tests.test_api_extras_boundaries import load_classes
from tools.tests.test_controllllite_tiling import anima as lllite

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "_test_october_spectrum", ROOT / "extensions-builtin/sd_forge_spectrum/lib_spectrum/forecaster.py"
)
spectrum = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spectrum)


class AnimaControlMappingTests(unittest.TestCase):
    @staticmethod
    def model(blocks):
        model = torch.nn.Module()
        model.blocks = torch.nn.ModuleList(torch.nn.Linear(1, 1) for _ in range(blocks))
        return model

    @staticmethod
    def weights(blocks):
        return {f"lllite_dit_blocks_{i}_self_attn_q_proj.down.weight": torch.ones(1, 1) for i in range(blocks)}

    def test_supported_expansions_hook_only_the_original_blocks(self):
        for source, target, indices in (
            (28, 40, ANIMA_BASE_TO_29B),
            (28, 52, tuple(ANIMA_29B_TO_38B[i] for i in ANIMA_BASE_TO_29B)),
            (40, 52, ANIMA_29B_TO_38B),
        ):
            with self.subTest(source=source, target=target):
                model = self.model(target)
                mapped = lllite.map_blocks(model, self.weights(source))
                self.assertEqual(len(mapped.blocks), source)
                for block, index in zip(mapped.blocks, indices, strict=True):
                    self.assertIs(block, model.blocks[index])

    def test_complete_native_layout_keeps_the_model(self):
        model = self.model(28)
        self.assertIs(lllite.map_blocks(model, self.weights(28)), model)

    def test_incomplete_and_unsupported_layouts_are_rejected(self):
        hole = self.weights(28)
        hole.pop("lllite_dit_blocks_5_self_attn_q_proj.down.weight")
        for weights, target in ((hole, 40), (self.weights(27), 40), ({}, 52), (self.weights(28), 30)):
            with self.subTest(count=len(weights), target=target):
                with self.assertRaisesRegex(ValueError, "layout|blocks"):
                    lllite.map_blocks(self.model(target), weights)

    def test_partial_weights_fail_before_modifying_any_parameter(self):
        net = torch.nn.Module()
        module = torch.nn.Module()
        module.lllite_name = "lllite_dit_blocks_0_self_attn_q_proj"
        module.down = torch.nn.Linear(1, 1, bias=False)
        net.lllite_modules = torch.nn.ModuleList([module])
        net.conditioning1 = torch.nn.Linear(1, 1, bias=False)
        before = net.conditioning1.weight.detach().clone()
        missing = {"lllite_conditioning1.weight": torch.full((1, 1), 99.0)}
        unexpected = missing | {module.lllite_name + ".down.weight": torch.ones(1, 1), "unknown.weight": torch.ones(1)}
        for weights, reason in ((missing, "Missing|missing"), (unexpected, "Unexpected|unexpected")):
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(RuntimeError, reason):
                    lllite.load_lllite_weights_from_dict(net, weights)
                torch.testing.assert_close(net.conditioning1.weight, before)

    def test_mapped_adapters_do_not_register_or_cast_the_backbone_and_restore_hooks(self):
        dit = torch.nn.Module()
        dit.blocks = torch.nn.ModuleList(control_fixture.SelfCrossAttention() for _ in range(40))
        mapped = lllite.map_blocks(dit, self.weights(28))
        net = lllite.ControlNetLLLiteDiT(mapped, cond_emb_dim=4, mlp_dim=4, cond_dim=8, cond_resblocks=0)
        self.addCleanup(net.restore)
        parameters = [(id(parameter), parameter.dtype, parameter.device) for parameter in dit.parameters()]
        forwards = [block.q_proj.forward for block in dit.blocks]
        self.assertEqual(len(net.lllite_modules), 28)
        self.assertFalse(any(name.startswith("blocks.") for name in net.state_dict()))
        net.to(dtype=torch.float64)
        self.assertEqual(parameters, [(id(p), p.dtype, p.device) for p in dit.parameters()])
        net.apply_to()
        for source, module in enumerate(net.lllite_modules):
            self.assertIs(module.org_module[0], dit.blocks[ANIMA_BASE_TO_29B[source]].q_proj)
            self.assertEqual(module.lllite_name, f"lllite_dit_blocks_{source}_q_proj")
        for index in set(range(40)) - set(ANIMA_BASE_TO_29B):
            self.assertEqual(dit.blocks[index].q_proj.forward, forwards[index])
        net.restore()
        self.assertEqual([block.q_proj.forward for block in dit.blocks], forwards)

    def test_failed_patcher_load_keeps_weights_and_does_not_publish_the_net(self):
        mapped = Mock()
        map_blocks = Mock(return_value=mapped)
        constructor = Mock()
        load = Mock(side_effect=RuntimeError("bad weights"))
        classes = load_classes(
            "extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py",
            {"ControlLLLiteAnimaPatcher"},
            {
                "torch": torch,
                "ControlModelPatcher": type("ControlModelPatcher", (), {}),
                "ControlNetLLLiteDiT": constructor,
                "infer_anima_config": lambda weights: {},
                "load_lllite_weights_from_dict": load,
                "map_blocks": map_blocks,
            },
        )
        weights = self.weights(28)
        patcher = classes.ControlLLLiteAnimaPatcher(weights, inpaint=False)
        dit = self.model(40)
        unet = SimpleNamespace(
            load_device=torch.device("cpu"), model=SimpleNamespace(computation_dtype=torch.float32, diffusion_model=dit)
        )
        process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
        with self.assertRaisesRegex(RuntimeError, "bad weights"):
            patcher.process_before_every_sampling(process, torch.zeros(1, 3, 8, 8), None)
        self.assertIsNone(patcher._lllite_net)
        self.assertIs(patcher.state_dict, weights)
        map_blocks.assert_called_once_with(dit, weights)
        constructor.assert_called_once_with(mapped)

    def test_changing_the_backbone_rebuilds_the_adapter_from_the_original_weights(self):
        nets = [Mock(), Mock()]
        for net in nets:
            net.eval.return_value = net
            net.to.return_value = net
        constructor = Mock(side_effect=nets)
        classes = load_classes(
            "extensions-builtin/sd_forge_controlllite/scripts/forge_controllllite.py",
            {"ControlLLLiteAnimaPatcher"},
            {
                "torch": torch,
                "ControlModelPatcher": type("ControlModelPatcher", (), {}),
                "ControlNetLLLiteDiT": constructor,
                "infer_anima_config": lambda weights: {},
                "load_lllite_weights_from_dict": Mock(),
                "map_blocks": lambda dit, weights: dit,
            },
        )
        weights = self.weights(28)
        patcher = classes.ControlLLLiteAnimaPatcher(weights, inpaint=False)
        patcher.strength, patcher.start_percent, patcher.end_percent = 1.0, 0.0, 1.0
        unet = SimpleNamespace(load_device=torch.device("cpu"), model=SimpleNamespace(computation_dtype=torch.float32))
        process = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)), steps=3)
        for blocks in (28, 40):
            unet.model.diffusion_model = self.model(blocks)
            patcher.process_before_every_sampling(process, torch.zeros(1, 3, 8, 8), None)
        self.assertEqual(constructor.call_count, 2)
        self.assertIs(patcher.state_dict, weights)
        self.assertIs(patcher._lllite_net, nets[1])
        nets[0].restore.assert_called_once()


class ConditionBatchTests(unittest.TestCase):
    def setUp(self):
        composition.SamplingCompositionBatchTests.setUp(self)

    def run_euler(self, *args, **kwargs):
        return composition.SamplingCompositionBatchTests.run_euler(self, *args, **kwargs)

    def test_runtime_setting_changes_condition_batch_without_splitting_images(self):
        opts = SimpleNamespace(batch_cond_uncond=True)
        original = composition.TinyConditionModel.apply_model
        batches = []

        def record(model, x, *args, **kwargs):
            batches.append(x.shape[0])
            return original(model, x, *args, **kwargs)

        with (
            patch.object(shared, "opts", opts),
            patch.object(composition.TinyConditionModel, "apply_model", record),
            patch.object(sampling_function.memory_management, "get_free_memory", return_value=10**9),
        ):
            for enabled, batch in ((True, 4), (False, 2), (True, 4)):
                opts.batch_cond_uncond = enabled
                batches.clear()
                self.run_euler(["a", "c"])
                self.assertTrue(batches)
                self.assertEqual(set(batches), {batch})

    def test_sequential_conditions_keep_and_and_zero_sum_weights(self):
        with patch.object(shared, "opts", SimpleNamespace(batch_cond_uncond=False)):
            self.run_euler(["a AND b", "c"])
            self.run_euler(["a:1 AND b:-1", "c:1 AND a:-1"], negative_prompt="b")

    def test_wrapper_knows_about_splitting_before_the_first_branch_runs(self):
        signals = []

        def wrapper(model, args):
            self.assertEqual(set(args), {"input", "timestep", "c", "cond_or_uncond"})
            signals.append(args["c"]["transformer_options"].get("split_conditions", False))
            return model(args["input"], args["timestep"], **args["c"])

        with patch.object(shared, "opts", SimpleNamespace(batch_cond_uncond=False)):
            self.run_euler(["a AND b", "c"], model_options={"model_function_wrapper": wrapper})
        self.assertTrue(signals[0])
        self.assertFalse(signals[-1])

    def test_minimum_memory_is_one_complete_image_batch(self):
        fixture = retention.SamplingModelRetentionTests("test_removed_control_is_not_loaded_by_next_sampling_pass")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.unet.controlnet_linked_list = None
        fixture.unet.extra_preserved_memory_during_sampling = 20
        latent = torch.zeros(2, 4, 2, 2)
        with patch.object(fixture.unet, "memory_required", side_effect=lambda shape: 10 + shape[0] ** 2):
            for enabled, required in ((True, 46), (False, 34)):
                with self.subTest(enabled=enabled):
                    with patch.object(shared, "opts", SimpleNamespace(batch_cond_uncond=enabled)):
                        sampling_function.sampling_prepare(fixture.unet, latent)
                    self.assertEqual(fixture.load.call_args.kwargs["memory_required"], required)
                    self.assertEqual(fixture.load.call_args.kwargs["minimum_memory_required"], 34)


class VaeSliceLimitTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(17)
        self.q = torch.randn(1, 5, 4, generator=generator)
        self.k = torch.randn(1, 4, 5, generator=generator)
        self.v = torch.randn(1, 4, 5, generator=generator)
        self.expected = torch.bmm(self.v, (torch.bmm(self.q, self.k) * 0.5).softmax(dim=-1).transpose(1, 2))

    def test_zero_free_memory_still_computes_one_row_at_a_time(self):
        with patch.object(attention.memory_management, "get_free_memory", return_value=0):
            actual = attention.slice_attention_vae(self.q, self.k, self.v)
        torch.testing.assert_close(actual, self.expected)

    def test_mps_slice_respects_int_max_after_rounding(self):
        q = torch.empty(1, 92681, 4, device="meta")
        k = v = torch.empty(1, 4, 92681, device="meta")
        original = torch.bmm
        sizes = []

        def record(left, right):
            if left.shape[-1] == 4:
                sizes.append(left.shape[0] * left.shape[1] * right.shape[2])
            return original(left, right)

        with (
            patch.object(attention.memory_management, "get_free_memory", return_value=10**14),
            patch.object(attention.memory_management, "is_device_mps", return_value=True),
            patch.object(torch, "bmm", side_effect=record),
        ):
            attention.slice_attention_vae(q, k, v)
        self.assertGreater(len(sizes), 1)
        self.assertLessEqual(max(sizes), 2**31 - 1)

    def test_int_max_retries_only_on_mps_and_stops_at_one_row(self):
        original = torch.bmm
        for mps, always_fail in ((True, False), (False, False), (True, True)):
            with self.subTest(mps=mps, always_fail=always_fail):
                sizes = []

                def limited(left, right, always_fail=always_fail, sizes=sizes):
                    if left.shape[-1] == 4:
                        sizes.append(left.shape[1])
                        if always_fail or left.shape[1] > 2:
                            raise RuntimeError("fixture INT_MAX")
                    return original(left, right)

                with (
                    patch.object(attention.memory_management, "get_free_memory", return_value=10**9),
                    patch.object(attention.memory_management, "is_device_mps", return_value=mps),
                    patch.object(attention.memory_management, "is_oom", return_value=False),
                    patch.object(attention.memory_management, "soft_empty_cache"),
                    patch.object(torch, "bmm", side_effect=limited),
                ):
                    if not mps or always_fail:
                        with self.assertRaisesRegex(RuntimeError, "INT_MAX"):
                            attention.slice_attention_vae(self.q, self.k, self.v)
                    else:
                        torch.testing.assert_close(attention.slice_attention_vae(self.q, self.k, self.v), self.expected)
                self.assertEqual(sizes, [5, 3, 2, 1] if always_fail else ([5, 3, 2, 2, 1] if mps else [5]))

    def test_pytorch_vae_routes_mps_int_max_to_the_sliced_path(self):
        inputs = (self.q.transpose(1, 2), self.k, self.v)
        with (
            patch.object(attention.operations, "scaled_dot_product_attention", side_effect=RuntimeError("INT_MAX")),
            patch.object(attention.memory_management, "get_free_memory", return_value=10**9),
            patch.object(attention.memory_management, "is_device_mps", return_value=True),
            patch.object(attention.memory_management, "is_oom", return_value=False),
            patch.object(attention.memory_management, "soft_empty_cache"),
        ):
            actual = attention.pytorch_attention_vae(*inputs)
        torch.testing.assert_close(actual, self.expected)


class SpectrumConditionTests(unittest.TestCase):
    @staticmethod
    def wrapper(warmup=1, w=0.5, window=4):
        captured = []
        model = SimpleNamespace(clone=lambda: SimpleNamespace(set_model_unet_function_wrapper=captured.append))
        spectrum.SpectrumNode.patch(model, 100, w, 1, 0.1, window, 0.5, warmup, 1.0)
        return captured[0]

    @staticmethod
    def args(x, timestep, branch=0):
        return {"input": x, "timestep": torch.tensor([float(timestep)]), "c": {}, "cond_or_uncond": [branch]}

    def test_shape_change_does_not_predict_from_the_previous_shape(self):
        wrapper = self.wrapper()
        model = Mock(side_effect=lambda x, t, **c: torch.ones_like(x))
        wrapper(model, self.args(torch.zeros(1, 2), 100))
        with patch.object(spectrum.FastChebyshevForecaster, "predict", side_effect=AssertionError("old shape")):
            out = wrapper(model, self.args(torch.zeros(2, 2), 99))
        self.assertEqual(model.call_count, 2)
        torch.testing.assert_close(out, torch.ones(2, 2))

    def test_first_call_always_runs_the_model_even_with_zero_warmup(self):
        wrapper = self.wrapper(warmup=0)
        model = Mock(side_effect=lambda x, t, **c: x + 3)
        torch.testing.assert_close(wrapper(model, self.args(torch.zeros(1, 2), 100)), torch.full((1, 2), 3.0))
        model.assert_called_once()

    def test_a_single_condition_batch_still_uses_prediction(self):
        wrapper = self.wrapper()
        model = Mock(side_effect=lambda x, t, **c: torch.ones_like(x))
        for timestep in range(100, 94, -1):
            out = wrapper(model, self.args(torch.zeros(1, 2), timestep))
            self.assertEqual(out.shape, (1, 2))
            self.assertTrue(torch.isfinite(out).all())
        self.assertLess(model.call_count, 6)

    def test_first_split_condition_is_actual_when_a_new_condition_starts(self):
        wrapper = self.wrapper(w=0.0, window=2)
        model = Mock(side_effect=lambda x, t, value, **c: torch.full_like(x, value))
        for timestep, value, split in ((100, 1.0, False), (99, 4.0, True), (99, 1.0, False)):
            args = self.args(torch.zeros(1, 2), timestep)
            args["c"] = {"value": value, "transformer_options": {"split_conditions": split}}
            torch.testing.assert_close(wrapper(model, args), torch.full((1, 2), value))
        self.assertEqual(model.call_count, 3)

    def test_split_cfg_and_and_conditions_never_share_predictions(self):
        for branches, values in (((0, 1), (1.0, 10.0)), ((0, 0), (1.0, 4.0))):
            with self.subTest(branches=branches):
                wrapper = self.wrapper()
                model = Mock(side_effect=lambda x, t, **c: x)
                for timestep in (100, 99, 98):
                    for branch, value in zip(branches, values, strict=True):
                        x = torch.full((1, 2), value)
                        torch.testing.assert_close(wrapper(model, self.args(x, timestep, branch)), x)
                self.assertEqual(model.call_count, 6)


if __name__ == "__main__":
    unittest.main()
