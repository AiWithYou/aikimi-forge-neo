"""CPU regressions for ControlLLLite condition and hook lifetime while tiling."""

import ast
import importlib.util
import logging
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch

from backend.args import dynamic_args

ROOT = Path(__file__).resolve().parents[2]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_controls():
    directory = ROOT / "extensions-builtin/sd_forge_controlllite/lib_controllllite"
    package = ModuleType("_test_lllite_tiling")
    package.__path__ = [str(directory)]
    names = (
        package.__name__,
        package.__name__ + ".lib_controllllite",
        package.__name__ + ".lib_controllllite_anima",
        "_test_tiled_diffusion",
    )
    previous = {name: sys.modules.get(name) for name in names}
    try:
        sys.modules[package.__name__] = package
        sd = _load_module(package.__name__ + ".lib_controllllite", directory / "lib_controllllite.py")
        anima = _load_module(package.__name__ + ".lib_controllllite_anima", directory / "lib_controllllite_anima.py")
        # Match Forge startup's import order across cldm/controlnet's existing cycle.
        importlib.import_module("backend.nn.cnets.cldm")
        tiled = _load_module(
            "_test_tiled_diffusion",
            ROOT / "extensions-builtin/sd_forge_multidiffusion/lib_multidiffusion/tiled_diffusion.py",
        )
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return sd, anima, tiled


lllite, anima, tiled = _load_controls()


def _load_cleanup_class():
    # Load the production cleanup methods without importing the Gradio UI/model index.
    path = ROOT / "extensions-builtin/sd_forge_controlnet/scripts/controlnet.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cached = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ControlNetCachedParameters"
    )
    script = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ControlNetForForgeOfficial"
    )
    script.bases = []
    methods = {
        "_restore_control_model",
        "process_unit_after_every_sampling",
        "postprocess_batch_list",
        "on_process_cleanup",
        "postprocess",
    }
    script.body = [node for node in script.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    namespace = {
        "ExitStack": ExitStack,
        "torch": torch,
        "logger": logging.getLogger("test_controllllite_cleanup"),
        "StableDiffusionProcessing": object,
        "ControlNetUnit": object,
    }
    exec(  # noqa: S102 - execute the selected local cleanup implementation
        compile(ast.fix_missing_locations(ast.Module(body=[cached, script], type_ignores=[])), str(path), "exec"),
        namespace,
    )
    return namespace["ControlNetForForgeOfficial"], namespace["ControlNetCachedParameters"]


ControlNetScript, CachedParameters = _load_cleanup_class()


class SelfCrossAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.is_SelfAttn = True
        self.q_proj = torch.nn.Linear(4, 4)


class ControlLLLiteTilingTests(unittest.TestCase):
    def setUp(self):
        self.active = mock.patch.object(dynamic_args, "ACTIVE_LLLITE_DIT", set(), create=True)
        self.active.start()
        self.addCleanup(self.active.stop)
        self.factor = mock.patch.object(tiled, "opt_f", 8)
        self.factor.start()
        self.addCleanup(self.factor.stop)

    @staticmethod
    def _sd_patch(*, batch=1, height=32, width=64, steps=0, start=0.0, end=1.0):
        name = "lllite_unet_input_blocks_0_1_transformer_blocks_0_attn1_to_q"
        module = lllite.LLLiteModule(name, False, 4, 1, 4, 4, 1.0, 0, 0, 0)
        state = {f"{name}.{key}": value for key, value in module.state_dict().items()}
        image = torch.linspace(0.0, 1.0, batch * height * width * 3).reshape(batch, height, width, 3)
        return lllite.load_control_net_lllite_patch(state, image, 1.0, steps, start, end, model_dtype=torch.float32)

    @staticmethod
    def _anima_net():
        dit = torch.nn.Sequential(SelfCrossAttention())
        net = anima.ControlNetLLLiteDiT(dit, cond_emb_dim=4, mlp_dim=4, cond_dim=8, cond_resblocks=0)
        return dit, net

    @staticmethod
    def _diffusion(method, tile_size=4, batch_size=1):
        implementation = method()
        implementation.tile_width = tile_size
        implementation.tile_height = tile_size
        implementation.tile_overlap = 0
        implementation.tile_batch_size = batch_size
        return implementation

    @staticmethod
    def _args(x, patch=None):
        conditioning = {"c_crossattn": torch.zeros(x.shape[0], 1, 4)}
        if patch is not None:
            conditioning["transformer_options"] = {"patches": {"attn1_patch": [patch], "attn2_patch": [patch]}}
        return {"input": x, "timestep": torch.ones(x.shape[0]), "c": conditioning, "cond_or_uncond": [0]}

    @staticmethod
    def _sd_forward(patch, x):
        q = x.flatten(2).transpose(1, 2)
        q, _, _ = patch(q, q, q, {"block": ("input", 0), "block_index": 0})
        return q.transpose(1, 2).reshape_as(x)

    def test_sdxl_full_tiled_full_preserves_values_and_batches(self):
        for method in (tiled.MultiDiffusion, tiled.MixtureOfDiffusers):
            for tile_batch in (1, 2):
                with self.subTest(method=method.__name__, tile_batch=tile_batch):
                    patch = self._sd_patch(batch=2)
                    x = torch.ones(2, 4, 4, 8)
                    full = self._sd_forward(patch, x)
                    implementation = self._diffusion(method, batch_size=tile_batch)
                    with mock.patch.object(patch, "prepare_tiled", wraps=patch.prepare_tiled) as prepare:
                        result = implementation(
                            lambda tile, _t, patch=patch, **_c: self._sd_forward(patch, tile), self._args(x, patch)
                        )
                    torch.testing.assert_close(result, full)
                    self.assertEqual(prepare.call_count, 2 // tile_batch)
                    for module in patch.modules.values():
                        self.assertIs(module.cond_image, patch._cond_image_original)
                        self.assertIsNone(module.cond_emb)
                    torch.testing.assert_close(self._sd_forward(patch, x), full)

    def test_sdxl_step_range_advances_once_per_evaluation(self):
        for method in (tiled.MultiDiffusion, tiled.MixtureOfDiffusers):
            with self.subTest(method=method.__name__):
                patch = self._sd_patch(steps=4, start=0.25, end=0.75)
                module = next(iter(patch.modules.values()))
                implementation = self._diffusion(method)
                x = torch.ones(1, 4, 4, 8)
                for step in range(4):
                    observed = []

                    def model(tile, _t, observed=observed, module=module, patch=patch, **_c):
                        observed.append(module.current_step)
                        return self._sd_forward(patch, tile)

                    result = implementation(model, self._args(x, patch))
                    self.assertEqual(observed, [step, step])
                    self.assertEqual(module.current_step, (step + 1) % 4)
                    if step in (0, 3):
                        torch.testing.assert_close(result, x)

    def test_sdxl_exception_or_cancel_restores_condition_cache_and_retry(self):
        for method in (tiled.MultiDiffusion, tiled.MixtureOfDiffusers):
            for error_type in (RuntimeError, KeyboardInterrupt):
                with self.subTest(method=method.__name__, error=error_type.__name__):
                    patch = self._sd_patch(steps=4)
                    module = next(iter(patch.modules.values()))
                    implementation = self._diffusion(method)
                    x = torch.ones(1, 4, 4, 8)
                    calls = 0

                    def fail(tile, _t, patch=patch, error_type=error_type, **_c):
                        nonlocal calls
                        calls += 1
                        result = self._sd_forward(patch, tile)
                        if calls == 2:
                            raise error_type("sampling stopped")
                        return result

                    with self.assertRaises(error_type):
                        implementation(fail, self._args(x, patch))
                    self.assertIs(module.cond_image, patch._cond_image_original)
                    self.assertIsNone(module.cond_emb)
                    self.assertEqual(module.current_step, 0)
                    self.assertEqual(patch._lllite_tiled_cache, {})
                    recovered = implementation(
                        lambda tile, _t, patch=patch, **_c: self._sd_forward(patch, tile), self._args(x, patch)
                    )
                    self.assertEqual(recovered.shape, x.shape)
                    self.assertEqual(module.current_step, 1)

    def test_resized_condition_repeats_input_batch_in_tile_order(self):
        patch = self._sd_patch(batch=2, height=16, width=32)
        implementation = self._diffusion(tiled.MultiDiffusion, batch_size=2)
        x = torch.ones(3, 4, 4, 8)
        resized = torch.nn.functional.interpolate(patch._cond_image_original, size=(32, 64), mode="nearest-exact")
        repeated = resized[[0, 1, 0]]
        expected = torch.cat((repeated[:, :, :, :32], repeated[:, :, :, 32:]), dim=0)

        def model(tile, _t, **_c):
            torch.testing.assert_close(next(iter(patch.modules.values())).cond_image, expected)
            return tile

        torch.testing.assert_close(implementation(model, self._args(x, patch)), x)

    def test_sdxl_move_invalidates_tiles_and_rebinds_original_condition(self):
        patch = self._sd_patch()
        implementation = self._diffusion(tiled.MultiDiffusion)
        implementation(lambda tile, _t, **_c: tile, self._args(torch.ones(1, 4, 4, 8), patch))
        self.assertTrue(patch._lllite_tiled_cache)
        patch.to(torch.float64)
        self.assertEqual(patch._lllite_tiled_cache, {})
        self.assertEqual(patch._cond_image_original.dtype, torch.float64)
        self.assertIs(next(iter(patch.modules.values())).cond_image, patch._cond_image_original)

    def test_anima_tiling_cache_restore_and_changed_condition(self):
        for method in (tiled.MultiDiffusion, tiled.MixtureOfDiffusers):
            with self.subTest(method=method.__name__):
                dit, net = self._anima_net()
                self.addCleanup(net.restore)
                original_forward = dit[0].q_proj.forward
                image = torch.randn(1, 3, 64, 128)
                net.set_cond_image(image)
                net.set_step_range(4, 0, 1)
                net.apply_to()
                full_embedding = net.lllite_modules[0].cond_emb
                implementation = self._diffusion(method, tile_size=8)
                x = torch.ones(1, 4, 1, 8, 16)
                seen_steps = []

                def model(tile, _t, seen_steps=seen_steps, net=net, dit=dit, **_c):
                    seen_steps.append(net.lllite_modules[0].current_step)
                    self.assertEqual(net.lllite_modules[0].cond_emb.shape, (1, 16, 4))
                    self.assertEqual(dit[0].q_proj(torch.ones(1, 16, 4)).shape, (1, 16, 4))
                    return tile

                with mock.patch.object(net.conditioning1, "forward", wraps=net.conditioning1.forward) as encode:
                    torch.testing.assert_close(implementation(model, self._args(x)), x)
                    torch.testing.assert_close(implementation(model, self._args(x)), x)
                    self.assertEqual(encode.call_count, 2)
                self.assertEqual(seen_steps, [0, 0, 1, 1])
                self.assertIs(net.lllite_modules[0].cond_emb, full_embedding)
                self.assertEqual(dit[0].q_proj(torch.ones(1, 32, 4)).shape, (1, 32, 4))
                net.set_cond_image(image + 0.5)
                self.assertEqual(net._lllite_tiled_cache, {})
                self.assertIsNot(net.lllite_modules[0].cond_emb, full_embedding)
                net.restore()
                self.assertEqual(dit[0].q_proj.forward, original_forward)
                self.assertNotIn(net, dynamic_args.ACTIVE_LLLITE_DIT)
                self.assertIsNone(net._cond_image_original)
                self.assertIsNone(net._cond_emb_original)

    def test_anima_preparation_failure_restores_all_registered_controls(self):
        nets = [self._anima_net()[1] for _ in range(2)]
        for net in nets:
            net.set_cond_image(torch.ones(1, 3, 64, 128))
            net.apply_to()
            self.addCleanup(net.restore)
        implementation = self._diffusion(tiled.MultiDiffusion, tile_size=8)
        with (
            mock.patch.object(nets[1].conditioning1, "forward", side_effect=RuntimeError("encoder failed")),
            self.assertRaisesRegex(RuntimeError, "encoder failed"),
        ):
            implementation(lambda tile, _t, **_c: tile, self._args(torch.ones(1, 4, 8, 16)))
        for net in nets:
            self.assertIs(net.lllite_modules[0].cond_emb, net._cond_emb_original)
            self.assertEqual(net._lllite_tiled_cache, {})

    def test_anima_move_clears_cache_and_moves_original_embedding(self):
        _, net = self._anima_net()
        self.addCleanup(net.restore)
        net.set_cond_image(torch.ones(1, 3, 64, 128))
        net.apply_to()
        self._diffusion(tiled.MultiDiffusion, tile_size=8)(
            lambda tile, _t, **_c: tile, self._args(torch.ones(1, 4, 8, 16))
        )
        self.assertTrue(net._lllite_tiled_cache)
        net.to(dtype=torch.float64)
        self.assertEqual(net._lllite_tiled_cache, {})
        self.assertEqual(net._cond_image_original.dtype, torch.float64)
        self.assertEqual(net._cond_emb_original.dtype, torch.float64)
        self.assertIs(net.lllite_modules[0].cond_emb, net._cond_emb_original)

    def test_final_cleanup_unpatches_cancelled_generation_and_allows_reuse(self):
        dit, net = self._anima_net()
        original_forward = dit[0].q_proj.forward
        self.addCleanup(net.restore)
        script = ControlNetScript()
        for image_value in (0.0, 1.0):
            params = CachedParameters()
            params.model = SimpleNamespace(process_after_every_sampling=lambda *_args: net.restore())
            params.model_cleanup_pending = True
            script.current_params = {0: params}
            net.set_cond_image(torch.full((1, 3, 64, 128), image_value))
            net.apply_to()
            self.assertIn(net, dynamic_args.ACTIVE_LLLITE_DIT)
            script.on_process_cleanup(object())
            script.on_process_cleanup(object())
            self.assertEqual(dit[0].q_proj.forward, original_forward)
            self.assertFalse(params.model_cleanup_pending)
            self.assertEqual(script.current_params, {})
            self.assertNotIn(net, dynamic_args.ACTIVE_LLLITE_DIT)
            self.assertIsNone(net.lllite_modules[0].cond_emb)

    def test_normal_batch_cleanup_is_reverse_order_and_runs_once(self):
        script = ControlNetScript()
        order = []
        image_order = []
        parameters = []
        for index in range(2):
            params = CachedParameters()
            params.preprocessor = mock.Mock()
            params.preprocessor.process_after_every_sampling.side_effect = lambda *_args, i=index: image_order.append(i)
            params.model = mock.Mock()
            params.model.process_after_every_sampling.side_effect = lambda *_args, i=index: order.append(i)
            params.model_cleanup_pending = True
            parameters.append(params)
        script.current_params = dict(enumerate(parameters))
        script.get_enabled_units = lambda _args: [object(), object()]
        script.postprocess_batch_list(object(), object())
        script.on_process_cleanup(object())
        self.assertEqual(order, [1, 0])
        self.assertEqual(image_order, [0, 1])
        for params in parameters:
            params.model.process_after_every_sampling.assert_called_once()

    def test_preprocessor_cleanup_failure_still_restores_model(self):
        params = CachedParameters()
        params.preprocessor = mock.Mock()
        params.preprocessor.process_after_every_sampling.side_effect = RuntimeError("preprocessor failed")
        params.model = mock.Mock()
        params.model_cleanup_pending = True
        with self.assertRaisesRegex(RuntimeError, "preprocessor failed"):
            ControlNetScript().process_unit_after_every_sampling(object(), object(), params)
        params.model.process_after_every_sampling.assert_called_once()
        self.assertFalse(params.model_cleanup_pending)


if __name__ == "__main__":
    unittest.main()
