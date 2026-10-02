"""Registered RescaleCFG preserves image statistics and video sample axes."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from backend.patcher.unet import UnetPatcher
from backend.sampling import sampling_function
from backend.sampling.condition import compile_conditions
from tools.tests.test_api_extras_boundaries import load_classes


class TinyRescaleModel(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.coefficient = torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.coefficient.weight.fill_(0.5 if mode in ("guided_flat", "cond_flat") else 1.0)
            self.coefficient.bias.fill_(0.5 if mode == "guided_flat" else -0.5 if mode == "cond_flat" else 0.0)

    def memory_required(self, shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, **kwargs):
        grid = torch.arange(x[0].numel(), dtype=x.dtype).reshape((1,) + x.shape[1:])
        value = self.coefficient(c_crossattn[:, 0]).reshape((-1,) + (1,) * (x.ndim - 1))
        if self.mode == "flat":
            return value.expand_as(x)
        if self.mode in ("guided_flat", "cond_flat"):
            return grid * value
        grid = grid / 10.0
        return grid.square() + grid * value


class RescaleCFGProtocolTests(unittest.TestCase):
    def setUp(self):
        self.script = load_classes(
            "modules/processing_scripts/rescale_cfg.py",
            {"ScriptRescaleCFG"},
            {"torch": torch, "scripts": SimpleNamespace(ScriptBuiltinUI=object)},
        ).ScriptRescaleCFG()
        self.enterContext(patch.object(sampling_function.dynamic_args, "context_handler", None))

    def run_fixture(self, shape, mode="spatial"):
        model = TinyRescaleModel(mode)
        cpu = torch.device("cpu")
        original = UnetPatcher(model, load_device=cpu, offload_device=cpu)
        process = SimpleNamespace(
            sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=original)), rescale_cfg=0.5, is_hr_pass=False
        )
        self.script.process_before_every_sampling(process)
        patched = process.sd_model.forge_objects.unet
        self.assertIsNot(patched, original)
        self.assertEqual(original.model_options, {"transformer_options": {}})
        x = torch.zeros(shape)
        if mode == "spatial":
            x = torch.arange(x.numel(), dtype=x.dtype).reshape(shape) / 100.0
        positive = torch.ones(shape[0], 1, 1)
        negative = torch.full_like(positive, 3.0)
        positive_before, negative_before = positive.clone(), negative.clone()
        cond, uncond = compile_conditions(positive), compile_conditions(negative)
        actual, cond_pred, uncond_pred = sampling_function.sampling_function_inner(
            model, x, torch.ones(shape[0]), uncond, cond, 2.0, patched.model_options, return_full=True
        )
        reduced_x = x / 2.0
        cond_eps = (reduced_x - cond_pred) * 2**0.5
        uncond_eps = (reduced_x - uncond_pred) * 2**0.5
        cfg_eps = uncond_eps + 2.0 * (cond_eps - uncond_eps)
        dims = tuple(range(1, x.ndim))
        positive_std = cond_eps.std(dim=dims, keepdim=True)
        guided_std = cfg_eps.std(dim=dims, keepdim=True)
        if mode in ("flat", "guided_flat"):
            torch.testing.assert_close(guided_std, torch.zeros_like(guided_std))
            expected = reduced_x - cfg_eps / 2**0.5
        else:
            ratio = positive_std / guided_std
            expected = reduced_x - (0.5 * cfg_eps * ratio + 0.5 * cfg_eps) / 2**0.5
        self.assertEqual(actual.shape, x.shape)
        self.assertEqual(actual.dtype, x.dtype)
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(positive, positive_before)
        torch.testing.assert_close(negative, negative_before)

    def test_image_rescale_preserves_existing_full_sample_formula(self):
        self.run_fixture((2, 4, 2, 3))

    def test_video_and_single_frame_image_rescale_reduce_all_feature_axes(self):
        for frames in (1, 2):
            with self.subTest(frames=frames):
                self.run_fixture((2, 4, frames, 2, 3))

    def test_flat_and_cancelled_guidance_keep_finite_ordinary_cfg(self):
        for shape in ((2, 4, 2, 3), (2, 4, 2, 2, 3)):
            for mode in ("flat", "guided_flat"):
                with self.subTest(shape=shape, mode=mode):
                    self.run_fixture(shape, mode)

    def test_flat_positive_with_varying_guidance_keeps_zero_target_variance(self):
        for shape in ((2, 4, 2, 3), (2, 4, 2, 2, 3)):
            with self.subTest(shape=shape):
                self.run_fixture(shape, "cond_flat")


if __name__ == "__main__":
    unittest.main()
