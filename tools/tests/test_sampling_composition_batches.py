"""Real prompt composition and Euler preserve each image's AND weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from backend.patcher.controlnet import compute_controlnet_weighting
from backend.sampling import sampling_function
from backend.sampling.condition import compile_weighted_conditions
from modules import prompt_parser
from modules_forge.packages.k_diffusion import sampling as sampler_impl
from tools.tests.test_api_extras_boundaries import load_classes


class TinyConditionModel:
    def __init__(self, dictionary):
        self.dictionary = dictionary
        self.diffusion_model = SimpleNamespace(lq_proj=SimpleNamespace(sr_scale=1, latent_spatial_down_factor=1))

    def get_learned_conditioning(self, texts):
        cond = torch.tensor([[[{"a": 1.0, "b": 2.0, "c": 4.0, "": 0.0}[text.strip()]]] for text in texts])
        if self.dictionary:
            return {"crossattn": cond, "vector": torch.zeros(len(texts), 2), "guidance": torch.ones(len(texts), 1)}
        return cond

    def memory_required(self, input_shape):
        return 0

    def apply_model(self, x, timestep, c_crossattn, **kwargs):
        return c_crossattn[:, 0, 0].reshape((-1,) + (1,) * (x.ndim - 1)).expand_as(x)


class FrameControl:
    previous_controlnet = None
    advanced_frame_weighting = [0.25, 0.75]

    def __init__(self):
        self.transformer_options = {}
        self.signals = []

    def get_control(self, x, timestep, cond, batched_number):
        control = {"input": [torch.ones_like(x)], "middle": [], "output": []}
        result = compute_controlnet_weighting(control, self)
        self.signals.append(result["input"][0])
        return result


class SamplingCompositionBatchTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(sampling_function.dynamic_args, "context_handler", None))

    def run_euler(
        self,
        prompts,
        *,
        dictionary=False,
        ndim=4,
        model_options=None,
        control=None,
        scale=2.0,
        negative_prompt="",
        frames=2,
        context_handler=None,
        expected=None,
    ):
        callbacks = load_classes("modules/script_callbacks.py", {"CFGDenoiserParams", "AfterCFGCallbackParams"}, {})
        classes = load_classes(
            "modules/sd_samplers_cfg_denoiser.py",
            {"CFGDenoiser"},
            {
                "torch": torch,
                "sampling_function": sampling_function.sampling_function,
                "prompt_parser": prompt_parser,
                "sd_samplers_common": SimpleNamespace(
                    InterruptedException=RuntimeError, apply_refiner=lambda *args: False, store_latent=Mock()
                ),
                "state": SimpleNamespace(interrupted=False, skipped=False, sampling_step=0, sampling_steps=1),
                "opts": SimpleNamespace(skip_early_cond=0, s_min_uncond_all=False),
                "CFGDenoiserParams": callbacks.CFGDenoiserParams,
                "AfterCFGCallbackParams": callbacks.AfterCFGCallbackParams,
                "cfg_denoiser_callback": Mock(),
                "cfg_after_cfg_callback": Mock(),
            },
        )
        model = TinyConditionModel(dictionary)
        unet = SimpleNamespace(
            model=model, controlnet_linked_list=control, extra_concat_condition=None, model_options=model_options or {}
        )
        wrapped = SimpleNamespace(
            inner_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet), is_inpaint=False)
        )

        class Denoiser(classes.CFGDenoiser):
            @property
            def inner_model(self):
                return wrapped

        denoiser = Denoiser(SimpleNamespace(sampler_extra_args={}))
        denoiser.p = SimpleNamespace(is_hr_pass=False, seeds=[1], extra_generation_params={})
        denoiser.total_steps = 1
        cond = prompt_parser.get_multicond_learned_conditioning(model, prompts, 1)
        uncond = prompt_parser.get_learned_conditioning(model, [negative_prompt] * len(prompts), 1)
        composition, stacked = prompt_parser.reconstruct_multicond_batch(cond, 0)
        uc = prompt_parser.reconstruct_cond_batch(uncond, 0)
        x = torch.zeros((len(prompts), 1) + (2,) * (ndim - 2))
        if ndim == 5:
            x = torch.zeros(len(prompts), 1, frames, 2, 2)
        cond_tensor = stacked["crossattn"] if dictionary else stacked
        uncond_tensor = uc["crossattn"] if dictionary else uc
        x_out = torch.cat(
            [
                tensor[:, 0, 0].reshape((-1,) + (1,) * (ndim - 1)).expand((-1,) + x.shape[1:])
                for tensor in (cond_tensor, uncond_tensor)
            ]
        )
        if expected is None:
            expected = denoiser.combine_denoised(x_out, composition, uc, scale, None, x, stacked)
        if context_handler is not None:
            context_handler.orig_lq_latent = torch.zeros_like(x)
            unet.model_options["transformer_options"] = {"sampling_sigmas": torch.tensor([1.0, 0.0])}
        with (
            patch.object(sampling_function.dynamic_args, "context_handler", context_handler),
            patch.object(sampling_function.dynamic_args, "lq_latent", [None, None]),
        ):
            actual = sampler_impl.sample_euler(
                denoiser,
                x,
                torch.tensor([1.0, 0.0]),
                extra_args={
                    "cond": cond,
                    "uncond": uncond,
                    "cond_scale": scale,
                    "s_min_uncond": 0.0,
                    "image_cond": None,
                },
                disable=True,
            )
        torch.testing.assert_close(actual, expected)
        return stacked, composition

    def test_different_component_counts_preserve_all_prompts_in_both_batch_orders(self):
        for prompts in (["a AND b", "c"], ["c", "a AND b"]):
            for dictionary in (False, True):
                with self.subTest(prompts=prompts, dictionary=dictionary):
                    self.run_euler(prompts, dictionary=dictionary)

    def test_different_image_weights_work_for_image_and_video_latents(self):
        for dictionary in (False, True):
            for ndim in (4, 5):
                with self.subTest(dictionary=dictionary, ndim=ndim):
                    self.run_euler(["a:2", "c:1"], dictionary=dictionary, ndim=ndim)

    def test_zero_and_negative_component_weights_keep_nonzero_row_totals(self):
        for prompts in (["a:0 AND b:2", "c:1 AND a:0.5"], ["a:-1 AND b:2", "c:0.5 AND a:1"]):
            with self.subTest(prompts=prompts):
                self.run_euler(prompts)

    def test_uniform_weights_keep_scalar_strengths(self):
        stacked, composition = self.run_euler(["a:2 AND b:0.5", "c:2 AND a:0.5"])
        compiled = compile_weighted_conditions(stacked, composition)
        self.assertEqual([item["strength"] for item in compiled], [2.0, 0.5])

    def test_cfg_one_weighted_prompts_keep_negative_conditioning(self):
        for prompts in (["a:2", "c:1"], ["a:2", "c:2"], ["a AND b", "c"]):
            with self.subTest(prompts=prompts):
                self.run_euler(prompts, scale=1.0, negative_prompt="b")

    def test_cfg_one_unit_total_keeps_existing_optimization(self):
        for prompts in (["a", "c"], ["a:0.2 AND b:0.8", "c:0.7 AND a:0.3"]):
            with self.subTest(prompts=prompts):
                calls = []

                def post_cfg(args, calls=calls):
                    calls.append(args)
                    return args["denoised"]

                self.run_euler(
                    prompts, scale=1.0, negative_prompt="b", model_options={"sampler_post_cfg_function": [post_cfg]}
                )
                self.assertEqual(len(calls), 1)
                torch.testing.assert_close(calls[0]["uncond_denoised"], torch.zeros(2, 1, 2, 2))

    def test_explicitly_disabled_cfg_one_optimization_still_evaluates_negative_conditioning(self):
        calls = []

        def post_cfg(args):
            calls.append(args)
            return args["denoised"]

        self.run_euler(
            ["a", "c"],
            scale=1.0,
            negative_prompt="b",
            model_options={"sampler_post_cfg_function": [post_cfg], "disable_cfg1_optimization": True},
        )
        self.assertEqual(len(calls), 1)
        torch.testing.assert_close(calls[0]["uncond_denoised"], torch.full((2, 1, 2, 2), 2.0))

    def test_post_cfg_receives_weighted_average_for_each_original_batch_row(self):
        calls = []

        def post_cfg(args):
            calls.append(args)
            return args["denoised"]

        self.run_euler(["a:2 AND b:1", "c:1 AND a:0.5"], model_options={"sampler_post_cfg_function": [post_cfg]})
        self.assertEqual(len(calls), 1)
        expected = torch.tensor([4 / 3, 3.0]).reshape(2, 1, 1, 1).expand(2, 1, 2, 2)
        torch.testing.assert_close(calls[0]["cond_denoised"], expected)
        torch.testing.assert_close(calls[0]["uncond_denoised"], torch.zeros_like(expected))

    def test_controlnet_frame_weighting_keeps_one_full_batch(self):
        control = FrameControl()
        self.run_euler(["a AND b", "c"], control=control)
        self.assertEqual(len(control.signals), 1)
        expected = torch.tensor([0.25, 0.75] * 3).reshape(6, 1, 1, 1).expand(6, 1, 2, 2)
        torch.testing.assert_close(control.signals[0], expected)


if __name__ == "__main__":
    unittest.main()
