"""A valid zero-strength DDIM/PLMS img2img request leaves its latent unchanged."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import tqdm

from tools.tests.test_runtime_efficiency import load_definitions


class TimestepZeroDenoisingTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        self.alphas = torch.linspace(0.999, 0.01, 1000)
        self.latent = torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4)
        self.noise = torch.linspace(0.1, 0.9, 16).reshape_as(self.latent)

    def sampler(self, name, fixed_steps):
        options = SimpleNamespace(
            always_discard_next_to_last_sigma=False, img2img_fix_steps=fixed_steps, img2img_extra_noise=0.0
        )
        setup_steps = load_definitions("modules/sd_samplers_common.py", {"setup_img2img_steps"}, {"opts": options})[
            "setup_img2img_steps"
        ]
        model = SimpleNamespace(alphas_cumprod=self.alphas.clone(), forge_objects=SimpleNamespace(unet=object()))
        function = load_definitions(
            "modules/sd_samplers_timesteps_impl.py",
            {name},
            {
                "torch": torch,
                "np": np,
                "tqdm": tqdm,
                "float64": lambda _x: torch.float64,
                "sampling": SimpleNamespace(torch=torch),
            },
        )[name]
        prepare, cleanup = Mock(), Mock()
        sampler_class = load_definitions(
            "modules/sd_samplers_timesteps.py",
            {"CompVisSampler"},
            {
                "torch": torch,
                "inspect": inspect,
                "opts": options,
                "devices": SimpleNamespace(device=torch.device("cpu")),
                "shared": SimpleNamespace(sd_model=model),
                "sd_samplers_common": SimpleNamespace(Sampler=object, setup_img2img_steps=setup_steps),
                "sampling_prepare": prepare,
                "sampling_cleanup": cleanup,
            },
        )["CompVisSampler"]
        sampler = object.__new__(sampler_class)
        sampler.model_wrap = SimpleNamespace(inner_model=model)
        denoiser = Mock(side_effect=lambda values, _timesteps, **_kwargs: torch.zeros_like(values))
        denoiser.inner_model = SimpleNamespace(inner_model=model)
        sampler.model_wrap_cfg = denoiser
        sampler.config = None
        sampler.func = function
        sampler.initialize = lambda _p: {}
        sampler.launch_sampling = lambda _steps, callback: callback()
        sampler.add_infotext = lambda _p: None
        sampler.callback_state = lambda _event: None
        sampler.s_min_uncond = 0.0
        return sampler, prepare, cleanup, denoiser, setup_steps

    @staticmethod
    def parameters(strength):
        return SimpleNamespace(steps=4, denoising_strength=strength, cfg_scale=1.0, extra_generation_params={})

    def test_zero_strength_with_default_or_fixed_steps_returns_the_original_latent(self):
        for name in ("ddim", "plms"):
            for fixed_steps in (False, True):
                with self.subTest(sampler=name, fixed_steps=fixed_steps):
                    sampler, prepare, cleanup, denoiser, _ = self.sampler(name, fixed_steps)
                    before = torch.random.get_rng_state().clone()
                    result = sampler.sample_img2img(self.parameters(0.0), self.latent, self.noise, None, None)
                    torch.testing.assert_close(result, self.latent)
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
                    denoiser.assert_not_called()
                    prepare.assert_not_called()
                    cleanup.assert_not_called()

    def test_positive_strength_retains_the_existing_timestep_sequence_and_arithmetic(self):
        for name in ("ddim", "plms"):
            for fixed_steps in (False, True):
                with self.subTest(sampler=name, fixed_steps=fixed_steps):
                    sampler, prepare, cleanup, denoiser, setup_steps = self.sampler(name, fixed_steps)
                    parameters = self.parameters(0.5)
                    steps, index = setup_steps(parameters)
                    timesteps = sampler.get_timesteps(parameters, steps)
                    alpha_start = self.alphas[timesteps[index]]
                    scheduled = timesteps[:index]
                    noisy = self.latent * alpha_start.sqrt() + self.noise * (1 - alpha_start).sqrt()
                    # Zero epsilon makes the existing DDIM/PLMS updates telescope.
                    expected = noisy * (self.alphas[scheduled[0]] / self.alphas[scheduled[-1]]).sqrt()
                    result = sampler.sample_img2img(parameters, self.latent, self.noise, None, None)
                    torch.testing.assert_close(result, expected)
                    self.assertGreater(denoiser.call_count, 0)
                    prepare.assert_called_once()
                    cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
