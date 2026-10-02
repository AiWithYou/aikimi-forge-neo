"""CPU Brownian noise follows the latent batch rather than video frame count."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from modules_forge.packages.k_diffusion import sampling

ROOT = Path(__file__).resolve().parents[2]


class TinyDenoiser:
    def __init__(self, prediction_type="const"):
        self.inner_model = SimpleNamespace(
            predictor=SimpleNamespace(prediction_type=prediction_type, percent_to_sigma=lambda percent: 1.0 - percent)
        )
        self.calls = []

    def __call__(self, values, sigmas, **_kwargs):
        result = values * 0.25 + sigmas.reshape((-1,) + (1,) * (values.ndim - 1)) * 0.1
        self.calls.append(result.clone())
        return result


class SamplerNoiseBatchContractTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.random.fork_rng(devices=[]))
        package = ModuleType("k_diffusion")
        package.sampling = sampling
        self.enterContext(patch.dict(sys.modules, {"k_diffusion": package, "k_diffusion.sampling": sampling}))
        tree = ast.parse((ROOT / "modules/sd_samplers_common.py").read_text(encoding="utf-8"))
        sampler = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Sampler")
        method = next(
            node for node in sampler.body if isinstance(node, ast.FunctionDef) and node.name == "create_noise_sampler"
        )
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), "modules/sd_samplers_common.py", "exec"), namespace)  # noqa: S102
        self.create_noise = namespace["create_noise_sampler"]
        self.sigmas = torch.tensor([0.9, 0.5, 0.1, 0.0])

    def test_video_frame_count_does_not_become_the_brownian_batch_size(self):
        # Existing WAN txt2img/img2img callers derive temporal latent length
        # from p.batch_size, then enforce a single video in x.shape[0].
        frame_count = 9
        values = torch.linspace(-0.5, 0.5, 2 * 3 * 4 * 4).reshape(1, 2, 3, 4, 4)
        for iteration in (0, 1):
            for name in ("sample_dpmpp_sde", "sample_dpmpp_2m_sde", "sample_dpmpp_3m_sde"):
                with self.subTest(iteration=iteration, sampler=name):
                    parameters = SimpleNamespace(
                        batch_size=frame_count, iteration=iteration, all_seeds=list(range(21, 39))
                    )
                    noise = self.create_noise(None, values, self.sigmas, parameters)
                    expected_noise = sampling.BrownianTreeNoiseSampler(
                        values, 0.1, 0.9, seed=[parameters.all_seeds[iteration * frame_count]]
                    )
                    torch.testing.assert_close(noise(0.9, 0.5), expected_noise(0.9, 0.5))
                    events = []
                    model = TinyDenoiser()
                    result = getattr(sampling, name)(
                        model, values.clone(), self.sigmas, noise_sampler=noise, callback=events.append, disable=True
                    )
                    self.assertEqual(result.shape, values.shape)
                    self.assertTrue(torch.isfinite(result).all())
                    torch.testing.assert_close(result, events[-1]["denoised"])
                    self.assertEqual([event["i"] for event in events], [0, 1, 2])

    def test_image_batch_matches_individual_seeded_brownian_outputs(self):
        values = torch.zeros(2, 2, 4, 4)
        parameters = SimpleNamespace(batch_size=2, iteration=1, all_seeds=[21, 22, 31, 32])
        noise = self.create_noise(None, values, self.sigmas, parameters)
        expected = torch.cat(
            [sampling.BrownianTreeNoiseSampler(values[:1], 0.1, 0.9, seed=[seed])(0.9, 0.5) for seed in (31, 32)]
        )
        torch.testing.assert_close(noise(0.9, 0.5), expected)

    def test_one_step_and_final_sigma_zero_do_not_request_brownian_noise(self):
        values = torch.zeros(1, 2, 4, 4)
        for name in ("sample_dpmpp_sde", "sample_dpmpp_2m_sde", "sample_dpmpp_3m_sde"):
            with self.subTest(sampler=name):
                model = TinyDenoiser("eps")
                result = getattr(sampling, name)(model, values.clone(), torch.tensor([0.9, 0.0]), disable=True)
                self.assertEqual(len(model.calls), 1)
                torch.testing.assert_close(result, model.calls[0])
                no_noise = Mock(side_effect=AssertionError("The final denoising step requested noise."))
                getattr(sampling, name)(
                    model, values.clone(), torch.tensor([0.9, 0.0]), disable=True, noise_sampler=no_noise
                )
                no_noise.assert_not_called()


if __name__ == "__main__":
    unittest.main()
