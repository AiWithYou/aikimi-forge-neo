"""CPU tensor regressions; no pretrained weights, GPU or downloads are used."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def load_vae_module(filename):
    """Load real VAE definitions while isolating application and weight I/O."""
    package = ModuleType("modules")
    package.__path__ = []
    for name in ("devices", "paths", "paths_internal", "shared"):
        setattr(package, name, ModuleType(f"modules.{name}"))
    backend = ModuleType("backend")
    backend.__path__ = []
    state_dict = ModuleType("backend.state_dict")
    state_dict.load_state_dict = Mock()
    utils = ModuleType("backend.utils")
    utils.load_torch_file = Mock(return_value={})
    spec = importlib.util.spec_from_file_location("_vae_efficiency_" + Path(filename).stem, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    stubs = {"modules": package, "backend": backend, "backend.state_dict": state_dict, "backend.utils": utils}
    stubs.update({
        f"modules.{name}": getattr(package, name)
        for name in ("devices", "paths", "paths_internal", "shared")
    })
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def reference_preview(sample, factors, bias=None):
    """The previous per-frame projection, including its singleton conventions."""
    sample = sample[0]
    factors = factors.to(sample)
    bias = bias.to(sample) if bias is not None else None
    if sample.ndim == 4:
        frames = [
            torch.nn.functional.linear(frame.movedim(0, -1), factors, bias).permute(2, 0, 1)
            for frame in sample.unbind(1)
        ]
        return torch.stack(frames).unsqueeze(0).squeeze(1)
    return torch.nn.functional.linear(sample.movedim(0, -1), factors, bias).permute(2, 0, 1).unsqueeze(0)


class PreviewProjectionTests(unittest.TestCase):
    def setUp(self):
        self.module = load_vae_module("modules/sd_vae_approx.py")
        self.module.shared.sd_model = SimpleNamespace(model_config=SimpleNamespace(latent_format=object()))
        self.generator = torch.Generator().manual_seed(23)

    def project(self, sample, factors, bias=None, reshape=None):
        with patch.object(self.module, "latent_format", return_value=(factors, bias, reshape)):
            return self.module.cheap_approximation(sample)

    def test_video_projection_matches_per_frame_values_and_shape(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            for frames in (1, 7):
                for use_bias in (False, True):
                    with self.subTest(dtype=dtype, frames=frames, bias=use_bias):
                        sample = torch.randn(2, 16, frames, 5, 9, generator=self.generator).to(dtype)
                        factors = torch.randn(3, 16, generator=self.generator)
                        bias = torch.randn(3, generator=self.generator) if use_bias else None
                        actual = self.project(sample, factors, bias)
                        expected = reference_preview(sample, factors, bias)
                        torch.testing.assert_close(actual, expected)
                        self.assertEqual(actual.dtype, dtype)
                        self.assertEqual(actual.device, sample.device)

    def test_video_uses_one_projection_and_no_frame_stack(self):
        sample = torch.ones(1, 16, 17, 4, 6)
        with patch("torch.nn.functional.linear", wraps=torch.nn.functional.linear) as linear:
            with patch("torch.stack", side_effect=AssertionError("unnecessary frame stack")):
                actual = self.project(sample, torch.ones(3, 16))
        self.assertEqual(linear.call_count, 1)
        self.assertEqual(actual.shape, (1, 17, 3, 4, 6))
        self.assertTrue(torch.all(actual == 16))

    def test_noncontiguous_video_and_reshape_hook_preserve_first_batch(self):
        sample = torch.randn(2, 4, 3, 5, 7, generator=self.generator).transpose(-1, -2)
        sample[1].fill_(1000)
        factors = torch.randn(3, 4, generator=self.generator)
        reshape = Mock(side_effect=lambda value: value * 0.5)
        before = sample.clone()
        actual = self.project(sample, factors, reshape=reshape)
        torch.testing.assert_close(actual, reference_preview(sample * 0.5, factors))
        torch.testing.assert_close(sample, before)
        reshape.assert_called_once_with(sample)
        frames = ((actual[0] + 1) * 127.5).clamp(0, 255).to(torch.uint8).movedim(1, -1).numpy()
        self.assertEqual(frames.shape, (3, 7, 5, 3))

    def test_still_image_projection_is_unchanged(self):
        sample = torch.randn(2, 4, 5, 9, generator=self.generator)
        factors = torch.randn(3, 4, generator=self.generator)
        bias = torch.randn(3, generator=self.generator)
        torch.testing.assert_close(self.project(sample, factors, bias), reference_preview(sample, factors, bias))


class TaesdPackingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        previous_threads = torch.get_num_threads()
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        torch.set_num_threads(1)
        cls.module = load_vae_module("modules/sd_vae_taesd.py")

    def test_128_channel_encoder_encodes_rgb_before_packing(self):
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(31)
            encoder = self.module.TAESDEncoder("unused.pth", 128).eval()
            image = torch.rand(2, 3, 32, 48)
            raw = encoder.encoder(image)
            with patch.object(encoder.encoder, "forward", wraps=encoder.encoder.forward) as forward:
                actual = encoder(image)
            forward.assert_called_once_with(image)
            self.assertEqual(actual.shape, (2, 128, 2, 3))
            # Exercise the real decoder wrapper's inverse layout, not a duplicate formula.
            decoder = self.module.TAESDDecoder.__new__(self.module.TAESDDecoder)
            nn.Module.__init__(decoder)
            decoder.latent_channels = 32
            decoder.decoder = nn.Identity()
            torch.testing.assert_close(decoder(actual), raw, rtol=0, atol=0)

    def test_legacy_encoders_return_the_unmodified_encoded_tensor(self):
        for channels in (4, 16):
            with self.subTest(channels=channels), torch.random.fork_rng(devices=[]), torch.inference_mode():
                torch.manual_seed(31)
                encoder = self.module.TAESDEncoder("unused.pth", channels).eval()
                image = torch.rand(1, 3, 16, 24)
                raw = encoder.encoder(image)
                with patch.object(encoder.encoder, "forward", return_value=raw) as forward:
                    actual = encoder(image)
                self.assertIs(actual, raw)
                self.assertEqual(actual.shape, (1, channels, 2, 3))
                forward.assert_called_once_with(image)


if __name__ == "__main__":
    unittest.main()
