"""Use small CPU autoencoders to exercise cloned VAE tiled paths."""

import types
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from backend.patcher.vae import VAE


class TinyAutoencoder(torch.nn.Module):
    def __init__(self, spatial_scale=8, video=False):
        super().__init__()
        self.config = types.SimpleNamespace(latent_channels=3, z_dim=3)
        self.spatial_scale = spatial_scale
        self.video = video

    def encode(self, pixels):
        if self.video:
            return pixels[:, :, ::4, :: self.spatial_scale, :: self.spatial_scale]
        return F.avg_pool2d(pixels, self.spatial_scale)

    def decode(self, latents):
        if self.video:
            size = (latents.shape[2] * 4 - 3, *(axis * self.spatial_scale for axis in latents.shape[3:]))
            return F.interpolate(latents, size=size, mode="nearest")
        return F.interpolate(latents, scale_factor=self.spatial_scale, mode="nearest")


class VaeCloneTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch("backend.memory_management.load_models_gpu"))
        self.enterContext(patch("backend.memory_management.vae_offload_device", return_value=torch.device("cpu")))
        self.enterContext(patch("backend.memory_management.intermediate_device", return_value=torch.device("cpu")))

    def make_vae(self, **kwargs):
        model = TinyAutoencoder(spatial_scale=16 if kwargs.get("is_flux2") else 8, video=kwargs.get("is_wan", False))
        return VAE(model=model, device=torch.device("cpu"), dtype=torch.float32, **kwargs)

    @torch.inference_mode()
    def test_cloned_image_vae_preserves_tiled_decode(self):
        for flags in ({}, {"is_flux2": True}):
            with self.subTest(flags=flags):
                vae = self.make_vae(**flags)
                clone = vae.clone()
                latents = torch.linspace(-1, 1, 12).reshape(1, 3, 2, 2)
                torch.testing.assert_close(clone.decode_tiled(latents), vae.decode_tiled(latents))
                self.assertIsNot(clone.patcher, vae.patcher)
                self.assertIs(clone.first_stage_model, vae.first_stage_model)

    @torch.inference_mode()
    def test_cloned_video_vae_preserves_tiled_encode_and_decode(self):
        vae = self.make_vae(is_wan=True)
        clone = vae.clone()
        pixels = torch.linspace(0, 1, 5 * 16 * 16 * 3).reshape(5, 16, 16, 3)
        original_latents = vae.encode_tiled(pixels)
        cloned_latents = clone.encode_tiled(pixels)
        torch.testing.assert_close(cloned_latents, original_latents)
        torch.testing.assert_close(clone.decode_tiled(cloned_latents), vae.decode_tiled(original_latents))


if __name__ == "__main__":
    unittest.main()
