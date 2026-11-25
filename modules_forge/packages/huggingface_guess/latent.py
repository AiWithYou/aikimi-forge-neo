# reference: https://github.com/Comfy-Org/ComfyUI/blob/v0.24.1/comfy/latent_formats.py

import torch


class LatentFormat:
    scale_factor: float = 1.0
    latent_channels: int = 4
    latent_rgb_factors: list[list[float]] = None
    latent_rgb_factors_bias: list[list[float]] = None
    taesd_decoder_name: str = None

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return latent * self.scale_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return latent / self.scale_factor


class SDXL(LatentFormat):
    def __init__(self):
        self.scale_factor = 0.13025
        self.latent_rgb_factors = [
            #      R        G        B
            [ 0.3651,  0.4232,  0.4341],
            [-0.2533, -0.0042,  0.1068],
            [ 0.1076,  0.1111, -0.0362],
            [-0.3165, -0.2492, -0.2188],
        ]
        self.latent_rgb_factors_bias = [0.1084, -0.0175, -0.0011]
        self.taesd_decoder_name = "taesdxl_decoder"


class Wan21(LatentFormat):
    def __init__(self):
        self.latent_channels = 16
        self.scale_factor = 1.0
        self.latent_rgb_factors = [
            [-0.1299, -0.1692,  0.2932],
            [ 0.0671,  0.0406,  0.0442],
            [ 0.3568,  0.2548,  0.1747],
            [ 0.0372,  0.2344,  0.1420],
            [ 0.0313,  0.0189, -0.0328],
            [ 0.0296, -0.0956, -0.0665],
            [-0.3477, -0.4059, -0.2925],
            [ 0.0166,  0.1902,  0.1975],
            [-0.0412,  0.0267, -0.1364],
            [-0.1293,  0.0740,  0.1636],
            [ 0.0680,  0.3019,  0.1128],
            [ 0.0032,  0.0581,  0.0639],
            [-0.1251,  0.0927,  0.1699],
            [ 0.0060, -0.0633,  0.0005],
            [ 0.3477,  0.2275,  0.2950],
            [ 0.1984,  0.0913,  0.1861],
        ]
        self.latent_rgb_factors_bias = [-0.1835, -0.0868, -0.3360]
        self.taesd_decoder_name = "taew2_1"

        self.latents_mean = torch.tensor([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).view(1, self.latent_channels, 1, 1, 1)
        self.latents_std = torch.tensor([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]).view(1, self.latent_channels, 1, 1, 1)

    def process_in(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return (latent - latents_mean) * self.scale_factor / latents_std

    def process_out(self, latent):
        latents_mean = self.latents_mean.to(latent.device, latent.dtype)
        latents_std = self.latents_std.to(latent.device, latent.dtype)
        return latent * latents_std / self.scale_factor + latents_mean
