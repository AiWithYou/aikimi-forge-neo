"""Generator adapter contract and deterministic CPU stub generator."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .color import image_to_linear_rgb, linear_rgb_to_image, luminance


@dataclass(frozen=True)
class GenerationRequest:
    stage_index: int
    pass_name: str
    candidate_index: int
    roi_id: int
    strength: float
    steps: int
    seed: int
    prompt_suffix: str
    negative_suffix: str
    coordinate_noise: np.ndarray
    absolute_input_box: tuple[int, int, int, int]
    canvas_size: tuple[int, int]


class GeneratorAdapter(Protocol):
    latent_channels: int
    latent_scale: int
    last_processed: object | None

    def generate(
        self, image: Image.Image, request: GenerationRequest
    ) -> Image.Image: ...

    def model_metadata(self) -> dict[str, object]: ...

    def runtime_metrics(self) -> dict[str, object]: ...

    def pass_cleanup(self) -> None: ...


def _seed_from_request(request: GenerationRequest) -> int:
    payload = (
        f"{request.seed}:{request.stage_index}:{request.pass_name}:"
        f"{request.candidate_index}:{request.roi_id}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")


class StubGenerator:
    """Model-free generator that exposes coherent, shifted, noisy, and crossing modes."""

    latent_channels = 4
    latent_scale = 8
    last_processed = None

    def __init__(self, mode_cycle: tuple[str, ...] = ("coherent", "shift", "noise", "cross")):
        self.mode_cycle = mode_cycle
        self.calls: list[GenerationRequest] = []

    def generate(
        self, image: Image.Image, request: GenerationRequest
    ) -> Image.Image:
        self.calls.append(request)
        rgb, _ = image_to_linear_rgb(image.convert("RGB"))
        y = luminance(rgb)
        strength = float(np.clip(request.strength, 0.0, 1.0))
        if request.pass_name == "anchor":
            mode = "anchor"
        else:
            mode = self.mode_cycle[request.candidate_index % len(self.mode_cycle)]
        rng = np.random.Generator(np.random.PCG64(_seed_from_request(request)))
        if mode == "localized_failure":
            x0, _, x1, _ = request.absolute_input_box
            mode = (
                "coherent"
                if (x0 + x1) * 0.5 <= request.canvas_size[0] * 0.5
                else "corrupt"
            )

        if mode == "anchor":
            smooth = cv2.GaussianBlur(rgb, (0, 0), 1.0)
            result = rgb + (rgb - smooth) * (0.10 + 0.15 * strength)
            yy, xx = np.mgrid[: y.shape[0], : y.shape[1]]
            phase = rng.uniform(0, 2 * np.pi)
            seeded_detail = np.sin(xx * 0.31 + yy * 0.17 + phase)
            edge = np.clip(
                cv2.magnitude(
                    cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3),
                    cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3),
                )
                * 2.0,
                0.0,
                1.0,
            )
            result += seeded_detail[..., None] * edge[..., None] * 0.0025
        elif mode == "coherent":
            gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            tangent = np.arctan2(gy, gx) + np.pi / 2.0
            yy, xx = np.mgrid[: y.shape[0], : y.shape[1]]
            phase = (
                xx * np.cos(tangent) + yy * np.sin(tangent)
            ) * 0.19 + rng.uniform(0, 2 * np.pi)
            line = np.sin(phase) * np.clip(magnitude * 2.5, 0.0, 1.0)
            detail = line[..., None] * np.array([0.7, 0.85, 1.0], dtype=np.float32)
            result = rgb + detail * (0.012 + 0.030 * strength)
            result += (rgb - cv2.GaussianBlur(rgb, (0, 0), 1.3)) * (
                0.12 + 0.20 * strength
            )
        elif mode == "shift":
            matrix = np.float32([[1, 0, 4 + round(8 * strength)], [0, 1, 0]])
            result = cv2.warpAffine(
                rgb,
                matrix,
                (rgb.shape[1], rgb.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        elif mode == "corrupt":
            result = 1.0 - rgb[..., ::-1]
        elif mode == "noise":
            noise = rng.standard_normal(rgb.shape, dtype=np.float32)
            result = rgb + noise * (0.035 + 0.08 * strength)
        else:
            yy, _ = np.mgrid[: rgb.shape[0], : rgb.shape[1]]
            stripes = np.sin(yy * 0.55 + rng.uniform(0, 2 * np.pi))
            result = rgb + stripes[..., None] * (0.025 + 0.055 * strength)
        return linear_rgb_to_image(np.clip(result, 0.0, 1.0))

    def model_metadata(self) -> dict[str, object]:
        return {
            "adapter": "StubGenerator",
            "model": "deterministic CPU stub",
            "latent_channels": self.latent_channels,
            "latent_scale": self.latent_scale,
        }

    def runtime_metrics(self) -> dict[str, object]:
        return {}

    def pass_cleanup(self) -> None:
        return
