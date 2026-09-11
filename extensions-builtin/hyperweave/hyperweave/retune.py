"""Generation-free final-frequency retuning for HyperWeave outputs.

This module deliberately does not import Forge, torch, a text encoder, a VAE,
or a generator adapter. It treats an existing HyperWeave result as the fixed
semantic solution and only rescales its multi-band residual relative to the
original source image on CPU.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

RETUNE_VERSION = "1.0.0"


@dataclass(frozen=True)
class RetuneSettings:
    high_detail: float = 1.0
    mid_detail: float = 1.0
    low_detail: float = 1.0
    chroma_detail: float = 1.0
    tile_size: int = 1024
    mask_channel: str = "Luminance"

    def validate(self) -> None:
        for name in ("high_detail", "mid_detail", "low_detail", "chroma_detail"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0 or value > 2.0:
                raise ValueError(f"{name} must be finite and between 0 and 2.")
        if int(self.tile_size) < 256 or int(self.tile_size) > 4096:
            raise ValueError("tile_size must be between 256 and 4096.")
        if self.mask_channel not in {"Luminance", "Alpha"}:
            raise ValueError("mask_channel must be Luminance or Alpha.")

    @property
    def neutral(self) -> bool:
        return all(
            abs(float(value) - 1.0) <= 1e-12
            for value in (
                self.high_detail,
                self.mid_detail,
                self.low_detail,
                self.chroma_detail,
            )
        )


@dataclass(frozen=True)
class RetuneMetrics:
    version: str
    elapsed_seconds: float
    width: int
    height: int
    tile_size: int
    neutral_fast_path: bool
    changed_pixel_fraction: float
    mean_absolute_difference: float
    max_absolute_difference: int
    settings: dict[str, object]


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _mask_image(mask: Image.Image | None, size: tuple[int, int], channel: str) -> Image.Image | None:
    if mask is None:
        return None
    mask = ImageOps.exif_transpose(mask)
    if channel == "Alpha" and "A" in mask.getbands():
        gray = mask.getchannel("A")
    else:
        gray = mask.convert("L")
    if gray.size != size:
        gray = gray.resize(size, Image.Resampling.BILINEAR)
    return gray


def _luma_chroma(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = (
        0.2126 * values[..., 0]
        + 0.7152 * values[..., 1]
        + 0.0722 * values[..., 2]
    ).astype(np.float32)
    return y, values - y[..., None]


def _scaled_band(band: np.ndarray, gain: float, chroma_gain: float) -> np.ndarray:
    y, chroma = _luma_chroma(band)
    return float(gain) * (y[..., None] + float(chroma_gain) * chroma)


def _retune_tile(
    source_rgb: np.ndarray,
    generated_rgb: np.ndarray,
    settings: RetuneSettings,
    protection: np.ndarray | None,
) -> np.ndarray:
    source_linear = _srgb_to_linear(source_rgb)
    generated_linear = _srgb_to_linear(generated_rgb)
    residual = generated_linear - source_linear

    smooth_high = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=1.1,
        sigmaY=1.1,
        borderType=cv2.BORDER_REFLECT_101,
    )
    smooth_mid = cv2.GaussianBlur(
        residual,
        (0, 0),
        sigmaX=3.0,
        sigmaY=3.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    high = residual - smooth_high
    mid = smooth_high - smooth_mid
    low = smooth_mid

    adjusted = source_linear.copy()
    adjusted += _scaled_band(high, settings.high_detail, settings.chroma_detail)
    adjusted += _scaled_band(mid, settings.mid_detail, settings.chroma_detail)
    adjusted += _scaled_band(low, settings.low_detail, settings.chroma_detail)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    if protection is not None:
        weight = np.clip(protection.astype(np.float32), 0.0, 1.0)[..., None]
        adjusted = adjusted * (1.0 - weight) + generated_linear * weight

    return np.clip(_linear_to_srgb(adjusted), 0.0, 1.0)


def retune_image(
    source: Image.Image,
    generated: Image.Image,
    settings: RetuneSettings,
    *,
    protection_mask: Image.Image | None = None,
) -> tuple[Image.Image, RetuneMetrics]:
    """Retune an existing generated image without invoking model code.

    White protection-mask pixels keep the existing HyperWeave result. A
    neutral setting returns a pixel-exact copy of ``generated`` before any
    floating-point conversion.
    """

    started = time.perf_counter()
    settings.validate()
    source = ImageOps.exif_transpose(source)
    generated = ImageOps.exif_transpose(generated)
    if min(source.size + generated.size) <= 0:
        raise ValueError("Source and generated image sizes must be positive.")

    if settings.neutral:
        result = generated.copy()
        result.info.update(generated.info)
        metrics = RetuneMetrics(
            version=RETUNE_VERSION,
            elapsed_seconds=time.perf_counter() - started,
            width=generated.width,
            height=generated.height,
            tile_size=int(settings.tile_size),
            neutral_fast_path=True,
            changed_pixel_fraction=0.0,
            mean_absolute_difference=0.0,
            max_absolute_difference=0,
            settings=asdict(settings),
        )
        return result, metrics

    source_resized = source.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS)
    protection = _mask_image(protection_mask, generated.size, settings.mask_channel)
    alpha = generated.getchannel("A") if "A" in generated.getbands() else None
    output_rgb = Image.new("RGB", generated.size)

    tile_size = int(settings.tile_size)
    margin = 24
    changed_pixels = 0
    total_pixels = generated.width * generated.height
    absolute_sum = 0.0
    absolute_count = 0
    maximum_difference = 0

    for y0 in range(0, generated.height, tile_size):
        y1 = min(generated.height, y0 + tile_size)
        for x0 in range(0, generated.width, tile_size):
            x1 = min(generated.width, x0 + tile_size)
            ex0 = max(0, x0 - margin)
            ey0 = max(0, y0 - margin)
            ex1 = min(generated.width, x1 + margin)
            ey1 = min(generated.height, y1 + margin)
            ext_box = (ex0, ey0, ex1, ey1)

            source_values = _rgb_array(source_resized.crop(ext_box))
            generated_values = _rgb_array(generated.crop(ext_box))
            mask_values = None
            if protection is not None:
                mask_values = np.asarray(protection.crop(ext_box), dtype=np.float32) / 255.0

            adjusted = _retune_tile(source_values, generated_values, settings, mask_values)
            cx0, cy0 = x0 - ex0, y0 - ey0
            cx1, cy1 = cx0 + (x1 - x0), cy0 + (y1 - y0)
            core = adjusted[cy0:cy1, cx0:cx1]
            core_u8 = np.clip(np.rint(core * 255.0), 0, 255).astype(np.uint8)
            output_rgb.paste(Image.fromarray(core_u8, mode="RGB"), (x0, y0))

            generated_core = np.asarray(
                generated.crop((x0, y0, x1, y1)).convert("RGB"), dtype=np.int16
            )
            difference = np.abs(core_u8.astype(np.int16) - generated_core)
            changed_pixels += int(np.count_nonzero(np.any(difference != 0, axis=2)))
            absolute_sum += float(np.sum(difference, dtype=np.float64))
            absolute_count += int(difference.size)
            maximum_difference = max(maximum_difference, int(difference.max(initial=0)))

    if alpha is None:
        result = output_rgb
    else:
        result = output_rgb.convert("RGBA")
        result.putalpha(alpha)
    result.info.update(generated.info)

    metrics = RetuneMetrics(
        version=RETUNE_VERSION,
        elapsed_seconds=time.perf_counter() - started,
        width=generated.width,
        height=generated.height,
        tile_size=tile_size,
        neutral_fast_path=False,
        changed_pixel_fraction=changed_pixels / max(1, total_pixels),
        mean_absolute_difference=absolute_sum / max(1, absolute_count),
        max_absolute_difference=maximum_difference,
        settings=asdict(settings),
    )
    return result, metrics
