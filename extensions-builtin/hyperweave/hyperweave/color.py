"""Stable sRGB/linear-RGB and premultiplied-alpha image operations."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.clip(values, 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def image_to_linear_rgb(image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.float32) / 255.0
    alpha = array[..., 3]
    rgb = srgb_to_linear(array[..., :3])
    if image.mode == "RGBA" or "transparency" in image.info:
        return rgb, alpha
    return rgb, None


def linear_rgb_to_image(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    info: dict[str, str] | None = None,
) -> Image.Image:
    rgb = np.asarray(rgb, dtype=np.float32)
    if not np.isfinite(rgb).all():
        raise ValueError("Linear RGB contains NaN or Inf.")
    srgb = np.clip(np.rint(linear_to_srgb(rgb) * 255.0), 0, 255).astype(np.uint8)
    if alpha is None:
        image = Image.fromarray(srgb, mode="RGB")
    else:
        alpha = np.asarray(alpha, dtype=np.float32)
        if alpha.shape != rgb.shape[:2]:
            raise ValueError("Alpha shape does not match RGB.")
        rgba = np.dstack(
            [srgb, np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)]
        )
        image = Image.fromarray(rgba, mode="RGBA")
    if info:
        image.info.update(info)
    return image


def _resize_float(
    array: np.ndarray, size: tuple[int, int], interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    width, height = size
    result = cv2.resize(array, (width, height), interpolation=interpolation)
    if array.ndim == 3 and result.ndim == 2:
        result = result[..., None]
    return np.asarray(result, dtype=np.float32)


def resize_linear_rgb(
    image: Image.Image | np.ndarray,
    size: tuple[int, int],
    *,
    preserve_alpha: bool = True,
) -> Image.Image | np.ndarray:
    if isinstance(image, Image.Image):
        rgb, alpha = image_to_linear_rgb(image)
        if alpha is None or not preserve_alpha:
            resized = _resize_float(rgb, size)
            return linear_rgb_to_image(resized)

        premultiplied = rgb * alpha[..., None]
        resized_alpha = np.clip(_resize_float(alpha, size), 0.0, 1.0)
        resized_premultiplied = _resize_float(premultiplied, size)
        safe_alpha = np.maximum(resized_alpha[..., None], 1e-6)
        resized_rgb = np.where(
            resized_alpha[..., None] > 1e-6,
            resized_premultiplied / safe_alpha,
            0.0,
        )
        return linear_rgb_to_image(
            np.clip(resized_rgb, 0.0, 1.0), resized_alpha, info=dict(image.info)
        )
    return _resize_float(np.asarray(image, dtype=np.float32), size)


def flatten_for_model(
    image: Image.Image,
    *,
    background: tuple[int, int, int] | None = None,
) -> Image.Image:
    if image.mode != "RGBA":
        return image.convert("RGB")
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    if background is None:
        border = np.concatenate(
            [
                rgba[0, :, :3],
                rgba[-1, :, :3],
                rgba[:, 0, :3],
                rgba[:, -1, :3],
            ],
            axis=0,
        )
        bg = np.median(border, axis=0)
    else:
        bg = np.asarray(background, dtype=np.float32) / 255.0
    flattened = rgba[..., :3] * alpha + bg[None, None, :] * (1.0 - alpha)
    return Image.fromarray(
        np.clip(np.rint(flattened * 255.0), 0, 255).astype(np.uint8), mode="RGB"
    )


def luminance(linear_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(linear_rgb, dtype=np.float32)
    return (
        0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    ).astype(np.float32)


def rgb_to_luma_chroma(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.asarray(rgb, dtype=np.float32)
    y = luminance(rgb)
    chroma = rgb - y[..., None]
    return y, chroma


def luma_chroma_to_rgb(y: np.ndarray, chroma: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float32)[..., None] + np.asarray(
        chroma, dtype=np.float32
    )
