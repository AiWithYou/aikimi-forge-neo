from __future__ import annotations

import cv2
import numpy as np
from PIL import Image
from skimage import color as skimage_color
from skimage.segmentation import slic

from modules.krea2_quality import adaptive_chroma_correct

SMART_MODE = "Smart Adaptive Chroma"
FAST_MODE = "Fast Lab Chroma"
SUPERPIXEL_MODE = "Superpixel Lab Chroma"
GRADIENT_MODE = "Smooth Gradient / AI Noise"

DEFAULT_GRADIENT_RADIUS = 12.0
DEFAULT_GRADIENT_DETAIL_THRESHOLD = 8.0
MAX_GRADIENT_RADIUS = 64.0
MAX_GRADIENT_DETAIL_THRESHOLD = 24.0
GRADIENT_TILE_SIZE = 768


def _validate_bgr_image(bgr: np.ndarray) -> None:
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Color Flatten expects a BGR uint8 image with 3 channels")


def _validate_strength(strength: float) -> float:
    strength = float(strength)
    if not np.isfinite(strength) or strength < 0.0 or strength > 1.0:
        raise ValueError("Color Flatten strength must be between 0.0 and 1.0")

    return strength


def _validate_positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")

    return value


def _validate_positive_float(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be greater than 0")

    return value


def _edge_strength_map(l_channel: np.ndarray, strength: float) -> np.ndarray:
    l_float = l_channel.astype(np.float32)
    grad_x = cv2.Sobel(l_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(l_float, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(grad_x, grad_y)
    scale = float(np.percentile(grad, 95))

    if scale <= 1e-6:
        return np.full(l_channel.shape, strength, dtype=np.float32)

    grad = np.clip(grad / scale, 0.0, 1.0)
    grad = cv2.GaussianBlur(grad, (0, 0), 1.0)
    return (strength * (1.0 - 0.85 * grad)).astype(np.float32)


def _light_bilateral_u8(channel: np.ndarray, sigma_space: float) -> np.ndarray:
    return cv2.bilateralFilter(
        channel, d=5, sigmaColor=10.0, sigmaSpace=max(3.0, sigma_space * 0.5)
    )


def _light_bilateral_float(channel: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(
        channel.astype(np.float32), d=5, sigmaColor=4.0, sigmaSpace=3.0
    )


def _smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    if high <= low:
        raise ValueError("smoothstep high must be greater than low")
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _normalized_gaussian_lab(
    lab: np.ndarray, alpha: np.ndarray, radius: float
) -> np.ndarray:
    border = cv2.BORDER_REFLECT_101
    if np.all(alpha == 255):
        return cv2.GaussianBlur(lab, (0, 0), sigmaX=radius, sigmaY=radius, borderType=border)

    weights = alpha.astype(np.float32) / 255.0
    numerator = cv2.GaussianBlur(
        lab * weights[..., None],
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
        borderType=border,
    )
    denominator = cv2.GaussianBlur(
        weights,
        (0, 0),
        sigmaX=radius,
        sigmaY=radius,
        borderType=border,
    )
    return np.divide(
        numerator,
        denominator[..., None],
        out=lab.copy(),
        where=denominator[..., None] > 1e-5,
    )


def _alpha_boundary_protection(alpha: np.ndarray, radius: float) -> np.ndarray:
    alpha_float = alpha.astype(np.float32)
    grad_x = cv2.Sobel(alpha_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(alpha_float, cv2.CV_32F, 0, 1, ksize=3)
    boundary = cv2.magnitude(grad_x, grad_y) > 1.0
    if not np.any(boundary):
        return np.zeros(alpha.shape, dtype=np.float32)

    distance = cv2.distanceTransform(
        (~boundary).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    guard = max(1.0, 2.0 * radius)
    return 1.0 - _smoothstep(guard, 2.0 * guard, distance)


def _structure_edge_protection(
    lab: np.ndarray, radius: float, detail_threshold: float
) -> np.ndarray:
    guide = cv2.GaussianBlur(
        lab,
        (0, 0),
        sigmaX=1.0,
        sigmaY=1.0,
        borderType=cv2.BORDER_REFLECT_101,
    )
    gradient_squared = np.zeros(lab.shape[:2], dtype=np.float32)
    for channel in range(guide.shape[2]):
        grad_x = cv2.Sobel(guide[..., channel], cv2.CV_32F, 1, 0, ksize=3) / 8.0
        grad_y = cv2.Sobel(guide[..., channel], cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gradient_squared += grad_x * grad_x + grad_y * grad_y

    edge_threshold = max(1.0, detail_threshold * 0.5)
    edges = gradient_squared > edge_threshold * edge_threshold
    if not np.any(edges):
        return np.zeros(lab.shape[:2], dtype=np.float32)

    distance = cv2.distanceTransform(
        (~edges).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    guard = max(1.0, 2.0 * radius)
    return 1.0 - _smoothstep(guard, 1.5 * guard, distance)


def _smooth_gradient_tile_rgba(
    rgba: np.ndarray,
    *,
    strength: float,
    radius: float,
    detail_threshold: float,
    edge_protect: bool,
) -> np.ndarray:
    source_rgb = np.ascontiguousarray(rgba[..., :3])
    alpha = rgba[..., 3]
    rgb_float = source_rgb.astype(np.float32) / 255.0
    source_lab = cv2.cvtColor(rgb_float, cv2.COLOR_RGB2LAB)
    reference_lab = _normalized_gaussian_lab(source_lab, alpha, radius)

    if edge_protect:
        residual = np.linalg.norm(source_lab - reference_lab, axis=2)
        raw_protection = _smoothstep(detail_threshold * 0.5, detail_threshold, residual)
        expanded = cv2.dilate(
            raw_protection, np.ones((3, 3), dtype=np.uint8), iterations=1
        )
        softened = cv2.GaussianBlur(
            expanded,
            (0, 0),
            sigmaX=0.8,
            sigmaY=0.8,
            borderType=cv2.BORDER_REFLECT_101,
        )
        detail_protection = np.maximum(
            np.maximum(raw_protection, softened),
            _structure_edge_protection(
                source_lab, radius=radius, detail_threshold=detail_threshold
            ),
        )
    else:
        detail_protection = np.zeros(alpha.shape, dtype=np.float32)

    protection = np.maximum(
        detail_protection, _alpha_boundary_protection(alpha, radius)
    )
    visible = alpha > 8
    amount = strength * (1.0 - np.clip(protection, 0.0, 1.0))
    amount *= visible.astype(np.float32)
    result_lab = source_lab + (reference_lab - source_lab) * amount[..., None]
    shift = np.linalg.norm(result_lab - source_lab, axis=2)
    changed = visible & (shift > 0.01)

    result = rgba.copy()
    if np.any(changed):
        result_rgb_float = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        result_rgb = np.clip(np.rint(result_rgb_float * 255.0), 0, 255).astype(
            np.uint8
        )
        result[..., :3][changed] = result_rgb[changed]
    return result


def smooth_gradient_pil(
    image: Image.Image,
    *,
    strength: float = 1.0,
    radius: float = DEFAULT_GRADIENT_RADIUS,
    detail_threshold: float = DEFAULT_GRADIENT_DETAIL_THRESHOLD,
    edge_protect: bool = True,
    tile_size: int = GRADIENT_TILE_SIZE,
) -> Image.Image:
    """Suppress low-contrast AI texture while retaining strong edges and alpha."""
    if not isinstance(image, Image.Image):
        raise TypeError("Smooth Gradient expects a PIL.Image.Image input")
    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError("Smooth Gradient supports RGB and RGBA images")

    strength = _validate_strength(strength)
    radius = _validate_positive_float("Smooth Gradient Radius", radius)
    detail_threshold = _validate_positive_float(
        "Smooth Gradient Detail Threshold", detail_threshold
    )
    if radius > MAX_GRADIENT_RADIUS:
        raise ValueError(
            f"Smooth Gradient Radius must be at most {MAX_GRADIENT_RADIUS:g}"
        )
    if detail_threshold > MAX_GRADIENT_DETAIL_THRESHOLD:
        raise ValueError(
            "Smooth Gradient Detail Threshold must be at most "
            f"{MAX_GRADIENT_DETAIL_THRESHOLD:g}"
        )
    tile_size = _validate_positive_int("Smooth Gradient Tile Size", tile_size)
    if strength == 0.0:
        return image.copy()

    source_rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    result_rgba = source_rgba.copy()
    halo = max(4, int(np.ceil(radius * 4.0)))

    for y in range(0, image.height, tile_size):
        y2 = min(image.height, y + tile_size)
        outer_y = max(0, y - halo)
        outer_y2 = min(image.height, y2 + halo)
        for x in range(0, image.width, tile_size):
            x2 = min(image.width, x + tile_size)
            outer_x = max(0, x - halo)
            outer_x2 = min(image.width, x2 + halo)
            tile = np.ascontiguousarray(
                source_rgba[outer_y:outer_y2, outer_x:outer_x2]
            )
            filtered = _smooth_gradient_tile_rgba(
                tile,
                strength=strength,
                radius=radius,
                detail_threshold=detail_threshold,
                edge_protect=edge_protect,
            )
            tile_y = y - outer_y
            tile_x = x - outer_x
            result_rgba[y:y2, x:x2] = filtered[
                tile_y : tile_y + (y2 - y), tile_x : tile_x + (x2 - x)
            ]

    result = Image.fromarray(result_rgba, mode="RGBA")
    return result.convert("RGB") if image.mode == "RGB" else result


def chroma_flatten_fast_bgr(
    bgr: np.ndarray,
    strength: float,
    edge_protect: bool,
    mean_shift_sp: int,
    mean_shift_sr: int,
) -> np.ndarray:
    _validate_bgr_image(bgr)
    strength = _validate_strength(strength)
    mean_shift_sp = _validate_positive_int("Mean Shift Spatial", mean_shift_sp)
    mean_shift_sr = _validate_positive_int("Mean Shift Color", mean_shift_sr)

    if strength == 0.0:
        return bgr.copy()

    source_lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    shifted = cv2.pyrMeanShiftFiltering(bgr, sp=mean_shift_sp, sr=mean_shift_sr)
    shifted_lab = cv2.cvtColor(shifted, cv2.COLOR_BGR2LAB)

    target_a = _light_bilateral_u8(shifted_lab[:, :, 1], mean_shift_sp)
    target_b = _light_bilateral_u8(shifted_lab[:, :, 2], mean_shift_sp)
    target_ab = np.stack([target_a, target_b], axis=2).astype(np.float32)
    source_ab = source_lab[:, :, 1:3].astype(np.float32)

    if edge_protect:
        blend = _edge_strength_map(source_lab[:, :, 0], strength)[:, :, None]
    else:
        blend = np.float32(strength)

    result_ab = source_ab + (target_ab - source_ab) * blend
    result_lab = source_lab.copy()
    result_lab[:, :, 1:3] = np.clip(np.rint(result_ab), 0, 255).astype(np.uint8)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def superpixel_chroma_flatten_bgr(
    bgr: np.ndarray,
    strength: float,
    edge_protect: bool,
    n_segments: int,
    compactness: float,
) -> np.ndarray:
    _validate_bgr_image(bgr)
    strength = _validate_strength(strength)
    n_segments = _validate_positive_int("SLIC Segments", n_segments)
    compactness = _validate_positive_float("SLIC Compactness", compactness)

    if strength == 0.0:
        return bgr.copy()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_float = rgb.astype(np.float32) / 255.0
    labels = slic(
        rgb_float,
        n_segments=n_segments,
        compactness=compactness,
        start_label=0,
        channel_axis=-1,
    )
    lab = skimage_color.rgb2lab(rgb_float).astype(np.float32)

    labels_flat = labels.reshape(-1)
    order = np.argsort(labels_flat, kind="stable")
    ordered_labels = labels_flat[order]
    starts = np.empty(ordered_labels.shape[0], dtype=bool)
    starts[0] = True
    starts[1:] = ordered_labels[1:] != ordered_labels[:-1]
    starts = np.flatnonzero(starts)
    ends = np.concatenate([starts[1:], np.array([ordered_labels.shape[0]])])

    source_a = lab[:, :, 1].reshape(-1)
    source_b = lab[:, :, 2].reshape(-1)
    ordered_a = source_a[order]
    ordered_b = source_b[order]
    target_a = np.empty_like(source_a)
    target_b = np.empty_like(source_b)

    for start, end in zip(starts, ends):
        segment_indices = order[start:end]
        target_a[segment_indices] = np.median(ordered_a[start:end])
        target_b[segment_indices] = np.median(ordered_b[start:end])

    target_a = target_a.reshape(labels.shape)
    target_b = target_b.reshape(labels.shape)
    target_a = _light_bilateral_float(target_a)
    target_b = _light_bilateral_float(target_b)

    if edge_protect:
        blend = _edge_strength_map(lab[:, :, 0], strength)
    else:
        blend = np.float32(strength)

    result_lab = lab.copy()
    result_lab[:, :, 1] = lab[:, :, 1] + (target_a - lab[:, :, 1]) * blend
    result_lab[:, :, 2] = lab[:, :, 2] + (target_b - lab[:, :, 2]) * blend

    result_rgb = np.clip(skimage_color.lab2rgb(result_lab), 0.0, 1.0)
    result_rgb = np.rint(result_rgb * 255.0).astype(np.uint8)
    return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)


def color_flatten_pil(
    image: Image.Image,
    mode: str,
    strength: float,
    edge_protect: bool,
    mean_shift_sp: int,
    mean_shift_sr: int,
    n_segments: int,
    compactness: float,
    gradient_radius: float = DEFAULT_GRADIENT_RADIUS,
    gradient_detail_threshold: float = DEFAULT_GRADIENT_DETAIL_THRESHOLD,
) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Color Flatten expects a PIL.Image.Image input")

    if image.mode not in {"RGB", "RGBA"}:
        raise ValueError("Color Flatten supports RGB and RGBA images")

    if mode == SMART_MODE:
        result, _ = adaptive_chroma_correct(image, strength=strength)
        return result
    if mode == GRADIENT_MODE:
        return smooth_gradient_pil(
            image,
            strength=strength,
            radius=gradient_radius,
            detail_threshold=gradient_detail_threshold,
            edge_protect=edge_protect,
        )

    if image.mode == "RGBA":
        alpha = image.getchannel("A")
        rgb_image = image.convert("RGB")
    else:
        alpha = None
        rgb_image = image

    rgb = np.asarray(rgb_image, dtype=np.uint8)
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    if mode == FAST_MODE:
        result_bgr = chroma_flatten_fast_bgr(
            bgr, strength, edge_protect, mean_shift_sp, mean_shift_sr
        )
    elif mode == SUPERPIXEL_MODE:
        result_bgr = superpixel_chroma_flatten_bgr(
            bgr, strength, edge_protect, n_segments, compactness
        )
    else:
        raise ValueError(f"Unknown Color Flatten mode: {mode}")

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    result = Image.fromarray(result_rgb, "RGB")

    if alpha is not None:
        result.putalpha(alpha)

    return result
