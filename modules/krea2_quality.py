from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image


REFERENCE_LONG_EDGE = 1536
DEFAULT_ANALYSIS_LONG_EDGE = 1536


@dataclass(frozen=True)
class ChromaMuraParams:
    analysis_long_edge: int = DEFAULT_ANALYSIS_LONG_EDGE
    blur_sigma: float = 18.0
    edge_percentile: float = 96.0
    edge_dilate: int = 5
    alpha_threshold: int = 8
    ignore_near_white_bg: bool = False
    white_bg_luma_threshold: int = 245
    white_bg_chroma_threshold: float = 5.0
    smoothness_threshold: float = 0.35


@dataclass(frozen=True)
class ChromaMuraMetrics:
    valid_area_pct: float
    mean_chroma_delta: float
    median_chroma_delta: float
    p90_chroma_delta: float
    p95_chroma_delta: float
    p99_chroma_delta: float
    max_chroma_delta: float
    area_chroma_delta_gt_2_pct: float
    area_chroma_delta_gt_5_pct: float
    area_chroma_delta_gt_10_pct: float
    rough_judgement: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _AnalysisData:
    rgba: np.ndarray
    lab: np.ndarray
    reference_ab: np.ndarray
    delta_chroma: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    metrics: ChromaMuraMetrics


def _validate_positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than 0")
    return value


def _validate_unit_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return value


def rgb_to_lab_float(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("rgb_to_lab_float expects an RGB uint8 image")

    lab8 = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab = np.empty_like(lab8, dtype=np.float32)
    lab[..., 0] = lab8[..., 0] * (100.0 / 255.0)
    lab[..., 1] = lab8[..., 1] - 128.0
    lab[..., 2] = lab8[..., 2] - 128.0
    return lab


def _lab_float_to_rgb(lab: np.ndarray) -> np.ndarray:
    lab8 = np.empty_like(lab, dtype=np.uint8)
    lab8[..., 0] = np.clip(np.rint(lab[..., 0] * (255.0 / 100.0)), 0, 255).astype(
        np.uint8
    )
    lab8[..., 1] = np.clip(np.rint(lab[..., 1] + 128.0), 0, 255).astype(np.uint8)
    lab8[..., 2] = np.clip(np.rint(lab[..., 2] + 128.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab8, cv2.COLOR_LAB2RGB)


def _analysis_rgba(image: Image.Image, analysis_long_edge: int) -> np.ndarray:
    analysis_long_edge = _validate_positive_int(
        analysis_long_edge, "analysis long edge"
    )
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    height, width = rgba.shape[:2]
    long_edge = max(width, height)
    if long_edge <= analysis_long_edge:
        return rgba.copy()

    scale = analysis_long_edge / long_edge
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    # Resize premultiplied color and alpha separately. Straight-alpha resizing lets
    # arbitrary hidden RGB from transparent pixels bleed into visible boundaries.
    alpha = rgba[..., 3].astype(np.float32) / 255.0
    premultiplied = rgba[..., :3].astype(np.float32) * alpha[..., None]
    resized_alpha = cv2.resize(
        alpha, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    resized_premultiplied = cv2.resize(
        premultiplied, (target_width, target_height), interpolation=cv2.INTER_AREA
    )
    resized_rgb = np.zeros_like(resized_premultiplied, dtype=np.float32)
    np.divide(
        resized_premultiplied,
        resized_alpha[..., None],
        out=resized_rgb,
        where=resized_alpha[..., None] > 1e-6,
    )
    return np.dstack(
        (
            np.clip(np.rint(resized_rgb), 0, 255).astype(np.uint8),
            np.clip(np.rint(resized_alpha * 255.0), 0, 255).astype(np.uint8),
        )
    )


def _normalized_gaussian_blur(
    values: np.ndarray, valid: np.ndarray, sigma: float
) -> np.ndarray:
    """Blur only valid samples so transparent hidden RGB cannot bias a reference."""
    weights = valid.astype(np.float32)
    numerator = cv2.GaussianBlur(
        values * weights[..., None] if values.ndim == 3 else values * weights,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    denominator = cv2.GaussianBlur(weights, (0, 0), sigmaX=sigma, sigmaY=sigma)
    if values.ndim == 3:
        denominator = denominator[..., None]
    fallback = values.astype(np.float32, copy=False)
    return np.divide(
        numerator,
        denominator,
        out=fallback.copy(),
        where=denominator > 1e-5,
    )


def _gradient_magnitude(channel: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy) / 8.0


def _scaled_sigma(params: ChromaMuraParams, analysis_shape: tuple[int, int]) -> float:
    analysis_long_edge = max(analysis_shape)
    return max(1.0, float(params.blur_sigma) * analysis_long_edge / REFERENCE_LONG_EDGE)


def _smoothstep(low: float, high: float, values: np.ndarray | float):
    if high <= low:
        raise ValueError("smoothstep high must be greater than low")
    values_array = np.asarray(values, dtype=np.float32)
    t = np.clip((values_array - low) / (high - low), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _metrics(delta_chroma: np.ndarray, valid: np.ndarray) -> ChromaMuraMetrics:
    if np.any(valid):
        values = delta_chroma[valid]
        valid_area_pct = float(np.mean(valid) * 100.0)
        mean_delta = float(np.mean(values))
        median_delta = float(np.median(values))
        p90 = float(np.percentile(values, 90))
        p95 = float(np.percentile(values, 95))
        p99 = float(np.percentile(values, 99))
        max_delta = float(np.max(values))
        gt2 = float(np.mean(values > 2.0) * 100.0)
        gt5 = float(np.mean(values > 5.0) * 100.0)
        gt10 = float(np.mean(values > 10.0) * 100.0)
    else:
        valid_area_pct = 0.0
        mean_delta = median_delta = p90 = p95 = p99 = max_delta = 0.0
        gt2 = gt5 = gt10 = 0.0

    if p95 <= 2.5 and gt5 <= 0.5:
        judgement = "OK: chroma mura is minor"
    elif p95 <= 5.0 and gt5 <= 3.0:
        judgement = "CHECK: light chroma mura detected"
    else:
        judgement = "NG: visible chroma mura candidate detected"

    return ChromaMuraMetrics(
        valid_area_pct=valid_area_pct,
        mean_chroma_delta=mean_delta,
        median_chroma_delta=median_delta,
        p90_chroma_delta=p90,
        p95_chroma_delta=p95,
        p99_chroma_delta=p99,
        max_chroma_delta=max_delta,
        area_chroma_delta_gt_2_pct=gt2,
        area_chroma_delta_gt_5_pct=gt5,
        area_chroma_delta_gt_10_pct=gt10,
        rough_judgement=judgement,
    )


def _build_analysis(image: Image.Image, params: ChromaMuraParams) -> _AnalysisData:
    rgba = _analysis_rgba(image, params.analysis_long_edge)
    rgb = np.ascontiguousarray(rgba[..., :3])
    lab = rgb_to_lab_float(rgb)
    l_channel = lab[..., 0]
    ab = lab[..., 1:3]

    alpha_valid = rgba[..., 3] > int(np.clip(params.alpha_threshold, 0, 255))
    sigma = _scaled_sigma(params, l_channel.shape)
    reference_ab = _normalized_gaussian_blur(ab, alpha_valid, sigma)
    delta_chroma = np.linalg.norm(ab - reference_ab, axis=2).astype(np.float32)

    detail_sigma = max(0.6, sigma / 8.0)
    l_smooth = _normalized_gaussian_blur(l_channel, alpha_valid, detail_sigma)
    ab_smooth = _normalized_gaussian_blur(ab, alpha_valid, detail_sigma)
    l_gradient = _gradient_magnitude(l_smooth)
    a_gradient = _gradient_magnitude(ab_smooth[..., 0])
    b_gradient = _gradient_magnitude(ab_smooth[..., 1])
    chroma_gradient = np.hypot(a_gradient, b_gradient)
    l_detail = np.abs(l_channel - l_smooth)

    structure = (
        (l_gradient / 1.5) ** 2 + (chroma_gradient / 2.0) ** 2 + (l_detail / 1.5) ** 2
    )
    confidence = (1.0 / (1.0 + structure)).astype(np.float32)

    edge_source = np.maximum(l_gradient, chroma_gradient)
    edge_percentile = float(np.clip(params.edge_percentile, 0.0, 100.0))
    if edge_percentile < 100.0:
        percentile_source = (
            edge_source[alpha_valid] if np.any(alpha_valid) else edge_source
        )
        edge_threshold = max(
            float(np.percentile(percentile_source, edge_percentile)), 1.0
        )
        edges = edge_source > edge_threshold
    else:
        edges = np.zeros_like(alpha_valid, dtype=bool)

    dilate = max(
        0,
        int(
            round(
                float(params.edge_dilate) * max(l_channel.shape) / REFERENCE_LONG_EDGE
            )
        ),
    )
    if dilate > 0:
        kernel_size = 2 * dilate + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        edges = cv2.dilate(edges.astype(np.uint8), kernel, iterations=1).astype(bool)

    # Treat the visible/transparent boundary as protected structure. Even though
    # the normalized blur ignores hidden RGB, changing edge colors can produce a
    # halo once the image is composited over another background.
    alpha_boundary = cv2.morphologyEx(
        alpha_valid.astype(np.uint8),
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    if dilate > 0:
        alpha_boundary = cv2.dilate(
            alpha_boundary.astype(np.uint8), kernel, iterations=1
        ).astype(bool)
    edges |= alpha_boundary

    # A Gaussian color reference can otherwise mix two perfectly flat colors on
    # opposite sides of a hard boundary and create a colored halo just outside
    # the ordinary edge mask. Keep correction at least 2.5 sigma away from
    # strong luminance/chroma transitions. Distance transform is linear in image
    # area and cheaper than a very large morphology kernel at the 1536px cap.
    l_curvature = np.abs(cv2.Laplacian(l_smooth, cv2.CV_32F, ksize=3)) / 8.0
    a_curvature = cv2.Laplacian(ab_smooth[..., 0], cv2.CV_32F, ksize=3) / 8.0
    b_curvature = cv2.Laplacian(ab_smooth[..., 1], cv2.CV_32F, ksize=3) / 8.0
    chroma_curvature = np.hypot(a_curvature, b_curvature)
    hard_boundaries = (l_curvature > 2.0) | (chroma_curvature > 3.5) | alpha_boundary
    if np.any(hard_boundaries):
        distance_to_boundary = cv2.distanceTransform(
            (~hard_boundaries).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
        )
        boundary_guard = max(1.0, 2.5 * sigma)
        edges |= distance_to_boundary <= boundary_guard

    valid = alpha_valid & (confidence >= float(params.smoothness_threshold)) & ~edges

    if params.ignore_near_white_bg:
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        chroma = np.linalg.norm(ab, axis=2)
        near_white = (gray >= float(params.white_bg_luma_threshold)) & (
            chroma <= float(params.white_bg_chroma_threshold)
        )
        valid &= ~near_white

    return _AnalysisData(
        rgba=rgba,
        lab=lab,
        reference_ab=reference_ab,
        delta_chroma=delta_chroma,
        confidence=confidence,
        valid=valid,
        metrics=_metrics(delta_chroma, valid),
    )


def analyze_chroma_mura(
    image: Image.Image,
    params: ChromaMuraParams | None = None,
    *,
    full_resolution: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ChromaMuraMetrics, tuple[int, int]]:
    if not isinstance(image, Image.Image):
        raise TypeError("analyze_chroma_mura expects a PIL.Image.Image input")

    params = params or ChromaMuraParams()
    data = _build_analysis(image, params)
    target_size = image.size
    analysis_size = (data.rgba.shape[1], data.rgba.shape[0])

    if not full_resolution or analysis_size == target_size:
        delta = data.delta_chroma.copy()
        confidence = data.confidence.copy()
        valid = data.valid.copy()
    else:
        delta = cv2.resize(
            data.delta_chroma, target_size, interpolation=cv2.INTER_LINEAR
        )
        confidence = cv2.resize(
            data.confidence, target_size, interpolation=cv2.INTER_LINEAR
        )
        valid = cv2.resize(
            data.valid.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)

    return delta, confidence, valid, data.metrics, analysis_size


def _mura_score(metrics: ChromaMuraMetrics) -> float:
    return (
        metrics.p95_chroma_delta
        + 0.08 * metrics.area_chroma_delta_gt_5_pct
        + 0.015 * metrics.p99_chroma_delta
    )


def adaptive_chroma_correct(
    image: Image.Image,
    *,
    strength: float = 0.8,
    params: ChromaMuraParams | None = None,
    max_chroma_shift: float = 6.0,
) -> tuple[Image.Image, dict[str, Any]]:
    if not isinstance(image, Image.Image):
        raise TypeError("adaptive_chroma_correct expects a PIL.Image.Image input")
    strength = _validate_unit_float(strength, "adaptive chroma strength")
    max_chroma_shift = float(max_chroma_shift)
    if not np.isfinite(max_chroma_shift) or max_chroma_shift <= 0.0:
        raise ValueError("max chroma shift must be greater than 0")

    params = params or ChromaMuraParams()
    data = _build_analysis(image, params)
    before = data.metrics
    severity = max(
        float(_smoothstep(1.5, 6.0, before.p95_chroma_delta)),
        0.7 * float(_smoothstep(0.5, 12.0, before.area_chroma_delta_gt_2_pct)),
    )
    effective_strength = strength * severity

    base_report: dict[str, Any] = {
        "applied": False,
        "accepted": False,
        "requested_strength": strength,
        "effective_strength": effective_strength,
        "analysis_size": [data.rgba.shape[1], data.rgba.shape[0]],
        "before": before.as_dict(),
        "after": before.as_dict(),
        "correction_area_pct": 0.0,
        "mean_chroma_shift": 0.0,
        "max_chroma_shift": 0.0,
        "reason": "mura below adaptive threshold",
    }
    if strength == 0.0 or effective_strength < 0.05 or not np.any(data.valid):
        return image.copy(), base_report

    signal = _smoothstep(1.0, 5.0, data.delta_chroma)
    blend = (
        effective_strength * data.confidence * signal * data.valid.astype(np.float32)
    )
    blend = cv2.GaussianBlur(blend, (0, 0), sigmaX=0.8, sigmaY=0.8)
    # Gaussian feathering must not re-introduce corrections inside a hard
    # alpha/edge/detail exclusion.
    blend *= data.valid.astype(np.float32)
    correction = (data.reference_ab - data.lab[..., 1:3]) * blend[..., None]
    correction_norm = np.linalg.norm(correction, axis=2)
    limiter = np.minimum(1.0, max_chroma_shift / np.maximum(correction_norm, 1e-6))
    correction *= limiter[..., None]

    analysis_size = (data.rgba.shape[1], data.rgba.shape[0])
    if analysis_size != image.size:
        correction = cv2.resize(correction, image.size, interpolation=cv2.INTER_LINEAR)
        full_valid = cv2.resize(
            data.valid.astype(np.uint8), image.size, interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        correction *= full_valid[..., None]

    source_rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    candidate_rgba = source_rgba.copy()
    shift = np.linalg.norm(correction, axis=2)
    changed = (shift > 0.05) & (
        source_rgba[..., 3] > int(np.clip(params.alpha_threshold, 0, 255))
    )
    # Work in bounded tiles. This avoids a full-resolution float Lab copy (about
    # 100 MiB at UHD) and, crucially, writes only genuinely corrected pixels.
    tile_size = 512
    for y in range(0, image.height, tile_size):
        y2 = min(image.height, y + tile_size)
        for x in range(0, image.width, tile_size):
            x2 = min(image.width, x + tile_size)
            tile_changed = changed[y:y2, x:x2]
            if not np.any(tile_changed):
                continue
            source_rgb_tile = np.ascontiguousarray(source_rgba[y:y2, x:x2, :3])
            source_lab_tile = rgb_to_lab_float(source_rgb_tile)
            source_lab_tile[..., 1:3] += correction[y:y2, x:x2]
            candidate_rgb_tile = _lab_float_to_rgb(source_lab_tile)
            target_rgb_tile = candidate_rgba[y:y2, x:x2, :3]
            target_rgb_tile[tile_changed] = candidate_rgb_tile[tile_changed]
    candidate = Image.fromarray(candidate_rgba, mode="RGBA")
    if image.mode == "RGB":
        candidate = candidate.convert("RGB")

    candidate_analysis = _build_analysis(candidate, params)
    # Evaluate against the exact pre-correction validity mask. Recomputing the
    # mask could make a candidate look better merely by turning difficult pixels
    # into newly excluded edge/detail pixels.
    after = _metrics(candidate_analysis.delta_chroma, data.valid)
    before_score = _mura_score(before)
    after_score = _mura_score(after)
    accepted = after_score <= before_score * 0.995

    report = dict(base_report)
    report.update(
        {
            "applied": accepted,
            "accepted": accepted,
            "after": after.as_dict(),
            "correction_area_pct": float(np.mean(changed) * 100.0),
            "mean_chroma_shift": float(np.mean(shift[changed]))
            if np.any(changed)
            else 0.0,
            "max_chroma_shift": float(np.max(shift)) if shift.size else 0.0,
            "reason": "chroma metrics improved"
            if accepted
            else "candidate rejected because chroma metrics did not improve",
        }
    )
    if not accepted:
        report["after"] = before.as_dict()
        return image.copy(), report
    return candidate, report


def adaptive_detail_guard(
    image: Image.Image,
    *,
    strength: float = 0.55,
    radius: float = 1.0,
    detail_threshold: float = 1.0,
    max_detail_delta: float = 4.0,
    stripe_height: int = 512,
) -> tuple[Image.Image, dict[str, Any]]:
    """Increase only coherent source microdetail without inventing flat-region texture."""

    if not isinstance(image, Image.Image):
        raise TypeError("adaptive_detail_guard expects a PIL.Image.Image input")
    strength = _validate_unit_float(strength, "detail guard strength")
    radius = float(radius)
    detail_threshold = float(detail_threshold)
    max_detail_delta = float(max_detail_delta)
    stripe_height = int(stripe_height)
    for value, name in (
        (radius, "detail guard radius"),
        (detail_threshold, "detail guard threshold"),
        (max_detail_delta, "maximum detail delta"),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be greater than 0")
    if stripe_height <= 0:
        raise ValueError("detail guard stripe height must be greater than 0")

    source_rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    candidate_rgba = source_rgba.copy()
    visible_pixels = int(np.count_nonzero(source_rgba[..., 3] > 8))
    base_report: dict[str, Any] = {
        "applied": False,
        "accepted": True,
        "requested_strength": strength,
        "radius": radius,
        "detail_threshold": detail_threshold,
        "max_detail_delta": max_detail_delta,
        "stripe_height": stripe_height,
        "changed_pixels": 0,
        "changed_percent": 0.0,
        "flat_region_changed_pixels": 0,
        "clipped_channel_fraction": 0.0,
        "mean_abs_luma_delta": 0.0,
        "max_abs_luma_delta": 0.0,
        "weighted_detail_energy_before": 0.0,
        "weighted_detail_energy_after": 0.0,
        "detail_energy_ratio": 1.0,
        "reason": "detail guard disabled",
    }
    if strength == 0.0 or visible_pixels == 0:
        return image.copy(), base_report

    energy_sigma = max(1.0, radius * 1.4)
    tensor_sigma = max(1.0, radius * 1.2)
    halo = max(8, int(np.ceil(max(radius, energy_sigma, tensor_sigma) * 4.0)) + 2)
    changed_pixels = 0
    flat_region_changed_pixels = 0
    clipped_channels = 0
    visible_channels = visible_pixels * 3
    abs_delta_sum = 0.0
    max_abs_delta_seen = 0.0
    detail_before_sum = 0.0
    detail_after_sum = 0.0
    detail_weight_sum = 0.0

    for core_y0 in range(0, image.height, stripe_height):
        core_y1 = min(image.height, core_y0 + stripe_height)
        context_y0 = max(0, core_y0 - halo)
        context_y1 = min(image.height, core_y1 + halo)
        core_slice = slice(core_y0 - context_y0, core_y1 - context_y0)

        context_rgba = source_rgba[context_y0:context_y1]
        rgb = context_rgba[..., :3].astype(np.float32)
        luma = (
            rgb[..., 0] * 0.299
            + rgb[..., 1] * 0.587
            + rgb[..., 2] * 0.114
        )
        low = cv2.GaussianBlur(luma, (0, 0), sigmaX=radius, sigmaY=radius)
        high = luma - low
        energy = np.sqrt(
            np.maximum(
                cv2.GaussianBlur(
                    high * high,
                    (0, 0),
                    sigmaX=energy_sigma,
                    sigmaY=energy_sigma,
                ),
                0.0,
            )
        )

        gx = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3) / 8.0
        gy = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3) / 8.0
        gradient = np.hypot(gx, gy)
        jxx = cv2.GaussianBlur(
            gx * gx, (0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma
        )
        jyy = cv2.GaussianBlur(
            gy * gy, (0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma
        )
        jxy = cv2.GaussianBlur(
            gx * gy, (0, 0), sigmaX=tensor_sigma, sigmaY=tensor_sigma
        )
        coherence = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy * jxy) / (
            jxx + jyy + 1e-6
        )

        texture_gate = _smoothstep(
            detail_threshold,
            max(detail_threshold + 0.5, detail_threshold * 3.0),
            energy,
        )
        coherence_gate = _smoothstep(0.18, 0.65, coherence)
        strong_edge_guard = 1.0 - _smoothstep(10.0, 24.0, gradient)
        dark_guard = _smoothstep(4.0, 16.0, luma)
        highlight_guard = 1.0 - _smoothstep(239.0, 251.0, luma)
        alpha_valid = context_rgba[..., 3] > 8
        gate = (
            texture_gate
            * coherence_gate
            * strong_edge_guard
            * dark_guard
            * highlight_guard
            * alpha_valid.astype(np.float32)
        )
        delta = np.clip(strength * high * gate, -max_detail_delta, max_detail_delta)
        candidate_float = rgb + delta[..., None]
        clipped = (candidate_float < 0.0) | (candidate_float > 255.0)
        candidate_rgb = np.clip(np.rint(candidate_float), 0, 255).astype(np.uint8)
        changed = np.any(candidate_rgb != context_rgba[..., :3], axis=2) & alpha_valid

        core_changed = changed[core_slice]
        core_gate = gate[core_slice]
        core_high = high[core_slice]
        core_energy = energy[core_slice]
        core_alpha_valid = alpha_valid[core_slice]
        target = candidate_rgba[core_y0:core_y1, :, :3]
        target[core_changed] = candidate_rgb[core_slice][core_changed]

        changed_count = int(np.count_nonzero(core_changed))
        changed_pixels += changed_count
        flat_region_changed_pixels += int(
            np.count_nonzero(
                core_changed & (core_energy < detail_threshold) & core_alpha_valid
            )
        )
        clipped_channels += int(np.count_nonzero(clipped[core_slice] & core_alpha_valid[..., None]))
        applied_delta = (
            candidate_rgb[core_slice].astype(np.float32)
            - context_rgba[core_slice, :, :3].astype(np.float32)
        )
        applied_luma_delta = (
            applied_delta[..., 0] * 0.299
            + applied_delta[..., 1] * 0.587
            + applied_delta[..., 2] * 0.114
        )
        abs_delta_sum += float(np.sum(np.abs(applied_luma_delta[core_changed])))
        if changed_count:
            max_abs_delta_seen = max(
                max_abs_delta_seen,
                float(np.max(np.abs(applied_luma_delta[core_changed]))),
            )

        candidate_luma = (
            candidate_rgb[..., 0].astype(np.float32) * 0.299
            + candidate_rgb[..., 1].astype(np.float32) * 0.587
            + candidate_rgb[..., 2].astype(np.float32) * 0.114
        )
        candidate_low = cv2.GaussianBlur(
            candidate_luma, (0, 0), sigmaX=radius, sigmaY=radius
        )
        candidate_high = candidate_luma - candidate_low
        detail_before_sum += float(np.sum(np.abs(core_high) * core_gate))
        detail_after_sum += float(
            np.sum(np.abs(candidate_high[core_slice]) * core_gate)
        )
        detail_weight_sum += float(np.sum(core_gate))

    if changed_pixels == 0 or detail_weight_sum <= 1e-6:
        report = dict(base_report)
        report["reason"] = "no coherent source microdetail exceeded the guard threshold"
        return image.copy(), report

    detail_before = detail_before_sum / detail_weight_sum
    detail_after = detail_after_sum / detail_weight_sum
    detail_ratio = detail_after / max(detail_before, 1e-6)
    clipped_fraction = clipped_channels / max(visible_channels, 1)
    accepted = (
        detail_ratio >= 1.002
        and flat_region_changed_pixels == 0
        and clipped_fraction <= 0.0005
    )
    report = dict(base_report)
    report.update(
        {
            "applied": accepted,
            "accepted": accepted,
            "changed_pixels": changed_pixels if accepted else 0,
            "changed_percent": changed_pixels * 100.0 / visible_pixels
            if accepted
            else 0.0,
            "flat_region_changed_pixels": flat_region_changed_pixels,
            "clipped_channel_fraction": clipped_fraction,
            "mean_abs_luma_delta": abs_delta_sum / changed_pixels
            if accepted
            else 0.0,
            "max_abs_luma_delta": max_abs_delta_seen if accepted else 0.0,
            "weighted_detail_energy_before": detail_before,
            "weighted_detail_energy_after": detail_after if accepted else detail_before,
            "detail_energy_ratio": detail_ratio if accepted else 1.0,
            "reason": "coherent source detail increased without flat-region edits"
            if accepted
            else "candidate rejected by detail-energy, flat-region, or clipping guard",
        }
    )
    if not accepted:
        return image.copy(), report

    candidate = Image.fromarray(candidate_rgba, mode="RGBA")
    if image.mode == "RGB":
        candidate = candidate.convert("RGB")
    return candidate, report


def _odd_kernel(value: float, minimum: int = 3, maximum: int = 9) -> int:
    value = int(round(value))
    value = min(maximum, max(minimum, value))
    if value % 2 == 0:
        value += 1 if value < maximum else -1
    return value


def _component_filter(
    mask: np.ndarray, min_area: int, max_area: int, max_span: int
) -> tuple[np.ndarray, dict[str, int]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    areas = stats[1:, cv2.CC_STAT_AREA]
    widths = stats[1:, cv2.CC_STAT_WIDTH]
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    area_ok = (areas >= min_area) & (areas <= max_area)
    span_ok = (widths <= max_span) & (heights <= max_span)
    selected = area_ok & span_ok
    keep_lookup = np.zeros(num_labels, dtype=bool)
    keep_lookup[1:] = selected
    filtered = np.where(keep_lookup[labels], 255, 0).astype(np.uint8)
    kept = int(np.count_nonzero(selected))
    rejected_area = int(np.count_nonzero(~area_ok))
    rejected_span = int(np.count_nonzero(area_ok & ~span_ok))

    return filtered, {
        "candidate_components": max(0, num_labels - 1),
        "kept_components": kept,
        "rejected_by_area": rejected_area,
        "rejected_by_span": rejected_span,
    }


def build_adaptive_speckle_mask(
    image: Image.Image,
    *,
    threshold: int = 0,
    polarity: str = "both",
    max_masked_percent: float = 0.35,
) -> tuple[Image.Image, dict[str, Any]]:
    if not isinstance(image, Image.Image):
        raise TypeError("build_adaptive_speckle_mask expects a PIL.Image.Image input")
    if threshold < 0 or threshold > 255:
        raise ValueError("speckle threshold must be between 0 and 255")
    if polarity not in {"bright", "dark", "both", "all"}:
        raise ValueError(f"unsupported speckle polarity: {polarity}")
    max_masked_percent = float(max_masked_percent)
    if not np.isfinite(max_masked_percent) or max_masked_percent <= 0.0:
        raise ValueError("max masked percent must be greater than 0")

    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = np.ascontiguousarray(rgba[..., :3])
    height, width = rgb.shape[:2]
    scale = max(1.0, max(width, height) / REFERENCE_LONG_EDGE)
    median_size = _odd_kernel(5.0 * scale)
    alpha_valid = rgba[..., 3] > 8
    median_radius = median_size // 2
    if median_radius > 0:
        interior_kernel = np.ones((median_size, median_size), dtype=np.uint8)
        alpha_interior = cv2.erode(
            alpha_valid.astype(np.uint8), interior_kernel, iterations=1
        ).astype(bool)
    else:
        alpha_interior = alpha_valid
    median = cv2.medianBlur(rgb, median_size)

    residual = np.max(np.abs(rgb.astype(np.int16) - median.astype(np.int16)), axis=2)
    rgb_float = rgb.astype(np.float32)
    median_float = median.astype(np.float32)
    source_luma = (
        rgb_float[..., 0] * 0.299
        + rgb_float[..., 1] * 0.587
        + rgb_float[..., 2] * 0.114
    )
    median_luma = (
        median_float[..., 0] * 0.299
        + median_float[..., 1] * 0.587
        + median_float[..., 2] * 0.114
    )
    luma_delta = source_luma - median_luma

    guide = cv2.GaussianBlur(median_luma, (0, 0), sigmaX=0.8, sigmaY=0.8)
    guide_gradient = _gradient_magnitude(guide)
    gradient_sample = (
        guide_gradient[alpha_interior]
        if np.any(alpha_interior)
        else np.zeros(1, dtype=np.float32)
    )
    smooth_threshold = max(float(np.percentile(gradient_sample, 80)), 4.0)
    smooth = guide_gradient <= smooth_threshold

    if threshold == 0:
        sample = residual[smooth & alpha_interior]
        if sample.size:
            median_residual = float(np.median(sample))
            mad = float(np.median(np.abs(sample - median_residual)))
            robust_threshold = median_residual + 8.0 * 1.4826 * mad
        else:
            robust_threshold = 0.0
        used_threshold = int(np.clip(round(max(18.0, robust_threshold)), 1, 255))
    else:
        used_threshold = int(threshold)

    candidate = (residual >= used_threshold) & alpha_interior
    luma_floor = max(1.0, used_threshold * 0.35)
    if polarity == "bright":
        candidate &= luma_delta >= luma_floor
    elif polarity == "dark":
        candidate &= luma_delta <= -luma_floor
    elif polarity == "both":
        candidate &= np.abs(luma_delta) >= luma_floor

    edge_threshold = max(float(np.percentile(gradient_sample, 90)), 6.0)
    edges = guide_gradient > edge_threshold
    edge_dilate = max(1, int(round(scale)))
    edge_kernel_size = 2 * edge_dilate + 1
    edge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (edge_kernel_size, edge_kernel_size)
    )
    edges = cv2.dilate(edges.astype(np.uint8), edge_kernel, iterations=1).astype(bool)
    candidate &= ~edges

    min_area = 1
    max_area = max(2, int(round(32 * scale * scale)))
    max_span = max(3, int(round(10 * scale)))
    filtered, component_stats = _component_filter(
        candidate.astype(np.uint8) * 255, min_area, max_area, max_span
    )

    repair_dilate = max(1, int(round(scale)))
    repair_kernel_size = 2 * repair_dilate + 1
    repair_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (repair_kernel_size, repair_kernel_size)
    )
    filtered = cv2.dilate(filtered, repair_kernel, iterations=1)
    filtered[~alpha_valid] = 0

    masked_pixels = int(np.count_nonzero(filtered))
    visible_pixels = int(np.count_nonzero(alpha_valid))
    masked_percent = masked_pixels * 100.0 / visible_pixels if visible_pixels else 0.0
    accepted = masked_percent <= max_masked_percent
    if not accepted:
        filtered.fill(0)
        masked_pixels = 0
        masked_percent = 0.0

    report: dict[str, Any] = {
        "accepted": accepted,
        "width": width,
        "height": height,
        "visible_pixels": visible_pixels,
        "threshold": used_threshold,
        "adaptive_threshold": threshold == 0,
        "median_size": median_size,
        "polarity": polarity,
        "max_area": max_area,
        "max_span": max_span,
        "masked_pixels": masked_pixels,
        "masked_percent": masked_percent,
        "max_masked_percent": max_masked_percent,
        "reason": "mask accepted"
        if accepted
        else "mask rejected by maximum changed-area guard",
    }
    report.update(component_stats)
    return Image.fromarray(filtered, mode="L"), report


def adaptive_despeckle(
    image: Image.Image,
    *,
    threshold: int = 0,
    polarity: str = "both",
    max_masked_percent: float = 0.35,
) -> tuple[Image.Image, dict[str, Any]]:
    mask, report = build_adaptive_speckle_mask(
        image,
        threshold=threshold,
        polarity=polarity,
        max_masked_percent=max_masked_percent,
    )
    if report["masked_pixels"] == 0:
        report = dict(report)
        report["applied"] = False
        return image.copy(), report

    source_rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    radius = max(1.0, min(4.0, max(image.size) / REFERENCE_LONG_EDGE * 2.0))
    repaired_rgb = cv2.inpaint(
        np.ascontiguousarray(source_rgba[..., :3]),
        np.asarray(mask, dtype=np.uint8),
        radius,
        cv2.INPAINT_TELEA,
    )
    repaired_rgba = source_rgba.copy()
    repaired_rgba[..., :3] = repaired_rgb
    result = Image.fromarray(repaired_rgba, mode="RGBA")
    if image.mode == "RGB":
        result = result.convert("RGB")

    report = dict(report)
    report["applied"] = True
    report["inpaint_radius"] = radius
    return result, report


def smart_finish_image(
    image: Image.Image,
    *,
    color_strength: float = 0.8,
    analysis_long_edge: int = DEFAULT_ANALYSIS_LONG_EDGE,
    despeckle: bool = False,
    speckle_threshold: int = 0,
    max_speckle_percent: float = 0.35,
    detail_guard: bool = False,
    detail_strength: float = 0.55,
    detail_radius: float = 1.0,
    detail_threshold: float = 1.0,
    max_detail_delta: float = 4.0,
) -> tuple[Image.Image, dict[str, Any]]:
    if not isinstance(image, Image.Image):
        raise TypeError("smart_finish_image expects a PIL.Image.Image input")

    working = image.copy()
    if despeckle:
        working, speckle_report = adaptive_despeckle(
            working,
            threshold=speckle_threshold,
            polarity="both",
            max_masked_percent=max_speckle_percent,
        )
    else:
        speckle_report = {
            "applied": False,
            "accepted": True,
            "masked_pixels": 0,
            "masked_percent": 0.0,
            "reason": "despeckle disabled",
        }

    params = ChromaMuraParams(analysis_long_edge=analysis_long_edge)
    working, chroma_report = adaptive_chroma_correct(
        working, strength=color_strength, params=params
    )
    if detail_guard:
        working, detail_report = adaptive_detail_guard(
            working,
            strength=detail_strength,
            radius=detail_radius,
            detail_threshold=detail_threshold,
            max_detail_delta=max_detail_delta,
        )
    else:
        detail_report = {
            "applied": False,
            "accepted": True,
            "changed_pixels": 0,
            "changed_percent": 0.0,
            "detail_energy_ratio": 1.0,
            "reason": "detail guard disabled",
        }
    report = {
        "version": 2,
        "input_size": [image.width, image.height],
        "output_size": [working.width, working.height],
        "speckle": speckle_report,
        "chroma_mura": chroma_report,
        "detail_guard": detail_report,
    }
    return working, report


def smart_finish_summary(report: dict[str, Any]) -> str:
    speckle = report["speckle"]
    chroma = report["chroma_mura"]
    before = chroma["before"]
    after = chroma["after"]
    detail = report.get("detail_guard", {})
    return (
        f"speckles={speckle.get('masked_pixels', 0)}, "
        f"chroma={'applied' if chroma.get('applied') else 'no-op'}, "
        f"p95={before['p95_chroma_delta']:.2f}->{after['p95_chroma_delta']:.2f}, "
        f"detail={'applied' if detail.get('applied') else 'no-op'}, "
        f"energy={detail.get('detail_energy_ratio', 1.0):.3f}x"
    )
