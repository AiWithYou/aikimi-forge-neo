"""Frequency-aware residual decomposition, confidence maps, and composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import cv2
import numpy as np

from .color import luminance, rgb_to_luma_chroma
from .config import FrequencyGains


EPSILON = 1e-6
BAND_NAMES = ("high_0", "high_1", "mid_high", "mid", "mid_low", "low")
SIGMAS = (1.0, 2.0, 4.0, 8.0, 16.0)


def gaussian_blur(array: np.ndarray, sigma: float) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    if sigma <= 0:
        return source.copy()
    result = cv2.GaussianBlur(
        source,
        (0, 0),
        sigmaX=float(sigma),
        sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )
    if source.ndim == 3 and result.ndim == 2:
        result = result[..., None]
    return np.asarray(result, dtype=np.float32)


class FrequencyDecomposer:
    def decompose(self, residual: np.ndarray) -> dict[str, np.ndarray]:
        residual = np.asarray(residual, dtype=np.float32)
        if not np.isfinite(residual).all():
            raise ValueError("Frequency residual contains NaN or Inf.")
        g1, g2, g4, g8, g16 = (
            gaussian_blur(residual, sigma) for sigma in SIGMAS
        )
        return {
            "high_0": residual - g1,
            "high_1": g1 - g2,
            "mid_high": g2 - g4,
            "mid": g4 - g8,
            "mid_low": g8 - g16,
            "low": g16,
        }

    @staticmethod
    def reconstruct(bands: Mapping[str, np.ndarray]) -> np.ndarray:
        missing = set(BAND_NAMES) - set(bands)
        if missing:
            raise ValueError(f"Missing frequency bands: {sorted(missing)}")
        return np.sum([bands[name] for name in BAND_NAMES], axis=0, dtype=np.float32)

    @staticmethod
    def energy(bands: Mapping[str, np.ndarray]) -> dict[str, float]:
        return {
            name: float(np.mean(np.square(np.asarray(bands[name], dtype=np.float32))))
            for name in BAND_NAMES
        }


def robust_soft_clip(
    values: np.ndarray,
    *,
    mad_multiplier: float = 6.0,
    percentile: float = 99.5,
    epsilon: float = 1e-7,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    if not np.isfinite(source).all():
        raise ValueError("Cannot soft-clip NaN or Inf.")
    median = float(np.median(source))
    mad = float(np.median(np.abs(source - median)))
    percentile_limit = float(np.percentile(np.abs(source), percentile))
    mad_limit = mad_multiplier * mad
    if percentile_limit <= epsilon:
        return np.zeros_like(source)
    limit = percentile_limit if mad_limit <= epsilon else min(
        mad_limit, percentile_limit
    )
    limit = max(limit, epsilon)
    return (limit * np.tanh(source / limit)).astype(np.float32)


def robust_normalize(
    values: np.ndarray, low_percentile: float = 50.0, high_percentile: float = 95.0
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    low, high = np.percentile(source, (low_percentile, high_percentile))
    return np.clip((source - low) / max(float(high - low), EPSILON), 0.0, 1.0)


@dataclass(frozen=True)
class StructureMaps:
    protection: np.ndarray
    edge_magnitude: np.ndarray
    orientation_x: np.ndarray
    orientation_y: np.ndarray
    coherence: np.ndarray
    flatness: np.ndarray
    texture: np.ndarray


class StructureMapBuilder:
    def __init__(self, dilation_pixels: int = 3, feather_sigma: float = 2.0):
        self.dilation_pixels = dilation_pixels
        self.feather_sigma = feather_sigma

    def build(
        self,
        anchor_linear_rgb: np.ndarray,
        manual_protection: np.ndarray | None = None,
    ) -> StructureMaps:
        y = luminance(anchor_linear_rgb)
        gradients: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for sigma in (1.0, 2.0, 4.0):
            smooth = gaussian_blur(y, sigma)
            gx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(gx, gy)
            gradients.append((gx, gy, magnitude))
        combined = np.maximum.reduce([item[2] for item in gradients])
        normalized = robust_normalize(combined, 55.0, 98.5)
        if self.dilation_pixels > 0:
            kernel_size = self.dilation_pixels * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            normalized = cv2.dilate(normalized, kernel)
        protection = np.clip(
            gaussian_blur(normalized, self.feather_sigma), 0.0, 1.0
        )
        if manual_protection is not None:
            manual = np.asarray(manual_protection, dtype=np.float32)
            if manual.shape != y.shape:
                raise ValueError("Manual protection map size does not match anchor.")
            protection = np.maximum(protection, np.clip(manual, 0.0, 1.0))

        gx, gy, magnitude = gradients[1]
        jxx = gaussian_blur(gx * gx, 2.0)
        jyy = gaussian_blur(gy * gy, 2.0)
        jxy = gaussian_blur(gx * gy, 2.0)
        trace = jxx + jyy
        discriminant = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4.0 * jxy**2))
        coherence = np.clip(discriminant / (trace + EPSILON), 0.0, 1.0)
        norm = magnitude + EPSILON
        orientation_x = gx / norm
        orientation_y = gy / norm
        local_variance = np.maximum(
            0.0, gaussian_blur(y * y, 2.0) - gaussian_blur(y, 2.0) ** 2
        )
        texture = robust_normalize(local_variance, 45.0, 97.0)
        flatness = 1.0 - texture
        return StructureMaps(
            protection=protection.astype(np.float32),
            edge_magnitude=combined.astype(np.float32),
            orientation_x=orientation_x.astype(np.float32),
            orientation_y=orientation_y.astype(np.float32),
            coherence=coherence.astype(np.float32),
            flatness=flatness.astype(np.float32),
            texture=texture.astype(np.float32),
        )


def orientation_confidence(
    anchor_linear_rgb: np.ndarray, candidate_linear_rgb: np.ndarray
) -> np.ndarray:
    anchor_y = luminance(anchor_linear_rgb)
    candidate_y = luminance(candidate_linear_rgb)
    ax = cv2.Sobel(anchor_y, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(anchor_y, cv2.CV_32F, 0, 1, ksize=3)
    cx = cv2.Sobel(candidate_y, cv2.CV_32F, 1, 0, ksize=3)
    cy = cv2.Sobel(candidate_y, cv2.CV_32F, 0, 1, ksize=3)
    an = np.sqrt(ax * ax + ay * ay)
    cn = np.sqrt(cx * cx + cy * cy)
    cosine = (ax * cx + ay * cy) / (an * cn + EPSILON)
    confidence = np.clip((cosine - 0.25) / 0.75, 0.0, 1.0) ** 2
    threshold_a = np.percentile(an, 60.0)
    threshold_c = np.percentile(cn, 60.0)
    flat = (an <= threshold_a) | (cn <= threshold_c)
    confidence[flat] = 1.0
    return confidence.astype(np.float32)


def new_edge_confidence(
    anchor_linear_rgb: np.ndarray,
    candidate_linear_rgb: np.ndarray,
    *,
    allowed_distance: float = 1.5,
    reject_distance: float = 3.0,
    minimum_component_pixels: int = 9,
) -> np.ndarray:
    anchor_y = luminance(anchor_linear_rgb)
    candidate_y = luminance(candidate_linear_rgb)

    def edge_strength(y: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    anchor_strength = edge_strength(anchor_y)
    candidate_strength = edge_strength(candidate_y)
    anchor_threshold = max(
        float(np.percentile(anchor_strength, 85.0)), EPSILON
    )
    candidate_threshold = max(
        float(np.percentile(candidate_strength, 88.0)), EPSILON
    )
    anchor_edges = anchor_strength >= anchor_threshold
    candidate_edges = candidate_strength >= candidate_threshold
    distance = cv2.distanceTransform(
        (~anchor_edges).astype(np.uint8), cv2.DIST_L2, 3
    )
    confidence = np.ones(anchor_y.shape, dtype=np.float32)
    strong_new = candidate_edges & (distance > allowed_distance)

    components, labels, stats, _ = cv2.connectedComponentsWithStats(
        strong_new.astype(np.uint8), connectivity=8
    )
    continuous = np.zeros_like(strong_new)
    for label in range(1, components):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_component_pixels:
            continuous |= labels == label
    ramp = 1.0 - np.clip(
        (distance - allowed_distance)
        / max(reject_distance - allowed_distance, EPSILON),
        0.0,
        1.0,
    )
    confidence[continuous] = ramp[continuous]
    return gaussian_blur(confidence, 0.75).astype(np.float32)


@dataclass(frozen=True)
class ComposeMaps:
    structure: np.ndarray
    orientation: np.ndarray
    new_edge: np.ndarray
    tile: np.ndarray
    roundtrip: np.ndarray | float = 1.0
    region: np.ndarray | float = 1.0
    manual_boost: np.ndarray | float = 0.0


PROTECTION_MULTIPLIERS = {
    "high_0": 0.35,
    "high_1": 0.45,
    "mid_high": 0.60,
    "mid": 0.85,
    "mid_low": 1.00,
    "low": 1.00,
}
ORIENTATION_WEIGHTS = {
    "high_0": 0.20,
    "high_1": 0.30,
    "mid_high": 0.60,
    "mid": 0.85,
    "mid_low": 1.00,
    "low": 1.00,
}
ROUNDTRIP_EXPONENTS = {
    "high_0": 0.50,
    "high_1": 0.70,
    "mid_high": 1.00,
    "mid": 1.50,
    "mid_low": 2.00,
    "low": 2.00,
}


class FrequencyAwareComposer:
    def __init__(self):
        self.decomposer = FrequencyDecomposer()

    def compose(
        self,
        anchor_linear_rgb: np.ndarray,
        candidate_linear_rgb: np.ndarray,
        *,
        gains: FrequencyGains | Mapping[str, float],
        maps: ComposeMaps,
        structural_lock: float,
        low_frequency_lock: float,
        boost_strength: float = 0.75,
    ) -> tuple[np.ndarray, dict[str, float]]:
        anchor = np.asarray(anchor_linear_rgb, dtype=np.float32)
        candidate = np.asarray(candidate_linear_rgb, dtype=np.float32)
        if anchor.shape != candidate.shape or anchor.ndim != 3:
            raise ValueError("Frequency composer inputs must be matching RGB arrays.")
        anchor_y, anchor_chroma = rgb_to_luma_chroma(anchor)
        candidate_y, candidate_chroma = rgb_to_luma_chroma(candidate)
        y_bands = self.decomposer.decompose(candidate_y - anchor_y)
        chroma_bands = self.decomposer.decompose(candidate_chroma - anchor_chroma)

        def gain(name: str) -> float:
            if isinstance(gains, FrequencyGains):
                return float(getattr(gains, name))
            return float(gains[name])

        chroma_ratio = (
            gains.chroma_ratio
            if isinstance(gains, FrequencyGains)
            else float(gains.get("chroma_ratio", 0.35))
        )

        def spatial_map(value: np.ndarray | float, name: str) -> np.ndarray:
            result = np.asarray(value, dtype=np.float32)
            if result.ndim == 0:
                result = np.full(anchor.shape[:2], float(result), dtype=np.float32)
            if result.shape != anchor.shape[:2]:
                raise ValueError(
                    f"{name} map shape {result.shape} does not match "
                    f"{anchor.shape[:2]}."
                )
            if not np.isfinite(result).all():
                raise ValueError(f"{name} map contains NaN or Inf.")
            return np.clip(result, 0.0, 1.0)

        structure = spatial_map(maps.structure, "Structure")
        orientation = spatial_map(maps.orientation, "Orientation")
        new_edge = spatial_map(maps.new_edge, "New-edge")
        tile = spatial_map(maps.tile, "Tile confidence")
        roundtrip = spatial_map(maps.roundtrip, "Round-trip confidence")
        region = spatial_map(maps.region, "Region")
        boost = 1.0 + boost_strength * spatial_map(
            maps.manual_boost, "Manual boost"
        )

        output_y = anchor_y.copy()
        output_chroma = anchor_chroma.copy()
        adopted_energy: dict[str, float] = {}
        for name in BAND_NAMES:
            protection_amount = PROTECTION_MULTIPLIERS[name] * structural_lock
            protection_gate = np.clip(1.0 - protection_amount * structure, 0.0, 1.0)
            orientation_weight = ORIENTATION_WEIGHTS[name]
            compatibility = (
                (1.0 - orientation_weight)
                + orientation_weight * orientation * new_edge
            )
            if name == "low":
                compatibility *= max(0.0, 1.0 - low_frequency_lock)
            roundtrip_gate = np.power(
                roundtrip, ROUNDTRIP_EXPONENTS[name], dtype=np.float32
            )
            scalar_map = (
                protection_gate
                * compatibility
                * tile
                * roundtrip_gate
                * region
                * boost
            )
            clipped_y = robust_soft_clip(y_bands[name])
            clipped_chroma = robust_soft_clip(chroma_bands[name])
            y_contribution = clipped_y * scalar_map * gain(name)
            chroma_contribution = (
                clipped_chroma
                * scalar_map[..., None]
                * gain(name)
                * chroma_ratio
            )
            output_y += y_contribution
            output_chroma += chroma_contribution
            adopted_energy[name] = float(np.mean(np.square(y_contribution)))

        composed = output_y[..., None] + output_chroma
        if not np.isfinite(composed).all():
            raise RuntimeError("Frequency composition produced NaN or Inf.")
        return np.clip(composed, 0.0, 1.0).astype(np.float32), adopted_energy
