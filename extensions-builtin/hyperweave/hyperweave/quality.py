"""Round-trip, back-projection, and seam quality controls."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from .color import luminance
from .frequency import EPSILON, gaussian_blur


def resize_float(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    result = cv2.resize(
        np.asarray(array, dtype=np.float32),
        size,
        interpolation=cv2.INTER_AREA
        if size[0] < array.shape[1] or size[1] < array.shape[0]
        else cv2.INTER_LANCZOS4,
    )
    if array.ndim == 3 and result.ndim == 2:
        result = result[..., None]
    return np.asarray(result, dtype=np.float32)


@dataclass(frozen=True)
class RoundTripMetrics:
    ssim: float
    psnr: float
    low_frequency_mse: float
    color_drift: float
    edge_displacement: float
    confidence: float
    edge_precision: float = 1.0
    edge_recall: float = 1.0
    edge_f1: float = 1.0
    edge_displacement_forward: float = 0.0
    edge_displacement_reverse: float = 0.0
    evaluation_fraction: float = 1.0


@dataclass(frozen=True)
class EdgeMetrics:
    displacement: float
    precision: float
    recall: float
    f1: float
    displacement_forward: float
    displacement_reverse: float


def _evaluation_weights(
    evaluation_mask: np.ndarray | None,
    size: tuple[int, int],
) -> tuple[np.ndarray, float]:
    width, height = size
    if evaluation_mask is None:
        return np.ones((height, width), dtype=np.float32), 1.0
    mask = np.asarray(evaluation_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = np.mean(mask, axis=2, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("Evaluation mask must be a 2D array.")
    if not np.isfinite(mask).all():
        raise ValueError("Evaluation mask contains NaN or Inf.")
    if mask.shape != (height, width):
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_AREA)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    fraction = float(np.mean(mask))
    if float(np.sum(mask, dtype=np.float64)) <= EPSILON:
        # An empty user/component mask must remain finite and safe.  The zero
        # fraction records that the explicit mask could not be used.
        return np.ones((height, width), dtype=np.float32), 0.0
    return mask, fraction


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    source = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if source.ndim == 3:
        numerator = float(np.sum(source * weight[..., None], dtype=np.float64))
        denominator = float(np.sum(weight, dtype=np.float64)) * source.shape[2]
    else:
        numerator = float(np.sum(source * weight, dtype=np.float64))
        denominator = float(np.sum(weight, dtype=np.float64))
    return numerator / max(denominator, float(EPSILON))


def _weighted_ssim(
    reference: np.ndarray,
    candidate: np.ndarray,
    weights: np.ndarray,
) -> float:
    minimum = min(reference.shape[:2])
    if minimum < 3:
        mse = _weighted_mean(np.square(reference - candidate), weights)
        return float(np.clip(1.0 - 4.0 * mse, -1.0, 1.0))
    window = min(7, minimum if minimum % 2 else minimum - 1)
    if bool(np.all(weights >= 1.0 - 1e-7)):
        value = structural_similarity(
            reference,
            candidate,
            channel_axis=2,
            data_range=1.0,
            gaussian_weights=True,
            win_size=window,
        )
        return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else -1.0
    _, score_map = structural_similarity(
        reference,
        candidate,
        channel_axis=2,
        data_range=1.0,
        gaussian_weights=True,
        win_size=window,
        full=True,
    )
    if score_map.ndim == 3:
        score_map = np.mean(score_map, axis=2, dtype=np.float32)
    value = _weighted_mean(score_map, weights)
    return float(np.clip(value, -1.0, 1.0)) if np.isfinite(value) else -1.0


def _weighted_edge_mean(
    values: np.ndarray,
    edges: np.ndarray,
    weights: np.ndarray,
) -> float:
    selected_weights = weights[edges].astype(np.float64)
    if not selected_weights.size:
        return 0.0
    return float(
        np.sum(values[edges].astype(np.float64) * selected_weights)
        / max(float(np.sum(selected_weights)), float(EPSILON))
    )


def symmetric_edge_metrics(
    reference_y: np.ndarray,
    candidate_y: np.ndarray,
    *,
    evaluation_mask: np.ndarray | None = None,
    allowed_distance: float = 1.75,
    displacement_cap: float = 8.0,
) -> EdgeMetrics:
    """Measure missing and added edges using one reference-derived threshold."""

    reference = np.asarray(reference_y, dtype=np.float32)
    candidate = np.asarray(candidate_y, dtype=np.float32)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Edge metric inputs must be matching 2D arrays.")
    weights, _ = _evaluation_weights(
        evaluation_mask, (reference.shape[1], reference.shape[0])
    )
    support = weights > 1e-3

    def magnitude(y: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    reference_magnitude = magnitude(reference)
    candidate_magnitude = magnitude(candidate)
    samples = reference_magnitude[support]
    if samples.size:
        p85 = float(np.percentile(samples, 85.0))
        p99 = float(np.percentile(samples, 99.0))
        threshold = max(p85, 0.10 * p99, EPSILON)
    else:
        threshold = EPSILON
    reference_edges = support & (reference_magnitude >= threshold)
    candidate_edges = support & (candidate_magnitude >= threshold)
    has_reference = bool(np.any(reference_edges))
    has_candidate = bool(np.any(candidate_edges))

    if not has_reference and not has_candidate:
        return EdgeMetrics(0.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    if not has_reference:
        return EdgeMetrics(0.5, 0.0, 1.0, 0.0, 1.0, 0.0)
    if not has_candidate:
        return EdgeMetrics(0.5, 1.0, 0.0, 0.0, 0.0, 1.0)

    distance_to_reference = cv2.distanceTransform(
        (~reference_edges).astype(np.uint8), cv2.DIST_L2, 5
    )
    distance_to_candidate = cv2.distanceTransform(
        (~candidate_edges).astype(np.uint8), cv2.DIST_L2, 5
    )
    forward_pixels = np.minimum(distance_to_reference, displacement_cap)
    reverse_pixels = np.minimum(distance_to_candidate, displacement_cap)
    forward = _weighted_edge_mean(
        forward_pixels, candidate_edges, weights
    ) / max(displacement_cap, EPSILON)
    reverse = _weighted_edge_mean(
        reverse_pixels, reference_edges, weights
    ) / max(displacement_cap, EPSILON)

    precision_hits = (distance_to_reference <= allowed_distance).astype(np.float32)
    recall_hits = (distance_to_candidate <= allowed_distance).astype(np.float32)
    precision = _weighted_edge_mean(precision_hits, candidate_edges, weights)
    recall = _weighted_edge_mean(recall_hits, reference_edges, weights)
    f1 = (
        0.0
        if precision + recall <= EPSILON
        else 2.0 * precision * recall / (precision + recall)
    )
    return EdgeMetrics(
        displacement=float(np.clip(0.5 * (forward + reverse), 0.0, 1.0)),
        precision=float(np.clip(precision, 0.0, 1.0)),
        recall=float(np.clip(recall, 0.0, 1.0)),
        f1=float(np.clip(f1, 0.0, 1.0)),
        displacement_forward=float(np.clip(forward, 0.0, 1.0)),
        displacement_reverse=float(np.clip(reverse, 0.0, 1.0)),
    )


def roundtrip_metrics(
    reference_linear_rgb: np.ndarray,
    candidate_linear_rgb: np.ndarray,
    *,
    reference_size: tuple[int, int] | None = None,
    evaluation_mask: np.ndarray | None = None,
) -> RoundTripMetrics:
    reference = np.asarray(reference_linear_rgb, dtype=np.float32)
    candidate = np.asarray(candidate_linear_rgb, dtype=np.float32)
    if reference_size is None:
        reference_size = (reference.shape[1], reference.shape[0])
    if (candidate.shape[1], candidate.shape[0]) != reference_size:
        candidate = resize_float(candidate, reference_size)
    if (reference.shape[1], reference.shape[0]) != reference_size:
        reference = resize_float(reference, reference_size)
    reference = np.clip(reference, 0.0, 1.0)
    candidate = np.clip(candidate, 0.0, 1.0)
    weights, evaluation_fraction = _evaluation_weights(
        evaluation_mask, reference_size
    )
    score = _weighted_ssim(reference, candidate, weights)
    mse = _weighted_mean(np.square(reference - candidate), weights)
    psnr = float(99.0 if mse <= 1e-12 else 10.0 * np.log10(1.0 / mse))
    reference_low = gaussian_blur(reference, 2.0)
    candidate_low = gaussian_blur(candidate, 2.0)
    low_mse = _weighted_mean(
        np.square(reference_low - candidate_low), weights
    )
    weight_sum = max(float(np.sum(weights, dtype=np.float64)), float(EPSILON))
    reference_color = np.sum(
        reference.astype(np.float64) * weights[..., None], axis=(0, 1)
    ) / weight_sum
    candidate_color = np.sum(
        candidate.astype(np.float64) * weights[..., None], axis=(0, 1)
    ) / weight_sum
    color_drift = float(np.mean(np.abs(reference_color - candidate_color)))
    edge = symmetric_edge_metrics(
        luminance(reference),
        luminance(candidate),
        evaluation_mask=weights,
    )
    confidence = float(
        np.clip(
            max(score, 0.0)
            * np.exp(-12.0 * low_mse)
            * np.exp(-6.0 * color_drift)
            * np.exp(-2.0 * edge.displacement)
            * (0.35 + 0.65 * edge.f1),
            0.0,
            1.0,
        )
    )
    return RoundTripMetrics(
        ssim=score,
        psnr=psnr,
        low_frequency_mse=low_mse,
        color_drift=color_drift,
        edge_displacement=edge.displacement,
        confidence=confidence,
        edge_precision=edge.precision,
        edge_recall=edge.recall,
        edge_f1=edge.f1,
        edge_displacement_forward=edge.displacement_forward,
        edge_displacement_reverse=edge.displacement_reverse,
        evaluation_fraction=evaluation_fraction,
    )


@dataclass(frozen=True)
class BackProjectionReport:
    initial_error: float
    final_error: float
    iterations_completed: int
    rolled_back: bool


class LowFrequencyBackProjection:
    def apply(
        self,
        output_linear_rgb: np.ndarray,
        previous_linear_rgb: np.ndarray,
        *,
        iterations: int = 2,
        beta: float = 0.70,
        sigma: float = 2.0,
    ) -> tuple[np.ndarray, BackProjectionReport]:
        output = np.asarray(output_linear_rgb, dtype=np.float32).copy()
        previous = np.asarray(previous_linear_rgb, dtype=np.float32)
        previous_size = (previous.shape[1], previous.shape[0])

        def error_for(candidate: np.ndarray) -> float:
            down = resize_float(candidate, previous_size)
            return float(
                np.mean(
                    np.square(
                        gaussian_blur(previous - down, sigma)
                    )
                )
            )

        initial = error_for(output)
        current_error = initial
        completed = 0
        rolled_back = False
        current_beta = float(beta)
        for _ in range(max(0, int(iterations))):
            down = resize_float(output, previous_size)
            low_error = gaussian_blur(previous - down, sigma)
            correction = resize_float(
                low_error, (output.shape[1], output.shape[0])
            )
            proposal = np.clip(output + current_beta * correction, 0.0, 1.0)
            proposal_error = error_for(proposal)
            if proposal_error > current_error + 1e-12:
                retry_beta = current_beta * 0.5
                proposal = np.clip(output + retry_beta * correction, 0.0, 1.0)
                proposal_error = error_for(proposal)
                if proposal_error > current_error + 1e-12:
                    rolled_back = True
                    break
                current_beta = retry_beta
            output = proposal
            current_error = proposal_error
            completed += 1
        return output, BackProjectionReport(
            initial_error=initial,
            final_error=current_error,
            iterations_completed=completed,
            rolled_back=rolled_back,
        )


@dataclass(frozen=True)
class SeamReport:
    ratio: float
    boundary_energy: float
    reference_energy: float
    boundary_count: int


class SeamAnalyzer:
    def analyze(
        self,
        linear_rgb: np.ndarray,
        vertical_boundaries: list[int],
        horizontal_boundaries: list[int],
        *,
        band: int = 2,
    ) -> SeamReport:
        y = luminance(linear_rgb)
        gx = np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3))
        boundary_samples: list[np.ndarray] = []
        reference_samples: list[np.ndarray] = []
        height, width = y.shape
        for x in sorted(set(vertical_boundaries)):
            if band <= x < width - band:
                boundary_samples.append(gx[:, x - band : x + band + 1])
                reference_x = max(band, min(width - band - 1, x + width // 7))
                reference_samples.append(
                    gx[:, reference_x - band : reference_x + band + 1]
                )
        for y_pos in sorted(set(horizontal_boundaries)):
            if band <= y_pos < height - band:
                boundary_samples.append(
                    gy[y_pos - band : y_pos + band + 1, :]
                )
                reference_y = max(
                    band, min(height - band - 1, y_pos + height // 7)
                )
                reference_samples.append(
                    gy[reference_y - band : reference_y + band + 1, :]
                )
        if not boundary_samples:
            return SeamReport(0.0, 0.0, 0.0, 0)
        boundary_energy = float(
            np.mean([float(np.mean(sample)) for sample in boundary_samples])
        )
        reference_energy = float(
            np.mean([float(np.mean(sample)) for sample in reference_samples])
        )
        ratio = boundary_energy / max(reference_energy, EPSILON)
        return SeamReport(
            ratio=ratio,
            boundary_energy=boundary_energy,
            reference_energy=reference_energy,
            boundary_count=len(boundary_samples),
        )

    def harmonize(
        self,
        linear_rgb: np.ndarray,
        vertical_boundaries: list[int],
        horizontal_boundaries: list[int],
        *,
        threshold: float = 1.65,
        radius: int = 8,
    ) -> tuple[np.ndarray, SeamReport]:
        report = self.analyze(
            linear_rgb, vertical_boundaries, horizontal_boundaries
        )
        if report.ratio <= threshold:
            return np.asarray(linear_rgb, dtype=np.float32), report
        source = np.asarray(linear_rgb, dtype=np.float32)
        smooth = gaussian_blur(source, 1.0)
        mask = np.zeros(source.shape[:2], dtype=np.float32)
        for x in vertical_boundaries:
            mask[:, max(0, x - radius) : min(source.shape[1], x + radius + 1)] = 1
        for y in horizontal_boundaries:
            mask[max(0, y - radius) : min(source.shape[0], y + radius + 1), :] = 1
        mask = gaussian_blur(mask, max(1.0, radius / 3.0))[..., None] * 0.35
        result = source * (1.0 - mask) + smooth * mask
        return np.clip(result, 0.0, 1.0), self.analyze(
            result, vertical_boundaries, horizontal_boundaries
        )
