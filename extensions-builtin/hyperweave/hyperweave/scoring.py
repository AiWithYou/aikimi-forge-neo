"""Hard constraints and coherent-detail candidate ranking."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .color import luminance
from .frequency import (
    FrequencyDecomposer,
    gaussian_blur,
    new_edge_confidence,
)
from .quality import RoundTripMetrics, roundtrip_metrics
from .scoring_features import spectral_flatness


@dataclass
class CandidateScore:
    candidate_index: int
    accepted: bool
    total: float
    coherent_detail_gain: float
    mid_frequency_gain: float
    line_continuity: float
    material_richness: float
    style_consistency: float
    orientation_alignment: float
    noise_penalty: float
    duplicate_edge_penalty: float
    color_drift: float
    boundary_error: float
    structure_error: float
    clipping_ratio: float
    roundtrip: RoundTripMetrics
    rejection_reasons: list[str] = field(default_factory=list)
    raw_mid_frequency_energy: float = 0.0
    raw_high_frequency_energy: float = 0.0
    normalized_detail_score: float = 0.0
    evaluation_fraction: float = 1.0
    edge_precision: float = 1.0
    edge_recall: float = 1.0
    edge_f1: float = 1.0

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["roundtrip"] = asdict(self.roundtrip)
        return result


def _mask_for_shape(
    evaluation_mask: np.ndarray | None, shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    if evaluation_mask is None:
        return np.ones(shape, dtype=np.float32), 1.0
    mask = np.asarray(evaluation_mask, dtype=np.float32)
    if mask.ndim == 3:
        mask = np.mean(mask, axis=2, dtype=np.float32)
    if mask.ndim != 2:
        raise ValueError("Evaluation mask must be 2D.")
    if not np.isfinite(mask).all():
        raise ValueError("Evaluation mask contains NaN or Inf.")
    if mask.shape != shape:
        mask = cv2.resize(
            mask, (shape[1], shape[0]), interpolation=cv2.INTER_AREA
        )
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    fraction = float(np.mean(mask))
    if float(np.sum(mask, dtype=np.float64)) <= 1e-6:
        return np.ones(shape, dtype=np.float32), 0.0
    return mask, fraction


def _weighted_mean(values: np.ndarray, mask: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    weights = np.asarray(mask, dtype=np.float64)
    return float(
        np.sum(array * weights, dtype=np.float64)
        / max(float(np.sum(weights, dtype=np.float64)), 1e-12)
    )


def _gradient_coherence(y: np.ndarray, mask: np.ndarray) -> float:
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    jxx = gaussian_blur(gx * gx, 1.5)
    jyy = gaussian_blur(gy * gy, 1.5)
    jxy = gaussian_blur(gx * gy, 1.5)
    trace = jxx + jyy
    discriminant = np.sqrt(np.maximum(0.0, (jxx - jyy) ** 2 + 4 * jxy**2))
    return _weighted_mean(discriminant / (trace + 1e-6), mask)


def _orientation_alignment(
    anchor: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    *,
    neutral: float = 0.5,
) -> float:
    anchor_y = luminance(anchor)
    candidate_y = luminance(candidate)
    ax = cv2.Sobel(anchor_y, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(anchor_y, cv2.CV_32F, 0, 1, ksize=3)
    cx = cv2.Sobel(candidate_y, cv2.CV_32F, 1, 0, ksize=3)
    cy = cv2.Sobel(candidate_y, cv2.CV_32F, 0, 1, ksize=3)
    anchor_magnitude = cv2.magnitude(ax, ay)
    candidate_magnitude = cv2.magnitude(cx, cy)
    combined = np.maximum(anchor_magnitude, candidate_magnitude)
    samples = combined[mask > 1e-3]
    if not samples.size:
        return neutral
    threshold = max(
        float(np.percentile(samples, 60.0)),
        0.05 * float(np.percentile(samples, 95.0)),
        1e-6,
    )
    active = (mask > 1e-3) & (
        (anchor_magnitude >= threshold) | (candidate_magnitude >= threshold)
    )
    if not np.any(active):
        return neutral
    cosine = np.abs(ax * cx + ay * cy) / (
        anchor_magnitude * candidate_magnitude + 1e-6
    )
    weights = mask * combined
    return float(
        np.clip(
            np.sum(np.clip(cosine[active], 0.0, 1.0) * weights[active])
            / max(float(np.sum(weights[active])), 1e-6),
            0.0,
            1.0,
        )
    )


def _duplicate_edge_penalty(
    anchor: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> float:
    anchor_y = luminance(anchor)
    candidate_y = luminance(candidate)
    ax = cv2.Sobel(anchor_y, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(anchor_y, cv2.CV_32F, 0, 1, ksize=3)
    cx = cv2.Sobel(candidate_y, cv2.CV_32F, 1, 0, ksize=3)
    cy = cv2.Sobel(candidate_y, cv2.CV_32F, 0, 1, ksize=3)
    anchor_mag = cv2.magnitude(ax, ay)
    candidate_mag = cv2.magnitude(cx, cy)
    anchor_edge = anchor_mag > max(float(np.percentile(anchor_mag, 88)), 1e-6)
    candidate_edge = candidate_mag > max(float(np.percentile(candidate_mag, 88)), 1e-6)
    distance = cv2.distanceTransform(
        (~anchor_edge).astype(np.uint8), cv2.DIST_L2, 3
    )
    parallel = np.abs(ax * cx + ay * cy) / (
        anchor_mag * candidate_mag + 1e-6
    )
    suspicious = (
        candidate_edge
        & (distance >= 1.5)
        & (distance <= 6.0)
        & (mask > 1e-3)
    )
    if not np.any(suspicious):
        return 0.0
    # Nearby parallel edges are more suspicious than isolated fine texture.
    return float(
        np.average(
            np.clip(parallel[suspicious], 0.0, 1.0),
            weights=mask[suspicious],
        )
        * _weighted_mean(suspicious.astype(np.float32), mask)
        * 8.0
    )


class CandidateScorer:
    def __init__(
        self,
        *,
        strictness: float = 0.70,
        color_drift_tolerance: float = 0.08,
        new_edge_tolerance: float = 0.20,
    ):
        self.strictness = float(np.clip(strictness, 0.0, 1.0))
        self.color_drift_tolerance = float(color_drift_tolerance)
        self.new_edge_tolerance = float(new_edge_tolerance)
        self.decomposer = FrequencyDecomposer()

    def score(
        self,
        anchor: np.ndarray,
        candidate: np.ndarray,
        reference: np.ndarray,
        *,
        candidate_index: int,
        boundary_error: float = 0.0,
        evaluation_mask: np.ndarray | None = None,
    ) -> CandidateScore:
        anchor = np.asarray(anchor, dtype=np.float32)
        candidate = np.asarray(candidate, dtype=np.float32)
        reasons: list[str] = []
        if candidate.shape != anchor.shape:
            raise ValueError("Candidate and anchor shapes differ.")
        mask, evaluation_fraction = _mask_for_shape(
            evaluation_mask, anchor.shape[:2]
        )
        if not np.isfinite(candidate).all():
            dummy = roundtrip_metrics(
                reference,
                np.nan_to_num(candidate),
                evaluation_mask=evaluation_mask,
            )
            return CandidateScore(
                candidate_index=candidate_index,
                accepted=False,
                total=-1.0e9,
                coherent_detail_gain=0.0,
                mid_frequency_gain=0.0,
                line_continuity=0.0,
                material_richness=0.0,
                style_consistency=0.0,
                orientation_alignment=0.5,
                noise_penalty=1.0,
                duplicate_edge_penalty=1.0,
                color_drift=1.0,
                boundary_error=float(boundary_error),
                structure_error=1.0,
                clipping_ratio=1.0,
                roundtrip=dummy,
                raw_mid_frequency_energy=0.0,
                raw_high_frequency_energy=0.0,
                normalized_detail_score=0.0,
                evaluation_fraction=evaluation_fraction,
                edge_precision=dummy.edge_precision,
                edge_recall=dummy.edge_recall,
                edge_f1=dummy.edge_f1,
                rejection_reasons=["NaN or Inf"],
            )

        roundtrip = roundtrip_metrics(
            reference, candidate, evaluation_mask=evaluation_mask
        )
        anchor_y = luminance(anchor)
        anchor_bands = self.decomposer.decompose(anchor_y)
        anchor_energy = {
            name: _weighted_mean(np.square(band), mask)
            for name, band in anchor_bands.items()
        }
        del anchor_bands
        residual = candidate - anchor
        residual_y = luminance(residual)
        bands = self.decomposer.decompose(residual_y)
        energy = {
            name: _weighted_mean(np.square(band), mask)
            for name, band in bands.items()
        }
        raw_mid_energy = energy["mid_high"] + energy["mid"]
        raw_high_energy = energy["high_0"] + energy["high_1"]
        anchor_mid_rms = np.sqrt(
            max(anchor_energy["mid_high"] + anchor_energy["mid"], 0.0)
        )
        anchor_high_rms = np.sqrt(
            max(anchor_energy["high_0"] + anchor_energy["high_1"], 0.0)
        )
        signal_floor = max(
            0.002,
            0.015
            * np.sqrt(
                max(
                    _weighted_mean(
                        np.square(anchor_y - np.mean(anchor_y)),
                        mask,
                    ),
                    0.0,
                )
            ),
        )
        normalized_mid = float(
            np.tanh(
                np.sqrt(max(raw_mid_energy, 0.0))
                / max(anchor_mid_rms + signal_floor, 1e-6)
            )
        )
        normalized_high = float(
            np.tanh(
                np.sqrt(max(raw_high_energy, 0.0))
                / max(anchor_high_rms + signal_floor, 1e-6)
            )
        )
        normalized_detail = float(
            np.clip(0.72 * normalized_mid + 0.28 * normalized_high, 0.0, 1.0)
        )
        new_edge_map = new_edge_confidence(anchor, candidate)
        orientation_alignment = _orientation_alignment(anchor, candidate, mask)
        anchor_line_continuity = _gradient_coherence(anchor_y, mask)
        line_continuity = _gradient_coherence(luminance(candidate), mask)
        noise_penalty = spectral_flatness(residual_y * np.sqrt(mask))
        duplicate_penalty = _duplicate_edge_penalty(anchor, candidate, mask)
        raw_material = (
            _weighted_mean(np.abs(bands["mid_high"]), mask)
            + _weighted_mean(np.abs(bands["mid"]), mask)
        )
        material_richness = float(
            np.tanh(raw_material / max(anchor_mid_rms + signal_floor, 1e-6))
        )
        style_consistency = float(
            np.clip(
                1.0
                - abs(
                    anchor_line_continuity - line_continuity
                ),
                0.0,
                1.0,
            )
        )
        edge_admission = _weighted_mean(new_edge_map, mask)
        coherent_detail = float(
            normalized_detail
            * orientation_alignment
            * edge_admission
            * (1.0 - 0.75 * noise_penalty)
        )
        clipping = float(
            _weighted_mean(
                np.any(
                    (candidate <= 1e-5) | (candidate >= 1.0 - 1e-5),
                    axis=2,
                ).astype(np.float32),
                mask,
            )
        )
        structure_error = float(
            (1.0 - roundtrip.ssim)
            + roundtrip.low_frequency_mse * 4.0
            + roundtrip.edge_displacement
            + 0.5 * (1.0 - roundtrip.edge_f1)
        )
        color_drift = roundtrip.color_drift

        min_ssim = 0.50 + 0.28 * self.strictness
        max_edge = self.new_edge_tolerance * (1.45 - 0.65 * self.strictness)
        max_color = self.color_drift_tolerance * (
            1.45 - 0.65 * self.strictness
        )
        if roundtrip.ssim < min_ssim:
            reasons.append(
                f"round-trip SSIM {roundtrip.ssim:.4f} < {min_ssim:.4f}"
            )
        if roundtrip.edge_displacement > max_edge:
            reasons.append(
                f"edge displacement {roundtrip.edge_displacement:.4f} > {max_edge:.4f}"
            )
        if color_drift > max_color:
            reasons.append(f"color drift {color_drift:.4f} > {max_color:.4f}")
        if duplicate_penalty > 0.32:
            reasons.append(f"duplicate edge penalty {duplicate_penalty:.4f}")
        if clipping > 0.22:
            reasons.append(f"clipping ratio {clipping:.4f}")
        if boundary_error > 2.8:
            reasons.append(f"tile seam ratio {boundary_error:.4f}")

        total = (
            2.4 * coherent_detail
            + 1.4 * normalized_detail
            + 0.40 * material_richness
            + 0.25 * (line_continuity - anchor_line_continuity)
            - 0.40 * (1.0 - style_consistency)
            - 0.30 * normalized_detail * (1.0 - orientation_alignment)
            - 0.80 * noise_penalty * normalized_detail
            - 1.8 * duplicate_penalty
            - 3.0 * color_drift
            - 0.03 * boundary_error
            - 1.20 * structure_error
        )
        return CandidateScore(
            candidate_index=candidate_index,
            accepted=not reasons,
            total=float(total),
            coherent_detail_gain=coherent_detail,
            mid_frequency_gain=normalized_mid,
            line_continuity=line_continuity,
            material_richness=material_richness,
            style_consistency=style_consistency,
            orientation_alignment=orientation_alignment,
            noise_penalty=noise_penalty,
            duplicate_edge_penalty=duplicate_penalty,
            color_drift=color_drift,
            boundary_error=float(boundary_error),
            structure_error=structure_error,
            clipping_ratio=clipping,
            roundtrip=roundtrip,
            raw_mid_frequency_energy=float(raw_mid_energy),
            raw_high_frequency_energy=float(raw_high_energy),
            normalized_detail_score=normalized_detail,
            evaluation_fraction=roundtrip.evaluation_fraction,
            edge_precision=roundtrip.edge_precision,
            edge_recall=roundtrip.edge_recall,
            edge_f1=roundtrip.edge_f1,
            rejection_reasons=reasons,
        )

    @staticmethod
    def enforce_anchor_baseline(
        score: CandidateScore,
        anchor_baseline: CandidateScore,
        *,
        margin: float,
        minimum_detail: float = 1e-4,
    ) -> CandidateScore:
        if not np.isfinite(score.total):
            score.accepted = False
            return score
        if score.total < anchor_baseline.total + max(0.0, float(margin)):
            score.accepted = False
            reason = "candidate score did not beat Anchor baseline"
            if reason not in score.rejection_reasons:
                score.rejection_reasons.append(reason)
        if score.normalized_detail_score <= minimum_detail:
            score.accepted = False
            reason = "candidate added no useful detail"
            if reason not in score.rejection_reasons:
                score.rejection_reasons.append(reason)
        return score

    @staticmethod
    def select(
        anchor: np.ndarray,
        candidates: list[np.ndarray],
        scores: list[CandidateScore],
        *,
        anchor_baseline: CandidateScore | None = None,
        score_margin: float = 0.0,
    ) -> tuple[np.ndarray, int | None, CandidateScore | None]:
        if len(candidates) != len(scores):
            raise ValueError("Candidate and score counts differ.")
        if anchor_baseline is not None:
            for score in scores:
                CandidateScorer.enforce_anchor_baseline(
                    score, anchor_baseline, margin=score_margin
                )
        accepted = [
            (candidate, score)
            for candidate, score in zip(candidates, scores)
            if score.accepted
        ]
        if not accepted:
            return np.asarray(anchor, dtype=np.float32), None, None
        candidate, score = max(accepted, key=lambda item: item[1].total)
        return candidate, score.candidate_index, score
