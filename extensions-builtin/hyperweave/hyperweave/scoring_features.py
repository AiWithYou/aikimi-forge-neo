"""Shared, numerically stable scoring features."""

from __future__ import annotations

import cv2
import numpy as np


SCORING_VERSION = "proofweave-1.2.0"


def spectral_flatness(
    residual: np.ndarray,
    *,
    maximum_size: int = 512,
    zero_power_threshold: float = 1e-14,
    relative_floor: float = 1e-12,
) -> float:
    """Return bounded spectral flatness without treating zero residual as noise.

    The numerical floor follows the observed mean spectrum power.  A fixed
    epsilon added to every FFT bin would turn an all-zero residual into a
    perfectly flat (white) spectrum, which is the opposite of the intended
    noise penalty.
    """

    sample = np.asarray(residual, dtype=np.float32)
    if sample.ndim == 3:
        sample = np.mean(sample, axis=2, dtype=np.float32)
    if sample.ndim != 2 or sample.size == 0:
        return 0.0
    if not np.isfinite(sample).all():
        return 1.0

    rms_power = float(np.mean(np.square(sample, dtype=np.float64)))
    if not np.isfinite(rms_power) or rms_power <= zero_power_threshold:
        return 0.0

    if max(sample.shape) > maximum_size:
        scale = maximum_size / max(sample.shape)
        sample = cv2.resize(
            sample,
            (
                max(8, round(sample.shape[1] * scale)),
                max(8, round(sample.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )

    power = np.square(np.abs(np.fft.rfft2(sample)), dtype=np.float64)
    arithmetic = float(np.mean(power))
    if not np.isfinite(arithmetic) or arithmetic <= zero_power_threshold:
        return 0.0

    floor = max(
        arithmetic * max(float(relative_floor), np.finfo(np.float64).eps),
        np.finfo(np.float64).tiny,
    )
    positive = np.maximum(power, floor)
    geometric = float(np.exp(np.mean(np.log(positive))))
    if not np.isfinite(geometric):
        return 1.0
    return float(np.clip(geometric / arithmetic, 0.0, 1.0))
