"""RAM and disk-backed weighted residual accumulators."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np


EPSILON = 1e-8


@dataclass(frozen=True)
class AccumulatorResult:
    candidate: np.ndarray
    mean_delta: np.ndarray
    variance: np.ndarray
    weight_sum: np.ndarray
    tile_confidence: np.ndarray


class AccumulatorBackend(ABC):
    def __init__(self, base: np.ndarray):
        base = np.asarray(base, dtype=np.float32)
        if base.ndim != 3 or base.shape[2] != 3:
            raise ValueError("Accumulator base must be H×W×3.")
        if not np.isfinite(base).all():
            raise ValueError("Accumulator base contains NaN or Inf.")
        self.base = base
        self.height, self.width = base.shape[:2]
        self.closed = False

    @property
    @abstractmethod
    def delta_sum(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def weight_sum(self) -> np.ndarray: ...

    @property
    @abstractmethod
    def delta_luma_squared_sum(self) -> np.ndarray: ...

    def add(
        self,
        core_box: tuple[int, int, int, int],
        generated_core: np.ndarray,
        base_core: np.ndarray,
        weight: np.ndarray,
    ) -> None:
        if self.closed:
            raise RuntimeError("Accumulator is closed.")
        x0, y0, x1, y1 = core_box
        expected = (y1 - y0, x1 - x0)
        generated = np.asarray(generated_core, dtype=np.float32)
        base = np.asarray(base_core, dtype=np.float32)
        weights = np.asarray(weight, dtype=np.float32)
        if generated.shape != expected + (3,) or base.shape != generated.shape:
            raise ValueError("Tile RGB arrays do not match the core box.")
        if weights.shape != expected:
            raise ValueError("Tile weight does not match the core box.")
        if not (
            np.isfinite(generated).all()
            and np.isfinite(base).all()
            and np.isfinite(weights).all()
        ):
            raise ValueError("Accumulator input contains NaN or Inf.")
        if np.any(weights < 0):
            raise ValueError("Accumulator weights cannot be negative.")
        delta = generated - base
        section = np.s_[y0:y1, x0:x1]
        self.delta_sum[section] += delta * weights[..., None]
        self.weight_sum[section] += weights
        self.delta_luma_squared_sum[section] += (
            np.mean(np.square(delta), axis=2, dtype=np.float32) * weights
        )

    def finalize(self) -> AccumulatorResult:
        if self.closed:
            raise RuntimeError("Accumulator is closed.")
        if np.any(self.weight_sum <= 0):
            missing = int(np.count_nonzero(self.weight_sum <= 0))
            raise RuntimeError(f"Accumulator has {missing} uncovered pixels.")
        divisor = np.maximum(self.weight_sum, EPSILON)
        mean_delta = self.delta_sum / divisor[..., None]
        second_moment = self.delta_luma_squared_sum / divisor
        mean_square = np.mean(np.square(mean_delta), axis=2, dtype=np.float32)
        variance = np.maximum(0.0, second_moment - mean_square)
        finite = np.isfinite(mean_delta).all() and np.isfinite(variance).all()
        if not finite:
            raise RuntimeError("Accumulator produced NaN or Inf.")
        p50, p95 = np.percentile(variance, (50.0, 95.0))
        normalized = np.maximum(0.0, variance - p50) / max(p95 - p50, EPSILON)
        confidence = np.exp(-normalized).astype(np.float32)
        candidate = np.clip(self.base + mean_delta, 0.0, 1.0)
        return AccumulatorResult(
            candidate=candidate,
            mean_delta=mean_delta,
            variance=variance,
            weight_sum=np.asarray(self.weight_sum).copy(),
            tile_confidence=confidence,
        )

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> "AccumulatorBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class InMemoryAccumulator(AccumulatorBackend):
    def __init__(self, base: np.ndarray):
        super().__init__(base)
        self._delta_sum = np.zeros_like(self.base, dtype=np.float32)
        self._weight_sum = np.zeros((self.height, self.width), dtype=np.float32)
        self._square_sum = np.zeros((self.height, self.width), dtype=np.float32)

    @property
    def delta_sum(self) -> np.ndarray:
        return self._delta_sum

    @property
    def weight_sum(self) -> np.ndarray:
        return self._weight_sum

    @property
    def delta_luma_squared_sum(self) -> np.ndarray:
        return self._square_sum

    def close(self) -> None:
        self.closed = True


class MemmapAccumulator(AccumulatorBackend):
    def __init__(self, base: np.ndarray, directory: str | Path, stem: str):
        super().__init__(base)
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.paths = {
            "delta": root / f"{stem}_delta.float32",
            "weight": root / f"{stem}_weight.float32",
            "square": root / f"{stem}_square.float32",
        }
        self._delta_sum = np.memmap(
            self.paths["delta"],
            dtype=np.float32,
            mode="w+",
            shape=self.base.shape,
        )
        self._weight_sum = np.memmap(
            self.paths["weight"],
            dtype=np.float32,
            mode="w+",
            shape=(self.height, self.width),
        )
        self._square_sum = np.memmap(
            self.paths["square"],
            dtype=np.float32,
            mode="w+",
            shape=(self.height, self.width),
        )
        self._delta_sum[:] = 0
        self._weight_sum[:] = 0
        self._square_sum[:] = 0

    @property
    def delta_sum(self) -> np.ndarray:
        return self._delta_sum

    @property
    def weight_sum(self) -> np.ndarray:
        return self._weight_sum

    @property
    def delta_luma_squared_sum(self) -> np.ndarray:
        return self._square_sum

    def close(self) -> None:
        if self.closed:
            return
        for value in (self._delta_sum, self._weight_sum, self._square_sum):
            value.flush()
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self.closed = True

    def cleanup(self) -> None:
        self.close()
        for path in self.paths.values():
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass


def accumulator_bytes(width: int, height: int) -> int:
    """RGB delta + scalar weight + scalar squared luminance."""
    return int(width) * int(height) * (3 + 1 + 1) * 4
