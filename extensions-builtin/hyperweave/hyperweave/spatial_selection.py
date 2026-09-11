"""Fail-closed spatial rescue for globally rejected generative candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .quality import SeamAnalyzer
from .scoring import CandidateScore, CandidateScorer


@dataclass(frozen=True)
class DecisionCell:
    row: int
    column: int
    core_box: tuple[int, int, int, int]
    context_box: tuple[int, int, int, int]
    reference_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class SpatialSelectionResult:
    candidate: np.ndarray
    confidence: np.ndarray
    report: dict[str, object]


def _axis_edges(length: int, target: int) -> list[int]:
    if length <= 0:
        raise ValueError("Spatial selection axis length must be positive.")
    count = max(1, math.ceil(length / max(1, target)))
    edges = [round(index * length / count) for index in range(count + 1)]
    edges[0] = 0
    edges[-1] = length
    return edges


def _smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


class SpatialResidualSelector:
    """Choose one rejected candidate or the anchor per coarse decision cell.

    Candidates are considered sequentially. Only the current winning pixels and
    a coarse score grid remain resident, so peak memory does not scale with the
    number of generated candidates. Candidate-to-candidate transitions return
    through the anchor inside a smooth collar; candidate residuals are never
    averaged with each other.
    """

    def __init__(
        self,
        anchor: np.ndarray,
        reference: np.ndarray,
        *,
        decision_size: int = 480,
        transition_width: int = 48,
        score_margin: float = 0.05,
        fragmentation_limit: float = 0.45,
        minimum_component_cells: int = 2,
        strictness: float = 0.70,
        color_drift_tolerance: float = 0.08,
        new_edge_tolerance: float = 0.20,
        vertical_boundaries: list[int] | None = None,
        horizontal_boundaries: list[int] | None = None,
    ):
        anchor_array = np.asarray(anchor, dtype=np.float32)
        reference_array = np.asarray(reference, dtype=np.float32)
        if anchor_array.ndim != 3 or anchor_array.shape[2] != 3:
            raise ValueError("Spatial selection anchor must be H×W×3.")
        if reference_array.ndim != 3 or reference_array.shape[2] != 3:
            raise ValueError("Spatial selection reference must be H×W×3.")
        if not np.isfinite(anchor_array).all():
            raise ValueError("Spatial selection anchor contains NaN or Inf.")
        if not np.isfinite(reference_array).all():
            raise ValueError("Spatial selection reference contains NaN or Inf.")
        if decision_size < 32:
            raise ValueError("Spatial decision size must be at least 32 pixels.")
        if transition_width < 1:
            raise ValueError("Spatial transition width must be positive.")
        if score_margin < 0:
            raise ValueError("Spatial score margin cannot be negative.")
        if not 0.0 <= fragmentation_limit <= 1.0:
            raise ValueError("Spatial fragmentation limit must be in [0, 1].")
        if minimum_component_cells < 1:
            raise ValueError("Spatial minimum component cells must be positive.")

        self.anchor = anchor_array
        self.reference = reference_array
        self.height, self.width = anchor_array.shape[:2]
        self.reference_height, self.reference_width = reference_array.shape[:2]
        self.decision_size = int(decision_size)
        self.transition_width = int(transition_width)
        self.score_margin = float(score_margin)
        self.fragmentation_limit = float(fragmentation_limit)
        self.minimum_component_cells = int(minimum_component_cells)
        self.vertical_boundaries = sorted(set(vertical_boundaries or []))
        self.horizontal_boundaries = sorted(set(horizontal_boundaries or []))
        self.scorer = CandidateScorer(
            strictness=strictness,
            color_drift_tolerance=color_drift_tolerance,
            new_edge_tolerance=new_edge_tolerance,
        )
        self.seam_analyzer = SeamAnalyzer()

        self.x_edges = _axis_edges(self.width, self.decision_size)
        self.y_edges = _axis_edges(self.height, self.decision_size)
        self.rows = len(self.y_edges) - 1
        self.columns = len(self.x_edges) - 1
        self.cells = self._build_cells()
        self.labels = np.full((self.rows, self.columns), -1, dtype=np.int16)
        self.best_scores = np.full((self.rows, self.columns), -np.inf, dtype=np.float32)
        self.baseline_scores = np.full(
            (self.rows, self.columns), -np.inf, dtype=np.float32
        )
        self.selected = self.anchor.copy()
        self.selected_confidence = np.zeros((self.height, self.width), dtype=np.float32)
        self.candidate_stats: dict[int, dict[str, object]] = {}
        self.last_failure_reason: str | None = None
        self.pruned_cells = 0
        self._initialize_baselines()

    def _map_reference_box(
        self, box: tuple[int, int, int, int]
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = box
        rx0 = max(0, math.floor(x0 * self.reference_width / self.width))
        ry0 = max(0, math.floor(y0 * self.reference_height / self.height))
        rx1 = min(
            self.reference_width,
            math.ceil(x1 * self.reference_width / self.width),
        )
        ry1 = min(
            self.reference_height,
            math.ceil(y1 * self.reference_height / self.height),
        )
        if rx1 - rx0 < 8:
            rx1 = min(self.reference_width, rx0 + 8)
            rx0 = max(0, rx1 - 8)
        if ry1 - ry0 < 8:
            ry1 = min(self.reference_height, ry0 + 8)
            ry0 = max(0, ry1 - 8)
        return rx0, ry0, rx1, ry1

    def _build_cells(self) -> list[DecisionCell]:
        cells: list[DecisionCell] = []
        context = max(8, min(self.transition_width, self.decision_size // 4))
        for row, (y0, y1) in enumerate(zip(self.y_edges[:-1], self.y_edges[1:])):
            for column, (x0, x1) in enumerate(zip(self.x_edges[:-1], self.x_edges[1:])):
                context_box = (
                    max(0, x0 - context),
                    max(0, y0 - context),
                    min(self.width, x1 + context),
                    min(self.height, y1 + context),
                )
                cells.append(
                    DecisionCell(
                        row=row,
                        column=column,
                        core_box=(x0, y0, x1, y1),
                        context_box=context_box,
                        reference_box=self._map_reference_box(context_box),
                    )
                )
        return cells

    def _local_boundaries(self, cell: DecisionCell) -> tuple[list[int], list[int]]:
        x0, y0, x1, y1 = cell.context_box
        vertical = [
            value - x0 for value in self.vertical_boundaries if x0 + 2 <= value < x1 - 2
        ]
        horizontal = [
            value - y0
            for value in self.horizontal_boundaries
            if y0 + 2 <= value < y1 - 2
        ]
        return vertical, horizontal

    def _score_cell(
        self,
        cell: DecisionCell,
        candidate: np.ndarray,
        *,
        candidate_index: int,
    ) -> CandidateScore:
        x0, y0, x1, y1 = cell.context_box
        rx0, ry0, rx1, ry1 = cell.reference_box
        anchor_crop = self.anchor[y0:y1, x0:x1]
        candidate_crop = candidate[y0:y1, x0:x1]
        reference_crop = self.reference[ry0:ry1, rx0:rx1]
        vertical, horizontal = self._local_boundaries(cell)
        seam = self.seam_analyzer.analyze(
            candidate_crop - anchor_crop, vertical, horizontal
        )
        return self.scorer.score(
            anchor_crop,
            candidate_crop,
            reference_crop,
            candidate_index=candidate_index,
            boundary_error=seam.ratio,
        )

    def _initialize_baselines(self) -> None:
        for cell in self.cells:
            score = self._score_cell(cell, self.anchor, candidate_index=-1)
            self.baseline_scores[cell.row, cell.column] = score.total
            self.best_scores[cell.row, cell.column] = score.total

    def consider(
        self,
        candidate: np.ndarray,
        confidence: np.ndarray,
        *,
        candidate_index: int,
        global_score: CandidateScore,
    ) -> int:
        candidate_array = np.asarray(candidate, dtype=np.float32)
        confidence_array = np.asarray(confidence, dtype=np.float32)
        if candidate_array.shape != self.anchor.shape:
            raise ValueError("Spatial candidate shape does not match the anchor.")
        if confidence_array.shape != self.anchor.shape[:2]:
            raise ValueError("Spatial candidate confidence has the wrong shape.")

        stats: dict[str, object] = {
            "candidate_index": int(candidate_index),
            "cells_evaluated": 0,
            "cells_passing_hard_gates": 0,
            "cells_selected": 0,
            "globally_rejected_reasons": list(global_score.rejection_reasons),
            "fatal_rejection": None,
        }
        self.candidate_stats[int(candidate_index)] = stats
        if not (
            np.isfinite(candidate_array).all() and np.isfinite(confidence_array).all()
        ):
            stats["fatal_rejection"] = "NaN or Inf"
            return 0
        if global_score.clipping_ratio > 0.22:
            stats["fatal_rejection"] = "canvas-wide clipping ratio"
            return 0

        selected_count = 0
        for cell in self.cells:
            stats["cells_evaluated"] = int(stats["cells_evaluated"]) + 1
            score = self._score_cell(
                cell, candidate_array, candidate_index=candidate_index
            )
            if not score.accepted:
                continue
            stats["cells_passing_hard_gates"] = (
                int(stats["cells_passing_hard_gates"]) + 1
            )
            current = float(self.best_scores[cell.row, cell.column])
            baseline = float(self.baseline_scores[cell.row, cell.column])
            required = max(current, baseline) + self.score_margin
            if (
                score.total < required
                or score.mid_frequency_gain <= 1e-8
                or score.coherent_detail_gain <= 0.0
            ):
                continue
            x0, y0, x1, y1 = cell.core_box
            self.selected[y0:y1, x0:x1] = candidate_array[y0:y1, x0:x1]
            self.selected_confidence[y0:y1, x0:x1] = confidence_array[y0:y1, x0:x1]
            self.labels[cell.row, cell.column] = int(candidate_index)
            self.best_scores[cell.row, cell.column] = score.total
            selected_count += 1

        stats["cells_selected"] = selected_count
        return selected_count

    def _fragmentation(self) -> tuple[float, int, int]:
        horizontal = self.labels[:, 1:] != self.labels[:, :-1]
        vertical = self.labels[1:, :] != self.labels[:-1, :]
        switches = int(np.count_nonzero(horizontal) + np.count_nonzero(vertical))
        comparisons = int(horizontal.size + vertical.size)
        return switches / max(comparisons, 1), switches, comparisons

    def _prune_small_components(self) -> int:
        pruned = 0
        for label in np.unique(self.labels):
            if label < 0:
                continue
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (self.labels == label).astype(np.uint8),
                connectivity=4,
            )
            for component in range(1, count):
                area = int(stats[component, cv2.CC_STAT_AREA])
                if area >= self.minimum_component_cells:
                    continue
                cells = np.argwhere(components == component)
                for row, column in cells:
                    cell = self.cells[int(row) * self.columns + int(column)]
                    x0, y0, x1, y1 = cell.core_box
                    self.selected[y0:y1, x0:x1] = self.anchor[y0:y1, x0:x1]
                    self.selected_confidence[y0:y1, x0:x1] = 0.0
                    self.labels[row, column] = -1
                    self.best_scores[row, column] = self.baseline_scores[row, column]
                    pruned += 1
        return pruned

    def _pixel_labels(self) -> np.ndarray:
        result = np.full((self.height, self.width), -1, dtype=np.int16)
        for cell in self.cells:
            x0, y0, x1, y1 = cell.core_box
            result[y0:y1, x0:x1] = self.labels[cell.row, cell.column]
        return result

    def _candidate_stats_report(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for key in sorted(self.candidate_stats):
            item = dict(self.candidate_stats[key])
            item["cells_selected_final"] = int(np.count_nonzero(self.labels == key))
            result.append(item)
        return result

    def diagnostic_report(self) -> dict[str, object]:
        return {
            "decision_size": self.decision_size,
            "transition_width": self.transition_width,
            "score_margin": self.score_margin,
            "fragmentation_limit": self.fragmentation_limit,
            "minimum_component_cells": self.minimum_component_cells,
            "grid": [self.columns, self.rows],
            "labels": self.labels.tolist(),
            "selected_cells": int(np.count_nonzero(self.labels >= 0)),
            "total_cells": int(self.labels.size),
            "pruned_cells": self.pruned_cells,
            "failure_reason": self.last_failure_reason,
            "candidate_stats": self._candidate_stats_report(),
        }

    def finalize(self) -> SpatialSelectionResult | None:
        self.last_failure_reason = None
        self.pruned_cells = self._prune_small_components()
        selected_cells = int(np.count_nonzero(self.labels >= 0))
        total_cells = int(self.labels.size)
        if selected_cells == 0:
            self.last_failure_reason = "no connected locally safe cells"
            return None

        fragmentation, switches, comparisons = self._fragmentation()
        if fragmentation > self.fragmentation_limit:
            self.last_failure_reason = (
                f"fragmentation {fragmentation:.6f} exceeds "
                f"{self.fragmentation_limit:.6f}"
            )
            return None

        pixel_labels = self._pixel_labels()
        candidate_region = pixel_labels >= 0
        boundary = np.zeros((self.height, self.width), dtype=np.uint8)
        different_x = pixel_labels[:, 1:] != pixel_labels[:, :-1]
        different_y = pixel_labels[1:, :] != pixel_labels[:-1, :]
        boundary[:, 1:][different_x] = 1
        boundary[:, :-1][different_x] = 1
        boundary[1:, :][different_y] = 1
        boundary[:-1, :][different_y] = 1

        if np.any(boundary):
            distance = cv2.distanceTransform(
                (boundary == 0).astype(np.uint8), cv2.DIST_L2, 5
            )
            transition = _smoothstep(distance / self.transition_width)
        else:
            transition = np.ones((self.height, self.width), dtype=np.float32)
        admission = transition * candidate_region.astype(np.float32)
        candidate = self.anchor + (self.selected - self.anchor) * admission[..., None]
        confidence = np.clip(self.selected_confidence * admission, 0.0, 1.0).astype(
            np.float32
        )
        if not (np.isfinite(candidate).all() and np.isfinite(confidence).all()):
            self.last_failure_reason = "composite contains NaN or Inf"
            return None
        residual = candidate - self.anchor
        boundary_jumps: list[np.ndarray] = []
        if np.any(different_x):
            boundary_jumps.append(
                np.abs(residual[:, 1:] - residual[:, :-1])[different_x]
            )
        if np.any(different_y):
            boundary_jumps.append(
                np.abs(residual[1:, :] - residual[:-1, :])[different_y]
            )
        if boundary_jumps:
            jump_values = np.concatenate(
                [values.reshape(-1) for values in boundary_jumps]
            )
            boundary_jump_mean = float(np.mean(jump_values))
            boundary_jump_max = float(np.max(jump_values))
        else:
            boundary_jump_mean = 0.0
            boundary_jump_max = 0.0
        if boundary_jump_max > 1e-5:
            self.last_failure_reason = (
                f"boundary jump {boundary_jump_max:.8f} exceeds 0.00001000"
            )
            return None

        label_counts = {
            str(int(label)): int(np.count_nonzero(self.labels == label))
            for label in np.unique(self.labels)
            if label >= 0
        }
        report = {
            "mode": "spatial_residual_rescue",
            "decision_size": self.decision_size,
            "transition_width": self.transition_width,
            "score_margin": self.score_margin,
            "fragmentation_limit": self.fragmentation_limit,
            "minimum_component_cells": self.minimum_component_cells,
            "grid": [self.columns, self.rows],
            "labels": self.labels.tolist(),
            "total_cells": total_cells,
            "selected_cells": selected_cells,
            "pruned_cells": self.pruned_cells,
            "selected_cell_fraction": selected_cells / max(total_cells, 1),
            "label_counts": label_counts,
            "label_switches": switches,
            "label_comparisons": comparisons,
            "fragmentation": fragmentation,
            "transition_pixels": int(
                np.count_nonzero(candidate_region & (admission < 0.999))
            ),
            "boundary_jump_mean": boundary_jump_mean,
            "boundary_jump_max": boundary_jump_max,
            "admitted_pixels": int(np.count_nonzero(admission > 1e-4)),
            "candidate_stats": self._candidate_stats_report(),
        }
        return SpatialSelectionResult(
            candidate=np.clip(candidate, 0.0, 1.0).astype(np.float32),
            confidence=confidence,
            report=report,
        )
