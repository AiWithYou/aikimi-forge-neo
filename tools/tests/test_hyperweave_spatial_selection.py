from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.color import luminance
from hyperweave.quality import resize_float
from hyperweave.scoring import CandidateScorer
from hyperweave.spatial_selection import SpatialResidualSelector


def structured_canvas() -> np.ndarray:
    height, width = 128, 256
    yy, xx = np.mgrid[:height, :width]
    canvas = np.zeros((height, width, 3), dtype=np.float32)
    canvas[..., 0] = 0.18 + xx / width * 0.48
    canvas[..., 1] = 0.20 + yy / height * 0.42
    canvas[..., 2] = 0.30 + (xx + yy) / (width + height) * 0.25
    cv2.rectangle(canvas, (20, 18), (112, 108), (0.72, 0.42, 0.28), 3)
    cv2.line(canvas, (12, 96), (238, 34), (0.12, 0.16, 0.20), 3)
    cv2.circle(canvas, (76, 62), 24, (0.30, 0.72, 0.52), 3)
    return np.clip(canvas, 0.0, 1.0)


def coherent_detail(anchor: np.ndarray) -> np.ndarray:
    y = luminance(anchor)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    tangent = np.arctan2(gy, gx) + np.pi / 2.0
    yy, xx = np.mgrid[: y.shape[0], : y.shape[1]]
    phase = (xx * np.cos(tangent) + yy * np.sin(tangent)) * 0.24
    line = np.sin(phase) * np.clip(magnitude * 3.0, 0.0, 1.0)
    detail = line[..., None] * np.array([0.75, 0.90, 1.0], dtype=np.float32)
    highpass = anchor - cv2.GaussianBlur(anchor, (0, 0), 1.2)
    return np.clip(anchor + detail * 0.035 + highpass * 0.18, 0.0, 1.0)


class SpatialResidualSelectorTests(unittest.TestCase):
    def test_salvages_safe_region_from_globally_rejected_candidate(self):
        anchor = structured_canvas()
        reference = resize_float(anchor, (128, 64))
        candidate = coherent_detail(anchor)
        corrupt_from = 96
        candidate[:, corrupt_from:] = 1.0 - anchor[:, corrupt_from:, ::-1]
        scorer = CandidateScorer()
        global_score = scorer.score(anchor, candidate, reference, candidate_index=0)
        self.assertFalse(global_score.accepted, global_score.to_dict())

        selector = SpatialResidualSelector(
            anchor,
            reference,
            decision_size=64,
            transition_width=8,
            score_margin=0.05,
            fragmentation_limit=0.75,
        )
        selected = selector.consider(
            candidate,
            np.ones(anchor.shape[:2], dtype=np.float32),
            candidate_index=0,
            global_score=global_score,
        )
        result = selector.finalize()

        self.assertGreater(selected, 0)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertLess(result.report["selected_cells"], result.report["total_cells"])
        left_delta = np.mean(np.abs(result.candidate[:, 20:80] - anchor[:, 20:80]))
        right_delta = np.max(np.abs(result.candidate[:, 210:245] - anchor[:, 210:245]))
        self.assertGreater(float(left_delta), 1e-5)
        self.assertLess(float(right_delta), 1e-7, result.report)
        self.assertEqual(
            0,
            int(np.count_nonzero(result.confidence[:, 210:245])),
        )

    def test_candidate_transitions_return_through_anchor(self):
        anchor = structured_canvas()
        reference = resize_float(anchor, (128, 64))
        candidate = coherent_detail(anchor)
        scorer = CandidateScorer()
        global_score = scorer.score(anchor, candidate, reference, candidate_index=0)
        global_score.accepted = False
        global_score.rejection_reasons.append("synthetic global veto")

        selector = SpatialResidualSelector(
            anchor,
            reference,
            decision_size=128,
            transition_width=16,
            score_margin=0.0,
            fragmentation_limit=1.0,
            minimum_component_cells=1,
        )
        selector.consider(
            candidate,
            np.ones(anchor.shape[:2], dtype=np.float32),
            candidate_index=0,
            global_score=global_score,
        )
        selector.labels[0, 1] = -1
        selector.selected[:, 128:] = anchor[:, 128:]
        selector.selected_confidence[:, 128:] = 0.0
        result = selector.finalize()

        self.assertIsNotNone(result)
        assert result is not None
        boundary_delta = np.max(
            np.abs(result.candidate[:, 127:129] - anchor[:, 127:129])
        )
        interior_delta = np.mean(np.abs(result.candidate[:, 48:80] - anchor[:, 48:80]))
        self.assertLess(float(boundary_delta), 1e-7)
        self.assertGreater(float(interior_delta), 1e-5)
        self.assertEqual(0.0, result.report["boundary_jump_max"])


if __name__ == "__main__":
    unittest.main()
