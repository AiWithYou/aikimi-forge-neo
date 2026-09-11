from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.analysis import FaceDetection
from hyperweave.regions import (
    classify_face_size,
    expand_face_roi,
    face_core_mask,
    face_region_masks,
    hair_flow_score,
    merge_overlapping_regions,
)
from hyperweave.scoring import CandidateScorer
from hyperweave.scoring_features import spectral_flatness
import hyperweave.comparison as comparison_module


def source_pattern(size=128):
    image = np.full((size, size, 3), 0.4, np.float32)
    cv2.circle(image, (size // 2, size // 2), size // 4, (0.75, 0.65, 0.55), -1)
    cv2.line(image, (20, 25), (108, 25), (0.1, 0.1, 0.1), 2)
    cv2.rectangle(image, (20, 80), (108, 108), (0.2, 0.3, 0.7), -1)
    return image


class CandidateScorerTests(unittest.TestCase):
    def test_zero_residual_has_zero_noise_penalty(self):
        anchor = source_pattern()
        scorer = CandidateScorer(strictness=0.5)
        zero = scorer.score(anchor, anchor, anchor, candidate_index=0)
        yy, xx = np.mgrid[: anchor.shape[0], : anchor.shape[1]]
        structured = np.clip(
            anchor + 0.02 * np.sin(xx * 0.20)[..., None], 0.0, 1.0
        )
        rng = np.random.default_rng(2026)
        white = np.clip(
            anchor + rng.normal(0.0, 0.02, anchor.shape), 0.0, 1.0
        ).astype(np.float32)
        structured_score = scorer.score(
            anchor, structured, anchor, candidate_index=1
        )
        white_score = scorer.score(anchor, white, anchor, candidate_index=2)
        self.assertEqual(0.0, spectral_flatness(np.zeros((32, 32), np.float32)))
        self.assertLessEqual(zero.noise_penalty, 1e-8)
        self.assertGreater(
            white_score.noise_penalty, structured_score.noise_penalty
        )
        self.assertIs(comparison_module.spectral_flatness, spectral_flatness)

    def test_coherent_detail_beats_noise_and_color_drift_rejects(self):
        anchor = source_pattern()
        coherent = anchor.copy()
        yy, xx = np.mgrid[:128, :128]
        coherent += (
            np.sin(xx * 0.45)[..., None]
            * (cv2.Canny((anchor[..., 0] * 255).astype(np.uint8), 30, 100) > 0)[..., None]
            * 0.01
        )
        rng = np.random.default_rng(4)
        noisy = np.clip(anchor + rng.normal(0, 0.08, anchor.shape), 0, 1)
        drift = np.clip(anchor + np.array([0.2, -0.05, -0.05]), 0, 1)
        scorer = CandidateScorer(strictness=0.5)
        good_score = scorer.score(anchor, coherent, anchor, candidate_index=0)
        noise_score = scorer.score(anchor, noisy, anchor, candidate_index=1)
        drift_score = scorer.score(anchor, drift, anchor, candidate_index=2)
        self.assertGreater(good_score.total, noise_score.total)
        self.assertFalse(drift_score.accepted)
        selected, index, _ = scorer.select(
            anchor,
            [coherent, noisy, drift],
            [good_score, noise_score, drift_score],
        )
        self.assertEqual(0, index)
        np.testing.assert_array_equal(selected, coherent)

    def test_all_rejected_falls_back_to_anchor(self):
        anchor = source_pattern()
        candidates = [
            np.clip(anchor + 0.4, 0, 1),
            np.clip(1.0 - anchor, 0, 1),
        ]
        scorer = CandidateScorer(strictness=1.0, color_drift_tolerance=0.01)
        scores = [
            scorer.score(anchor, candidate, anchor, candidate_index=index)
            for index, candidate in enumerate(candidates)
        ]
        selected, index, score = scorer.select(anchor, candidates, scores)
        self.assertIsNone(index)
        self.assertIsNone(score)
        np.testing.assert_array_equal(selected, anchor)

    def test_nonfinite_candidate_fails_closed_with_finite_metadata(self):
        anchor = source_pattern()
        candidate = anchor.copy()
        candidate[4, 5, 1] = np.nan
        score = CandidateScorer().score(
            anchor, candidate, anchor, candidate_index=0
        )
        self.assertFalse(score.accepted)
        self.assertTrue(np.isfinite(score.total))
        self.assertIn("NaN or Inf", score.rejection_reasons)

    def test_face_mask_prevents_background_from_hiding_face_corruption(self):
        anchor = source_pattern(192)
        candidate = anchor.copy()
        candidate[84:108, 84:108] = 1.0 - candidate[84:108, 84:108]
        mask = np.zeros((192, 192), np.float32)
        cv2.circle(mask, (96, 96), 13, 1.0, -1)
        scorer = CandidateScorer(strictness=1.0)
        unmasked = scorer.score(
            anchor, candidate, anchor, candidate_index=0
        )
        masked = scorer.score(
            anchor,
            candidate,
            anchor,
            candidate_index=0,
            evaluation_mask=mask,
        )
        self.assertLess(masked.roundtrip.ssim, unmasked.roundtrip.ssim - 0.20)
        self.assertLess(masked.edge_f1, unmasked.edge_f1)
        self.assertFalse(masked.accepted, masked.to_dict())
        self.assertLess(masked.evaluation_fraction, 0.03)

    def test_candidate_must_beat_anchor_baseline(self):
        anchor = source_pattern()
        scorer = CandidateScorer(strictness=0.5)
        baseline = scorer.score(
            anchor, anchor, anchor, candidate_index=-1
        )
        candidate = scorer.score(
            anchor, anchor.copy(), anchor, candidate_index=0
        )
        self.assertTrue(candidate.accepted)
        scorer.enforce_anchor_baseline(candidate, baseline, margin=0.02)
        self.assertFalse(candidate.accepted)
        self.assertIn(
            "candidate score did not beat Anchor baseline",
            candidate.rejection_reasons,
        )
        self.assertIn(
            "candidate added no useful detail",
            candidate.rejection_reasons,
        )


class FaceRegionTests(unittest.TestCase):
    def detection(self, x0=40, y0=40, size=30):
        return FaceDetection(
            bbox=(x0, y0, x0 + size, y0 + size),
            confidence=1.0,
            landmarks=None,
            mask=None,
            detector_name="test",
            source_resolution=(128, 128),
            original_bbox_size=(size, size),
        )

    def test_source_face_size_classification(self):
        self.assertEqual("micro", classify_face_size(self.detection(size=20)))
        self.assertEqual("tiny", classify_face_size(self.detection(size=40)))
        self.assertEqual("small", classify_face_size(self.detection(size=90)))
        self.assertEqual("medium", classify_face_size(self.detection(size=150)))
        self.assertEqual("large", classify_face_size(self.detection(size=220)))

    def test_roi_expansion_transform_and_boundary_clip(self):
        region = expand_face_roi(
            self.detection(x0=4, y0=3, size=30),
            scale_x=2.0,
            scale_y=2.0,
            stage_size=(256, 256),
            region_id=0,
        )
        self.assertEqual(0, region.stage_box[0])
        self.assertEqual(0, region.stage_box[1])
        self.assertLessEqual(region.stage_box[2], 256)
        self.assertLessEqual(region.stage_box[3], 256)
        self.assertEqual((30, 30), region.original_face_size)

    def test_overlapping_rois_merge(self):
        first = expand_face_roi(
            self.detection(40, 40, 30),
            scale_x=1,
            scale_y=1,
            stage_size=(128, 128),
            region_id=0,
        )
        second = expand_face_roi(
            self.detection(44, 42, 30),
            scale_x=1,
            scale_y=1,
            stage_size=(128, 128),
            region_id=1,
        )
        self.assertEqual(1, len(merge_overlapping_regions([first, second])))

    def test_face_write_mask_does_not_modify_other_face_core(self):
        detections = [
            self.detection(38, 45, 28),
            self.detection(62, 45, 28),
        ]
        context = (0, 0, 128, 128)
        masks = face_region_masks(
            detections,
            0,
            scale_x=1.0,
            scale_y=1.0,
            stage_size=(128, 128),
            context_box=context,
        )
        second_masks = face_region_masks(
            detections,
            1,
            scale_x=1.0,
            scale_y=1.0,
            stage_size=(128, 128),
            context_box=context,
        )
        other_core = face_core_mask(
            detections[1],
            scale_x=1.0,
            scale_y=1.0,
            stage_size=(128, 128),
            output_box=context,
        )
        simulated = np.ones((128, 128), np.float32) * masks.write
        self.assertEqual(
            0.0,
            float(np.max(simulated[other_core > 0.90])),
        )
        self.assertGreater(float(np.max(simulated[masks.core > 0.90])), 0.2)
        self.assertLessEqual(
            float(np.max(masks.write + second_masks.write)), 1.0 + 1e-6
        )


class HairFlowTests(unittest.TestCase):
    def test_aligned_lines_score_above_crossing_lines(self):
        anchor = np.zeros((128, 128, 3), np.float32)
        for x in range(16, 112, 8):
            cv2.line(anchor, (x, 10), (x, 118), (0.8, 0.8, 0.8), 1)
        aligned = anchor.copy()
        for x in range(20, 112, 8):
            cv2.line(aligned, (x, 10), (x, 118), (0.4, 0.4, 0.4), 1)
        crossing = anchor.copy()
        for y in range(16, 112, 8):
            cv2.line(crossing, (10, y), (118, y), (0.4, 0.4, 0.4), 1)
        mask = np.ones((128, 128), np.float32)
        aligned_score = hair_flow_score(anchor, aligned, mask)
        crossing_score = hair_flow_score(anchor, crossing, mask)
        self.assertGreater(aligned_score.total, crossing_score.total)
        self.assertGreater(crossing_score.crossing_penalty, aligned_score.crossing_penalty)


if __name__ == "__main__":
    unittest.main()
