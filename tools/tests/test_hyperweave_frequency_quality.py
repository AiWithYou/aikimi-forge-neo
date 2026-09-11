from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.frequency import (
    ComposeMaps,
    FrequencyDecomposer,
    FrequencyAwareComposer,
    StructureMapBuilder,
    new_edge_confidence,
    orientation_confidence,
    robust_soft_clip,
)
from hyperweave.config import FrequencyGains
from hyperweave.engine import _roundtrip_confidence_map
from hyperweave.quality import LowFrequencyBackProjection, SeamAnalyzer, roundtrip_metrics


def rgb_from_gray(gray):
    return np.repeat(np.asarray(gray, np.float32)[..., None], 3, axis=2)


class FrequencyTests(unittest.TestCase):
    def setUp(self):
        self.decomposer = FrequencyDecomposer()

    def test_constant_residual_is_low_and_reconstructs(self):
        residual = np.full((96, 96), 0.2, dtype=np.float32)
        bands = self.decomposer.decompose(residual)
        self.assertLess(sum(float(np.mean(np.abs(bands[name]))) for name in ("high_0", "high_1", "mid_high", "mid", "mid_low")), 1e-5)
        np.testing.assert_allclose(
            self.decomposer.reconstruct(bands), residual, atol=2e-6
        )

    def test_thin_line_has_more_high_energy_than_thick_shape(self):
        thin = np.zeros((128, 128), np.float32)
        thin[:, 64] = 1
        thick = np.zeros_like(thin)
        thick[:, 48:80] = 1
        thin_energy = self.decomposer.energy(self.decomposer.decompose(thin))
        thick_energy = self.decomposer.energy(self.decomposer.decompose(thick))
        thin_high_fraction = (
            thin_energy["high_0"] + thin_energy["high_1"]
        ) / sum(thin_energy.values())
        thick_high_fraction = (
            thick_energy["high_0"] + thick_energy["high_1"]
        ) / sum(thick_energy.values())
        self.assertGreater(thin_high_fraction, thick_high_fraction)
        self.assertGreater(
            thick_energy["mid_low"] + thick_energy["low"],
            thin_energy["mid_low"] + thin_energy["low"],
        )

    def test_soft_clip_suppresses_outlier_and_handles_zero(self):
        values = np.full(1000, 0.01, np.float32)
        values[-1] = 10.0
        clipped = robust_soft_clip(values)
        self.assertLess(float(clipped[-1]), 0.1)
        self.assertGreater(float(np.median(clipped)), 0.005)
        np.testing.assert_array_equal(
            robust_soft_clip(np.zeros((8, 8), np.float32)),
            np.zeros((8, 8), np.float32),
        )


class ConfidenceMapTests(unittest.TestCase):
    def test_structure_map_protects_edge_and_manual_mask(self):
        gray = np.zeros((128, 128), np.float32)
        gray[:, 64:] = 1
        manual = np.zeros_like(gray)
        manual[10:20, 10:20] = 1
        maps = StructureMapBuilder().build(rgb_from_gray(gray), manual)
        self.assertGreater(float(maps.protection[:, 62:67].mean()), 0.7)
        self.assertGreater(float(maps.protection[12:18, 12:18].mean()), 0.9)
        self.assertLess(float(maps.protection[80:100, 10:30].mean()), 0.2)

    def test_orientation_same_is_high_orthogonal_is_low(self):
        horizontal = np.zeros((128, 128), np.float32)
        horizontal[63:66, :] = 1
        vertical = np.zeros_like(horizontal)
        vertical[:, 63:66] = 1
        same = orientation_confidence(
            rgb_from_gray(horizontal), rgb_from_gray(horizontal)
        )
        orthogonal = orientation_confidence(
            rgb_from_gray(horizontal), rgb_from_gray(vertical)
        )
        self.assertGreater(float(same.mean()), float(orthogonal.mean()))
        self.assertTrue(np.isfinite(orthogonal).all())

    def test_new_edge_far_from_anchor_is_rejected_but_texture_survives(self):
        anchor = np.zeros((128, 128), np.float32)
        anchor[:, 32:34] = 1
        candidate = anchor.copy()
        candidate[:, 80:82] = 1
        confidence = new_edge_confidence(
            rgb_from_gray(anchor), rgb_from_gray(candidate)
        )
        self.assertLess(float(confidence[:, 80:82].mean()), 0.35)

        texture = anchor.copy()
        texture[::16, ::16] = 0.15
        texture_confidence = new_edge_confidence(
            rgb_from_gray(anchor), rgb_from_gray(texture)
        )
        self.assertGreater(float(texture_confidence.mean()), 0.8)

    def test_composer_rejects_nonfinite_roundtrip_map(self):
        anchor = np.full((32, 32, 3), 0.4, np.float32)
        candidate = anchor + 0.01
        invalid = np.ones((32, 32), np.float32)
        invalid[4, 5] = np.nan
        maps = ComposeMaps(
            structure=np.zeros((32, 32), np.float32),
            orientation=np.ones((32, 32), np.float32),
            new_edge=np.ones((32, 32), np.float32),
            tile=np.ones((32, 32), np.float32),
            roundtrip=invalid,
        )
        with self.assertRaisesRegex(ValueError, "NaN or Inf"):
            FrequencyAwareComposer().compose(
                anchor,
                candidate,
                gains=FrequencyGains(1, 1, 1, 1, 1, 0),
                maps=maps,
                structural_lock=0.0,
                low_frequency_lock=1.0,
            )


class QualityTests(unittest.TestCase):
    def test_missing_reference_edge_reduces_recall_and_f1(self):
        reference = np.full((96, 96), 0.4, np.float32)
        cv2.line(reference, (48, 12), (48, 84), 0.9, 2)
        identical = roundtrip_metrics(
            rgb_from_gray(reference), rgb_from_gray(reference)
        )
        missing = roundtrip_metrics(
            rgb_from_gray(reference),
            rgb_from_gray(np.full_like(reference, 0.4)),
        )
        self.assertLess(missing.edge_recall, identical.edge_recall)
        self.assertLess(missing.edge_f1, identical.edge_f1)
        self.assertGreater(
            missing.edge_displacement, identical.edge_displacement
        )
        self.assertGreater(missing.edge_displacement_reverse, 0.0)

    def test_added_false_edge_reduces_precision(self):
        reference = np.full((96, 96), 0.4, np.float32)
        cv2.line(reference, (24, 12), (24, 84), 0.9, 2)
        candidate = reference.copy()
        cv2.line(candidate, (72, 12), (72, 84), 0.9, 2)
        identical = roundtrip_metrics(
            rgb_from_gray(reference), rgb_from_gray(reference)
        )
        added = roundtrip_metrics(
            rgb_from_gray(reference), rgb_from_gray(candidate)
        )
        self.assertLess(added.edge_precision, identical.edge_precision)
        self.assertLess(added.edge_f1, identical.edge_f1)
        self.assertGreater(added.edge_displacement_forward, 0.0)

    def test_local_roundtrip_map_suppresses_corrupt_region(self):
        yy, xx = np.mgrid[:128, :128]
        anchor_gray = 0.35 + 0.25 * xx / 127 + 0.10 * yy / 127
        anchor = rgb_from_gray(anchor_gray)
        residual = 0.018 * np.sin(xx * 0.45)
        candidate = np.clip(anchor + residual[..., None], 0.0, 1.0)
        candidate[44:84, 44:84] = np.clip(
            candidate[44:84, 44:84] + 0.20, 0.0, 1.0
        )
        local = _roundtrip_confidence_map(anchor, candidate, 1.0)
        maps = ComposeMaps(
            structure=np.zeros((128, 128), np.float32),
            orientation=np.ones((128, 128), np.float32),
            new_edge=np.ones((128, 128), np.float32),
            tile=np.ones((128, 128), np.float32),
            roundtrip=local,
            region=np.ones((128, 128), np.float32),
        )
        gains = FrequencyGains(1.0, 1.0, 1.0, 1.0, 0.8, 0.0)
        composed, _ = FrequencyAwareComposer().compose(
            anchor,
            candidate,
            gains=gains,
            maps=maps,
            structural_lock=0.0,
            low_frequency_lock=1.0,
        )
        raw = np.mean(np.abs(candidate - anchor), axis=2)
        adopted = np.mean(np.abs(composed - anchor), axis=2)
        corrupt_ratio = float(
            np.mean(adopted[48:80, 48:80])
            / max(float(np.mean(raw[48:80, 48:80])), 1e-8)
        )
        safe_ratio = float(
            np.mean(adopted[12:36, 12:36])
            / max(float(np.mean(raw[12:36, 12:36])), 1e-8)
        )
        self.assertLess(float(np.mean(local[48:80, 48:80])), 0.35)
        self.assertGreater(float(np.mean(local[12:36, 12:36])), 0.80)
        self.assertLess(corrupt_ratio, safe_ratio * 0.65)

    def test_roundtrip_accepts_high_frequency_and_detects_shape_change(self):
        source = np.full((64, 64, 3), 0.5, np.float32)
        high = cv2.resize(source, (128, 128), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.mgrid[:128, :128]
        high += (((xx + yy) % 2) * 2 - 1)[..., None] * 0.02
        changed = high.copy()
        changed[:, :50] *= 0.5
        good = roundtrip_metrics(source, np.clip(high, 0, 1))
        bad = roundtrip_metrics(source, np.clip(changed, 0, 1))
        self.assertGreater(good.ssim, bad.ssim)
        self.assertLess(good.low_frequency_mse, bad.low_frequency_mse)
        self.assertLess(good.color_drift, bad.color_drift)

    def test_backprojection_reduces_low_error_and_preserves_high_frequency(self):
        previous = np.zeros((64, 64, 3), np.float32)
        previous[:, 32:] = 0.8
        output = cv2.resize(previous, (128, 128), interpolation=cv2.INTER_LINEAR)
        yy, xx = np.mgrid[:128, :128]
        detail = (((xx + yy) % 2) * 2 - 1)[..., None] * 0.03
        output = np.clip(output * 0.85 + 0.05 + detail, 0, 1)
        before_high = output - cv2.GaussianBlur(output, (0, 0), 1)
        result, report = LowFrequencyBackProjection().apply(output, previous)
        after_high = result - cv2.GaussianBlur(result, (0, 0), 1)
        self.assertLess(report.final_error, report.initial_error)
        self.assertGreater(
            float(np.mean(np.abs(after_high))),
            float(np.mean(np.abs(before_high))) * 0.75,
        )

    def test_seam_analyzer_detects_artificial_boundary(self):
        clean = np.tile(
            np.linspace(0, 1, 128, dtype=np.float32)[None, :, None],
            (128, 1, 3),
        )
        artificial = clean.copy()
        artificial[:, 64:] = np.clip(artificial[:, 64:] + 0.25, 0, 1)
        analyzer = SeamAnalyzer()
        clean_report = analyzer.analyze(clean, [64], [])
        bad_report = analyzer.analyze(artificial, [64], [])
        self.assertGreater(bad_report.ratio, clean_report.ratio)


if __name__ == "__main__":
    unittest.main()
