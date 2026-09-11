from __future__ import annotations

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.retune import RetuneSettings, retune_image  # noqa: E402


def source_image(mode="RGB"):
    yy, xx = np.mgrid[:96, :128]
    rgb = np.zeros((96, 128, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(xx * 2, 0, 255)
    rgb[..., 1] = np.clip(yy * 2, 0, 255)
    rgb[..., 2] = 90
    cv2.line(rgb, (8, 48), (120, 48), (235, 235, 235), 2)
    if mode == "RGBA":
        alpha = np.clip((xx + yy) * 2, 0, 255).astype(np.uint8)
        return Image.fromarray(np.dstack([rgb, alpha]), mode="RGBA")
    return Image.fromarray(rgb, mode="RGB")


def generated_image(mode="RGB"):
    base = np.asarray(source_image("RGB"), dtype=np.int16)
    yy, xx = np.mgrid[:96, :128]
    detail = ((np.sin(xx * 0.65) + np.cos(yy * 0.7)) * 9).astype(np.int16)
    result = np.clip(base + detail[..., None], 0, 255).astype(np.uint8)
    if mode == "RGBA":
        alpha = np.asarray(source_image("RGBA").getchannel("A"))
        return Image.fromarray(np.dstack([result, alpha]), mode="RGBA")
    return Image.fromarray(result, mode="RGB")


class HyperWeaveRetuneTests(unittest.TestCase):
    def test_neutral_is_pixel_exact_and_preserves_metadata(self):
        source = source_image("RGBA")
        generated = generated_image("RGBA")
        generated.info["hyperweave"] = "fixed"
        result, metrics = retune_image(source, generated, RetuneSettings())
        self.assertEqual(result.mode, "RGBA")
        self.assertTrue(np.array_equal(np.asarray(result), np.asarray(generated)))
        self.assertEqual(result.info["hyperweave"], "fixed")
        self.assertTrue(metrics.neutral_fast_path)
        self.assertEqual(metrics.changed_pixel_fraction, 0.0)

    def test_high_detail_adjustment_changes_rgb_but_not_alpha(self):
        source = source_image("RGBA")
        generated = generated_image("RGBA")
        settings = RetuneSettings(high_detail=0.0, tile_size=512)
        result, metrics = retune_image(source, generated, settings)
        self.assertFalse(np.array_equal(np.asarray(result)[..., :3], np.asarray(generated)[..., :3]))
        self.assertTrue(np.array_equal(np.asarray(result)[..., 3], np.asarray(generated)[..., 3]))
        self.assertGreater(metrics.changed_pixel_fraction, 0.0)
        self.assertGreater(metrics.max_absolute_difference, 0)

    def test_white_protection_mask_keeps_generated_pixels(self):
        source = source_image("RGB")
        generated = generated_image("RGB")
        mask = np.zeros((96, 128), dtype=np.uint8)
        mask[:, :64] = 255
        result, _ = retune_image(
            source,
            generated,
            RetuneSettings(high_detail=0.0, mid_detail=0.0, tile_size=512),
            protection_mask=Image.fromarray(mask, mode="L"),
        )
        actual = np.asarray(result)
        expected = np.asarray(generated)
        self.assertTrue(np.array_equal(actual[:, :64], expected[:, :64]))
        self.assertFalse(np.array_equal(actual[:, 64:], expected[:, 64:]))

    def test_tile_boundaries_are_stable(self):
        source = source_image("RGB").resize((1800, 1200), Image.Resampling.LANCZOS)
        generated = generated_image("RGB").resize((1800, 1200), Image.Resampling.LANCZOS)
        settings_a = RetuneSettings(high_detail=0.6, mid_detail=1.2, tile_size=512)
        settings_b = RetuneSettings(high_detail=0.6, mid_detail=1.2, tile_size=1024)
        result_a, _ = retune_image(source, generated, settings_a)
        result_b, _ = retune_image(source, generated, settings_b)
        difference = np.abs(
            np.asarray(result_a, dtype=np.int16) - np.asarray(result_b, dtype=np.int16)
        )
        self.assertLessEqual(int(difference.max()), 1)
        self.assertLess(float(np.mean(difference)), 0.01)

    def test_invalid_settings_fail_closed(self):
        with self.assertRaises(ValueError):
            retune_image(
                source_image(),
                generated_image(),
                RetuneSettings(high_detail=float("nan")),
            )


if __name__ == "__main__":
    unittest.main()
