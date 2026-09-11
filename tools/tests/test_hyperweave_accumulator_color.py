from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.accumulator import InMemoryAccumulator, MemmapAccumulator
from hyperweave.color import (
    image_to_linear_rgb,
    linear_to_srgb,
    resize_linear_rgb,
    srgb_to_linear,
)


class AccumulatorTests(unittest.TestCase):
    def _populate(self, accumulator, reverse=False):
        base = accumulator.base
        operations = [
            ((0, 0, 4, 4), np.full((4, 4, 3), 0.6, np.float32), np.ones((4, 4), np.float32)),
            ((2, 0, 6, 4), np.full((4, 4, 3), 0.8, np.float32), np.full((4, 4), 2.0, np.float32)),
        ]
        if reverse:
            operations.reverse()
        for box, generated, weight in operations:
            x0, y0, x1, y1 = box
            accumulator.add(box, generated, base[y0:y1, x0:x1], weight)

    def test_known_weighted_average_and_variance(self):
        base = np.full((4, 6, 3), 0.2, dtype=np.float32)
        accumulator = InMemoryAccumulator(base)
        self._populate(accumulator)
        result = accumulator.finalize()
        expected_overlap = 0.2 + ((0.4 * 1.0 + 0.6 * 2.0) / 3.0)
        np.testing.assert_allclose(result.candidate[:, 2:4], expected_overlap, atol=1e-6)
        self.assertTrue(np.all(result.variance >= 0))
        self.assertTrue(np.isfinite(result.candidate).all())
        accumulator.close()

    def test_ram_and_memmap_match_and_cleanup(self):
        base = np.full((4, 6, 3), 0.2, dtype=np.float32)
        memory = InMemoryAccumulator(base)
        self._populate(memory)
        memory_result = memory.finalize()
        with tempfile.TemporaryDirectory() as temporary:
            disk = MemmapAccumulator(base, temporary, "test")
            self._populate(disk)
            disk_result = disk.finalize()
            paths = list(disk.paths.values())
            np.testing.assert_allclose(memory_result.candidate, disk_result.candidate)
            np.testing.assert_allclose(memory_result.variance, disk_result.variance)
            disk.cleanup()
            self.assertFalse(any(path.exists() for path in paths))
        memory.close()

    def test_tile_commit_order_is_stable_for_known_values(self):
        base = np.full((4, 6, 3), 0.2, dtype=np.float32)
        first = InMemoryAccumulator(base)
        second = InMemoryAccumulator(base)
        self._populate(first, reverse=False)
        self._populate(second, reverse=True)
        np.testing.assert_array_equal(first.finalize().candidate, second.finalize().candidate)

    def test_uncovered_pixels_fail(self):
        base = np.zeros((4, 4, 3), dtype=np.float32)
        accumulator = InMemoryAccumulator(base)
        accumulator.add(
            (0, 0, 2, 2),
            np.zeros((2, 2, 3), np.float32),
            np.zeros((2, 2, 3), np.float32),
            np.ones((2, 2), np.float32),
        )
        with self.assertRaisesRegex(RuntimeError, "uncovered"):
            accumulator.finalize()


class ColorTests(unittest.TestCase):
    def test_srgb_linear_roundtrip(self):
        values = np.linspace(0, 1, 4096, dtype=np.float32)
        restored = linear_to_srgb(srgb_to_linear(values))
        np.testing.assert_allclose(values, restored, atol=2e-6)

    def test_rgba_alpha_and_visible_color_survive_resize(self):
        array = np.zeros((16, 16, 4), dtype=np.uint8)
        array[:, :8, :3] = (255, 32, 16)
        array[:, :8, 3] = 255
        array[:, 8:, :3] = (0, 0, 255)  # hidden RGB must not bleed
        image = Image.fromarray(array, mode="RGBA")
        resized = resize_linear_rgb(image, (64, 64))
        self.assertEqual("RGBA", resized.mode)
        result = np.asarray(resized)
        self.assertEqual(0, int(result[:, -1, 3].max()))
        visible = result[:, :28, :3]
        self.assertGreater(float(visible[..., 0].mean()), 220)
        self.assertLess(float(visible[..., 2].mean()), 40)

    def test_image_to_linear_separates_alpha(self):
        image = Image.new("RGBA", (5, 7), (12, 34, 56, 128))
        rgb, alpha = image_to_linear_rgb(image)
        self.assertEqual((7, 5, 3), rgb.shape)
        self.assertEqual((7, 5), alpha.shape)
        self.assertAlmostEqual(128 / 255, float(alpha.mean()), places=6)


if __name__ == "__main__":
    unittest.main()
