from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from tools.apply_krea2_identity_guard import ProtectionEllipse
from tools.krea2_b5_tile_regenerate import (
    apply_stage_protection,
    axis_blend_weight,
    low_frequency_detail,
    remove_scoped_tree,
    scale_protection_regions,
    source_gated_detail,
    tile_blend_weight,
    tile_origins,
)


class TilePlanTests(unittest.TestCase):
    def test_balanced_origins_cover_axis_with_complete_tiles(self):
        origins = tile_origins(1448, 256, 64)

        self.assertEqual(origins[0], 0)
        self.assertEqual(origins[-1], 1448 - 256)
        self.assertEqual(origins, sorted(set(origins)))
        self.assertTrue(all(b - a <= 256 - 64 for a, b in zip(origins, origins[1:])))

    def test_default_b5_working_canvas_uses_165_complete_tiles(self):
        self.assertEqual(
            len(tile_origins(2048, 256, 64)) * len(tile_origins(2896, 256, 64)),
            165,
        )

    def test_blend_weight_keeps_outer_canvas_edge_and_feathers_interior(self):
        weight = tile_blend_weight(
            512,
            128,
            x0=0,
            y0=256,
            canvas_size=(2048, 2896),
        )

        self.assertEqual(float(weight[256, 0]), 1.0)
        self.assertLess(float(weight[0, 256]), 0.001)
        self.assertGreater(float(weight[256, 256]), 0.99)

    def test_axis_weight_without_feather_is_one(self):
        np.testing.assert_array_equal(
            axis_blend_weight(
                8,
                0,
                touches_start=False,
                touches_end=False,
            ),
            np.ones(8, dtype=np.float32),
        )


class TileFinishTests(unittest.TestCase):
    def test_source_gated_detail_rejects_uniform_color_shift(self):
        base = Image.new("RGB", (64, 64), (100, 120, 140))
        shifted = Image.new("RGB", (64, 64), (130, 150, 170))

        result, report = source_gated_detail(
            shifted,
            base,
            radius=8,
            gain=1.0,
            max_delta=32.0,
            structure_sigma=12.0,
            base_detail_sigma=2.5,
        )

        np.testing.assert_array_equal(np.asarray(result), np.asarray(base))
        self.assertLess(report["mean_abs_rgb_delta"], 1e-4)

    def test_low_frequency_detail_rejects_uniform_color_shift(self):
        base = Image.new("RGB", (64, 64), (100, 120, 140))
        shifted = Image.new("RGB", (64, 64), (130, 150, 170))

        result, report = low_frequency_detail(
            shifted,
            base,
            sigma=8.0,
            max_delta=16.0,
        )

        np.testing.assert_array_equal(np.asarray(result), np.asarray(base))
        self.assertLess(report["mean_abs_rgb_delta"], 1e-4)

    def test_protection_regions_scale_and_restore_deterministic_base(self):
        regions = [ProtectionEllipse("face", (10, 20, 50, 80), 6)]
        scaled = scale_protection_regions(regions, 2.0)
        self.assertEqual(scaled[0].box, (20, 40, 100, 160))
        self.assertEqual(scaled[0].feather, 12)

        candidate = Image.new("RGB", (160, 200), (200, 210, 220))
        base = Image.new("RGB", (160, 200), (20, 30, 40))
        result, report = apply_stage_protection(candidate, base, scaled)

        self.assertEqual(result.getpixel((60, 100)), (20, 30, 40))
        self.assertEqual(result.getpixel((150, 190)), (200, 210, 220))
        self.assertIsNotNone(report)

    def test_scoped_cleanup_rejects_wrong_parent(self):
        with self.assertRaises(RuntimeError):
            remove_scoped_tree(
                Path("H:/AI/example/child"),
                expected_parent=Path("H:/AI/different"),
            )

    def test_scoped_cleanup_removes_only_exact_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            child = parent / "child"
            sibling = parent / "sibling"
            child.mkdir()
            sibling.mkdir()
            (child / "generated.tmp").write_text("temporary", encoding="utf-8")

            remove_scoped_tree(child, expected_parent=parent)

            self.assertFalse(child.exists())
            self.assertTrue(sibling.is_dir())


if __name__ == "__main__":
    unittest.main()
