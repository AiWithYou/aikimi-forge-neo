from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.config import HyperWeaveConfig
from hyperweave.geometry import TilePlanner, plan_upscale_stages
from hyperweave.noise import CoordinateNoiseProvider, derive_seed


class StagePlannerTests(unittest.TestCase):
    def test_1024_to_4096_uses_two_stages(self):
        stages = plan_upscale_stages(
            1024, 1024, 4096, 4096, config=HyperWeaveConfig()
        )
        self.assertEqual([(2048, 2048), (4096, 4096)], [s.target_size for s in stages])

    def test_1024_to_8192_uses_three_stages(self):
        stages = plan_upscale_stages(
            1024, 1024, 8192, 8192, config=HyperWeaveConfig()
        )
        self.assertEqual(
            [(2048, 2048), (4096, 4096), (8192, 8192)],
            [s.target_size for s in stages],
        )

    def test_1536_to_4096_is_progressive_and_exact(self):
        stages = plan_upscale_stages(
            1536, 1024, 4096, 2731, config=HyperWeaveConfig()
        )
        self.assertEqual((3072, 2048), stages[0].target_size)
        self.assertEqual((4096, 2731), stages[-1].target_size)
        for stage in stages:
            self.assertLessEqual(stage.scale_x, 2.0)
            self.assertLessEqual(stage.scale_y, 2.0)
        self.assertAlmostEqual(1536 / 1024, 4096 / 2731, places=3)

    def test_internal_sizes_are_aligned_but_delivery_is_exact(self):
        stages = plan_upscale_stages(
            1664, 2353, 2897, 4096, config=HyperWeaveConfig()
        )
        self.assertEqual((2897, 4096), stages[-1].target_size)
        self.assertEqual(0, stages[-1].processing_size[0] % 8)
        self.assertEqual(0, stages[-1].processing_size[1] % 8)


class TilePlannerTests(unittest.TestCase):
    def test_default_geometry_and_coverage(self):
        planner = TilePlanner(2048, 1536)
        tiles = planner.plan()
        self.assertTrue(np.all(planner.coverage(tiles) > 0))
        self.assertEqual(192, planner.core_size - planner.stride)
        self.assertEqual(768, tiles[1].grid_core_box[0] - tiles[0].grid_core_box[0])
        self.assertEqual(2048, max(tile.core_box[2] for tile in tiles))
        self.assertEqual(1536, max(tile.core_box[3] for tile in tiles))

    def test_weight_sum_is_positive_everywhere(self):
        planner = TilePlanner(2048, 1536)
        weight = np.zeros((1536, 2048), dtype=np.float32)
        for tile in planner.plan():
            x0, y0, x1, y1 = tile.core_box
            weight[y0:y1, x0:x1] += planner.weight_window(tile)
        self.assertGreater(float(weight.min()), 0.0)

    def test_small_and_nonsquare_images_use_fixed_payload(self):
        planner = TilePlanner(
            320,
            192,
            tile_input_size=128,
            core_size=96,
            context_size=16,
            stride=72,
            alignment=8,
        )
        tiles = planner.plan()
        self.assertTrue(np.all(planner.coverage(tiles) > 0))
        for tile in tiles:
            self.assertEqual(128, tile.input_box[2] - tile.input_box[0])
            self.assertEqual(128, tile.input_box[3] - tile.input_box[1])

    def test_unaligned_canvas_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "latent-aligned"):
            TilePlanner(321, 192)


class CoordinateNoiseTests(unittest.TestCase):
    def setUp(self):
        self.planner = TilePlanner(2048, 1024)
        self.tiles = self.planner.plan()

    def _intersection_slices(self, left, right, scale=8):
        x0 = max(left.input_box[0], right.input_box[0])
        y0 = max(left.input_box[1], right.input_box[1])
        x1 = min(left.input_box[2], right.input_box[2])
        y1 = min(left.input_box[3], right.input_box[3])

        def local(tile):
            return (
                (y0 - tile.input_box[1]) // scale,
                (y1 - tile.input_box[1]) // scale,
                (x0 - tile.input_box[0]) // scale,
                (x1 - tile.input_box[0]) // scale,
            )

        return local(left), local(right)

    def test_overlapping_absolute_coordinates_are_bit_exact(self):
        provider = CoordinateNoiseProvider(1234)
        left, right = self.tiles[0], self.tiles[1]
        a = provider.crop_for_tile(
            left,
            stage_index=0,
            pass_name="global",
            candidate_index=0,
            latent_channels=4,
        )
        b = provider.crop_for_tile(
            right,
            stage_index=0,
            pass_name="global",
            candidate_index=0,
            latent_channels=4,
        )
        sa, sb = self._intersection_slices(left, right)
        np.testing.assert_array_equal(
            a[:, sa[0] : sa[1], sa[2] : sa[3]],
            b[:, sb[0] : sb[1], sb[2] : sb[3]],
        )

    def test_namespace_changes_noise(self):
        provider = CoordinateNoiseProvider(1234)
        kwargs = dict(
            latent_width=16,
            latent_height=12,
            latent_channels=4,
            pass_name="global",
            candidate_index=0,
            stage_index=0,
        )
        base = provider.canvas(**kwargs)
        candidate = provider.canvas(**{**kwargs, "candidate_index": 1})
        stage = provider.canvas(**{**kwargs, "stage_index": 1})
        self.assertFalse(np.array_equal(base, candidate))
        self.assertFalse(np.array_equal(base, stage))

    def test_seed_minus_one_resolves_only_once(self):
        provider = CoordinateNoiseProvider(-1)
        first = provider.resolved_seed
        a = provider.canvas(
            stage_index=0,
            pass_name="anchor",
            candidate_index=0,
            latent_width=8,
            latent_height=8,
            latent_channels=4,
        ).copy()
        b = provider.canvas(
            stage_index=0,
            pass_name="anchor",
            candidate_index=0,
            latent_width=8,
            latent_height=8,
            latent_channels=4,
        )
        self.assertEqual(first, provider.resolved_seed)
        np.testing.assert_array_equal(a, b)

    def test_derived_seed_does_not_depend_on_python_hash(self):
        code = (
            "from hyperweave.noise import derive_seed;"
            "print(derive_seed(7,2,'face',3,4))"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(EXTENSION)
        env["PYTHONHASHSEED"] = "1"
        first = subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True
        ).strip()
        env["PYTHONHASHSEED"] = "999"
        second = subprocess.check_output(
            [sys.executable, "-c", code], env=env, text=True
        ).strip()
        self.assertEqual(first, second)
        self.assertEqual(str(derive_seed(7, 2, "face", 3, 4)), first)


if __name__ == "__main__":
    unittest.main()
