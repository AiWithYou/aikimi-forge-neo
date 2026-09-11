from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from tools.krea2_tiled_refine import (
    axis_weights,
    low_frequency_box_blur,
    match_low_frequency_chroma,
    match_low_frequency_lab_chroma,
    pad_tile_for_diffusion,
    rgb_to_lab_float,
    target_size_from_args,
    tile_positions,
    tile_weight_mask,
    validate_args,
)


def valid_args(**overrides):
    values = {
        "width": None,
        "height": None,
        "long_edge": 5760,
        "scale": 1.5,
        "tile_size": 1024,
        "overlap": 192,
        "steps": 4,
        "timeout": 1800,
        "max_output_pixels": 24_000_000,
        "max_tile_pixels": 1_638_400,
        "denoise": 0.10,
        "cfg": 1.0,
        "distilled_cfg": 1.15,
        "progress_interval": 20.0,
        "no_progress_timeout": 600.0,
        "color_match_radius": 32,
        "color_match_max_shift": 4.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ValidationTests(unittest.TestCase):
    def test_accepts_default_arguments(self):
        validate_args(valid_args())

    def test_rejects_single_explicit_dimension(self):
        with self.assertRaisesRegex(ValueError, "--width and --height"):
            validate_args(valid_args(width=5760))

    def test_rejects_overlap_at_tile_size(self):
        with self.assertRaisesRegex(ValueError, "--overlap must be < --tile-size"):
            validate_args(valid_args(overlap=1024))

    def test_rejects_large_tile(self):
        with self.assertRaisesRegex(ValueError, "--tile-size exceeds"):
            validate_args(valid_args(tile_size=2048))

    def test_rejects_tile_whose_aligned_payload_exceeds_limit(self):
        with self.assertRaisesRegex(ValueError, "--tile-size exceeds"):
            validate_args(valid_args(tile_size=1279, max_tile_pixels=1279 * 1279))

    def test_rejects_no_progress_timeout_without_progress_polling(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout requires"):
            validate_args(valid_args(no_progress_timeout=120, progress_interval=0))

    def test_rejects_non_finite_color_match_shift(self):
        with self.assertRaisesRegex(ValueError, "--color-match-max-shift"):
            validate_args(valid_args(color_match_max_shift=float("nan")))

    def test_rejects_color_match_radius_larger_than_tile(self):
        with self.assertRaisesRegex(ValueError, "--color-match-radius"):
            validate_args(valid_args(color_match_radius=1025))


class TargetSizeTests(unittest.TestCase):
    def test_uses_long_edge_when_set(self):
        self.assertEqual(
            target_size_from_args(3840, 1664, valid_args(long_edge=5760)), (5760, 2496)
        )

    def test_uses_scale_when_long_edge_is_not_set(self):
        self.assertEqual(
            target_size_from_args(3840, 1664, valid_args(long_edge=None, scale=1.5)),
            (5760, 2496),
        )

    def test_rejects_target_above_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeding --max-output-pixels"):
            target_size_from_args(
                3840, 1664, valid_args(long_edge=7680, max_output_pixels=10_000_000)
            )


class TileLayoutTests(unittest.TestCase):
    def test_single_tile_when_length_fits(self):
        self.assertEqual(tile_positions(900, 1024, 192), [0])

    def test_positions_include_end(self):
        self.assertEqual(tile_positions(2496, 1024, 192), [0, 832, 1472])

    def test_axis_weights_feather_edges(self):
        weights = axis_weights(8, 3, 3)
        self.assertEqual(weights[0], 0.0)
        self.assertEqual(weights[-1], 0.0)
        self.assertEqual(weights[3], 1.0)

    def test_axis_weights_use_smoothstep_not_linear_feather(self):
        weights = axis_weights(7, 4, 0)
        np.testing.assert_allclose(
            weights[:4], [0.0, 7.0 / 27.0, 20.0 / 27.0, 1.0], atol=1e-6
        )

    def test_neighboring_smoothstep_feathers_are_complementary(self):
        left = axis_weights(7, 0, 4)[-4:]
        right = axis_weights(7, 4, 0)[:4]
        np.testing.assert_allclose(left + right, np.ones(4), atol=1e-6)

    def test_weight_mask_shape(self):
        x_positions = [0, 832, 1472]
        y_positions = [0, 832]
        mask = tile_weight_mask(x_positions, y_positions, 1, 0, 1024, 1024)
        self.assertEqual(mask.shape, (1024, 1024))
        self.assertTrue(np.all(mask >= 0))
        self.assertTrue(np.all(mask <= 1))

    def test_unaligned_tile_is_edge_padded_for_diffusion(self):
        values = np.arange(777 * 1001 * 3, dtype=np.uint8).reshape(777, 1001, 3)
        tile = Image.fromarray(values, "RGB")

        padded = pad_tile_for_diffusion(tile)
        padded_values = np.asarray(padded)

        self.assertEqual(padded.size, (1008, 784))
        np.testing.assert_array_equal(padded_values[:777, :1001], values)
        np.testing.assert_array_equal(
            padded_values[776, 1001:], np.repeat(values[776, -1:, :], 7, axis=0)
        )
        np.testing.assert_array_equal(
            padded_values[777:, 1000], np.repeat(values[-1:, 1000, :], 7, axis=0)
        )

    def test_aligned_tile_keeps_size_and_pixels(self):
        tile = Image.new("RGB", (256, 272), (10, 20, 30))

        padded = pad_tile_for_diffusion(tile)

        self.assertEqual(padded.size, tile.size)
        np.testing.assert_array_equal(np.asarray(padded), np.asarray(tile))


class ColorMatchTests(unittest.TestCase):
    def test_low_frequency_blur_preserves_constant_array(self):
        values = np.full((3, 5, 2), [2.5, -7.0], dtype=np.float32)
        np.testing.assert_allclose(low_frequency_box_blur(values, 3), values)

    def test_lab_match_preserves_l_and_clips_chroma_correction(self):
        refined = np.zeros((4, 5, 3), dtype=np.float32)
        refined[..., 0] = np.arange(20, dtype=np.float32).reshape(4, 5)
        refined[..., 1] = 12.0
        refined[..., 2] = -5.0
        base = refined.copy()
        base[..., 1] = -20.0
        base[..., 2] = 19.0

        matched = match_low_frequency_lab_chroma(
            refined, base, radius=2, max_chroma_shift=4.0
        )

        np.testing.assert_array_equal(matched[..., 0], refined[..., 0])
        shifts = np.linalg.norm(matched[..., 1:3] - refined[..., 1:3], axis=2)
        np.testing.assert_allclose(shifts, 4.0, atol=1e-5)

    def test_rgb_match_moves_chroma_toward_base_without_large_l_shift(self):
        refined = np.full((12, 12, 3), [190, 105, 85], dtype=np.uint8)
        base = np.full((12, 12, 3), [170, 120, 95], dtype=np.uint8)

        matched = match_low_frequency_chroma(
            refined, base, radius=2, max_chroma_shift=4.0
        )
        refined_lab = rgb_to_lab_float(refined)
        base_lab = rgb_to_lab_float(base)
        matched_lab = rgb_to_lab_float(matched)

        before = np.linalg.norm(refined_lab[..., 1:3] - base_lab[..., 1:3])
        after = np.linalg.norm(matched_lab[..., 1:3] - base_lab[..., 1:3])
        self.assertLess(after, before)
        self.assertLess(
            float(np.max(np.abs(matched_lab[..., 0] - refined_lab[..., 0]))), 0.5
        )


if __name__ == "__main__":
    unittest.main()
