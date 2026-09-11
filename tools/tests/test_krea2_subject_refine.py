from types import SimpleNamespace
import unittest

from tools.krea2_subject_refine import (
    RegionBox,
    build_feather_mask,
    expand_box,
    parse_normalized_box,
    parse_pixel_box,
    process_size_for_crop,
    resolve_boxes,
    validate_args,
)


def valid_args(**overrides):
    values = {
        "box": ["10,20,110,220"],
        "box_normalized": None,
        "padding": 96,
        "feather": 96,
        "process_long_edge": 1536,
        "max_process_pixels": 1536 * 1536,
        "steps": 4,
        "timeout": 1800,
        "denoise": 0.10,
        "cfg": 1.0,
        "distilled_cfg": 1.15,
        "progress_interval": 20.0,
        "no_progress_timeout": 600.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class ArgumentValidationTests(unittest.TestCase):
    def test_accepts_default_arguments(self):
        validate_args(valid_args())

    def test_rejects_missing_region(self):
        with self.assertRaisesRegex(ValueError, "Pass at least one"):
            validate_args(valid_args(box=None))

    def test_rejects_mixed_region_modes(self):
        with self.assertRaisesRegex(ValueError, "either --box or --box-normalized"):
            validate_args(valid_args(box_normalized=["0.1,0.2,0.3,0.4"]))

    def test_rejects_no_progress_timeout_without_progress_polling(self):
        with self.assertRaisesRegex(ValueError, "--no-progress-timeout requires"):
            validate_args(valid_args(no_progress_timeout=120, progress_interval=0))

    def test_rejects_large_denoise(self):
        with self.assertRaisesRegex(ValueError, "--denoise"):
            validate_args(valid_args(denoise=1.1))


class BoxParsingTests(unittest.TestCase):
    def test_parses_pixel_box(self):
        self.assertEqual(
            parse_pixel_box("10,20,110,220", 500, 300),
            RegionBox(10, 20, 110, 220),
        )

    def test_rejects_pixel_box_outside_image(self):
        with self.assertRaisesRegex(ValueError, "fit inside"):
            parse_pixel_box("10,20,600,220", 500, 300)

    def test_parses_normalized_box(self):
        self.assertEqual(
            parse_normalized_box("0.10,0.20,0.30,0.60", 1000, 500),
            RegionBox(100, 100, 300, 300),
        )

    def test_rejects_normalized_box_outside_unit_interval(self):
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            parse_normalized_box("0.1,0.2,1.2,0.6", 1000, 500)

    def test_resolves_multiple_boxes(self):
        args = valid_args(box=["10,20,110,220", "120,20,220,220"])
        self.assertEqual(
            resolve_boxes(args, 500, 300),
            [RegionBox(10, 20, 110, 220), RegionBox(120, 20, 220, 220)],
        )


class RegionProcessingTests(unittest.TestCase):
    def test_expands_box_and_clamps_to_image(self):
        self.assertEqual(
            expand_box(RegionBox(20, 30, 120, 230), 64, 500, 300),
            RegionBox(0, 0, 184, 294),
        )

    def test_process_size_uses_requested_long_edge(self):
        self.assertEqual(
            process_size_for_crop(300, 600, 1024, 1024 * 1024), (512, 1024)
        )

    def test_rejects_process_size_above_pixel_limit(self):
        with self.assertRaisesRegex(ValueError, "exceeding --max-process-pixels"):
            process_size_for_crop(1200, 1200, 1536, 1_000_000)

    def test_rectangle_feather_mask_has_soft_edges(self):
        mask = build_feather_mask((128, 96), 24, "rectangle")
        self.assertEqual(mask.size, (128, 96))
        self.assertLess(mask.getpixel((0, 0)), mask.getpixel((64, 48)))

    def test_ellipse_mask_keeps_corners_transparent(self):
        mask = build_feather_mask((128, 96), 16, "ellipse")
        self.assertEqual(mask.getpixel((0, 0)), 0)
        self.assertGreater(mask.getpixel((64, 48)), 200)


if __name__ == "__main__":
    unittest.main()
