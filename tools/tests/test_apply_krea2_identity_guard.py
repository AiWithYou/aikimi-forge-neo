from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from tools.apply_krea2_identity_guard import (
    ProtectionEllipse,
    apply_identity_guard,
    parse_protection_ellipse,
    protection_mask,
    validate_paths,
)


class ProtectionEllipseTests(unittest.TestCase):
    def test_parse_named_ellipse(self):
        region = parse_protection_ellipse("face:10,20,110,140,16")

        self.assertEqual(region.label, "face")
        self.assertEqual(region.box, (10, 20, 110, 140))
        self.assertEqual(region.feather, 16)

    def test_parse_rejects_invalid_geometry(self):
        with self.assertRaisesRegex(ValueError, "x1 > x0"):
            parse_protection_ellipse("face:20,20,10,40,4")

    def test_mask_has_exact_core_smooth_transition_and_zero_exterior(self):
        mask, report = protection_mask(
            (100, 100),
            [ProtectionEllipse("face", (20, 20, 80, 80), 12)],
        )
        values = np.asarray(mask)

        self.assertEqual(int(values[50, 50]), 255)
        self.assertEqual(int(values[10, 10]), 0)
        self.assertGreater(int(values[50, 22]), 0)
        self.assertLess(int(values[50, 22]), 255)
        self.assertGreater(report["reference_exact_pixels"], 0)
        self.assertGreater(report["transition_pixels"], 0)


class IdentityGuardTests(unittest.TestCase):
    def test_exact_core_comes_from_reference_and_exterior_stays_candidate(self):
        candidate = Image.new("RGB", (100, 100), (10, 20, 30))
        reference = Image.new("RGB", (100, 100), (210, 220, 230))

        result, mask, report = apply_identity_guard(
            candidate,
            reference,
            [ProtectionEllipse("face", (20, 20, 80, 80), 12)],
        )

        self.assertEqual(result.getpixel((50, 50)), (210, 220, 230))
        self.assertEqual(result.getpixel((10, 10)), (10, 20, 30))
        self.assertNotEqual(result.getpixel((22, 50)), candidate.getpixel((22, 50)))
        self.assertNotEqual(result.getpixel((22, 50)), reference.getpixel((22, 50)))
        self.assertEqual(mask.getpixel((10, 10)), 0)
        self.assertGreater(report["result_delta_from_candidate"]["changed_pixels"], 0)

    def test_rejects_mismatched_dimensions(self):
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            apply_identity_guard(
                Image.new("RGB", (64, 64)),
                Image.new("RGB", (32, 32)),
                [ProtectionEllipse("face", (8, 8, 56, 56), 8)],
            )

    def test_paths_must_be_distinct(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "distinct"):
                validate_paths(
                    root / "candidate.png",
                    root / "reference.png",
                    root / "output.png",
                    root / "output.png",
                    None,
                )


if __name__ == "__main__":
    unittest.main()
