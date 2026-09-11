import unittest

import cv2
import numpy as np
from PIL import Image, ImageCms

from modules.grain_cleaner import GrainSettings, clean_grain, gaussian, prepare_image


class GrainCleanerTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        pixels = np.clip(128 + rng.normal(0, 4, (160, 160, 3)), 0, 255).astype(np.uint8)
        self.image = Image.fromarray(pixels)

    def test_zero_and_protected_pixels_are_exact(self):
        for kwargs in (
            {"settings": GrainSettings(strength=0)},
            {"protect_mask": Image.new("L", self.image.size, 255)},
            {"apply_mask": Image.new("L", self.image.size, 0)},
        ):
            out, _, _ = clean_grain(self.image, **kwargs)
            np.testing.assert_array_equal(out, self.image)
        mask = np.zeros((160, 160), np.uint8)
        mask[:, :80] = 255
        for key, area in (("protect_mask", np.s_[:, :80]), ("apply_mask", np.s_[:, 80:])):
            out, report, _ = clean_grain(self.image, GrainSettings(strength=1), **{key: Image.fromarray(mask)})
            np.testing.assert_array_equal(np.asarray(out)[area], np.asarray(self.image)[area])
            self.assertGreater(report["changed_fraction"], 0)

    def test_flat_and_small_auto_images_are_unchanged(self):
        for image in (Image.new("RGB", (160, 160), (100, 128, 140)), self.image.crop((0, 0, 64, 64))):
            out, report, _ = clean_grain(image)
            np.testing.assert_array_equal(out, image)
            self.assertIn(report["status"], ("LOW_ACTIVITY", "INSUFFICIENT_REFERENCE"))

    def test_grain_reduces_and_line_is_preserved(self):
        pixels = np.asarray(self.image).copy()
        pixels[:, 100:102] = 20
        out, report, views = clean_grain(Image.fromarray(pixels), GrainSettings(strength=0.75), diagnostics=True)
        result = np.asarray(out)
        self.assertLess(result[:, :64].astype(float).std(), pixels[:, :64].astype(float).std())
        self.assertLess(np.abs(result[:, 100:102].astype(float) - pixels[:, 100:102]).max(), 2)
        self.assertEqual(set(views), {"removed_x8", "suppression", "protection", "support"})
        self.assertEqual(report["status"], "APPLIED")

    def test_sample_and_output_diagnostics_match(self):
        settings = GrainSettings(mode="sample")
        first, _, _ = clean_grain(self.image, settings, sample_roi=(0, 0, 64, 64))
        second, _, views = clean_grain(self.image, settings, sample_roi=(0, 0, 64, 64), diagnostics=True)
        np.testing.assert_array_equal(first, second)
        difference = np.clip(128 + 8 * (np.asarray(self.image).astype(float) - np.asarray(first)), 0, 255)
        np.testing.assert_array_equal(views["removed_x8"], difference.astype(np.uint8))

    def test_validation(self):
        for image, kwargs in (
            (self.image.convert("L"), {}),
            (Image.new("RGBA", (64, 64), (1, 2, 3, 0)), {}),
            (self.image, {"protect_mask": Image.new("L", (64, 64))}),
            (self.image, {"settings": GrainSettings(grain_scale=float("nan"))}),
            (self.image, {"settings": GrainSettings(mode="sample"), "sample_roi": (0, 0, 31, 64)}),
        ):
            with self.assertRaises(ValueError):
                clean_grain(image, **kwargs)

    def test_orientation_icc_alpha_and_reconstruction(self):
        image = self.image.crop((0, 0, 128, 96))
        image.getexif()[274] = 6
        baseline, _ = prepare_image(image)
        self.assertEqual(baseline.size, (96, 128))
        self.assertNotIn(274, baseline.getexif())
        image.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        _, status = prepare_image(image)
        self.assertEqual(status, "ICC converted to sRGB")
        rgba = self.image.convert("RGBA")
        out, _, _ = clean_grain(rgba)
        self.assertEqual(out.mode, "RGBA")
        self.assertEqual(out.getchannel("A").getextrema(), (255, 255))
        x = cv2.cvtColor(np.asarray(self.image).astype(np.float32) / 255, cv2.COLOR_RGB2Lab)
        b1, b2, b3 = (gaussian(x, s) for s in (0.8, 1.6, 3.2))
        np.testing.assert_allclose((x - b1) + (b1 - b2) + (b2 - b3) + b3, x, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
