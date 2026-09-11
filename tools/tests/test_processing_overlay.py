import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "modules_forge" / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import modules.shared_init as shared_init

shared_init.initialize()
sys.argv = _ORIGINAL_ARGV

from modules import processing
from modules.processing import (
    _INPAINT_FULL_RES_OVERLAY_MASK,
    _scale_inpaint_crop_region,
    apply_overlay,
)


class InpaintOverlayTests(unittest.TestCase):
    def test_scaled_crop_region_is_stable_across_batch_images(self):
        crop_region = (2, 3, 6, 8)

        first = _scale_inpaint_crop_region(crop_region, 2.0)
        second = _scale_inpaint_crop_region(crop_region, 2.0)

        self.assertEqual(first, (4, 6, 12, 16))
        self.assertEqual(second, first)
        self.assertEqual(crop_region, (2, 3, 6, 8))

    def test_scaled_crop_mask_matches_scaled_paste_region(self):
        overlay = Image.new("RGB", (20, 24), (0, 0, 255))
        full_mask = Image.new("L", overlay.size, 255)
        crop_region = _scale_inpaint_crop_region((2, 3, 6, 8), 2.0)
        x1, y1, x2, y2 = crop_region
        crop_mask = full_mask.crop(crop_region)
        overlay.info[_INPAINT_FULL_RES_OVERLAY_MASK] = crop_mask

        generated = Image.new("RGB", crop_mask.size, (255, 0, 0))
        result, _ = apply_overlay(
            generated,
            (x1, y1, x2 - x1, y2 - y1),
            overlay,
        )

        self.assertEqual(crop_mask.size, (x2 - x1, y2 - y1))
        self.assertEqual(result.size, overlay.size)

    def test_crop_overlay_keeps_original_pixels_outside_mask(self):
        generated = Image.new("RGB", (4, 4), (255, 0, 0))
        overlay = Image.new("RGB", (8, 8), (0, 0, 255))

        crop_mask = Image.new("L", (4, 4), 0)
        crop_mask.putpixel((1, 1), 255)
        overlay.info[_INPAINT_FULL_RES_OVERLAY_MASK] = crop_mask

        result, original_denoised = apply_overlay(generated, (2, 2, 4, 4), overlay)

        self.assertEqual(result.size, (8, 8))
        self.assertEqual(original_denoised.size, (8, 8))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
        self.assertEqual(result.getpixel((2, 2)), (0, 0, 255))
        self.assertEqual(result.getpixel((3, 3)), (255, 0, 0))

    def test_crop_overlay_rejects_mask_size_mismatch(self):
        generated = Image.new("RGB", (4, 4), (255, 0, 0))
        overlay = Image.new("RGB", (8, 8), (0, 0, 255))
        overlay.info[_INPAINT_FULL_RES_OVERLAY_MASK] = Image.new("L", (3, 4), 255)

        with self.assertRaisesRegex(ValueError, "does not match paste size"):
            apply_overlay(generated, (2, 2, 4, 4), overlay)


class OverrideSettingsTests(unittest.TestCase):
    def test_capture_includes_default_for_setting_omitted_from_config(self):
        class FakeOptions:
            data = {}
            data_labels = {"img2img_fix_steps": object()}

            @staticmethod
            def get_default(key):
                return False if key == "img2img_fix_steps" else None

        original_opts = processing.opts
        processing.opts = FakeOptions()
        try:
            captured = processing._capture_override_settings(
                {"img2img_fix_steps": True, "unknown_extension_option": 7}
            )
        finally:
            processing.opts = original_opts

        self.assertEqual(captured, {"img2img_fix_steps": False})


class HiresModuleSwitchTests(unittest.TestCase):
    def test_unspecified_modules_keep_the_current_choices(self):
        with patch.object(processing.main_entry, "modules_change") as modules_change:
            self.assertFalse(processing._change_hires_modules_if_needed(None))

        modules_change.assert_not_called()

    def test_empty_modules_select_the_builtin_hires_vae(self):
        with patch.object(
            processing.main_entry,
            "modules_change",
            return_value=True,
        ) as modules_change:
            self.assertTrue(processing._change_hires_modules_if_needed([]))

        modules_change.assert_called_once_with(
            [],
            preset=None,
            save=False,
            refresh=False,
        )

    def test_same_choices_marker_keeps_the_current_choices(self):
        with patch.object(processing.main_entry, "modules_change") as modules_change:
            self.assertFalse(
                processing._change_hires_modules_if_needed(["Use same choices"])
            )

        modules_change.assert_not_called()

    def test_explicit_modules_request_a_temporary_switch(self):
        selected = ["qwen_image_vae.safetensors", "qwen3vl_4b.safetensors"]
        with patch.object(
            processing.main_entry,
            "modules_change",
            return_value=True,
        ) as modules_change:
            self.assertTrue(processing._change_hires_modules_if_needed(selected))

        modules_change.assert_called_once_with(
            selected,
            preset=None,
            save=False,
            refresh=False,
        )


if __name__ == "__main__":
    unittest.main()
