"""Offline geometry, alpha preservation, and real Gradio callback contracts."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from modules_forge.qwen_image21.outpaint import MAX_CANVAS_PIXELS, PROMPT, Plan, normalize_image, prepare, recipe, stitch

ROOT = Path(__file__).resolve().parents[2]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.source = Image.new("RGB", (256, 256), (12, 34, 56))

    def test_all_fifteen_nonempty_direction_combinations(self):
        for bits in range(1, 16):
            with self.subTest(directions=bits):
                pads = tuple(32 if bits & (1 << i) else 0 for i in range(4))
                original, canvas, plan = prepare(self.source, *pads)
                self.assertEqual(tuple((plan.left, plan.top, plan.right, plan.bottom)), pads)
                self.assertEqual(canvas.crop(plan.box).tobytes(), original.tobytes())
                self.assertEqual(canvas.size, plan.size)
                corner = (0, 0) if plan.left or plan.top else (canvas.width - 1, canvas.height - 1)
                self.assertEqual(canvas.getpixel(corner), (128, 128, 128))

    def test_one_sided_alignment_never_enables_opposite_side(self):
        _, _, plan = prepare(self.source, 1, 0, 0, 0)
        self.assertEqual((plan.left, plan.top, plan.right, plan.bottom), (32, 0, 0, 0))

    def test_two_sided_alignment_splits_extra(self):
        _, _, plan = prepare(self.source, 1, 0, 1, 0)
        self.assertEqual((plan.left, plan.right), (16, 16))

    def test_unaligned_source_is_not_resized(self):
        original, padded, plan = prepare(Image.new("RGB", (257, 259)), 1, 1, 1, 1)
        self.assertEqual(original.size, (257, 259))
        self.assertEqual(plan.size, (288, 288))
        self.assertEqual(padded.crop(plan.box).size, original.size)

    def test_unextended_unaligned_axis_has_actionable_error(self):
        with self.assertRaisesRegex(ValueError, "高さ.*32"):
            prepare(Image.new("RGB", (256, 257)), 32, 0, 0, 0)

    def test_zero_padding_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.source, 0, 0, 0, 0)

    def test_invalid_padding_values(self):
        for value in (-1, 1.5, True, "32", None, math.nan, math.inf, 10**1000):
            with self.subTest(value=str(value)[:32]), self.assertRaises(ValueError):
                prepare(self.source, value, 0, 0, 0)

    def test_integral_gradio_floats_normalized(self):
        _, _, plan = prepare(self.source, 32.0, 0.0, 0.0, 0.0)
        self.assertIs(type(plan.left), int)

    def test_large_canvas_rejected_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "2 MP"):
            prepare(self.source, 1024, 1024, 1024, 1024)

    def test_axis_limit_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.source, 4096, 0, 0, 0)

    def test_minimum_canvas_size(self):
        with self.assertRaises(ValueError):
            prepare(Image.new("RGB", (16, 16)), 1, 1, 1, 1)

    def test_snapshot_is_independent(self):
        original, padded, plan = prepare(self.source)
        self.source.putpixel((0, 0), (255, 0, 0))
        self.assertEqual(original.getpixel((0, 0)), (12, 34, 56))
        self.assertEqual(padded.getpixel((plan.left, plan.top)), (12, 34, 56))

    def test_missing_input(self):
        with self.assertRaises(ValueError):
            prepare(None)

    def test_source_pixel_limit_without_large_allocation(self):
        image = Image.new("RGB", (1, 1))
        with patch.object(type(image), "width", new_callable=unittest.mock.PropertyMock, return_value=50_000_000):
            with self.assertRaises(ValueError):
                normalize_image(image)

    def test_unsupported_bit_depth_and_color_modes_are_not_silently_lost(self):
        for mode in ("I", "I;16", "F", "CMYK"):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "16-bit"):
                normalize_image(Image.new(mode, (256, 256)))

    def test_exif_orientation_is_applied_once(self):
        source = Image.new("RGB", (256, 288))
        source.getexif()[274] = 6
        original, _, plan = prepare(source, 32, 0, 0, 0)
        self.assertEqual(original.size, (288, 256))
        self.assertEqual(normalize_image(original).size, original.size)
        self.assertEqual(plan.size, (320, 256))

    def test_animated_image_rejected(self):
        stream = io.BytesIO()
        self.source.save(stream, format="GIF", save_all=True, append_images=[Image.new("RGB", (256, 256), "red")])
        stream.seek(0)
        with Image.open(stream) as animated, self.assertRaises(ValueError):
            normalize_image(animated)

    def test_palette_transparency_retained_in_snapshot(self):
        source = Image.new("P", (256, 256))
        source.info["transparency"] = 0
        original, padded, plan = prepare(source)
        self.assertEqual(original.mode, "RGBA")
        self.assertEqual(original.getpixel((0, 0))[3], 0)
        self.assertEqual(padded.getpixel((plan.left, plan.top)), (128, 128, 128))


class StitchTests(unittest.TestCase):
    def setUp(self):
        self.source, _, self.plan = prepare(Image.new("RGB", (256, 256), (19, 101, 202)), 32, 0, 0, 0)
        self.generated = Image.new("RGB", self.plan.size, (220, 30, 40))

    def test_zero_feather_exact_pixels(self):
        result = stitch(self.source, self.generated, self.plan, 0)
        self.assertEqual(result.crop(self.plan.box).tobytes(), self.source.tobytes())
        self.assertEqual(result.crop((0, 0, 32, 256)).tobytes(), self.generated.crop((0, 0, 32, 256)).tobytes())

    def test_positive_feather_changes_only_extended_edge(self):
        result = stitch(self.source, self.generated, self.plan, 32)
        self.assertEqual(result.getpixel((32, 0)), (220, 30, 40))
        self.assertEqual(result.getpixel((64, 0)), self.source.getpixel((32, 0)))
        self.assertEqual(result.getpixel((287, 255)), self.source.getpixel((255, 255)))
        self.assertNotEqual(result.getpixel((48, 0)), self.source.getpixel((16, 0)))

    def test_all_directions_keep_exact_core(self):
        for bits in range(1, 16):
            with self.subTest(directions=bits):
                source, _, plan = prepare(self.source, *(32 if bits & (1 << i) else 0 for i in range(4)))
                result = stitch(source, Image.new("RGB", plan.size), plan, 16)
                core = result.crop((plan.left + 16, plan.top + 16, plan.left + 240, plan.top + 240))
                self.assertEqual(core.tobytes(), source.crop((16, 16, 240, 240)).tobytes())

    def test_wrong_result_size_not_silently_resized(self):
        with self.assertRaisesRegex(ValueError, "自動リサイズ"):
            stitch(self.source, Image.new("RGB", (256, 256)), self.plan)

    def test_wrong_source_size_rejected(self):
        with self.assertRaises(ValueError):
            stitch(Image.new("RGB", (128, 128)), self.generated, self.plan)

    def test_missing_plan_or_result(self):
        with self.assertRaises(ValueError):
            stitch(self.source, self.generated, None)
        with self.assertRaises(ValueError):
            stitch(self.source, None, self.plan)

    def test_bad_feather(self):
        for value in (-1, 128, 1.5, True, math.nan, math.inf, "32"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                stitch(self.source, self.generated, self.plan, value)

    def test_rgba_original_exact_including_hidden_rgb(self):
        for alpha in (0, 1, 33, 128, 254, 255):
            with self.subTest(alpha=alpha):
                source, _, plan = prepare(Image.new("RGBA", (256, 256), (31, 75, 211, alpha)), 32, 0, 0, 0)
                generated = Image.new("RGBA", plan.size, (113, 219, 33, 101))
                result = stitch(source, generated, plan, 32)
                self.assertEqual(result.getpixel((80, 80)), (31, 75, 211, alpha))
                self.assertEqual(result.getpixel((32, 80)), generated.getpixel((32, 80)))
                self.assertEqual(result.getpixel((0, 80)), generated.getpixel((0, 80)))
                exact = stitch(source, generated, plan, 0)
                self.assertEqual(exact.crop(plan.box).tobytes(), source.tobytes())

    def test_opaque_source_over_rgba_result(self):
        result = stitch(self.source, Image.new("RGBA", self.plan.size, (0, 0, 0, 0)), self.plan, 0)
        self.assertEqual(result.mode, "RGBA")
        self.assertEqual(result.getpixel((32, 0)), (19, 101, 202, 255))
        self.assertEqual(result.getpixel((0, 0)), (0, 0, 0, 0))

    def test_inputs_are_not_mutated(self):
        before = self.generated.tobytes()
        stitch(self.source, self.generated, self.plan)
        self.assertEqual(self.generated.tobytes(), before)

    def test_tampered_plan_rejected(self):
        for plan in (replace(self.plan, left=1), replace(self.plan, right=2048, bottom=2048)):
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                stitch(self.source, self.generated, plan)


class RecipeTests(unittest.TestCase):
    def setUp(self):
        _, _, self.plan = prepare(Image.new("RGB", (256, 256)))

    def test_external_recipe_and_settings(self):
        result = recipe(self.plan)
        self.assertEqual(result["comfyui"]["steps"], 25)
        self.assertFalse(result["comfyui"]["latent_noise_mask"])
        self.assertEqual(result["comfyui"]["reference_resolution"], 0)
        self.assertIn("v2.safetensors", result["weights"])
        self.assertIn("別途ComfyUI", result["warnings"][0])
        json.dumps(result, allow_nan=False)

    def test_optional_scene_follows_instruction(self):
        self.assertEqual(recipe(self.plan)["prompt"], PROMPT)
        self.assertEqual(recipe(self.plan, scene="  mountains  ")["prompt"], PROMPT + "\nScene: mountains")

    def test_unknown_version_and_invalid_scene_rejected(self):
        for version in ("v3", None, []):
            with self.subTest(version=version), self.assertRaises(ValueError):
                recipe(self.plan, version)
        for scene in (None, "x" * 10001):
            with self.assertRaises(ValueError):
                recipe(self.plan, scene=scene)

    def test_training_coverage_warnings(self):
        _, _, large = prepare(Image.new("RGB", (1024, 1024)), 32, 0, 0, 0)
        self.assertLessEqual(math.prod(large.size), MAX_CANVAS_PIXELS)
        self.assertEqual(len(recipe(large, "v1")["warnings"]), 2)
        _, _, tiny = prepare(Image.new("RGB", (64, 64)), 128, 128, 128, 128)
        self.assertIn("15%", recipe(tiny)["warnings"][1])


class UIContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        callbacks = types.SimpleNamespace(on_ui_tabs=lambda fn: None)
        fake_modules = types.ModuleType("modules")
        fake_modules.script_callbacks = callbacks
        path = ROOT / "extensions-builtin/qwen-image21-studio/scripts/qwen_image21_outpaint.py"
        spec = importlib.util.spec_from_file_location("aikimi_outpaint_ui_test", path)
        cls.ui = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"modules": fake_modules}):
            spec.loader.exec_module(cls.ui)

    def test_prepare_restore_callback_roundtrip(self):
        state, padded, prompt, settings, summary, cleared, old_generated = self.ui.prepare_canvas(
            Image.new("RGB", (256, 256), (91, 33, 202)), 32, 0, 0, 0, "v2", "room",
        )
        self.assertIsNone(cleared)
        self.assertIsNone(old_generated)
        self.assertIn("288 × 256", summary)
        self.assertEqual(prompt, settings["prompt"])
        result = self.ui.restore_original(state, padded, 0)
        self.assertEqual(result.crop(state[1].box).tobytes(), state[0].tobytes())

    def test_real_gradio_tab_build_and_private_callbacks(self):
        tabs = self.ui.on_ui_tabs()
        self.assertEqual(tabs[0][2], "qwen_image21_outpaint")
        config = tabs[0][0].get_config_file()
        self.assertEqual(len(config["dependencies"]), 11)
        for dependency in config["dependencies"]:
            self.assertEqual(dependency["api_visibility"], "private")
        self.assertIn("画像生成を行いません", str(config))
        tabs[0][0].close()

    def test_changes_invalidate_prepared_state_and_result(self):
        state, preview, prompt, settings, summary, result, generated = self.ui.clear_prepared()
        self.assertIsNone(state)
        self.assertIsNone(preview)
        self.assertEqual(prompt, "")
        self.assertIsNone(settings)
        self.assertIsNone(result)
        self.assertIsNone(generated)
        self.assertIsNone(self.ui.clear_result())

    def test_missing_snapshot_is_actionable(self):
        with self.assertRaises(ValueError):
            self.ui.restore_original(None, Image.new("RGB", (256, 256)), 0)


if __name__ == "__main__":
    unittest.main()
