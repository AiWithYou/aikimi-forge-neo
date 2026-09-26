"""Offline geometry, alpha preservation, and real Gradio callback contracts."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from modules_forge.qwen_image21.outpaint import (
    MAX_CANVAS_PIXELS,
    PROMPT,
    Plan,
    normalize_image,
    prepare,
    recipe,
    stitch,
)

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
        self.assertIsInstance(plan, Plan)

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

    def setUp(self):
        # Download buttons write named PNGs; keep them out of the real temp dir.
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        patcher = patch.object(self.ui, "_EXPORT_DIRECTORY", temporary)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_prepare_restore_callback_roundtrip(self):
        state, padded, prompt, settings, summary = self.ui.prepare_canvas(
            Image.new("RGB", (256, 256), (91, 33, 202)),
            32,
            0,
            0,
            0,
            "v2",
            "room",
        )
        self.assertIn("288 × 256", summary)
        self.assertEqual(prompt, settings["prompt"])
        binding, _, button = self.ui.generated_status(state, padded, False)
        self.assertTrue(button["interactive"])
        result = self.ui.restore_original(state, padded, 0, binding, False)
        self.assertEqual(result.crop(state.plan.box).tobytes(), state.original.tobytes())

    def test_real_gradio_tab_build_and_private_callbacks(self):
        tabs = self.ui.on_ui_tabs()
        self.assertEqual(tabs[0][2], "qwen_image21_outpaint")
        config = tabs[0][0].get_config_file()
        self.assertEqual(len(config["dependencies"]), 17)
        for dependency in config["dependencies"]:
            self.assertEqual(dependency["api_visibility"], "private")
        text = str(config)
        self.assertIn("画像生成を行いません", text)
        self.assertIn("完成画像を作成", text)
        self.assertNotIn("復元", text)  # The final output is named 完成画像, not an ambiguous restore.
        ids = {component["props"].get("elem_id"): component for component in config["components"]}
        for elem_id in (
            "qwen21-outpaint-source",
            "qwen21-outpaint-prepare",
            "qwen21-outpaint-stitch",
            "qwen21-outpaint-restored",
            "qwen21-outpaint-reference-download",
            "qwen21-outpaint-final-download",
        ):
            self.assertIn(elem_id, ids)
        self.assertFalse(ids["qwen21-outpaint-preview-column"]["props"]["visible"])
        tabs[0][0].close()

    def test_preview_marks_extension_and_keeps_original_visible(self):
        source = Image.new("RGB", (256, 256), (10, 200, 30))
        column, preview, caption, direction = self.ui.preview_canvas(source, 128, 0, 0, 0)
        image = preview["value"]
        self.assertTrue(column["visible"])
        self.assertLessEqual(max(image.size), self.ui.PREVIEW_SIDE)
        self.assertEqual(image.size[0] * 256, image.size[1] * 384)  # Same aspect as the 384 × 256 canvas.
        self.assertEqual(image.getpixel((image.width - 10, image.height // 2)), (10, 200, 30))
        stripes = {image.getpixel((x, y)) for x in range(0, 40) for y in range(0, 40)}
        self.assertEqual(stripes, {(128, 128, 128), (158, 158, 158)})
        self.assertIn("384 × 256", caption)
        self.assertEqual(direction["value"], "左")

    def test_preview_reports_alignment_and_errors_inline(self):
        _, _, caption, _ = self.ui.preview_canvas(Image.new("RGB", (256, 256)), 1, 0, 0, 0)
        self.assertIn("32 px単位", caption)
        column, preview, caption, _ = self.ui.preview_canvas(Image.new("RGB", (1536, 1024)), 128, 128, 128, 128)
        self.assertTrue(column["visible"])
        self.assertFalse(preview["visible"])
        self.assertIn("2 MP", caption)
        column, preview, caption, _ = self.ui.preview_canvas(None, 128, 128, 128, 128)
        self.assertFalse(column["visible"])
        self.assertIsNone(preview)

    def test_preview_transparent_source_matches_gray_reference(self):
        _, preview, _, _ = self.ui.preview_canvas(Image.new("RGBA", (256, 256), (255, 0, 0, 0)), 128, 0, 0, 0)
        image = preview["value"]
        inner = {image.getpixel((x, y)) for x in range(image.width - 40, image.width - 4) for y in range(4, 40)}
        self.assertEqual(inner, {(128, 128, 128)})

    def test_direction_shortcuts_keep_manual_amounts(self):
        values = lambda updates: tuple(update["value"] for update in updates)  # noqa: E731
        self.assertEqual(values(self.ui.apply_direction("左右", 200, 0, 0, 0)), (200, 0, 200, 0))
        self.assertEqual(values(self.ui.apply_direction("左", 64, 128, 256, 128)), (64, 0, 0, 0))
        self.assertEqual(values(self.ui.apply_direction("四方", 0, 0, None, 0)), (128, 128, 128, 128))
        self.assertEqual(values(self.ui.apply_direction("上下", 96.0, 0, 0, 0)), (0, 96, 0, 96))
        self.assertTrue(all("value" not in update for update in self.ui.apply_direction(None, 1, 2, 3, 4)))
        self.assertEqual(self.ui.direction_of(0, 32, 0, 64), "上下")
        self.assertIsNone(self.ui.direction_of(32, 32, 0, 0))

    def test_prepare_offers_named_reference_and_concrete_handoff(self):
        outputs = self.ui.prepare_for_ui(Image.new("RGB", (256, 256), "blue"), 32, 0, 0, 0, "v1", "", None)
        download, steps, final = outputs[13], outputs[14], outputs[15]
        self.assertTrue(download["interactive"])
        self.assertTrue(download["value"].endswith("qwen-outpaint-reference-288x256.png"))
        with Image.open(download["value"]) as saved:
            self.assertEqual(saved.tobytes(), outputs[1]["value"].tobytes())
        self.assertIn("qwen-image-2.1-outpaint.safetensors", steps)
        self.assertIn("288 × 256", steps)
        self.assertIn("25 steps · CFG 1 · euler / simple · denoise 1", steps)
        self.assertFalse(final["interactive"])
        self.assertIsNone(final["value"])

    def test_finish_saves_final_image_and_changes_clear_it(self):
        state, padded, *_ = self.ui.prepare_canvas(Image.new("RGB", (256, 256), "red"), 32, 0, 0, 0, "v2", "")
        result, download = self.ui.finish_for_ui(state, padded, 0, state.token, False)
        self.assertTrue(download["value"].endswith("qwen-outpaint-288x256.png"))
        with Image.open(download["value"]) as saved:
            self.assertEqual(saved.tobytes(), result.tobytes())
        for restored, cleared in (self.ui.feather_changed(), self.ui.uploaded(state, padded, False)[3:]):
            self.assertIsNone(restored["value"])
            self.assertIsNone(cleared["value"])
            self.assertFalse(cleared["interactive"])

    def test_draft_change_keeps_snapshot_but_blocks_stale_restore(self):
        source = Image.new("RGB", (256, 256), "red")
        state, padded, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "")
        dirty, message, prepare_button, restore_button, result, validation, reference, prompt, *downloads = (
            self.ui.draft_changed(
                source,
                33,
                0,
                0,
                0,
                "v2",
                "",
                state,
                state.token,
                padded,
            )
        )
        for download in downloads:
            self.assertIn("前回", download["label"])
            self.assertNotIn("value", download)  # Previous files remain downloadable.
        self.assertTrue(dirty)
        self.assertTrue(prepare_button["interactive"])
        self.assertFalse(restore_button["interactive"])
        self.assertIn("未反映", message)
        self.assertIn("未反映", validation)
        self.assertIn("未反映", reference["label"])
        self.assertNotIn("value", reference)
        self.assertNotIn("value", prompt)
        self.assertNotIn("value", result)  # Preserve the previous result for download.
        self.assertEqual(state.original.tobytes(), source.tobytes())
        with self.assertRaisesRegex(ValueError, "未反映"):
            self.ui.restore_original(state, padded, 0, state.token, dirty)

    def test_new_preparation_rejects_old_binding_even_at_same_dimensions(self):
        source = Image.new("RGB", (256, 256))
        first, padded, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "")
        second, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v1", "different scene")
        with self.assertRaisesRegex(ValueError, "選び直して"):
            self.ui.restore_original(second, padded, 0, first.token, False)

    def test_wrong_size_is_explained_before_restore(self):
        state, *_ = self.ui.prepare_canvas(Image.new("RGB", (256, 256)), 32, 0, 0, 0, "v2", "")
        binding, message, button = self.ui.generated_status(state, Image.new("RGB", (256, 256)), False)
        self.assertIsNone(binding)
        self.assertFalse(button["interactive"])
        self.assertIn("288 × 256", message)
        self.assertIn("256 × 256", message)

    def test_upload_during_dirty_draft_cannot_reenable_restore(self):
        state, padded, *_ = self.ui.prepare_canvas(Image.new("RGB", (256, 256)), 32, 0, 0, 0, "v2", "")
        binding, _, button = self.ui.generated_status(state, padded, True)
        self.assertIsNone(binding)
        self.assertFalse(button["interactive"])

    def test_small_source_has_valid_default_feather(self):
        outputs = self.ui.prepare_for_ui(Image.new("RGB", (32, 32)), 128, 128, 128, 128, "v2", "", None)
        state, padded = outputs[0], outputs[1]["value"]
        feather = outputs[9]
        self.assertEqual(feather["maximum"], 15)
        result = self.ui.restore_original(state, padded, feather["value"], state.token, False)
        self.assertEqual(result.size, padded.size)

    def test_repreparation_keeps_generated_input_unbound(self):
        image = Image.new("RGB", (512, 512))
        outputs = self.ui.prepare_for_ui(Image.new("RGB", (256, 256)), 128, 128, 128, 128, "v2", "", image)
        self.assertIsNone(outputs[6])
        self.assertIn("選び直して", outputs[10])
        self.assertFalse(outputs[11]["interactive"])

    def test_missing_source_disables_prepare(self):
        self.assertFalse(self.ui.draft_changed(None, 128, 128, 128, 128, "v2", "", None, None, None)[2]["interactive"])
        chosen = self.ui.draft_changed(Image.new("RGB", (256, 256)), 128, 128, 128, 128, "v2", "", None, None, None)
        self.assertTrue(chosen[2]["interactive"])
        self.assertEqual(chosen[1], self.ui.READY)  # Says what the prepare click produces.

    def test_returning_to_prepared_input_resumes_without_reupload(self):
        source = Image.new("RGB", (256, 256))
        state, padded, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "forest")
        changed = self.ui.draft_changed(source, 33, 0, 0, 0, "v2", "forest", state, state.token, padded)
        self.assertTrue(changed[0])
        reverted = self.ui.draft_changed(source, 32, 0, 0, 0, "v2", "forest", state, state.token, padded)
        self.assertFalse(reverted[0])
        self.assertTrue(reverted[3]["interactive"])

    def test_same_preparation_keeps_binding_and_feather_choice(self):
        source = Image.new("RGB", (256, 256))
        state, padded, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "forest")
        result = self.ui.prepare_for_ui(source.copy(), 32.0, 0, 0, 0, "v2", " forest ", padded, state, state.token)
        self.assertEqual(result[0].token, state.token)
        self.assertEqual(result[6], state.token)
        self.assertNotIn("value", result[9])
        self.assertTrue(result[11]["interactive"])

    def test_prompt_and_pixels_are_part_of_generation_identity(self):
        source = Image.new("RGBA", (256, 256), (1, 2, 3, 0))
        state, padded, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "forest")
        changed_prompt, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "desert")
        source.putpixel((0, 0), (4, 5, 6, 0))
        changed_pixels, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "forest")
        self.assertNotEqual(state.token, changed_prompt.token)
        self.assertNotEqual(state.token, changed_pixels.token)
        for new_state in (changed_prompt, changed_pixels):
            with self.assertRaisesRegex(ValueError, "選び直して"):
                self.ui.restore_original(new_state, padded, 0, state.token, False)

    def test_binding_does_not_override_wrong_image_dimensions(self):
        source = Image.new("RGB", (256, 256))
        state, *_ = self.ui.prepare_canvas(source, 32, 0, 0, 0, "v2", "")
        result = self.ui.prepare_for_ui(source, 32, 0, 0, 0, "v2", "", source, state, state.token)
        self.assertFalse(result[11]["interactive"])
        draft = self.ui.draft_changed(source, 32, 0, 0, 0, "v2", "", state, state.token, source)
        self.assertFalse(draft[3]["interactive"])

    def test_changed_feather_removes_stale_download(self):
        restored, download = self.ui.feather_changed()
        self.assertIsNone(restored["value"])
        self.assertIsNone(download["value"])

    def test_missing_snapshot_is_actionable(self):
        with self.assertRaises(ValueError):
            self.ui.restore_original(None, Image.new("RGB", (256, 256)), 0, None, False)


if __name__ == "__main__":
    unittest.main()
