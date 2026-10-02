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

import gradio as gr
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
        sys.modules[spec.name] = cls.ui
        cls.addClassCleanup(sys.modules.pop, spec.name, None)
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

    def test_no_lora_needs_no_download_and_external_recipe_disables_adapter(self):
        with patch("modules_forge.qwen_image21.outpaint_lora.installed", side_effect=AssertionError("No download")):
            self.assertEqual(self.ui.adapter_status("none"), "")
            self.assertFalse(self.ui.setup_visibility("none")["visible"])
            self.assertEqual(self.ui.prepare_native("none"), "")
        _, _, prompt, settings, _ = self.ui.prepare_canvas(
            Image.new("RGB", (256, 256), "red"), 32, 0, 32, 0, "none", "room"
        )
        self.assertIsNone(settings["weights"])
        self.assertIn("LoRAを無効にする", self.ui.handoff_steps(settings))
        self.assertIn("Outpaint the image", prompt)

    def test_canvas_commit_reaches_native_generation_with_all_four_sides(self):
        source = Image.new("RGB", (736, 512), "blue")
        pads = self.ui.commit_canvas(json.dumps({"w": 736, "h": 512, "pads": [192, 0, 64, 0]}), source)
        captured = {}
        studio = types.SimpleNamespace(
            start=lambda generation, owner: captured.update(request=generation, owner=owner) or "job-gui"
        )
        browser = types.SimpleNamespace(session_hash="gui-owner", username=None)
        with patch.object(self.ui, "native_studio", return_value=studio):
            result = self.ui.start_native(source, *pads, "none", "room", 0, 25, 42, "base_q4_k_m", "", browser)
        self.assertEqual(result[0], "job-gui")
        self.assertEqual(captured["request"].outpaint_margins, (192, 0, 64, 0))
        self.assertEqual((captured["request"].width, captured["request"].height), (992, 512))

    def test_shared_profile_is_injected_after_request_and_frozen_per_generation(self):
        from gradio.helpers import special_args

        from modules_forge.qwen_image21.core import Request
        from modules_forge.qwen_image21.outpaint_profile import FIELDS, resolve, summary

        source = Image.new("RGB", (256, 256), "blue")
        first = Request("test", precision="w4a8", fun_acc=True, steps=4).to_dict()
        first["lora_strengths"] = [["a.safetensors", 0.65]]
        first["style_loras"] = ["a.safetensors"]
        values = [first[name] for name in FIELDS]
        captured = {}
        studio = types.SimpleNamespace(
            start=lambda generation, owner: captured.update(request=generation, owner=owner) or "job-shared"
        )
        browser = gr.Request(session_hash="client-a")
        inputs = [source, 32, 0, 32, 0, "v2", "room", 0, 25, 42, "base_q4_k_m", "", *values]
        args, _, _, _ = special_args(self.ui.start_native, inputs, request=browser)
        with patch.object(self.ui, "native_studio", return_value=studio):
            result = self.ui.start_native(*args)
        self.assertEqual(result[0], "job-shared", str(result[1]) + " " + str(self.ui.start_native.__annotations__))
        request = captured["request"].resolved()
        self.assertEqual((request.precision, request.steps, request.fun_acc), ("w4a8", 4, True))
        self.assertEqual(request.style_loras, ({"name": "a.safetensors", "strength": 0.65},))
        self.assertEqual(captured["owner"], ":client-a")
        # A second client and edits to the first client's table cannot mutate a running job.
        values[FIELDS.index("lora_strengths")][0][1] = 0.1
        other = Request("other", precision="bf16").to_dict()
        other["style_loras"], other["lora_strengths"] = [], []
        other_profile = resolve([other[name] for name in FIELDS], 25)
        self.assertEqual(other_profile["precision"], "bf16")
        self.assertEqual(request.style_loras[0]["strength"], 0.65)
        self.assertIn(
            "0.65", summary(resolve([first[name] for name in FIELDS], 4) | {"style_loras": request.style_loras}, "v2")
        )

    def test_distilled_steps_restore_outpaint_steps_and_conflicts_are_visible(self):
        fixed, remembered = self.ui.outpaint_step_settings("turbo_q4_k_m", False, 31, 25)
        self.assertEqual(fixed["value"], 4)
        self.assertFalse(fixed["interactive"])
        restored, remembered = self.ui.outpaint_step_settings("bf16", False, 4, remembered)
        self.assertEqual(restored["value"], 31)
        self.assertTrue(restored["interactive"])

    def test_control_readiness_uses_the_same_exif_orientation_as_job_copy(self):
        from modules_forge.qwen_image21.core import Request, copy_control_image
        from modules_forge.qwen_image21.outpaint_profile import FIELDS

        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            source = Image.new("RGB", (512, 384), "blue")
            control = Image.new("RGB", (384, 512), "white")
            exif = control.getexif()
            exif[274] = 6
            path = directory / "portrait-control.jpg"
            control.save(path, exif=exif)
            copied = copy_control_image(str(path), directory)
            with Image.open(copied) as normalized:
                self.assertEqual(normalized.size, source.size)
            profile = Request("test", control_kind="canny", control_image=str(path)).to_dict()
            profile["lora_strengths"] = []
            studio = types.SimpleNamespace(runtime=directory)
            with patch.object(self.ui, "native_studio", return_value=studio):
                button, _ = self.ui.native_readiness(
                    source,
                    64,
                    32,
                    96,
                    64,
                    "",
                    gr.Request(session_hash="orientation"),
                    25,
                    *[profile[name] for name in FIELDS],
                )
            self.assertTrue(button["interactive"])

    def test_real_gradio_tab_build_and_private_callbacks(self):
        tabs = self.ui.on_ui_tabs()
        self.assertEqual(tabs[0][2], "qwen_image21_outpaint")
        config = tabs[0][0].get_config_file()
        for dependency in config["dependencies"]:
            self.assertEqual(dependency["api_visibility"], "private")
        text = str(config)
        self.assertNotIn("画像生成を行いません", text)
        self.assertIn("いつものQwen 2.1で描き足し", text)
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
            "qwen21-outpaint-generate",
            "qwen21-outpaint-cancel",
            "qwen21-outpaint-result",
            "qwen21-outpaint-download",
            "qwen21-outpaint-reuse",
            "qwen21-outpaint-reset-source",
        ):
            self.assertIn(elem_id, ids)
        self.assertFalse(ids["qwen21-outpaint-preview-column"]["props"]["visible"])
        tabs[0][0].close()

    def test_native_start_snapshots_inputs_and_passes_owner_and_selected_model(self):
        studio = types.SimpleNamespace(
            start=lambda generation, owner: captured.update(request=generation, owner=owner) or "job-1"
        )
        captured = {}
        browser = types.SimpleNamespace(session_hash="session", username="user")
        with patch.object(self.ui, "native_studio", return_value=studio):
            values = self.ui.start_native(
                Image.new("RGB", (256, 256), "blue"),
                32,
                0,
                32,
                0,
                "v2",
                "forest",
                0,
                25,
                42,
                "base_q4_k_m",
                None,
                browser,
            )
        self.assertEqual(values[0], "job-1")
        self.assertEqual(captured["owner"], "user:session")
        request = captured["request"].resolved()
        self.assertEqual((request.width, request.height), (320, 256))
        self.assertEqual(request.precision, "base_q4_k_m")
        self.assertEqual(request.outpaint_feather, 0)
        self.assertFalse(values[2]["interactive"])
        self.assertTrue(values[3]["interactive"])
        self.assertTrue(values[4]["active"])

    def test_native_invalid_start_preserves_job_and_previous_result(self):
        browser = types.SimpleNamespace(session_hash="session", username=None)
        with patch.object(self.ui, "native_studio") as studio:
            result = self.ui.start_native(None, 32, 0, 32, 0, "v2", "", 0, 25, 42, "int8", "previous.png", browser)
        studio.assert_not_called()
        self.assertNotIn("value", result[0])
        self.assertNotIn("value", result[5])
        self.assertNotIn("value", result[6])

    def test_native_failed_poll_keeps_previous_download(self):
        browser = types.SimpleNamespace(session_hash="session", username=None)
        for state in ("failed", "cancelled"):
            with self.subTest(state=state):
                studio = types.SimpleNamespace(
                    status=lambda job, owner, state=state: {
                        "done": True,
                        "state": state,
                        "message": "stopped",
                        "elapsed": 1,
                    }
                )
                with patch.object(self.ui, "native_studio", return_value=studio):
                    result = self.ui.poll_native("job-1", "previous-job", True, browser)
                self.assertTrue(result[1]["interactive"])
                self.assertFalse(result[2]["interactive"])
                self.assertFalse(result[3]["active"])
                self.assertNotIn("value", result[4])
                self.assertNotIn("value", result[5])
                self.assertEqual(result[4]["label"], "前回の生成結果 · 開始時の設定")

    def test_missing_job_resets_running_label_without_replacing_previous_image(self):
        from modules_forge.qwen_image21.service import JobNotFound

        browser = types.SimpleNamespace(session_hash="session", username=None)
        studio = types.SimpleNamespace(status=lambda *args: (_ for _ in ()).throw(JobNotFound("gone")))
        with patch.object(self.ui, "native_studio", return_value=studio):
            result = self.ui.poll_native("missing-job", "previous-job", True, browser)
        self.assertEqual(result[4]["label"], "前回の生成結果 · 開始時の設定")
        self.assertNotIn("value", result[4])
        self.assertFalse(result[2]["interactive"])

    def test_preview_marks_extension_and_keeps_original_visible(self):
        source = Image.new("RGB", (256, 256), (10, 200, 30))
        column, preview, caption = self.ui.preview_canvas(source, 128, 0, 0, 0)
        image = preview["value"]
        self.assertTrue(column["visible"])
        self.assertLessEqual(max(image.size), self.ui.PREVIEW_SIDE)
        self.assertEqual(image.size[0] * 256, image.size[1] * 384)  # Same aspect as the 384 × 256 canvas.
        self.assertEqual(image.getpixel((image.width - 10, image.height // 2)), (10, 200, 30))
        stripes = {image.getpixel((x, y)) for x in range(0, 40) for y in range(0, 40)}
        self.assertEqual(stripes, {(128, 128, 128), (158, 158, 158)})
        self.assertIn("384 × 256", caption)

    def test_preview_reports_alignment_and_errors_inline(self):
        _, _, caption = self.ui.preview_canvas(Image.new("RGB", (256, 256)), 1, 0, 0, 0)
        self.assertIn("32 px単位", caption)
        self.assertIn("左 +31 px", caption)
        column, preview, caption = self.ui.preview_canvas(Image.new("RGB", (1536, 1024)), 128, 128, 128, 128)
        self.assertTrue(column["visible"])
        self.assertFalse(preview["visible"])
        self.assertIn("2 MP", caption)
        column, preview, caption = self.ui.preview_canvas(None, 128, 128, 128, 128)
        self.assertFalse(column["visible"])
        self.assertIsNone(preview)

    def test_preview_transparent_source_matches_gray_reference(self):
        _, preview, _ = self.ui.preview_canvas(Image.new("RGBA", (256, 256), (255, 0, 0, 0)), 128, 0, 0, 0)
        image = preview["value"]
        inner = {image.getpixel((x, y)) for x in range(image.width - 40, image.width - 4) for y in range(4, 40)}
        self.assertEqual(inner, {(128, 128, 128)})

    def test_completed_result_reuse_reads_owned_artifact_without_changing_alpha(self):
        original = Image.new("RGBA", (256, 256), (9, 80, 70, 0))
        path = self.ui.save_png(original, "server-output")
        calls = []
        studio = types.SimpleNamespace(
            status=lambda job, owner: {"done": True},
            artifact=lambda job, owner: calls.append((job, owner)) or path,
        )
        browser = types.SimpleNamespace(session_hash="session", username="user")
        with patch.object(self.ui, "native_studio", return_value=studio):
            source, stage, reset, _ = self.ui.source_from_result("last-success", "failed-new-job", browser)
        self.assertEqual(calls, [("last-success", "user:session")])
        self.assertEqual(source.mode, "RGBA")
        self.assertEqual(source.tobytes(), original.tobytes())
        self.assertEqual(stage["selected"], "range")
        self.assertTrue(reset["interactive"])

    def test_reuse_cannot_change_source_while_a_job_is_running(self):
        studio = types.SimpleNamespace(status=lambda job, owner: {"done": False})
        browser = types.SimpleNamespace(session_hash="session", username=None)
        with (
            patch.object(self.ui, "native_studio", return_value=studio),
            self.assertRaisesRegex(ValueError, "生成が終わって"),
        ):
            self.ui.source_from_result("last-success", "running", browser)

    def test_first_source_is_restored_losslessly_and_new_upload_replaces_baseline(self):
        first = Image.new("RGBA", (256, 256), (90, 80, 70, 0))
        baseline, *_ = self.ui.remember_original(first)
        recovered, stage, reset, _ = self.ui.original_source(baseline)
        self.assertEqual(recovered.tobytes(), first.tobytes())
        self.assertEqual(stage["selected"], "range")
        self.assertFalse(reset["interactive"])
        replacement, *_ = self.ui.remember_original(Image.new("RGB", (320, 256), "blue"))
        self.assertNotEqual(replacement, baseline)
        self.assertIsNone(self.ui.remember_original(None)[0])

    def test_completion_selects_result_but_does_not_enable_generation_for_invalid_draft(self):
        original = Image.new("RGBA", (320, 256), (1, 2, 3, 4))
        path = self.ui.save_png(original, "server-output")
        studio = types.SimpleNamespace(
            status=lambda job, owner: {"done": True, "state": "complete", "message": "完了 Seed 42", "elapsed": 3},
            artifact=lambda job, owner: path,
        )
        browser = types.SimpleNamespace(session_hash="session", username=None)
        with patch.object(self.ui, "native_studio", return_value=studio):
            result = self.ui.poll_native("job-123456789", "old", False, browser)
        self.assertFalse(result[1]["interactive"])
        self.assertEqual(result[6], "job-123456789")
        self.assertEqual(result[7]["selected"], "result")
        self.assertTrue(result[8]["interactive"])
        self.assertIn("Seed 42", result[9])
        self.assertNotIn("Seed", result[4]["label"])
        self.assertIn("job-1234", result[5]["value"])
        with Image.open(result[5]["value"]) as saved:
            self.assertEqual(saved.tobytes(), original.tobytes())

    def test_invalid_padding_and_active_job_keep_generate_disabled(self):
        browser = types.SimpleNamespace(session_hash="session", username=None)
        image = Image.new("RGB", (256, 256))
        self.assertFalse(self.ui.native_readiness(image, 0, 0, 0, 0, "", browser)[0]["interactive"])
        self.assertTrue(self.ui.native_readiness(image, 128, 0, 128, 0, "", browser)[0]["interactive"])
        studio = types.SimpleNamespace(status=lambda job, owner: {"done": False})
        with patch.object(self.ui, "native_studio", return_value=studio):
            self.assertFalse(self.ui.native_readiness(image, 128, 0, 128, 0, "running", browser)[0]["interactive"])

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
