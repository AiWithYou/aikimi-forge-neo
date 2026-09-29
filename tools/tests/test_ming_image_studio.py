"""Offline checks for Ming prompts, real ComfyUI contracts and RGBA preservation."""

from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge import ming_image_studio as studio
from tools import setup_ming_image


def load_ui():
    location = Path(__file__).resolve().parents[2] / "extensions-builtin/ming-image-studio/scripts/ming_image_studio.py"
    spec = importlib.util.spec_from_file_location("ming_ui_test", location)
    ui = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"modules.script_callbacks": SimpleNamespace(on_ui_tabs=Mock())}):
        spec.loader.exec_module(ui)
    return ui


class MingImageContracts(unittest.TestCase):
    def test_quantization_and_upstream_pins(self):
        manifest = setup_ming_image.manifest()
        self.assertEqual(manifest["repository"], "Comfy-Org/Ming-Image")
        self.assertEqual(len(manifest["revision"]), 40)
        self.assertEqual(len(manifest["comfy_revision"]), 40)
        models = setup_ming_image.profiles()["models"].artifacts
        self.assertEqual(len(models), 3)
        self.assertTrue(all(len(item.sha256) == 64 and item.size > 0 for item in models))
        self.assertTrue(any(studio.ENCODER in item.relative_path for item in models))
        self.assertLess(sum(item.size for item in models), 20_000_000_000)

    def test_prompt_preserves_json_and_rejects_contradictory_controls(self):
        prompt = '  { "layers": [{"description": "春の光"}] }\n'
        self.assertEqual(studio.MingImageRequest(prompt).effective_prompt(), prompt)
        for request in (
            studio.MingImageRequest(prompt, transparent=True),
            studio.MingImageRequest(prompt, text="text"),
            studio.MingImageRequest('{"bad":}'),
            studio.MingImageRequest("[]"),
        ):
            with self.subTest(request=request), self.assertRaises(studio.MingImageError):
                request.effective_prompt()

    def test_alpha_prefix_once_and_verbatim_text_not_duplicated(self):
        request = studio.MingImageRequest(
            studio.RGBA_PREFIXES[1] + "\nPoster of 春の光", text='春の光\nA "B"\nA "B"', transparent=True
        )
        prompt = request.effective_prompt()
        self.assertEqual(prompt.count(studio.RGBA_PREFIXES[0]), 1)
        self.assertNotIn(studio.RGBA_PREFIXES[1], prompt)
        self.assertEqual(prompt.count("春の光"), 1)
        self.assertIn(json.dumps('A "B"', ensure_ascii=False), prompt)

    def test_input_limits_are_checked_before_any_runtime_work(self):
        request = studio.MingImageRequest("poster")
        for field, value in (
            ("width", 513),
            ("height", True),
            ("width", 4112),
            ("steps", 0),
            ("steps", 12.5),
            ("seed", 2**53),
            ("seed", True),
            ("text", None),
            ("transparent", "yes"),
        ):
            with self.subTest(field=field, value=value), self.assertRaises(studio.MingImageError):
                replace(request, **{field: value}).validate()

    def test_graph_uses_supported_loader_and_size_dependent_sampling(self):
        graph = studio.build_workflow(studio.MingImageRequest("poster", width=2560, height=1440), 42)
        self.assertEqual(graph["model"]["inputs"]["unet_name"], studio.MODEL)
        self.assertEqual(graph["clip"]["inputs"]["clip_name"], studio.ENCODER)
        self.assertEqual(graph["clip"]["inputs"]["type"], "qwen_image")
        self.assertEqual(graph["sample"]["inputs"]["cfg"], 1)
        self.assertEqual(graph["sample"]["inputs"]["seed"], 42)
        self.assertEqual(graph["sampling"]["inputs"]["width"], 2560)
        self.assertEqual(graph["negative"]["class_type"], "ConditioningZeroOut")
        self.assertEqual(graph["save"]["inputs"]["images"], ["decode", 0])

    def test_png_bytes_and_alpha_survive_publication_and_seed_is_recorded(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.png"
            image = Image.new("RGBA", (256, 256), (128, 64, 32, 0))
            image.putpixel((10, 10), (20, 30, 40, 255))
            image.save(source)
            request = studio.MingImageRequest("transparent logo", transparent=True, width=256, height=256)
            result = studio.save_result(
                source, request, 42, "job", SimpleNamespace(core_revision="pin", comfy_version="0.37"), root / "out"
            )
            self.assertEqual(Path(result["path"]).read_bytes(), source.read_bytes())
            self.assertEqual(result["metadata"]["alpha_range"], [0, 255])
            self.assertTrue(result["metadata"]["has_transparency"])
            saved = json.loads(Path(result["files"][1]).read_text(encoding="utf-8"))
            self.assertEqual(saved["seed"], 42)
            self.assertEqual(Path(result["files"][2]).read_text(encoding="utf-8"), request.effective_prompt())

    def test_transparency_reports_area_not_a_single_translucent_pixel(self):
        ui = load_ui()
        image = Image.new("RGBA", (100, 100), (200, 100, 50, 255))
        image.putpixel((0, 0), (200, 100, 50, 254))
        stats = studio.transparency_stats(image)
        self.assertTrue(stats["has_transparency"])
        self.assertEqual(stats["near_transparent_fraction"], 0)
        data = {**stats, "width": 100, "height": 100, "seed": 42, "request": {"transparent": True}}
        self.assertIn("ほぼ不透明", ui._caption(data))
        for x in range(5):
            for y in range(100):
                image.putpixel((x, y), (200, 100, 50, 0))
        stats = studio.transparency_stats(image)
        self.assertEqual(stats["near_transparent_fraction"], 0.05)
        self.assertIn("約5%", ui._caption({**data, **stats}))
        self.assertEqual(
            studio.transparency_stats(Image.new("RGBA", (4, 4), (0, 0, 0, 10)))["near_transparent_fraction"], 1
        )
        self.assertEqual(studio.transparency_stats(Image.new("RGB", (4, 4)))["near_transparent_fraction"], 0)

    def test_bad_output_is_not_published(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.png"
            Image.new("RGB", (32, 32)).save(source)
            with self.assertRaises(studio.MingImageError):
                studio.save_result(
                    source,
                    studio.MingImageRequest("poster"),
                    1,
                    "job",
                    SimpleNamespace(core_revision="pin", comfy_version="0.37"),
                    root / "out",
                )
            self.assertEqual(list((root / "out").iterdir()), [])

    def test_ui_restores_resolved_seed_and_double_uses_result_not_edited_form(self):
        ui = load_ui()
        request = studio.MingImageRequest("original", text="HELLO", width=1024, height=1024)
        data = {
            "request": studio.asdict(request),
            "seed": 321,
            "width": 1024,
            "height": 1024,
            "steps": 12,
            "prompt_id": "parent",
        }
        record = {"metadata": data}
        self.assertEqual(ui._restore(record)[6], "321")
        with patch.object(studio, "run_generation", return_value=iter(())) as run:
            list(ui._double(record))
        doubled = run.call_args.args[0]
        self.assertEqual((doubled.width, doubled.height, doubled.seed), (2048, 2048, 321))
        self.assertEqual((doubled.prompt, doubled.parent), ("original", "parent"))

    def test_json_mode_preserves_form_values_but_sends_only_json(self):
        ui = load_ui()
        prompt = '  {"description": "透明な花"}\n'
        hint, text_control, alpha_control = ui._prompt_mode(prompt)
        for control in (text_control, alpha_control):
            self.assertFalse(control["interactive"])
            self.assertNotIn("value", control)
        self.assertIn("入力は保持", hint)
        request = ui._request(prompt, "stored text", True, 1024, 1024, 12, "123")
        self.assertEqual(request.effective_prompt(), prompt)
        self.assertEqual((request.text, request.transparent), ("", False))
        with self.assertRaises(studio.MingImageError):
            ui._request('{"broken":}', "", False, 1024, 1024, 12, "123")

    def test_failed_setup_keeps_actionable_error_and_allows_retry(self):
        ui = load_ui()
        failure = "準備を完了できませんでした: disk full <detail>"
        with (
            patch.object(ui, "_setup", return_value=iter(("準備中", failure))),
            patch.object(setup_ming_image, "runtime_ready", return_value=False),
        ):
            updates = list(ui._prepare())
        message, generate, prepare, status = updates[-1]
        self.assertEqual(message, failure)
        self.assertFalse(generate["visible"])
        self.assertTrue(prepare["interactive"])
        self.assertIn("&lt;detail&gt;", status)

    def test_setup_streams_real_download_progress(self):
        ui = load_ui()

        def install(**kwargs):
            self.assertFalse(kwargs["repair"])
            output = setup_ming_image.ProgressOutput(kwargs["progress"])
            output.write("ming-image-" + studio.ENCODER + ": 1.00 GiB")
            output.write(" / 11.93 GiB\n")
            return {"ok": True}

        bridge = Mock()
        bridge.runtime_setup_session.return_value = nullcontext()
        with (
            patch.object(studio, "_bridge", return_value=bridge),
            patch.object(setup_ming_image, "run", side_effect=install),
        ):
            updates = list(ui._setup())
        self.assertIn("テキストエンコーダーを取得中 · 1.00 GiB / 11.93 GiB", updates)
        self.assertTrue(updates[-1].startswith("準備できました"))

    def test_ui_renders_offline_and_private_callbacks(self):
        ui = load_ui()
        with patch.object(studio, "ensure_runtime") as runtime:
            tabs = ui.on_ui_tabs()
            runtime.assert_not_called()
        self.assertEqual(tabs[0][1:], ("Ming Image", "ming_image_studio"))
        config = tabs[0][0].get_config_file()
        self.assertTrue(all(d["api_visibility"] == "private" for d in config["dependencies"] if d.get("backend_fn")))


if __name__ == "__main__":
    unittest.main()
