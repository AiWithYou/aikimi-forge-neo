"""Exercise annotation state and Gradio's request injection without a GPU."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image, ImageDraw

from modules_forge.qwen_image21.service import JobNotFound
from tools.tests.test_qwen_image21_service import load_ui


class AnnotationUiTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen-annotation-ui-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ui = load_ui()
        self.source = self.root / "source.png"
        self.second = self.root / "second.png"
        self.layer = self.root / "layer.png"
        Image.new("RGBA", (40, 32), (10, 20, 30, 255)).save(self.source)
        Image.new("RGBA", (40, 32), (50, 60, 70, 255)).save(self.second)
        with Image.new("RGBA", (40, 32), (0, 0, 0, 0)) as image:
            ImageDraw.Draw(image).ellipse((10, 10, 25, 24), outline="red", width=2)
            image.save(self.layer)
        self.gallery = [(str(self.source), "Image 1"), (str(self.second), "Image 2")]
        self.request = gr.Request(session_hash="browser-owner")

    def test_editor_load_normalizes_phone_orientation(self):
        source = self.root / "phone.jpg"
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (40, 32), "blue").save(source, exif=exif)
        target, editor, panel = self.ui.open_annotation([str(source)], -1)
        self.assertEqual(target, str(source))
        self.assertEqual(editor["value"].size, (32, 40))
        self.assertEqual(editor["value"].mode, "RGBA")
        self.assertTrue(panel["visible"])
        with Image.open(source) as original:
            self.assertEqual(original.size, (40, 32))
            self.assertEqual(original.getexif()[274], 6)

    def test_reorder_updates_number_without_replacing_strokes_and_removal_closes(self):
        updates = self.ui.refresh_references(list(reversed(self.gallery)), str(self.source))
        self.assertEqual(updates[1], str(self.source))
        self.assertIn("Image 2", updates[2]["label"])
        self.assertNotIn("value", updates[2])
        removed = self.ui.refresh_references([self.gallery[1]], str(self.source))
        self.assertEqual(removed[1:3], ("", None))
        self.assertEqual(removed[3]["visible"], "hidden")
        # An excess upload must not strand the remove controls.
        self.assertTrue(self.ui.refresh_references(self.gallery * 6, "")[0]["visible"])

    def test_continue_edit_uses_only_the_owned_result_and_opens_editor(self):
        with patch.object(self.ui.STUDIO, "artifact", return_value=self.source) as artifact:
            gallery, selected, controls, target, editor, panel, view = self.ui.continue_edit("job", self.request)
        artifact.assert_called_once_with("job", ":browser-owner")
        self.assertEqual(self.ui.reference_paths(gallery["value"]), [str(self.source)])
        self.assertEqual((selected, target), (0, str(self.source)))
        self.assertTrue(controls["visible"] and panel["visible"])
        self.assertEqual(editor["value"].size, (40, 32))
        self.assertEqual(view["value"], "edit")

    def test_cached_result_keeps_annotation_panel_open(self):
        cached = self.root / "gradio-cache.png"
        cached.write_bytes(self.source.read_bytes())
        controls, target, editor, panel = self.ui.refresh_references([str(cached)], str(self.source))
        self.assertTrue(controls["visible"] and panel["visible"])
        self.assertEqual(target, str(cached))
        self.assertNotIn("value", editor)

    def test_stale_background_never_submits_generation(self):
        editor = {"background": str(self.second), "layers": [str(self.layer)], "composite": str(self.source)}
        with patch.object(self.ui.STUDIO, "start") as submit:
            result = self.ui.start(
                "edit",
                self.gallery,
                "1024x1024",
                False,
                "int8",
                "offload",
                "42",
                40,
                self.request,
                str(self.source),
                editor,
            )
        submit.assert_not_called()
        self.assertIn("画像が変わりました", result[1])
        self.assertNotIn("interactive", result[2])

    def test_gradio_injects_request_between_settings_and_editor_values(self):
        demo = self.ui.on_ui_tabs()[0][0]
        function_id = next(index for index, function in demo.fns.items() if function.fn is self.ui.start_canvas)
        background = Image.open(self.source).copy()
        foreground = Image.open(self.layer).copy()
        values = [
            "edit",
            list(reversed(self.gallery)),
            "1024x1024",
            False,
            "int8",
            "offload",
            "42",
            40,
            str(self.source),
            background,
            foreground,
            None,
        ]
        with patch.object(self.ui.STUDIO, "start", return_value="accepted") as submit:
            result = asyncio.run(demo.call_function(function_id, values, requests=self.request))
        self.assertEqual(result["prediction"][0], "accepted")
        generation, owner = submit.call_args.args
        self.assertEqual(owner, ":browser-owner")
        self.assertEqual(generation.annotation_reference, 1)
        self.assertEqual(len(generation.annotation_layers), 1)
        self.assertFalse(Path(generation.annotation_layers[0]).exists())

    def test_empty_editor_keeps_ordinary_generation(self):
        with patch.object(self.ui.STUDIO, "start", return_value="accepted") as submit:
            result = self.ui.start(
                "edit",
                self.gallery,
                "1024x1024",
                False,
                "int8",
                "offload",
                "42",
                40,
                self.request,
                str(self.source),
                {"background": str(self.source), "layers": [], "composite": None},
            )
        self.assertEqual(result[0], "accepted")
        self.assertEqual(submit.call_args.args[0].annotation_reference, -1)
        self.assertEqual(submit.call_args.args[0].annotation_layers, ())

    def test_missing_job_stops_timer_and_disables_both_result_actions(self):
        with patch.object(self.ui.STUDIO, "status", side_effect=JobNotFound("gone")):
            values = self.ui.poll("missing", self.request)
        self.assertEqual(len(values), 9)
        self.assertFalse(values[3]["active"])
        self.assertFalse(values[6]["interactive"])
        self.assertFalse(values[7]["interactive"])

    def test_single_upload_opens_drawing_surface_without_extra_click(self):
        controls, target, editor, panel = self.ui.refresh_references([str(self.source)], "")
        self.assertTrue(controls["visible"] and panel["visible"])
        self.assertEqual(target, str(self.source))
        self.assertEqual(editor["value"].size, (40, 32))

    def test_next_generation_keeps_previous_image_with_explicit_label(self):
        with patch.object(self.ui.STUDIO, "start", return_value="next-job"):
            values = self.ui.start(
                "edit",
                self.gallery,
                "1024x1024",
                False,
                "int8",
                "offload",
                "42",
                40,
                self.request,
                previous_output=str(self.source),
            )
        self.assertEqual(values[0], "next-job")
        self.assertNotIn("value", values[5])
        self.assertIn("前の結果", values[5]["label"])
        self.assertFalse(values[7]["interactive"])
        self.assertFalse(values[8]["interactive"])

    def test_failed_generation_does_not_erase_previous_image(self):
        state = {"done": True, "state": "failed", "message": "failed", "elapsed": 1}
        with patch.object(self.ui.STUDIO, "status", return_value=state):
            values = self.ui.poll("failed-job", self.request)
        self.assertEqual(values[4], {"__type__": "update", "label": "生成結果"})
        self.assertFalse(values[3]["active"])


if __name__ == "__main__":
    unittest.main()
