"""Temporary image regressions using real Pillow and Gradio, without a server."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
from PIL import Image

from modules import ui_tempdir


class TempImageEfficiencyTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.registry = set()
        self.shared = SimpleNamespace(
            opts=SimpleNamespace(temp_dir=""), demo=SimpleNamespace(temp_file_sets=[self.registry])
        )
        shared_patch = patch.object(ui_tempdir, "shared", self.shared)
        shared_patch.start()
        self.addCleanup(shared_patch.stop)
        self.image = Image.new("RGB", (12, 8), (10, 20, 30))
        self.addCleanup(self.image.close)

    def test_register_updates_existing_registry_without_copying(self):
        self.registry.update(str(self.root / f"old-{i}.png") for i in range(100))
        target = self.root / "new.png"
        ui_tempdir.register_tmp_file(self.shared.demo, target)
        ui_tempdir.register_tmp_file(self.shared.demo, target)
        self.assertIs(self.shared.demo.temp_file_sets[0], self.registry)
        self.assertEqual(len(self.registry), 101)
        self.assertTrue(ui_tempdir.check_tmp_file(self.shared.demo, target))
        self.assertFalse(ui_tempdir.check_tmp_file(self.shared.demo, self.root / "unknown.png"))

    def test_custom_directory_is_created_and_png_metadata_preserved(self):
        target = self.root / "custom" / "nested"
        self.shared.opts.temp_dir = str(target)
        self.image.info.update(parameters="prompt\nSteps: 20", note="日本語", ignored=42)
        filename = Path(ui_tempdir.save_pil_to_file(self.image, cache_dir=self.root / "unused"))
        self.assertEqual(filename.parent, target)
        self.assertTrue(filename.name.startswith(ui_tempdir.MANAGED_TEMP_PREFIX))
        self.assertFalse((self.root / "unused").exists())
        with Image.open(filename) as reopened:
            self.assertEqual(reopened.info["parameters"], self.image.info["parameters"])
            self.assertEqual(reopened.info["note"], "日本語")
            self.assertNotIn("ignored", reopened.info)
            self.assertEqual(reopened.tobytes(), self.image.tobytes())

    def test_failed_save_removes_only_its_partial_file_and_reraises(self):
        unrelated = self.root / "keep.png"
        unrelated.write_bytes(b"keep")
        error = OSError("simulated encode failure")

        def fail(filename, **kwargs):
            Path(filename).write_bytes(b"partial")
            raise error

        with patch.object(self.image, "save", side_effect=fail):
            with self.assertRaises(OSError) as caught:
                ui_tempdir.save_pil_to_file(self.image, cache_dir=self.root)
        self.assertIs(caught.exception, error)
        self.assertEqual(list(self.root.iterdir()), [unrelated])
        self.assertEqual(unrelated.read_bytes(), b"keep")

    def test_cleanup_error_does_not_mask_original_save_failure(self):
        error = OSError("original encode failure")
        with (
            patch.object(self.image, "save", side_effect=error),
            patch.object(ui_tempdir.os, "unlink", side_effect=PermissionError),
        ):
            with self.assertRaises(OSError) as caught:
                ui_tempdir.save_pil_to_file(self.image, cache_dir=self.root)
        self.assertIs(caught.exception, error)

    def test_already_saved_image_is_registered_without_reencoding(self):
        target = self.root / "saved.png"
        self.image.save(target)
        self.image.already_saved_as = str(target)
        before = target.read_bytes()
        with patch.object(self.image, "save", side_effect=AssertionError("must reuse saved file")):
            filename = ui_tempdir.save_pil_to_file(self.image, cache_dir=self.root)
        self.assertEqual(filename, str(target))
        self.assertTrue(ui_tempdir.check_tmp_file(self.shared.demo, target))
        self.assertTrue(ui_tempdir.check_tmp_file(self.shared.demo, filename))
        self.assertEqual(target.read_bytes(), before)

    def test_saved_gallery_image_passes_gradio_cache_file_reader(self):
        target = self.root / "saved.png"
        self.image.save(target)
        self.image.already_saved_as = str(target)
        with gr.Blocks():
            gallery = gr.Gallery(format="png")
        with patch.object(gr.processing_utils, "save_pil_to_cache", ui_tempdir.save_pil_to_file):
            result = gallery.postprocess([self.image])
        cached = asyncio.run(gallery.async_move_resource_to_block_cache(result.root[0].image.path))
        with Image.open(cached) as reopened:
            self.assertEqual(reopened.tobytes(), self.image.tobytes())

    def test_non_png_formats_skip_png_metadata_work(self):
        self.image.info["parameters"] = "unused for JPEG"
        with patch.object(ui_tempdir.PngImagePlugin, "PngInfo", side_effect=AssertionError("PNG-only metadata")):
            filename = Path(ui_tempdir.save_pil_to_file(self.image, cache_dir=self.root, format="jpeg"))
        self.assertEqual(filename.suffix, ".jpg")
        with Image.open(filename) as reopened:
            self.assertEqual(reopened.format, "JPEG")
            self.assertEqual(reopened.size, self.image.size)

    def test_gradio_gallery_postprocess_keeps_png_metadata(self):
        self.shared.opts.temp_dir = str(self.root / "gallery")
        self.image.info["parameters"] = "gallery metadata"
        # Replace only the intended serializer, not Gradio's file validators.
        with patch.object(gr.processing_utils, "save_pil_to_cache", ui_tempdir.save_pil_to_file):
            result = gr.Gallery(format="png").postprocess([self.image])
        with Image.open(result.root[0].image.path) as reopened:
            self.assertEqual(reopened.info["parameters"], "gallery metadata")


if __name__ == "__main__":
    unittest.main()
