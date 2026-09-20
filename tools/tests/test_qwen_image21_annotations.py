"""Visual region guides must bind to their source and preserve clean snapshots."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from modules_forge.qwen_image21 import annotations, core, service


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen-annotations-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.first = self.root / "first.png"
        self.second = self.root / "second.png"
        self.background = self.root / "editor-background.png"
        self.layer = self.root / "layer.png"
        self.empty = self.root / "empty.png"
        Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(self.first)
        Image.new("RGBA", (8, 6), (80, 100, 120, 200)).save(self.second)
        Image.new("RGBA", (8, 6), (10, 20, 30, 255)).save(self.background)
        with Image.new("RGBA", (8, 6), (0, 0, 0, 0)) as layer:
            ImageDraw.Draw(layer).rectangle((2, 1, 5, 3), fill=(255, 0, 0, 128))
            layer.save(self.layer)
        Image.new("RGBA", (8, 6), (100, 100, 100, 0)).save(self.empty)
        self.paths = [str(self.first), str(self.second)]
        self.editor = {
            "background": str(self.background),
            "layers": [str(self.layer)],
            # The browser composite must never be trusted or opened.
            "composite": str(self.root / "nonexistent-untrusted-composite.png"),
        }

    def test_cached_background_matches_and_reorder_follows_source(self):
        index, layers = annotations.resolve_annotation(self.paths, str(self.first), self.editor)
        self.assertEqual(index, 0)
        self.assertEqual(layers, (str(self.layer),))
        index, _ = annotations.resolve_annotation(list(reversed(self.paths)), str(self.first), self.editor)
        self.assertEqual(index, 1)

    def test_missing_or_duplicated_target_with_strokes_is_rejected(self):
        for paths in ([str(self.second)], [str(self.first), str(self.first)]):
            with self.subTest(paths=paths), self.assertRaisesRegex(core.QwenImage21Error, "選択し直"):
                annotations.resolve_annotation(paths, str(self.first), self.editor)

    def test_result_moved_into_gradio_cache_matches_only_identical_file(self):
        cached = self.root / "cached-result.png"
        cached.write_bytes(self.first.read_bytes())
        index, _ = annotations.resolve_annotation([str(self.second), str(cached)], str(self.first), self.editor)
        self.assertEqual(index, 1)
        duplicate = self.root / "duplicate.png"
        duplicate.write_bytes(self.first.read_bytes())
        with self.assertRaisesRegex(core.QwenImage21Error, "複数"):
            annotations.reference_index([str(cached), str(duplicate)], str(self.first))

    def test_stale_editor_background_is_rejected(self):
        with Image.open(self.background) as image:
            image.putpixel((0, 0), (50, 20, 30, 255))
            image.save(self.background)
        with self.assertRaisesRegex(core.QwenImage21Error, "画像が変わり"):
            annotations.resolve_annotation(self.paths, str(self.first), self.editor)

    def test_transparent_hidden_rgb_does_not_make_source_stale(self):
        for path, color in ((self.first, (10, 20, 30, 0)), (self.background, (0, 0, 0, 0))):
            with Image.open(path) as image:
                image.putpixel((0, 0), color)
                image.save(path)
        self.assertEqual(annotations.resolve_annotation(self.paths, str(self.first), self.editor)[0], 0)

    def test_alpha_change_in_background_is_stale_even_with_same_rgb(self):
        with Image.open(self.background) as image:
            image.putpixel((0, 0), (10, 20, 30, 200))
            image.save(self.background)
        with self.assertRaisesRegex(core.QwenImage21Error, "画像が変わり"):
            annotations.resolve_annotation(self.paths, str(self.first), self.editor)

    def test_opaque_editor_preview_preserves_original_alpha_and_binds_to_source(self):
        original_bytes = self.second.read_bytes()
        with annotations.annotation_preview(str(self.second)) as preview:
            self.assertEqual(preview.getchannel("A").getextrema(), (255, 255))
            preview.save(self.background)
        index, _ = annotations.resolve_annotation(self.paths, str(self.second), self.editor)
        self.assertEqual(index, 1)
        self.assertEqual(self.second.read_bytes(), original_bytes)
        with Image.open(self.background) as preview:
            preview.putpixel((0, 0), (20, 30, 40, 255))
            preview.save(self.background)
        with self.assertRaisesRegex(core.QwenImage21Error, "画像が変わり"):
            annotations.resolve_annotation(self.paths, str(self.second), self.editor)

    def test_no_target_or_no_strokes_is_no_op(self):
        for target, editor in (
            (None, self.editor),
            (str(self.first), None),
            (str(self.first), {**self.editor, "layers": []}),
            (str(self.first), {**self.editor, "layers": [str(self.empty)]}),
        ):
            with self.subTest(target=target, editor=editor):
                self.assertEqual(annotations.resolve_annotation(self.paths, target, editor), (-1, ()))

    def test_layer_requires_alpha_same_size_and_bounded_count(self):
        wrong = self.root / "wrong.png"
        Image.new("RGB", (8, 6), "red").save(wrong)
        with self.assertRaisesRegex(core.QwenImage21Error, "アルファ"):
            annotations.resolve_annotation(self.paths, str(self.first), {**self.editor, "layers": [str(wrong)]})
        Image.new("RGBA", (6, 8), "red").save(wrong)
        with self.assertRaisesRegex(core.QwenImage21Error, "サイズ"):
            annotations.resolve_annotation(self.paths, str(self.first), {**self.editor, "layers": [str(wrong)]})
        with self.assertRaisesRegex(core.QwenImage21Error, "最大8"):
            annotations.resolve_annotation(
                self.paths, str(self.first), {**self.editor, "layers": [str(self.layer)] * 9}
            )

    def test_request_validates_annotation_pair_before_gpu_work(self):
        invalid = (
            {"annotation_reference": -1, "annotation_layers": (str(self.layer),)},
            {"annotation_reference": 0, "annotation_layers": ()},
            {"annotation_reference": 2, "annotation_layers": (str(self.layer),)},
            {"annotation_reference": True, "annotation_layers": (str(self.layer),)},
            {"annotation_reference": 0, "annotation_layers": (str(self.empty),)},
            {"annotation_reference": 0, "annotation_layers": "not-a-list"},
        )
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(core.QwenImage21Error):
                core.Request("edit", input_images=tuple(self.paths), **fields).resolved()
        request = core.Request(
            "edit", input_images=tuple(self.paths), annotation_reference=0, annotation_layers=(str(self.layer),)
        ).resolved()
        self.assertEqual(request.annotation_reference, 0)
        self.assertEqual(request.annotation_layers, (str(self.layer),))

    def test_snapshot_replaces_only_target_and_survives_uploaded_file_removal(self):
        original_bytes = [Path(path).read_bytes() for path in self.paths]
        job = self.root / "job"
        job.mkdir()
        clean = core.copy_inputs(self.paths, job)
        model, suffix, info = annotations.snapshot_annotation(clean, 1, (str(self.layer),), job)
        self.assertEqual(model[0], clean[0])
        self.assertEqual(model[1], str(job / "reference-02-annotated.png"))
        self.assertEqual(len(model), len(self.paths))
        self.assertIn("Image 2", suffix)
        self.assertIn("remove the annotation marks", suffix)
        self.assertEqual(info["reference_index"], 2)
        self.assertEqual(info["bounds"], [2, 1, 6, 4])
        self.assertEqual(info["original_path"], clean[1])
        self.assertEqual([Path(path).read_bytes() for path in self.paths], original_bytes)
        with Image.open(clean[1]) as original, Image.open(self.layer) as layer:
            expected = Image.alpha_composite(original, layer)
            with Image.open(model[1]) as generated:
                self.assertEqual(generated.mode, "RGBA")
                self.assertEqual(generated.tobytes(), expected.tobytes())
            self.assertEqual(original.getpixel((2, 1)), (80, 100, 120, 200))
        for path in (self.first, self.second, self.background, self.layer, self.empty):
            path.unlink()
        for path in [*model, info["original_path"], *info["layer_paths"]]:
            with Image.open(path) as image:
                image.load()
        self.assertEqual(annotations.snapshot_annotation(clean, -1, (), job), (clean, "", {}))

    def test_multiple_layers_keep_order_and_union_bounds(self):
        second_layer = self.root / "second-layer.png"
        with Image.new("RGBA", (8, 6), (0, 0, 0, 0)) as image:
            ImageDraw.Draw(image).rectangle((4, 2, 7, 5), fill=(0, 255, 0, 192))
            image.save(second_layer)
        job = self.root / "job"
        job.mkdir()
        clean = core.copy_inputs(self.paths, job)
        model, _, info = annotations.snapshot_annotation(clean, 0, (str(self.layer), str(second_layer)), job)
        self.assertEqual(info["bounds"], [2, 1, 8, 6])
        with Image.open(clean[0]) as base, Image.open(self.layer) as first, Image.open(second_layer) as second:
            expected = Image.alpha_composite(Image.alpha_composite(base, first), second)
            with Image.open(model[0]) as generated:
                self.assertEqual(generated.tobytes(), expected.tobytes())

    def test_snapshot_rejects_clean_file_outside_job(self):
        job = self.root / "job"
        job.mkdir()
        with self.assertRaisesRegex(core.QwenImage21Error, "外側"):
            annotations.snapshot_annotation(self.paths, 0, (str(self.layer),), job)


class AnnotationServiceTests(unittest.TestCase):
    def test_service_archives_original_and_layers_before_worker_uses_annotated_reference(self):
        from tools.tests.test_qwen_image21_service import FakeResident, Lease, Residency, installed_runtime

        with tempfile.TemporaryDirectory(prefix="qwen-annotation-service-") as temporary:
            root = Path(temporary)
            runtime = root / "runtime"
            installed_runtime(runtime)
            original, layer = root / "upload.png", root / "strokes.png"
            Image.new("RGBA", (32, 32), (20, 30, 40, 200)).save(original)
            with Image.new("RGBA", (32, 32), (0, 0, 0, 0)) as image:
                ImageDraw.Draw(image).ellipse((8, 8, 22, 22), outline=(255, 0, 0, 255), width=2)
                image.save(layer)
            worker = FakeResident("wait")
            studio = service.Studio(
                runtime,
                root / "outputs",
                ownership_factory=Lease,
                release_vram=lambda: None,
                worker_factory=lambda *_: worker,
                residency=Residency(),
                poll_interval=0.01,
            )
            try:
                request = core.Request(
                    "Make the indicated object blue.",
                    width=256,
                    height=256,
                    input_images=(str(original),),
                    annotation_reference=0,
                    annotation_layers=(str(layer),),
                )
                identifier = studio.start(request, "owner")
                original.unlink()
                layer.unlink()
                saved = core.read_json(root / "outputs" / identifier / "request.json")
                self.assertEqual(saved["user_prompt"], request.prompt)
                self.assertTrue(saved["prompt"].startswith(request.prompt))
                self.assertIn("Image 1", saved["prompt"])
                self.assertEqual(saved["annotation_layers"], saved["annotation"]["layer_paths"])
                self.assertEqual(saved["input_images"], [saved["annotation"]["annotated_path"]])
                for path in [saved["annotation"]["original_path"], *saved["annotation_layers"], *saved["input_images"]]:
                    self.assertTrue(Path(path).is_file())
                    self.assertTrue(Path(path).is_relative_to(root / "outputs" / identifier))
                worker.mode = "complete"
                self.assertTrue(studio._jobs[identifier].done.wait(5))
                self.assertEqual(studio.status(identifier, "owner")["state"], "complete")
            finally:
                studio.shutdown()


if __name__ == "__main__":
    unittest.main()
