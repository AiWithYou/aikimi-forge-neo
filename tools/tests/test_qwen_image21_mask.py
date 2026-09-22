"""Explicit edit masks preserve every outside RGBA byte and both result variants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image, ImageChops, ImageDraw

from modules_forge.qwen_image21 import annotations, core, service
from tools.tests.test_qwen_image21_service import FakeResident, Lease, Residency, installed_runtime, load_ui


class EditMaskTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen-edit-mask-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.original = self.root / "original.png"
        self.mask = self.root / "mask.png"
        with Image.new("RGBA", (256, 256), (12, 34, 56, 72)) as image:
            image.putpixel((0, 0), (98, 76, 54, 0))
            image.save(self.original)
        with Image.new("L", (256, 256), 0) as mask:
            ImageDraw.Draw(mask).rectangle((70, 60, 180, 170), fill=255)
            ImageDraw.Draw(mask).rectangle((95, 85, 110, 100), fill=0)
            mask.save(self.mask)

    def request(self, **changes):
        fields = {
            "width": 256,
            "height": 256,
            "input_images": (str(self.original),),
            "preserve_unmasked": True,
            "edit_mask_reference": 0,
            "edit_mask_path": str(self.mask),
        }
        fields.update(changes)
        return core.Request("Change the selected region.", **fields)

    def test_mask_requires_exact_original_output_size_and_nonempty_coverage(self):
        self.assertEqual(self.request().resolved().edit_mask_path, str(self.mask))
        with self.assertRaisesRegex(core.QwenImage21Error, "同じ出力サイズ"):
            self.request(width=512).resolved()
        Image.new("L", (128, 256), 255).save(self.mask)
        with self.assertRaisesRegex(core.QwenImage21Error, "サイズ"):
            self.request().resolved()
        Image.new("L", (256, 256), 0).save(self.mask)
        with self.assertRaisesRegex(core.QwenImage21Error, "空"):
            self.request().resolved()

    def test_disabled_mask_and_edit_rewrite_remain_optional(self):
        request = self.request(preserve_unmasked=False, edit_mask_path="missing.png").resolved()
        self.assertEqual((request.edit_mask_reference, request.edit_mask_path), (-1, ""))
        self.assertFalse(request.rewrite_edit_prompt)
        for fields in ({"preserve_unmasked": "yes"}, {"rewrite_edit_prompt": "yes"}, {"mask_feather": float("nan")}):
            with self.subTest(fields=fields), self.assertRaises(core.QwenImage21Error):
                self.request(**fields).resolved()

    def test_inside_feather_preserves_outside_hidden_rgb_alpha_and_holes_exactly(self):
        with Image.open(self.original) as original, Image.open(self.mask) as mask:
            generated = Image.new("RGBA", original.size, (210, 180, 90, 250))
            for feather in (0, 8):
                with self.subTest(feather=feather):
                    result = annotations.composite_preserving_outside(original, generated, mask, feather)
                    outside = mask.point(lambda value: 255 if value == 0 else 0)
                    for channel in ImageChops.difference(original, result).split():
                        self.assertIsNone(ImageChops.multiply(channel, outside).getbbox())
                    for x, y in ((0, 0), (69, 60), (180, 171), (100, 90), (255, 255)):
                        self.assertEqual(result.getpixel((x, y)), original.getpixel((x, y)))
                    self.assertEqual(result.mode, "RGBA")
                    self.assertEqual(result.getpixel((140, 130)), generated.getpixel((140, 130)))
                    self.assertNotEqual(result.getpixel((70, 60)), original.getpixel((70, 60)))
                    if feather:
                        self.assertNotEqual(result.getpixel((70, 60)), generated.getpixel((70, 60)))
            with self.assertRaisesRegex(core.QwenImage21Error, "サイズ"):
                annotations.composite_preserving_outside(original, Image.new("RGBA", (128, 256)), mask)

    def test_uploaded_transparent_pixels_never_become_edit_coverage(self):
        with Image.new("RGBA", (256, 256), (255, 255, 255, 0)) as image:
            image.putpixel((100, 100), (255, 255, 255, 255))
            image.putpixel((90, 90), (0, 0, 0, 255))
            image.save(self.mask)
        _, bounds = annotations.validate_edit_mask(str(self.original), str(self.mask))
        self.assertEqual(bounds, (100, 100, 101, 101))

    def test_transparent_boundaries_blend_premultiplied_colors_without_dark_fringes(self):
        cases = [
            ((0, 0, 0, 0), (255, 0, 0, 255), 128, (255, 0, 0, 128)),
            ((0, 255, 0, 0), (255, 0, 0, 255), 128, (255, 0, 0, 128)),
            ((0, 0, 255, 255), (255, 0, 0, 0), 128, (0, 0, 255, 127)),
            ((0, 0, 0, 0), (128, 67, 39, 1), 128, (128, 67, 39, 1)),
            ((10, 20, 30, 128), (220, 170, 120, 64), 128, (80, 70, 60, 96)),
            ((91, 72, 53, 0), (220, 170, 120, 64), 0, (91, 72, 53, 0)),
            ((91, 72, 53, 0), (128, 67, 39, 1), 255, (128, 67, 39, 1)),
            ((10, 20, 30, 128), (211, 73, 19, 0), 255, (211, 73, 19, 0)),
        ]
        original, generated, mask = (
            Image.new("RGBA", (len(cases), 1)),
            Image.new("RGBA", (len(cases), 1)),
            Image.new("L", (len(cases), 1)),
        )
        for index, (source, target, weight, _) in enumerate(cases):
            original.putpixel((index, 0), source)
            generated.putpixel((index, 0), target)
            mask.putpixel((index, 0), weight)
        result = annotations.composite_preserving_outside(original, generated, mask)
        for index, (*_, expected) in enumerate(cases):
            with self.subTest(index=index):
                self.assertEqual(result.getpixel((index, 0)), expected)

    def test_snapshot_and_worker_hook_keep_clean_original_and_raw_output(self):
        job = self.root / "job"
        job.mkdir()
        clean = core.copy_inputs([str(self.original)], job)
        info = annotations.snapshot_edit_mask(clean, 0, str(self.mask), 5, job)
        request = {"preserve_unmasked": True, "edit_mask": info}
        generated = Image.new("RGBA", (256, 256), (220, 110, 22, 180))
        raw = job / "output.png"
        generated.save(raw)
        raw_bytes = raw.read_bytes()
        self.original.unlink()
        self.mask.unlink()
        result = annotations.save_preserved_output(request, generated, job)
        self.assertEqual(raw.read_bytes(), raw_bytes)
        self.assertEqual(result["output_paths"], [str(job / "output-preserved.png"), str(raw)])
        with Image.open(result["preserved_output_path"]) as image:
            self.assertEqual(image.getpixel((0, 0)), (98, 76, 54, 0))
            self.assertEqual(image.getpixel((140, 130)), generated.getpixel((140, 130)))
        self.assertEqual(annotations.save_preserved_output({}, generated, job), {})
        with self.assertRaisesRegex(core.QwenImage21Error, "外側"):
            annotations.save_preserved_output(
                {"preserve_unmasked": True, "edit_mask": {**info, "original_path": str(self.root / "elsewhere.png")}},
                generated,
                job,
            )

    def test_guide_strokes_do_not_create_an_edit_mask(self):
        layer = self.root / "guide.png"
        Image.new("RGBA", (256, 256), (255, 0, 0, 100)).save(layer)
        request = core.Request(
            "edit", input_images=(str(self.original),), annotation_reference=0, annotation_layers=(str(layer),)
        ).resolved()
        self.assertFalse(request.preserve_unmasked)
        self.assertEqual(request.edit_mask_path, "")

    def studio(self, worker):
        runtime = self.root / "runtime"
        installed_runtime(runtime)
        studio = service.Studio(
            runtime,
            self.root / "outputs",
            ownership_factory=Lease,
            release_vram=lambda: None,
            worker_factory=lambda *_: worker,
            residency=Residency(),
            poll_interval=0.01,
        )
        self.addCleanup(studio.shutdown)
        return studio

    def test_service_snapshots_mask_and_returns_owned_raw_and_preserved_variants(self):
        class MaskResident(FakeResident):
            def result(self):
                response = super().result()
                directory = Path(self.payload["job_dir"])
                request = core.read_json(directory / "request.json")
                result = core.read_json(directory / "result.json")
                with Image.open(directory / "output.png") as image:
                    result.update(annotations.save_preserved_output(request, image, directory))
                core.atomic_json(directory / "result.json", result)
                return response

        studio = self.studio(MaskResident())
        guide = self.root / "guide.png"
        Image.new("RGBA", (256, 256), (255, 0, 0, 200)).save(guide)
        identifier = studio.start(self.request(annotation_reference=0, annotation_layers=(str(guide),)), "owner")
        self.assertTrue(studio._jobs[identifier].done.wait(5))
        state = studio.status(identifier, "owner")
        self.assertEqual(state["state"], "complete", state)
        paths = studio.artifacts(identifier, "owner")
        self.assertEqual([path.name for path in paths], ["output-preserved.png", "output.png"])
        self.assertEqual(studio.artifact(identifier, "owner", "original"), paths[1])
        self.assertEqual(state["output_path"], str(paths[0]))
        saved = core.read_json(paths[0].with_name("request.json"))
        self.assertEqual(saved["edit_mask_path"], str(paths[0].with_name("edit-mask.png")))
        self.assertEqual(saved["edit_mask"]["original_path"], saved["clean_input_images"][0])
        self.assertNotEqual(saved["input_images"][0], saved["clean_input_images"][0])
        with Image.open(paths[0]) as preserved:
            self.assertEqual(preserved.getpixel((0, 0)), (98, 76, 54, 0))
        with self.assertRaises(service.JobNotFound):
            studio.artifact(identifier, "another-owner", "original")

    def test_service_requires_requested_preserved_result(self):
        studio = self.studio(FakeResident())
        identifier = studio.start(self.request(), "owner")
        self.assertTrue(studio._jobs[identifier].done.wait(5))
        self.assertEqual(studio.status(identifier, "owner")["state"], "failed")

    def test_edit_rewriter_preflight_is_independent_and_before_gpu_lease(self):
        studio = self.studio(FakeResident())
        with (
            patch.object(
                service, "rewriter_manifest", side_effect=core.QwenImage21Error("missing edit model")
            ) as manifest,
            patch.object(studio, "_ownership_factory") as acquire,
        ):
            with self.assertRaisesRegex(core.QwenImage21Error, "missing edit model"):
                studio.start(self.request(rewrite_edit_prompt=True), "owner")
            manifest.assert_called_once_with(studio.runtime, editing=True)
            acquire.assert_not_called()

    def test_canvas_mask_and_edit_rewrite_reach_request_without_reusing_guide(self):
        ui = load_ui()
        background = annotations.annotation_preview(str(self.original))
        with Image.open(self.mask) as mask:
            foreground = Image.new("RGBA", mask.size, (0, 180, 220, 0))
            foreground.putalpha(mask)
        submitted = []

        def submit(request, owner):
            checked = request.resolved()
            with Image.open(checked.edit_mask_path) as mask:
                submitted.append((checked, mask.tobytes()))
            return "accepted"

        with patch.object(ui.STUDIO, "start", side_effect=submit):
            result = ui.start_canvas(
                "edit",
                [str(self.original)],
                "reference",
                False,
                "int8",
                "offload",
                "1",
                2,
                gr.Request(session_hash="owner"),
                rewrite_edit_prompt=True,
                preserve_unmasked=True,
                mask_target=str(self.original),
                mask_background=background,
                mask_foreground=foreground,
            )
        self.assertEqual(result[0], "accepted", result)
        request, actual_mask = submitted[0]
        with Image.open(self.mask) as mask:
            self.assertEqual(actual_mask, mask.tobytes())
        self.assertTrue(request.rewrite_edit_prompt)
        self.assertEqual(request.annotation_reference, -1)
        self.assertEqual((request.width, request.height), (256, 256))
        self.assertFalse(Path(request.edit_mask_path).exists())

    def test_result_selection_does_not_reset_after_viewing_raw(self):
        ui = load_ui()
        state = {"done": True, "state": "complete", "preserved_output_path": "fixed.png"}
        with patch.object(ui.STUDIO, "status", return_value=state):
            first = ui.result_variants("job", gr.Request(session_hash="owner"))
            following = ui.result_variants("job", gr.Request(session_hash="owner"), "job")
        self.assertEqual(first[0]["value"], "preferred")
        self.assertTrue(first[0]["visible"])
        self.assertNotIn("value", following[0])

    def test_mask_surface_is_cleared_on_source_change_or_disable(self):
        ui = load_ui()
        other = self.root / "other.png"
        Image.new("RGBA", (256, 256), "blue").save(other)
        paths = [str(self.original), str(other)]
        unchanged = ui.refresh_mask(list(reversed(paths)), str(self.original), str(self.original))
        self.assertNotIn("value", unchanged[2])
        changed = ui.refresh_mask(paths, str(other), str(self.original))
        self.assertEqual(changed[0], str(other))
        self.assertIsNone(changed[2])
        self.assertIsNone(changed[3])
        self.assertEqual(ui.refresh_mask(paths, str(other), str(other), False), ("", None, None, None))

    def test_uploaded_mask_uses_selected_reference_and_rejects_stale_paint_background(self):
        ui = load_ui()
        accepted = []

        def submit(request, owner):
            accepted.append(request.resolved())
            return "accepted"

        with patch.object(ui.STUDIO, "start", side_effect=submit):
            values = ui.start_canvas(
                "edit",
                [str(self.original)],
                "reference",
                False,
                "int8",
                "offload",
                "1",
                2,
                gr.Request(session_hash="owner"),
                preserve_unmasked=True,
                mask_target=str(self.original),
                mask_source="upload",
                mask_upload=str(self.mask),
            )
            self.assertEqual(values[0], "accepted", values)
            foreground = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
            stale = ui.start_canvas(
                "edit",
                [str(self.original)],
                "reference",
                False,
                "int8",
                "offload",
                "1",
                2,
                gr.Request(session_hash="owner"),
                preserve_unmasked=True,
                mask_target=str(self.original),
                mask_background=Image.new("RGBA", (256, 256), "red"),
                mask_foreground=foreground,
            )
        self.assertEqual(len(accepted), 1)
        self.assertIn("画像が変わりました", stale[1])


if __name__ == "__main__":
    unittest.main()
