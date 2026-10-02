"""Result publication must reject PNGs whose pixel stream cannot decode."""

from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from modules_forge import ming_image_studio, nanosaur2_studio, sensenova_u15_bridge
from modules_forge.minimax_h3_images import H3ImageError, extract_image_outputs
from modules_forge.qwen_image21 import core
from tools.tests import test_qwen_image21_mask as mask_tests
from tools.tests import test_qwen_image21_service as qwen_tests


def incomplete_pixel_png(path, width=256, height=256):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00"))
        + chunk(b"IEND", b"")
    )


class StudioOutputPixelTests(unittest.TestCase):
    def test_nanosaur_keeps_rgba_bytes_and_metadata_at_unicode_destination(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = root / "透過画像.png"
        Image.new("RGBA", (256, 256), (98, 76, 54, 0)).save(source)
        request = nanosaur2_studio.Nanosaur2Request("夜の雪", width=256, height=256, seed=42)
        result = nanosaur2_studio.save_result(
            source,
            request,
            42,
            "owned-job",
            SimpleNamespace(core_revision="fixture", comfy_version="fixture"),
            root / "結果",
        )
        self.assertEqual(Path(result["path"]).read_bytes(), source.read_bytes())
        with Image.open(result["path"]) as image:
            image.load()
            self.assertEqual(image.getpixel((0, 0)), (98, 76, 54, 0))
        metadata = json.loads(Path(result["files"][1]).read_text(encoding="utf-8"))
        self.assertEqual((metadata["prompt"], metadata["prompt_id"], metadata["seed"]), ("夜の雪", "owned-job", 42))

    def test_result_paths_reject_native_symlink_outside_output_directory(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        base = root / "runtime" / "output"
        base.mkdir(parents=True)
        outside = root / "outside.png"
        Image.new("RGB", (256, 256)).save(outside)
        link = base / "linked.png"
        try:
            link.symlink_to(outside)
        except OSError as error:
            self.skipTest(f"native symlinks unavailable: {error}")
        with self.assertRaises(core.QwenImage21Error):
            core.inside(base, link)
        self.assertFalse(sensenova_u15_bridge._is_within(link, base))
        history = {"job": {"outputs": {"save": {"images": [{"type": "output", "filename": link.name}]}}}}
        for request in (nanosaur2_studio.Nanosaur2Request("test"), ming_image_studio.MingImageRequest("test")):
            with self.subTest(request=type(request).__name__), self.assertRaises(H3ImageError):
                extract_image_outputs(history, "job", base.parent, request)

    def test_nanosaur_does_not_publish_png_with_incomplete_pixel_stream(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = root / "incomplete.png"
        incomplete_pixel_png(source)
        with Image.open(source) as image:
            self.assertEqual(image.size, (256, 256))
            image.verify()  # Chunk checksums are intact, but pixels are missing.
        with Image.open(source) as image, self.assertRaises(OSError):
            image.load()
        output = root / "outputs"
        request = nanosaur2_studio.Nanosaur2Request("test", width=256, height=256)
        with self.assertRaises(OSError):
            nanosaur2_studio.save_result(
                source,
                request,
                42,
                "job",
                SimpleNamespace(core_revision="fixture", comfy_version="fixture"),
                output,
            )
        self.assertEqual(list(output.iterdir()), [])

    def test_qwen_does_not_complete_png_with_incomplete_pixel_stream(self):
        fixture = qwen_tests.QwenServiceTests("test_success_reuses_worker_and_locks_runtime_until_eviction")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        result = fixture.worker.result

        def corrupt_pixels():
            response = result()
            directory = Path(fixture.worker.payload["job_dir"])
            incomplete_pixel_png(directory / "output.png")
            return response

        with patch.object(fixture.worker, "result", side_effect=corrupt_pixels):
            identifier = fixture.studio.start(fixture.request(), "owner")
            state = fixture.wait_done(identifier)
        self.assertEqual(state["state"], "failed", state)
        self.assertTrue(fixture.worker.closed.is_set())
        self.assertFalse(fixture.lease.owned)
        with self.assertRaises(core.QwenImage21Error):
            fixture.studio.artifact(identifier, "owner")

    def test_qwen_does_not_complete_corrupt_preserved_variant(self):
        fixture = mask_tests.EditMaskTests("test_service_snapshots_mask_and_returns_owned_raw_and_preserved_variants")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        class CorruptPreservedResident(qwen_tests.FakeResident):
            def result(self):
                response = super().result()
                directory = Path(self.payload["job_dir"])
                preserved = directory / "output-preserved.png"
                incomplete_pixel_png(preserved)
                result = core.read_json(directory / "result.json")
                result["preserved_output_path"] = str(preserved)
                core.atomic_json(directory / "result.json", result)
                return response

        worker = CorruptPreservedResident()
        studio = fixture.studio(worker)
        identifier = studio.start(fixture.request(), "owner")
        self.assertTrue(studio._jobs[identifier].done.wait(5))
        state = studio.status(identifier, "owner")
        self.assertEqual(state["state"], "failed", state)
        self.assertTrue(worker.closed.is_set())
        with self.assertRaises(core.QwenImage21Error):
            studio.artifact(identifier, "owner", "preserved")


if __name__ == "__main__":
    unittest.main()
