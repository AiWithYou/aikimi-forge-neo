"""Native Outpaint input isolation, optional download integrity, and UI job ownership."""

from __future__ import annotations

import hashlib
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from PIL import Image

from modules_forge.qwen_image21 import outpaint_lora, outpaint_native
from modules_forge.qwen_image21.core import Request, copy_inputs


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "upload.png"
        Image.new("RGBA", (256, 256), (11, 77, 133, 23)).save(self.source)
        self.job = self.root / "job"
        self.job.mkdir()
        self.request = Request(
            "outpaint",
            width=320,
            height=256,
            input_images=(str(self.source),),
            outpaint_version="v2",
            outpaint_margins=(32, 0, 32, 0),
            outpaint_feather=0,
        )

    def snapshot(self):
        payload = self.request.resolved().to_dict()
        payload["clean_input_images"] = copy_inputs(payload["input_images"], self.job)
        outpaint_native.snapshot(payload, self.job)
        return payload

    def test_snapshot_survives_upload_removed_and_preserves_exact_rgba(self):
        payload = self.snapshot()
        self.source.unlink()
        outpaint_native.validate_snapshot(payload, self.job)
        generated = Image.new("RGBA", (320, 256), (220, 90, 50, 255))
        finished = outpaint_native.finish(generated, payload)
        self.assertEqual(finished.getpixel((0, 0)), (220, 90, 50, 255))
        self.assertEqual(
            finished.crop((32, 0, 288, 256)).tobytes(), Image.new("RGBA", (256, 256), (11, 77, 133, 23)).tobytes()
        )

    def test_rejects_unsupported_combinations_before_gpu(self):
        for fields in (
            {"precision": "w4a8"},
            {"precision": "bf16"},
            {"fun_acc": True, "steps": 4},
            {"sparse_mode": "fixed"},
            {"rewrite_edit_prompt": True},
            {"preserve_unmasked": True},
            {"transparent": True},
            {"outpaint_version": "unknown"},
            {"outpaint_version": True},
            {"input_images": ()},
            {"width": 288},
            {"outpaint_feather": 1.5},
            {"outpaint_margins": (32, 0, 0, False)},
        ):
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                replace(self.request, **fields).resolved()

    def test_regular_gguf_keeps_the_same_outpaint_request(self):
        resolved = replace(self.request, precision="base_q4_k_m").resolved()
        self.assertEqual(resolved.outpaint_version, "v2")
        self.assertEqual(resolved.outpaint_margins, (32, 0, 32, 0))

    def test_no_lora_uses_the_same_snapshot_validation_and_composition(self):
        self.request = replace(self.request, outpaint_version="none")
        payload = self.snapshot()
        outpaint_native.validate_snapshot(payload, self.job)
        result = outpaint_native.finish(Image.new("RGBA", (320, 256), "white"), payload)
        self.assertEqual(result.crop((32, 0, 288, 256)).tobytes(), Image.open(self.source).tobytes())

    def test_worker_rejects_changed_reference_and_plan(self):
        payload = self.snapshot()
        payload["outpaint"]["plan"]["left"] = 0
        with self.assertRaisesRegex(ValueError, "保存情報"):
            outpaint_native.validate_snapshot(payload, self.job)
        payload = self.snapshot()
        Image.new("RGB", (320, 256), "white").save(payload["input_images"][0])
        with self.assertRaisesRegex(ValueError, "参照画像"):
            outpaint_native.validate_snapshot(payload, self.job)

    def test_worker_rejects_source_outside_job_and_feather_mismatch(self):
        payload = self.snapshot()
        payload["outpaint"]["source"] = str(self.source)
        with self.assertRaisesRegex(ValueError, "ジョブ内"):
            outpaint_native.validate_snapshot(payload, self.job)
        payload = self.snapshot()
        payload["outpaint"]["feather"] = 32
        with self.assertRaisesRegex(ValueError, "保存情報"):
            outpaint_native.validate_snapshot(payload, self.job)

    def test_no_resizing_generated_images(self):
        payload = self.snapshot()
        with self.assertRaises(ValueError):
            outpaint_native.finish(Image.new("RGBA", (256, 256)), payload)


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = b"verified adapter fixture"
        for name, value in (("SIZE", len(self.data)), ("HASHES", {"v2": hashlib.sha256(self.data).hexdigest()})):
            patcher = mock.patch.object(outpaint_lora, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_explicit_install_pins_revision_and_reuses_verified_file(self):
        with mock.patch.object(outpaint_lora.urllib.request, "urlopen", return_value=io.BytesIO(self.data)) as download:
            info = outpaint_lora.install(self.root, "v2")
            self.assertIn(outpaint_lora.REVISION, download.call_args.args[0])
            self.assertEqual(Path(info["path"]).read_bytes(), self.data)
        with mock.patch.object(outpaint_lora.urllib.request, "urlopen", side_effect=AssertionError("No download")):
            self.assertEqual(outpaint_lora.install(self.root, "v2"), info)

    def test_corrupt_download_never_becomes_installed(self):
        with mock.patch.object(outpaint_lora.urllib.request, "urlopen", return_value=io.BytesIO(b"x" * len(self.data))):
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                outpaint_lora.install(self.root, "v2")
        self.assertFalse(outpaint_lora.adapter_path(self.root, "v2").exists())
        self.assertEqual(list((self.root / "outpaint").glob("*.part")), [])
        with self.assertRaises(ValueError):
            outpaint_lora.installed(self.root, "v2")

    def test_verified_load_detects_same_size_corruption(self):
        with mock.patch.object(outpaint_lora.urllib.request, "urlopen", return_value=io.BytesIO(self.data)):
            outpaint_lora.install(self.root, "v2")
        outpaint_lora.adapter_path(self.root, "v2").write_bytes(b"x" * len(self.data))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            outpaint_lora.installed(self.root, "v2", verify=True)


if __name__ == "__main__":
    unittest.main()
