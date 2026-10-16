"""Uploaded pixels, completed artifacts and confirmed worker exit own the GPU lease."""

import hashlib
import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, PngImagePlugin

from modules_forge.flux3_klein.core import validate_request
from modules_forge.flux3_klein.service import Service


class ServiceTests(unittest.TestCase):
    def test_redirected_output_ancestor_is_rejected_before_acquiring_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            service = Service(root, outputs=root / "outputs")
            original = Path.lstat

            def redirected(path):
                return (
                    SimpleNamespace(st_mode=stat.S_IFDIR, st_reparse_tag=0xA0000003) if path == root else original(path)
                )

            with (
                patch.object(Path, "lstat", redirected),
                patch("modules_forge.flux3_klein.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership") as ownership,
                patch("modules_forge.flux3_klein.service.threading.Thread"),
            ):
                with self.assertRaises(ValueError):
                    service.start({"prompt": "a fox"})
                ownership.assert_not_called()
            self.assertFalse(service.outputs.exists())

    def test_completion_during_poll_keeps_one_consistent_snapshot(self):
        from modules_forge.flux3_klein.api import public_result

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            service.jobs["job"] = job
            original = Path.read_text

            def finish_during_progress(path, *args, **kwargs):
                if path == root / "progress.json":
                    with service._guard:
                        job["status"] = "complete"
                    return json.dumps({"stage": "complete", "current": 1, "total": 1})
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", finish_during_progress):
                result = service.poll("job")
            self.assertEqual(result["status"], "running")
            self.assertNotIn("files", result)
            self.assertEqual(public_result(result)["status"], "running")
            Image.new("RGB", (64, 64)).save(root / "output.png")
            (root / "result.json").write_text(
                json.dumps({"image_path": str(root / "output.png"), "width": 64, "height": 64}), encoding="utf-8"
            )
            self.assertIn("image", public_result(service.poll("job")))

    def test_transparent_reference_pixels_survive_until_stage_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root, outputs=root / "outputs")
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.flux3_klein.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.flux3_klein.service.threading.Thread"),
            ):
                identifier = service.start({"prompt": "a fox", "images": [Image.new("RGBA", (64, 64), (255, 0, 0, 0))]})
            request = json.loads((root / "outputs" / identifier / "request.json").read_text(encoding="utf-8"))
            with Image.open(request["images"][0]) as image:
                self.assertEqual(image.mode, "RGBA")
                self.assertEqual(image.getpixel((0, 0)), (255, 0, 0, 0))

    def test_reference_pixels_are_snapshotted_in_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root, outputs=root / "outputs")
            lease = Mock()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.flux3_klein.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.flux3_klein.service.threading.Thread"),
            ):
                job = service.start(
                    {
                        "prompt": "preserve order",
                        "pup": False,
                        "images": [Image.new("RGB", (16, 16), "red"), Image.new("RGB", (32, 16), "blue")],
                    }
                )
            request = json.loads((root / "outputs" / job / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(len(request["images"]), 2)
            for path, expected in zip(request["images"], [(255, 0, 0), (0, 0, 255)], strict=True):
                with Image.open(path) as image:
                    self.assertEqual(image.convert("RGB").getpixel((0, 0)), expected)
            self.assertEqual(len(request["reference_sha256"]), 2)

    def test_caller_paths_are_rejected_without_reading(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            with (
                patch("modules_forge.flux3_klein.service.Image.open") as opener,
                self.assertRaisesRegex(ValueError, "アップロード"),
            ):
                service.start({"prompt": "test", "images": ["C:/private.png"]})
            opener.assert_not_called()

    def test_spawn_failure_confirms_exit_before_releasing(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            service.worker = Mock()
            service.worker.start.side_effect = RuntimeError("spawn failed")
            service.worker.close.side_effect = lambda: events.append("stopped")
            lease = Mock()
            lease.release.side_effect = lambda: events.append("released")
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(events, ["stopped", "released"])
            self.assertEqual(job["status"], "error")

    def test_success_requires_a_real_png_and_matching_dimensions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            Image.new("RGB", (64, 64)).save(root / "output.png")
            (root / "result.json").write_text(
                json.dumps({"image_path": str(root / "output.png"), "width": 128, "height": 64}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "サイズ"):
                service.validate_result(root)

    def test_completion_rechecks_reference_content_and_redirects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            reference = root / "reference-00.png"
            Image.new("RGB", (64, 64), "red").save(reference)
            digest = hashlib.sha256(reference.read_bytes()).hexdigest()
            request = validate_request({"prompt": "a fox", "width": 64, "height": 64, "seed": 7})
            request.update(images=[str(reference)], reference_sha256=[digest])
            record = {
                key: request[key] for key in ("model", "precision", "width", "height", "steps", "cfg", "seed", "boxes")
            }
            record.update(
                input_prompt=request["prompt"], prompt=request["prompt"], reference_hashes=[digest], lora=None
            )
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("flux3_klein", json.dumps(record))
            Image.new("RGB", (64, 64)).save(root / "output.png", pnginfo=metadata)
            (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
            record["image_path"] = str(root / "output.png")
            (root / "result.json").write_text(json.dumps(record), encoding="utf-8")
            self.assertEqual(Service.validate_result(root)["seed"], 7)
            original = Path.lstat

            def redirected(path):
                return SimpleNamespace(st_mode=stat.S_IFLNK) if path == reference else original(path)

            with self.subTest(mutation="redirect"), patch.object(Path, "lstat", redirected):
                with self.assertRaises(ValueError):
                    Service.validate_result(root)
            Image.new("RGB", (64, 64), "blue").save(reference)
            with self.subTest(mutation="pixels"), self.assertRaisesRegex(ValueError, "参照画像"):
                Service.validate_result(root)

    def test_worker_result_must_match_the_original_request(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            request = {
                "prompt": "a fox",
                "model": "distilled",
                "precision": "int8",
                "steps": 16,
                "cfg": 1,
                "seed": 7,
                "images": [],
                "reference_sha256": [],
                "width": 64,
                "height": 64,
                "boxes": [],
            }
            record = {
                "input_prompt": "a fox",
                "prompt": "a fox",
                "model": "distilled",
                "precision": "int8",
                "steps": 16,
                "cfg": 1,
                "seed": 42,
                "width": 64,
                "height": 64,
                "boxes": [],
                "reference_hashes": [],
            }
            (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("flux3_klein", json.dumps(record))
            Image.new("RGB", (64, 64)).save(root / "output.png", pnginfo=metadata)
            (root / "result.json").write_text(
                json.dumps({**record, "image_path": str(root / "output.png")}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "生成条件"):
                service.validate_result(root)
            record["seed"] = 7
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("flux3_klein", json.dumps(record))
            Image.new("RGB", (64, 64)).save(root / "output.png", pnginfo=metadata)
            (root / "result.json").write_text(
                json.dumps({**record, "image_path": str(root / "output.png")}), encoding="utf-8"
            )
            self.assertEqual(service.validate_result(root)["seed"], 7)
            (root / "result.json").write_text(
                json.dumps({"image_path": str(root.parent / "outside.png"), "width": 64, "height": 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "出力"):
                service.validate_result(root)

    def test_cancelled_job_is_not_published_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            service.worker = Mock()
            (root / "cancel").touch()
            lease = Mock()
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(job["status"], "cancelled")
            service.worker.close.assert_called_once()
            lease.release.assert_called_once()

    def test_failure_to_write_error_record_still_returns_stopped_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            service.worker = Mock()
            service.worker.start.side_effect = RuntimeError("spawn failed")
            lease = Mock()
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.flux3_klein.service.atomic_json", side_effect=OSError("disk unavailable")),
                self.assertRaises(OSError),
            ):
                service.execute(job, lease)
            service.worker.close.assert_called_once()
            lease.release.assert_called_once()
            self.assertEqual(job["status"], "error")

    def test_cancellation_during_worker_response_wins_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            service.worker = Mock()
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            service.jobs["job"] = job

            def response():
                self.assertIn("中断", service.cancel("job"))
                return {"ok": True}

            service.worker.result.side_effect = response
            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch.object(service, "validate_result", return_value={}),
            ):
                service.execute(job, Mock())
            self.assertEqual(job["status"], "cancelled")
            service.worker.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
