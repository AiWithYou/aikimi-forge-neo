"""MOSS jobs publish validated PNGs and hold the GPU until the worker is idle."""

import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image, PngImagePlugin

from modules_forge import gpu_fault
from modules_forge.moss_image import lora
from modules_forge.moss_image.core import MODEL_ID, MODEL_REVISION, MossImageError, validate_request
from modules_forge.moss_image.service import Service


def mock_lease():
    lease = Mock()
    lease.release.side_effect = lambda **kwargs: kwargs.get("before_unlock", lambda: None)()
    return lease


class ServiceTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(gpu_fault, "STATE", gpu_fault.FaultState(root / "fault.json")))

    def artifact(self, root, **changes):
        request = validate_request({"prompt": "a fox", "width": 512, "height": 512, "seed": 7})
        lora.bind(root, request)
        (root / "request.json").write_text(json.dumps(request), encoding="utf-8")
        record = dict(
            request,
            model_id=MODEL_ID,
            model_revision=MODEL_REVISION,
            elapsed_seconds=1.25,
            lora=lora.record(request),
            lora_applied_targets=0,
        )
        record.update(changes)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("moss_image", json.dumps(record))
        Image.new("RGB", (512, 512)).save(root / "output.png", pnginfo=metadata)
        result = dict(record, image_path=str(root / "output.png"))
        (root / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def test_start_snapshots_validated_request_before_dispatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root, outputs=root / "outputs")
            lease = mock_lease()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.moss_image.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.moss_image.service.threading.Thread") as thread,
            ):
                identifier = service.start({"prompt": "a fox", "negative_prompt": "blur", "seed": 7})
            self.assertEqual(len(identifier), 32)
            request = json.loads((root / "outputs" / identifier / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["prompt"], "a fox")
            self.assertEqual(request["negative_prompt"], "blur")
            self.assertEqual(request["seed"], 7)
            self.assertEqual(lease.engine, "moss-image")
            thread.return_value.start.assert_called_once()
            lease.release.assert_not_called()

    def test_precision_and_bound_lora_identity_reach_readiness_and_dispatch(self):
        from tools.tests.test_moss_image_lora import adapter

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root / "loras/style.safetensors")
            service = Service(root, outputs=root / "outputs")
            lease = mock_lease()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.moss_image.service.model_ready", return_value=True) as ready,
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.moss_image.service.threading.Thread"),
            ):
                identifier = service.start(
                    {
                        "prompt": "cat",
                        "precision": "bf16",
                        "text_precision": "w4a8",
                        "loras": [{"name": "style.safetensors", "strength": 0.75}],
                    }
                )
            request = json.loads((service.outputs / identifier / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["lora_identities"][0]["name"], "style.safetensors")
            self.assertEqual((request["precision"], request["text_precision"]), ("bf16", "w4a8"))
            ready.assert_called_once_with(root, request)

    def test_wrong_lora_application_count_cannot_publish_a_complete_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root, lora_applied_targets=1)
            with self.assertRaisesRegex(ValueError, "LoRA"):
                Service.validate_result(root)

    def test_png_must_include_the_lora_application_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.artifact(root)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text(
                "moss_image",
                json.dumps(
                    {key: value for key, value in result.items() if key not in {"image_path", "lora_applied_targets"}}
                ),
            )
            Image.new("RGB", (512, 512)).save(root / "output.png", pnginfo=metadata)
            with self.assertRaisesRegex(ValueError, "PNG内"):
                Service.validate_result(root)

    def test_missing_model_and_invalid_request_do_not_acquire_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            with patch("modules_forge.gpu_ownership.GPUOwnership") as ownership:
                with self.assertRaises(ValueError):
                    service.start({"prompt": ""})
                with self.assertRaisesRegex(ValueError, "aikimi-moss-image-setup.bat"):
                    service.start({"prompt": "a fox"})
            ownership.assert_not_called()

    def test_redirected_output_ancestor_is_rejected_before_gpu(self):
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
                patch("modules_forge.moss_image.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership") as ownership,
            ):
                with self.assertRaises(ValueError):
                    service.start({"prompt": "a fox"})
            ownership.assert_not_called()

    def test_thread_dispatch_failure_returns_lease_and_removes_job(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root, outputs=root / "outputs")
            lease = mock_lease()
            lease.acquire.return_value = True
            with (
                patch("modules_forge.moss_image.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch("modules_forge.moss_image.service.threading.Thread") as thread,
            ):
                thread.return_value.start.side_effect = RuntimeError("dispatch failed")
                with self.assertRaisesRegex(RuntimeError, "dispatch failed"):
                    service.start({"prompt": "a fox"})
            lease.release.assert_called_once()
            self.assertEqual(service.jobs, {})

    def test_artifact_requires_request_identity_dimensions_and_embedded_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root)
            self.assertEqual(Service.validate_result(root)["seed"], 7)
            for changes in ({"seed": 8}, {"width": 1024}, {"model_revision": "other"}, {"negative_prompt": "changed"}):
                with self.subTest(changes=changes):
                    self.artifact(root, **changes)
                    with self.assertRaises(ValueError):
                        Service.validate_result(root)
            self.artifact(root)
            Image.new("RGB", (512, 512)).save(root / "output.png")
            with self.assertRaisesRegex(ValueError, "PNG内"):
                Service.validate_result(root)

    def test_artifact_rejects_path_escape_and_mismatched_png_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = self.artifact(root)
            record["image_path"] = str(root.parent / "private.png")
            (root / "result.json").write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "出力"):
                Service.validate_result(root)
            self.artifact(root)
            metadata = PngImagePlugin.PngInfo()
            metadata.add_text("moss_image", json.dumps({"seed": 8}))
            Image.new("RGB", (512, 512)).save(root / "output.png", pnginfo=metadata)
            with self.assertRaisesRegex(ValueError, "PNG内"):
                Service.validate_result(root)

    def test_complete_worker_is_idle_before_lease_release_and_retained(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root)
            service = Service(root)
            service.worker = Mock()
            service.worker.result.side_effect = lambda: events.append("idle") or {"ok": True}
            lease = mock_lease()
            lease.release.side_effect = lambda **kwargs: (events.append("released"), kwargs.get("before_unlock", lambda: None)())
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            service.jobs["job"] = job
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(events, ["idle", "released"])
            self.assertEqual(job["status"], "complete")
            self.assertTrue((root / "output.png").is_file())
            self.assertIn("ありません", service.cancel("job"))
            service.worker.close.assert_not_called()

    def test_failed_worker_is_confirmed_stopped_before_release(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            service.worker = Mock()
            service.worker.start.side_effect = RuntimeError("spawn failed")
            service.worker.close.side_effect = lambda: events.append("stopped")
            lease = mock_lease()
            lease.release.side_effect = lambda **kwargs: (events.append("released"), kwargs.get("before_unlock", lambda: None)())
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(events, ["stopped", "released"])
            self.assertEqual(job["status"], "error")

    def test_internal_failures_keep_private_details_in_local_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            for failure in (
                FileNotFoundError("H:/private/model.safetensors"),
                MossImageError("セットアップを実行してください。"),
            ):
                with self.subTest(failure=type(failure).__name__):
                    service.worker = Mock()
                    service.worker.start.side_effect = failure
                    job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
                    with (
                        patch("modules_forge.gpu_ownership.release_forge_vram"),
                        patch("modules_forge.gpu_residency.register"),
                    ):
                        service.execute(job, mock_lease())
                    self.assertNotIn("H:/private", job["error"])
                    self.assertIn(str(failure), (root / "run.log").read_text(encoding="utf-8"))
                    if isinstance(failure, MossImageError):
                        self.assertEqual(job["error"], str(failure))

    def test_stop_confirmation_is_retried_while_lease_remains_owned(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            service.worker = Mock()
            service.worker.start.side_effect = RuntimeError("spawn failed")

            def stop():
                events.append("stop")
                if events.count("stop") == 1:
                    raise RuntimeError("still stopping")

            service.worker.close.side_effect = stop
            lease = mock_lease()
            lease.release.side_effect = lambda **kwargs: (events.append("released"), kwargs.get("before_unlock", lambda: None)())
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.moss_image.service.time.sleep"),
            ):
                service.execute(job, lease)
            self.assertEqual(events, ["stop", "stop", "released"])
            self.assertEqual(job["status"], "error")

    def test_cancel_before_dispatch_does_not_launch_worker(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "cancel").touch()
            service = Service(root)
            service.worker = Mock()
            lease = mock_lease()
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(job["status"], "cancelled")
            service.worker.start.assert_not_called()
            service.worker.close.assert_called_once()
            lease.release.assert_called_once()

    def test_cancel_during_result_validation_wins_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root)
            service = Service(root)
            service.worker = Mock()
            service.worker.result.return_value = {"ok": True}
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            service.jobs["job"] = job
            events = []

            def stop():
                self.assertTrue((root / "output.png").exists())
                (root / ".output.png.tmp").write_bytes(b"late write")
                events.append("stopped")

            service.worker.close.side_effect = stop
            lease = mock_lease()
            lease.release.side_effect = lambda **kwargs: (events.append("released"), kwargs.get("before_unlock", lambda: None)())

            def validate(_directory):
                self.assertIn("中断", service.cancel("job"))
                return Service.validate_result(_directory)

            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch.object(service, "validate_result", side_effect=validate),
            ):
                service.execute(job, lease)
            self.assertEqual(job["status"], "cancelled")
            self.assertEqual(events, ["stopped", "released"])
            self.assertFalse((root / "output.png").exists())
            self.assertFalse((root / ".output.png.tmp").exists())
            self.assertEqual(json.loads((root / "result.json").read_text(encoding="utf-8")), {"status": "cancelled", "error": ""})
            service.worker.close.assert_called_once()

    def test_worker_failure_after_png_save_removes_output_only_after_stop(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root)
            service = Service(root)
            service.worker = Mock()
            service.worker.result.return_value = {"ok": False, "error": "worker failed after saving"}
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}

            def stop():
                self.assertTrue((root / "output.png").exists())
                (root / ".output.png.tmp").touch()

            service.worker.close.side_effect = stop
            lease = mock_lease()
            with patch("modules_forge.gpu_ownership.release_forge_vram"), patch("modules_forge.gpu_residency.register"):
                service.execute(job, lease)
            self.assertEqual(job["status"], "error")
            self.assertFalse((root / "output.png").exists())
            self.assertFalse((root / ".output.png.tmp").exists())
            self.assertEqual(json.loads((root / "result.json").read_text(encoding="utf-8"))["status"], "error")
            lease.release.assert_called_once()

    def test_cleanup_error_is_logged_and_stopped_gpu_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.artifact(root)
            service = Service(root)
            service.worker = Mock()
            service.worker.result.return_value = {"ok": False, "error": "worker failed"}
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            lease = mock_lease()
            original = Path.unlink

            def locked(path, *args, **kwargs):
                if path == root / "output.png":
                    raise PermissionError("output file is locked")
                return original(path, *args, **kwargs)

            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch.object(Path, "unlink", locked),
            ):
                service.execute(job, lease)
            self.assertEqual(job["status"], "error")
            self.assertIn("output file is locked", (root / "run.log").read_text(encoding="utf-8"))
            lease.release.assert_called_once()

    def test_error_record_write_failure_still_releases_stopped_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = Service(Path(temporary))
            service.worker = Mock()
            service.worker.start.side_effect = RuntimeError("spawn failed")
            lease = mock_lease()
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            with (
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.moss_image.service.atomic_json", side_effect=OSError("disk unavailable")),
                self.assertRaises(OSError),
            ):
                service.execute(job, lease)
            service.worker.close.assert_called_once()
            lease.release.assert_called_once()
            self.assertEqual(job["status"], "error")

    def test_poll_is_a_consistent_snapshot_and_does_not_publish_progress_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            job = {"id": "job", "directory": temporary, "status": "running", "error": ""}
            service.jobs["job"] = job
            original = Path.read_text

            def finish_during_read(path, *args, **kwargs):
                if path == root / "progress.json":
                    with service._guard:
                        job["status"] = "complete"
                    return json.dumps({"stage": "sampling", "current": 1, "total": 2, "image_path": "untrusted.png"})
                return original(path, *args, **kwargs)

            with patch.object(Path, "read_text", finish_during_read):
                result = service.poll("job")
            self.assertEqual(result["status"], "running")
            self.assertNotIn("files", result)
            self.assertNotIn("image_path", result)
            self.artifact(root)
            result = service.poll("job")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["files"], [str(root / "output.png"), str(root / "result.json")])
            self.assertEqual(service.poll("missing"), {"status": "idle"})


if __name__ == "__main__":
    unittest.main()
