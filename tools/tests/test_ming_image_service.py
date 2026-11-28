import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge.ming_image import service
from modules_forge.ming_image.service import SERVICE


class ContractTests(unittest.TestCase):
    def test_unknown_job(self):
        assert SERVICE.poll("unknown") == {"status": "idle"}

    def test_cancel_before_worker_start_releases_lease_and_records_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            instance = service.Service(root=root, outputs=root / "outputs", worker_root=root / "cache")
            lease = Mock()
            with (
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch.object(service, "model_ready", return_value=True),
                patch.object(service.threading.Thread, "start"),
            ):
                job = instance.start(dict(prompt="poster", profile="comfy_org_w4a8"))
            lease.acquire.assert_called_once_with(False)
            self.assertEqual(lease.engine, "ming-image")
            instance.cancel(job)
            instance.execute(instance.jobs[job], lease)
            self.assertEqual(instance.poll(job)["status"], "cancelled")
            lease.release.assert_called_once_with()
            self.assertFalse((root / "outputs" / job / "output.png").exists())

    def test_unresponsive_worker_stops_before_artifact_cleanup_and_unlock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            instance = service.Service(outputs=root / "outputs", worker_root=root / "cache")
            lease = Mock()
            with (
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch.object(service, "model_ready", return_value=True),
                patch.object(service.threading.Thread, "start"),
                patch.object(service.gpu_fault, "STATE") as fault,
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.gpu_residency.register"),
                patch.object(service, "CANCEL_GRACE_SECONDS", 0),
            ):
                instance.worker = Mock()
                fault.blocked.return_value = False
                job = instance.start(dict(prompt="api_key=dummy-secret", profile="comfy_org_w4a8"))
                directory = instance.outputs / job
                private = root / "cache/jobs" / job
                self.assertTrue((private / "request.json").is_file())
                self.assertFalse((directory / "request.json").exists())
                artifact = directory / "output.png"
                artifact.touch()
                instance.worker.start.side_effect = lambda *args: instance.cancel(job)
                instance.worker.result.side_effect = [None, None, AssertionError("cancel timed out")]
                events = []
                instance.worker.close.side_effect = lambda: events.append(("stop", artifact.exists()))

                def unlock(before_unlock):
                    events.append(("unlock", artifact.exists(), private.exists()))
                    before_unlock()

                lease.release.side_effect = unlock
                unlink = Path.unlink
                locked = True

                def transient_lock(path, **kwargs):
                    nonlocal locked
                    if path == artifact and locked:
                        locked = False
                        raise PermissionError("output still open")
                    unlink(path, **kwargs)

                with patch.object(Path, "unlink", transient_lock):
                    instance.execute(instance.jobs[job], lease)
                self.assertEqual(instance.poll(job)["status"], "cancelled")
                lease.release.assert_called_once()
                self.assertEqual(events, [("stop", True), ("unlock", False, False)])

    def test_worker_entrypoint_cleans_private_data_on_success_and_failure(self):
        from PIL import Image

        from modules_forge.ming_image import core, inference
        from tools import ming_image_worker

        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                instance = service.Service(root=root, outputs=root / "outputs", worker_root=root / "cache")
                runtime = Mock()
                runtime.generate.return_value = Image.new("RGBA", (512, 512), (0, 0, 0, 255))
                if fail:
                    runtime.generate.side_effect = ValueError("private worker failure")
                with (
                    patch.object(service, "model_ready", return_value=True),
                    patch.object(core, "model_ready", return_value=True),
                    patch.object(service.threading.Thread, "start"),
                    patch.object(service.gpu_fault, "STATE") as fault,
                    patch("modules_forge.gpu_ownership.GPUOwnership") as ownership,
                    patch("modules_forge.gpu_ownership.release_forge_vram"),
                    patch("modules_forge.gpu_residency.register"),
                    patch.object(ming_image_worker, "_runner", inference.Runner(root, runtime)),
                    patch.object(ming_image_worker, "_root", root.resolve()),
                ):
                    instance.worker = Mock()
                    fault.blocked.return_value = False
                    instance.worker.start.side_effect = lambda python, script, env, payload, log: (
                        ming_image_worker.resident_run(payload)
                    )
                    instance.worker.result.return_value = {"ok": True}
                    job = instance.start(
                        dict(
                            profile="comfy_org_w4a8",
                            prompt="api_key=dummy-secret",
                            width=512,
                            height=512,
                            transparent=True,
                        )
                    )
                    instance.execute(instance.jobs[job], ownership.return_value)
                    record = instance.poll(job)
                    self.assertEqual(record["status"], "error" if fail else "complete")
                    self.assertFalse((root / "cache/jobs" / job).exists())
                    directory = instance.outputs / job
                    self.assertEqual(
                        {p.name for p in directory.iterdir()},
                        {"result.json"} if fail else {"output.png", "result.json"},
                    )
                    public = (directory / "result.json").read_text(encoding="utf-8")
                    self.assertNotIn("dummy-secret", public)
                    self.assertNotIn(root.as_posix(), public.replace("\\", "/"))
