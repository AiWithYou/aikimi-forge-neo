"""GPU lease is returned only after a failed/cancelled worker is stopped."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from modules_forge.iris.core import IrisError
from modules_forge.iris.service import Service


class Lease:
    def __init__(self, events):
        self.events = events

    def acquire(self, blocking=False):
        self.events.append("acquire")
        return True

    def release(self):
        self.events.append("release")


class Worker:
    def __init__(self, events):
        self.events = events

    def start(self, *args):
        self.events.append("start")
        raise RuntimeError("failed spawn")

    def close(self):
        self.events.append("stopped")


class ServiceTests(unittest.TestCase):
    def test_uploaded_pixels_are_canonicalized_without_opening_caller_paths(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            image = Image.new("RGBA", (3, 2), (255, 0, 0, 0))
            with (
                patch("modules_forge.iris.service.ROOT", root),
                patch("modules_forge.iris.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=Lease(events)),
                patch("modules_forge.iris.service.threading.Thread") as thread,
            ):
                identifier = service.start({"task": "depth", "precision": "int8", "image": image})
            directory = root / "outputs/iris" / identifier
            request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(request["image"], str(directory / "input.png"))
            with Image.open(request["image"]) as saved:
                self.assertEqual((saved.mode, saved.size, saved.getpixel((0, 0))), ("RGB", (3, 2), (255, 255, 255)))
            thread.return_value.start.assert_called_once()

    def test_caller_file_paths_are_rejected_without_filesystem_reads(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "private.png"
            Image.new("RGB", (2, 2)).save(image)
            service = Service(root)
            with (
                patch("modules_forge.iris.service.ROOT", root),
                patch("modules_forge.iris.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=Lease(events)),
                patch("modules_forge.iris.service.threading.Thread"),
                patch("modules_forge.iris.service.Image.open") as open_image,
                self.assertRaisesRegex(IrisError, "アップロード"),
            ):
                service.start({"task": "upscale", "precision": "int8", "image": str(image)})
            open_image.assert_not_called()
            self.assertEqual(events, ["acquire", "release"])

    def test_pixel_limit_applies_before_encoding(self):
        events = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            image = Image.new("L", (6400, 6400))
            with (
                patch("modules_forge.iris.service.ROOT", root),
                patch("modules_forge.iris.service.model_ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=Lease(events)),
                patch.object(image, "save") as save,
                self.assertRaisesRegex(IrisError, "40メガピクセル"),
            ):
                service.start({"task": "depth", "precision": "int8", "image": image})
            save.assert_not_called()
            self.assertEqual(events, ["acquire", "release"])

    def test_failed_dispatch_stops_worker_before_gpu_release(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            service = Service(Path(directory))
            service.worker = Worker(events)
            job = {"id": "job", "directory": str(Path(directory) / "job"), "status": "running"}
            Path(job["directory"]).mkdir()
            with patch("modules_forge.gpu_residency.register"), patch("modules_forge.gpu_ownership.release_forge_vram"):
                service.execute(job, Lease(events))
            self.assertEqual(job["status"], "error")
            self.assertEqual(events, ["start", "stopped", "release"])

    def test_cleanup_failure_keeps_lease_until_stop_confirmed(self):
        events = []
        worker = Worker(events)
        attempts = 0

        def close():
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                events.append("still-running")
                raise OSError("sharing conflict")
            events.append("stopped")

        worker.close = close
        with tempfile.TemporaryDirectory() as directory:
            service = Service(Path(directory))
            service.worker = worker
            job = {"id": "job", "directory": directory, "status": "running"}
            with (
                patch("modules_forge.gpu_residency.register"),
                patch("modules_forge.gpu_ownership.release_forge_vram"),
                patch("modules_forge.iris.service.time.sleep"),
            ):
                service.execute(job, Lease(events))
            self.assertEqual(events, ["start", "still-running", "stopped", "release"])


if __name__ == "__main__":
    unittest.main()
