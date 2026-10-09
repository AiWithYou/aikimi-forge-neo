"""GPU lease is returned only after a failed/cancelled worker is stopped."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
