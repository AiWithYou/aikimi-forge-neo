"""CPU regressions for worker ownership and staged-input lifetime."""

from __future__ import annotations

import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules.fifo_lock import FIFOLock
from modules_forge import gpu_ownership
from modules_forge import sensenova_u15_bridge as bridge


class WorkerOutput(io.StringIO):
    fail_close = False

    def close(self):
        if self.fail_close:
            self.fail_close = False
            raise OSError("injected stream-close failure")
        super().close()


class WorkerProcess:
    def __init__(self):
        self.returncode = None
        self.fail_stop = False
        self.stdout = WorkerOutput(
            'SENSENOVA_EVENT {"stage":"loading","message":"test","progress":0.1}\n'
        )

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.fail_stop:
            raise OSError("injected stop failure")
        self.returncode = -15

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("test-worker", timeout)
        return self.returncode


class WorkerCleanupTests(unittest.TestCase):
    def setUp(self):
        self.gpu_lock = FIFOLock()
        self.enterContext(patch.object(gpu_ownership, "queue_lock", self.gpu_lock))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.worker = self.root / "worker.py"
        self.worker.touch()
        self.request = bridge.SenseNovaRequest(
            mode=bridge.MODE_TEXT,
            prompt="test",
            generation_profile=bridge.PROFILE_QUALITY,
        )
        for name, value in (
            ("_ACTIVE_JOB_ID", None),
            ("_ACTIVE_PROCESS", None),
            ("_CANCELLED_JOB_IDS", set()),
            ("_PENDING_CLEANUP", None),
            ("_GPU_OWNERSHIP", None),
        ):
            self.enterContext(patch.object(bridge, name, value, create=True))
        self.enterContext(
            patch.object(bridge, "inspect_runtime", return_value=SimpleNamespace(ready=True))
        )
        self.enterContext(patch.object(bridge, "_release_forge_vram"))

    def generation(self):
        return bridge.run_generation(
            self.request,
            output_directory=self.root / "output",
            cache_directory=self.root / "cache",
            log_directory=self.root / "logs",
            worker_path=self.worker,
        )

    def running_worker(self):
        process = WorkerProcess()
        self.addCleanup(process.stdout.close)
        self.enterContext(patch.object(bridge.subprocess, "Popen", return_value=process))
        generation = self.generation()
        self.addCleanup(generation.close)
        self.assertEqual(next(generation)["stage"], "prepare")
        self.assertEqual(next(generation)["stage"], "vram")
        event = next(generation)
        self.assertEqual(event["stage"], "loading")
        directory = self.root / "cache" / "jobs" / event["job_id"]
        return generation, process, event["job_id"], directory

    def assert_next_job_starts(self):
        generation = self.generation()
        try:
            self.assertEqual(next(generation)["stage"], "prepare")
        finally:
            generation.close()

    def test_exited_worker_stream_error_does_not_poison_next_job(self):
        generation, process, _, directory = self.running_worker()
        process.returncode = 0
        process.stdout.fail_close = True

        with self.assertRaisesRegex(OSError, "stream-close"):
            generation.close()

        self.assertIsNone(bridge._ACTIVE_JOB_ID)
        self.assertIsNone(bridge._ACTIVE_PROCESS)
        self.assertFalse(directory.exists())
        self.assert_next_job_starts()

    def test_failed_stop_preserves_inputs_until_successful_recancel(self):
        generation, process, job_id, directory = self.running_worker()
        process.fail_stop = True
        with self.assertRaisesRegex(OSError, "stop failure"):
            generation.close()

        self.assertEqual(bridge._ACTIVE_JOB_ID, job_id)
        self.assertIs(bridge._ACTIVE_PROCESS, process)
        self.assertTrue((directory / "request.json").is_file())
        self.assertFalse(process.stdout.closed)
        self.assertFalse(self.gpu_lock.acquire(False))
        with self.assertRaisesRegex(bridge.SenseNovaBridgeError, "別のSenseNova"):
            next(self.generation())

        process.fail_stop = False
        bridge.cancel_generation(job_id)

        self.assertIsNone(bridge._ACTIVE_JOB_ID)
        self.assertIsNone(bridge._ACTIVE_PROCESS)
        self.assertFalse(directory.exists())
        self.assertTrue(self.gpu_lock.acquire(False))
        self.gpu_lock.release()
        self.assert_next_job_starts()

    def test_waiting_cancel_does_not_unload_forge_or_release_its_lock(self):
        self.gpu_lock.acquire()
        self.addCleanup(self.gpu_lock.release)
        generation = self.generation()
        job_id = next(generation)["job_id"]
        self.assertEqual(next(generation)["stage"], "queued")
        bridge._release_forge_vram.assert_not_called()
        bridge.cancel_generation(job_id)
        with self.assertRaises(bridge.SenseNovaGenerationCancelled):
            next(generation)
        self.assertFalse(self.gpu_lock.acquire(False))

    def test_next_job_reaps_worker_that_exited_after_failed_stop(self):
        generation, process, _, directory = self.running_worker()
        process.fail_stop = True
        with self.assertRaisesRegex(OSError, "stop failure"):
            generation.close()
        process.returncode = 1

        self.assert_next_job_starts()

        self.assertFalse(directory.exists())
        self.assertTrue(process.stdout.closed)

    def test_cancel_keeps_ownership_until_running_generator_finishes(self):
        generation, _, job_id, directory = self.running_worker()
        bridge.cancel_generation(job_id)
        self.assertEqual(bridge._ACTIVE_JOB_ID, job_id)
        self.assertTrue(directory.exists())

        with self.assertRaises(bridge.SenseNovaGenerationCancelled):
            list(generation)

        self.assertIsNone(bridge._ACTIVE_JOB_ID)
        self.assertFalse(directory.exists())
        self.assert_next_job_starts()


if __name__ == "__main__":
    unittest.main()
