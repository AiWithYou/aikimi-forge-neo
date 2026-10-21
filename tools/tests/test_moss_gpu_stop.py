"""CPU failure injection for MOSS stop admission, cancellation and cleanup."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import gpu_fault
from modules_forge.moss_image.service import Service


def mock_lease():
    lease = Mock()
    lease.release.side_effect = lambda **kwargs: kwargs.get("before_unlock", lambda: None)()
    return lease


class MossGPUStopTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = gpu_fault.FaultState(self.root / "fault.json")
        self.enterContext(patch.object(gpu_fault, "STATE", self.state))
        self.service = Service(self.root)
        self.service.worker = Mock()
        self.lease = mock_lease()
        self.job = {"id": "job", "directory": str(self.root), "status": "running", "error": ""}
        self.enterContext(patch("modules_forge.gpu_ownership.release_forge_vram"))
        self.enterContext(patch("modules_forge.gpu_residency.register"))

    def test_cuda_failure_latches_before_worker_stop_and_release(self):
        self.service.worker.result.side_effect = RuntimeError("CUDA error: unknown error")
        observations = []
        self.service.worker.close.side_effect = lambda: observations.append(self.state.blocked())
        self.lease.release.side_effect = lambda **kwargs: (
            observations.append(self.state.blocked()),
            kwargs.get("before_unlock", lambda: None)(),
        )
        self.service.execute(self.job, self.lease)
        self.assertEqual(observations, [True, True])
        self.assertEqual(self.job["error_code"], "gpu_stopped")
        self.assertEqual(self.job["status"], "error")
        self.assertTrue(self.state.pending_path.exists())
        self.lease.release.assert_called_once()

    def test_cancel_race_does_not_hide_fatal_response(self):
        def start(*args):
            (self.root / "cancel").touch()

        self.service.worker.start.side_effect = start
        self.service.worker.result.return_value = {"ok": False, "error": "GPU is lost", "gpu_fault": "gpu_execution"}
        self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "error")
        self.assertTrue(self.state.blocked())
        self.assertEqual(self.job["error_code"], "gpu_stopped")

    def test_regular_oom_does_not_latch_or_retry(self):
        self.service.worker.result.return_value = {"ok": False, "error": "CUDA out of memory"}
        self.service.execute(self.job, self.lease)
        self.assertFalse(self.state.blocked())
        self.assertFalse(self.state.pending_path.exists())
        self.service.worker.start.assert_called_once()
        self.service.worker.close.assert_called_once()

    def test_slow_cooperative_cancel_keeps_ownership_until_response(self):
        self.service.worker.start.side_effect = lambda *args: (self.root / "cancel").touch()
        elapsed = [0]
        clock = SimpleNamespace(monotonic=lambda: elapsed[0], sleep=Mock())

        def response():
            elapsed[0] += 20
            self.lease.release.assert_not_called()
            self.service.worker.close.assert_not_called()
            self.assertTrue(self.state.pending_path.exists())
            self.assertFalse(self.state.blocked())
            if elapsed[0] > 20:
                self.assertEqual(self.job["status"], "stopping")
            if elapsed[0] < 80:
                return None
            return {"ok": False, "error_code": "cancelled", "error": "cancelled"}

        self.service.worker.result.side_effect = response
        with patch("modules_forge.moss_image.service.time", clock):
            self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "cancelled")
        self.assertEqual(elapsed[0], 80)
        self.assertFalse(self.state.blocked())
        self.assertFalse(self.state.pending_path.exists())
        self.service.worker.start.assert_called_once()
        self.service.worker.close.assert_called_once()

    def test_fatal_response_during_cancel_wait_still_latches(self):
        self.service.worker.start.side_effect = lambda *args: (self.root / "cancel").touch()
        self.service.worker.result.side_effect = [
            None,
            None,
            {"ok": False, "error": "GPU is lost", "gpu_fault": "gpu_execution"},
        ]
        clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 16]), sleep=Mock())
        with patch("modules_forge.moss_image.service.time", clock):
            self.service.execute(self.job, self.lease)
        self.assertEqual(self.service.worker.result.call_count, 3)
        self.assertEqual(self.job["status"], "error")
        self.assertTrue(self.state.blocked())
        self.assertTrue(self.state.pending_path.exists())

    def test_native_exit_during_cancel_wait_still_latches(self):
        from modules_forge.resident_worker import WorkerExited

        self.service.worker.start.side_effect = lambda *args: (self.root / "cancel").touch()
        self.service.worker.result.side_effect = [None, WorkerExited("native exit", 23)]
        with patch("modules_forge.moss_image.service.time.sleep"):
            self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "error")
        self.assertTrue(self.state.blocked())
        self.assertTrue(self.state.pending_path.exists())
        self.service.worker.close.assert_called_once()
        self.lease.release.assert_called_once()

    def test_cancel_lifecycle_survives_completion_and_repeated_clicks(self):
        self.service.jobs["job"] = self.job

        def start(*args):
            self.service.cancel("job")
            self.service.cancel("job")

        self.service.worker.start.side_effect = start
        self.service.worker.result.side_effect = [None, {"ok": False, "error_code": "cancelled", "error": "cancelled"}]
        with patch("modules_forge.moss_image.service.time.sleep"):
            self.service.execute(self.job, self.lease)
        records = [
            json.loads(line) for line in (self.root / "run.lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [item["event"] for item in records], ["cancel_requested", "cancel_observed", "cancel_acknowledged"]
        )
        self.assertEqual(self.job["status"], "cancelled")

    def test_cooperative_cancel_is_not_a_fault(self):
        self.service.worker.start.side_effect = lambda *args: (self.root / "cancel").touch()
        self.service.worker.result.return_value = {"ok": False, "error_code": "cancelled", "error": "cancelled"}
        self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "cancelled")
        self.assertFalse(self.state.blocked())

    def test_death_after_result_before_release_is_not_success(self):
        from modules_forge.resident_worker import WorkerExited

        self.service.worker.result.return_value = {"ok": True}
        self.service.worker.ensure_alive.side_effect = [None, WorkerExited("native death", 23)]
        with patch.object(self.service, "validate_result"):
            self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "error")
        self.assertTrue(self.state.blocked())
        self.assertTrue(self.state.pending_path.exists())

    def test_latched_start_rejects_before_readiness_or_worker(self):
        self.state.trip("gpu_execution", "moss-image")
        with patch("modules_forge.moss_image.service.model_ready") as ready:
            with self.assertRaises(gpu_fault.GPUFaultError):
                self.service.start({"prompt": "cat"})
        ready.assert_not_called()
        self.service.worker.start.assert_not_called()

    def test_fault_during_lease_release_cannot_publish_success(self):
        self.service.worker.result.return_value = {"ok": True}
        (self.root / "output.png").write_bytes(b"unpublished")
        self.lease.release.side_effect = RuntimeError("CUDA error: unknown error")
        with patch.object(self.service, "validate_result"):
            self.service.execute(self.job, self.lease)
        self.assertEqual(self.job["status"], "error")
        self.assertEqual(self.job["error_code"], "gpu_stopped")
        self.assertTrue(self.state.blocked())
        self.assertTrue(self.state.pending_path.exists())
        self.assertFalse((self.root / "output.png").exists())
        self.service.worker.close.assert_called_once()

    def test_pending_record_is_retained_until_lease_release_completes(self):
        self.service.worker.result.return_value = {"ok": True}
        observed = []
        self.lease.release.side_effect = lambda **kwargs: (
            observed.append(self.state.pending_path.exists()),
            kwargs.get("before_unlock", lambda: None)(),
        )
        with patch.object(self.service, "validate_result"):
            self.service.execute(self.job, self.lease)
        self.assertEqual(observed, [True])
        self.assertFalse(self.state.pending_path.exists())
        self.assertEqual(self.job["status"], "complete")


if __name__ == "__main__":
    unittest.main()
