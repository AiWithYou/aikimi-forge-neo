"""CPU contracts for persistent GPU admission stops; never initialize CUDA."""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge import gpu_fault, gpu_ownership, gpu_residency


class GPUFaultTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = gpu_fault.FaultState(root / "gpu-fault.json")
        self.enterContext(patch.object(gpu_fault, "STATE", self.state))

    def test_persisted_and_corrupt_marker_both_block_without_gpu(self):
        self.state.trip("gpu_execution", "moss-image")
        self.assertTrue(self.state.path.is_file())
        reloaded = gpu_fault.FaultState(self.state.path)
        with self.assertRaises(gpu_fault.GPUFaultError):
            reloaded.ensure_available()
        self.state.path.write_text("{broken", encoding="utf-8")
        with self.assertRaises(gpu_fault.GPUFaultError):
            gpu_fault.FaultState(self.state.path).ensure_available()

    def test_failed_persistence_remains_blocked_in_process(self):
        with patch.object(gpu_fault, "atomic_json", side_effect=OSError("disk unavailable")):
            self.state.trip("gpu_execution", "moss-image")
        with self.assertRaises(gpu_fault.GPUFaultError):
            self.state.ensure_available()

    def test_removing_marker_does_not_unlock_running_process(self):
        self.state.trip("gpu_execution", "moss-image")
        self.state.path.unlink()
        with self.assertRaises(gpu_fault.GPUFaultError):
            self.state.ensure_available()

    def test_explicit_clear_requires_acknowledgement(self):
        self.state.trip("gpu_execution", "moss-image")
        with self.assertRaises(ValueError):
            self.state.clear(acknowledge_recovery=False)
        self.assertTrue(self.state.blocked())
        self.state.clear(acknowledge_recovery=True)
        self.assertFalse(gpu_fault.FaultState(self.state.path).blocked())
        self.assertTrue(self.state.blocked(), "an existing process must stay stopped after CLI clearance")

    def test_classification_distinguishes_fatal_from_normal_failures(self):
        for message in (
            "CUDA error: an illegal memory access was encountered",
            "GPU is lost",
            "CUDA error: unknown error",
            "Expected mha_graph.execute(...).is_good() to be true",
            "CUDNN_STATUS_EXECUTION_FAILED",
        ):
            with self.subTest(message=message):
                self.assertEqual(gpu_fault.classify(RuntimeError(message)), "gpu_execution")
        for message in (
            "CUDA out of memory",
            "No available kernel. Aborting execution.",
            "cancelled",
            "model missing",
            "CUDNN_STATUS_NOT_SUPPORTED",
        ):
            with self.subTest(message=message):
                self.assertIsNone(gpu_fault.classify(RuntimeError(message)))

    def test_fatal_cleanup_after_oom_takes_priority(self):
        failure = RuntimeError("CUDA error: unknown error")
        failure.__context__ = RuntimeError("CUDA out of memory")
        self.assertEqual(gpu_fault.classify(failure), "gpu_execution")

    def test_fault_detected_during_engine_eviction_denies_admission(self):
        lock = gpu_ownership.GPUQueueLock()
        lock._restored = True
        with patch.object(gpu_residency, "begin", side_effect=lambda engine: self.state.trip("worker_exit", engine)):
            with self.assertRaises(gpu_fault.GPUFaultError):
                lock.acquire(False)
        self.assertTrue(lock._lock.acquire(False))
        lock._lock.release()

    def test_completion_callback_runs_after_cleanup_before_unlock(self):
        lock = gpu_ownership.GPUQueueLock()
        lock._restored = True
        events = []
        with (
            patch.object(gpu_residency, "begin"),
            patch.object(gpu_residency, "finish", side_effect=lambda: events.append("cleanup")),
        ):
            lock.acquire()

            def complete():
                events.append("journal")
                self.assertFalse(lock._lock.acquire(False))

            lock.release(before_unlock=complete)
        self.assertEqual(events, ["cleanup", "journal"])
        self.assertTrue(lock._lock.acquire(False))
        lock._lock.release()

    def test_completion_io_failure_keeps_stop_and_unlocks_fifo(self):
        lock = gpu_ownership.GPUQueueLock()
        lock._restored = True
        self.state.begin("job")
        with patch.object(gpu_residency, "begin"), patch.object(gpu_residency, "finish"):
            lock.acquire()
            with patch.object(Path, "unlink", side_effect=PermissionError("busy")):
                with self.assertRaises(gpu_fault.GPUFaultError):
                    lock.release(before_unlock=lambda: self.state.complete("job"))
        self.assertTrue(self.state.blocked())
        self.assertTrue(self.state.pending_path.exists())
        self.assertTrue(lock._lock.acquire(False))
        lock._lock.release()

    def test_fault_denies_queue_before_residency_or_pending_recovery(self):
        lock = gpu_ownership.GPUQueueLock()
        self.state.trip("gpu_execution", "moss-image")
        with patch.object(gpu_residency, "begin") as begin:
            with self.assertRaises(gpu_fault.GPUFaultError):
                lock.acquire(False)
        begin.assert_not_called()
        self.assertFalse(lock._restored)

    def test_waiting_job_is_rejected_after_fault_and_does_not_strand_fifo(self):
        lock = gpu_ownership.GPUQueueLock()
        lock._restored = True
        outcomes = []
        waiting = threading.Event()
        with patch.object(gpu_residency, "begin"), patch.object(gpu_residency, "finish") as finish:
            self.assertTrue(lock.acquire())

            def enter():
                waiting.set()
                try:
                    lock.acquire()
                except gpu_fault.GPUFaultError:
                    outcomes.append("blocked")

            thread = threading.Thread(target=enter)
            thread.start()
            try:
                self.assertTrue(waiting.wait(2))
                self.state.trip("gpu_execution", "moss-image")
                lock.release()
            finally:
                thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(outcomes, ["blocked"])
            finish.assert_not_called()
        self.assertTrue(lock._lock.acquire(False), "rejected waiter must release raw FIFO ownership")
        lock._lock.release()

    def test_fault_release_cancels_timer_without_gpu_cleanup(self):
        timer, cleanup = Mock(), Mock()
        with (
            patch.object(gpu_residency, "_resources", {"forge": ("forge", cleanup)}),
            patch.object(gpu_residency, "_timer", timer),
            patch.object(gpu_residency, "_active", True),
            patch.object(gpu_residency, "policy", return_value="release"),
        ):
            self.state.trip("gpu_execution", "moss-image")
            gpu_residency.finish()
            timer.cancel.assert_called_once()
            cleanup.assert_not_called()
            self.assertFalse(gpu_residency._active)
            gpu_residency.expire_idle()
            self.assertIn("停止", gpu_residency.release_idle())
            cleanup.assert_not_called()

    def test_fault_race_at_idle_queue_acquisition_is_handled(self):
        for action in (gpu_residency.expire_idle, gpu_residency.release_idle, gpu_residency.apply_policy):
            with (
                self.subTest(action=action.__name__),
                patch.object(gpu_ownership.queue_lock, "acquire", side_effect=gpu_fault.GPUFaultError()),
            ):
                action()

    def test_fatal_manual_cleanup_latches_before_any_next_cleanup(self):
        cleanup = Mock(side_effect=RuntimeError("CUDA error: unknown error"))
        with patch.object(gpu_residency, "_resources", {"forge": ("forge", cleanup)}):
            with self.assertRaisesRegex(RuntimeError, "unknown error"):
                gpu_residency.release_resource("forge")
            self.assertTrue(self.state.blocked())
            with self.assertRaises(gpu_fault.GPUFaultError):
                gpu_residency.release_resource("forge")
            cleanup.assert_called_once()


if __name__ == "__main__":
    unittest.main()
