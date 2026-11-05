"""Retention must reuse a runtime without releasing the queue during cleanup."""

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge import gpu_ownership, gpu_residency
from modules_forge.irodori.core import atomic_json, validate_request
from modules_forge.irodori.service import Service


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.worker = Mock()
        self.worker.result.return_value = {"ok": True}
        self.service = Service(root=self.root, outputs=self.root, worker=self.worker)
        self.queue = gpu_ownership.GPUQueueLock()
        self.queue._restored = True
        self.resources = {}
        for scope in (
            patch.object(gpu_ownership, "queue_lock", self.queue),
            patch.object(gpu_ownership, "release_forge_vram"),
            patch.object(gpu_residency, "_resources", self.resources),
            patch.object(gpu_residency, "_timer", None),
            patch.object(gpu_residency, "policy", return_value="keep"),
            patch("modules_forge.irodori.service.verify_result", return_value={}),
        ):
            scope.start()
            self.addCleanup(scope.stop)

    def lease(self, engine="irodori"):
        lease = gpu_ownership.GPUOwnership()
        lease.engine = engine
        return lease

    def job(self, name, precision="bf16"):
        directory = self.root / name
        directory.mkdir()
        atomic_json(directory / "request.json", validate_request({"text": "文", "precision": precision}))
        return self.service.new_job("owner", directory)

    def complete(self, job):
        lease = self.lease()
        self.assertTrue(lease.acquire(False))
        self.service.execute(job, lease)
        self.assertEqual(job["status"], "complete")

    def test_keep_reuses_worker_and_precision_change_stops_old_one(self):
        events = []
        self.worker.start.side_effect = lambda *args: events.append("start")
        self.worker.close.side_effect = lambda: events.append("close")
        self.complete(self.job("a"))
        self.assertEqual(events, ["start"])
        self.assertIn("irodori", self.resources)
        self.complete(self.job("b"))
        self.assertEqual(events, ["start", "start"])
        self.complete(self.job("c", "fp32"))
        self.assertEqual(events, ["start", "start", "close", "start"])

    def test_same_name_new_content_and_new_selection_restart_resident_worker(self):
        events = []
        self.worker.start.side_effect = lambda *args: events.append("start")
        self.worker.close.side_effect = lambda: events.append("close")
        for index, (name, digest) in enumerate((
            ("A.safetensors", "a" * 64), ("A.safetensors", "a" * 64),
            ("A.safetensors", "b" * 64), ("B.safetensors", "b" * 64),
        )):
            job = self.job(f"model-{index}")
            request_file = job["directory"] / "request.json"
            import json

            request = json.loads(request_file.read_text(encoding="utf-8"))
            request.update(checkpoint=name, checkpoint_identity={"name": name, "sha256": digest, "format": "float"})
            atomic_json(request_file, request)
            with (
                patch("modules_forge.irodori.service.checkpoints.bind"),
                patch("modules_forge.irodori.service.checkpoints.verify_components"),
            ):
                self.complete(job)
        self.assertEqual(events, ["start", "start", "close", "start", "close", "start"])

    def test_changed_checkpoint_before_worker_start_fails_and_releases_lease(self):
        import json

        job = self.job("changed")
        path = job["directory"] / "request.json"
        request = json.loads(path.read_text(encoding="utf-8"))
        request.update(checkpoint="A.safetensors", checkpoint_identity={"name": "A.safetensors", "sha256": "a" * 64})
        atomic_json(path, request)
        lease = self.lease()
        self.assertTrue(lease.acquire(False))
        with patch("modules_forge.irodori.service.checkpoints.bind", side_effect=ValueError("changed")):
            self.service.execute(job, lease)
        self.assertEqual(job["status"], "error")
        self.worker.start.assert_not_called()
        self.assertFalse(lease._owned)

    def blocking_cleanup(self):
        entered, proceed = threading.Event(), threading.Event()
        attempts = []

        def close():
            attempts.append(1)
            if len(attempts) == 1:
                entered.set()
                if not proceed.wait(10):
                    raise RuntimeError("test gate timed out")
                raise PermissionError("sharing conflict")

        self.worker.close.side_effect = close
        return entered, proceed, attempts

    def test_release_holds_real_queue_through_failed_cleanup(self):
        job = self.job("release")
        lease = self.lease()
        self.assertTrue(lease.acquire(False))
        entered, proceed, attempts = self.blocking_cleanup()
        with patch.object(gpu_residency, "policy", return_value="release"):
            thread = threading.Thread(target=self.service.execute, args=(job, lease))
            thread.start()
            competitor = self.lease()
            try:
                self.assertTrue(entered.wait(10))
                self.assertFalse(competitor.acquire(False))
            finally:
                proceed.set()
                thread.join(10)
                if competitor._owned:
                    competitor.release()
            self.assertFalse(thread.is_alive())
            self.assertEqual(job["status"], "complete")
            self.assertEqual(len(attempts), 2)
            self.assertTrue(competitor.acquire(False))
            competitor.release()

    def test_switching_engine_cannot_lose_queue_on_cleanup_failure(self):
        self.complete(self.job("kept"))
        entered, proceed, attempts = self.blocking_cleanup()
        other, competitor = self.lease("other"), self.lease()
        thread = threading.Thread(target=lambda: other.acquire(True))
        thread.start()
        try:
            self.assertTrue(entered.wait(10))
            self.assertFalse(competitor.acquire(False))
        finally:
            proceed.set()
            thread.join(10)
            if competitor._owned:
                competitor.release()
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(attempts), 2)
        other.release()

    def test_auto_expiry_blocks_a_new_job_until_old_worker_exits(self):
        self.complete(self.job("kept"))
        entered, proceed, attempts = self.blocking_cleanup()
        with (
            patch.object(gpu_residency, "policy", return_value="auto"),
            patch.object(gpu_residency, "_idle_since", time.monotonic() - 301),
            patch.object(gpu_residency, "_active", False),
        ):
            thread = threading.Thread(target=gpu_residency.expire_idle)
            thread.start()
            competitor = self.lease()
            try:
                self.assertTrue(entered.wait(10))
                self.assertFalse(competitor.acquire(False))
            finally:
                proceed.set()
                thread.join(10)
                if competitor._owned:
                    competitor.release()
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(attempts), 2)
            self.assertTrue(competitor.acquire(False))
            competitor.release()

    def test_cached_runtime_loads_once_and_rejects_different_key(self):
        from tools.irodori_worker import _STATE, cached_runtime

        factory = Mock()
        with patch.dict(_STATE, {"key": None, "runtime": None}):
            first = cached_runtime("bf16", factory)
            self.assertIs(cached_runtime("bf16", factory), first)
            factory.assert_called_once_with("bf16")
            with self.assertRaises(ValueError):
                cached_runtime("fp32", factory)

    def test_worker_cache_distinguishes_same_model_path_with_new_sha(self):
        from tools.irodori_worker import _STATE, cached_runtime

        factory = Mock()
        first_key = ("same/path.safetensors", "a" * 64, "bf16")
        second_key = ("same/path.safetensors", "b" * 64, "bf16")
        with patch.dict(_STATE, {"key": None, "runtime": None}):
            cached_runtime(first_key, factory)
            cached_runtime(first_key, factory)
            factory.assert_called_once_with(first_key)
            with self.assertRaises(ValueError):
                cached_runtime(second_key, factory)

    def test_cancel_accepted_before_commit_cannot_be_reported_complete(self):
        job = self.job("cancel")
        self.service.jobs["cancel"] = job

        def verification(*args):
            self.assertEqual(self.service.cancel("cancel", "owner"), "中断処理中…")
            return {}

        lease = self.lease()
        self.assertTrue(lease.acquire(False))
        with patch("modules_forge.irodori.service.verify_result", side_effect=verification):
            self.service.execute(job, lease)
        self.assertEqual(job["status"], "cancelled")
        self.worker.close.assert_called_once()
        self.assertNotIn("irodori", self.resources)

    def test_cancel_during_commit_is_explicitly_rejected(self):
        job = self.job("finishing")
        self.service.jobs["finishing"] = job

        def writing_state(*args):
            self.assertEqual(self.service.cancel("finishing", "owner"), "終了処理中です。")
            self.assertFalse(job["cancel"].is_set())

        with patch("modules_forge.irodori.service.atomic_json", side_effect=writing_state):
            self.complete(job)


if __name__ == "__main__":
    unittest.main()
