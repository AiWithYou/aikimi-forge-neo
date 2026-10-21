"""Unexpected resident death must not be hidden by success or worker replacement."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import gpu_fault, resident_worker
from modules_forge.resident_worker import ResidentWorker, WorkerExited
from modules_forge.yue2_studio.core import atomic_json, safe_environment


class ResidentSafetyTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.state = gpu_fault.FaultState(self.root / "fault.json")
        self.enterContext(patch.object(gpu_fault, "STATE", self.state))
        self.worker = ResidentWorker("moss-image", self.root, gpu_safety=True)
        self.worker.directory = self.root
        self.worker.log_path = self.root / "run.log"
        self.worker.identifier = "job"
        (self.root / "startup.log").write_text("native death", encoding="utf-8")
        self.worker.process = Mock()
        self.worker.process.stdin = None
        self.worker.process.poll.return_value = 0
        self.addCleanup(self.cleanup_worker)

    def cleanup_worker(self):
        if isinstance(self.worker.process, Mock):
            self.worker.process.poll.side_effect = None
            self.worker.process.poll.return_value = 0
        self.worker.close()

    def test_success_response_cannot_hide_unexpected_exit_zero(self):
        atomic_json(self.root / "response.json", {"id": "job", "ok": True})
        with self.assertRaises(WorkerExited):
            self.worker.result()
        self.assertTrue(self.state.blocked())

    def test_idle_worker_death_is_never_silently_restarted(self):
        script = self.root / "engine.py"
        script.write_text("def resident_run(payload): pass\n", encoding="utf-8")
        with patch("modules_forge.resident_worker.subprocess.Popen") as spawn:
            with self.assertRaises(WorkerExited):
                self.worker.start("python", script, {}, {}, self.root / "next.log")
        spawn.assert_not_called()
        self.assertTrue(self.state.blocked())

    def test_close_detects_idle_death_before_native_log_cleanup(self):
        self.worker.close()
        self.assertTrue(self.state.blocked())
        self.assertEqual((self.root / "run.native.log").read_text(encoding="utf-8"), "native death")

    def test_unexpected_exit_code_remains_in_lifecycle_after_close(self):
        self.worker.close()
        records = [
            json.loads(line) for line in (self.root / "run.lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "exit_observed")
        self.assertEqual(records[0]["exit_code"], 0)
        self.assertFalse(records[0]["stop_requested"])
        self.assertIn("time_utc", records[0])
        self.assertIsInstance(records[0]["monotonic_ns"], int)
        self.assertIsNone(self.worker.process)

    def test_stop_request_is_durable_before_termination_and_exit_is_retained(self):
        expected = 130 if os.name == "nt" else -9
        self.worker.process.pid = 123
        self.worker.tree = Mock()
        self.worker.tree.handle = object()
        self.worker.process.poll.side_effect = [None, None, expected, expected]

        def terminate():
            records = [
                json.loads(line)
                for line in (self.root / "run.lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(records[-1]["event"], "stop_requested")
            self.assertEqual(records[-1]["pid"], 123)

        self.worker.tree.terminate.side_effect = terminate
        self.worker.close()
        records = [
            json.loads(line) for line in (self.root / "run.lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([item["event"] for item in records], ["stop_requested", "exit_observed"])
        self.assertEqual(records[-1]["exit_code"], expected)
        self.assertEqual(records[-1]["expected_exit_code"], expected)
        self.assertTrue(records[-1]["stop_requested"])
        self.assertFalse(self.state.blocked())

    def test_lifecycle_write_failure_does_not_prevent_os_stop(self):
        expected = 1 if os.name == "nt" else -9
        self.worker.process.poll.side_effect = [None, None, expected, expected]
        process = self.worker.process
        original_open = Path.open

        def open_path(path, *args, **kwargs):
            if path.name.endswith(".lifecycle.jsonl"):
                raise PermissionError("diagnostic unavailable")
            return original_open(path, *args, **kwargs)

        with (
            patch.object(Path, "open", new=open_path),
            self.assertLogs("modules_forge.resident_worker", level="WARNING"),
        ):
            self.worker.close()
        process.kill.assert_called_once()
        self.assertIsNone(self.worker.process)
        self.assertFalse(self.state.blocked())

    def test_stop_request_survives_exit_confirmation_retry(self):
        expected = 130 if os.name == "nt" else -9
        self.worker.tree = Mock()
        self.worker.tree.handle = object()
        self.worker.process.poll.return_value = None
        self.worker.process.wait.side_effect = subprocess.TimeoutExpired("fixture", 15)
        clock = SimpleNamespace(monotonic=Mock(side_effect=[0, 0, 16]), monotonic_ns=time.monotonic_ns)
        with patch("modules_forge.resident_worker.time", clock):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.worker.close()
        self.assertTrue(self.state.blocked())
        self.worker.process.poll.return_value = expected
        self.worker.close()
        records = [
            json.loads(line) for line in (self.root / "run.lifecycle.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([item["event"] for item in records], ["stop_requested", "exit_observed"])
        self.assertTrue(records[-1]["stop_requested"])
        self.assertEqual(records[-1]["expected_exit_code"], expected)
        self.assertEqual(json.loads(self.state.path.read_text(encoding="utf-8"))["reason"], "worker_stop_unconfirmed")

    def test_death_between_initial_poll_and_stop_loop_is_detected(self):
        self.worker.process.poll.side_effect = [None, 0, 0]
        self.worker.close()
        self.assertTrue(self.state.blocked())

    def test_native_failure_during_requested_stop_is_detected(self):
        self.worker.process.poll.side_effect = [None, None, 0xC0000409, 0xC0000409]
        self.worker.close()
        self.assertTrue(self.state.blocked())

    @unittest.skipUnless(os.name == "nt", "Windows Job Object exit-code contract")
    def test_job_object_stop_cannot_mask_fatal_self_exit_one(self):
        self.worker.tree = Mock()
        self.worker.tree.handle = object()
        self.worker.process.poll.side_effect = [None, None, 1, 1]
        self.worker.close()
        self.assertTrue(self.state.blocked())

    def test_real_cpu_worker_stops_with_expected_os_code(self):
        script = self.root / "cpu_engine.py"
        script.write_text("def resident_run(payload): pass\n", encoding="utf-8")
        worker = ResidentWorker("cpu-fixture", self.root / "sessions", gpu_safety=True)
        try:
            worker.start(sys.executable, script, safe_environment(), {}, self.root / "cpu.log")
            deadline = time.monotonic() + 15
            while worker.result() is None:
                self.assertLess(time.monotonic(), deadline)
                time.sleep(0.05)
            process = worker.process
            worker.close()
            self.assertEqual(process.poll(), 130 if os.name == "nt" else -9)
            self.assertFalse(self.state.blocked())
        finally:
            worker.close()

    def test_expected_requested_stop_is_not_a_new_fault(self):
        expected = 1 if os.name == "nt" else -9
        self.worker.process.poll.side_effect = [None, None, expected, expected]
        self.worker.close()
        self.assertFalse(self.state.blocked())

    def test_known_fault_preserves_native_log_even_if_parent_kills_first(self):
        self.state.trip("gpu_execution", "moss-image")
        expected = 1 if os.name == "nt" else -9
        self.worker.process.poll.side_effect = [None, None, expected, expected]
        self.worker.close()
        self.assertEqual((self.root / "run.native.log").read_text(encoding="utf-8"), "native death")

    def test_fatal_exit_occurs_even_if_response_save_or_logging_fails(self):
        for target in ("atomic_json", "traceback.print_exc"):
            with self.subTest(target=target):
                command = {"id": "job", "payload": {}, "log": str(self.root / "command.log")}
                (self.root / "command.json").touch()
                module = Mock()
                module.resident_run.side_effect = RuntimeError("CUDA error: unknown error")
                stream = Mock()
                stream.to.side_effect = lambda log: nullcontext()
                with (
                    patch.object(resident_worker.importlib.util, "spec_from_file_location"),
                    patch.object(resident_worker.importlib.util, "module_from_spec", return_value=module),
                    patch.object(resident_worker, "_read_control_json", return_value=command),
                    patch("modules_forge.resident_worker." + target, side_effect=OSError("disk")),
                    patch.object(resident_worker.os, "_exit", side_effect=SystemExit(1)) as exit_process,
                ):
                    with self.assertRaises(SystemExit):
                        resident_worker._serve_commands("unused", self.root, stream, stream)
                exit_process.assert_called_once_with(1)
