"""Startup failures must reclaim only newly owned ComfyUI process trees."""

from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import psutil

from modules_forge import minimax_h3_bridge as bridge
from modules_forge.minimax_h3_acceleration import H3Acceleration


class StartupCleanupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temporary = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(self.temporary) / "ComfyUI"
        self.root.mkdir()
        (self.root / "main.py").write_text("# fixture\n", encoding="utf-8")
        (self.root / "models").mkdir()
        self.url = "http://127.0.0.1:8189"
        self.identity = (self.root.resolve(), self.url)
        self.process = Mock(pid=12345)
        self.process.poll.return_value = None
        self.stack.enter_context(patch.object(bridge, "_MANAGED_PROCESS", None))
        self.stack.enter_context(patch.object(bridge, "_MANAGED_PROCESS_IDENTITY", None))
        self.stack.enter_context(patch.object(bridge, "server_runtime_root", return_value=None))
        self.stack.enter_context(patch.object(bridge, "_python_for_runtime", return_value=Path(sys.executable)))
        self.stack.enter_context(patch.object(bridge.workflow_handoff, "install_bundle", return_value=False))
        self.stack.enter_context(patch("modules_forge.nanosaur2_studio.source_ready", return_value=False))
        self.stack.enter_context(patch("modules_forge.ming_local.source_ready", return_value=False))
        self.spawn = self.stack.enter_context(patch.object(bridge.subprocess, "Popen", return_value=self.process))
        self.disconnected = bridge.RuntimeReadiness(self.root, self.url, connected=False, acceleration=H3Acceleration())
        self.connected = bridge.RuntimeReadiness(self.root, self.url, connected=True, acceleration=H3Acceleration())

    def start(self, **kwargs):
        return bridge._start_runtime_locked(
            self.root, self.url, Path(self.temporary) / "logs", initial_readiness=self.disconnected, **kwargs
        )

    def test_new_spawn_timeout_is_stopped_before_original_error_returns(self):
        with patch.object(bridge, "_stop_managed_runtime") as stop:
            with self.assertRaisesRegex(bridge.H3BridgeError, "起動を 0 秒"):
                self.start(wait_seconds=0)
        self.spawn.assert_called_once()
        stop.assert_called_once()

    def test_new_spawn_readiness_exception_is_stopped(self):
        with (
            patch.object(bridge.time, "sleep"),
            patch.object(bridge, "_server_api_responding", return_value=True),
            patch.object(bridge, "inspect_readiness", side_effect=OSError("fixture inspection failed")),
            patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            with self.assertRaisesRegex(OSError, "fixture inspection failed"):
                self.start(wait_seconds=5)
        stop.assert_called_once()

    def test_reused_managed_runtime_timeout_does_not_stop_existing_process(self):
        bridge._MANAGED_PROCESS = self.process
        bridge._MANAGED_PROCESS_IDENTITY = self.identity
        with patch.object(bridge, "_stop_managed_runtime") as stop:
            with self.assertRaises(bridge.H3BridgeError):
                self.start(wait_seconds=0)
        self.spawn.assert_not_called()
        stop.assert_not_called()

    def test_external_runtime_timeout_is_not_stopped(self):
        with (
            patch.object(bridge, "server_runtime_root", return_value=self.root),
            patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            with self.assertRaises(bridge.H3BridgeError):
                self.start(wait_seconds=0)
        self.spawn.assert_not_called()
        stop.assert_not_called()

    def test_new_spawn_success_is_retained(self):
        with (
            patch.object(bridge.time, "sleep"),
            patch.object(bridge, "_server_api_responding", return_value=True),
            patch.object(bridge, "inspect_readiness", return_value=self.connected),
            patch.object(bridge, "_stop_managed_runtime") as stop,
        ):
            self.assertIs(self.start(wait_seconds=5), self.connected)
        stop.assert_not_called()

    def test_failed_startup_with_real_cleanup_does_not_reenter_process_lock(self):
        errors = []

        def stopped(**_):
            self.process.poll.return_value = 0

        def attempt():
            try:
                self.start(wait_seconds=0)
            except BaseException as error:
                errors.append(error)

        self.process.wait.side_effect = stopped
        with (
            patch("psutil.Process", side_effect=psutil.NoSuchProcess(self.process.pid)),
            patch.object(bridge, "_loopback_server_process", return_value=None),
        ):
            worker = threading.Thread(target=attempt, daemon=True)
            worker.start()
            worker.join(timeout=2)
            deadlocked = worker.is_alive()
            if deadlocked:
                # Repair the deliberately detected non-reentrant deadlock so
                # the rest of the test suite can still use the real lock.
                bridge._PROCESS_LOCK.release()
                worker.join(timeout=2)
        self.assertFalse(deadlocked, "Failed startup reentered the non-reentrant process lock")
        self.assertIsInstance(errors[0], bridge.H3BridgeError)
        self.process.terminate.assert_called_once()
        self.assertIsNone(bridge._MANAGED_PROCESS)


class ManagedTreeCleanupTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.parent = Mock(pid=12345)
        self.parent.poll.return_value = None
        self.parent.wait.side_effect = self.stopped
        self.child = Mock(pid=23456)
        self.child.create_time.return_value = 10.0
        self.child.is_running.return_value = True
        self.fresh_child = Mock(pid=23456)
        self.fresh_child.create_time.return_value = 10.0
        self.parent_info = Mock(pid=12345)
        self.parent_info.children.return_value = [self.child]
        self.stack.enter_context(patch.object(bridge, "_MANAGED_PROCESS", self.parent))
        self.stack.enter_context(
            patch.object(bridge, "_MANAGED_PROCESS_IDENTITY", (Path("runtime"), "http://127.0.0.1:8189"))
        )
        self.stack.enter_context(patch.object(bridge, "_loopback_server_process", return_value=None))
        self.lookup = self.stack.enter_context(patch.object(psutil, "Process", side_effect=self.lookup_process))
        self.wait = self.stack.enter_context(patch.object(psutil, "wait_procs", return_value=([self.child], [])))

    def stopped(self, timeout):
        self.parent.poll.return_value = 0
        return 0

    def lookup_process(self, pid):
        if pid == self.parent.pid:
            return self.parent_info
        if pid == self.child.pid:
            return self.fresh_child
        raise psutil.NoSuchProcess(pid)

    def test_prebind_child_exits_before_launcher_is_stopped(self):
        calls = []
        self.child.terminate.side_effect = lambda: calls.append("child")
        self.parent.terminate.side_effect = lambda: calls.append("parent")
        bridge._stop_managed_runtime()
        self.assertEqual(calls, ["child", "parent"])
        self.assertIsNone(bridge._MANAGED_PROCESS)

    def test_reused_child_pid_is_not_terminated(self):
        self.fresh_child.create_time.return_value = 20.0
        bridge._stop_managed_runtime()
        self.child.terminate.assert_not_called()
        self.fresh_child.terminate.assert_not_called()
        self.parent.terminate.assert_called_once()

    def test_failed_old_spawn_cannot_stop_a_new_managed_process(self):
        bridge._stop_managed_runtime(expected_process=Mock(pid=34567))
        self.lookup.assert_not_called()
        self.parent.terminate.assert_not_called()
        self.assertIs(bridge._MANAGED_PROCESS, self.parent)

    def test_external_listener_survives_owned_tree_cleanup(self):
        foreign = Mock(pid=34567)
        with patch.object(bridge, "_owns_runtime_process", return_value=False):
            bridge._stop_managed_runtime(foreign)
        foreign.terminate.assert_not_called()
        foreign.kill.assert_not_called()
        self.child.terminate.assert_called_once()

    def test_unconfirmed_descendants_preserve_parent_and_management(self):
        self.parent_info.children.side_effect = psutil.AccessDenied(self.parent.pid)
        bridge._stop_managed_runtime()
        self.parent.terminate.assert_not_called()
        self.parent.kill.assert_not_called()
        self.assertIs(bridge._MANAGED_PROCESS, self.parent)


if __name__ == "__main__":
    unittest.main()
