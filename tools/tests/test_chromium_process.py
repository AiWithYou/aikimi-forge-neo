"""Ownership stays bounded even when a browser root exits before its children."""

from __future__ import annotations

import ctypes as c
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.tests import chromium_process


class _FakeKernel:
    def __init__(self):
        self.events = []
        self.terminated = False
        self.member = True
        self.wait_result = 0
        self.pids = [42]
        self.open_processes = set()
        self.active_until_handles_closed = False

    def CreateJobObjectW(self, *_args):
        return 10

    def SetInformationJobObject(self, _handle, _kind, data, _size):
        limits = c.cast(data, c.POINTER(chromium_process._ExtendedLimit)).contents
        self.events.append(("limits", limits.basic.flags, limits.basic.active))
        return True

    def QueryInformationJobObject(self, _handle, kind, data, _size, _length):
        if kind == 3:
            self.events.append(("snapshot",))
            listed = min(len(self.pids), (_size - 8) // c.sizeof(c.c_size_t))
            c.cast(data, c.POINTER(c.c_uint32))[0] = len(self.pids)
            c.cast(data, c.POINTER(c.c_uint32))[1] = listed
            for index, pid in enumerate(self.pids[:listed]):
                c.cast(c.addressof(data) + 8, c.POINTER(c.c_size_t))[index] = pid
            return listed == len(self.pids)
        else:
            active = not self.terminated or (self.active_until_handles_closed and self.open_processes)
            c.cast(data, c.POINTER(chromium_process._Accounting)).contents.active = int(bool(active))
        return True

    def OpenProcess(self, rights, inherit, pid):
        self.events.append(("open", rights, inherit, pid))
        self.open_processes.add(20)
        return 20

    def IsProcessInJob(self, process, job, member):
        self.events.append(("member", process, job))
        c.cast(member, c.POINTER(c.c_int))[0] = self.member
        return True

    def TerminateJobObject(self, handle, code):
        self.events.append(("terminate", handle, code))
        self.terminated = True
        return True

    def WaitForSingleObject(self, handle, milliseconds):
        self.events.append(("wait", handle, milliseconds))
        return self.wait_result

    def CloseHandle(self, handle):
        self.events.append(("close", handle))
        self.open_processes.discard(handle)
        return True


class WindowsJobLifecycleTests(unittest.TestCase):
    def make_job(self, kernel):
        with patch.object(chromium_process, "_kernel32", return_value=kernel):
            return chromium_process._WindowsJob()

    def test_close_retains_owned_handles_before_termination_and_waits_for_rundown(self):
        kernel = _FakeKernel()
        self.make_job(kernel).close()
        self.assertEqual(
            kernel.events[:6],
            [
                ("limits", 0x2000, 0),
                ("limits", 0x2008, 1),
                ("snapshot",),
                ("open", 0x101000, False, 42),
                ("member", 20, 10),
                ("terminate", 10, 1),
            ],
        )
        self.assertEqual(kernel.events[6][0:2], ("wait", 20))
        self.assertGreater(kernel.events[6][2], 0)
        self.assertEqual([event for event in kernel.events if event[0] == "close"], [("close", 20), ("close", 10)])

    def test_process_wait_timeout_is_failure_and_closes_all_handles(self):
        kernel = _FakeKernel()
        kernel.wait_result = 258
        with self.assertRaisesRegex(TimeoutError, "rundown"):
            self.make_job(kernel).close(timeout=0.1)
        self.assertEqual(kernel.events[-2:], [("close", 20), ("close", 10)])
        self.assertTrue(kernel.terminated)

    def test_pid_reuse_never_waits_or_terminates_an_unrelated_process(self):
        kernel = _FakeKernel()
        kernel.member = False
        with self.assertRaisesRegex(RuntimeError, "membership"):
            self.make_job(kernel).close()
        self.assertTrue(kernel.terminated)
        self.assertFalse(any(event[0] == "wait" for event in kernel.events))
        self.assertEqual([event for event in kernel.events if event[0] == "close"], [("close", 20), ("close", 10)])

    def test_failed_snapshot_still_terminates_job(self):
        kernel = _FakeKernel()
        kernel.QueryInformationJobObject = Mock(return_value=False)
        with (
            patch.object(c, "get_last_error", return_value=5, create=True),
            patch.object(chromium_process, "_native_error", return_value=OSError("snapshot failed")),
            self.assertRaisesRegex(OSError, "snapshot failed"),
        ):
            self.make_job(kernel).close()
        self.assertTrue(kernel.terminated)
        self.assertEqual(kernel.events[-1], ("close", 10))

    def test_partial_snapshot_failure_waits_previously_verified_member(self):
        kernel = _FakeKernel()
        kernel.pids = [42, 43]
        kernel.OpenProcess = Mock(side_effect=[20, 0])
        with (
            patch.object(chromium_process, "_native_error", return_value=OSError("open failed")),
            self.assertRaisesRegex(OSError, "open failed"),
        ):
            self.make_job(kernel).close()
        self.assertTrue(kernel.terminated)
        self.assertEqual([event[1] for event in kernel.events if event[0] == "wait"], [20])
        self.assertEqual(kernel.events[-2:], [("close", 20), ("close", 10)])

    def test_process_list_buffer_grows_to_capture_every_member(self):
        kernel = _FakeKernel()
        kernel.pids = list(range(100, 133))
        with patch.object(c, "get_last_error", return_value=234, create=True):
            self.make_job(kernel).close()
        self.assertEqual(kernel.events.count(("snapshot",)), 2)
        self.assertEqual(len([event for event in kernel.events if event[0] == "wait"]), 33)

    def test_accounting_response_after_cleanup_deadline_is_not_success(self):
        kernel = _FakeKernel()
        clock = [100.0]
        query = kernel.QueryInformationJobObject

        def delayed_accounting(*args):
            result = query(*args)
            if args[1] == 1:
                clock[0] = 106.0
            return result

        kernel.QueryInformationJobObject = delayed_accounting
        with (
            patch.object(chromium_process.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaisesRegex(TimeoutError, "accounting deadline"),
        ):
            self.make_job(kernel).close(timeout=5)
        self.assertEqual(kernel.events[-2:], [("close", 20), ("close", 10)])

    def test_gate_wait_uses_only_cleanup_budget_remaining_after_job(self):
        clock = [100.0]
        job, gate = Mock(), Mock()
        gate.poll.return_value = 0

        def close_job(timeout):
            self.assertEqual(timeout, 5)
            clock[0] = 104.0

        job.close.side_effect = close_job
        with patch.object(chromium_process.time, "monotonic", side_effect=lambda: clock[0]):
            chromium_process._WindowsProcess(gate, job, 42, ["chrome"]).close_tree(timeout=5)
        gate.wait.assert_called_once_with(timeout=1)

    def test_signaled_process_handles_are_closed_before_active_count_confirmation(self):
        kernel = _FakeKernel()
        kernel.active_until_handles_closed = True
        self.make_job(kernel).close(timeout=0.05)
        self.assertFalse(kernel.open_processes)


class ChromiumGateTests(unittest.TestCase):
    def test_gate_assigns_itself_before_spawning_any_child(self):
        order = []
        child = Mock(pid=1234)
        child.wait.return_value = 0

        def spawn(*_args, **_kwargs):
            order.append("spawn")
            return child

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(chromium_process, "_join_windows_job", side_effect=lambda _name: order.append("join")),
            patch.object(chromium_process.subprocess, "Popen", side_effect=spawn),
        ):
            status = Path(directory) / "pid.json"
            self.assertEqual(chromium_process._gate("job", status, ["chrome"]), 0)
            self.assertEqual(json.loads(status.read_text()), {"pid": 1234})
        self.assertEqual(order, ["join", "spawn"])

    def test_assignment_failure_never_starts_browser(self):
        with (
            patch.object(chromium_process, "_join_windows_job", side_effect=OSError("assignment failed")),
            patch.object(chromium_process.subprocess, "Popen") as spawn,
            self.assertRaisesRegex(OSError, "assignment failed"),
        ):
            chromium_process._gate("job", Path("unused.json"), ["chrome"])
        spawn.assert_not_called()

    def test_pre_handshake_failure_stops_gate_before_closing_job(self):
        order = []
        gate = Mock()
        gate.poll.return_value = None

        def kill_gate():
            self.assertTrue(Path(spawn.call_args.args[0][6]).parent.is_dir())
            order.append("kill gate")

        gate.kill.side_effect = kill_gate
        gate.wait.side_effect = lambda timeout: order.append("wait gate") or 1
        job = Mock(name="job")
        job.close.side_effect = lambda timeout=5: order.append("close job")
        with (
            patch.object(chromium_process, "_windows", return_value=True),
            patch.object(chromium_process, "_WindowsJob", return_value=job),
            patch.object(chromium_process.subprocess, "Popen", return_value=gate) as spawn,
            patch.object(chromium_process, "_read_pid", side_effect=TimeoutError("handshake deadline")),
            self.assertRaisesRegex(TimeoutError, "handshake deadline"),
        ):
            chromium_process.spawn_chromium(["chrome"], stderr=subprocess.DEVNULL)
        self.assertEqual(order, ["kill gate", "wait gate", "close job"])
        self.assertFalse(Path(spawn.call_args.args[0][6]).parent.exists())

    def test_process_creation_failure_closes_empty_job(self):
        job = Mock()
        with (
            patch.object(chromium_process, "_windows", return_value=True),
            patch.object(chromium_process, "_WindowsJob", return_value=job),
            patch.object(chromium_process.subprocess, "Popen", side_effect=OSError("creation failed")),
            self.assertRaisesRegex(OSError, "creation failed"),
        ):
            chromium_process.spawn_chromium(["chrome"], stderr=subprocess.DEVNULL)
        job.close.assert_called_once()

    def test_exited_gate_without_handshake_is_failure(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "gate exited"):
            gate = Mock()
            gate.poll.return_value = 1
            chromium_process._read_pid(gate, Path(directory) / "missing.json", time.monotonic() + 1)

    def test_malformed_or_oversized_handshake_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            status = Path(directory) / "pid.json"
            for value in (b"{", b"x" * 129, b'{"pid":true}', b'{"pid":-1}', b'{"pid":1,"extra":2}'):
                with self.subTest(value=value[:20]), self.assertRaises(ValueError):
                    status.write_bytes(value)
                    chromium_process._read_pid(Mock(), status, time.monotonic() + 1)

    def test_atomic_publish_sharing_window_uses_same_handshake_deadline(self):
        gate = Mock()
        gate.poll.return_value = None
        with (
            patch.object(
                chromium_process, "_read_status", side_effect=[PermissionError("publishing"), b'{"pid":1234}']
            ),
            patch.object(chromium_process.time, "sleep"),
        ):
            self.assertEqual(chromium_process._read_pid(gate, Path("pid.json"), time.monotonic() + 1), 1234)

    def test_ready_handshake_after_deadline_is_rejected(self):
        clock = [100.0]

        def late_read(*_args):
            clock[0] = 102
            return b'{"pid":1234}'

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(chromium_process.time, "monotonic", side_effect=lambda: clock[0]),
            self.assertRaises(TimeoutError),
        ):
            status = Path(directory) / "pid.json"
            status.write_text('{"pid":1234}')
            with patch.object(chromium_process, "_read_status", side_effect=late_read):
                chromium_process._read_pid(Mock(), status, 101)

    def test_posix_preserves_existing_popen_contract(self):
        with (
            patch.object(chromium_process, "_windows", return_value=False),
            patch.object(chromium_process.subprocess, "Popen") as spawn,
        ):
            actual = chromium_process.spawn_chromium(["chrome"], deadline=1, start_new_session=True)
        self.assertIs(actual, spawn.return_value)
        spawn.assert_called_once_with(["chrome"], start_new_session=True)

    def test_import_is_dependency_free(self):
        root = str(Path(__file__).resolve().parents[2])
        script = (
            f"import sys;sys.path.insert(0,{root!r});import tools.tests.chromium_process;"
            "assert not {'torch','gradio','psutil','websockets'}.intersection(sys.modules)"
        )
        result = subprocess.run(  # noqa: S603 - current Python and fixed import probe
            [sys.executable, "-I", "-S", "-c", script], capture_output=True, text=True, timeout=10, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(os.name == "nt", "Windows Job Object native checks")
class WindowsChromiumOwnershipTests(unittest.TestCase):
    def test_admission_barrier_preserves_existing_members_and_rejects_new_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready, attempt, result, marker = [root / name for name in ("ready", "attempt", "result", "unexpected")]
            code = """
import json, subprocess, sys, time
from pathlib import Path
ready, attempt, result, marker = map(Path, sys.argv[1:])
child = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time;time.sleep(90)'])
ready.write_text('ready')
while not attempt.exists():
    time.sleep(0.01)
try:
    subprocess.Popen([sys.executable, '-I', '-S', '-c', 'from pathlib import Path;import sys;Path(sys.argv[1]).touch()', str(marker)])
except OSError as error:
    result.write_text(json.dumps({'error': error.winerror, 'existing_child_alive': child.poll() is None}))
else:
    result.write_text('{}')
time.sleep(90)
"""
            owned = chromium_process.spawn_chromium(
                [sys._base_executable, "-I", "-S", "-c", code, *map(str, (ready, attempt, result, marker))],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                deadline=time.monotonic() + 10,
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
                owned._job._freeze_admission()
                self.assertIsNone(owned.poll())
                attempt.touch()
                while not result.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(result.exists())
                self.assertEqual(json.loads(result.read_text()), {"error": 1816, "existing_child_alive": True})
                self.assertFalse(marker.exists())
            finally:
                owned.close_tree(timeout=5)

    def test_orphaned_grandchild_releases_lock_without_stopping_unrelated_process(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "browser-cache.tmp"
            ready = Path(directory) / "child.ready"
            child_code = (
                "import sys,time;from pathlib import Path;"
                "handle=open(sys.argv[1],'wb');Path(sys.argv[2]).write_text('ready');time.sleep(90)"
            )
            root_code = (
                "import subprocess,sys;"
                "subprocess.Popen([sys.executable,'-I','-S','-c',sys.argv[1],sys.argv[2],sys.argv[3]])"
            )
            unrelated = subprocess.Popen(  # noqa: S603 - test-owned Python sleeper
                [sys._base_executable, "-I", "-S", "-c", "import time;time.sleep(90)"],
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            owned = None
            try:
                owned = chromium_process.spawn_chromium(
                    [sys._base_executable, "-I", "-S", "-c", root_code, child_code, str(lock), str(ready)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    deadline=time.monotonic() + 10,
                )
                self.assertEqual(owned.wait(timeout=10), 0)
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(ready.exists())
                with self.assertRaises(PermissionError):
                    lock.unlink()
                owned.close_tree(timeout=5)
                lock.unlink()
                self.assertIsNone(unrelated.poll())
            finally:
                try:
                    if owned is not None:
                        owned.close_tree(timeout=5)
                finally:
                    unrelated.kill()
                    unrelated.wait(timeout=5)

    def test_failed_browser_creation_is_reaped(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(RuntimeError, "gate exited"):
            chromium_process.spawn_chromium(
                [str(Path(directory) / "missing.exe")], stderr=subprocess.DEVNULL, deadline=time.monotonic() + 10
            )
