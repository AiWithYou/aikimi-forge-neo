"""Resident control reads tolerate brief Windows locks without masking real failures."""

from __future__ import annotations

import ctypes
import errno
import os
import tempfile
import unittest
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge import resident_worker
from modules_forge.yue2_studio import worker as yue_worker
from modules_forge.yue2_studio.core import YuE2Error, atomic_json, read_json


@contextmanager
def exclusive_file_read(path):
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel32.CreateFileW
    create.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = create(str(path), 0x80000000, 0, None, 3, 0, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())

    def release():
        nonlocal handle
        if handle is not None:
            if not kernel32.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            handle = None

    try:
        yield release
    finally:
        release()


class ResidentControlReadTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.response = self.root / "response.json"
        atomic_json(self.response, {"id": "current", "ok": True})
        (self.root / "startup.log").write_text("fixture worker stopped", encoding="utf-8")
        self.worker = resident_worker.ResidentWorker("fixture", self.root)
        self.worker.directory = self.root
        self.worker.identifier = "current"
        self.worker.process = Mock()
        self.worker.process.poll.return_value = None
        self.addCleanup(self.clear_worker)

    def clear_worker(self):
        self.worker.process = None  # Only a fake process is attached by this fixture.
        self.worker.close()

    def sharing_error(self):
        error = PermissionError(errno.EACCES, "sharing violation", str(self.response))
        error.winerror = 32
        return error

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing semantics")
    def test_real_response_lock_is_released_before_second_read(self):
        attempts = []
        with exclusive_file_read(self.response) as release:
            self.assertTrue(self.response.exists())

            def read_after_release(path):
                attempts.append(path)
                if len(attempts) == 2:
                    release()
                try:
                    return read_json(path)
                except PermissionError as error:
                    self.assertEqual(error.errno, errno.EACCES)
                    self.assertIsNone(getattr(error, "winerror", None))
                    raise

            with patch.object(resident_worker, "read_json", side_effect=read_after_release):
                result = self.worker.result()
        self.assertEqual(result, {"id": "current", "ok": True})
        self.assertEqual(len(attempts), 2)

    def test_retried_old_response_does_not_complete_current_job(self):
        with (
            patch.object(
                resident_worker, "read_json", side_effect=[self.sharing_error(), {"id": "old", "ok": True}]
            ) as read,
            patch.object(resident_worker.time, "sleep"),
        ):
            self.assertIsNone(self.worker.result())
        self.assertEqual(read.call_count, 2)

    def test_permanent_permission_error_stops_at_deadline(self):
        error = self.sharing_error()
        with (
            patch.object(resident_worker, "read_json", side_effect=error) as read,
            patch.object(resident_worker.time, "monotonic", side_effect=[0.0, 0.0, 2.0]),
            patch.object(resident_worker.time, "sleep") as sleep,
            self.assertRaises(PermissionError) as caught,
        ):
            self.worker.result()
        self.assertIs(caught.exception, error)
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_worker_exit_reports_startup_error_without_retry(self):
        self.worker.process.poll.return_value = 1
        with (
            patch.object(resident_worker, "read_json", side_effect=self.sharing_error()) as read,
            patch.object(resident_worker.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "fixture worker stopped"),
        ):
            self.worker.result()
        read.assert_called_once_with(self.response)
        sleep.assert_not_called()

    def test_worker_exit_during_retry_stops_wait_and_reports_startup_log(self):
        self.worker.process.poll.side_effect = [None, 1, 1, 1]
        with (
            patch.object(resident_worker, "read_json", side_effect=self.sharing_error()) as read,
            patch.object(resident_worker.time, "monotonic", return_value=0.0),
            patch.object(resident_worker.time, "sleep") as sleep,
            self.assertRaisesRegex(RuntimeError, "fixture worker stopped"),
        ):
            self.worker.result()
        self.assertEqual(read.call_count, 2)
        sleep.assert_called_once_with(0.01)

    def test_other_permission_error_is_not_retried(self):
        error = PermissionError(errno.EPERM, "access policy")
        with (
            patch.object(resident_worker, "read_json", side_effect=error) as read,
            patch.object(resident_worker.time, "sleep") as sleep,
            self.assertRaises(PermissionError) as caught,
        ):
            self.worker.result()
        self.assertIs(caught.exception, error)
        read.assert_called_once_with(self.response)
        sleep.assert_not_called()

    def test_invalid_json_is_not_retried(self):
        self.response.write_text("{", encoding="utf-8")
        with (
            patch.object(resident_worker, "read_json", wraps=read_json) as read,
            patch.object(resident_worker.time, "sleep") as sleep,
            self.assertRaises(YuE2Error),
        ):
            self.worker.result()
        read.assert_called_once_with(self.response)
        sleep.assert_not_called()

    def test_matching_completed_response_is_kept_after_process_exit(self):
        self.worker.process.poll.return_value = 0
        self.assertEqual(self.worker.result(), {"id": "current", "ok": True})

    @unittest.skipUnless(os.name == "nt", "Windows file-sharing semantics")
    def test_serve_retries_real_command_lock(self):
        module_key = "aikimi_resident_engine"
        original_module = resident_worker.sys.modules.get(module_key)
        if original_module is None:
            self.addCleanup(resident_worker.sys.modules.pop, module_key, None)
        else:
            self.addCleanup(resident_worker.sys.modules.__setitem__, module_key, original_module)
        script = self.root / "engine.py"
        script.write_text('def resident_run(payload):\n    raise KeyboardInterrupt("fixture done")\n', encoding="utf-8")
        command = self.root / "command.json"
        atomic_json(command, {"id": "current", "payload": {}, "log": str(self.root / "job.log")})
        attempts = []
        with exclusive_file_read(command) as release:

            def read_after_release(path):
                attempts.append(path)
                if len(attempts) == 2:
                    release()
                return read_json(path)

            with (
                patch.object(yue_worker, "parent_guard"),
                patch.object(resident_worker, "read_json", side_effect=read_after_release),
                self.assertRaisesRegex(KeyboardInterrupt, "fixture done"),
            ):
                resident_worker.serve(script, self.root)
        self.assertEqual(attempts, [command, command])


if __name__ == "__main__":
    unittest.main()
