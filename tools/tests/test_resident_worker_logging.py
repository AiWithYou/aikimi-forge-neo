"""Job logs keep receiving library warnings after the first log is closed."""

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from modules_forge import resident_worker
from modules_forge.yue2_studio import worker as yue_worker
from modules_forge.yue2_studio.core import safe_environment


class ResidentLoggingTests(unittest.TestCase):
    def test_close_preserves_native_log_before_removing_control_directory(self):
        for exit_code in (0, 23):
            with self.subTest(exit_code=exit_code):
                self.check_close_preserves_native_log(exit_code)

    def check_close_preserves_native_log(self, exit_code):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = resident_worker.ResidentWorker("fixture", root / "sessions")
            worker.root.mkdir()
            worker._temporary = tempfile.TemporaryDirectory(dir=worker.root, delete=False)
            worker.directory = Path(worker._temporary.name)
            (worker.directory / "startup.log").write_text("native cause", encoding="utf-8")
            worker.log_path = root / "job.log"
            worker.process = Mock()
            worker.process.poll.return_value = exit_code
            worker.process.stdin = None
            try:
                worker.close()
                self.assertEqual((root / "job.native.log").read_text(encoding="utf-8"), "native cause")
                self.assertEqual(list(worker.root.iterdir()), [])
            finally:
                worker.close()

    def test_native_log_save_failure_keeps_exit_code_and_cause(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "startup.log").write_text("native cause", encoding="utf-8")
            worker = resident_worker.ResidentWorker("fixture", root)
            worker.directory = root
            worker.log_path = root / "job.log"
            worker.process = Mock()
            worker.process.poll.return_value = 23
            try:
                with patch("shutil.copyfile", side_effect=PermissionError("archive denied")):
                    with self.assertRaises(RuntimeError) as caught:
                        worker.result()
                message = str(caught.exception)
                self.assertIn("23", message)
                self.assertIn("native cause", message)
                self.assertIn("archive denied", message)
            finally:
                worker.process = None
                worker.close()

    def test_missing_native_log_keeps_exit_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            worker = resident_worker.ResidentWorker("fixture", root)
            worker.directory = root
            worker.log_path = root / "job.log"
            worker.process = Mock()
            worker.process.poll.return_value = 23
            try:
                with self.assertRaises(RuntimeError) as caught:
                    worker.result()
                self.assertIn("23", str(caught.exception))
                self.assertIn("startup.log", str(caught.exception))
            finally:
                worker.process = None
                worker.close()

    def wait_result(self, worker):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            result = worker.result()
            if result is not None:
                return result
            time.sleep(0.01)
        self.fail("CPU fixture worker did not finish")

    def test_native_crash_log_survives_cleanup_after_worker_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "engine.py"
            script.write_text(
                "import os\n"
                "def resident_run(payload):\n"
                "    if payload['crash']:\n"
                "        os.write(2, b'NATIVE ROOT CAUSE\\n' + b'x' * 6000 + b'\\nNATIVE END\\n')\n"
                "        os._exit(23)\n"
                "    os.write(2, b'EARLIER JOB NOISE' * 1000)\n"
                "    print('completed fixture job')\n",
                encoding="utf-8",
            )
            worker = resident_worker.ResidentWorker("fixture", root / "sessions")
            try:
                self.assertFalse(worker.start(sys.executable, script, safe_environment(), {"crash": False}, root / "first.log"))
                self.assertTrue(self.wait_result(worker)["ok"])
                self.assertTrue(worker.start(sys.executable, script, safe_environment(), {"crash": True}, root / "second.log"))
                with self.assertRaises(RuntimeError) as caught:
                    self.wait_result(worker)
                self.assertIn("23", str(caught.exception))
                self.assertIn("second.native.log", str(caught.exception))
                self.assertIn("NATIVE ROOT CAUSE", str(caught.exception))
                with patch("shutil.copyfile", side_effect=PermissionError("second poll")) as copy:
                    with self.assertRaises(RuntimeError) as repeated:
                        worker.result()
                    self.assertEqual(str(repeated.exception), str(caught.exception))
                    copy.assert_not_called()
                worker.close()
                native = (root / "second.native.log").read_bytes()
                self.assertIn(b'NATIVE ROOT CAUSE\n' + b'x' * 6000 + b'\nNATIVE END\n', native)
                self.assertFalse((root / "first.native.log").exists())
                self.assertEqual(list((root / "sessions").iterdir()), [])
            finally:
                worker.close()

    def run_two_jobs(self, first_fails=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "engine.py"
            script.write_text(
                "import logging, sys\n"
                "handler = None\n"
                "def resident_run(payload):\n"
                "    global handler\n"
                "    if handler is None:\n"
                "        handler = logging.StreamHandler(sys.stderr)\n"
                "    handler.emit(logging.LogRecord('fixture', logging.WARNING, '', 0, "
                "payload['label'] + ' warning', (), None))\n"
                "    print(payload['label'] + ' stdout')\n"
                "    print(payload['label'] + ' stderr', file=sys.stderr)\n"
                "    if payload['fail']:\n"
                "        raise ValueError('fixture failure')\n",
                encoding="utf-8",
            )
            commands = [
                {
                    "id": label,
                    "payload": {"label": label, "fail": first_fails and label == "first"},
                    "log": str(root / (label + ".log")),
                }
                for label in ("first", "second")
            ]
            (root / "command.json").write_text("{}", encoding="utf-8")
            module_key = "aikimi_resident_engine"
            previous = resident_worker.sys.modules.get(module_key)
            try:
                with (
                    patch.object(yue_worker, "parent_guard"),
                    patch.object(resident_worker, "_read_control_json", side_effect=[*commands, KeyboardInterrupt]),
                    self.assertRaises(KeyboardInterrupt),
                ):
                    resident_worker.serve(script, root)
                first = (root / "first.log").read_text(encoding="utf-8")
                second = (root / "second.log").read_text(encoding="utf-8")
                for label, text in (("first", first), ("second", second)):
                    self.assertIn(label + " warning", text.splitlines())
                    self.assertIn(label + " stdout", text)
                    self.assertIn(label + " stderr", text)
                    self.assertNotIn("Logging error", text)
                self.assertNotIn("second", first)
                self.assertNotIn("first", second)
                if first_fails:
                    self.assertIn("ValueError: fixture failure", first)
                    self.assertNotIn("fixture failure", second)
            finally:
                if previous is None:
                    resident_worker.sys.modules.pop(module_key, None)
                else:
                    resident_worker.sys.modules[module_key] = previous

    def test_library_handler_follows_successive_job_logs(self):
        self.run_two_jobs()

    def test_library_handler_follows_log_after_failed_job(self):
        self.run_two_jobs(first_fails=True)


if __name__ == "__main__":
    unittest.main()
