"""CPU regressions for failed resident-worker startup and command dispatch."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge import resident_worker
from modules_forge.yue2_studio.core import safe_environment


class ResidentStartupCleanupTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.script = self.root / "engine.py"
        self.script.write_text("def resident_run(payload):\n    pass\n", encoding="utf-8")
        self.worker = resident_worker.ResidentWorker("test", self.root / "sessions")
        self.addCleanup(self.worker.close)

    def start(self, log_path=None):
        return self.worker.start(
            sys.executable,
            self.script,
            safe_environment(),
            {},
            log_path or self.root / "job.log",
        )

    def assert_closed(self, process=None):
        if process is not None:
            self.assertIsNotNone(process.poll(), "failed startup must confirm worker exit")
        self.assertIsNone(self.worker.process)
        self.assertIsNone(self.worker.directory)
        self.assertIsNone(self.worker._temporary)
        self.assertEqual(list((self.root / "sessions").iterdir()), [])

    def test_spawn_failure_removes_control_directory(self):
        with patch.object(resident_worker.subprocess, "Popen", side_effect=OSError("spawn failed")):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                self.start()

        self.assert_closed()

    def test_log_failure_after_spawn_confirms_exit_and_removes_control_directory(self):
        log_path = self.root / "log-is-directory"
        log_path.mkdir()
        processes = []
        original_popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(resident_worker.subprocess, "Popen", side_effect=spawn):
            with self.assertRaises(OSError):
                self.start(log_path)

        self.assertEqual(len(processes), 1)
        self.assert_closed(processes[0])

    def test_command_failure_confirms_exit_and_removes_control_directory(self):
        processes = []
        original_popen = subprocess.Popen

        def spawn(*args, **kwargs):
            process = original_popen(*args, **kwargs)
            processes.append(process)
            return process

        with (
            patch.object(resident_worker.subprocess, "Popen", side_effect=spawn),
            patch.object(resident_worker, "atomic_json", side_effect=OSError("command failed")),
            self.assertRaisesRegex(OSError, "command failed"),
        ):
            self.start()

        self.assertEqual(len(processes), 1)
        self.assert_closed(processes[0])


if __name__ == "__main__":
    unittest.main()
