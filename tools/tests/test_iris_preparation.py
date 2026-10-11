"""Preparation stays busy until its process tree has been closed."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.iris import preparation as preparation_module
from modules_forge.iris.core import IrisError
from modules_forge.iris.preparation import Preparation


class PreparationTests(unittest.TestCase):
    def test_start_preserves_job_contract_and_forwards_component_mode_to_thread(self):
        preparation = Preparation()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(preparation_module, "ROOT", Path(directory)),
            patch.object(preparation_module.threading, "Thread") as thread,
        ):
            identifier = preparation.start("generate", "w4a8", components_only=True)
            job = preparation.jobs[identifier]
            self.assertEqual(job["status"], "running")
            self.assertEqual(thread.call_args.kwargs["args"], (job, "generate", "w4a8", True))
            thread.return_value.start.assert_called_once_with()
            preparation.cancel(identifier)
            self.assertTrue(job["cancel"])
            with self.assertRaises(IrisError):
                preparation.start("generate", "int8", components_only=True)

    def test_start_rejects_non_generation_components_before_creating_job(self):
        preparation = Preparation()
        with self.assertRaisesRegex(IrisError, "画像生成|generate"):
            preparation.start("depth", "normal", components_only=True)
        self.assertEqual(preparation.jobs, {})

    def test_component_child_command_preserves_handshake_cleanup_and_cancellation(self):
        for cancelled in (False, True):
            with self.subTest(cancelled=cancelled), tempfile.TemporaryDirectory() as directory:
                job = {
                    "id": "test",
                    "status": "running",
                    "directory": directory,
                    "cancel": cancelled,
                    "process": None,
                    "tree": None,
                }
                events = []

                class Process:
                    def __init__(self, cancelled=cancelled):
                        self.stdin = io.BytesIO()
                        self.returncode = None if cancelled else 0

                    def poll(self):
                        return self.returncode

                class Tree:
                    def __init__(self, process):
                        self.process = process

                    def handshake(self):
                        return b"guard"

                    def terminate(self, events=events, job=job):
                        events.append(("terminate", job["status"]))
                        self.process.returncode = 130

                    def close(self, events=events, job=job):
                        events.append(("close", job["status"]))

                process = Process()
                with (
                    patch.object(preparation_module.subprocess, "Popen", return_value=process) as popen,
                    patch("modules_forge.yue2_studio.service.ProcessTree", Tree),
                    patch.object(preparation_module.time, "sleep"),
                ):
                    Preparation().execute(job, "generate", "int8", components_only=True)
                command = popen.call_args.args[0]
                self.assertEqual(
                    command[-6:], ["--task", "generate", "--precision", "int8", "--parent-guard", "--components-only"]
                )
                self.assertEqual(command.count("--components-only"), 1)
                self.assertEqual(job["status"], "cancelled" if cancelled else "complete")
                self.assertEqual(events[-1], ("close", "running"))
                self.assertTrue(process.stdin.closed)

    def test_handshake_failure_is_terminal_only_after_cleanup(self):
        events = []
        with tempfile.TemporaryDirectory() as directory:
            job = {
                "id": "test",
                "status": "running",
                "directory": directory,
                "cancel": False,
                "process": None,
                "tree": None,
            }

            class Process:
                stdin = io.BytesIO()
                returncode = None

                def poll(self):
                    return self.returncode

            class Tree:
                def __init__(self, process):
                    self.process = process

                def handshake(self):
                    raise OSError("handshake failed")

                def terminate(self):
                    events.append(job["status"])
                    self.process.returncode = 130

                def close(self):
                    events.append(job["status"])

            preparation = Preparation()
            with (
                patch("modules_forge.iris.preparation.subprocess.Popen", return_value=Process()),
                patch("modules_forge.yue2_studio.service.ProcessTree", Tree),
                patch("modules_forge.iris.preparation.time.sleep"),
            ):
                preparation.execute(job, "depth", "normal")
            self.assertEqual(events, ["running", "running"])
            self.assertEqual(job["status"], "error")
            self.assertIn("handshake", job["error"])
            self.assertTrue((Path(directory) / "setup.log").is_file())


if __name__ == "__main__":
    unittest.main()
