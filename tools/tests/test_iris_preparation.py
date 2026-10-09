"""Preparation stays busy until its process tree has been closed."""

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.iris.preparation import Preparation


class PreparationTests(unittest.TestCase):
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
