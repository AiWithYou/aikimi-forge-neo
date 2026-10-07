"""Job logs keep receiving library warnings after the first log is closed."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge import resident_worker
from modules_forge.yue2_studio import worker as yue_worker


class ResidentLoggingTests(unittest.TestCase):
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
