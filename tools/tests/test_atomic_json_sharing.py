"""Windows result polling must not make an atomic writer lose its process."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.yue2_studio import core


def sharing_error(code):
    error = PermissionError("fixture file is open")
    error.winerror = code
    return error


class AtomicJsonSharingTests(unittest.TestCase):
    def test_short_windows_read_lock_retries_without_removing_previous_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.json"
            core.atomic_json(path, {"id": "old"})
            replace = core.os.replace
            attempts = []

            def locked_then_replace(source, destination):
                self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"id": "old"})
                attempts.append(1)
                if len(attempts) <= 3:
                    raise sharing_error((5, 32, 33)[len(attempts) - 1])
                replace(source, destination)

            with patch.object(core.os, "replace", side_effect=locked_then_replace), patch.object(core.time, "sleep"):
                core.atomic_json(path, {"id": "new", "text": "日本語"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["id"], "new")
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_permanent_denial_is_bounded_preserves_old_record_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.json"
            core.atomic_json(path, {"id": "old"})
            with (
                patch.object(core.os, "replace", side_effect=sharing_error(5)) as replace,
                patch.object(core.time, "monotonic", side_effect=[0, 0.1, 1.1]),
                patch.object(core.time, "sleep"),
                self.assertRaises(PermissionError),
            ):
                core.atomic_json(path, {"id": "new"})
            self.assertEqual(replace.call_count, 2)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"id": "old"})
            self.assertEqual(list(Path(temporary).iterdir()), [path])

    def test_other_io_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "response.json"
            with (
                patch.object(core.os, "replace", side_effect=OSError("disk full")) as replace,
                patch.object(core.time, "sleep") as sleep,
                self.assertRaises(OSError),
            ):
                core.atomic_json(path, {"id": "new"})
            replace.assert_called_once()
            sleep.assert_not_called()
            self.assertEqual(list(Path(temporary).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
