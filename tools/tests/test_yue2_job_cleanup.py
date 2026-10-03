"""CPU regressions for YuE2 startup failures before job-process publication."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge.yue2_studio import service


class YuE2StartupCleanupTests(unittest.TestCase):
    def test_unprintable_exception_still_publishes_failed_job_status(self):
        class UnprintableError(RuntimeError):
            def __str__(self):
                raise ValueError("invalid exception message")

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            studio = service.Studio(root / "runtime", root / "outputs")
            job = service.Job("job", "owner", root)
            lease = Mock()
            with patch.object(service, "runtime_manifest", side_effect=UnprintableError()):
                with self.assertRaisesRegex(ValueError, "invalid exception message"):
                    studio._run(job, SimpleNamespace(engine="official"), lease, None, None)

            self.assertTrue(job.done.is_set())
            self.assertEqual(job.final["state"], "failed")
            self.assertEqual(service.read_json(root / "status.json"), job.final)
            lease.release.assert_called_once()

    def test_startup_failures_keep_gpu_until_worker_exit_is_confirmed(self):
        for boundary in ("startup", "session_metadata"):
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as root:
                root = Path(root)
                studio = service.Studio(root / "runtime", root / "outputs")
                process = Mock()
                process.poll.return_value = None
                lease = Mock()
                resident = Mock()
                resident.process = process
                resident.tree = Mock()

                def shutdown(studio=studio, resident=resident, process=process):
                    resident.close.side_effect = None
                    process.poll.return_value = 0
                    studio.shutdown()

                self.addCleanup(shutdown)
                if boundary == "startup":
                    resident.start.side_effect = OSError("startup failed")
                else:
                    resident.start.return_value = False
                stop_attempts = []

                def close(lease=lease, stop_attempts=stop_attempts, process=process):
                    lease.release.assert_not_called()
                    stop_attempts.append(True)
                    if len(stop_attempts) == 1:
                        raise OSError("stop failed")
                    process.poll.return_value = 0

                resident.close.side_effect = close
                studio._resident = resident
                runtime_lock = studio._runtime_lock = Mock()
                job = service.Job("job", "owner", root)
                original_atomic_json = service.atomic_json

                def atomic_json(path, value, boundary=boundary, original_atomic_json=original_atomic_json):
                    if boundary == "session_metadata" and path.name == "worker-session.json":
                        raise OSError("session metadata failed")
                    original_atomic_json(path, value)

                with (
                    patch.object(service, "runtime_manifest", return_value={"python": sys.executable}),
                    patch.object(service, "atomic_json", side_effect=atomic_json),
                    patch.object(service.time, "sleep") as sleep,
                ):
                    studio._run(job, SimpleNamespace(engine="official"), lease, None, runtime_lock)

                self.assertEqual(len(stop_attempts), 2)
                sleep.assert_called_once_with(0.5)
                self.assertEqual(process.poll(), 0)
                lease.release.assert_called_once_with()
                runtime_lock.close.assert_called_once_with()
                self.assertTrue(job.done.is_set())
                self.assertEqual(job.final["state"], "failed")


if __name__ == "__main__":
    unittest.main()
