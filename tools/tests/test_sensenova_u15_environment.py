import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge import sensenova_u15_environment as environment


class SenseNovaEnvironmentTests(unittest.TestCase):
    def test_missing_worker_is_not_reported_ready_by_forge_packages(self):
        with mock.patch.object(environment, "WORKER_PYTHON", Path("missing-worker-python")):
            ready, message = environment.environment_status()
        self.assertFalse(ready)
        self.assertIn("-RuntimeOnly", message)

    def test_worker_package_versions_must_match_the_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            python = Path(directory) / "python"
            python.touch()
            with (
                mock.patch.object(environment, "WORKER_PYTHON", python),
                mock.patch.object(environment, "ENVIRONMENT", Path(directory)),
                mock.patch.object(environment, "expected_versions", return_value={"transformers": "4.57.6"}),
            ):
                for actual, expected_ready in (("5.10.4", False), ("4.57.6", True)):
                    distribution = SimpleNamespace(metadata={"Name": "transformers"}, version=actual)
                    with mock.patch.object(
                        environment.importlib.metadata, "distributions", return_value=[distribution]
                    ):
                        ready, _ = environment.environment_status()
                    self.assertEqual(ready, expected_ready)

    def test_worker_rejects_incompatible_running_environment(self):
        with (
            mock.patch.object(environment, "expected_versions", return_value={"transformers": "4.57.6"}),
            mock.patch.object(environment.importlib.metadata, "version", return_value="5.10.4"),
            self.assertRaisesRegex(RuntimeError, "4.57.6"),
        ):
            environment.validate_running_versions()

    def test_all_worker_requirements_are_exactly_pinned(self):
        versions = environment.expected_versions()
        self.assertEqual(versions["transformers"], "4.57.6")
        self.assertEqual(versions["huggingface-hub"], "0.36.2")
        self.assertTrue(
            all(version and not any(character in version for character in "*<>,;") for version in versions.values())
        )
