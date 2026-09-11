import os
import sys
import unittest
from pathlib import Path
from unittest import mock

from tools import run_ci_tests


class RunCiTestsTests(unittest.TestCase):
    def test_environment_forces_cpu_offline_and_disables_live_tests(self):
        environment = {
            "CUDA_VISIBLE_DEVICES": "0",
            "HF_HUB_OFFLINE": "0",
            "KREA2_LIVE_API_TEST": "1",
        }

        run_ci_tests.configure_ci_environment(environment)

        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(environment["KREA2_LIVE_API_TEST"], "0")
        self.assertTrue(all(environment[name] == "0" for name in run_ci_tests.DISABLED_LIVE_TESTS))
        self.assertTrue(all(environment[name] == "1" for name in run_ci_tests.OFFLINE_ENVIRONMENT))

    def test_discovery_rejects_repository_escape(self):
        with self.assertRaisesRegex(ValueError, "inside the repository"):
            run_ci_tests.discover_tests("../outside", "test_*.py")

    def test_repository_root_is_first_on_import_path(self):
        search_path = ["tools", str(run_ci_tests.REPOSITORY_ROOT), "stdlib"]

        run_ci_tests.configure_import_path(search_path)

        self.assertEqual(search_path[0], str(run_ci_tests.REPOSITORY_ROOT))
        self.assertEqual(search_path.count(str(run_ci_tests.REPOSITORY_ROOT)), 1)

    def test_discovery_rejects_missing_directory(self):
        with self.assertRaisesRegex(FileNotFoundError, "does not exist"):
            run_ci_tests.discover_tests("tools/missing-tests", "test_*.py")

    def test_explicit_modules_bypass_discovery(self):
        suite = unittest.TestSuite()
        modules = ["tools.tests.test_run_ci_tests"]

        with (
            mock.patch.object(
                run_ci_tests.unittest.defaultTestLoader,
                "loadTestsFromNames",
                return_value=suite,
            ) as load,
            mock.patch.object(run_ci_tests, "discover_tests") as discover,
        ):
            actual = run_ci_tests.load_tests("tools/tests", "test_*.py", modules)

        self.assertIs(actual, suite)
        load.assert_called_once_with(modules)
        discover.assert_not_called()

    def test_preload_imports_modules_after_policy_setup(self):
        with mock.patch.object(run_ci_tests.importlib, "import_module") as import_module:
            run_ci_tests.preload_modules(["modules.shared", "modules.ui_tempdir"])

        self.assertEqual(
            import_module.call_args_list,
            [mock.call("modules.shared"), mock.call("modules.ui_tempdir")],
        )

    def test_main_injects_cpu_argument_before_discovery(self):
        observed: dict[str, object] = {}

        class SuccessfulResult:
            @staticmethod
            def wasSuccessful():
                return True

        def load_tests(start_directory, pattern, modules):
            observed["environment"] = dict(os.environ)
            observed["start_directory"] = start_directory
            observed["pattern"] = pattern
            observed["modules"] = modules
            observed["argv"] = list(sys.argv)
            observed["search_path"] = list(sys.path)
            return unittest.TestSuite()

        with (
            mock.patch.object(run_ci_tests, "validate_python_version"),
            mock.patch.object(run_ci_tests, "load_tests", side_effect=load_tests),
            mock.patch.object(run_ci_tests, "preload_modules") as preload,
            mock.patch.object(
                run_ci_tests.unittest,
                "TextTestRunner",
                return_value=mock.Mock(run=mock.Mock(return_value=SuccessfulResult())),
            ),
            mock.patch.object(run_ci_tests.os, "chdir"),
            mock.patch.object(run_ci_tests.sys, "path", ["tools", "stdlib"]),
            mock.patch.dict(os.environ, {}, clear=True),
        ):
            exit_code = run_ci_tests.main(["--start-directory", "tools/tests", "--pattern", "test_security*.py"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(observed["argv"], [str(Path(run_ci_tests.__file__).resolve()), "--cpu"])
        self.assertEqual(observed["start_directory"], "tools/tests")
        self.assertEqual(observed["pattern"], "test_security*.py")
        self.assertEqual(observed["modules"], [])
        preload.assert_called_once_with([])
        self.assertEqual(observed["search_path"][0], str(run_ci_tests.REPOSITORY_ROOT))
        environment = observed["environment"]
        self.assertIsInstance(environment, dict)
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertEqual(environment["TRANSFORMERS_OFFLINE"], "1")


if __name__ == "__main__":
    unittest.main()
