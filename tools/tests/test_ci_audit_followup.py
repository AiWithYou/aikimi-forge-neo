"""CPU-only regression tests for the shared test runner and audit coverage."""

from __future__ import annotations

import io
import os
import re
import sys
import unittest
import unittest.mock as mock
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from tools import run_ci_tests as runner

ROOT = Path(__file__).resolve().parents[2]


class RunnerExitTests(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.isolation_active = False
        self.enterContext(redirect_stdout(self.stdout))
        self.enterContext(redirect_stderr(self.stderr))
        self.enterContext(mock.patch.dict(os.environ))
        self.enterContext(mock.patch.object(sys, "path", list(sys.path)))
        self.enterContext(mock.patch.object(sys, "argv", list(sys.argv)))
        self.enterContext(mock.patch.object(sys, "dont_write_bytecode", sys.dont_write_bytecode))
        self.enterContext(mock.patch.object(runner.os, "chdir"))
        # Version gating is tested separately; these tests exercise runner policy
        # without importing the application's pending-job store or GPU stack.
        self.enterContext(mock.patch.object(runner, "validate_python_version"))
        self.enterContext(mock.patch.object(runner, "isolated_h3_records", self.isolated_records))

    @contextmanager
    def isolated_records(self):
        self.assertFalse(self.isolation_active)
        self.isolation_active = True
        try:
            yield
        finally:
            self.isolation_active = False

    def run_suite(self, suite):
        with mock.patch.object(runner, "load_tests", return_value=suite):
            code = runner.main(["--verbosity", "0"])
        self.assertFalse(self.isolation_active)
        return code

    def test_empty_suite_is_not_a_success(self):
        self.assertEqual(self.run_suite(unittest.TestSuite()), 5)
        self.assertIn("No tests", self.stderr.getvalue())

    def test_passing_suite_succeeds(self):
        case = unittest.FunctionTestCase(lambda: None)
        self.assertEqual(self.run_suite(unittest.TestSuite([case])), 0)

    def test_assertion_failure_fails(self):
        def fail():
            raise AssertionError("regression sentinel")

        self.assertEqual(self.run_suite(unittest.TestSuite([unittest.FunctionTestCase(fail)])), 1)

    def test_fixture_error_is_failure_not_empty_suite(self):
        class BrokenFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise RuntimeError("fixture sentinel")

            def test_never_runs(self):
                pass

        suite = unittest.TestLoader().loadTestsFromTestCase(BrokenFixture)
        self.assertEqual(self.run_suite(suite), 1)

    def test_class_level_skip_remains_successful(self):
        class SkippedFixture(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                raise unittest.SkipTest("GPU is intentionally unavailable")

            def test_never_runs(self):
                pass

        suite = unittest.TestLoader().loadTestsFromTestCase(SkippedFixture)
        self.assertEqual(self.run_suite(suite), 0)

    def test_expected_failure_remains_successful(self):
        class ExpectedFailure(unittest.TestCase):
            @unittest.expectedFailure
            def test_known_failure(self):
                self.fail("expected sentinel")

        suite = unittest.TestLoader().loadTestsFromTestCase(ExpectedFailure)
        self.assertEqual(self.run_suite(suite), 0)

    def test_unexpected_success_fails(self):
        class UnexpectedSuccess(unittest.TestCase):
            @unittest.expectedFailure
            def test_fixed_case(self):
                pass

        suite = unittest.TestLoader().loadTestsFromTestCase(UnexpectedSuccess)
        self.assertEqual(self.run_suite(suite), 1)

    def test_explicit_unittest_module_selection_is_preserved(self):
        suite = unittest.TestSuite([unittest.FunctionTestCase(lambda: None)])
        with mock.patch.object(runner, "load_tests", return_value=suite) as loader:
            self.assertEqual(runner.main(["--module", "tests.one", "--module", "tests.two"]), 0)
        loader.assert_called_once_with("tools/tests", "test_*.py", ["tests.one", "tests.two"])

    def assert_offline_profile(self):
        for name, expected in (runner.OFFLINE_ENVIRONMENT | runner.DISABLED_LIVE_TESTS).items():
            self.assertEqual(os.environ[name], expected, name)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "-1")
        self.assertTrue(sys.dont_write_bytecode)
        self.assertEqual(sys.argv[1:], ["--cpu"])
        self.assertEqual(sys.path[0], str(runner.REPOSITORY_ROOT))
        self.assertTrue(self.isolation_active)

    def test_pytest_receives_policy_before_import_and_collection(self):
        os.environ.update({name: "1" for name in runner.DISABLED_LIVE_TESTS})
        os.environ.update({name: "0" for name in runner.OFFLINE_ENVIRONMENT})
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        received = []

        def pytest_main(arguments):
            self.assert_offline_profile()
            received.append(arguments)
            return 0

        def import_pytest(name):
            self.assertEqual(name, "pytest")
            self.assert_offline_profile()
            return SimpleNamespace(main=pytest_main)

        with (
            mock.patch.object(runner.importlib, "import_module", side_effect=import_pytest),
            mock.patch.object(runner, "load_tests") as unittest_loader,
        ):
            self.assertEqual(runner.main(["--pytest", "tests/yue2", "-q", "-k", "roundtrip"]), 0)
        self.assertEqual(received, [["tests/yue2", "-q", "-k", "roundtrip"]])
        unittest_loader.assert_not_called()
        self.assertFalse(self.isolation_active)

    def test_pytest_preserves_all_exit_codes(self):
        for expected in range(6):
            with self.subTest(exit_code=expected):
                module = SimpleNamespace(main=mock.Mock(return_value=expected))
                with mock.patch.object(runner.importlib, "import_module", return_value=module):
                    self.assertEqual(runner.main(["--pytest", "tests/yue2", "-q"]), expected)
                self.assertFalse(self.isolation_active)

    def test_preload_runs_inside_policy_before_pytest_import(self):
        order = []

        def import_module(name):
            self.assert_offline_profile()
            order.append(name)
            return SimpleNamespace(main=lambda _: 0)

        with mock.patch.object(runner.importlib, "import_module", side_effect=import_module):
            self.assertEqual(runner.main(["--preload", "test_preload", "--pytest", "tests/yue2"]), 0)
        self.assertEqual(order, ["test_preload", "pytest"])

    def test_pytest_exception_still_leaves_record_isolation(self):
        module = SimpleNamespace(main=mock.Mock(side_effect=RuntimeError("pytest sentinel")))
        with (
            mock.patch.object(runner.importlib, "import_module", return_value=module),
            self.assertRaisesRegex(RuntimeError, "pytest sentinel"),
        ):
            runner.main(["--pytest", "tests/yue2"])
        self.assertFalse(self.isolation_active)

    def test_pytest_requires_arguments(self):
        with self.assertRaises(SystemExit) as raised:
            runner.main(["--pytest"])
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("requires", self.stderr.getvalue())
        self.assertFalse(self.isolation_active)

    def test_pytest_rejects_mixed_unittest_module_mode(self):
        with self.assertRaises(SystemExit) as raised:
            runner.main(["--module", "tests.one", "--pytest", "tests/yue2"])
        self.assertEqual(raised.exception.code, 2)
        self.assertFalse(self.isolation_active)


class RecordIsolationTests(unittest.TestCase):
    def check_record_isolation(self, *, fail_inside):
        from modules_forge import minimax_h3_pending as pending

        with TemporaryDirectory(prefix="aikimi-owned-records-") as directory:
            original = Path(directory)
            with mock.patch.object(pending, "DIRECTORY", original):
                record = {"version": 1, "prompt_id": str(uuid.uuid4())}
                pending.write(record)
                saved = {p.name: p.read_bytes() for p in original.iterdir()}
                try:
                    with runner.isolated_h3_records():
                        isolated = pending.DIRECTORY
                        self.assertNotEqual(isolated, original)
                        self.assertEqual(pending.read_all(), [])
                        pending.write({"version": 1, "prompt_id": str(uuid.uuid4())})
                        if fail_inside:
                            raise RuntimeError("isolation sentinel")
                except RuntimeError as exc:
                    if not fail_inside or str(exc) != "isolation sentinel":
                        raise
                self.assertEqual(pending.DIRECTORY, original)
                self.assertFalse(isolated.exists())
                self.assertEqual({p.name: p.read_bytes() for p in original.iterdir()}, saved)
                self.assertEqual(pending.read_all(), [record])

    def test_real_record_store_is_restored_without_touching_original_files(self):
        self.check_record_isolation(fail_inside=False)

    def test_real_record_store_is_restored_after_exception(self):
        self.check_record_isolation(fail_inside=True)


class WorkflowCoverageTests(unittest.TestCase):
    def test_python_contract_steps_share_the_cpu_offline_runner(self):
        source = (ROOT / ".github/workflows/unit-tests.yml").read_text(encoding="utf-8")
        self.assertNotIn("python -m pytest", source)
        self.assertIn("python tools/run_ci_tests.py --pytest\n", source)
        self.assertIn("python tools/run_ci_tests.py --pytest tests/yue2 -q", source)
        self.assertIn("python tools/run_ci_tests.py --pytest tests/jev_sparse -q", source)
        self.assertIn("node --test tools/tests/h3_handoff_core.test.mjs", source)

    def test_qwen_audit_is_independent_resolved_and_strict(self):
        source = (ROOT / ".github/workflows/security.yml").read_text(encoding="utf-8")
        match = re.search(r"(?ms)^  qwen-pip-audit:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)
        self.assertIsNotNone(match, "The separate Qwen environment needs its own audit job")
        job = match.group(1)
        self.assertIn("inputs: tools/requirements-qwen-image21.txt", job)
        self.assertIn("internal-be-careful-extra-flags: --strict", job)
        for bypass in ("no-deps: true", "ignore-vulns:", "continue-on-error:", "needs:"):
            self.assertNotIn(bypass, job)
        actions = re.findall(r"uses:\s+(\S+)", job)
        self.assertTrue(actions)
        for action in actions:
            self.assertRegex(action, r"@[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
