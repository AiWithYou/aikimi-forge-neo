from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CiWorkflowBoundaryTests(unittest.TestCase):
    def test_clef_and_qwen_release_code_is_linted_and_clipboard_is_tested(self):
        lint = self.workflow("lint.yml")
        paths = (
            "modules_forge/clef/**",
            "extensions-builtin/clef-studio/**",
            "tools/*clef*.py",
            "tools/tests/test_clef*.py",
            "tools/prepare_qwen21_consistency_lora.py",
            "tools/tests/test_qwen21_consistency*.py",
            "tools/tests/test_qwen_int8_convrot.py",
            "tools/tests/test_qwen_gguf_normalization.py",
            "tools/tests/test_resident_worker_logging.py",
        )
        for path in paths:
            with self.subTest(path=path):
                self.assertEqual(lint.count(f'"{path}"'), 2)
        for path in (
            "modules_forge/clef",
            "extensions-builtin/clef-studio/scripts",
            "tools/*clef*.py",
            "tools/tests/test_clef*.py",
            "tools/prepare_qwen21_consistency_lora.py",
            "tools/tests/test_qwen21_consistency*.py",
            "tools/tests/test_qwen_int8_convrot.py",
            "tools/tests/test_qwen_gguf_normalization.py",
            "tools/tests/test_resident_worker_logging.py",
        ):
            with self.subTest(linted=path):
                self.assertTrue(f"            {path}\n" in lint, f"Missing lint target: {path}")
        self.assertTrue(
            "node --test tools/tests/clef_clipboard.test.mjs" in self.workflow("unit-tests.yml"),
            "Clef clipboard JavaScript tests must run in CI",
        )
        self.assertTrue(
            "python tools/run_ci_tests.py --pytest tools/tests/test_clef*.py -q" in self.workflow("unit-tests.yml"),
            "Clef pytest functions must run independently of unittest discovery",
        )
        self.assertEqual(self.workflow("unit-tests.yml").count('"**/*.mjs"'), 2)

    def test_clef_dependency_audit_runs_independently_without_ignores(self):
        security = self.workflow("security.yml")
        self.assertTrue("  clef-pip-audit:" in security, "Clef needs its own runtime audit")
        job = security.split("  clef-pip-audit:", 1)[1].split("  gitleaks-full-history:", 1)[0]
        self.assertIn("inputs: tools/requirements-clef.txt", job)
        self.assertIn("vulnerability-service: OSV", job)
        self.assertIn("internal-be-careful-extra-flags: --strict", job)
        self.assertIn("extra-index-urls: https://download.pytorch.org/whl/cu130", job)
        for bypass in ("ignore-vulns:", "no-deps:", "needs:", "continue-on-error:"):
            self.assertNotIn(bypass, job)

    def workflow(self, name: str) -> str:
        return (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8")

    def test_asset_job_tracks_its_dependency_light_import_closure(self):
        workflow = self.workflow("aikimi-assets.yml")

        self.assertEqual(workflow.count('"modules/aikimi_status.py"'), 2)
        self.assertEqual(workflow.count('"modules/aikimi_security/**"'), 2)

    def test_windows_job_tracks_and_runs_its_runner_and_chromium_smokes(self):
        workflow = self.workflow("windows-smoke.yml")
        tracked_twice = (
            "backend/**",
            "launch.py",
            "modules_forge/**",
            "modules/call_queue.py",
            "modules_forge/forge_canvas/**",
            "script.js",
            "tools/run_ci_tests.py",
            "tools/tests/chromium_helpers.py",
            "tools/tests/test_chromium_helpers.py",
            "tools/tests/test_ci_workflow_boundaries.py",
            "tools/tests/test_gpu_ownership.py",
            "tools/tests/test_minimax_h3_submission.py",
            "tools/tests/test_sensenova_worker_cleanup.py",
            "tools/tests/test_sensenova_u15_bridge.py",
            "tools/tests/test_additional_module_identity.py",
            "tools/tests/test_extra_networks_lora_filter.py",
            "tools/tests/test_gradio_chromium_smoke.py",
            "tools/tests/test_option_migrations.py",
            "tools/tests/test_patcher_cleanup.py",
            "tools/tests/test_quant_ops_zero_scale.py",
            "tools/tests/test_repository_privacy.py",
            "tools/tests/test_run_ci_tests.py",
            "tools/tests/test_upstream_neo_regressions.py",
        )
        for path in tracked_twice:
            with self.subTest(path=path):
                self.assertEqual(sum(line.strip() == f'- "{path}"' for line in workflow.splitlines()), 2)

        for module in (
            "tools.tests.test_additional_module_identity",
            "tools.tests.test_chromium_helpers",
            "tools.tests.test_ci_workflow_boundaries",
            "tools.tests.test_gpu_ownership",
            "tools.tests.test_minimax_h3_submission",
            "tools.tests.test_sensenova_worker_cleanup",
            "tools.tests.test_sensenova_u15_bridge",
            "tools.tests.test_extra_networks_lora_filter",
            "tools.tests.test_gradio_chromium_smoke",
            "tools.tests.test_option_migrations",
            "tools.tests.test_patcher_cleanup",
            "tools.tests.test_quant_ops_zero_scale",
            "tools.tests.test_repository_privacy",
            "tools.tests.test_run_ci_tests",
            "tools.tests.test_upstream_neo_regressions",
        ):
            with self.subTest(module=module):
                self.assertEqual(sum(line.strip() == f"--module {module}" for line in workflow.splitlines()), 1)
        self.assertIn("Start the real UI and run the full Chromium smoke", workflow)
        self.assertIn("Chrome or Chromium is required for the Windows smoke tests.", workflow)
        self.assertIn("GradioFullUiChromiumTests", workflow)
        self.assertIn("taskkill.exe", workflow)
        self.assertIn("The full UI process tree did not exit after taskkill.", workflow)
        self.assertIn("$global:LASTEXITCODE = 0", workflow)
        self.assertIn('$env:SD_WEBUI_RESTARTING = "1"', workflow)
        self.assertIn("The full UI process did not release port", workflow)

    def test_lint_job_tracks_and_checks_new_python_boundaries(self):
        workflow = self.workflow("lint.yml")

        for path in (
            "modules/gradio_file_url.py",
            "modules/gradio_runtime.py",
            "tools/tests/chromium_helpers.py",
            "tools/tests/test_additional_module_identity.py",
            "tools/tests/test_chromium_helpers.py",
            "tools/tests/test_ci_workflow_boundaries.py",
            "tools/tests/test_option_migrations.py",
            "tools/tests/test_patcher_cleanup.py",
            "tools/tests/test_quant_ops_zero_scale.py",
            "tools/tests/test_repository_privacy.py",
        ):
            with self.subTest(path=path):
                self.assertGreaterEqual(workflow.count(path), 3)
        self.assertIn("python -m unittest -v tools.tests.test_ci_workflow_boundaries", workflow)

    def test_security_and_dependabot_cover_the_separate_preprocessor_manifest(self):
        security = self.workflow("security.yml")
        dependabot = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        manifest = "extensions-builtin/forge_legacy_preprocessors/requirements.txt"

        self.assertIn(f"inputs: {manifest}", security)
        self.assertIn('directory: "/extensions-builtin/forge_legacy_preprocessors"', dependabot)

    def test_runtime_and_asset_toolchain_share_the_pinned_scipy_boundary(self):
        runtime = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        assets = (ROOT / "tools" / "requirements-aikimi-assets.txt").read_text(encoding="utf-8").splitlines()

        self.assertIn("scipy==1.18.0", runtime)
        self.assertIn("scipy==1.18.0", assets)

    def test_sensenova_dependency_audit_runs_independently_without_ignores(self):
        security = self.workflow("security.yml")
        job = security.split("  sensenova-pip-audit:", 1)[1].split("  gitleaks-full-history:", 1)[0]
        self.assertIn("inputs: tools/requirements-sensenova.txt", job)
        self.assertIn("extra-index-urls: https://download.pytorch.org/whl/cu130", job)
        for bypass in ("ignore-vulns:", "no-deps:", "needs:", "continue-on-error:"):
            self.assertNotIn(bypass, job)


if __name__ == "__main__":
    unittest.main()
