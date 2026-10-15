from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class CiWorkflowBoundaryTests(unittest.TestCase):
    def test_shared_image_workspace_is_linted_in_ci(self):
        lint = self.workflow("lint.yml")
        for path in (
            "modules_forge/image_workspace.py",
            "tools/tests/test_image_workspace*.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(lint.count(f'"{path}"'), 2)
                self.assertIn(f"            {path}\n", lint)

    def test_krea_text_conditioning_cache_is_checked_in_ci(self):
        lint = self.workflow("lint.yml")
        windows = self.workflow("windows-smoke.yml")
        for path in (
            "tools/tests/test_krea2_text_conditioning_cache.py",
            "tools/tests/test_krea2_reference_conditioning_cache.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(lint.count(f'"{path}"'), 2)
                self.assertIn(f"            {path}\n", lint)
                self.assertEqual(windows.count(f'"{path}"'), 2)
                self.assertIn(f"--module tools.tests.{Path(path).stem}\n", windows)

    def test_anima_reference_canvas_is_checked_in_ci(self):
        lint = self.workflow("lint.yml")
        windows = self.workflow("windows-smoke.yml")
        for path in (
            "tools/tests/test_image_stitch_anima_canvas.py",
            "tools/tests/test_anima_reference_conditioning.py",
        ):
            self.assertEqual(lint.count(f'"{path}"'), 2)
            self.assertIn(f"            {path}\n", lint)
            self.assertEqual(windows.count(f'"{path}"'), 2)
            module = Path(path).stem
            self.assertIn(f"--module tools.tests.{module}\n", windows)
        self.assertEqual(windows.count('"extensions-builtin/sd_forge_image_stitch/**"'), 2)

    def test_orbit_installation_and_ui_are_checked_in_ci(self):
        lint = self.workflow("lint.yml")
        for path in (
            "modules_forge/minimax_h3_orbit_assets.py",
            "tools/prepare_minimax_h3_orbit.py",
            "tools/tests/test_h3_orbit*.py",
            "tools/tests/test_comfy_startup_cleanup.py",
        ):
            with self.subTest(path=path):
                self.assertEqual(lint.count(f'"{path}"'), 2)
                self.assertIn(f"            {path}\n", lint)
        windows = self.workflow("windows-smoke.yml")
        for path in ("tools/tests/test_h3_orbit*.py", "tools/tests/test_comfy_startup_cleanup.py"):
            self.assertEqual(windows.count(f'"{path}"'), 2)
        for module in ("test_h3_orbit", "test_h3_orbit_assets", "test_h3_orbit_ui", "test_comfy_startup_cleanup"):
            self.assertIn(f"--module tools.tests.{module}\n", windows)

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

    def test_flux3_lint_tracks_runtime_studio_installers_and_asset_inputs(self):
        lint = self.workflow("lint.yml")
        for path in (
            "modules_forge/flux3_klein/**",
            "extensions-builtin/flux3-klein-studio/**",
            "tools/*flux3*.py",
            "tools/tests/test_flux3*.py",
            "tools/tests/flux3_layout.test.mjs",
            "tools/flux3_klein_manifest.json",
            "tools/requirements-flux3-klein.*",
        ):
            with self.subTest(trigger=path):
                self.assertEqual(lint.count(f'"{path}"'), 2)
        for path in (
            "modules_forge/flux3_klein",
            "extensions-builtin/flux3-klein-studio/scripts",
            "tools/*flux3*.py",
            "tools/tests/test_flux3*.py",
        ):
            with self.subTest(ruff=path):
                self.assertTrue(f"            {path}\n" in lint, f"Missing FLUX 3 lint target: {path}")

    def test_flux3_layout_is_syntax_checked_and_its_node_contracts_run(self):
        self.assertTrue(
            "node --check extensions-builtin/flux3-klein-studio/javascript/flux3_layout.js"
            in self.workflow("lint.yml"),
            "FLUX 3 layout JavaScript must pass a syntax check",
        )
        self.assertTrue(
            "node --test tools/tests/flux3_layout.test.mjs" in self.workflow("unit-tests.yml"),
            "FLUX 3 layout contracts must run in CI",
        )
        for path in ("tools/flux3_klein_manifest.json", "tools/requirements-flux3-klein.*"):
            with self.subTest(unit_trigger=path):
                self.assertEqual(self.workflow("unit-tests.yml").count(f'"{path}"'), 2)

    def test_flux3_windows_cpu_discovers_current_and_future_feature_contracts(self):
        windows = self.workflow("windows-smoke.yml")
        for path in (
            "modules_forge/flux3_klein/**",
            "extensions-builtin/flux3-klein-studio/**",
            "tools/*flux3*.py",
            "tools/tests/test_flux3_klein*.py",
            "tools/flux3_klein_manifest.json",
            "tools/requirements-flux3-klein.*",
        ):
            with self.subTest(trigger=path):
                self.assertEqual(windows.count(f'"{path}"'), 2)
        self.assertTrue(
            'python tools/run_ci_tests.py --pattern "test_flux3_klein*.py" --verbosity 2' in windows,
            "Windows must discover all FLUX 3 contracts, including future feature tests",
        )

    def test_flux3_lock_rebuild_command_keeps_the_required_cuda_index(self):
        lock = (ROOT / "tools/requirements-flux3-klein.lock").read_text(encoding="utf-8")
        command = lock.splitlines()[1]
        self.assertIn("--index-url https://pypi.org/simple", command)
        self.assertIn("--extra-index-url https://download.pytorch.org/whl/cu130", command)

    def test_flux3_dependency_audit_checks_every_pin_without_platform_resolution(self):
        security = self.workflow("security.yml")
        self.assertTrue("  flux3-pip-audit:" in security, "FLUX 3 needs an independent complete-lock audit")
        job = security.split("  flux3-pip-audit:", 1)[1].split("  gitleaks-full-history:", 1)[0]
        self.assertIn("inputs: tools/requirements-flux3-klein.lock", job)
        self.assertIn("vulnerability-service: OSV", job)
        self.assertIn("internal-be-careful-extra-flags: --strict --no-deps --disable-pip", job)
        for bypass in (
            "ignore-vulns:",
            "--ignore-vuln",
            "needs:",
            "continue-on-error:",
            "internal-be-careful-allow-failure:",
        ):
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
