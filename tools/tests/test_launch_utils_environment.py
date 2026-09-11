import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from modules import launch_utils


class RequirementsMetTests(unittest.TestCase):
    def _check(self, content, installed):
        with tempfile.TemporaryDirectory() as temp_dir:
            requirements = Path(temp_dir) / "requirements.txt"
            requirements.write_text(content, encoding="utf-8")

            def installed_version(name):
                if name not in installed:
                    from importlib.metadata import PackageNotFoundError

                    raise PackageNotFoundError(name)
                return installed[name]

            with patch("importlib.metadata.version", side_effect=installed_version):
                return launch_utils.requirements_met(requirements)

    def test_exact_pin_must_match_instead_of_accepting_a_newer_version(self):
        self.assertTrue(self._check("gradio==6.17.3\n", {"gradio": "6.17.3"}))
        self.assertFalse(self._check("gradio==6.17.3\n", {"gradio": "6.18.0"}))

    def test_specifiers_markers_and_bare_requirements_are_evaluated(self):
        requirements = """
        package-a>=2.0 # supported range
        ignored-package==1; python_version < "1.0"
        torch
        """
        self.assertTrue(self._check(requirements, {"package-a": "2.5", "torch": "2.11.0+cu130"}))
        self.assertFalse(self._check(requirements, {"package-a": "2.5"}))

    def test_invalid_requirement_fails_closed(self):
        self.assertFalse(self._check("not a valid requirement !!!\n", {}))


class SafeSubprocessRunnerTests(unittest.TestCase):
    def test_run_uses_argument_list_without_a_shell_and_fixed_cwd(self):
        completed = SimpleNamespace(returncode=0, stdout="ok", stderr="")
        runner = MagicMock(return_value=completed)

        with patch.object(launch_utils.subprocess, "run", runner):
            result = launch_utils.run(
                [launch_utils.python, "-c", "print('safe')"],
                live=False,
            )

        self.assertEqual(result, "ok")
        call = runner.call_args.kwargs
        self.assertIs(call["shell"], False)
        self.assertEqual(call["cwd"], launch_utils.script_path)
        self.assertIsInstance(call["args"], list)

    def test_shell_metacharacters_remain_plain_arguments(self):
        tokens = launch_utils._split_command("install package;echo-not-executed")
        self.assertIn("package;echo-not-executed", tokens)


class PrepareEnvironmentTests(unittest.TestCase):
    @staticmethod
    def _args(**overrides):
        values = {
            "skip_python_version_check": True,
            "reinstall_torch": False,
            "skip_torch_cuda_test": True,
            "xformers": False,
            "reinstall_xformers": False,
            "sage": False,
            "flash": False,
            "nunchaku": False,
            "bnb": False,
            "ngrok": None,
            "onnxruntime_gpu": False,
            "skip_install": True,
            "update_all_extensions": False,
            "ui_settings_file": "config.json",
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_default_packaging_and_bnb_packages_are_preserved(self):
        installed = []

        def is_installed(package):
            return package not in {"packaging", "bitsandbytes"}

        with (
            patch.object(launch_utils, "args", self._args(bnb=True)),
            patch.object(launch_utils, "is_installed", side_effect=is_installed),
            patch.object(launch_utils, "requirements_met", return_value=True),
            patch.object(launch_utils, "_torch_version", return_value=("2.11.0", "cu130")),
            patch.object(launch_utils, "run_pip", side_effect=lambda command, _label: installed.append(command)),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os.path, "isfile", return_value=True),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("PACKAGING_PACKAGE", None)
            os.environ.pop("BNB_PACKAGE", None)
            launch_utils.prepare_environment()

        self.assertIn("install packaging==26.2", installed)
        self.assertIn("install bitsandbytes==0.49.2", installed)

    def test_gpu_probe_reports_a_generic_compute_device_error(self):
        with (
            patch.object(launch_utils, "args", self._args(skip_torch_cuda_test=False)),
            patch.object(launch_utils, "is_installed", return_value=True),
            patch.object(launch_utils, "check_run_python", return_value=(False, "probe failed")),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
        ):
            with self.assertRaisesRegex(RuntimeError, "any compute device"):
                launch_utils.prepare_environment()

    def test_old_driver_probe_keeps_the_manual_pytorch_fallback(self):
        with (
            patch.object(launch_utils, "args", self._args(skip_torch_cuda_test=False)),
            patch.object(launch_utils, "is_installed", return_value=True),
            patch.object(launch_utils, "check_run_python", return_value=(False, "older driver")),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
        ):
            with self.assertRaisesRegex(SystemError, "manually install older version of PyTorch"):
                launch_utils.prepare_environment()


if __name__ == "__main__":
    unittest.main()
