from __future__ import annotations

import subprocess
import unittest
from unittest import mock

from modules_forge.uv_hook import patch, rewrite_pip_to_uv


class RewritePipToUvTests(unittest.TestCase):
    def test_python_module_pip_preserves_interpreter(self):
        rewritten = rewrite_pip_to_uv(
            [
                r"C:\repo\venv\Scripts\python.exe",
                "-m",
                "pip",
                "install",
                "torch==2.13.0+cu130",
            ]
        )
        self.assertEqual(
            rewritten,
            [
                "uv",
                "pip",
                "install",
                "--python",
                r"C:\repo\venv\Scripts\python.exe",
                "torch==2.13.0+cu130",
            ],
        )

    def test_existing_python_option_is_not_duplicated(self):
        rewritten = rewrite_pip_to_uv(
            [
                r"C:\repo\venv\Scripts\python.exe",
                "-m",
                "pip",
                "install",
                "--python",
                r"C:\other\python.exe",
                "starlette==1.6.0",
            ]
        )
        self.assertEqual(rewritten.count("--python"), 1)
        self.assertEqual(rewritten[-1], "starlette==1.6.0")

    def test_bare_pip_command_keeps_no_interpreter(self):
        rewritten = rewrite_pip_to_uv(["pip", "install", "packaging"])
        self.assertEqual(rewritten, ["uv", "pip", "install", "packaging"])

    def test_bad_flags_are_stripped(self):
        rewritten = rewrite_pip_to_uv(
            [
                r"C:\repo\venv\Scripts\python.exe",
                "-m",
                "pip",
                "install",
                "--prefer-binary",
                "-I",
                "rich",
            ]
        )
        self.assertNotIn("--prefer-binary", rewritten)
        self.assertNotIn("-I", rewritten)
        self.assertIn("--python", rewritten)

    def test_symlink_mode_is_appended(self):
        rewritten = rewrite_pip_to_uv(
            [
                r"C:\repo\venv\Scripts\python.exe",
                "-m",
                "pip",
                "install",
                "tqdm",
            ],
            symlink=True,
        )
        self.assertEqual(rewritten[-2:], ["--link-mode", "symlink"])

    def test_non_pip_command_is_ignored(self):
        self.assertIsNone(rewrite_pip_to_uv([r"C:\repo\venv\Scripts\python.exe", "launch.py", "--exit"]))

    def test_pip_module_alias_preserves_interpreter(self):
        rewritten = rewrite_pip_to_uv(
            [
                "python",
                "-m",
                "pip.__main__",
                "check",
            ]
        )
        self.assertEqual(rewritten, ["uv", "pip", "check", "--python", "python"])


class PatchedRunTests(unittest.TestCase):
    def setUp(self):
        self._real_run = subprocess.run
        uv_check = mock.patch("modules_forge.uv_hook._pre_check")
        uv_check.start()
        self.addCleanup(uv_check.stop)

    def tearDown(self):
        subprocess.run = self._real_run
        if hasattr(subprocess, "__original_run"):
            delattr(subprocess, "__original_run")

    def test_patch_routes_pip_through_uv_with_interpreter(self):
        patch(symlink=False, local=False)
        with mock.patch.object(subprocess, "__original_run", return_value="ok") as original_run:
            result = subprocess.run(
                [r"C:\repo\venv\Scripts\python.exe", "-m", "pip", "install", "starlette==1.6.0"],
                capture_output=True,
            )
        self.assertEqual(result, "ok")
        original_run.assert_called_once()
        rewritten = original_run.call_args.args[0]
        self.assertEqual(rewritten[:3], ["uv", "pip", "install"])
        self.assertIn("--python", rewritten)
        self.assertEqual(rewritten[rewritten.index("--python") + 1], r"C:\repo\venv\Scripts\python.exe")

    def test_patch_leaves_non_pip_commands_alone(self):
        patch(symlink=False, local=False)
        command = [r"C:\repo\venv\Scripts\python.exe", "launch.py", "--exit"]
        with mock.patch.object(subprocess, "__original_run", return_value="ok") as original_run:
            subprocess.run(command, capture_output=True)  # noqa: S603 - モック済みの固定コマンド
        original_run.assert_called_once()
        self.assertEqual(original_run.call_args.args[0], command)


if __name__ == "__main__":
    unittest.main()
