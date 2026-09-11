from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PWSH = shutil.which("pwsh")


@unittest.skipUnless(os.name == "nt" and PWSH, "Windows PowerShell 7 test")
class ModelSetupLauncherTests(unittest.TestCase):
    def test_h3_download_batch_returns_from_python_cmd_launcher(self):
        with tempfile.TemporaryDirectory(prefix="python launcher ") as directory:
            launcher = Path(directory) / "python.cmd"
            launcher.write_text(f'@"{sys.executable}" %*\n', encoding="utf-8")
            environment = dict(os.environ, PATH=directory + os.pathsep + os.environ["PATH"])
            result = subprocess.run(  # noqa: S603 - 固定BATをダウンロードなしで実行する。
                [os.environ["COMSPEC"], "/d", "/c", str(ROOT / "download_minimax_h3_models.bat"), "--dry-run"],
                input=b"\r\n",
                capture_output=True,
                env=environment,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(b"model_download_bytes", result.stdout)

    def run_ps(self, code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(  # noqa: S603 - 固定のテストスクリプトだけを実行する。
            [PWSH, "-NoProfile", "-Command", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )

    def test_all_profiles_route_and_restore_environment(self):
        with tempfile.TemporaryDirectory(prefix="Neo setup 日本語 ") as directory:
            root = Path(directory)
            for relative in (
                "launch.py",
                "tools/aikimi_setup.py",
                "tools/setup_minimax_h3.py",
                "download_sensenova_u15_int8.ps1",
            ):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "requirements.txt").write_text("starlette==1.0.0\n", encoding="utf-8")
            for model in ("krea2", "anima38", "sensenova", "h3"):
                for fail in (False, True):
                    with self.subTest(model=model, fail=fail):
                        result = self.run_ps(f"""
$ErrorActionPreference = 'Stop'
. '{ROOT / "aikimi-setup.ps1"}'
$script:calls = [Collections.Generic.List[object]]::new()
function Assert-SetupIdle {{}}
function Get-Command {{ [pscustomobject]@{{ Source='mock.exe' }} }}
function Invoke-SetupCommand {{
    param([string]$Executable, [string[]]$CommandArguments)
    $script:calls.Add(@($CommandArguments))
    if (${str(fail).lower()} -and $script:calls.Count -eq 3) {{ throw 'injected failure' }}
}}
$env:COMMANDLINE_ARGS = '--original-option'
$before = (Get-Location).Path
$caught = $false
try {{
    Invoke-AikimiModelSetup -RepositoryRoot '{root}' -SelectedModel '{model}' {("-PreserveSource" if model == "anima38" else "")}
}} catch {{ $caught = $true }}
@{{ calls=$script:calls.ToArray(); caught=$caught; arguments=$env:COMMANDLINE_ARGS; locationRestored=((Get-Location).Path -eq $before) }} | ConvertTo-Json -Depth 8 -Compress
""")
                        self.assertEqual(result.returncode, 0, result.stderr)
                        data = json.loads(result.stdout.strip().splitlines()[-1])
                        self.assertEqual(data["caught"], fail)
                        self.assertEqual(data["arguments"], "--original-option")
                        self.assertTrue(data["locationRestored"])
                        calls = data["calls"]
                        self.assertEqual(calls[0][-1], str(root / "venv"))
                        if fail:
                            self.assertEqual(len(calls), 3)
                            continue
                        self.assertEqual(calls[3][-3:], ["--exit", "--uv", "--bnb"])
                        if model == "h3":
                            self.assertEqual(calls[4], ["-B", str(root / "tools/setup_minimax_h3.py")])
                        else:
                            self.assertEqual(calls[4][2:4], ["install", model])
                        if model == "anima38":
                            self.assertEqual(calls[4][-1], "--keep-source")
                        self.assertEqual(len(calls), 6 if model == "sensenova" else 5)
                        if model == "sensenova":
                            self.assertEqual(calls[-1][-1], "-RuntimeOnly")

    def test_bat_dry_run_each_profile(self):
        for model in ("krea2", "anima38", "sensenova", "h3"):
            with self.subTest(model=model):
                result = self.run_ps(
                    f"& '{ROOT / 'aikimi-setup.bat'}' -Model {model} -DryRun -NoPause; exit $LASTEXITCODE"
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(model, result.stdout)
                self.assertIn("実行予定", result.stdout)

    def test_external_command_failure_is_not_ignored(self):
        result = self.run_ps(f"""
. '{ROOT / "aikimi-setup.ps1"}'
try {{ Invoke-SetupCommand '{PWSH}' @('-NoProfile', '-Command', 'exit 23') }}
catch {{ Write-Output $_.Exception.Message; exit 9 }}
exit 0
""")
        self.assertEqual(result.returncode, 9)
        self.assertIn("23", result.stdout)


if __name__ == "__main__":
    unittest.main()
