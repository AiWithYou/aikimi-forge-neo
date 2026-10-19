"""Managed core updates preserve user edits and fail closed after interruption."""

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_runtime as runtime_paths
from modules_forge import minimax_h3_union2_vae as union2
from tools import setup_minimax_h3 as setup
from tools.aikimi_setup import SetupError


class CoreUpdateTests(unittest.TestCase):
    def test_normal_generation_accepts_the_bundled_orbit_pack_but_rejects_other_extras(self):
        from modules_forge.minimax_h3_acceleration import H3Acceleration

        with tempfile.TemporaryDirectory() as folder:
            option = H3Acceleration()
            arguments = bridge._runtime_command(Path("python"), 8189, "low_ram", option,
                                                trusted_custom_nodes=(bridge.orbit_assets.PACK,))[1:]
            ready = bridge.RuntimeReadiness(runtime_root=Path(folder), server_url="http://127.0.0.1:8189",
                connected=True, ck_attention_available=True, h3_core_optimized=True,
                package_versions={"comfy-kitchen": "0.2.39"}, runtime_profile="low_ram",
                runtime_args=tuple(arguments), acceleration=option,
                model_files=dict.fromkeys(bridge.MODEL_FILES, True),
                server_model_files=dict.fromkeys(bridge.MODEL_FILES, True))
            bridge.validate_readiness(ready, "low_ram")
            for extra in (bridge.FAST_VAE_PACK, "user-custom-pack"):
                with self.assertRaisesRegex(bridge.H3BridgeError, "許可設定"):
                    bridge.validate_readiness(replace(ready, runtime_args=tuple(arguments + [extra])), "low_ram")

    def test_lightvae_cli_runs_by_filename_without_pythonpath(self):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(  # noqa: S603 -- Fixed interpreter and repository CLI, no user input.
            [sys.executable, "-E", "-X", "utf8", str(root / "tools/setup_h3_lightvae.py"), "--help"],
            cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--variant", result.stdout)

    def test_current_setup_ignores_obsolete_receipts_but_both_paths_reject_new_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            runtime = runtime_paths.managed_runtime_root(root)
            runtime.mkdir(parents=True)
            (runtime / "main.py").touch()
            (runtime / "extra_model_paths.yaml").write_text(runtime_paths.model_config(root / "models"), encoding="utf-8")
            python = runtime.parent / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            for name in ("tools/minimax_h3_runtime_manifest.json", "tools/requirements-minimax-h3.lock",
                         "patches/minimax-h3/comfyui-b9b290e4-compiler.patch"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"verified fixture")
            (runtime.parent / "setup.json").write_text(json.dumps({"schema_version": 1,
                "fingerprint": runtime_paths.setup_fingerprint(root)}), encoding="utf-8")
            ready = SimpleNamespace(runtime_root=runtime, connected=True, runtime_args=(),
                                    package_versions={"comfy-kitchen": "0.2.39", "comfy-aimdo": "0.5.6"})
            for obsolete in ("union2-vae-update-in-progress.json", "union2-vae-runtime.json"):
                record = runtime.parent / obsolete
                record.write_text(json.dumps({"schema_version": 1, "revision": union2.VAE_FIXED_COMMIT}), encoding="utf-8")
                with self.subTest(obsolete=obsolete):
                    self.assertEqual(bridge._python_for_runtime(runtime), python)
                    union2.check_runtime(ready, ancestor=lambda commit: True)
                record.unlink()
            (runtime.parent / "runtime-update-in-progress.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(bridge.H3BridgeError, "更新が未完了"):
                bridge._python_for_runtime(runtime)
            with self.assertRaisesRegex(ValueError, "更新が未完了"):
                union2.check_runtime(ready, ancestor=lambda commit: True)

    def test_known_requirement_change_is_removed_only_after_all_files_validate(self):
        with tempfile.TemporaryDirectory() as folder:
            installer = setup.RuntimeInstaller(Path(folder))
            installer.runtime.mkdir(parents=True)
            (installer.runtime / "requirements.txt").write_text("comfy-aimdo==0.5.3\n", encoding="utf-8")
            (installer.runtime / "custom.py").write_text("user work\n", encoding="utf-8")
            responses = ["requirements.txt\ncustom.py\n", "comfy-aimdo==0.5.2\n", "original\n"]
            installer.run = Mock(side_effect=[SimpleNamespace(stdout=value) for value in responses])
            with self.assertRaisesRegex(SetupError, "管理対象外"):
                installer.prepare_core_update("efa6c8f804bff78b46a0fd458ebd2e47bba07a30")
            self.assertFalse(any("restore" in call.args[0] for call in installer.run.call_args_list))
            self.assertEqual((installer.runtime / "custom.py").read_text("utf-8"), "user work\n")

    def test_custom_requirement_change_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            installer = setup.RuntimeInstaller(Path(folder))
            installer.runtime.mkdir(parents=True)
            (installer.runtime / "requirements.txt").write_text("my-package==1\n", encoding="utf-8")
            installer.run = Mock(side_effect=[SimpleNamespace(stdout=value) for value in
                                             ("requirements.txt\n", "comfy-aimdo==0.5.2\n")])
            with self.assertRaisesRegex(SetupError, "管理対象外"):
                installer.prepare_core_update("efa6c8f804bff78b46a0fd458ebd2e47bba07a30")
            self.assertEqual(installer.run.call_count, 2)

    def test_pending_update_blocks_old_python_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder) / "repositories/minimax-h3/ComfyUI"
            runtime.mkdir(parents=True)
            python = runtime.parent / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            (runtime.parent / "runtime-update-in-progress.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(bridge.H3BridgeError, "更新が未完了"):
                bridge._python_for_runtime(runtime)


if __name__ == "__main__":
    unittest.main()
