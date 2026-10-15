"""Installer contracts without network traffic, model weights, or GPU imports."""

import hashlib
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from tools import setup_flux3_klein as setup


class Flux3SetupTests(unittest.TestCase):
    def test_cli_defaults_skip_pup_and_allow_explicit_opt_in(self):
        with patch.object(setup, "run", return_value={"ok": True}) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(setup.main([]), 0)
            self.assertIs(run.call_args.kwargs["with_pup"], False)
            self.assertEqual(setup.main(["--without-pup"]), 0)
            self.assertIs(run.call_args.kwargs["with_pup"], False)
            self.assertEqual(setup.main(["--with-pup"]), 0)
            self.assertIs(run.call_args.kwargs["with_pup"], True)

    def test_download_plan_excludes_optional_pup_by_default(self):
        plan = setup.download_plan()
        self.assertEqual(len(plan), 3)
        self.assertFalse(any("upsampler" in entry["path"] or "system_prompts" in entry["path"] for entry in plan))

    def test_new_checkout_checks_out_pin_before_checking_dirty_worktree(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "ComfyUI"
            calls = []

            def command(arguments, **kwargs):
                calls.append([str(part) for part in arguments])
                if "clone" in arguments:
                    (runtime / ".git").mkdir(parents=True)
                return ""

            setup.prepare_comfy_checkout(runtime, command, "git")
        self.assertTrue(any("checkout" in call and setup.core.COMFY_REVISION in call for call in calls))
        self.assertFalse(any("status" in call for call in calls))

    def test_existing_dirty_checkout_is_kept_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary) / "ComfyUI"
            (runtime / ".git").mkdir(parents=True)
            (runtime / ".git/index").write_bytes(b"index")
            command = Mock(return_value=" M main.py")
            with self.assertRaises(setup.Flux3SetupError):
                setup.prepare_comfy_checkout(runtime, command, "git")
            self.assertEqual(command.call_count, 1)
            self.assertIn("status", command.call_args.args[0])

    def test_model_plan_pins_every_download_and_pup_prompts(self):
        plan = setup.download_plan(model="distilled", precision="int8", with_pup=True)
        self.assertEqual(len(plan), 5)
        self.assertTrue(any(entry["path"] == "text_encoders/system_prompts.json" for entry in plan))
        for entry in plan:
            self.assertRegex(entry["revision"], r"^[0-9a-f]{40}$")
            self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(entry["size"], 0)
            self.assertNotIn("token", entry)

    def test_dry_run_does_not_touch_disk_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            with patch.object(setup, "prepare_runtime") as runtime, patch.object(setup, "download_asset") as download:
                result = setup.run(model="base", precision="bf16", with_pup=False, dry_run=True, root=root)
            self.assertTrue(result["dry_run"])
            self.assertEqual(len(result["assets"]), 3)
            self.assertFalse(root.exists())
            runtime.assert_not_called()
            download.assert_not_called()

    def test_runtime_only_never_reports_missing_models_as_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(setup, "prepare_runtime"),
                patch.object(setup, "runtime_ready", return_value=True),
                patch.object(setup.core, "model_ready", return_value=False),
                patch.object(setup, "download_asset") as download,
            ):
                result = setup.run(runtime_only=True, root=Path(temporary))
            self.assertTrue(result["ok"])
            self.assertTrue(result["runtime_ready"])
            self.assertFalse(result["model_ready"])
            download.assert_not_called()

    def test_verified_local_file_never_downloads_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "asset.bin").write_bytes(b"good")
            spec = {
                "repository": "test/model",
                "revision": "a" * 40,
                "filename": "asset.bin",
                "path": "asset.bin",
                "size": 4,
                "sha256": hashlib.sha256(b"good").hexdigest(),
            }
            with patch.object(setup, "_hub_download") as download:
                self.assertEqual(setup.download_asset(root, spec), root.resolve() / "asset.bin")
            download.assert_not_called()

    def test_hub_download_uses_pinned_revision_existing_auth_and_validates_before_publish(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "cached.bin"
            cache.write_bytes(b"good")
            spec = {
                "repository": "test/model",
                "revision": "a" * 40,
                "filename": "remote/asset.bin",
                "path": "dit/asset.bin",
                "size": 4,
                "sha256": hashlib.sha256(b"good").hexdigest(),
            }
            with patch.object(setup, "_hub_download", return_value=str(cache)) as download:
                target = setup.download_asset(root, spec)
            self.assertEqual(target.read_bytes(), b"good")
            self.assertEqual(download.call_args.kwargs["revision"], "a" * 40)
            self.assertIs(download.call_args.kwargs["token"], True)
            self.assertFalse(any("token" in str(path).lower() for path in root.rglob("*")))
            target.unlink()
            cache.write_bytes(b"evil")
            with (
                patch.object(setup, "_hub_download", return_value=str(cache)),
                self.assertRaises(setup.Flux3SetupError),
            ):
                setup.download_asset(root, spec)
            self.assertFalse(target.exists())

    def test_paths_cannot_escape_managed_root(self):
        spec = {
            "repository": "test/model",
            "revision": "a" * 40,
            "filename": "asset.bin",
            "path": "../escaped.bin",
            "size": 1,
            "sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as temporary, self.assertRaises(setup.Flux3SetupError):
            setup.download_asset(Path(temporary), spec)

    def test_auth_failure_is_actionable_and_never_includes_raw_exception(self):
        spec = setup.download_plan()[0]
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(
                setup, "_hub_download", side_effect=RuntimeError("Authorization: Bearer hf_private_token")
            ):
                with self.assertRaises(setup.Flux3SetupError) as caught:
                    setup.download_asset(Path(temporary), spec)
            self.assertNotIn("hf_private_token", str(caught.exception))
            self.assertIn("Hugging Face", str(caught.exception))

    def test_failed_runtime_command_is_redacted_and_does_not_use_shell(self):
        result = Mock(returncode=1, stdout="Authorization: Bearer hf_private_token", stderr="token=abc123")
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "setup.log"
            with patch.object(setup.subprocess, "run", return_value=result) as run:
                with self.assertRaises(setup.Flux3SetupError):
                    setup.runtime_command(["python", "--version"], Path(temporary), {}, log)
            self.assertNotIn("hf_private_token", log.read_text(encoding="utf-8"))
            self.assertFalse(run.call_args.kwargs.get("shell", False))


if __name__ == "__main__":
    unittest.main()
