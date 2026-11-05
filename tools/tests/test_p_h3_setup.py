"""P-H3 and video H3 share a verified runtime without replacing the old one."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge import minimax_h3_runtime as runtime


class SharedRuntimeTransitionTests(unittest.TestCase):
    def test_new_runtime_activates_only_after_complete_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "repositories/minimax-h3/ComfyUI"
            legacy.mkdir(parents=True)
            (legacy / "main.py").write_text("old core", encoding="utf-8")
            models = root / "existing-shared-models"
            models.mkdir()
            (legacy / "extra_model_paths.yaml").write_text(runtime.model_config(models), encoding="utf-8")
            self.assertEqual(runtime.managed_runtime_root(root), legacy)
            self.assertEqual(runtime.configured_model_root(root), models)

            candidate = runtime.candidate_runtime_root(root)
            candidate.mkdir(parents=True)
            (candidate / "main.py").write_text("new core", encoding="utf-8")
            (candidate / "extra_model_paths.yaml").write_text(runtime.model_config(models), encoding="utf-8")
            python = candidate.parent / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.touch()
            self.assertEqual(runtime.managed_runtime_root(root), legacy)
            with patch.object(runtime, "setup_fingerprint", return_value="verified"):
                record = candidate.parent / "setup.json"
                record.write_text(json.dumps({"schema_version": 1, "fingerprint": "verified"}), encoding="utf-8")
                pending = candidate.parent / "runtime-update-in-progress.json"
                pending.write_text("{}", encoding="utf-8")
                self.assertEqual(runtime.managed_runtime_root(root), legacy)
                pending.unlink()
                self.assertEqual(runtime.managed_runtime_root(root), candidate)
                self.assertEqual(runtime.installed_runtime_root(root), candidate)
            self.assertEqual((legacy / "main.py").read_text(encoding="utf-8"), "old core")
            self.assertEqual(runtime.model_root(candidate), models)

    def test_old_and_new_runtime_share_the_setup_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "repositories/minimax-h3/ComfyUI"
            new = runtime.candidate_runtime_root(root)
            with runtime.setup_lock(old), self.assertRaises(ValueError):
                with runtime.setup_lock(new):
                    self.fail("a second setup must not enter")


class PH3SetupTests(unittest.TestCase):
    def test_profile_installs_only_four_official_p_h3_files(self):
        from modules_forge import p_h3_image
        from tools import setup_p_h3_image as setup

        profile = setup.model_profile()
        names = {Path(item.relative_path).name for item in profile.artifacts}
        self.assertEqual(names, {p_h3_image.MODEL, p_h3_image.TEXT_ENCODER, p_h3_image.VAE, p_h3_image.UPSAMPLER})
        self.assertTrue(all("44a0fb73f299da1bb715bdb5ff59ce3fd3c6f543" in item.url for item in profile.artifacts))
        self.assertTrue(all(item.size > 0 and len(item.sha256) == 64 for item in profile.artifacts))
        self.assertEqual(
            {item.relative_path for item in profile.artifacts},
            {f"diffusion_models/{p_h3_image.MODEL}", f"text_encoders/{p_h3_image.TEXT_ENCODER}",
             f"vae/{p_h3_image.VAE}", f"text_encoders/{p_h3_image.UPSAMPLER}"},
        )

    def test_p_h3_setup_selects_its_profile_without_video_weights(self):
        from tools import setup_p_h3_image as setup

        with (
            patch.object(setup, "RuntimeInstaller") as installer,
            patch.object(setup.sys, "argv", ["setup_p_h3_image.py"]),
        ):
            self.assertEqual(setup.main(), 0)
        args = installer.return_value.install.call_args.kwargs
        self.assertIs(args["profile"], setup.model_profile)
        self.assertIs(args["verify"], setup.verify_models)

    def test_existing_external_model_root_is_read_only(self):
        from tools import setup_minimax_h3, setup_p_h3_image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "neo"
            external = Path(folder) / "external"
            external.mkdir()
            legacy = root / "repositories/minimax-h3/ComfyUI"
            legacy.mkdir(parents=True)
            (legacy / "main.py").touch()
            (legacy / "extra_model_paths.yaml").write_text(runtime.model_config(external), encoding="utf-8")
            plan = setup_minimax_h3.plan(root, profile=setup_p_h3_image.model_profile())
            self.assertTrue(plan["shared"])
            self.assertEqual(plan["model_download_bytes"], 0)
            self.assertEqual(Path(plan["models"]), external)

    def test_shared_model_verification_never_repairs_or_downloads(self):
        from tools import setup_p_h3_image
        from tools.aikimi_setup import ArtifactSpec, ProfileSpec, SetupError

        artifact = ArtifactSpec("p-h3", "diffusion_models/a.bin", "https://example.com/a.bin", 3,
                                hashlib.sha256(b"abc").hexdigest(), "https://example.com/LICENSE")
        profile = ProfileSpec("p_h3_image", "fixture", (artifact,), (artifact.license_url,), 3)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / artifact.relative_path
            path.parent.mkdir()
            path.write_bytes(b"bad")
            with patch.object(setup_p_h3_image, "model_profile", return_value=profile), \
                    patch.object(setup_p_h3_image.Installer, "_download") as download:
                with self.assertRaises(SetupError):
                    setup_p_h3_image.verify_models(root, shared=True)
                download.assert_not_called()
            self.assertEqual(path.read_bytes(), b"bad")

    def test_failed_new_runtime_leaves_old_environment_in_place(self):
        from tools import setup_minimax_h3, setup_p_h3_image
        from tools.aikimi_setup import SetupError

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "repositories/minimax-h3/ComfyUI"
            old.mkdir(parents=True)
            (old / "main.py").write_text("old core", encoding="utf-8")
            installer = setup_minimax_h3.RuntimeInstaller(root)
            with (
                patch.object(setup_minimax_h3.sys, "platform", "win32"),
                patch.object(setup_minimax_h3.shutil, "which", return_value="git"),
                patch.object(setup_minimax_h3.shutil, "disk_usage") as disk,
                patch.object(setup_minimax_h3.socket, "socket") as socket,
                patch.object(installer, "install_core", side_effect=SetupError("interrupted")),
            ):
                disk.return_value.free = 100 * 1024**3
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                with self.assertRaisesRegex(SetupError, "interrupted"):
                    installer.install(profile=setup_p_h3_image.model_profile, runtime_only=True)
            self.assertEqual(runtime.managed_runtime_root(root), old)
            self.assertEqual((old / "main.py").read_text(encoding="utf-8"), "old core")
            self.assertTrue((installer.base / "runtime-update-in-progress.json").is_file())

    def test_existing_runtime_updates_a_new_verified_shared_model_root(self):
        from tools import setup_minimax_h3, setup_p_h3_image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            installer = setup_minimax_h3.RuntimeInstaller(root)
            installer.runtime.mkdir(parents=True)
            old, new = root / "old-models", root / "new-models"
            old.mkdir()
            new.mkdir()
            (installer.runtime / "extra_model_paths.yaml").write_text(runtime.model_config(old), encoding="utf-8")
            (installer.base / "setup.json").write_text('{"shared_models":false}', encoding="utf-8")
            with (
                patch.object(setup_minimax_h3.sys, "platform", "win32"),
                patch.object(setup_minimax_h3.shutil, "which", return_value="git"),
                patch.object(setup_minimax_h3.shutil, "disk_usage") as disk,
                patch.object(setup_minimax_h3.socket, "socket") as socket,
                patch.object(setup_minimax_h3, "installed_runtime_root", return_value=installer.runtime),
                patch.object(setup_minimax_h3, "setup_fingerprint", return_value="verified"),
                patch.object(setup_p_h3_image, "verify_models") as verify,
            ):
                disk.return_value.free = 100 * 1024**3
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                installer.install(new, profile=setup_p_h3_image.model_profile, verify=verify)
                verify.assert_called_once_with(new, shared=True)
            self.assertEqual(runtime.model_root(installer.runtime), new)
            self.assertTrue(json.loads((installer.base / "setup.json").read_text(encoding="utf-8"))["shared_models"])

    def test_existing_shared_folder_inside_repository_remains_read_only(self):
        from tools import setup_minimax_h3, setup_p_h3_image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            legacy = root / "repositories/minimax-h3/ComfyUI"
            legacy.mkdir(parents=True)
            (legacy / "main.py").touch()
            shared = root / "existing-shared"
            shared.mkdir()
            (legacy / "extra_model_paths.yaml").write_text(runtime.model_config(shared), encoding="utf-8")
            (legacy.parent / "setup.json").write_text('{"shared_models":true}', encoding="utf-8")
            plan = setup_minimax_h3.plan(root, profile=setup_p_h3_image.model_profile())
            self.assertTrue(plan["shared"])
            self.assertEqual(plan["model_download_bytes"], 0)

    def test_runtime_only_never_downloads_or_verifies_models(self):
        from tools import setup_minimax_h3, setup_p_h3_image

        with tempfile.TemporaryDirectory() as folder:
            installer = setup_minimax_h3.RuntimeInstaller(Path(folder))
            with (
                patch.object(setup_minimax_h3.sys, "platform", "win32"),
                patch.object(setup_minimax_h3.shutil, "which", return_value="git"),
                patch.object(setup_minimax_h3.shutil, "disk_usage") as disk,
                patch.object(setup_minimax_h3.socket, "socket") as socket,
                patch.object(setup_minimax_h3, "installed_runtime_root", return_value=None),
                patch.object(installer, "install_core"),
                patch.object(installer, "bootstrap"),
                patch.object(installer, "install_python"),
                patch.object(installer, "install_nodes"),
                patch.object(installer, "complete_setup"),
                patch.object(setup_p_h3_image, "verify_models") as verify,
            ):
                disk.return_value.free = 100 * 1024**3
                socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                installer.install(profile=setup_p_h3_image.model_profile, verify=verify, runtime_only=True)
                verify.assert_not_called()

    def test_h3_ui_returns_the_activated_runtime_after_setup(self):
        from contextlib import nullcontext

        from modules_forge import minimax_h3_setup_ui as ui

        old, new = Path("old/ComfyUI"), Path("new/ComfyUI")
        with (
            patch.object(ui, "managed_runtime_root", return_value=old),
            patch.object(ui, "installed_runtime_root", return_value=new),
            patch.object(ui, "runtime_setup_session", return_value=nullcontext()),
            patch.object(ui, "_setup_process", return_value=iter(())),
        ):
            result = list(ui.setup_updates("", Path.cwd()))
        self.assertEqual(result[-1][0], str(new))


if __name__ == "__main__":
    unittest.main()
