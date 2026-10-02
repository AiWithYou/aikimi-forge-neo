from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import setup_ming_image, setup_minimax_h3, setup_nanosaur2
from tools.aikimi_setup import SetupError


class StudioSetupBoundaryTests(unittest.TestCase):
    modules = ((setup_ming_image, "ming-image"), (setup_nanosaur2, "nanosaur2"))

    def symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            self.skipTest(f"Symlinks are unavailable: {exc}")

    def test_external_runtime_parent_is_rejected_before_the_setup_lock_write(self):
        for module, name in self.modules:
            with self.subTest(model=name), tempfile.TemporaryDirectory(prefix="Neo setup 日本語 ") as temporary:
                root = Path(temporary) / "Neo"
                outside = Path(temporary) / "external"
                (root / "repositories").mkdir(parents=True)
                outside.mkdir()
                self.symlink(root / "repositories" / name, outside, directory=True)
                runtime = Mock(side_effect=SetupError("mock stopped before any installation"))
                with patch.object(module, "install_runtime", runtime), patch.object(module.socket, "socket") as socket:
                    socket.return_value.__enter__.return_value.connect_ex.return_value = 1
                    with self.assertRaises(SetupError):
                        module.run(root=root)

                self.assertFalse((outside / "setup.lock").exists(), "Setup wrote outside the managed repository.")
                runtime.assert_not_called()

    def test_external_interpreter_or_receipt_is_rejected_before_the_readiness_shortcut(self):
        for module, name in self.modules:
            for relative in (".venv/Scripts/python.exe", "setup.json"):
                with self.subTest(model=name, path=relative), tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary) / "Neo"
                    root.mkdir()
                    outside = Path(temporary) / "external-file"
                    outside.write_bytes(b"external")
                    link = module.runtime_root(root).parent / relative
                    link.parent.mkdir(parents=True)
                    self.symlink(link, outside)
                    with patch.object(module, "runtime_ready", return_value=True) as ready:
                        with self.assertRaisesRegex(SetupError, "リンク"):
                            module.install_runtime(root)
                    ready.assert_not_called()

    def test_h3_completion_does_not_write_through_an_external_temporary_file(self):
        for relative in ("ComfyUI/extra_model_paths.yaml.tmp", "setup.json.tmp"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "Neo"
                runtime = setup_minimax_h3.managed_runtime_root(root)
                runtime.mkdir(parents=True)
                outside = Path(temporary) / "external-file"
                outside.write_text("preserve external data", encoding="utf-8")
                self.symlink(runtime.parent / relative, outside)

                try:
                    setup_minimax_h3.RuntimeInstaller(root).complete_setup(root / "models", shared=False)
                except SetupError:
                    pass

                self.assertEqual(outside.read_text(encoding="utf-8"), "preserve external data")

    def test_h3_external_runtime_executables_are_rejected_before_bootstrap(self):
        for relative in (".venv/Scripts/python.exe", "bootstrap/uv.exe"):
            with self.subTest(path=relative), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "Neo"
                root.mkdir()
                outside = Path(temporary) / "external-file"
                outside.write_bytes(b"external executable")
                link = setup_minimax_h3.managed_runtime_root(root).parent / relative
                link.parent.mkdir(parents=True)
                self.symlink(link, outside)

                with self.assertRaisesRegex(SetupError, "リンク"):
                    setup_minimax_h3.RuntimeInstaller(root)

    def test_interrupted_runtime_repair_invalidates_the_previous_completion_record(self):
        for module, name in self.modules:
            with self.subTest(model=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                record = module.runtime_root(root).parent / "setup.json"
                record.parent.mkdir(parents=True)
                record.write_text('{"schema_version": 1, "fingerprint": "previous"}', encoding="utf-8")
                failed = SimpleNamespace(returncode=1, stdout="", stderr="injected failure")
                with (
                    patch.object(module, "runtime_ready", return_value=False),
                    patch.object(module.shutil, "which", return_value="git"),
                    patch.object(module.subprocess, "run", return_value=failed) as run,
                    patch.object(module.sys, "platform", "win32"),
                ):
                    with self.assertRaises(SetupError):
                        module.install_runtime(root)

                self.assertEqual(run.call_count, 1)
                self.assertFalse(record.exists(), "An interrupted repair must not retain a completion record.")

    def test_failed_ming_bootstrap_cannot_make_a_restored_core_appear_ready(self):
        module = setup_ming_image
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = module.runtime_root(root)
            (runtime / ".git").mkdir(parents=True)
            (runtime / "requirements.txt").write_text("fixture requirements", encoding="utf-8")
            python = runtime.parent / ".venv/Scripts/python.exe"
            python.parent.mkdir(parents=True)
            python.write_bytes(b"fixture interpreter")
            for name in ("ming_image_manifest.json", "requirements-ming-image.lock"):
                path = root / "tools" / name
                path.parent.mkdir(exist_ok=True)
                path.write_text("fixture fingerprint input", encoding="utf-8")
            record = runtime.parent / "setup.json"
            record.write_text(
                json.dumps({"schema_version": 1, "fingerprint": module.runtime_fingerprint(root)}), encoding="utf-8"
            )

            def git(arguments, **_kwargs):
                if "checkout" in arguments:
                    (runtime / "main.py").write_text("fixture core restored", encoding="utf-8")
                stdout = module.manifest()["comfy_revision"] if "rev-parse" in arguments else ""
                return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

            with (
                patch.object(module.shutil, "which", return_value="git"),
                patch.object(module.subprocess, "run", side_effect=git),
                patch.object(module.sys, "platform", "win32"),
                patch.object(module.Installer, "install", side_effect=SetupError("injected bootstrap failure")),
            ):
                self.assertFalse(module.runtime_ready(root))
                with self.assertRaisesRegex(SetupError, "bootstrap failure"):
                    module.install_runtime(root)
                self.assertFalse(module.runtime_ready(root), "A failed bootstrap must not become ready after checkout.")


if __name__ == "__main__":
    unittest.main()
