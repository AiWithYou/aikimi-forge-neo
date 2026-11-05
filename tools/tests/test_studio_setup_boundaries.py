from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tools import setup_minimax_h3
from tools.aikimi_setup import SetupError


class StudioSetupBoundaryTests(unittest.TestCase):

    def symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            self.skipTest(f"Symlinks are unavailable: {exc}")



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




if __name__ == "__main__":
    unittest.main()
