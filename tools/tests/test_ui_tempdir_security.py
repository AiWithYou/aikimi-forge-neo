import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules import ui_tempdir


class ManagedTempCleanupTests(unittest.TestCase):
    def test_cleanup_removes_only_aikimi_owned_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ordinary = root / "photo.png"
            managed_png = root / f"{ui_tempdir.MANAGED_TEMP_PREFIX}one.png"
            managed_webp = root / f"{ui_tempdir.MANAGED_TEMP_PREFIX}two.webp"
            ordinary.write_bytes(b"keep")
            managed_png.write_bytes(b"remove")
            managed_webp.write_bytes(b"remove")

            with mock.patch.object(
                ui_tempdir,
                "shared",
                SimpleNamespace(opts=SimpleNamespace(temp_dir=str(root)), demo=None),
            ):
                ui_tempdir.cleanup_tmpdr()

            self.assertTrue(ordinary.is_file())
            self.assertFalse(managed_png.exists())
            self.assertFalse(managed_webp.exists())

    def test_cleanup_does_not_follow_directory_symlinks(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside_directory:
            root = Path(directory)
            outside = Path(outside_directory)
            protected = outside / f"{ui_tempdir.MANAGED_TEMP_PREFIX}protected.png"
            protected.write_bytes(b"keep")
            link = root / "outside-link"
            try:
                link.symlink_to(outside, target_is_directory=True)
            except OSError:
                self.skipTest("directory symlinks require additional privileges on this Windows host")

            with mock.patch.object(
                ui_tempdir,
                "shared",
                SimpleNamespace(opts=SimpleNamespace(temp_dir=str(root)), demo=None),
            ):
                ui_tempdir.cleanup_tmpdr()

            self.assertTrue(protected.is_file())


if __name__ == "__main__":
    unittest.main()
