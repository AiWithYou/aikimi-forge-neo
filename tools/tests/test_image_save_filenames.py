from __future__ import annotations

import ast
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class ImageSaveFilenameTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[2] / "modules/images.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "save_image")
        self.options = SimpleNamespace(save_images_replace_action="Override", target_side_length=1024, export_for_4chan=False, save_txt=True)
        host_os = SimpleNamespace(**{key: getattr(os, key) for key in dir(os) if not key.startswith("__")})
        host_os.statvfs = lambda path: SimpleNamespace(f_namemax=48)
        self.saved = []
        callbacks = SimpleNamespace(
            ImageSaveParams=lambda image, p, filename, pnginfo: SimpleNamespace(image=image, filename=filename, pnginfo=pnginfo),
            before_image_saved_callback=lambda params: None,
            image_saved_callback=lambda params: self.saved.append(params.filename),
        )
        self.host_os, self.callbacks = host_os, callbacks
        scope = {
            "os": host_os, "opts": self.options, "shared": SimpleNamespace(opts=self.options),
            "FilenameGenerator": lambda *args, **kwargs: None, "script_callbacks": callbacks,
            "save_image_with_geninfo": lambda image, info, filename, extension, **kwargs: image.save(filename, format="PNG"),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102 -- isolated production save path with real file writes
        self.save = scope["save_image"]

    def test_collision_numbers_fit_byte_limit_and_callbacks_report_saved_path(self):
        self.options.save_images_replace_action = "Add number"
        with tempfile.TemporaryDirectory() as temporary:
            outputs = []
            for index in range(12):
                filename, metadata = self.save(Image.new("RGB", (8, 8), (index, 0, 0)), temporary, "",
                                               forced_filename="画像" * 30, save_to_dirs=False, info=str(index))
                outputs.append(filename)
                self.assertLessEqual(len(os.fsencode(Path(filename).name)), 48)
                self.assertEqual(self.saved[-1], filename)
                self.assertEqual(Path(metadata).read_text(encoding="utf-8"), f"{index}\n")
            self.assertEqual(len(set(outputs)), 12)
            for index, filename in enumerate(outputs):
                with Image.open(filename) as saved:
                    self.assertEqual(saved.getpixel((0, 0)), (index, 0, 0))

    def test_callback_redirect_uses_destination_filesystem_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "redirect"
            destination.mkdir()
            checked = []

            def limits(path):
                checked.append(Path(path))
                return SimpleNamespace(f_namemax=32 if Path(path) == destination else 48)

            def redirect(params):
                params.filename = str(destination / ("x" * 70 + ".png"))

            self.host_os.statvfs = limits
            self.callbacks.before_image_saved_callback = redirect
            filename, _ = self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="original", save_to_dirs=False)
            self.assertEqual(Path(filename).parent, destination)
            self.assertLessEqual(len(os.fsencode(Path(filename).name)), 32)
            self.assertIn(destination, checked)

    def test_override_policy_replaces_existing_image(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = []
            for color in ("red", "blue"):
                filename, _ = self.save(Image.new("RGB", (8, 8), color), temporary, "", forced_filename="same", save_to_dirs=False)
                paths.append(filename)
            self.assertEqual(paths[0], paths[1])
            with Image.open(paths[1]) as saved:
                self.assertEqual(saved.getpixel((0, 0)), (0, 0, 255))

    def test_without_statvfs_collision_still_reports_actual_filename(self):
        del self.host_os.statvfs
        self.options.save_images_replace_action = "Add number"
        with tempfile.TemporaryDirectory() as temporary:
            for index in range(2):
                filename, _ = self.save(Image.new("RGB", (8, 8)), temporary, "", forced_filename="image", save_to_dirs=False)
                self.assertEqual(Path(filename).name, "image.png" if index == 0 else "image-1.png")
                self.assertEqual(self.saved[-1], filename)

    def test_long_directory_is_preserved_and_filename_respects_byte_limit(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "deep-output-directory-longer-than-the-filename-limit" / "images"
            for name in ("short", "画像" * 30, "long-name-" * 20):
                with self.subTest(name=name):
                    image = Image.new("RGB", (8, 8), "red")
                    filename, metadata = self.save(image, str(directory), "", forced_filename=name, save_to_dirs=False, info="test")
                    output = Path(filename)
                    self.addCleanup(output.unlink, missing_ok=True)
                    self.addCleanup(Path(metadata).unlink, missing_ok=True)
                    self.assertEqual(output.parent, directory)
                    self.assertLessEqual(len(os.fsencode(output.name)), 48)
                    self.assertEqual(output.suffix, ".png")
                    with Image.open(output) as saved:
                        self.assertEqual(saved.getpixel((0, 0)), (255, 0, 0))
                    self.assertEqual(Path(metadata).read_text(encoding="utf-8"), "test\n")
                    self.assertEqual(image.already_saved_as, filename)


if __name__ == "__main__":
    unittest.main()
