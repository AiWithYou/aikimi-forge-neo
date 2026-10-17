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
        scope = {
            "os": host_os, "opts": self.options, "shared": SimpleNamespace(opts=self.options),
            "FilenameGenerator": lambda *args, **kwargs: None, "script_callbacks": callbacks,
            "save_image_with_geninfo": lambda image, info, filename, extension, **kwargs: image.save(filename, format="PNG"),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), scope)  # noqa: S102 -- isolated production save path with real file writes
        self.save = scope["save_image"]

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
