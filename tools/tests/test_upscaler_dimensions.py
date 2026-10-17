from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class UpscalerDimensionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[2] / "modules/upscaler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Upscaler")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "upscale")
        scope = {
            "Image": Image,
            "LANCZOS": Image.Resampling.LANCZOS,
            "UPSCALE_ITERATIONS": 4,
            "shared": SimpleNamespace(state=SimpleNamespace(interrupted=False)),
        }
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102 -- isolated production method
        self.upscale = scope["upscale"]
        self.scaler = SimpleNamespace(do_upscale=lambda image, model: image)

    def test_small_images_keep_positive_eight_pixel_dimensions(self):
        for size, scale, expected in (
            ((1, 1), 1.0, (8, 8)),
            ((2, 3), 2.0, (8, 8)),
            ((1, 17), 2.0, (8, 32)),
            ((17, 1), 2.0, (32, 8)),
        ):
            with self.subTest(size=size, scale=scale):
                result = self.upscale(self.scaler, Image.new("RGB", size, "red"), scale)
                self.assertEqual(result.size, expected)
                self.assertEqual(result.getpixel((0, 0)), (255, 0, 0))

    def test_normal_dimensions_preserve_existing_rounding(self):
        for size, scale, expected in (
            ((64, 80), 2.0, (128, 160)),
            ((17, 29), 1.0, (16, 32)),
            ((100, 100), 1.5, (152, 152)),
        ):
            with self.subTest(size=size, scale=scale):
                result = self.upscale(self.scaler, Image.new("RGB", size), scale)
                self.assertEqual(result.size, expected)


if __name__ == "__main__":
    unittest.main()
