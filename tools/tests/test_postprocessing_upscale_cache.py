"""Upscale cache identity must distinguish geometry and effective palette colors."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


class UpscaleCacheTests(unittest.TestCase):
    def setUp(self):
        source = ROOT / "scripts/postprocessing_upscale.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        script = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScriptPostprocessingUpscale"
        )
        method = next(node for node in script.body if isinstance(node, ast.FunctionDef) and node.name == "_upscale")
        namespace = {
            "Image": Image,
            "np": np,
            "shared": SimpleNamespace(opts=SimpleNamespace(upscaling_max_images_in_cache=4)),
            "upscale_cache": {},
        }
        exec(  # noqa: S102 - execute only the extracted repository method in isolation
            compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace
        )
        self.upscale = namespace["_upscale"]
        self.calls = 0

        def scale(image, factor, _path):
            self.calls += 1
            return image.convert("RGBA").resize((int(image.width * factor), int(image.height * factor)))

        self.upscaler = SimpleNamespace(name="fixture", data_path=None, scaler=SimpleNamespace(upscale=scale))

    def run_upscale(self, image):
        return self.upscale(None, image, {}, self.upscaler, 0, 2.0, 0, 1024, 1024, False)

    def test_equal_flat_pixels_with_different_shapes_do_not_reuse_output(self):
        first = self.run_upscale(Image.new("RGB", (2, 4), "red"))
        second = self.run_upscale(Image.new("RGB", (4, 2), "red"))
        self.assertEqual(first.size, (4, 8))
        self.assertEqual(second.size, (8, 4))
        self.assertEqual(self.calls, 2)

    def test_equal_palette_indices_with_different_colors_do_not_reuse_output(self):
        first, second = Image.new("P", (4, 4)), Image.new("P", (4, 4))
        first.putpalette([255, 0, 0] + [0] * 765)
        second.putpalette([0, 0, 255] + [0] * 765)
        self.assertEqual(self.run_upscale(first).getpixel((0, 0)), (255, 0, 0, 255))
        self.assertEqual(self.run_upscale(second).getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(self.calls, 2)

    def test_identical_copy_still_reuses_cached_output(self):
        source = Image.new("RGBA", (4, 4), (10, 20, 30, 40))
        first, second = self.run_upscale(source), self.run_upscale(source.copy())
        self.assertEqual(second.tobytes(), first.tobytes())
        self.assertEqual(self.calls, 1)


if __name__ == "__main__":
    unittest.main()
