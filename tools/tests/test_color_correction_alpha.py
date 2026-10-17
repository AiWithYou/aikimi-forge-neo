import ast
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage.exposure import match_histograms


class ColorCorrectionAlphaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "modules/processing.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"setup_color_correction", "apply_color_correction"}]
        namespace = dict(cv2=cv2, np=np, Image=Image, match_histograms=match_histograms)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.setup_correction = staticmethod(namespace["setup_color_correction"])
        cls.apply = staticmethod(namespace["apply_color_correction"])

    def test_rgba_preserves_each_alpha_value_without_mutating_source(self):
        source = Image.new("RGBA", (2, 2), (180, 70, 20, 255))
        source.putalpha(Image.fromarray(np.array([[0, 85], [170, 255]], dtype=np.uint8)))
        before = source.tobytes()
        target = self.setup_correction(Image.new("RGB", (2, 2), "blue"))
        output = self.apply(target, source)
        self.assertEqual(output.mode, "RGBA")
        self.assertEqual(output.getchannel("A").tobytes(), source.getchannel("A").tobytes())
        self.assertEqual(source.tobytes(), before)

    def test_grayscale_and_palette_inputs_use_effective_rgb_colors(self):
        gray = Image.new("L", (2, 2), 100)
        palette = Image.new("P", (2, 2), 0)
        palette.putpalette([255, 0, 0] + [0] * 765)
        palette.info["transparency"] = 0
        for source in (gray, palette):
            with self.subTest(mode=source.mode):
                target = self.setup_correction(source)
                np.testing.assert_array_equal(target, self.setup_correction(source.convert("RGB")))
                output = self.apply(target, source)
                self.assertEqual(output.size, source.size)
                self.assertEqual(output.mode, "RGBA" if source.mode == "P" else "RGB")
                if source.mode == "P":
                    self.assertEqual(output.getchannel("A").getextrema(), (0, 0))


if __name__ == "__main__":
    unittest.main()
