"""Exercise builtin resamplers before the shared final dimension adjustment."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class BuiltinUpscalerThinImageTests(unittest.TestCase):
    def test_builtin_resamplers_keep_intermediate_dimensions_positive(self):
        path = Path(__file__).resolve().parents[2] / "modules/upscaler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in ("UpscalerLanczos", "UpscalerNearest"):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "do_upscale")
            scope = {"Image": Image, "LANCZOS": Image.Resampling.LANCZOS, "NEAREST": Image.Resampling.NEAREST}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
            for size, scale, expected in (((1000, 1), 0.064, (64, 1)), ((1, 1000), 0.064, (1, 64)), ((40, 60), 0.5, (20, 30))):
                with self.subTest(upscaler=name, size=size):
                    result = scope["do_upscale"](SimpleNamespace(scale=scale), Image.new("RGBA", size, (20, 40, 60, 85)))
                    self.assertEqual(result.size, expected)
                    self.assertEqual(result.mode, "RGBA")
                    self.assertEqual(result.getchannel("A").getextrema(), (85, 85))


if __name__ == "__main__":
    unittest.main()
