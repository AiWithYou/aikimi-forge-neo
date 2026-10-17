"""Exercise builtin resamplers before the shared final dimension adjustment."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class BuiltinUpscalerThinImageTests(unittest.TestCase):
    def test_thin_image_shrinks_only_once_before_final_size_adjustment(self):
        path = Path(__file__).resolve().parents[2] / "modules/upscaler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        base = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Upscaler")
        upscale = next(n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == "upscale")
        for name in ("UpscalerLanczos", "UpscalerNearest"):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "do_upscale")
            scope = {
                "Image": Image, "LANCZOS": Image.Resampling.LANCZOS, "NEAREST": Image.Resampling.NEAREST,
                "UPSCALE_ITERATIONS": 4, "shared": SimpleNamespace(state=SimpleNamespace(interrupted=False)),
            }
            exec(compile(ast.Module(body=[upscale, method], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
            for size in ((1000, 1), (1, 1000)):
                with self.subTest(upscaler=name, size=size):
                    image = Image.new("RGB", size)
                    image.putdata([(x % 256, (x * 3) % 256, (x * 7) % 256) for x in range(1000)])
                    scaler = SimpleNamespace()
                    calls = []

                    def resize(source, _model, calls=calls, scope=scope, scaler=scaler):
                        calls.append(source.size)
                        return scope["do_upscale"](scaler, source)

                    scaler.do_upscale = resize
                    result = scope["upscale"](scaler, image, 0.064)
                    expected_size = (64, 8) if size[0] > size[1] else (8, 64)
                    intermediate = (64, 1) if size[0] > size[1] else (1, 64)
                    filter_ = Image.Resampling.LANCZOS if name == "UpscalerLanczos" else Image.Resampling.NEAREST
                    expected = image.resize(intermediate, filter_).resize(expected_size, Image.Resampling.LANCZOS)
                    self.assertEqual(calls, [size])
                    self.assertEqual(result.size, expected_size)
                    self.assertEqual(result.tobytes(), expected.tobytes())

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
