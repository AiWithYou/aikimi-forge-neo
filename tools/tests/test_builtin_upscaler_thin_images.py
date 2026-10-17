"""Verify builtin resamplers use one pass and preserve their selected filter."""

import ast
import unittest
from abc import abstractmethod
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


class BuiltinUpscalerThinImageTests(unittest.TestCase):
    def test_nearest_final_rounding_does_not_introduce_colors(self):
        path = Path(__file__).resolve().parents[2] / "modules/upscaler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in {"Upscaler", "UpscalerData", "UpscalerNearest"}]
        scope = {
            "Image": Image, "LANCZOS": Image.Resampling.LANCZOS, "NEAREST": Image.Resampling.NEAREST,
            "abstractmethod": abstractmethod, "UPSCALE_ITERATIONS": 4,
            "shared": SimpleNamespace(state=SimpleNamespace(interrupted=False)),
            "opts": SimpleNamespace(ESRGAN_tile=0, ESRGAN_tile_overlap=0),
            "devices": SimpleNamespace(device_esrgan="cpu"),
        }
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
        image = Image.new("L", (17, 17))
        image.putdata([255 if i % 17 >= 8 else 0 for i in range(17 * 17)])
        result = scope["UpscalerNearest"]().upscale(image, 1)
        self.assertEqual(result.size, (16, 16))
        self.assertEqual(result.tobytes(), image.resize((16, 16), Image.Resampling.NEAREST).tobytes())

    def test_builtin_resamplers_resize_once_to_final_dimensions(self):
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
            for size, factor in (((1000, 1), 0.064), ((1, 1000), 0.064), ((1000, 1), 1.5), ((14, 14), 2.0)):
                with self.subTest(upscaler=name, size=size):
                    image = Image.new("RGB", size)
                    image.putdata([(x % 256, (x * 3) % 256, (x * 7) % 256) for x in range(size[0] * size[1])])
                    scaler = SimpleNamespace()
                    calls = []

                    def resize(source, _model, calls=calls, scope=scope, scaler=scaler):
                        calls.append(source.size)
                        return scope["do_upscale"](scaler, source)

                    scaler.do_upscale = resize
                    result = scope["upscale"](scaler, image, factor)
                    expected_size = tuple(max(8, round(dimension * factor / 8) * 8) for dimension in size)
                    filter_ = Image.Resampling.LANCZOS if name == "UpscalerLanczos" else Image.Resampling.NEAREST
                    expected = image.resize(expected_size, filter_)
                    self.assertEqual(calls, [size])
                    self.assertEqual(result.size, expected_size)
                    self.assertEqual(result.tobytes(), expected.tobytes())

    def test_builtin_resamplers_preserve_alpha_and_final_dimensions(self):
        path = Path(__file__).resolve().parents[2] / "modules/upscaler.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in ("UpscalerLanczos", "UpscalerNearest"):
            cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
            method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "do_upscale")
            scope = {"Image": Image, "LANCZOS": Image.Resampling.LANCZOS, "NEAREST": Image.Resampling.NEAREST}
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
            for size, scale, expected in (((1000, 1), 0.064, (64, 8)), ((1, 1000), 0.064, (8, 64)), ((40, 60), 0.5, (16, 32))):
                with self.subTest(upscaler=name, size=size):
                    result = scope["do_upscale"](SimpleNamespace(scale=scale), Image.new("RGBA", size, (20, 40, 60, 85)))
                    self.assertEqual(result.size, expected)
                    self.assertEqual(result.mode, "RGBA")
                    self.assertEqual(result.getchannel("A").getextrema(), (85, 85))


if __name__ == "__main__":
    unittest.main()
