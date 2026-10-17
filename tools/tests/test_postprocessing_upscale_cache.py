"""Upscale cache identity must distinguish geometry and effective palette colors."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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
        methods = [node for node in script.body if isinstance(node, ast.FunctionDef) and node.name in {"_upscale", "process", "process_firstpass"}]
        helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "limit_size_by_one_dimension"]
        namespace = {
            "Image": Image,
            "np": np,
            "shared": SimpleNamespace(opts=SimpleNamespace(upscaling_max_images_in_cache=4, ESRGAN_tile=256, ESRGAN_tile_overlap=16)),
            "upscale_cache": {},
            "scripts_postprocessing": SimpleNamespace(PostprocessedImage=object),
            "setup_color_correction": Mock(return_value="correction"),
            "apply_color_correction": Mock(side_effect=lambda _correction, image: Image.new("RGB", image.size, "blue")),
        }
        exec(  # noqa: S102 - execute only the extracted repository method in isolation
            compile(ast.Module(body=[*helpers, *methods], type_ignores=[]), str(source), "exec"), namespace
        )
        self.upscale = namespace["_upscale"]
        self.namespace = namespace
        self.calls = 0

        def scale(image, factor, _path):
            self.calls += 1
            return image.convert("RGBA").resize((int(image.width * factor), int(image.height * factor)))

        self.upscaler = SimpleNamespace(name="fixture", data_path=None, scaler=SimpleNamespace(upscale=scale))
        namespace["shared"].sd_upscalers = [self.upscaler]

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

    def test_transparent_color_metadata_is_part_of_cache_identity(self):
        for mode, color, transparent, opaque in (
            ("L", 10, 10, 20),
            ("RGB", (10, 20, 30), (10, 20, 30), (1, 2, 3)),
        ):
            with self.subTest(mode=mode):
                self.namespace["upscale_cache"].clear()
                first = Image.new(mode, (4, 4), color)
                second = first.copy()
                first.info["transparency"] = transparent
                second.info["transparency"] = opaque
                self.assertEqual(self.run_upscale(first).getpixel((0, 0))[3], 0)
                self.assertEqual(self.run_upscale(second).getpixel((0, 0))[3], 255)

    def test_color_correction_survives_disabled_or_single_image_cache(self):
        script = SimpleNamespace()
        script._upscale = self.upscale.__get__(script)
        for limit in (0, 1):
            with self.subTest(limit=limit):
                self.namespace["upscale_cache"].clear()
                self.namespace["shared"].opts.upscaling_max_images_in_cache = limit
                pp = SimpleNamespace(image=Image.new("RGB", (4, 4), "red"), info={}, shared=SimpleNamespace())
                args = dict(upscale_mode=0, upscale_by=2.0, upscale_cc=True, upscaler_1_name="fixture")
                self.namespace["process_firstpass"](script, pp, **args)
                self.namespace["process"](script, pp, **args)
                self.assertEqual(pp.image.getpixel((0, 0)), (0, 0, 255))
                self.assertLessEqual(len(self.namespace["upscale_cache"]), limit)

    def test_lowering_cache_limit_evicts_all_excess_images(self):
        for color in ("red", "blue", "green", "yellow"):
            self.run_upscale(Image.new("RGB", (4, 4), color))
        self.namespace["shared"].opts.upscaling_max_images_in_cache = 1
        self.run_upscale(Image.new("RGB", (4, 4), "white"))
        self.assertEqual(len(self.namespace["upscale_cache"]), 1)

    def test_unlimited_firstpass_preserves_requested_dimensions(self):
        pp = SimpleNamespace(image=Image.new("RGB", (30, 20)), shared=SimpleNamespace())
        self.namespace["process_firstpass"](None, pp, upscale_mode=0, upscale_by=2.0, max_side_length=0)
        self.assertEqual((pp.shared.target_width, pp.shared.target_height), (60, 40))

    def test_limit_never_produces_zero_for_thin_images(self):
        limit = self.namespace["limit_size_by_one_dimension"]
        self.assertEqual(limit(1000, 1, 64), (64, 1))
        self.assertEqual(limit(1, 1000, 64), (1, 64))
        self.assertEqual(limit(100, 50, 64), (64, 32))

    def test_tiling_setting_changes_recompute_the_image(self):
        options = self.namespace["shared"].opts
        self.upscaler.scaler.upscale = lambda image, factor, path: Image.new("RGB", image.size, (options.ESRGAN_tile // 2, options.ESRGAN_tile_overlap, 0))
        image = Image.new("RGB", (4, 4))
        self.assertEqual(self.run_upscale(image).getpixel((0, 0)), (128, 16, 0))
        options.ESRGAN_tile = 128
        self.assertEqual(self.run_upscale(image).getpixel((0, 0)), (64, 16, 0))
        options.ESRGAN_tile_overlap = 32
        self.assertEqual(self.run_upscale(image).getpixel((0, 0)), (64, 32, 0))

    def test_same_name_different_model_path_does_not_reuse_output(self):
        self.upscaler.scaler.upscale = lambda image, factor, path: Image.new("RGB", image.size, path)
        image = Image.new("RGB", (4, 4))
        self.upscaler.data_path = "red"
        self.assertEqual(self.run_upscale(image).getpixel((0, 0)), (255, 0, 0))
        self.upscaler.data_path = "blue"
        self.assertEqual(self.run_upscale(image).getpixel((0, 0)), (0, 0, 255))

    def test_thin_image_limit_uses_long_edge_for_the_scale(self):
        image = Image.new("RGB", (1000, 1))
        upscale = Mock(return_value=image)
        self.upscaler.scaler.upscale = upscale
        self.upscale(None, image, {}, self.upscaler, 0, 2.0, 64, 1024, 1024, False)
        self.assertEqual(upscale.call_args.args[1], 64 / 1000)

    def test_planned_size_matches_minimum_output_dimensions(self):
        path = ROOT / "modules/extras_workflow.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "planned_size")
        scope = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
        values = dict(upscale_enabled=True, upscaler_1_name="fixture", upscale_mode=0, upscale_crop=False, upscale_by=2, max_side_length=64)
        self.assertEqual(scope["planned_size"]((1000, 1), values), (64, 8))
        self.assertEqual(scope["planned_size"]((1, 1000), values), (8, 64))


if __name__ == "__main__":
    unittest.main()
