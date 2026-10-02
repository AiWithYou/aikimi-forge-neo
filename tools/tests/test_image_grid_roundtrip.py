"""Real Pillow grids and CPU upscalers preserve small and partial tiles."""

import io
import logging
import math
import re
import unittest
from collections import namedtuple
from collections.abc import Callable
from contextlib import nullcontext, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import torch
import tqdm
from PIL import Image, ImageColor

from tools.tests.test_api_extras_boundaries import load_classes
from tools.tests.test_gpu_ownership import load_function
from tools.tests.test_runtime_efficiency import load_definitions


class TinyNearestUpscaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, tensor):
        return torch.nn.functional.interpolate(tensor, scale_factor=2, mode="nearest")


class ImageGridRoundTripTests(unittest.TestCase):
    def setUp(self):
        self.opts = SimpleNamespace(n_rows=-1, grid_prevent_empty_spots=False, grid_background_color="white")
        callbacks = load_classes("modules/script_callbacks.py", {"ImageGridLoopParams"}, {})
        namespace = {
            "Image": Image,
            "ImageColor": ImageColor,
            "math": math,
            "np": np,
            "namedtuple": namedtuple,
            "opts": self.opts,
            "script_callbacks": SimpleNamespace(
                ImageGridLoopParams=callbacks.ImageGridLoopParams, image_grid_callback=Mock()
            ),
        }
        self.images = SimpleNamespace(
            **load_definitions(
                "modules/images.py", {"Grid", "split_grid", "combine_grid", "image_grid", "flatten"}, namespace
            )
        )
        upscaler_namespace = {
            "Image": Image,
            "torch": torch,
            "np": np,
            "Callable": Callable,
            "images": self.images,
            "tqdm": tqdm,
            "logger": logging.getLogger(__name__),
            "devices": SimpleNamespace(without_autocast=nullcontext),
            "torch_utils": SimpleNamespace(
                get_param=load_function("modules/torch_utils.py", "get_param", {"torch": torch})
            ),
            "shared": SimpleNamespace(
                opts=SimpleNamespace(enable_upscale_progressbar=False), state=SimpleNamespace(interrupted=False)
            ),
        }
        self.upscale = load_definitions(
            "modules/upscaler_utils.py",
            {
                "_model",
                "pil_image_to_torch_bgr",
                "torch_bgr_to_pil_image",
                "upscale_pil_patch",
                "upscale_with_model_cpu",
            },
            upscaler_namespace,
        )["upscale_with_model_cpu"]
        self.enterContext(redirect_stdout(io.StringIO()))

    @staticmethod
    def pattern(size):
        width, height = size
        return Image.fromarray(np.arange(width * height * 3, dtype=np.uint8).reshape(height, width, 3))

    def test_split_and_combine_keep_odd_partial_and_thin_tiles(self):
        for size in ((73, 69), (17, 23), (9, 7), (15, 15), (15, 35), (35, 15)):
            for overlap in (0, 4):
                with self.subTest(size=size, overlap=overlap):
                    image = self.pattern(size)
                    grid = self.images.split_grid(image, tile_w=16, tile_h=16, overlap=overlap)
                    result = self.images.combine_grid(grid)
                    self.assertEqual(result.size, image.size)
                    np.testing.assert_array_equal(np.asarray(result), np.asarray(image))

    def test_default_split_keeps_one_tile_for_small_positive_dimensions(self):
        for size in ((1, 1), (8, 8), (16, 16), (64, 64), (64, 100), (100, 64)):
            with self.subTest(size=size):
                image = self.pattern(size)
                grid = self.images.split_grid(image)
                self.assertEqual(grid.tile_count, 1)
                self.assertEqual((grid.image_w, grid.image_h), size)
                np.testing.assert_array_equal(np.asarray(self.images.combine_grid(grid)), np.asarray(image))

    def test_actual_cpu_nearest_upscale_preserves_small_and_tile_boundary_pixels(self):
        sizes = {
            (8, 8): 1,
            (16, 16): 1,
            (16, 40): 1,
            (40, 16): 1,
            (17, 40): 1,
            (16, 600): 3,
            (600, 16): 3,
            (250, 255): 1,
            (255, 250): 1,
            (256, 257): 2,
            (257, 256): 2,
            (513, 509): 9,
        }
        for size, tile_count in sizes.items():
            with self.subTest(size=size):
                image = self.pattern(size)
                grid = self.images.split_grid(image, tile_w=256, tile_h=256, overlap=16)
                self.assertEqual(grid.tile_count, tile_count)
                result = self.upscale(TinyNearestUpscaler(), image, tile_size=256, tile_overlap=16)
                self.assertEqual(result.size, (size[0] * 2, size[1] * 2))
                expected = image.resize(result.size, Image.Resampling.NEAREST)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))

    def test_image_grid_keeps_rgba_palette_and_luminance_alpha_pixels(self):
        rgba = Image.new("RGBA", (3, 2), (80, 120, 160, 64))
        luminance = Image.new("LA", (3, 2), (110, 80))
        palette = Image.new("P", (3, 2), 0)
        palette.putpalette([80, 120, 160] + [0] * 765)
        palette.info["transparency"] = 0
        for image in (rgba, luminance, palette):
            with self.subTest(mode=image.mode):
                result = self.images.image_grid([image], rows=1)
                np.testing.assert_array_equal(np.asarray(result), np.asarray(image.convert("RGBA")))

    def test_grid_rows_and_centered_cells_keep_original_dimensions_and_pixels(self):
        images = [
            Image.new("RGB", size, color)
            for size, color in (
                ((3, 2), "red"),
                ((1, 4), "green"),
                ((5, 1), "blue"),
                ((2, 2), "yellow"),
                ((4, 3), "orange"),
            )
        ]
        layouts = ((-1, False, 1, 2, 3), (-1, True, 1, 1, 5), (0, False, 3, 3, 2), (16, False, 1, 5, 1))
        for setting, prevent_empty, batch_size, rows, cols in layouts:
            with self.subTest(setting=setting, prevent_empty=prevent_empty):
                self.opts.n_rows = setting
                self.opts.grid_prevent_empty_spots = prevent_empty
                result = self.images.image_grid(images, batch_size=batch_size)
                self.assertEqual(result.size, (cols * 5, rows * 4))
                for index, image in enumerate(images):
                    x = index % cols * 5 + (5 - image.width) // 2
                    y = index // cols * 4 + (4 - image.height) // 2
                    cell = result.crop((x, y, x + image.width, y + image.height))
                    np.testing.assert_array_equal(np.asarray(cell), np.asarray(image.convert("RGBA")))

    def test_sd_upscale_small_source_schedules_and_recombines_one_tile(self):
        renders = []

        def render(process):
            renders.append(process.init_images[0].size)
            return SimpleNamespace(
                info="Size: 512x512", seed=process.seed, images=[image.copy() for image in process.init_images]
            )

        namespace = {
            "scripts": SimpleNamespace(Script=object),
            "Image": Image,
            "math": math,
            "re": re,
            "images": self.images,
            "opts": SimpleNamespace(img2img_background_color="white", samples_save=False),
            "state": SimpleNamespace(job_count=1),
            "shared": SimpleNamespace(sd_upscalers=[SimpleNamespace(name="None")]),
            "devices": SimpleNamespace(torch_gc=Mock()),
            "processing": SimpleNamespace(fix_seed=Mock(), process_images=render),
            "Processed": lambda process, images, seed, info: SimpleNamespace(images=images, seed=seed, info=info),
        }
        script = load_classes("scripts/sd_upscale.py", {"SDUpscale"}, namespace).SDUpscale()
        image = self.pattern((64, 64))
        process = SimpleNamespace(
            extra_generation_params={}, seed=1, init_images=[image], width=512, height=512, batch_size=1, n_iter=1
        )
        result = script.run(process, overlap=64, upscaler_index=0, scale_factor=1.0, override=False)
        self.assertEqual(renders, [(512, 512)])
        self.assertEqual(result.info, "Size: 64x64")
        self.assertEqual(len(result.images), 1)
        np.testing.assert_array_equal(np.asarray(result.images[0]), np.asarray(image))


if __name__ == "__main__":
    unittest.main()
