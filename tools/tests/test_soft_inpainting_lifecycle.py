"""Exercise consecutive soft-inpaint callbacks with real CPU masks and Pillow images."""

import ast
import hashlib
import math
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from joblib import delayed
from PIL import Image, ImageFilter, ImageOps

from modules import masking

ROOT = Path(__file__).resolve().parents[2]
SOFT_PATH = ROOT / "extensions-builtin/soft-inpainting/scripts/soft_inpainting.py"
SETTINGS = (True, 1, 0.5, 4, 0, 0.5, 2)


class SerialParallel:
    """Keep the real histogram algorithm while running its tasks on this CPU."""

    def __init__(self, **_kwargs):
        pass

    def __call__(self, jobs):
        return [function(*args, **kwargs) for function, args, kwargs in jobs]


def load_functions(relative_path, names, scope):
    path = ROOT / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)  # noqa: S102
    return tree


class SoftInpaintingLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.scope = {
            "Image": Image,
            "ImageOps": ImageOps,
            "ImageFilter": ImageFilter,
            "np": np,
            "torch": torch,
            "hashlib": hashlib,
            "LANCZOS": Image.Resampling.LANCZOS,
            "opts": SimpleNamespace(
                upscaler_for_img2img="None",
                img2img_background_color="#ffffff",
                img2img_inpaint_precise_mask=False,
                img2img_color_correction=False,
                save_init_img=False,
                sd_vae_encode_method="Full",
            ),
        }
        load_functions("modules/images.py", {"resize_image", "flatten"}, self.scope)
        images = types.ModuleType("modules.images")
        images.resize_image = self.scope["resize_image"]
        images.flatten = self.scope["flatten"]
        self.scope["images"] = images
        tree = load_functions(
            "modules/processing.py",
            {"create_binary_mask", "uncrop", "_scale_inpaint_crop_region", "apply_overlay"},
            self.scope,
        )
        constant = next(
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_INPAINT_FULL_RES_OVERLAY_MASK"
                for target in node.targets
            )
        )
        exec(  # noqa: S102
            compile(ast.Module(body=[constant], type_ignores=[]), str(ROOT / "modules/processing.py"), "exec"),
            self.scope,
        )
        proc = types.ModuleType("modules.processing")
        proc.create_binary_mask = self.scope["create_binary_mask"]
        proc.uncrop = self.scope["uncrop"]
        shared = types.ModuleType("modules.shared")
        shared.opts = self.scope["opts"]
        shared.device = torch.device("cpu")
        modules_stub = types.ModuleType("modules")
        modules_stub.__path__ = []
        modules_stub.images = images
        modules_stub.processing = proc
        modules_stub.shared = shared
        module_patch = mock.patch.dict(
            sys.modules,
            {"modules": modules_stub, "modules.images": images, "modules.processing": proc, "modules.shared": shared},
        )
        module_patch.start()
        self.addCleanup(module_patch.stop)
        self.scope.update(
            sd_samplers=SimpleNamespace(create_sampler=lambda *_: None),
            args=SimpleNamespace(dynamic_args=SimpleNamespace(kontext=False, edit=False, wan=False, pid=False)),
            masking=masking,
            logger=SimpleNamespace(warning=lambda *_: None),
            shared=shared,
            devices=SimpleNamespace(dtype=torch.float32, torch_gc=lambda: None),
            approximation_indexes={"Full": None},
            images_tensor_to_samples=lambda image, *_: torch.zeros((image.shape[0], 4, 8, 8)),
        )
        owner = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "StableDiffusionProcessingImg2Img"
        )
        init_method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "init")
        exec(  # noqa: S102
            compile(ast.Module(body=[init_method], type_ignores=[]), str(ROOT / "modules/processing.py"), "exec"),
            self.scope,
        )

        self.soft = {
            "dataclass": dataclass,
            "Image": Image,
            "torch": torch,
            "np": np,
            "math": math,
            "Parallel": SerialParallel,
            "cpu_count": lambda: 8,
            "delayed": delayed,
            "scripts": SimpleNamespace(
                ScriptBuiltinUI=object, PostSampleArgs=object, PostProcessMaskOverlayArgs=object
            ),
        }
        tree = ast.parse(SOFT_PATH.read_text(encoding="utf-8"))
        names = {
            "processing_uses_inpainting",
            "apply_adaptive_masks",
            "weighted_histogram_filter",
            "get_gaussian_kernel",
            "smootherstep",
        }
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        settings = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SoftInpaintingSettings"
        )
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Script")
        owner.body = [
            node
            for node in owner.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"__init__", "process", "post_sample", "postprocess_maskoverlay"}
        ]
        exec(  # noqa: S102
            compile(ast.Module(body=[settings, *functions, owner], type_ignores=[]), str(SOFT_PATH), "exec"), self.soft
        )
        self.soft["enabled_gen_param_label"] = "Soft inpainting enabled"
        fields = (
            "mask_blend_power",
            "mask_blend_scale",
            "inpaint_detail_preservation",
            "composite_mask_influence",
            "composite_difference_threshold",
            "composite_difference_contrast",
        )
        self.soft["gen_param_labels"] = SimpleNamespace(**{field: field for field in fields})
        self.script = self.soft["Script"]()

    def job(self, color, value):
        return SimpleNamespace(
            extra_generation_params={},
            denoising_strength=0.0,
            sampler_name="Euler",
            sd_model=SimpleNamespace(comments=[]),
            image_mask=Image.new("L", (64, 64), value),
            mask=None,
            nmask=None,
            mask_round=True,
            inpainting_mask_invert=0,
            mask_blur=0,
            inpaint_full_res=True,
            inpaint_full_res_padding=0,
            width=64,
            height=64,
            mask_for_overlay=None,
            latent_mask=None,
            scripts=None,
            color_corrections=None,
            init_images=[Image.new("RGB", (64, 64), color)],
            batch_size=1,
            overlay_images=None,
            resize_mode=0,
            inpainting_fill=1,
            paste_to=None,
            img2img_image_conditioning=lambda *_: None,
        )

    def run_job(self, p, settings=SETTINGS):
        self.script.process(p, *settings)
        self.scope["init"](p, [], [1], [2])
        self.script.post_sample(p, SimpleNamespace(samples=p.init_latent.clone()), *settings)
        return p

    def overlay(self, p, settings=SETTINGS):
        result = SimpleNamespace(index=0, mask_for_overlay=p.mask_for_overlay, overlay_image=None)
        self.script.postprocess_maskoverlay(p, result, *settings)
        return result

    def test_empty_mask_fallback_does_not_composite_the_previous_job(self):
        self.run_job(self.job("red", 255))
        second = self.run_job(self.job("blue", 0))
        self.assertFalse(second.inpaint_full_res)
        self.assertIsNone(second.nmask)
        self.assertIsNone(second.mask_for_overlay)
        overlay = self.overlay(second)
        output, _ = self.scope["apply_overlay"](
            Image.new("RGB", (64, 64), "blue"), second.paste_to, overlay.overlay_image
        )
        self.assertEqual(output.getpixel((0, 0)), (0, 0, 255))
        self.assertIsNone(overlay.mask_for_overlay)

    def test_valid_next_job_rebuilds_its_own_overlay(self):
        self.run_job(self.job("red", 255))
        second = self.run_job(self.job("blue", 255))
        overlay = self.overlay(second)
        self.assertEqual(overlay.overlay_image.getpixel((0, 0)), (0, 0, 255, 255))
        self.assertEqual(overlay.mask_for_overlay.convert("L").getextrema(), (0, 0))

    def test_disable_after_enable_releases_overlay_state(self):
        self.run_job(self.job("red", 255))
        second = self.run_job(self.job("blue", 255), (False, *SETTINGS[1:]))
        self.assertIsNone(self.overlay(second, (False, *SETTINGS[1:])).overlay_image)
        self.assertIsNone(self.script.masks_for_overlay)
        self.assertIsNone(self.script.overlay_images)

    def test_post_sample_failure_cannot_reuse_an_earlier_batch_mask(self):
        first = self.run_job(self.job("red", 255))
        with mock.patch.dict(
            self.soft, {"apply_adaptive_masks": mock.Mock(side_effect=RuntimeError("histogram failure"))}
        ):
            with self.assertRaisesRegex(RuntimeError, "histogram failure"):
                self.script.post_sample(first, SimpleNamespace(samples=first.init_latent.clone()), *SETTINGS)
        self.assertIsNone(self.overlay(first).overlay_image)
        self.assertIsNone(self.script.masks_for_overlay)


if __name__ == "__main__":
    unittest.main()
