import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from modules_forge import krea2_upscale


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "krea2_2stage_upscale.py"
LEGACY_ARGUMENTS = [
    "final_long_edge",
    "first_pass_long_edge",
    "diffusion_long_edge_cap",
    "final_width",
    "final_height",
    "first_pass_denoise",
    "final_denoise",
    "method",
    "tile_width",
    "tile_height",
    "tile_overlap",
    "tile_batch_size",
    "save_stage1",
]
APPENDED_ARGUMENTS = [
    "model_profile",
    "allow_non_native_diffusion",
    "smart_finish",
    "smart_despeckle",
    "smart_color_strength",
]


def load_script_module(process_snapshots, saved_images):
    modules_package = ModuleType("modules")
    scripts_module = ModuleType("modules.scripts")
    devices_module = ModuleType("modules.devices")
    images_module = ModuleType("modules.images")
    processing_module = ModuleType("modules.processing")
    quality_module = ModuleType("modules.krea2_quality")
    shared_module = ModuleType("modules.shared")

    class Script:
        pass

    class Processed:
        def __init__(self, p, image_list, seed, info):
            self.images = image_list
            self.seed = seed
            self.info = info

    scripts_module.Script = Script
    devices_module.torch_gc = lambda: None
    images_module.flatten = lambda image, _background: image.convert("RGB")
    images_module.save_image = lambda image, *args, **kwargs: saved_images.append(
        (image.copy(), args, kwargs)
    )
    processing_module.Processed = Processed
    processing_module.fix_seed = lambda _p: None

    def process_images(p):
        process_snapshots.append(
            {
                "size": (p.width, p.height),
                "do_not_save_samples": p.do_not_save_samples,
                "flow": p.extra_generation_params["Krea2 Upscale flow"],
                "stage1_status": p.extra_generation_params["Krea2 Stage 1 status"],
            }
        )
        prompt_line = f"prompt keeps Size: {p.width}x{p.height} as literal"
        infotext = (
            f"{prompt_line}\n"
            f"Steps: 8, Sampler: test, Size: {p.width}x{p.height}, Seed: 1"
        )
        return SimpleNamespace(
            images=[Image.new("RGB", (p.width, p.height), (96, 112, 128))],
            info=infotext,
            infotexts=[infotext],
            index_of_first_image=0,
            width=p.width,
            height=p.height,
        )

    processing_module.process_images = process_images
    quality_module.smart_finish_image = lambda image, **_kwargs: (image, {})
    quality_module.smart_finish_summary = lambda _report: "unused"
    shared_module.opts = SimpleNamespace(
        img2img_background_color="#ffffff",
        enable_pnginfo=True,
        samples_format="png",
    )
    shared_module.state = SimpleNamespace(
        interrupted=False,
        skipped=False,
        stopping_generation=False,
        job="",
        textinfo="",
    )

    modules_package.scripts = scripts_module
    modules_package.devices = devices_module
    modules_package.images = images_module
    modules_package.processing = processing_module

    stubs = {
        "modules": modules_package,
        "modules.scripts": scripts_module,
        "modules.devices": devices_module,
        "modules.images": images_module,
        "modules.processing": processing_module,
        "modules.krea2_quality": quality_module,
        "modules.shared": shared_module,
        "modules_forge.krea2_upscale": krea2_upscale,
    }
    spec = importlib.util.spec_from_file_location(
        "_test_krea2_2stage_upscale_script", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    module.KreaTwoStageUpscale._enable_multidiffusion = staticmethod(
        lambda *_args: None
    )
    return module


def make_processing(source_size, save_samples_result):
    return SimpleNamespace(
        batch_size=1,
        n_iter=1,
        init_images=[Image.new("RGB", source_size, (80, 96, 112))],
        image_mask=None,
        latent_mask=None,
        do_not_save_samples=False,
        do_not_save_grid=False,
        resize_mode=1,
        width=source_size[0],
        height=source_size[1],
        denoising_strength=0.5,
        script_args=[],
        scripts=SimpleNamespace(alwayson_scripts=[]),
        extra_generation_params={},
        latents_after_sampling=[],
        pixels_after_sampling=[],
        outpath_samples="unused",
        seed=123,
        prompt="prompt",
        save_samples=lambda: save_samples_result,
    )


def run_script(script, p, *, save_stage1=True):
    return script.run(
        p,
        512,
        0,
        256,
        0,
        0,
        0.10,
        0.12,
        "Mixture of Diffusers",
        256,
        256,
        32,
        1,
        save_stage1,
        "custom",
        False,
        False,
        False,
        0.8,
    )


class PositionalApiTests(unittest.TestCase):
    def test_ui_return_and_run_keep_legacy_arguments_first(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        script_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "KreaTwoStageUpscale"
        )
        run_method = next(
            node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        run_arguments = [argument.arg for argument in run_method.args.args[2:]]
        self.assertEqual(run_arguments, LEGACY_ARGUMENTS + APPENDED_ARGUMENTS)

        ui_method = next(
            node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "ui"
        )
        ui_return = next(
            node for node in ui_method.body if isinstance(node, ast.Return)
        )
        return_names = [element.id for element in ui_return.value.elts]
        self.assertEqual(return_names, LEGACY_ARGUMENTS + APPENDED_ARGUMENTS)

    def test_quick_8k_target_includes_matching_live_summary(self):
        module = load_script_module([], [])
        values = module.KreaTwoStageUpscale._quick_target_values_with_summary(
            8192,
            "custom",
            False,
            0.10,
            0.12,
            768,
            768,
            96,
            True,
        )

        self.assertEqual(values[:5], (8192, 0, 0, 0, 0))
        self.assertIn("大判納品", values[-1])
        self.assertIn("長辺 8192 px", values[-1])

    def test_legacy_thirteen_argument_call_uses_safe_new_defaults(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing((256, 128), save_samples_result=False)

        result = module.KreaTwoStageUpscale().run(
            p,
            512,
            0,
            256,
            0,
            0,
            0.10,
            0.12,
            "Mixture of Diffusers",
            256,
            256,
            32,
            1,
            True,
        )

        self.assertEqual(result.images[0].size, (512, 256))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(p.extra_generation_params["Krea2 Model profile"], "custom")
        self.assertTrue(p.extra_generation_params["Krea2 Smart Finish"])


class UpscaleFlowTests(unittest.TestCase):
    def test_profile_auto_cap_uses_raw_and_turbo_native_limits(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        script = module.KreaTwoStageUpscale()

        self.assertEqual(
            script._effective_diffusion_cap(0, "raw"),
            krea2_upscale.KREA2_RAW_NATIVE_LONG_EDGE,
        )
        self.assertEqual(
            script._effective_diffusion_cap(0, "turbo"),
            krea2_upscale.KREA2_TURBO_NATIVE_LONG_EDGE,
        )
        self.assertEqual(script._effective_diffusion_cap(1536, "raw"), 1536)

    def test_profile_auto_cap_records_effective_value(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing((256, 128), save_samples_result=False)

        module.KreaTwoStageUpscale().run(
            p,
            512,
            0,
            0,
            0,
            0,
            0.10,
            0.12,
            "Mixture of Diffusers",
            256,
            256,
            32,
            1,
            True,
            "raw",
            False,
            False,
            False,
            0.8,
        )

        self.assertEqual(
            p.extra_generation_params["Krea2 2-Stage Diffusion cap"],
            krea2_upscale.KREA2_RAW_NATIVE_LONG_EDGE,
        )

    def test_skips_stage1_near_proxy_and_only_rewrites_metadata_size(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing((256, 128), save_samples_result=False)

        result = run_script(module.KreaTwoStageUpscale(), p)

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["size"], (256, 128))
        self.assertEqual(snapshots[0]["flow"], "single-stage")
        self.assertIn("skipped", snapshots[0]["stage1_status"])
        self.assertEqual(result.images[0].size, (512, 256))
        self.assertIn("prompt keeps Size: 256x128 as literal", result.info)
        self.assertIn("Steps: 8, Sampler: test, Size: 512x256", result.info)
        self.assertEqual(saved_images, [])
        self.assertEqual((p.width, p.height), (256, 128))
        self.assertFalse(p.do_not_save_samples)

    def test_two_stage_save_policy_and_final_manual_save(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing((64, 32), save_samples_result=True)

        result = run_script(module.KreaTwoStageUpscale(), p, save_stage1=True)

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["size"], (128, 64))
        self.assertFalse(snapshots[0]["do_not_save_samples"])
        self.assertEqual(snapshots[0]["flow"], "two-stage")
        self.assertEqual(snapshots[1]["size"], (256, 128))
        self.assertTrue(snapshots[1]["do_not_save_samples"])
        self.assertEqual(result.images[0].size, (512, 256))
        self.assertEqual(len(saved_images), 1)


if __name__ == "__main__":
    unittest.main()
