import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image

# Load the shared core before replacing WebUI modules with lightweight GUI stubs.
import tools.krea2_b5_tile_regenerate  # noqa: F401


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "krea2_b5_tile_regenerate.py"
ARGUMENTS = [
    "working_scale",
    "stages",
    "maximum_tile_count",
    "prompt_mode",
    "tile_size",
    "process_edge",
    "overlap",
    "steps",
    "denoising_strength",
    "merge_mode",
    "low_anchor_sigma",
    "detail_radius",
    "detail_gain",
    "max_tile_delta",
    "structure_sigma",
    "base_detail_sigma",
    "protection_rows",
    "print_detail_strength",
    "print_detail_radius",
    "print_detail_threshold",
    "print_max_detail_delta",
    "save_stage_images",
    "save_manifest",
]


def krea2_model():
    Krea2 = type("Krea2", (), {})
    Qwen3VLTextProcessingEngine = type("Qwen3VLTextProcessingEngine", (), {})
    Qwen2DVAE = type("Qwen2DVAE", (), {})
    model = Krea2()
    model.model_config = Krea2()
    text_engine = Qwen3VLTextProcessingEngine()
    text_engine.text_encoder = object()
    text_engine.tokenizer = object()
    model.text_processing_engine_qwen = text_engine
    vae_model = Qwen2DVAE()
    vae_model.process_in = lambda value: value
    vae_model.process_out = lambda value: value
    vae = SimpleNamespace(
        first_stage_model=vae_model,
        is_wan=True,
        latent_channels=16,
        encode=lambda value: value,
        decode=lambda value: value,
    )
    model.forge_objects = SimpleNamespace(
        vae=vae,
        unet=object(),
        clip=object(),
    )
    return model


def load_script_module(process_snapshots, saved_paths):
    gradio_module = gr
    backend_package = ModuleType("backend")
    memory_module = ModuleType("backend.memory_management")
    modules_package = ModuleType("modules")
    scripts_module = ModuleType("modules.scripts")
    devices_module = ModuleType("modules.devices")
    images_module = ModuleType("modules.images")
    processing_module = ModuleType("modules.processing")
    quality_module = ModuleType("modules.krea2_quality")
    shared_module = ModuleType("modules.shared")

    class Script:
        def elem_id(self, item):
            return f"test_{item}"

    scripts_module.Script = Script
    memory_module.is_oom = lambda _exc: False
    backend_package.memory_management = memory_module
    devices_module.torch_gc = lambda: None
    images_module.flatten = lambda image, _background: image.convert("RGB")
    processing_module.get_fixed_seed = lambda seed: 777 if int(seed) == -1 else int(seed)
    processing_module.manage_model_and_prompt_cache = lambda p: setattr(
        p, "sd_model", krea2_model()
    )
    quality_module.adaptive_detail_guard = lambda image, **_kwargs: (
        image,
        {"applied": False},
    )

    def nextjob():
        shared_module.state.job_no += 1

    shared_module.state = SimpleNamespace(
        interrupted=False,
        skipped=False,
        stopping_generation=False,
        job="",
        textinfo="",
        job_count=-1,
        job_no=0,
        nextjob=nextjob,
    )
    shared_module.opts = SimpleNamespace(
        img2img_background_color="#ffffff",
        temp_dir="",
        enable_pnginfo=True,
        sd_model_checkpoint="Krea2_Center NF4.safetensors",
        forge_additional_modules=[
            "qwen_image_vae.safetensors",
            "qwen3vl_4b.safetensors",
        ],
    )

    def process_images(p):
        process_snapshots.append(
            {
                "size": (p.width, p.height),
                "seed": p.seed,
                "steps": p.steps,
                "denoise": p.denoising_strength,
                "override_settings": dict(p.override_settings),
                "restore_afterwards": p.override_settings_restore_afterwards,
                "do_not_save_samples": p.do_not_save_samples,
                "restore_faces": p.restore_faces,
                "tiling": p.tiling,
                "image_mask": p.image_mask,
                "prompt": p.prompt,
            }
        )
        p.latents_after_sampling.append("temporary")
        p.pixels_after_sampling.append("temporary")
        shared_module.state.nextjob()
        if getattr(p, "interrupt_after_process", False):
            shared_module.state.interrupted = True
        infotext = (
            f"{p.prompt}\nNegative prompt: {p.negative_prompt}\n"
            f"Steps: {p.steps}, Sampler: test, Size: {p.width}x{p.height}, Seed: {p.seed}"
        )
        return SimpleNamespace(
            images=[p.init_images[0].copy()],
            extra_images=[],
            info=infotext,
            infotexts=[infotext],
            seed=p.seed,
            all_seeds=[p.seed],
            video_path=None,
        )

    processing_module.process_images = process_images

    def save_image(image, path, _basename, *args, **kwargs):
        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        current = kwargs.get("p")
        image.test_save_context = {
            "width": current.width,
            "height": current.height,
            "seed": current.seed,
            "subseed": current.subseed,
            "prompt": current.prompt,
            "negative_prompt": current.negative_prompt,
        }
        forced = kwargs.get("forced_filename")
        name = f"{forced}.png" if forced else f"saved_{len(saved_paths) + 1}.png"
        output = destination / name
        image.save(output, format="PNG")
        image.already_saved_as = str(output)
        saved_paths.append(output)
        return str(output), None

    images_module.save_image = save_image
    modules_package.scripts = scripts_module
    modules_package.devices = devices_module
    modules_package.images = images_module
    modules_package.processing = processing_module
    stubs = {
        "gradio": gradio_module,
        "backend": backend_package,
        "backend.memory_management": memory_module,
        "modules": modules_package,
        "modules.scripts": scripts_module,
        "modules.devices": devices_module,
        "modules.images": images_module,
        "modules.processing": processing_module,
        "modules.krea2_quality": quality_module,
        "modules.shared": shared_module,
    }
    spec = importlib.util.spec_from_file_location("_test_krea2_b5_gui", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def make_processing(output_path: Path, *, save_samples: bool = False):
    source = Image.new("RGB", (32, 48), (80, 100, 120))
    source.info["upstream"] = "kept"
    return SimpleNamespace(
        batch_size=1,
        n_iter=1,
        init_images=[source],
        image_mask=None,
        latent_mask=None,
        do_not_save_samples=False,
        do_not_save_grid=False,
        resize_mode=1,
        width=32,
        height=48,
        denoising_strength=0.5,
        seed=123,
        subseed=456,
        subseed_strength=0.0,
        steps=8,
        sampler_name="DPM++ 2M SDE",
        scheduler="Simple",
        cfg_scale=1.0,
        distilled_cfg_scale=1.15,
        all_seeds=None,
        all_subseeds=None,
        extra_generation_params={},
        latents_after_sampling=["original-latent-buffer"],
        pixels_after_sampling=["original-pixel-buffer"],
        outpath_samples=str(output_path),
        prompt="source prompt",
        negative_prompt="negative prompt",
        restore_faces=True,
        tiling=True,
        override_settings={},
        sd_model=krea2_model(),
        save_samples=lambda: save_samples,
    )


def run_small(module, p, **overrides):
    settings = {
        "working_scale": 1,
        "stages": 1,
        "maximum_tile_count": 10,
        "prompt_mode": module.PROMPT_MODE_LOCAL,
        "tile_size": 32,
        "process_edge": 80,
        "overlap": 0,
        "steps": 3,
        "denoising_strength": 0.25,
        "merge_mode": "low_anchor",
        "low_anchor_sigma": 4.0,
        "detail_radius": 2,
        "detail_gain": 1.0,
        "max_tile_delta": 8.0,
        "structure_sigma": 4.0,
        "base_detail_sigma": 1.0,
        "protection_rows": [],
        "print_detail_strength": 0.0,
        "print_detail_radius": 1.0,
        "print_detail_threshold": 0.7,
        "print_max_detail_delta": 3.0,
        "save_stage_images": False,
        "save_manifest": False,
    }
    settings.update(overrides)
    with patch.object(module, "DEFAULT_TARGET_SIZE", (64, 96)):
        return module.Krea2B5TileRegenerate().run(p, **settings)


class PositionalApiTests(unittest.TestCase):
    def test_ui_return_and_run_argument_order_match(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        script_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "Krea2B5TileRegenerate"
        )
        run_method = next(
            node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "run"
        )
        run_arguments = [argument.arg for argument in run_method.args.args[2:]]
        self.assertEqual(run_arguments, ARGUMENTS)
        ui_method = next(
            node
            for node in script_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "ui"
        )
        ui_return = next(
            node for node in ui_method.body if isinstance(node, ast.Return)
        )
        return_names = [element.id for element in ui_return.value.elts]
        self.assertEqual(return_names, ARGUMENTS)

    def test_ui_renders_with_real_gradio_and_exposes_protection_table(self):
        module = load_script_module([], [])
        with gr.Blocks():
            components = module.Krea2B5TileRegenerate().ui(True)
        self.assertEqual(len(components), len(ARGUMENTS))
        table = components[ARGUMENTS.index("protection_rows")]
        self.assertIsInstance(table, gr.Dataframe)
        self.assertEqual(
            table.headers,
            ["名前", "左", "上", "右", "下", "フェザー"],
        )
        self.assertEqual(table.type, "array")
        save_stages = components[ARGUMENTS.index("save_stage_images")]
        self.assertFalse(save_stages.value)


class ProtectionTableTests(unittest.TestCase):
    def test_table_parses_dynamic_rows_and_ignores_blank_rows(self):
        module = load_script_module([], [])
        regions = module.Krea2B5TileRegenerate._parse_protection_rows(
            [
                ["face", 2, 3, 18, 24, 4],
                ["", "", "", "", "", ""],
                ["", float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            ],
            (32, 48),
        )
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].label, "face")
        self.assertEqual(regions[0].box, (2, 3, 18, 24))
        self.assertEqual(regions[0].feather, 4)

    def test_table_rejects_partial_and_duplicate_rows(self):
        module = load_script_module([], [])
        with self.assertRaisesRegex(ValueError, "フェザー"):
            module.Krea2B5TileRegenerate._parse_protection_rows(
                [["face", 2, 3, 18, 24, ""]],
                (32, 48),
            )
        with self.assertRaisesRegex(ValueError, "重複"):
            module.Krea2B5TileRegenerate._parse_protection_rows(
                [
                    ["face", 2, 3, 18, 24, 4],
                    ["face", 8, 9, 20, 26, 3],
                ],
                (32, 48),
            )


class PlanTests(unittest.TestCase):
    def test_recommended_1024x1448_plan_has_165_tile_passes(self):
        module = load_script_module([], [])
        count = module.Krea2B5TileRegenerate._total_tile_count(
            (1024, 1448),
            working_scale=2,
            stages=1,
            tile_size=256,
            overlap=64,
        )
        self.assertEqual(count, 165)
        values = module.Krea2B5TileRegenerate._recommended_values_with_summary()
        self.assertEqual(values[:12], (2, 1, 256, "safe_local", 256, 1024, 64, 6, 0.35, "low_anchor", 16.0, 16.0))
        self.assertIn("JIS B5 2896×4096", values[-1])
        self.assertIn("165 tile pass", values[-1])


class GUIRunTests(unittest.TestCase):
    def test_run_uses_internal_exact_steps_progress_and_restores_processing(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(Path(directory))
            result = run_small(module, p)

        self.assertEqual(len(snapshots), 2)
        self.assertTrue(all(call["size"] == (80, 80) for call in snapshots))
        self.assertTrue(
            all(call["override_settings"]["img2img_fix_steps"] for call in snapshots)
        )
        self.assertTrue(all(call["restore_afterwards"] for call in snapshots))
        self.assertTrue(all(call["do_not_save_samples"] for call in snapshots))
        self.assertTrue(all(not call["restore_faces"] for call in snapshots))
        self.assertTrue(all(not call["tiling"] for call in snapshots))
        self.assertTrue(all(call["image_mask"] is None for call in snapshots))
        self.assertTrue(all(call["prompt"].startswith("An s5p2style") for call in snapshots))
        self.assertEqual(result.images[0].size, (64, 96))
        summary = json.loads(result.images[0].info["krea2_b5_tile_regenerate"])
        self.assertEqual(summary["tile_count"], 2)
        self.assertTrue(summary["exact_img2img_steps"])
        self.assertEqual(summary["exact_img2img_steps_scope"], "internal_tiles_only")
        self.assertEqual(module.state.job_count, 3)
        self.assertEqual(module.state.job_no, 3)
        self.assertEqual((p.width, p.height, p.steps), (32, 48, 8))
        self.assertEqual((p.seed, p.subseed), (123, 456))
        self.assertEqual(p.prompt, "source prompt")
        self.assertEqual(p.negative_prompt, "negative prompt")
        self.assertTrue(p.restore_faces)
        self.assertTrue(p.tiling)
        self.assertEqual(p.override_settings, {})
        self.assertEqual(p.extra_generation_params, {})
        self.assertEqual(p.latents_after_sampling, ["original-latent-buffer"])
        self.assertEqual(p.pixels_after_sampling, ["original-pixel-buffer"])

    def test_print_finish_is_applied_before_final_protection(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(Path(directory))

            def brighten(image, **_kwargs):
                return Image.new("RGB", image.size, (200, 210, 220)), {
                    "applied": True
                }

            with patch.object(module, "adaptive_detail_guard", brighten):
                result = run_small(
                    module,
                    p,
                    protection_rows=[["face", 4, 8, 20, 32, 3]],
                )
            final = result.images[0]
            self.assertEqual(final.getpixel((24, 40)), (80, 100, 120))
            self.assertEqual(final.getpixel((60, 90)), (200, 210, 220))

    def test_b5_aspect_mismatch_is_rejected_before_tiles(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(Path(directory))
            p.init_images = [Image.new("RGB", (32, 44), (80, 100, 120))]
            with self.assertRaisesRegex(ValueError, "aspect ratio"):
                run_small(module, p)
        self.assertEqual(snapshots, [])

    def test_stage_temp_png_is_not_created_when_normal_sample_saving_is_off(self):
        snapshots = []
        saved_paths = []
        observed_save_stage = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, saved_paths)
            p = make_processing(Path(directory), save_samples=False)
            original = module.regenerate_stage

            def record_stage_setting(*args, **kwargs):
                observed_save_stage.append(kwargs["save_stage"])
                return original(*args, **kwargs)

            with patch.object(module, "regenerate_stage", record_stage_setting):
                run_small(module, p, save_stage_images=True)
        self.assertEqual(observed_save_stage, [False])
        self.assertEqual(saved_paths, [])

    def test_save_writes_final_stage_and_adjacent_manifest(self):
        snapshots = []
        saved_paths = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, saved_paths)
            p = make_processing(Path(directory), save_samples=True)
            result = run_small(
                module,
                p,
                save_stage_images=True,
                save_manifest=True,
                protection_rows=[["face", 4, 8, 20, 32, 3]],
            )
            final_path = Path(result.images[0].already_saved_as)
            manifest_path = final_path.with_suffix(".json")
            self.assertTrue(final_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertEqual(len(saved_paths), 2)
            self.assertEqual(
                result.images[0].test_save_context,
                {
                    "width": 64,
                    "height": 96,
                    "seed": 123,
                    "subseed": 456,
                    "prompt": module.LOCAL_TILE_PROMPT,
                    "negative_prompt": "negative prompt",
                },
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["output"]["path"], str(final_path))
            self.assertEqual(manifest["tile_count"], 2)
            self.assertEqual(len(manifest["protection_regions"]), 1)
            self.assertIsNotNone(manifest["final_protection"])
            self.assertTrue(Path(manifest["stage_reports"][0]["output"]).is_file())

    def test_interruption_returns_no_partial_image_and_restores_processing(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(Path(directory))
            p.interrupt_after_process = True
            with self.assertRaisesRegex(RuntimeError, "no unfinished tile"):
                run_small(module, p)
            self.assertEqual((p.width, p.height, p.steps), (32, 48, 8))
            self.assertEqual(p.prompt, "source prompt")
            self.assertEqual(p.override_settings, {})
            self.assertEqual(p.latents_after_sampling, ["original-latent-buffer"])


if __name__ == "__main__":
    unittest.main()
