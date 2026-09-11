import ast
import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import cv2  # noqa: F401 - keep the native module loaded across isolated script imports
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "krea2_local_supersample_detail.py"
ARGUMENTS = [
    "mode",
    "profile",
    "roi_boxes",
    "crop_payload",
    "core_size",
    "core_overlap",
    "process_edge",
    "steps",
    "denoising_strength",
    "candidate_count",
    "luma_residual_cap",
    "chroma_residual_cap",
    "low_frequency_reject_radius",
    "focused_context_scale",
    "focused_rewrite_feather",
    "strong_edge_protection",
    "append_guidance",
    "save_qa_crops",
    "allow_expensive_2048_full_grid",
    "maximum_tile_count",
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


def load_script_module(process_snapshots, saved_images):
    gradio_module = ModuleType("gradio")
    backend_package = ModuleType("backend")
    memory_module = ModuleType("backend.memory_management")
    modules_package = ModuleType("modules")
    scripts_module = ModuleType("modules.scripts")
    devices_module = ModuleType("modules.devices")
    images_module = ModuleType("modules.images")
    processing_module = ModuleType("modules.processing")
    shared_module = ModuleType("modules.shared")

    class Script:
        pass

    scripts_module.Script = Script
    memory_module.is_oom = lambda _exc: False
    backend_package.memory_management = memory_module
    devices_module.torch_gc = lambda: None
    images_module.flatten = lambda image, _background: image.convert("RGB")
    images_module.save_image = lambda image, *args, **kwargs: saved_images.append((image.copy(), args, kwargs))
    processing_module.get_fixed_seed = lambda seed: 777 if int(seed) == -1 else int(seed)
    processing_module.model_load_calls = 0

    def manage_model_and_prompt_cache(p):
        processing_module.model_load_calls += 1
        p.sd_model = krea2_model()

    processing_module.manage_model_and_prompt_cache = manage_model_and_prompt_cache

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
                "do_not_save_samples": p.do_not_save_samples,
                "do_not_save_grid": p.do_not_save_grid,
                "restore_faces": p.restore_faces,
                "tiling": p.tiling,
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "init_count": len(p.init_images),
                "image_mask": p.image_mask,
                "override_settings": dict(p.override_settings),
                "override_settings_restore_afterwards": getattr(
                    p, "override_settings_restore_afterwards", None
                ),
            }
        )
        p.all_prompts = [p.prompt]
        p.all_negative_prompts = [p.negative_prompt]
        p.main_prompt = p.prompt
        p.main_negative_prompt = p.negative_prompt
        p.init_latent = object()
        p.image_conditioning = object()
        p.override_settings["temporary_internal_value"] = True
        p.latents_after_sampling.append("temporary")
        p.pixels_after_sampling.append("temporary")
        shared_module.state.nextjob()
        infotext = f"{p.prompt}\nNegative prompt: {p.negative_prompt}\n" f"Steps: {p.steps}, Sampler: test, Size: {p.width}x{p.height}, Seed: {p.seed}"
        candidate = p.init_images[0].copy()
        candidate_transform = getattr(p, "test_candidate_transform", None)
        if callable(candidate_transform):
            candidate = candidate_transform(candidate)
        return SimpleNamespace(
            images=[candidate],
            extra_images=[Image.new("RGB", (1, 1))],
            info=infotext,
            infotexts=[infotext],
            seed=p.seed,
            all_seeds=[p.seed],
            width=p.width,
            height=p.height,
        )

    processing_module.process_images = process_images
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
        "modules.shared": shared_module,
    }
    spec = importlib.util.spec_from_file_location("_test_krea2_local_supersample_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    if module.np is not np:
        raise RuntimeError("GUI test loaded a second NumPy module instance")
    return module


def make_processing(output_path):
    yy, xx = np.indices((24, 32))
    values = np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=2,
    ).astype(np.uint8)
    source = Image.fromarray(values, mode="RGB")
    source.info["vram_canvas"] = '{"upstream":true}'
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
        height=24,
        denoising_strength=0.5,
        seed=123,
        subseed=456,
        steps=8,
        all_seeds=None,
        all_subseeds=None,
        extra_generation_params={},
        latents_after_sampling=["original-latent-buffer"],
        pixels_after_sampling=["original-pixel-buffer"],
        outpath_samples=str(output_path),
        prompt="test prompt with trailing spaces  ",
        negative_prompt="negative prompt",
        restore_faces=True,
        tiling=True,
        override_settings={},
        sd_model=krea2_model(),
        save_samples=lambda: False,
    )


def run_small(module, p, **overrides):
    settings = {
        "mode": "Full Image Grid",
        "profile": "Safe 1536",
        "roi_boxes": "",
        "crop_payload": 64,
        "core_size": 48,
        "core_overlap": 8,
        "process_edge": 1536,
        "steps": 4,
        "denoising_strength": 0.10,
        "candidate_count": 1,
        "luma_residual_cap": 8.0,
        "chroma_residual_cap": 2.0,
        "low_frequency_reject_radius": 12.0,
        "focused_context_scale": 2.0,
        "focused_rewrite_feather": 20.0,
        "strong_edge_protection": True,
        "append_guidance": True,
        "save_qa_crops": False,
        "allow_expensive_2048_full_grid": False,
        "maximum_tile_count": 10,
    }
    settings.update(overrides)
    return module.Krea2LocalSupersampleDetail().run(p, **settings)


class PositionalApiTests(unittest.TestCase):
    def test_ui_return_and_run_argument_order_match(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        script_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Krea2LocalSupersampleDetail")
        run_method = next(node for node in script_class.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        run_arguments = [argument.arg for argument in run_method.args.args[2:]]
        self.assertEqual(run_arguments, ARGUMENTS)
        ui_method = next(node for node in script_class.body if isinstance(node, ast.FunctionDef) and node.name == "ui")
        ui_return = next(node for node in ui_method.body if isinstance(node, ast.Return))
        return_names = [element.id for element in ui_return.value.elts]
        self.assertEqual(return_names, ARGUMENTS)


class ProfileTests(unittest.TestCase):
    def test_profile_callback_sets_every_algorithm_slider(self):
        module = load_script_module([], [])
        safe = module.Krea2LocalSupersampleDetail._profile_values("Safe 1536")
        ultra = module.Krea2LocalSupersampleDetail._profile_values("Ultra Detail 1536")
        roi = module.Krea2LocalSupersampleDetail._profile_values("ROI Ultra 2048")
        focused = module.Krea2LocalSupersampleDetail._profile_values(
            "Focused Face Rewrite 1536"
        )
        self.assertEqual(
            safe,
            (512, 384, 64, 1536, 4, 0.10, 1, 8.0, 2.0, 12.0, 2.0, 20.0),
        )
        self.assertEqual(
            ultra,
            (512, 384, 64, 1536, 5, 0.15, 2, 12.0, 3.0, 12.0, 2.0, 20.0),
        )
        self.assertEqual(
            roi,
            (512, 384, 64, 2048, 5, 0.14, 2, 12.0, 3.0, 12.0, 2.0, 20.0),
        )
        self.assertEqual(
            focused,
            (512, 384, 64, 1536, 6, 0.38, 2, 12.0, 3.0, 12.0, 2.0, 20.0),
        )
        focused_with_summary = (
            module.Krea2LocalSupersampleDetail._profile_values_with_summary(
                "Focused Face Rewrite 1536",
                "Focused ROI Rewrite",
                "8,4,24,20",
                False,
                256,
            )
        )
        self.assertEqual(focused_with_summary[:-1], focused)
        self.assertIn("Focused ROI", focused_with_summary[-1])
        self.assertIn("1 box", focused_with_summary[-1])


class GUIFlowTests(unittest.TestCase):
    def test_focused_rewrite_processes_one_enlarged_context_and_changes_only_target(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(directory)
            source = np.asarray(p.init_images[0]).copy()

            def brighten(image):
                values = np.asarray(image, dtype=np.uint16)
                return Image.fromarray(
                    np.minimum(values + 24, 255).astype(np.uint8),
                    mode="RGB",
                )

            p.test_candidate_transform = brighten
            result = run_small(
                module,
                p,
                mode="Focused ROI Rewrite",
                profile="Focused Face Rewrite 1536",
                roi_boxes="8,4,24,20",
                process_edge=1536,
                steps=6,
                denoising_strength=0.38,
                candidate_count=1,
                focused_context_scale=2.0,
                focused_rewrite_feather=4.0,
            )
            output = np.asarray(result.images[0])
            manifest = json.loads(result.images[0].info["krea2_local_supersample"])

            self.assertEqual(len(snapshots), 1)
            self.assertEqual(snapshots[0]["size"], (1536, 1536))
            self.assertEqual(snapshots[0]["steps"], 6)
            self.assertEqual(snapshots[0]["denoise"], 0.38)
            self.assertIn("deliberately magnified context crop", snapshots[0]["prompt"])
            self.assertNotIn("This input is an enlarged local crop", snapshots[0]["prompt"])
            self.assertTrue(manifest["focused_rewrite"])
            self.assertEqual(manifest["focused_region_count"], 1)
            self.assertEqual(manifest["rejected_noop_tile_count"], 0)
            self.assertEqual(manifest["quality_gate_override_count"], 1)
            self.assertEqual(manifest["tiles"][0]["selected_candidate"], 1)
            self.assertEqual(manifest["tiles"][0]["payload_box"], [0, -4, 32, 28])
            self.assertEqual(manifest["tiles"][0]["payload_side"], 32)
            self.assertEqual(manifest["tiles"][0]["effective_zoom"], 48.0)
            self.assertIsNone(manifest["tiles"][0]["agreement_coverage"])
            self.assertEqual(
                manifest["tiles"][0]["quality_gate_override_reason"],
                "excessive_low_frequency_drift",
            )
            outside = np.ones(source.shape[:2], dtype=bool)
            outside[4:20, 8:24] = False
            np.testing.assert_array_equal(output[outside], source[outside])
            self.assertGreater(np.count_nonzero(output[4:20, 8:24] != source[4:20, 8:24]), 0)

    def test_echo_candidate_returns_one_bit_identical_image_with_metadata_and_restores_p(self):
        snapshots = []
        saved_images = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, saved_images)
            p = make_processing(directory)
            original_pixels = np.asarray(p.init_images[0]).copy()
            result = run_small(module, p)

            self.assertEqual(len(snapshots), 1)
            call = snapshots[0]
            self.assertEqual(call["size"], (1536, 1536))
            self.assertEqual(call["steps"], 4)
            self.assertEqual(call["denoise"], 0.10)
            self.assertTrue(call["do_not_save_samples"])
            self.assertTrue(call["do_not_save_grid"])
            self.assertFalse(call["restore_faces"])
            self.assertFalse(call["tiling"])
            self.assertIsNone(call["image_mask"])
            self.assertTrue(call["override_settings"]["img2img_fix_steps"])
            self.assertTrue(call["override_settings_restore_afterwards"])
            self.assertTrue(call["prompt"].startswith(p.prompt))
            self.assertEqual(call["prompt"].count("This input is an enlarged local crop"), 1)

            self.assertEqual(len(result.images), 1)
            self.assertEqual(result.extra_images, [])
            self.assertEqual(result.images[0].size, (32, 24))
            np.testing.assert_array_equal(np.asarray(result.images[0]), original_pixels)
            self.assertIn("Size: 32x24, Seed: 123", result.info)
            self.assertIn("vram_canvas", result.images[0].info)
            manifest = json.loads(result.images[0].info["krea2_local_supersample"])
            self.assertEqual(manifest["input_size"], [32, 24])
            self.assertEqual(manifest["output_size"], [32, 24])
            self.assertEqual(manifest["processed_tile_count"], 1)
            self.assertEqual(manifest["rejected_noop_tile_count"], 1)
            self.assertTrue(manifest["exact_img2img_steps"])
            self.assertEqual(
                manifest["exact_img2img_steps_scope"], "internal_tiles_only"
            )
            self.assertEqual(len(manifest["tiles"][0]["candidate_seed"]), 1)
            self.assertNotIn("test prompt", result.images[0].info["krea2_local_supersample"])

            self.assertEqual((p.width, p.height), (32, 24))
            self.assertEqual((p.seed, p.subseed, p.steps), (123, 456, 8))
            self.assertEqual(p.prompt, "test prompt with trailing spaces  ")
            self.assertEqual(p.negative_prompt, "negative prompt")
            self.assertFalse(p.do_not_save_samples)
            self.assertFalse(p.do_not_save_grid)
            self.assertTrue(p.restore_faces)
            self.assertTrue(p.tiling)
            self.assertEqual(p.override_settings, {})
            self.assertEqual(p.extra_generation_params, {})
            self.assertEqual(p.latents_after_sampling, ["original-latent-buffer"])
            self.assertEqual(p.pixels_after_sampling, ["original-pixel-buffer"])
            self.assertFalse(hasattr(p, "all_prompts"))
            self.assertFalse(hasattr(p, "main_prompt"))
            self.assertFalse(hasattr(p, "init_latent"))
            self.assertEqual(saved_images, [])
            self.assertEqual(module.state.job_no, module.state.job_count)
            self.assertEqual(list(Path(directory).glob("krea2_local_supersample_*")), [])
            self.assertFalse((Path(directory) / "krea2_local_supersample_qa").exists())

    def test_global_save_is_one_png_with_existing_and_new_metadata(self):
        snapshots = []
        saved_images = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, saved_images)
            p = make_processing(directory)
            p.save_samples = lambda: True
            run_small(module, p)
            self.assertEqual(len(saved_images), 1)
            saved, args, kwargs = saved_images[0]
            self.assertEqual(saved.size, (32, 24))
            self.assertEqual(args[4], "png")
            self.assertIn("vram_canvas", kwargs["existing_info"])
            self.assertIn("parameters", kwargs["existing_info"])
            self.assertIn("krea2_local_supersample", kwargs["existing_info"])

    def test_two_candidates_use_distinct_deterministic_seeds(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(directory)
            result = run_small(
                module,
                p,
                profile="Ultra Detail 1536",
                candidate_count=2,
                steps=5,
                denoising_strength=0.15,
                luma_residual_cap=12,
                chroma_residual_cap=3,
            )
            self.assertEqual(len(snapshots), 2)
            self.assertNotEqual(snapshots[0]["seed"], snapshots[1]["seed"])
            manifest = json.loads(result.images[0].info["krea2_local_supersample"])
            self.assertEqual(
                manifest["tiles"][0]["candidate_seed"],
                [snapshots[0]["seed"], snapshots[1]["seed"]],
            )

    def test_interruption_restores_every_processing_field_and_returns_no_partial(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(directory)
            original_process = module.processing.process_images

            def interrupting_process(processing_object):
                result = original_process(processing_object)
                module.state.interrupted = True
                return result

            module.processing.process_images = interrupting_process
            with self.assertRaisesRegex(RuntimeError, "no unfinished tile was returned"):
                run_small(module, p)
            self.assertEqual((p.width, p.height, p.seed, p.steps), (32, 24, 123, 8))
            self.assertEqual(p.prompt, "test prompt with trailing spaces  ")
            self.assertTrue(p.restore_faces)
            self.assertTrue(p.tiling)
            self.assertEqual(p.override_settings, {})
            self.assertFalse(hasattr(p, "all_prompts"))

    def test_skip_and_stop_return_no_partial_and_restore_state(self):
        for flag in ("skipped", "stopping_generation"):
            with self.subTest(flag=flag), TemporaryDirectory() as directory:
                snapshots = []
                module = load_script_module(snapshots, [])
                p = make_processing(directory)
                original_process = module.processing.process_images

                def stopping_process(processing_object):
                    result = original_process(processing_object)
                    setattr(module.state, flag, True)
                    return result

                module.processing.process_images = stopping_process
                with self.assertRaisesRegex(RuntimeError, "no unfinished tile was returned"):
                    run_small(module, p)
                self.assertEqual((p.width, p.height, p.seed, p.steps), (32, 24, 123, 8))
                self.assertTrue(p.restore_faces)
                self.assertTrue(p.tiling)
                self.assertFalse(hasattr(p, "all_prompts"))

    def test_processing_exception_and_oom_restore_state_without_fallback(self):
        class FakeOOM(Exception):
            pass

        with TemporaryDirectory() as directory:
            module = load_script_module([], [])
            p = make_processing(directory)

            def raise_oom(_p):
                raise FakeOOM("allocation failed")

            module.processing.process_images = raise_oom
            module.memory_management.is_oom = lambda exc: isinstance(exc, FakeOOM)
            with self.assertRaisesRegex(
                module.Krea2LocalSupersampleMemoryError,
                "Retry with Process Edge 1536",
            ):
                run_small(
                    module,
                    p,
                    process_edge=2048,
                    allow_expensive_2048_full_grid=True,
                )
            self.assertEqual((p.width, p.height, p.seed, p.steps), (32, 24, 123, 8))
            self.assertTrue(p.restore_faces)
            self.assertTrue(p.tiling)
            self.assertFalse(hasattr(p, "all_prompts"))
            self.assertEqual(list(Path(directory).glob("krea2_local_supersample_*")), [])

        with TemporaryDirectory() as directory:
            module = load_script_module([], [])
            p = make_processing(directory)

            def raise_regular(_p):
                raise RuntimeError("ordinary processing failure")

            module.processing.process_images = raise_regular
            with self.assertRaisesRegex(RuntimeError, "ordinary processing failure"):
                run_small(module, p)
            self.assertEqual((p.width, p.height, p.seed, p.steps), (32, 24, 123, 8))
            self.assertTrue(p.restore_faces)
            self.assertTrue(p.tiling)
            self.assertFalse(hasattr(p, "all_prompts"))

    def test_qa_option_saves_only_timestamped_diagnostic_files(self):
        with TemporaryDirectory() as directory:
            module = load_script_module([], [])
            p = make_processing(directory)
            run_small(module, p, save_qa_crops=True)
            qa_root = Path(directory) / "krea2_local_supersample_qa"
            run_dirs = list(qa_root.iterdir())
            self.assertEqual(len(run_dirs), 1)
            self.assertEqual(
                {path.name for path in run_dirs[0].iterdir()},
                {
                    "source_payload.png",
                    "process_input.png",
                    "high_resolution_candidate.png",
                    "downsampled_candidate_c1.png",
                    "roundtrip_baseline_c0.png",
                    "residual_visualization.png",
                    "before_payload.png",
                    "after_payload.png",
                    "qa_manifest.json",
                },
            )
            qa_manifest = json.loads((run_dirs[0] / "qa_manifest.json").read_text(encoding="utf-8"))
            self.assertIsNone(qa_manifest["selected_candidate"])
            self.assertNotIn("test prompt", json.dumps(qa_manifest))

    def test_preflight_rejects_wrong_model_and_batch_before_any_candidate(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(directory)
            p.sd_model = SimpleNamespace(model_config=SimpleNamespace())
            module.processing.manage_model_and_prompt_cache = lambda _p: None
            with self.assertRaisesRegex(ValueError, "loaded Krea2 checkpoint"):
                run_small(module, p)
            self.assertEqual(snapshots, [])

            p = make_processing(directory)
            p.batch_size = 2
            with self.assertRaisesRegex(ValueError, "Batch Count 1"):
                run_small(module, p)
            self.assertEqual(snapshots, [])

    def test_preflight_loads_configured_krea2_after_forge_lazy_startup(self):
        snapshots = []
        with TemporaryDirectory() as directory:
            module = load_script_module(snapshots, [])
            p = make_processing(directory)
            p.sd_model = None
            result = run_small(module, p)
            self.assertEqual(len(result.images), 1)
            self.assertEqual(module.processing.model_load_calls, 1)
            self.assertEqual(type(p.sd_model).__name__, "Krea2")
            self.assertGreater(len(snapshots), 0)


if __name__ == "__main__":
    unittest.main()
