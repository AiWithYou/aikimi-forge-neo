import ast
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "vram_canvas_highres.py"
ARGUMENTS = [
    "final_long_edge",
    "final_width",
    "final_height",
    "vram_budget_gib",
    "tile_size",
    "model_reserve_gib",
    "max_stage_scale",
    "phase_count",
    "minimum_steps",
    "maximum_steps",
    "detail_knee",
    "coarse_denoise",
    "final_denoise",
    "low_pass_radius",
    "detail_gain",
    "max_detail_delta",
    "structure_sigma",
    "base_detail_sigma",
    "consensus_sigma",
    "novel_detail_gain",
    "novel_detail_max_delta",
    "save_stages",
    "append_krea2_detail_prompt",
    "smart_finish",
    "smart_color_strength",
    "detail_guard",
    "finish_detail_strength",
    "finish_detail_radius",
    "finish_detail_threshold",
    "finish_max_detail_delta",
    "novel_detail_inner_radius",
    "novel_detail_outer_radius",
    "novel_detail_structure_sigma",
    "novel_detail_consensus_sigma",
    "novel_detail_consensus_strength",
    "merge_mode",
]


def load_script_module(process_snapshots, saved_images):
    gradio_module = ModuleType("gradio")
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
        pass

    scripts_module.Script = Script
    memory_module.get_total_memory = lambda _device: 8 * 1024**3
    backend_package.memory_management = memory_module
    devices_module.device = "cuda:0"
    devices_module.torch_gc = lambda: None
    images_module.flatten = lambda image, _background: image.convert("RGB")
    images_module.save_image = lambda image, *args, **kwargs: saved_images.append((image.copy(), args, kwargs))
    processing_module.fix_seed = lambda _p: None
    processing_module.get_fixed_seed = lambda seed: 777 if int(seed) == -1 else int(seed)
    quality_module.smart_finish_image = lambda image, **_kwargs: (
        image.copy(),
        {
            "version": 2,
            "detail_guard": {"applied": True, "accepted": True},
            "chroma_mura": {
                "applied": False,
                "before": {"p95_chroma_delta": 0.0},
                "after": {"p95_chroma_delta": 0.0},
            },
        },
    )
    quality_module.smart_finish_summary = lambda _report: "detail=applied"

    def process_images(p):
        p.all_prompts = [p.prompt]
        p.main_prompt = p.prompt
        p.prompts = [p.prompt]
        process_snapshots.append(
            {
                "size": (p.width, p.height),
                "seed": p.seed,
                "steps": p.steps,
                "denoise": p.denoising_strength,
                "do_not_save_samples": p.do_not_save_samples,
                "restore_faces": p.restore_faces,
                "tiling": p.tiling,
                "prompt": p.prompt,
                "override_settings": dict(getattr(p, "override_settings", {})),
                "override_settings_restore_afterwards": getattr(
                    p, "override_settings_restore_afterwards", None
                ),
            }
        )
        shared_module.state.nextjob()
        infotext = f"prompt literal Seed: 999 and Size: {p.width}x{p.height}\n" f"Steps: {p.steps}, Sampler: test, Size: {p.width}x{p.height}, Seed: {p.seed}"
        return SimpleNamespace(
            images=[p.init_images[0].copy()],
            info=infotext,
            infotexts=[infotext],
            seed=p.seed,
            all_seeds=[p.seed],
            width=p.width,
            height=p.height,
        )

    processing_module.process_images = process_images
    shared_module.opts = SimpleNamespace(
        img2img_background_color="#ffffff",
        temp_dir="",
        enable_pnginfo=True,
        samples_format="png",
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
    spec = importlib.util.spec_from_file_location("_test_vram_canvas_highres_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    if module.np is not np:
        raise RuntimeError("VRAM-Canvas GUI test loaded a second NumPy module instance")
    return module


def make_processing():
    source = Image.new("RGB", (64, 32), (80, 100, 120))
    return SimpleNamespace(
        batch_size=1,
        n_iter=1,
        init_images=[source],
        image_mask=None,
        latent_mask=None,
        do_not_save_samples=False,
        do_not_save_grid=False,
        resize_mode=1,
        width=64,
        height=32,
        denoising_strength=0.5,
        seed=123,
        subseed=456,
        steps=8,
        all_seeds=None,
        all_subseeds=None,
        extra_generation_params={},
        latents_after_sampling=[],
        pixels_after_sampling=[],
        outpath_samples="unused",
        prompt="test prompt",
        restore_faces=True,
        tiling=True,
        save_samples=lambda: False,
    )


def run_small(module, p, **overrides):
    settings = {
        "final_long_edge": 128,
        "final_width": 0,
        "final_height": 0,
        "vram_budget_gib": 8,
        "tile_size": 0,
        "model_reserve_gib": 5.5,
        "max_stage_scale": 2.0,
        "phase_count": 1,
        "minimum_steps": 2,
        "maximum_steps": 4,
        "detail_knee": 0.035,
        "coarse_denoise": 0.12,
        "final_denoise": 0.08,
        "low_pass_radius": 12,
        "detail_gain": 1.0,
        "max_detail_delta": 32,
        "structure_sigma": 18,
        "base_detail_sigma": 6,
        "consensus_sigma": 8,
        "novel_detail_gain": 0,
        "novel_detail_max_delta": 8,
        "save_stages": False,
        "smart_finish": False,
    }
    settings.update(overrides)
    return module.VRAMCanvasHighres().run(p, **settings)


class PositionalApiTests(unittest.TestCase):
    def test_ui_return_and_run_argument_order_match(self):
        tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
        script_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VRAMCanvasHighres")
        run_method = next(node for node in script_class.body if isinstance(node, ast.FunctionDef) and node.name == "run")
        run_arguments = [argument.arg for argument in run_method.args.args[2:]]
        self.assertEqual(run_arguments, ARGUMENTS)

        ui_method = next(node for node in script_class.body if isinstance(node, ast.FunctionDef) and node.name == "ui")
        ui_return = next(node for node in ui_method.body if isinstance(node, ast.Return))
        return_names = [element.id for element in ui_return.value.elts]
        self.assertEqual(return_names, ARGUMENTS)

class GUIFlowTests(unittest.TestCase):
    def test_gui_run_uses_internal_processing_and_restores_state(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing()
        p.latents_after_sampling[:] = ["original-latent"]
        p.pixels_after_sampling[:] = ["original-pixel"]

        result = module.VRAMCanvasHighres().run(
            p,
            128,
            0,
            0,
            8,
            0,
            5.5,
            2.0,
            1,
            2,
            4,
            0.035,
            0.12,
            0.08,
            12,
            1.0,
            32,
            18,
            6,
            8,
            0,
            8,
            False,
        )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]["size"], (128, 64))
        self.assertNotEqual(snapshots[0]["seed"], 123)
        self.assertTrue(snapshots[0]["do_not_save_samples"])
        self.assertFalse(snapshots[0]["restore_faces"])
        self.assertFalse(snapshots[0]["tiling"])
        self.assertTrue(snapshots[0]["override_settings"]["img2img_fix_steps"])
        self.assertTrue(snapshots[0]["override_settings_restore_afterwards"])
        self.assertEqual(result.images[0].size, (128, 64))
        self.assertIn("prompt literal Seed: 999", result.info)
        self.assertIn("Steps: 2, Sampler: test, Size: 128x64, Seed: 123", result.info)
        self.assertEqual(result.seed, 123)
        self.assertEqual(saved_images, [])
        self.assertEqual((p.width, p.height), (64, 32))
        self.assertEqual(p.seed, 123)
        self.assertEqual(p.steps, 8)
        self.assertFalse(p.do_not_save_samples)
        self.assertTrue(p.restore_faces)
        self.assertTrue(p.tiling)
        self.assertEqual(p.prompt, "test prompt")
        self.assertFalse(hasattr(p, "all_prompts"))
        self.assertFalse(hasattr(p, "main_prompt"))
        self.assertFalse(hasattr(p, "prompts"))
        self.assertFalse(hasattr(p, "override_settings"))
        self.assertFalse(hasattr(p, "override_settings_restore_afterwards"))
        self.assertEqual(p.latents_after_sampling, ["original-latent"])
        self.assertEqual(p.pixels_after_sampling, ["original-pixel"])
        self.assertIn("vram_canvas", result.images[0].info)
        self.assertIn("krea2_smart_finish", result.images[0].info)
        self.assertEqual(module.state.job_no, module.state.job_count)
        self.assertEqual(
            p.extra_generation_params["VRAM-Canvas"],
            "progressive base-detail-structure-consensus residual",
        )
        self.assertEqual(p.extra_generation_params["VRAM-Canvas base detail protection"], 6.0)
        self.assertEqual(p.extra_generation_params["VRAM-Canvas consensus floor"], 8.0)

    def test_quality_profiles_set_dense_detail_without_changing_positional_layout(self):
        module = load_script_module([], [])

        safe = module.VRAMCanvasHighres._quality_profile_values("Structure Safe")
        dense_4k = module.VRAMCanvasHighres._quality_profile_values(
            "Krea2 Dense Detail 4K"
        )
        texture_4k = module.VRAMCanvasHighres._quality_profile_values(
            "Krea2 Texture Rich 4K (Experimental)"
        )
        dense_8k = module.VRAMCanvasHighres._quality_profile_values(
            "Krea2 Dense Detail 8K"
        )
        phaseweave_4k = module.VRAMCanvasHighres._quality_profile_values(
            "Krea2 PhaseWeave 4K (Experimental)"
        )

        self.assertEqual(safe[0], 1)
        self.assertFalse(safe[-2])
        self.assertEqual(safe[-1], "consensus")
        self.assertEqual(dense_4k[0], 2)
        self.assertEqual(dense_4k[4:6], (0.16, 0.13))
        self.assertEqual(dense_8k[4:6], (0.12, 0.11))
        self.assertEqual(dense_4k[12:14], (1.0, 8.0))
        self.assertEqual(dense_8k[12:14], (0.8, 6.0))
        self.assertEqual(texture_4k[0:3], (2, 6, 6))
        self.assertEqual(texture_4k[4:6], (0.22, 0.18))
        self.assertEqual(texture_4k[7:12], (1.55, 40.0, 22.0, 1.5, 12.0))
        self.assertEqual(texture_4k[12:19], (1.6, 12.0, 1, 5, 10.0, 2.0, 4.0))
        self.assertEqual(texture_4k[19:23], (0.85, 1.4, 0.4, 10.0))
        self.assertTrue(dense_4k[-2])
        self.assertEqual(dense_4k[-1], "consensus")
        self.assertEqual(phaseweave_4k[0:3], (2, 6, 6))
        self.assertEqual(phaseweave_4k[4:6], (0.20, 0.16))
        self.assertTrue(phaseweave_4k[-2])
        self.assertEqual(phaseweave_4k[-1], "phase_weave")

        quick_4k = module.VRAMCanvasHighres._quick_profile_values(
            4096, "Krea2 Dense Detail 4K"
        )
        quick_8k = module.VRAMCanvasHighres._quick_profile_values(
            8192, "Krea2 Dense Detail 8K"
        )
        quick_texture_4k = module.VRAMCanvasHighres._quick_profile_values(
            4096, "Krea2 Texture Rich 4K (Experimental)"
        )
        quick_phaseweave_4k = module.VRAMCanvasHighres._quick_profile_values(
            4096, "Krea2 PhaseWeave 4K (Experimental)"
        )
        self.assertEqual(
            quick_4k[:4], (4096, 0, 0, "Krea2 Dense Detail 4K")
        )
        self.assertEqual(quick_4k[4], 0)
        self.assertEqual(quick_4k[5:], dense_4k)
        self.assertEqual(
            quick_8k[:4], (8192, 0, 0, "Krea2 Dense Detail 8K")
        )
        self.assertEqual(quick_8k[4], 1024)
        self.assertEqual(quick_8k[5:], dense_8k)
        self.assertEqual(
            quick_texture_4k[:4],
            (4096, 0, 0, "Krea2 Texture Rich 4K (Experimental)"),
        )
        self.assertEqual(quick_texture_4k[4], 896)
        self.assertEqual(quick_texture_4k[5:], texture_4k)
        self.assertEqual(
            quick_phaseweave_4k[:4],
            (4096, 0, 0, "Krea2 PhaseWeave 4K (Experimental)"),
        )
        self.assertEqual(quick_phaseweave_4k[4], 896)
        self.assertEqual(quick_phaseweave_4k[5:], phaseweave_4k)
        quick_8k_with_summary = (
            module.VRAMCanvasHighres._quick_profile_values_with_summary(
                8192,
                "Krea2 Dense Detail 8K",
                0,
                True,
            )
        )
        self.assertEqual(quick_8k_with_summary[:-1], quick_8k)
        self.assertIn("4K承認後", quick_8k_with_summary[-1])
        self.assertIn("長辺 8192 px", quick_8k_with_summary[-1])
        with self.assertRaisesRegex(ValueError, "4096 or 8192"):
            module.VRAMCanvasHighres._quick_profile_values(
                2048, "Structure Safe"
            )

    def test_krea2_detail_prompt_mode_rejects_wrong_model_before_tiles(self):
        module = load_script_module([], [])
        wrong = SimpleNamespace(sd_model=SimpleNamespace(model_config=SimpleNamespace()))

        with self.assertRaisesRegex(ValueError, "requires a loaded Krea2 checkpoint"):
            module.VRAMCanvasHighres._require_krea2_model(wrong)

        Krea2 = type("Krea2", (), {})
        model = Krea2()
        model.model_config = Krea2()
        module.VRAMCanvasHighres._require_krea2_model(
            SimpleNamespace(sd_model=model)
        )

    def test_krea2_detail_prompt_reaches_tiles_once_and_is_restored(self):
        snapshots = []
        module = load_script_module(snapshots, [])
        p = make_processing()
        base = (
            "wide anime illustration of a snowy railway station, a small white-haired "
            "woman reading a newspaper, reflective floor, dark wooden interior."
        )
        p.prompt = base
        Krea2 = type("Krea2", (), {})
        p.sd_model = Krea2()
        p.sd_model.model_config = Krea2()

        result = module.VRAMCanvasHighres().run(
            p,
            128,
            0,
            0,
            8,
            0,
            5.5,
            2.0,
            1,
            2,
            4,
            0.035,
            0.12,
            0.08,
            12,
            1.0,
            32,
            18,
            6,
            8,
            0,
            8,
            False,
            True,
            False,
            0.8,
            True,
            0.75,
            1.0,
            0.6,
            5.0,
        )

        self.assertEqual(len(snapshots), 1)
        effective = snapshots[0]["prompt"]
        self.assertTrue(effective.startswith(f"{base} Preserve"))
        self.assertEqual(effective.count("Preserve the exact source image"), 1)
        self.assertEqual(effective[: len(base)], base)
        self.assertEqual(p.prompt, base)
        metadata = json.loads(result.images[0].info["krea2_smart_highres"])
        self.assertTrue(metadata["detail_prompt_appended"])

    def test_global_save_receives_machine_readable_metadata(self):
        snapshots = []
        saved_images = []
        module = load_script_module(snapshots, saved_images)
        p = make_processing()
        p.save_samples = lambda: True

        module.VRAMCanvasHighres().run(
            p,
            128,
            0,
            0,
            8,
            0,
            5.5,
            2.0,
            1,
            2,
            4,
            0.035,
            0.12,
            0.08,
            12,
            1.0,
            32,
            18,
            6,
            8,
            0,
            8,
            False,
        )

        self.assertEqual(len(saved_images), 1)
        saved, _, kwargs = saved_images[0]
        self.assertEqual(saved.size, (128, 64))
        self.assertIn("vram_canvas", kwargs["existing_info"])
        self.assertIn("krea2_smart_highres", kwargs["existing_info"])
        self.assertIn("krea2_smart_finish", kwargs["existing_info"])

    def test_random_seed_sentinel_is_resolved_for_output_and_restored(self):
        snapshots = []
        module = load_script_module(snapshots, [])
        p = make_processing()
        p.seed = -1
        p.subseed = -1

        result = run_small(module, p)

        self.assertEqual(result.seed, 777)
        self.assertEqual(result.all_seeds, [777])
        self.assertEqual(p.seed, -1)
        self.assertEqual(p.subseed, -1)

    def test_phaseweave_gui_uses_two_completed_phases_and_writes_metadata(self):
        snapshots = []
        module = load_script_module(snapshots, [])
        p = make_processing()

        result = run_small(
            module,
            p,
            phase_count=2,
            merge_mode="phase_weave",
        )

        self.assertEqual(len(snapshots), 2)
        self.assertTrue(
            all(
                call["override_settings"]["img2img_fix_steps"]
                for call in snapshots
            )
        )
        manifest = json.loads(result.images[0].info["vram_canvas"])
        phaseweave = json.loads(result.images[0].info["krea2_phaseweave"])
        self.assertEqual(manifest["merge_mode"], "phase_weave")
        self.assertEqual(manifest["grid_layout"], "uniform_virtual_edge_balanced")
        self.assertEqual(manifest["grid_padding_mode"], "edge")
        self.assertTrue(manifest["exact_img2img_steps"])
        self.assertEqual(
            manifest["exact_img2img_steps_scope"], "internal_tiles_only"
        )
        self.assertEqual(
            manifest["stage_reports"][0]["consensus_stats"]["merge_mode"],
            "phase_weave",
        )
        self.assertEqual(phaseweave["profile_key"], "phaseweave_4k")
        self.assertEqual(phaseweave["merge_mode"], "phase_weave")
        self.assertEqual(phaseweave["grid_layout"], "uniform_virtual_edge_balanced")
        self.assertEqual(phaseweave["grid_origin"], manifest["grid_origin"])
        self.assertEqual(phaseweave["selection_mode"], "ternary_input_fallback")
        self.assertTrue(phaseweave["input_fallback"])
        self.assertAlmostEqual(phaseweave["selection_margin"], 0.03)
        self.assertEqual(phaseweave["island_min_area"], 3000)
        self.assertEqual(phaseweave["input_island_min_area"], 512)
        self.assertEqual(phaseweave["fidelity_guided_radius"], 8)
        self.assertEqual(phaseweave["feather_radius"], 5)
        self.assertAlmostEqual(phaseweave["support_mix"], 0.10)
        self.assertFalse(hasattr(p, "override_settings"))

    def test_8k_requires_an_approved_4k_source_before_processing(self):
        snapshots = []
        module = load_script_module(snapshots, [])
        p = make_processing()

        with self.assertRaisesRegex(ValueError, "approved 4K input"):
            run_small(module, p, final_long_edge=8192)

        self.assertEqual(snapshots, [])

    def test_interruption_never_returns_an_unfinished_tile(self):
        snapshots = []
        module = load_script_module(snapshots, [])
        p = make_processing()
        original_process = module.processing.process_images

        def interrupting_process(processing):
            result = original_process(processing)
            module.state.interrupted = True
            return result

        module.processing.process_images = interrupting_process

        with self.assertRaisesRegex(RuntimeError, "unfinished tile was not returned"):
            run_small(module, p)

        self.assertEqual(p.seed, 123)
        self.assertEqual(p.prompt, "test prompt")
        self.assertFalse(hasattr(p, "override_settings"))

    def test_krea2_mode_rejects_per_run_checkpoint_override(self):
        module = load_script_module([], [])
        Krea2 = type("Krea2", (), {})
        model = Krea2()
        model.model_config = Krea2()
        p = SimpleNamespace(
            sd_model=model,
            override_settings={"sd_model_checkpoint": "other.safetensors"},
        )

        with self.assertRaisesRegex(ValueError, "does not allow per-run"):
            module.VRAMCanvasHighres._require_krea2_model(p)

    def test_rejects_batch_and_inpaint_modes(self):
        module = load_script_module([], [])
        p = make_processing()
        p.batch_size = 2
        with self.assertRaisesRegex(ValueError, "Batch"):
            module.VRAMCanvasHighres._validate_run(
                p,
                max_stage_scale=2.0,
                phase_count=1,
                minimum_steps=2,
                maximum_steps=4,
                detail_knee=0.035,
                coarse_denoise=0.12,
                final_denoise=0.08,
                low_pass_radius=12,
                detail_gain=1.0,
                max_detail_delta=32,
                structure_sigma=18,
                base_detail_sigma=6,
                consensus_sigma=8,
                novel_detail_gain=0,
                novel_detail_max_delta=8,
            )

    def test_rejects_invalid_texture_rich_novel_detail_controls(self):
        module = load_script_module([], [])
        p = make_processing()
        with self.assertRaisesRegex(ValueError, "Inner Radius"):
            module.VRAMCanvasHighres._validate_run(
                p,
                max_stage_scale=2.0,
                phase_count=2,
                minimum_steps=6,
                maximum_steps=6,
                detail_knee=0.025,
                coarse_denoise=0.22,
                final_denoise=0.18,
                low_pass_radius=10,
                detail_gain=1.55,
                max_detail_delta=40,
                structure_sigma=22,
                base_detail_sigma=1.5,
                consensus_sigma=12,
                novel_detail_gain=1.6,
                novel_detail_max_delta=12,
                novel_detail_inner_radius=5,
                novel_detail_outer_radius=5,
            )


if __name__ == "__main__":
    unittest.main()
