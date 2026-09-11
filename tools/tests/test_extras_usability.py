import importlib
import json
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import gradio as gr
import numpy as np
from PIL import Image

from modules import grain_cleaner as grain
from modules import shared_init
from modules.extras_workflow import bind_workflow, plan_html, planned_size, result_html
from modules.grain_selection import make_reference_preview, select_reference
from modules.scripts_postprocessing import PostprocessedImage
from scripts.postprocessing_color_flatten import (
    FAST_MODE,
    GRADIENT_MODE,
    SMART_MODE,
    SUPERPIXEL_MODE,
    ScriptPostprocessingColorFlatten,
    mode_visibility,
)
from scripts.postprocessing_color_mura_checker import ScriptPostprocessingColorMuraChecker
from scripts.postprocessing_grain_cleaner import ScriptPostprocessingGrainCleaner

shared_init.initialize()


class ExtrasUsabilityTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.fromarray(np.random.default_rng(5).integers(120, 136, (128, 160, 3), dtype=np.uint8))

    def test_cache_reuses_analysis_but_not_render_and_invalidates_changed_content(self):
        cache = grain.GrainAnalysisCache()
        reference = self.image.copy()
        first = grain.GrainSettings()
        second = replace(first, strength=0.9, preserve_detail=0.3, chroma_strength=0.4)
        with patch.object(grain, "analyze_grain", wraps=grain.analyze_grain) as analyze:
            grain.clean_grain(self.image, first, cache=cache)
            out, report, _ = grain.clean_grain(self.image.copy(), second, cache=cache)
            self.assertTrue(report["analysis_cache_hit"])
            self.assertEqual(analyze.call_count, 1)
            self.image.putpixel((0, 0), (0, 0, 0))
            grain.clean_grain(self.image, second, cache=cache)
            self.assertEqual(analyze.call_count, 2)
            grain.clean_grain(self.image, replace(second, grain_scale=1.5), cache=cache)
            self.assertEqual(analyze.call_count, 3)
        cache.clear()
        self.assertIsNone(cache.analysis)
        fresh, _, _ = grain.clean_grain(reference, second)
        np.testing.assert_array_equal(out, fresh)

    def test_cache_does_not_reuse_masks_or_roi_and_coordinates_are_allocated_once(self):
        cache = grain.GrainAnalysisCache()
        with patch.object(grain.np, "meshgrid", wraps=np.meshgrid) as grids:
            grain.clean_grain(self.image, cache=cache)
            self.assertEqual(grids.call_count, 1)
        mask = np.zeros((128, 160), np.uint8)
        mask[:, :80] = 255
        out, report, _ = grain.clean_grain(self.image, cache=cache, protect_mask=Image.fromarray(mask))
        self.assertTrue(report["analysis_cache_hit"])
        np.testing.assert_array_equal(np.asarray(out)[:, :80], np.asarray(self.image)[:, :80])
        settings = grain.GrainSettings(mode="sample")
        grain.clean_grain(self.image, settings, sample_roi=(0, 0, 64, 64), cache=cache)
        _, report, _ = grain.clean_grain(self.image, settings, sample_roi=(64, 64, 64, 64), cache=cache)
        self.assertFalse(report["analysis_cache_hit"])

    def test_checker_skips_when_no_output_is_requested(self):
        with patch("scripts.postprocessing_color_mura_checker.analyze_chroma_mura") as analyze:
            pp = PostprocessedImage(self.image)
            ScriptPostprocessingColorMuraChecker().process(pp, color_mura_outputs=[], color_mura_add_metrics=False)
        analyze.assert_not_called()
        self.assertEqual(pp.info, {})

    def test_only_selected_mode_controls_are_visible(self):
        for mode, expected in (
            (SMART_MODE, (True, False, False, False, False)),
            (FAST_MODE, (False, True, False, False, True)),
            (SUPERPIXEL_MODE, (False, False, True, False, True)),
            (GRADIENT_MODE, (False, False, False, True, True)),
        ):
            self.assertEqual(tuple(v["visible"] for v in mode_visibility(mode)), expected)

    def test_summary_retains_json_and_explains_missing_diagnostics(self):
        _, report, views = grain.clean_grain(self.image, grain.GrainSettings(strength=0), diagnostics=True)
        self.assertEqual(views, {})
        rendered = result_html({"Grain Cleaner": json.dumps(report), "custom": "<script>"})
        self.assertIn("強度が0", rendered)
        self.assertIn("診断画像を省略", rendered)
        self.assertIn("<details>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertNotIn("<script>", rendered)

    def test_plan_obeys_order_and_skips_disabled_operations(self):
        values = dict(
            upscale_enabled=True,
            upscaler_1_name="Lanczos",
            upscale_mode=0,
            upscale_by=4,
            max_side_length=0,
            upscale_to_width=1024,
            upscale_to_height=1024,
            upscale_crop=False,
        )
        self.assertEqual(planned_size((1024, 1536), values), (4096, 6144))
        self.assertEqual(planned_size((1024, 1536), dict(values, max_side_length=2048)), (1368, 2048))
        self.assertEqual(planned_size((1024, 1536), dict(values, upscale_mode=1, upscale_crop=True)), (1024, 1024))
        operations = [
            ("Upscale", values),
            ("Grain Cleaner", {"enable": True}),
            ("Color Flatten", {"enable": False}),
            (
                "Color Mura Checker",
                {"color_mura_enabled": True, "color_mura_outputs": [], "color_mura_add_metrics": False},
            ),
        ]
        summary = plan_html((1024, 1536), 0, operations)
        self.assertIn("Upscale → Grain Cleaner", summary)
        self.assertNotIn("Color Flatten", summary)
        self.assertNotIn("Color Mura Checker", summary)
        self.assertIn("4096 × 6144", summary)

    def test_preview_rectangle_maps_to_original_and_draws_without_mutating_master(self):
        image = Image.new("RGB", (1536, 1024), "white")
        preview, master, size, _, _, _ = make_reference_preview(image)
        self.assertEqual(preview.size, (768, 512))
        selected, corner, roi, _ = select_reference(master, size, None, False, SimpleNamespace(index=(100, 100)))
        self.assertEqual(roi, "")
        selected, corner, roi, _ = select_reference(master, size, corner, False, SimpleNamespace(index=(200, 200)))
        self.assertEqual(roi, "200, 200, 202, 202")
        self.assertIsNone(corner)
        self.assertNotEqual(selected.tobytes(), master.tobytes())
        self.assertEqual(master.getpixel((100, 100)), (255, 255, 255))
        _, _, roi, message = select_reference(master, size, (1, 1), False, SimpleNamespace(index=(2, 2)))
        self.assertEqual(roi, "")
        self.assertIn("小さすぎ", message)
        with patch.object(gr, "Warning") as warning:
            select_reference(master, size, None, True, SimpleNamespace(index=(20, 20)))
        warning.assert_called_once()

    def test_event_wrappers_bind_solo_and_source_summary_without_image_slider_inputs(self):
        importlib.import_module("modules.gradio_extensions")
        with gr.Blocks() as demo:
            values = dict(
                upscale_enabled=True,
                upscaler_1_name="Lanczos",
                upscale_mode=0,
                upscale_by=4,
                max_side_length=0,
                upscale_to_width=1024,
                upscale_to_height=1024,
                upscale_crop=False,
            )
            controls = {
                key: gr.Checkbox(value)
                if isinstance(value, bool)
                else gr.Textbox(value)
                if isinstance(value, str)
                else gr.Number(value)
                for key, value in values.items()
            }
            upscale = SimpleNamespace(name="Upscale", controls=controls)
            cleaner = ScriptPostprocessingGrainCleaner()
            cleaner.controls = cleaner.ui()
            flatten = ScriptPostprocessingColorFlatten()
            flatten.controls = flatten.ui()
            checker = ScriptPostprocessingColorMuraChecker()
            checker.controls = checker.ui()
            scripts = [upscale, cleaner, flatten, checker]
            runner = SimpleNamespace(scripts_in_preferred_order=lambda: scripts, image_changed=unittest.mock.Mock())
            source, tab, plan, solo = gr.Image(type="pil"), gr.State(0), gr.HTML(), gr.Button()
            bind_workflow(runner, source, tab, plan, solo)
        refresh = next(fn for fn in demo.fns.values() if fn.name == "refresh_source")
        result = refresh.fn(self.image, "auto", 0, *(c.value for c in refresh.inputs[3:]))
        self.assertEqual(result[0], (160, 128))
        self.assertIn("640 × 512", result[1])
        runner.image_changed.assert_called_once()
        update = next(fn for fn in demo.fns.values() if fn.name == "update_summary")
        self.assertFalse(any(isinstance(component, gr.Image) for component in update.inputs))
        self.assertNotIn(cleaner.controls["strength"], update.inputs)
        targets = [
            upscale.controls["upscale_enabled"],
            cleaner.controls["enable"],
            flatten.controls["enable"],
            checker.controls["color_mura_enabled"],
        ]
        preset = next(fn for fn in demo.fns.values() if fn.outputs == targets)
        self.assertEqual(preset.fn(), (False, True, False, False))


if __name__ == "__main__":
    unittest.main()
