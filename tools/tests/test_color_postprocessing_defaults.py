import sys
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import gradio as gr
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ROOT / "modules_forge" / "packages"
if str(PACKAGES) not in sys.path:
    sys.path.insert(0, str(PACKAGES))

_ORIGINAL_ARGV = sys.argv[:]
sys.argv = [sys.argv[0]]

import modules.shared_init as shared_init

shared_init.initialize()
sys.argv = _ORIGINAL_ARGV

from modules.color_flatten import (
    FAST_MODE,
    GRADIENT_MODE,
    SMART_MODE,
    SUPERPIXEL_MODE,
)
from modules.scripts_postprocessing import PostprocessedImage
from scripts.postprocessing_color_flatten import ScriptPostprocessingColorFlatten
from scripts.postprocessing_color_mura_checker import (
    ScriptPostprocessingColorMuraChecker,
    compute_mura,
)


class ColorPostprocessingDefaultsTests(unittest.TestCase):
    def test_postprocessing_public_name_remains_compatible(self):
        self.assertEqual(ScriptPostprocessingColorFlatten.name, "Color Flatten")

    def test_smooth_gradient_controls_are_appended_compatibly(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with gr.Blocks():
                controls = ScriptPostprocessingColorFlatten().ui()

        self.assertEqual(
            list(controls),
            [
                "enable",
                "mode",
                "strength",
                "edge_protect",
                "despeckle",
                "analysis_long_edge",
                "mean_shift_sp",
                "mean_shift_sr",
                "n_segments",
                "compactness",
                "gradient_radius",
                "gradient_detail_threshold",
            ],
        )
        self.assertEqual(
            [value for _, value in controls["mode"].choices],
            [SMART_MODE, FAST_MODE, SUPERPIXEL_MODE, GRADIENT_MODE],
        )
        self.assertEqual(controls["gradient_radius"].value, 12.0)
        self.assertEqual(controls["gradient_detail_threshold"].value, 8.0)

    def test_smart_finish_runs_by_default_and_clean_image_is_noop(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorFlatten().process(pp)

        self.assertEqual(pp.info["Krea2 Smart Finish"], SMART_MODE)
        self.assertEqual(pp.info["Krea2 Smart Finish strength"], 0.8)
        self.assertFalse(pp.info["Krea2 Smart Finish chroma applied"])
        self.assertFalse(pp.info["Krea2 Smart Finish despeckle"])

    def test_manual_mode_metadata_is_truthful_and_reproducible(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorFlatten().process(
            pp, mode=FAST_MODE, strength=0.0, edge_protect=False
        )

        self.assertEqual(pp.info["Color Flatten"], FAST_MODE)
        self.assertEqual(pp.info["Color Flatten strength"], 0.0)
        self.assertFalse(pp.info["Color Flatten edge protect"])
        self.assertNotIn("Krea2 Smart Finish chroma p95 before", pp.info)

    def test_smooth_gradient_mode_records_its_own_settings(self):
        pp = PostprocessedImage(Image.new("RGB", (32, 32), (120, 100, 90)))

        with patch(
            "scripts.postprocessing_color_flatten.color_flatten_pil",
            return_value=pp.image.copy(),
        ) as flatten:
            ScriptPostprocessingColorFlatten().process(
                pp,
                mode=GRADIENT_MODE,
                strength=1.0,
                edge_protect=True,
                gradient_radius=12.0,
                gradient_detail_threshold=8.0,
            )

        self.assertEqual(pp.info["Color Flatten"], GRADIENT_MODE)
        self.assertEqual(pp.info["Color Flatten strength"], 1.0)
        self.assertTrue(pp.info["Color Flatten edge protect"])
        self.assertEqual(pp.info["Smooth Gradient radius"], 12.0)
        self.assertEqual(pp.info["Smooth Gradient detail threshold"], 8.0)
        self.assertFalse(pp.info["Smooth Gradient despeckle applied"])
        self.assertEqual(pp.info["Smooth Gradient despeckle pixels"], 0)
        self.assertEqual(
            pp.info["Smooth Gradient despeckle reason"], "despeckle disabled"
        )
        self.assertNotIn("Krea2 Smart Finish", pp.info)
        self.assertEqual(flatten.call_args.args[-2:], (12.0, 8.0))

    def test_legacy_manual_process_positional_arguments_remain_valid(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorFlatten().process(
            pp,
            True,
            FAST_MODE,
            0.0,
            False,
            False,
            1536,
            12,
            24,
            1200,
            12.0,
        )

        self.assertEqual(pp.info["Color Flatten"], FAST_MODE)
        self.assertEqual(pp.info["Color Flatten strength"], 0.0)

    def test_color_mura_checker_defaults_to_metrics_only(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorMuraChecker().process(pp)

        self.assertIn("Color mura summary", pp.info)
        self.assertEqual(
            pp.info["Color mura metric"], "Lab chroma-only delta (L excluded)"
        )
        self.assertEqual(len(pp.extra_images), 0)

    def test_color_mura_checker_creates_requested_review_outputs(self):
        pp = PostprocessedImage(Image.new("RGB", (16, 16), (120, 100, 90)))

        ScriptPostprocessingColorMuraChecker().process(
            pp, color_mura_outputs=("Overlay", "Heatmap")
        )

        self.assertEqual(len(pp.extra_images), 2)
        self.assertEqual(pp.extra_images[0].nametags[-1], "mura-overlay")
        self.assertEqual(pp.extra_images[1].nametags[-1], "mura-heatmap")

    def test_color_mura_checker_allocates_only_requested_review_view(self):
        image = Image.new("RGB", (16, 16), (120, 100, 90))

        _, heatmap, overlay, _, _ = compute_mura(image, requested_views={"Overlay"})
        self.assertIsNone(heatmap)
        self.assertIsNotNone(overlay)

        _, heatmap, overlay, _, _ = compute_mura(image, requested_views={"Heatmap"})
        self.assertIsNotNone(heatmap)
        self.assertIsNone(overlay)


if __name__ == "__main__":
    unittest.main()
