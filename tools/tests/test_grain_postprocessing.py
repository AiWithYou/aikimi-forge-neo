import json
import unittest
import warnings

import gradio as gr
import numpy as np
from PIL import Image

from modules import shared_init
from modules.grain_cleaner import GrainSettings, clean_grain
from modules.scripts_postprocessing import PostprocessedImage
from scripts.postprocessing_grain_cleaner import ScriptPostprocessingGrainCleaner

shared_init.initialize()


class GrainPostprocessingTests(unittest.TestCase):
    def test_controls_and_default_disabled(self):
        with warnings.catch_warnings(), gr.Blocks():
            warnings.simplefilter("ignore", DeprecationWarning)
            controls = ScriptPostprocessingGrainCleaner().ui()
        self.assertFalse(controls["enable"].value)
        self.assertEqual(controls["strength"].value, 0.5)
        pp = PostprocessedImage(Image.new("RGB", (16, 16)))
        ScriptPostprocessingGrainCleaner().process(pp)
        self.assertEqual(pp.info, {})

    def test_pipeline_pixels_and_diagnostics(self):
        data = np.random.default_rng(11).integers(120, 136, (128, 128, 3), dtype=np.uint8)
        image = Image.fromarray(data)
        expected, _, _ = clean_grain(image, GrainSettings(strength=0.75))
        pp = PostprocessedImage(image)
        ScriptPostprocessingGrainCleaner().process(pp, enable=True, strength=0.75, diagnostics=True)
        np.testing.assert_array_equal(pp.image, expected)
        self.assertEqual(len(pp.extra_images), 4)
        self.assertTrue(all(extra.disable_processing for extra in pp.extra_images))
        self.assertEqual(json.loads(pp.info["Grain Cleaner"])["settings"]["strength"], 0.75)

    def test_invalid_roi_surfaces_error(self):
        pp = PostprocessedImage(Image.new("RGB", (128, 128)))
        with self.assertRaises(gr.Error):
            ScriptPostprocessingGrainCleaner().process(pp, enable=True, mode="sample", sample_roi="oops")


if __name__ == "__main__":
    unittest.main()
