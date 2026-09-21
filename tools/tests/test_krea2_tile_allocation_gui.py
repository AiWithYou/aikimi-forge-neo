"""The actual tile/merge loop preserves its base when diffusion is skipped."""

import json
import unittest
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from modules_forge.jev_sparse import krea2, krea2_jobs
from tools.tests.test_vram_canvas_gui import load_script_module, make_processing, run_small


class Krea2TileAllocationGuiTests(unittest.TestCase):
    def test_cold_model_is_prepared_before_architecture_validation(self):
        module = load_script_module([], [])
        p = make_processing()
        p.sd_model = None
        prepared = []
        engine = type("Krea2", (), {})()
        engine.model_config = type("Krea2", (), {})()

        def prepare(request):
            prepared.append(request)
            request.sd_model = engine

        module.processing.manage_model_and_prompt_cache = prepare
        module.VRAMCanvasHighres._require_krea2_model(p)
        self.assertEqual(prepared, [p])
        p.override_settings = {"sd_model_checkpoint": "other-model"}
        with self.assertRaises(ValueError):
            module.VRAMCanvasHighres._require_krea2_model(p)
        self.assertEqual(prepared, [p])

    def test_all_flat_tiles_skip_diffusion_and_preserve_overlap_coverage(self):
        calls, saved = [], []
        module = load_script_module(calls, saved)
        p = make_processing()
        original = p.init_images[0].copy()

        def processed(_p, images, seed=-1, info=""):
            return SimpleNamespace(images=images, info=info, infotexts=[info], seed=seed, all_seeds=[seed])

        module.processing.Processed = processed
        with TemporaryDirectory() as directory:
            session = krea2_jobs.Session(krea2.Options(tile_mode="rules"), directory)
            with (
                patch.object(krea2_jobs, "processing_options", return_value=session.options),
                patch.object(krea2_jobs, "Session", return_value=session),
                patch.object(krea2_jobs, "cancelled", return_value=False),
                patch.object(krea2_jobs, "krea_selected", return_value=True),
            ):
                result = run_small(module, p, final_long_edge=256, tile_size=384, phase_count=2)
            self.assertTrue(session.closed)
            self.assertIsNone(krea2_jobs.current_session())
            self.assertEqual(calls, [])
            self.assertTrue(
                np.array_equal(
                    np.asarray(result.images[0]), np.asarray(original.resize((256, 128), Image.Resampling.LANCZOS))
                )
            )
            metadata = json.loads(result.images[0].info["vram_canvas"])
            tiles = [tile for stage in metadata["stage_reports"] for tile in stage["tiles"]]
            self.assertGreater(len(tiles), 1)
            self.assertTrue(all(tile["steps"] == 0 and tile["diffusion_skipped"] for tile in tiles))
            self.assertEqual((p.width, p.height, p.steps, p.seed), (64, 32, 8, 123))

    def test_cancelled_allocation_does_not_run_a_tile_or_leak_scope(self):
        calls, saved = [], []
        module = load_script_module(calls, saved)
        p = make_processing()
        with TemporaryDirectory() as directory:
            session = krea2_jobs.Session(krea2.Options(tile_mode="rules"), directory)
            with (
                patch.object(krea2_jobs, "processing_options", return_value=session.options),
                patch.object(krea2_jobs, "Session", return_value=session),
                patch.object(krea2_jobs, "cancelled", return_value=True),
                patch.object(krea2_jobs, "krea_selected", return_value=True),
            ):
                with self.assertRaises(RuntimeError):
                    run_small(module, p)
            self.assertTrue(session.closed)
            self.assertIsNone(krea2_jobs.current_session())
            self.assertEqual(calls, [])
            self.assertEqual((p.width, p.height, p.steps, p.seed), (64, 32, 8, 123))


if __name__ == "__main__":
    unittest.main()
