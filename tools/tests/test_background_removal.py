import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import shared_init
from modules.background_removal import (
    DEFAULT_MODEL,
    MODEL_FILES,
    MODELS,
    BackgroundRemover,
    apply_alpha,
    download_model,
)
from modules.extras_workflow import plan_html
from modules.scripts_postprocessing import PostprocessedImage
from scripts.postprocessing_background_removal import ScriptPostprocessingBackgroundRemoval

shared_init.initialize()


class BackgroundRemovalTests(unittest.TestCase):
    def test_disabled_ui_does_not_load_or_download(self):
        script = ScriptPostprocessingBackgroundRemoval()
        with patch.object(script.remover, "_load") as load, gr.Blocks():
            controls = script.ui()
            self.assertFalse(controls["enable"].value)
            script.process(PostprocessedImage(Image.new("RGB", (8, 8))))
        load.assert_not_called()
        self.assertEqual(controls["model_id"].value, DEFAULT_MODEL)
        self.assertNotIn("Background Removal", plan_html(None, 0, [(script.name, {"enable": False})]))
        self.assertIn("Background Removal", plan_html(None, 0, [(script.name, {"enable": True})]))

    def test_existing_complete_model_never_contacts_hub(self):
        with tempfile.TemporaryDirectory() as root, patch("huggingface_hub.snapshot_download") as download:
            directory = Path(root) / DEFAULT_MODEL / MODELS[DEFAULT_MODEL].revision
            directory.mkdir(parents=True)
            for name in MODEL_FILES:
                (directory / name).touch()
            self.assertEqual(download_model(DEFAULT_MODEL, root), directory)
        download.assert_not_called()

    def test_missing_model_download_is_selected_and_revision_pinned(self):
        with tempfile.TemporaryDirectory() as root, patch("huggingface_hub.snapshot_download") as download:
            download_model("birefnet", root)
        self.assertEqual(download.call_args.kwargs["repo_id"], MODELS["birefnet"].repository)
        self.assertEqual(download.call_args.kwargs["revision"], MODELS["birefnet"].revision)
        self.assertEqual(download.call_args.kwargs["allow_patterns"], list(MODEL_FILES))

    def test_invalid_model_fails_before_download(self):
        with patch("huggingface_hub.snapshot_download") as download, self.assertRaises(ValueError):
            download_model("unknown", "unused")
        download.assert_not_called()

    def test_alpha_multiplies_original_transparency_and_preserves_rgb(self):
        image = Image.new("RGBA", (3, 1), (10, 20, 30, 255))
        image.putalpha(Image.fromarray(np.array([[0, 128, 255]], dtype=np.uint8)))
        result, mask = apply_alpha(image, Image.new("L", image.size, 128))
        self.assertEqual(list(mask.getdata()), [0, 64, 128])
        self.assertEqual(result.convert("RGB").tobytes(), image.convert("RGB").tobytes())
        self.assertEqual(list(image.getchannel("A").getdata()), [0, 128, 255])

    def test_inference_restores_dimensions_soft_alpha_and_cpu_ownership(self):
        model = Mock(return_value=[torch.zeros(1, 1, 8, 8)])
        remover = BackgroundRemover()
        with patch.object(remover, "_load", return_value=model):
            result, mask = remover.remove(Image.new("RGB", (13, 7)), DEFAULT_MODEL, "unused", torch.device("cpu"))
        self.assertEqual(result.size, (13, 7))
        self.assertEqual(mask.getextrema(), (128, 128))
        self.assertEqual(model.to.call_args.kwargs, {"device": "cpu", "dtype": torch.float32})

    def test_failure_offloads_model(self):
        model = Mock(side_effect=RuntimeError("inference failed"))
        remover = BackgroundRemover()
        with (
            patch.object(remover, "_load", return_value=model),
            self.assertRaisesRegex(RuntimeError, "inference failed"),
        ):
            remover.remove(Image.new("RGB", (8, 8)), DEFAULT_MODEL, "unused", torch.device("cpu"))
        self.assertEqual(model.to.call_args.kwargs, {"device": "cpu", "dtype": torch.float32})

    def test_mask_is_separate_and_exempt_from_following_operations(self):
        script = ScriptPostprocessingBackgroundRemoval()
        pp = PostprocessedImage(Image.new("RGB", (8, 8)))
        with (
            patch.object(script.remover, "remove", return_value=(Image.new("RGBA", (8, 8)), Image.new("L", (8, 8)))),
            patch("backend.memory_management.unload_all_models"),
            patch("modules.devices.torch_gc"),
        ):
            script.process(pp, enable=True, output_mask=True)
        self.assertEqual(pp.image.mode, "RGBA")
        self.assertEqual(pp.extra_images[0].nametags, ["mask"])
        self.assertTrue(pp.extra_images[0].disable_processing)


if __name__ == "__main__":
    unittest.main()
