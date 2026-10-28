"""Reference image attention weights and request validation."""

import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

from modules_forge.moss_image import core
from modules_forge.moss_image.inference import ComfyRuntime


def reference_request(**extra):
    return {"prompt": "make it blue", "mode": "img2img", "edit_method": "reference",
            "images": [Image.new("RGB", (512, 512))], "loras": [{"name": "edit.safetensors", "strength": 1}], **extra}


class ReferenceImageStrengthTests(unittest.TestCase):
    def test_reference_strength_is_validated_and_persisted(self):
        assert core.validate_request(reference_request())["image_strength"] == 1
        for strength in (0, 0.2, 1, 4):
            request = core.validate_request(reference_request(image_strength=strength))
            assert request["image_strength"] == strength
            internal = {**request, "images": ["reference-0.png"]}
            assert core.validate_request(internal, internal=True) == internal
        for invalid in (-0.01, 4.01, float("nan"), float("inf"), True, "1"):
            with self.assertRaises(ValueError):
                core.validate_request(reference_request(image_strength=invalid))
        with self.assertRaises(ValueError):
            core.validate_request({"prompt": "cat", "image_strength": 0.5})


    def test_image_token_bias_follows_expanded_vision_tokens(self):
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch = torch
        images = [torch.zeros(1, 64, 96, 3), torch.zeros(1, 96, 64, 3)]
        rows = [[(10, 1.0), (248056, 1.0), (11, 1.0), (248056, 1.0), (12, 1.0)]]
        clip = SimpleNamespace(
            tokenizer=SimpleNamespace(qwen35_9b=SimpleNamespace(tokenize_with_weights=lambda *a, **k: rows)),
            encode_from_tokens_scheduled=lambda tokens: [[torch.zeros(1, 15, 4), {}]],
        )
        result, count = runtime.encode_text(clip, "edit", images=images, image_strength=0.5)
        assert count == 15
        bias = result[0][1]["context_attention_bias"][0]
        torch.testing.assert_close(bias, torch.tensor([0] + [math.log(0.5)] * 6 + [0] + [math.log(0.5)] * 6 + [0]))


    def test_public_metadata_keeps_image_strength(self):
        from modules_forge.moss_image.api import public_result

        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        image = Path(folder.name) / "image.png"
        image.write_bytes(b"image")
        assert public_result({"status": "complete", "image_strength": 0.35, "image_path": str(image)})["image_strength"] == 0.35


    def test_zero_omits_vision_but_keeps_vae_references(self):
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.root = core.MODEL_ROOT
        runtime.torch = torch
        runtime.mapper = None
        latent = torch.zeros(1, 128, 32, 32)
        runtime.load_vae = Mock(return_value=SimpleNamespace(encode=lambda pixels: latent))
        runtime.load_clip = Mock(return_value=SimpleNamespace())
        runtime.encode_text = Mock(side_effect=lambda *a, **k: ([[torch.zeros(1, 4, 8), {}]], 4))
        runtime.offload = Mock()
        runtime.load_dit = Mock(return_value=SimpleNamespace(model=None))
        runtime.sample = Mock(return_value=latent)
        runtime.decode = Mock(return_value=Image.new("RGB", (512, 512)))
        request = core.validate_request(reference_request(image_strength=0, width=512, height=512))
        fake_module = SimpleNamespace(MOSSImageOstrisEditModelPatch=lambda: SimpleNamespace(patch=lambda model, **k: (model,)))
        with (
            patch.object(core, "asset_paths", return_value={"dit": "dit", "text_encoder": "te", "vae": "vae"}),
            patch("modules_forge.moss_image.inference.lora.apply", side_effect=lambda model, *args: (model, 0)),
            patch("modules_forge.moss_image.inference.lora.bind"),
            patch.dict(sys.modules, {"vendor.moss_image_ostris_edit": fake_module}),
        ):
            runtime.generate(request, lambda *args: None, lambda: None, images=request["images"])
        assert runtime.load_clip.call_args.kwargs == {}
        assert all(not call.kwargs for call in runtime.encode_text.call_args_list)
        assert runtime.sample.call_args.args[1][0][1]["reference_latents"][0] is latent
