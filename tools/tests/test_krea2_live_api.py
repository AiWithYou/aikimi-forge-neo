"""Opt-in end-to-end smoke for a running Forge Krea2 checkpoint."""

import base64
import io
import json
import os
import unittest
from urllib import request as urlrequest

from PIL import Image, ImageStat


LIVE = os.environ.get("KREA2_LIVE_API_TEST") == "1"
API = os.environ.get("KREA2_LIVE_API", "http://127.0.0.1:7861").rstrip("/")
TIMEOUT = int(os.environ.get("KREA2_LIVE_TIMEOUT", "900"))


def post_json(endpoint: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urlrequest.Request(
        f"{API}{endpoint}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def encode_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@unittest.skipUnless(LIVE, "set KREA2_LIVE_API_TEST=1 for the live checkpoint smoke")
class Krea2LiveApiTests(unittest.TestCase):
    def test_loaded_runtime_is_krea2(self):
        status = post_json("/sdapi/v1/forge-model-status/ensure-loaded", {})

        self.assertTrue(status["loaded"])
        self.assertEqual(status["architecture"], "backend.diffusion_engine.krea.Krea2")
        self.assertEqual(status["transformer"], "backend.nn.krea.SingleStreamDiT")
        self.assertEqual(status["inspection_errors"], [])
        module_names = [str(value).lower() for value in status["additional_modules"]]
        self.assertTrue(any("qwen_image_vae" in value for value in module_names))
        self.assertTrue(any("qwen3vl" in value for value in module_names))
        expected_quant = os.environ.get("KREA2_EXPECTED_QUANT")
        if expected_quant:
            formats = status["quantization"]["formats"]
            self.assertGreater(int(formats.get(expected_quant, 0)), 0)

    def test_txt2img_executes_transformer_text_encoder_and_vae(self):
        width = int(os.environ.get("KREA2_LIVE_WIDTH", "512"))
        height = int(os.environ.get("KREA2_LIVE_HEIGHT", "512"))
        response = post_json(
            "/sdapi/v1/txt2img",
            {
                "prompt": (
                    "a small red ceramic teapot on a pale wooden table, soft window "
                    "light, clean product photograph"
                ),
                "negative_prompt": "",
                "seed": 20260715,
                "sampler_name": "Euler",
                "scheduler": "Simple",
                "steps": int(os.environ.get("KREA2_LIVE_STEPS", "1")),
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 1.15,
                "width": width,
                "height": height,
                "n_iter": 1,
                "batch_size": 1,
                "send_images": True,
                "save_images": False,
            },
        )

        images = response.get("images") or []
        self.assertEqual(len(images), 1)
        encoded = images[0].split(",", 1)[-1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
            image = opened.convert("RGB")
        self.assertEqual(image.size, (width, height))
        variance = sum(ImageStat.Stat(image).var) / 3.0
        self.assertGreater(variance, 1.0)

    def test_img2img_uses_the_same_krea2_runtime(self):
        width = int(os.environ.get("KREA2_LIVE_WIDTH", "512"))
        height = int(os.environ.get("KREA2_LIVE_HEIGHT", "512"))
        gradient = Image.linear_gradient("L").resize((width, height))
        source = Image.merge(
            "RGB",
            (gradient, gradient.transpose(Image.Transpose.FLIP_TOP_BOTTOM), gradient),
        )
        response = post_json(
            "/sdapi/v1/img2img",
            {
                "init_images": [encode_png(source)],
                "prompt": "a softly lit ceramic teapot, clean product photograph",
                "negative_prompt": "",
                "seed": 20260716,
                "sampler_name": "Euler",
                "scheduler": "Simple",
                "steps": int(os.environ.get("KREA2_LIVE_I2I_STEPS", "2")),
                "denoising_strength": 0.5,
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 1.15,
                "width": width,
                "height": height,
                "n_iter": 1,
                "batch_size": 1,
                "send_images": True,
                "save_images": False,
            },
        )

        images = response.get("images") or []
        self.assertEqual(len(images), 1)
        encoded = images[0].split(",", 1)[-1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
            image = opened.convert("RGB")
        self.assertEqual(image.size, (width, height))
        self.assertGreater(sum(ImageStat.Stat(image).var) / 3.0, 1.0)


if __name__ == "__main__":
    unittest.main()
