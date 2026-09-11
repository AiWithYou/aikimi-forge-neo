"""Opt-in real-file and generation smoke for Anima-2.9B plus a 28-block LoRA."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from urllib import request as urlrequest
import unittest

from PIL import Image, ImageStat
from PIL.PngImagePlugin import PngInfo
from safetensors import safe_open

from modules_forge.anima_lora import (
    anima_lora_block_indices,
    convert_anima_lora_layout,
    detect_anima_lora_block_count,
)


LIVE = os.environ.get("ANIMA_29B_LIVE_API_TEST") == "1"
API = os.environ.get("ANIMA_29B_LIVE_API", "http://127.0.0.1:7862").rstrip("/")
TIMEOUT = int(os.environ.get("ANIMA_29B_LIVE_TIMEOUT", "1200"))
ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "Stable-diffusion" / "Anima-2.9B-preview-v1.safetensors"
TEXT_ENCODER = ROOT / "models" / "text_encoder" / "qwen_3_06b_base.safetensors"
VAE = ROOT / "models" / "VAE" / "qwen_image_vae.safetensors"
LORA = ROOT / "models" / "Lora" / "anima-turbo-lora-v0.2.safetensors"
OUTPUT = Path(
    os.environ.get(
        "ANIMA_29B_LIVE_OUTPUT",
        str(ROOT / "outputs" / "anima_29b_smoke" / "anima_29b_lora_smoke.png"),
    )
)


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


def get_json(endpoint: str) -> dict:
    with urlrequest.urlopen(f"{API}{endpoint}", timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(LIVE, "set ANIMA_29B_LIVE_API_TEST=1 for the live checkpoint smoke")
class Anima29BLiveApiTests(unittest.TestCase):
    def test_downloaded_assets_and_real_28_block_lora(self):
        expected_sizes = {
            CHECKPOINT: 5_843_204_206,
            TEXT_ENCODER: 1_192_135_096,
            VAE: 253_806_246,
            LORA: 148_902_616,
        }
        for path, expected_size in expected_sizes.items():
            self.assertTrue(path.is_file(), path)
            self.assertEqual(path.stat().st_size, expected_size, path)
        self.assertEqual(
            sha256(LORA),
            "1b55e40bdb1d0e5a78cb498f245fccfdaae97823265db957d2aabdcf4cd3caf1",
        )

        with safe_open(LORA, framework="pt", device="cpu") as handle:
            state_dict = {key: key for key in handle.keys()}

        self.assertEqual(detect_anima_lora_block_count(state_dict), 28)
        converted, report = convert_anima_lora_layout(state_dict, 40)
        self.assertEqual(report.direction, "28_to_40")
        self.assertGreater(report.duplicated_entries, 0)
        self.assertEqual(anima_lora_block_indices(converted), tuple(range(40)))

    def test_txt2img_loads_40_blocks_and_auto_converts_28_block_lora(self):
        width = int(os.environ.get("ANIMA_29B_LIVE_WIDTH", "768"))
        height = int(os.environ.get("ANIMA_29B_LIVE_HEIGHT", "1024"))
        response = post_json(
            "/sdapi/v1/txt2img",
            {
                "prompt": (
                    "1girl, solo, an adventurous astronomer on a moonlit observatory "
                    "terrace, flowing navy coat, brass telescope, starry sky, anime "
                    "coloring, detailed expressive eyes <lora:anima-turbo-lora-v0.2:1>"
                ),
                "negative_prompt": "worst quality, low quality, blurry, malformed hands",
                "seed": 20260814,
                "sampler_name": "Euler",
                "scheduler": "Simple",
                "steps": int(os.environ.get("ANIMA_29B_LIVE_STEPS", "8")),
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 1.0,
                "width": width,
                "height": height,
                "n_iter": 1,
                "batch_size": 1,
                "send_images": True,
                "save_images": False,
                # Keep this dedicated test server on Anima so the runtime
                # status below describes the request that just completed.
                "override_settings_restore_afterwards": False,
                "override_settings": {
                    "sd_model_checkpoint": CHECKPOINT.name,
                    "forge_additional_modules": [VAE.name, TEXT_ENCODER.name],
                    "forge_unet_storage_dtype": "Automatic",
                },
            },
        )

        images = response.get("images") or []
        self.assertEqual(len(images), 1)
        encoded = images[0].split(",", 1)[-1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
            image = opened.convert("RGB")
        self.assertEqual(image.size, (width, height))
        self.assertGreater(sum(ImageStat.Stat(image).var) / 3.0, 1.0)

        output_info = PngInfo()
        output_info.add_text("parameters", str(response.get("info", "")))
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        image.save(OUTPUT, pnginfo=output_info)
        self.assertTrue(OUTPUT.is_file())

        # Inspect the model that actually served the request without triggering
        # another load.
        status = get_json("/sdapi/v1/forge-model-status")
        self.assertTrue(status["loaded"])
        self.assertEqual(status["architecture"], "backend.diffusion_engine.anima.Anima")
        self.assertEqual(status["transformer"], "backend.nn.anima.Anima")
        self.assertTrue(str(status["checkpoint"]).endswith(CHECKPOINT.name))
        self.assertEqual(status["inspection_errors"], [])


if __name__ == "__main__":
    unittest.main()
