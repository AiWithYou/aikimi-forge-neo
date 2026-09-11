"""Opt-in real-model HyperWeave smoke against a running local Forge API."""

from __future__ import annotations

import base64
import io
import json
import os
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib import request as urlrequest

from PIL import Image, ImageDraw, ImageStat
from PIL.PngImagePlugin import PngInfo


LIVE = os.environ.get("HYPERWEAVE_RUN_GPU_TESTS") == "1"
INTERRUPT_LIVE = os.environ.get("HYPERWEAVE_RUN_INTERRUPT_TESTS") == "1"
API = os.environ.get(
    "HYPERWEAVE_LIVE_API", "http://127.0.0.1:7861"
).rstrip("/")
TIMEOUT = int(os.environ.get("HYPERWEAVE_LIVE_TIMEOUT", "3600"))
TEST_MAX_EDGE = int(os.environ.get("HYPERWEAVE_TEST_MAX_EDGE", "512"))
LIVE_TARGET = os.environ.get("HYPERWEAVE_LIVE_TARGET", "x2")
FACE_ROIS = os.environ.get("HYPERWEAVE_TEST_FACE_ROIS", "left").casefold()
ROOT = Path(__file__).resolve().parents[2]
INPUT = Path(
    os.environ.get(
        "HYPERWEAVE_TEST_IMAGE", r"H:\dl\image-cropped.png"
    )
)
OUTPUT = Path(
    os.environ.get(
        "HYPERWEAVE_TEST_OUTPUT",
        str(ROOT / "outputs" / "hyperweave_live_test.png"),
    )
)


def get_json(endpoint: str) -> object:
    with urlrequest.urlopen(f"{API}{endpoint}", timeout=TIMEOUT) as response:
        return json.loads(response.read().decode("utf-8"))


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


def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def decode_image(value: str) -> Image.Image:
    encoded = value.split(",", 1)[-1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as opened:
        return opened.copy()


def save_png_with_text(image: Image.Image, path: Path) -> None:
    pnginfo = PngInfo()
    for key, value in image.info.items():
        if isinstance(value, str):
            pnginfo.add_text(str(key), value)
    image.save(path, format="PNG", pnginfo=pnginfo)


def proxy_and_manual_roi(path: Path) -> tuple[Image.Image, Image.Image]:
    with Image.open(path) as opened:
        source = opened.copy()
    if TEST_MAX_EDGE > 0 and max(source.size) > TEST_MAX_EDGE:
        scale = TEST_MAX_EDGE / max(source.size)
        size = (round(source.width * scale), round(source.height * scale))
        proxy = source.resize(size, Image.Resampling.LANCZOS)
    else:
        proxy = source
        size = source.size
    mask = Image.new("RGBA", size, (0, 0, 0, 255))
    draw = ImageDraw.Draw(mask)
    # Left character: head/face context on the supplied 1664×2353 anime image.
    draw.ellipse(
        (
            round(size[0] * 0.17),
            round(size[1] * 0.20),
            round(size[0] * 0.40),
            round(size[1] * 0.43),
        ),
        fill=(255, 255, 255, 255),
    )
    if FACE_ROIS == "both":
        draw.ellipse(
            (
                round(size[0] * 0.60),
                round(size[1] * 0.20),
                round(size[0] * 0.89),
                round(size[1] * 0.46),
            ),
            fill=(255, 255, 255, 255),
        )
    return proxy, mask


def expected_live_target(size: tuple[int, int]) -> tuple[int, int]:
    width, height = size
    if LIVE_TARGET == "x2":
        return width * 2, height * 2
    if LIVE_TARGET == "x4":
        return width * 4, height * 4
    long_edge = {
        "4K long edge": 4096,
        "8K long edge": 8192,
    }.get(LIVE_TARGET)
    if long_edge is None:
        raise AssertionError(f"Unsupported live-test target: {LIVE_TARGET}")
    scale = long_edge / max(size)
    return max(1, round(width * scale)), max(1, round(height * scale))


def hyperweave_script_info() -> dict:
    entries = get_json("/sdapi/v1/script-info")
    matches = [
        entry
        for entry in entries
        if str(entry.get("name", "")).lower() == "hyperweave 4k/8k"
        and entry.get("is_img2img")
        and not entry.get("is_alwayson")
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"Expected one img2img HyperWeave script, found {len(matches)}"
        )
    return matches[0]


def script_args(info: dict, overrides: dict[str, object]) -> list[object]:
    args = [item.get("value") for item in info["args"]]
    labels = [item.get("label") for item in info["args"]]
    missing = sorted(set(overrides) - set(labels))
    if missing:
        raise AssertionError(f"HyperWeave API labels missing: {missing}")
    for label, value in overrides.items():
        args[labels.index(label)] = value
    return args


@unittest.skipUnless(
    LIVE, "set HYPERWEAVE_RUN_GPU_TESTS=1 for the live HyperWeave smoke"
)
class HyperWeaveLiveApiTests(unittest.TestCase):
    def test_disabled_delegates_to_normal_img2img(self):
        self.assertTrue(INPUT.is_file(), INPUT)
        proxy, _ = proxy_and_manual_roi(INPUT)
        proxy = proxy.resize((360, 512), Image.Resampling.LANCZOS)
        info = hyperweave_script_info()
        values = script_args(info, {"Enable HyperWeave": False})
        response = post_json(
            "/sdapi/v1/img2img",
            {
                "init_images": [encode_image(proxy)],
                "prompt": "two anime girls reading in a sunlit library",
                "negative_prompt": "",
                "script_name": info["name"],
                "script_args": values,
                "sampler_name": "DPM++ 2M SDE",
                "scheduler": "Simple",
                "steps": 1,
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 1.15,
                "denoising_strength": 0.16,
                "seed": 976834651,
                "width": proxy.width,
                "height": proxy.height,
                "batch_size": 1,
                "n_iter": 1,
                "send_images": True,
                "save_images": False,
            },
        )
        images = response.get("images") or []
        self.assertEqual(1, len(images))
        result = decode_image(images[0])
        self.assertEqual(proxy.size, result.size)
        self.assertNotIn("hyperweave", result.info)

    def test_supplied_image_anchor_global_and_manual_face(self):
        self.assertTrue(INPUT.is_file(), INPUT)
        proxy, manual_roi = proxy_and_manual_roi(INPUT)
        info = hyperweave_script_info()
        exact_steps = int(os.environ.get("HYPERWEAVE_LIVE_STEPS", "1"))
        values = script_args(
            info,
            {
                "Enable HyperWeave": True,
                "Target resolution": LIVE_TARGET,
                "Preset": "Structure Safe",
                "Content profile": "Illustration / Anime",
                "HyperWeave seed (-1 = random once)": 976834651,
                "Exact Steps": exact_steps,
                "Overdraw Amount": 0.75,
                "Structural Lock": 0.90,
                "Low Frequency Lock": 1.00,
                "Anchor strength": 0.12,
                "Global overdraw strength": 0.24,
                "Face redraw strength": 0.22,
                "Global candidate count": 1,
                "Face candidate count": 1,
                "Enable face/head redraw": True,
                "Enable hair redraw": False,
                "Enable material redraw": False,
                "Enable micro pass": False,
                "Detector provider": "Manual ROI",
                "Manual Face Core Mask": encode_image(manual_roi),
                "ROI stages": "Final stage only",
                "Back projection iterations": 1,
                "Save debug images": False,
            },
        )
        before_temp = set(Path(tempfile.gettempdir()).glob("hyperweave_*"))
        response = post_json(
            "/sdapi/v1/img2img",
            {
                "init_images": [encode_image(proxy)],
                "prompt": (
                    "two anime girls reading a book in a sunlit circular library, "
                    "blue-haired horned girl, white-haired girl, translucent green "
                    "slime, exact original composition and character identities"
                ),
                "negative_prompt": "",
                "script_name": info["name"],
                "script_args": values,
                "sampler_name": "DPM++ 2M SDE",
                "scheduler": "Simple",
                "steps": max(1, exact_steps),
                "cfg_scale": 1.0,
                "distilled_cfg_scale": 1.15,
                "denoising_strength": 0.16,
                "seed": 976834651,
                "width": proxy.width,
                "height": proxy.height,
                "batch_size": 1,
                "n_iter": 1,
                "send_images": True,
                "save_images": False,
            },
        )
        images = response.get("images") or []
        self.assertEqual(1, len(images))
        result = decode_image(images[0])
        expected_target = expected_live_target(proxy.size)
        self.assertEqual(expected_target, result.size)
        self.assertGreater(sum(ImageStat.Stat(result.convert("RGB")).var) / 3, 1.0)
        parameters = str(result.info.get("parameters", ""))
        response_info = str(response.get("info", ""))
        self.assertIn("HyperWeave", parameters + response_info)
        self.assertIn("976834651", parameters + response_info)
        if "hyperweave" in result.info:
            manifest = json.loads(result.info["hyperweave"])
            expected_faces = 2 if FACE_ROIS == "both" else 1
            self.assertEqual(expected_faces, manifest["detected_faces"])
            self.assertEqual(list(expected_target), manifest["target_size"])
            self.assertIn("memory", manifest)
            self.assertIn("runtime", manifest)
            self.assertIn("quality", manifest)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        save_png_with_text(result, OUTPUT)
        self.assertTrue(OUTPUT.is_file())
        with Image.open(OUTPUT) as saved:
            self.assertIn("HyperWeave", str(saved.info.get("parameters", "")))
            self.assertIn("976834651", str(saved.info.get("parameters", "")))
            self.assertIn("hyperweave", saved.info)
        after_temp = set(Path(tempfile.gettempdir()).glob("hyperweave_*"))
        self.assertEqual(before_temp, after_temp)

    @unittest.skipUnless(
        INTERRUPT_LIVE,
        "set HYPERWEAVE_RUN_INTERRUPT_TESTS=1 for the live interrupt smoke",
    )
    def test_interrupt_cleans_temporary_state_and_api_recovers(self):
        self.assertTrue(INPUT.is_file(), INPUT)
        proxy, _ = proxy_and_manual_roi(INPUT)
        info = hyperweave_script_info()
        values = script_args(
            info,
            {
                "Enable HyperWeave": True,
                "Target resolution": "x2",
                "Preset": "Structure Safe",
                "Content profile": "Illustration / Anime",
                "HyperWeave seed (-1 = random once)": 976834651,
                "Exact Steps": int(
                    os.environ.get("HYPERWEAVE_INTERRUPT_STEPS", "20")
                ),
                "Global candidate count": 1,
                "Enable face/head redraw": False,
                "Enable hair redraw": False,
                "Enable material redraw": False,
                "Enable micro pass": False,
                "Back projection iterations": 1,
                "Save debug images": False,
            },
        )
        payload = {
            "init_images": [encode_image(proxy)],
            "prompt": "two anime girls reading in a sunlit circular library",
            "negative_prompt": "",
            "script_name": info["name"],
            "script_args": values,
            "sampler_name": "DPM++ 2M SDE",
            "scheduler": "Simple",
            "steps": 20,
            "cfg_scale": 1.0,
            "distilled_cfg_scale": 1.15,
            "denoising_strength": 0.16,
            "seed": 976834651,
            "width": proxy.width,
            "height": proxy.height,
            "batch_size": 1,
            "n_iter": 1,
            "send_images": False,
            "save_images": False,
        }
        before_temp = set(Path(tempfile.gettempdir()).glob("hyperweave_*"))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(post_json, "/sdapi/v1/img2img", payload)
            deadline = time.monotonic() + 90.0
            observed_running = False
            while time.monotonic() < deadline and not future.done():
                progress = get_json(
                    "/sdapi/v1/progress?skip_current_image=true"
                )
                state = (
                    progress.get("state", {})
                    if isinstance(progress, dict)
                    else {}
                )
                if (
                    "HyperWeave" in str(state.get("job", ""))
                    or int(state.get("job_count", 0) or 0) > 0
                    or float(progress.get("progress", 0.0) or 0.0) > 0.0
                ):
                    observed_running = True
                    post_json("/sdapi/v1/interrupt", {})
                    break
                time.sleep(0.25)
            self.assertTrue(observed_running, "live job never entered a running state")
            try:
                future.result(timeout=TIMEOUT)
            except URLError as exc:
                if isinstance(exc, HTTPError):
                    self.assertGreaterEqual(exc.code, 400)

        cleanup_deadline = time.monotonic() + 30.0
        while time.monotonic() < cleanup_deadline:
            after_temp = set(Path(tempfile.gettempdir()).glob("hyperweave_*"))
            if after_temp == before_temp:
                break
            time.sleep(0.25)
        self.assertEqual(before_temp, after_temp)
        self.assertIsInstance(get_json("/sdapi/v1/options"), dict)


if __name__ == "__main__":
    unittest.main()
