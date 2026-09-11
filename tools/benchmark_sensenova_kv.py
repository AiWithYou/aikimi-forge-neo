"""導入済みSenseNovaでKV方式の比較とキャンセル後の再生成を記録する。"""

import json
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from modules_forge.sensenova_u15_bridge import (
    DEFAULT_CHECKPOINT_PATH,
    MODE_EDIT,
    PROFILE_QUALITY,
    QUANT_INT8_CONVROT,
    SenseNovaGenerationCancelled,
    SenseNovaRequest,
    cancel_generation,
    run_generation,
)
from tools.tests.test_sensenova_u15_live import SenseNovaU15LiveTest


def main():
    root = Path("outputs") / ("sensenova_kv_compare_" + time.strftime("%Y%m%d_%H%M%S"))
    root.mkdir(parents=True)
    request = SenseNovaRequest(
        mode=MODE_EDIT,
        prompt="Use the blue circle from the first image with the warm red palette from the second image. Keep a clean simple background.",
        generation_profile=PROFILE_QUALITY,
        quantization=QUANT_INT8_CONVROT,
        checkpoint=str(DEFAULT_CHECKPOINT_PATH),
        input_images=SenseNovaU15LiveTest._reference_images(),
        width=512,
        height=512,
        input_max_pixels=str(512 * 512),
        steps=2,
        cfg_scale=4.0,
        img_cfg_scale=1.0,
        timestep_shift=3.0,
        seed=42,
        vram_mode="reference",
        attn_backend="sdpa",
        dtype="bfloat16",
    )
    report = {
        "size": 512,
        "input_max_pixels": 512 * 512,
        "steps": 2,
        "seed": 42,
        "resident_cap_gib": 0.015625,
        "timeout_seconds": 90,
        "runs": {},
    }
    baseline = None
    for label, adaptive, prefetch, cancel in [
        ("cpu", "0", "0", False),
        ("resident", "1", "0", False),
        ("prefetch", "1", "1", False),
        ("cancel", "1", "1", True),
        ("regenerate", "1", "1", False),
    ]:
        os.environ.update(
            AIKIMI_SENSENOVA_KV_ADAPTIVE=adaptive,
            AIKIMI_SENSENOVA_KV_PREFETCH=prefetch,
            AIKIMI_SENSENOVA_KV_MAX_RESIDENT_GIB="0.015625",
        )
        started = time.perf_counter()
        cancelled = False
        timed_out = False
        try:
            for event in run_generation(
                request, output_directory=root / label, cache_directory=root / "cache", log_directory=root / "logs"
            ):
                print(label, event["stage"], event.get("message", ""), flush=True)  # noqa: T201
                if not cancelled and time.perf_counter() - started > report["timeout_seconds"]:
                    cancel_generation(event["job_id"])
                    cancelled = timed_out = True
                if cancel and not cancelled and event["stage"] == "sampling":
                    cancel_generation(event["job_id"])
                    cancelled = True
                if event["stage"] == "complete":
                    with Image.open(event["path"]) as image:
                        pixels = np.asarray(image.convert("RGB")).astype(np.int16)
                    if label == "cpu":
                        baseline = pixels
                    diff = np.abs(pixels - baseline) if baseline is not None else None
                    report["runs"][label] = {
                        "status": "complete",
                        "wall_seconds": time.perf_counter() - started,
                        "max_pixel_difference": int(diff.max()) if diff is not None else None,
                        "mean_pixel_difference": float(diff.mean()) if diff is not None else None,
                        "different_channels": int(np.count_nonzero(diff)) if diff is not None else None,
                        "path": event["path"],
                        "metadata": event["metadata"],
                    }
        except SenseNovaGenerationCancelled:
            report["runs"][label] = {
                "status": "timeout" if timed_out else "cancelled",
                "wall_seconds": time.perf_counter() - started,
            }
        except Exception as error:
            report["runs"][label] = {"status": "failed", "error": str(error)}
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(root / "report.json", flush=True)  # noqa: T201
    if any(
        run["status"] != ("cancelled" if label == "cancel" else "complete") for label, run in report["runs"].items()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
