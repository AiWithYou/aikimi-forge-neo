"""Measure the actual Forge API with fixed Anima inputs, including OFF recovery."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "dense", "fixed", "rules", "jev"),
        default=["off", "fixed", "rules", "jev"],
    )
    parser.add_argument("--keep", type=float, default=25)
    parser.add_argument("--max-calls", type=int, default=1)
    parser.add_argument("--allow-cloud", action="store_true")
    args = parser.parse_args()
    if not args.allow_cloud:
        parser.error("--allow-cloud is required for the Jev comparison")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = f"http://127.0.0.1:{args.port}"

    def request(path, payload=None):
        encoded = None if payload is None else json.dumps(payload).encode()
        req = Request(api + path, data=encoded, headers={"Content-Type": "application/json"})  # noqa: S310 -- fixed loopback
        with urlopen(req, timeout=1800) as response:  # noqa: S310 -- fixed loopback origin
            return json.load(response)

    modules = [entry["model_name"] for entry in request("/sdapi/v1/sd-modules")]

    def module(name):
        hits = [m for m in modules if m == name or m.startswith(Path(name).stem + " [")]
        if len(hits) != 1:
            raise RuntimeError(f"Cannot identify the required Anima module: {name}")
        return hits[0]

    payload = {
        "prompt": "masterpiece, best quality, newest, anime illustration. A silver-haired astronomer in a navy coat "
        "on a moonlit observatory terrace, looking toward a bright comet. A brass telescope, detailed eyes, "
        "fine hair strands, intricate metalwork, blue and golden lighting.",
        "negative_prompt": "worst quality, low quality, blurry",
        "seed": 20260921,
        "sampler_name": "Res Multistep",
        "scheduler": "Beta",
        "steps": args.steps,
        "cfg_scale": 4,
        "distilled_cfg_scale": 3,
        "width": 1024,
        "height": 1024,
        "n_iter": 1,
        "batch_size": 1,
        "send_images": True,
        "save_images": False,
        "override_settings_restore_afterwards": False,
        "override_settings": {
            "sd_model_checkpoint": "Anima-3.8B-v1.1-int8-convrot.safetensors",
            "forge_additional_modules": [module("qwen_image_vae.safetensors"), module("qwen_3_06b_base.safetensors")],
            "forge_unet_storage_dtype": "Automatic",
        },
        "alwayson_scripts": {
            "Anima 3.8B": {
                "args": [True, "Anima-3.8B-expanded_adapter.safetensors", 1.0, False, 1.0, False, "None", 1.0]
            }
        },
    }
    modes = args.modes
    report = {"settings": payload, "runs": [], "protocol": "one warm-up, forward then reversed mode order"}
    schedule = [("off", 0)] + [
        (mode, n + 1) for n in range(args.repeats) for mode in (modes if n % 2 == 0 else list(reversed(modes)))
    ]
    previous = request("/sdapi/v1/options")
    restore = {key: previous[key] for key in payload["override_settings"] if key in previous}
    del previous
    try:
        for mode, repetition in schedule:
            label = "warmup" if not repetition else f"{mode}-{repetition}"
            directory = root / label
            directory.mkdir(exist_ok=False)
            body = json.loads(json.dumps(payload))
            body["alwayson_scripts"]["Anima Sparse Attention (experimental)"] = {
                "args": [mode, args.keep, 4096, 1, 4, args.max_calls, 10]
            }
            print(f"BENCHMARK_START {label}", flush=True)  # noqa: T201
            started = time.perf_counter()
            result = request("/sdapi/v1/txt2img", body)
            elapsed = time.perf_counter() - started
            info = json.loads(result["info"])
            output = directory / "output.png"
            output.write_bytes(base64.b64decode(result["images"][0].split(",")[-1]))
            extra = info.get("extra_generation_params", {})
            sparse = None
            if mode != "off":
                if extra.get("Anima Sparse status") != "active_experiment":
                    raise RuntimeError("The selected Anima experiment did not activate")
                records = [
                    json.loads(line)
                    for line in Path(extra["Anima Sparse log"]).read_text(encoding="utf-8").splitlines()
                ]
                sparse = next(row for row in reversed(records) if row["event"] == "end")
                if (
                    mode == "jev"
                    and args.max_calls
                    and not any(row["event"] == "decision" and row.get("source") == "jev" for row in records)
                ):
                    raise RuntimeError("Jev did not return a valid decision; fallback is not a Jev speed result")
            (directory / "generation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            row = {
                "label": label,
                "mode": mode,
                "repetition": repetition,
                "wall_seconds": elapsed,
                "output_path": str(output),
                "sparse": sparse,
            }
            report["runs"].append(row)
            (root / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print("BENCHMARK_RESULT " + json.dumps(row), flush=True)  # noqa: T201
        report["medians"] = {
            mode: statistics.median(r["wall_seconds"] for r in report["runs"] if r["mode"] == mode and r["repetition"])
            for mode in modes
        }
        (root / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        request("/sdapi/v1/options", restore)


if __name__ == "__main__":
    main()
