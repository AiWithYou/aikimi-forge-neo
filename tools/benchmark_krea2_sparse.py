"""Compare Krea2 attention and complete 4K upscales through the local Forge API."""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import time
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

PROMPT = (
    "A detailed editorial illustration of a young silver-haired astronomer in a navy coat on a stone observatory "
    "terrace, a brass telescope with engraved metalwork, small ceramic tea cups on a wooden table, distant mountains "
    "under a deep blue evening sky. One person, intricate fabric and hair, realistic lighting, clear natural detail."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=7864)
    parser.add_argument("--stage", choices=("native", "4k"), default="native")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1280, 2048])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--keep", type=float, default=10)
    parser.add_argument("--minimum-tokens", type=int, default=4096)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["off", "fixed", "jev"],
        choices=(
            "off",
            "fixed",
            "rules",
            "jev",
            "tiles-rules",
            "tiles-jev",
            "combined",
            "fixed-rules",
            "fixed-tiles-jev",
        ),
    )
    parser.add_argument("--source", type=Path)
    parser.add_argument("--allow-cloud", action="store_true")
    args = parser.parse_args()
    if any("jev" in mode or mode == "combined" for mode in args.modes) and not args.allow_cloud:
        parser.error("--allow-cloud is required to send aggregate statistics to Jev")
    if args.stage == "native" and any(mode not in {"off", "fixed", "rules", "jev"} for mode in args.modes):
        parser.error("Tile allocation modes require --stage 4k")
    if args.repeats < 1 or not 1 <= args.port <= 65535:
        parser.error("Positive repeats and a valid loopback port are required")
    if any(size < 256 or size > 2048 or size % 16 for size in args.sizes):
        parser.error("Native sizes must be 256..2048 and divisible by 16")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    api = f"http://127.0.0.1:{args.port}"

    def request(path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = Request(api + path, data=data, headers={"Content-Type": "application/json"})  # noqa: S310
        with urlopen(req, timeout=3600) as response:  # noqa: S310 -- fixed loopback origin
            return json.load(response)

    module_names = [m["model_name"] for m in request("/sdapi/v1/sd-modules")]
    required = ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"]
    if any(name not in module_names for name in required):
        raise RuntimeError("Krea2's VAE and Qwen3-VL 4B encoder must be installed and refreshed first")
    previous = request("/sdapi/v1/options")
    restore = {
        key: previous[key]
        for key in ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")
        if key in previous
    }
    checkpoint = previous["sd_model_checkpoint"]
    if "krea2" not in checkpoint.lower():
        raise RuntimeError("Select the Krea2 checkpoint to benchmark first")
    del previous
    payload = {
        "prompt": PROMPT,
        "negative_prompt": "",
        "seed": 20260921,
        "sampler_name": "DPM++ 2M SDE",
        "scheduler": "Simple",
        "steps": 4,
        "cfg_scale": 1,
        "distilled_cfg_scale": 1.15,
        "width": 1280,
        "height": 1280,
        "batch_size": 1,
        "n_iter": 1,
        "send_images": True,
        "save_images": False,
        "override_settings_restore_afterwards": True,
        "override_settings": {
            "sd_model_checkpoint": checkpoint,
            "forge_additional_modules": required,
            "forge_unet_storage_dtype": "Automatic",
        },
        "alwayson_scripts": {"Krea2 Jev / Sparse": {"args": ["off", args.keep, args.minimum_tokens, "off", 10]}},
    }
    report = {
        "stage": args.stage,
        "runs": [],
        "settings": payload,
        "protocol": (
            "Model preparation before timing; fixed seed and model; "
            + ("forward/reversed mode order" if args.repeats > 1 else "one forward pass")
            + "; API wait and output processing included"
        ),
    }

    def generate(label, mode, body, endpoint, repetition):
        directory = root / label
        directory.mkdir(exist_ok=False)
        attention_mode = (
            "jev"
            if mode == "combined"
            else "fixed"
            if mode in {"fixed-rules", "fixed-tiles-jev"}
            else "off"
            if mode.startswith("tiles-")
            else mode
        )
        tile_mode = (
            "jev"
            if mode in {"combined", "tiles-jev", "fixed-tiles-jev"}
            else "rules"
            if mode in {"tiles-rules", "fixed-rules"}
            else "off"
        )
        body = json.loads(json.dumps(body))
        body["alwayson_scripts"]["Krea2 Jev / Sparse"]["args"] = [
            attention_mode,
            args.keep,
            args.minimum_tokens,
            tile_mode,
            10,
        ]
        print(f"BENCHMARK_START {label}", flush=True)  # noqa: T201
        started = time.perf_counter()
        result = request(endpoint, body)
        elapsed = time.perf_counter() - started
        output = directory / "output.png"
        output.write_bytes(base64.b64decode(result["images"][0].split(",")[-1]))
        info = json.loads(result["info"])
        extra = info.get("extra_generation_params", {})
        log_summaries = {}
        for kind, key in (("attention", "Krea2 Sparse log"), ("tiles", "Krea2 tile allocation log")):
            if not extra.get(key):
                continue
            records = [json.loads(line) for line in Path(extra[key]).read_text(encoding="utf-8").splitlines()]
            end = next(row for row in reversed(records) if row["event"] == "end")
            decisions = [row for row in records if row["event"] == "decision"]
            log_summaries[kind] = {"log_path": extra[key], "end": end, "decisions": decisions}
        if attention_mode != "off":
            if extra.get("Krea2 Sparse status") != "active" or "attention" not in log_summaries:
                raise RuntimeError("Krea2 attention did not activate")
            if attention_mode in {"fixed", "rules", "jev"} and not log_summaries["attention"]["end"][
                "attention_calls"
            ].get("sparse"):
                raise RuntimeError("No real sparse attention calls were recorded")
        for kind, selected in (("attention", attention_mode), ("tiles", tile_mode)):
            if selected == "jev" and not any(
                d.get("source") == "jev" for d in log_summaries.get(kind, {}).get("decisions", [])
            ):
                raise RuntimeError(f"No valid Jev {kind} decision; fallback is not a Jev result")
        if any(entry["end"]["api_calls"] > 1 for entry in log_summaries.values()):
            raise RuntimeError("Jev was called repeatedly for nested tiles")
        (directory / "generation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        row = {
            "label": label,
            "mode": mode,
            "repetition": repetition,
            "width": info.get("width", body["width"]),
            "height": info.get("height", body["height"]),
            "wall_seconds": elapsed,
            "output_path": str(output),
            "logs": log_summaries,
        }
        report["runs"].append(row)
        (root / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("BENCHMARK_RESULT " + json.dumps({k: v for k, v in row.items() if k != "logs"}), flush=True)  # noqa: T201
        return output

    try:
        if args.stage == "native":
            for size in args.sizes:
                body = {**payload, "width": size, "height": size}
                generate(f"{size}-warmup", "off", body, "/sdapi/v1/txt2img", 0)
                for n in range(args.repeats):
                    for mode in args.modes if n % 2 == 0 else reversed(args.modes):
                        generate(f"{size}-{mode}-{n + 1}", mode, body, "/sdapi/v1/txt2img", n + 1)
        else:
            source = args.source
            if source is None:
                source = generate(
                    "source-2k", "off", {**payload, "width": 2048, "height": 1152}, "/sdapi/v1/txt2img", 0
                )
            scripts = request("/sdapi/v1/script-info")
            script = next(s for s in scripts if s["is_img2img"] and s["name"] == "vram-canvas 4k/8k highres")
            script_args = [entry["value"] for entry in script["args"]]
            # VRAM-Canvas uses the globally selected Krea2 and intentionally
            # rejects per-request checkpoint/VAE overrides.
            status = request("/sdapi/v1/forge-model-status/ensure-loaded", {})
            if not status["loaded"] or not status["architecture"].endswith(".Krea2"):
                raise RuntimeError("The globally selected Krea2 could not be prepared")
            # Fixed, documented speed profile for all modes. The same geometric
            # tile plan and normal adaptive steps are used for the baseline.
            with Image.open(source) as source_image:
                source_width, source_height = source_image.size
            scale = 4096 / max(source_width, source_height)
            target_width, target_height = round(source_width * scale), round(source_height * scale)
            updates = {
                0: 4096,
                1: target_width,
                2: target_height,
                3: 24,
                4: 1280,
                5: 14,
                6: 2.0,
                7: 1,
                8: 2,
                9: 4,
                11: 0.13,
                12: 0.13,
                19: 0,
                21: False,
                22: True,
                23: True,
            }
            for index, value in updates.items():
                script_args[index] = value
            body = {
                **payload,
                "override_settings": {},
                "override_settings_restore_afterwards": True,
                "width": source_width,
                "height": source_height,
                "denoising_strength": 0.13,
                "init_images": [base64.b64encode(source.read_bytes()).decode()],
                "script_name": script["name"],
                "script_args": script_args,
            }
            report["source"] = str(source.resolve())
            report["upscale_script_args"] = script_args
            for n in range(args.repeats):
                for mode in args.modes if n % 2 == 0 else reversed(args.modes):
                    generate(f"4k-{mode}-{n + 1}", mode, body, "/sdapi/v1/img2img", n + 1)
        report["medians"] = {
            str(size): {
                mode: statistics.median(
                    r["wall_seconds"]
                    for r in report["runs"]
                    if r["mode"] == mode and r["repetition"] and r["width"] == size
                )
                for mode in args.modes
            }
            for size in sorted({r["width"] for r in report["runs"] if r["repetition"]})
        }
        (root / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        request("/sdapi/v1/options", restore)


if __name__ == "__main__":
    main()
