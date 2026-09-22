"""Compare complete Krea2 native/4K images across cases, seeds, keeps and Jev replay.

--dry-run writes the matrix without contacting Forge or Jev. The local Forge API
must use the updated Krea2 sparse script for per-request replay and job budgets.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import math
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import sparse_benchmark as bench  # noqa: E402

SCRIPT = "Krea2 Jev / Sparse"
MODES = (
    "off",
    "dense",
    "fixed",
    "rules",
    "jev",
    "replay",
    "tiles-rules",
    "tiles-jev",
    "tiles-replay",
    "combined",
    "combined-replay",
    "fixed-rules",
    "fixed-tiles-jev",
    "fixed-tiles-replay",
)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--port", type=int, default=7864)
    result.add_argument("--stage", choices=("native", "4k"), default="native")
    result.add_argument("--sizes", type=int, nargs="+", default=[1280, 2048])
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument(
        "--keep", type=float, help="Legacy single fixed keep, or initial adaptive keep with --fixed-keeps"
    )
    result.add_argument("--fixed-keeps", type=float, nargs="+")
    result.add_argument("--minimum-tokens", type=int, default=4096)
    result.add_argument("--decision-cadence", choices=("once", "interval", "step"), default="once")
    result.add_argument("--decision-interval", type=int, default=2)
    result.add_argument("--tile-cadence", choices=("once", "stage"), default="once")
    result.add_argument("--job-max-calls", type=int, default=0, help="Extra job-wide cap; 0 leaves this cap unset")
    result.add_argument("--job-max-wait-seconds", type=float, default=0)
    result.add_argument("--modes", nargs="+", default=["off", "fixed", "rules", "jev", "replay"], choices=MODES)
    result.add_argument("--source", type=Path, help="Fixed source image for every selected 4K case")
    result.add_argument("--allow-cloud", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--case-manifest", type=Path)
    result.add_argument("--cases", nargs="+")
    result.add_argument("--reference-images", type=Path, nargs="+", default=[])
    result.add_argument("--replay-from", type=Path)
    seeds = result.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    result.add_argument("--prompt", help="One custom text case instead of the built-in case suite")
    return result


def make_plan(args):
    args.fixed_keeps = args.fixed_keeps or ([args.keep] if args.keep is not None else [3, 10, 25])
    args.keep = 10 if args.keep is None else args.keep
    if not 1 <= args.port <= 65535 or not 1 <= args.decision_interval <= 100:
        raise ValueError("A valid loopback port and decision interval 1..100 are required")
    if any(size < 256 or size > 2048 or size % 16 for size in args.sizes):
        raise ValueError("Native sizes must be 256..2048 and divisible by 16")
    if not 64 <= args.minimum_tokens <= 1048576:
        raise ValueError("minimum-tokens must be 64..1048576")
    if (
        not 0 <= args.job_max_calls <= 1000
        or not math.isfinite(args.job_max_wait_seconds)
        or not 0 <= args.job_max_wait_seconds <= 3600
    ):
        raise ValueError("job-max-calls must be 0..1000 and job-max-wait-seconds must be 0..3600")
    if args.stage == "native" and any(
        mode not in {"off", "dense", "fixed", "rules", "jev", "replay"} for mode in args.modes
    ):
        raise ValueError("Tile allocation modes require --stage 4k")
    cases = bench.load_cases(
        args.case_manifest,
        reference_images=args.reference_images,
        selected=args.cases,
        prompt=args.prompt,
        source=args.source,
    )
    settings = {
        "sampler_name": "DPM++ 2M SDE",
        "scheduler": "Simple",
        "steps": 4,
        "cfg_scale": 1,
        "distilled_cfg_scale": 1.15,
        "minimum_tokens": args.minimum_tokens,
        "decision_cadence": args.decision_cadence,
        "decision_interval": args.decision_interval,
        "tile_cadence": args.tile_cadence,
        "job_max_calls": args.job_max_calls,
        "job_max_wait_seconds": args.job_max_wait_seconds,
        "timeout": 10,
        "upscale_profile": "vram-canvas-speed-4k-v1" if args.stage == "4k" else None,
    }
    plan = bench.build_plan(
        model="krea2",
        stage=args.stage,
        cases=cases,
        seeds=[args.seed] if args.seed is not None else (args.seeds or [20260921, 20260922]),
        dimensions=[(size, size) for size in args.sizes] if args.stage == "native" else [(4096, 0)],
        variants=bench.build_variants(args.modes, args.fixed_keeps, args.keep),
        repeats=args.repeats,
        settings=settings,
        replay_from=args.replay_from,
    )
    plan["planned_source_generations"] = (
        sum(not g["case"]["source"] for g in plan["groups"]) if args.stage == "4k" else 0
    )
    if args.stage == "4k":
        plan["protocol"]["dimensions"] = (
            "4096px long edge, original aspect ratio; actual dimensions resolved before warmup"
        )
    return plan


def mode_pair(mode):
    mode = bench.REPLAY_SOURCES.get(mode, mode)
    attention = (
        "jev"
        if mode == "combined"
        else "fixed"
        if mode in {"fixed-rules", "fixed-tiles-jev"}
        else "off"
        if mode.startswith("tiles-")
        else mode
    )
    tiles = (
        "jev"
        if mode in {"combined", "tiles-jev", "fixed-tiles-jev"}
        else "rules"
        if mode in {"tiles-rules", "fixed-rules"}
        else "off"
    )
    return attention, tiles


class KreaRunner:
    def __init__(self, args, root, request=None):
        self.args, self.root = args, Path(root)
        self.request = request or self.api_request
        self.restore = None
        self.bodies = {}
        self.script = None

    def api_request(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            f"http://127.0.0.1:{self.args.port}" + path, data=data, headers={"Content-Type": "application/json"}
        )  # noqa: S310
        with urlopen(req, timeout=3600) as response:  # noqa: S310 -- fixed loopback origin
            return json.load(response)

    def connect(self):
        module_names = [m["model_name"] for m in self.request("/sdapi/v1/sd-modules")]
        required = ["qwen_image_vae.safetensors", "qwen3vl_4b_fp8_scaled.safetensors"]
        if any(name not in module_names for name in required):
            raise RuntimeError("Krea2's VAE and Qwen3-VL 4B encoder must be installed and refreshed first")
        previous = self.request("/sdapi/v1/options")
        self.restore = {
            key: previous[key]
            for key in ("sd_model_checkpoint", "forge_additional_modules", "forge_unet_storage_dtype")
            if key in previous
        }
        checkpoint = previous["sd_model_checkpoint"]
        if "krea2" not in checkpoint.lower():
            raise RuntimeError("Select the Krea2 checkpoint to benchmark first")
        scripts = self.request("/sdapi/v1/script-info")
        adapter = next((s for s in scripts if s["name"].lower() == SCRIPT.lower()), None)
        if adapter is None or len(adapter.get("args", [])) < 11:
            raise RuntimeError(
                "Restart Forge with the updated Krea2 sparse script (replay/job-budget arguments required)"
            )
        if self.args.stage == "4k":
            self.script = next(
                (s for s in scripts if s["is_img2img"] and s["name"] == "vram-canvas 4k/8k highres"), None
            )
            if self.script is None:
                raise RuntimeError("VRAM-Canvas 4K/8K highres img2img script was not found")
        self.payload = {
            "negative_prompt": "",
            "sampler_name": "DPM++ 2M SDE",
            "scheduler": "Simple",
            "steps": 4,
            "cfg_scale": 1,
            "distilled_cfg_scale": 1.15,
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
            "alwayson_scripts": {SCRIPT: {"args": self.script_arguments("off", self.args.keep)}},
        }
        return {"checkpoint": checkpoint, "additional_modules": required, "unet_storage_dtype": "Automatic"}

    def script_arguments(self, mode, keep, replay=None):
        attention, tiles = mode_pair(mode)
        args = self.args
        return [
            attention,
            keep,
            args.minimum_tokens,
            tiles,
            10,
            args.decision_cadence,
            args.decision_interval,
            args.tile_cadence,
            args.job_max_calls,
            args.job_max_wait_seconds,
            json.dumps(replay["logs"]) if replay else "",
        ]

    def request_image(self, body, endpoint, output, *, expected_size=None):
        started = time.perf_counter()
        result = self.request(endpoint, body)
        if not isinstance(result, dict) or not result.get("images"):
            raise RuntimeError("Forge did not return a completed image")
        output.write_bytes(base64.b64decode(result["images"][0].split(",")[-1], validate=True))
        artifact = bench.validate_output(output, expected_size=expected_size)
        elapsed = time.perf_counter() - started
        info = json.loads(result["info"]) if isinstance(result.get("info"), str) else result.get("info")
        if not isinstance(info, dict):
            raise RuntimeError("Forge response is missing generation metadata")
        return elapsed, artifact, info

    def prepare(self, group):
        case, identity = group["case"], group["identity"]
        body = copy.deepcopy(self.payload)
        body.update(prompt=case["prompt"], seed=identity["seed"], width=identity["width"], height=identity["height"])
        identity["runtime"] = self.payload["override_settings"]
        if self.args.stage == "native":
            self.bodies[group["id"]] = body
            return
        from PIL import Image

        source = Path(case["source"]) if case["source"] else self.root / f"{group['id']}-source.png"
        if not case["source"]:
            if source.exists():
                raise RuntimeError("Source image already exists; use a fresh benchmark directory")
            elapsed, artifact, info = self.request_image(
                {**body, "width": 2048, "height": 1152}, "/sdapi/v1/txt2img", source, expected_size=(2048, 1152)
            )
            group["source_generation"] = {"status": "completed", "wall_seconds": elapsed, "artifact": artifact}
            bench.atomic_json(source.with_suffix(".json"), info)
        with Image.open(source) as image:
            source_width, source_height = image.size
        scale = 4096 / max(source_width, source_height)
        target_width, target_height = round(source_width * scale), round(source_height * scale)
        identity.update(source_sha256=bench.digest(source), width=target_width, height=target_height)
        group["source_path"] = str(source.resolve())
        status = self.request("/sdapi/v1/forge-model-status/ensure-loaded", {})
        if not status.get("loaded") or not status.get("architecture", "").endswith(".Krea2"):
            raise RuntimeError("The globally selected Krea2 could not be prepared")
        script_args = [entry["value"] for entry in self.script["args"]]
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
        group["upscale_script_args"] = script_args
        body.update(
            override_settings={},
            width=source_width,
            height=source_height,
            denoising_strength=0.13,
            init_images=[base64.b64encode(source.read_bytes()).decode("ascii")],
            script_name=self.script["name"],
            script_args=script_args,
        )
        self.bodies[group["id"]] = body

    def generate(self, group, spec, directory, replay):
        body = copy.deepcopy(self.bodies[group["id"]])
        body["alwayson_scripts"][SCRIPT]["args"] = self.script_arguments(spec["mode"], spec["keep"], replay)
        # Persist a reviewable request without duplicating large source image data.
        bench.atomic_json(
            directory / "request.json", {key: value for key, value in body.items() if key != "init_images"}
        )
        output = directory / "output.png"
        identity = group["identity"]
        elapsed, artifact, info = self.request_image(
            body,
            "/sdapi/v1/txt2img" if self.args.stage == "native" else "/sdapi/v1/img2img",
            output,
            expected_size=(identity["width"], identity["height"]),
        )
        bench.atomic_json(directory / "generation.json", info)
        attention, tiles = mode_pair(spec["mode"])
        extra, logs = info.get("extra_generation_params", {}), {}
        for kind, key in (("attention", "Krea2 Sparse log"), ("tiles", "Krea2 tile allocation log")):
            if extra.get(key):
                logs[kind] = bench.read_log(extra[key])
        if attention != "off":
            if extra.get("Krea2 Sparse status") != "active" or "attention" not in logs:
                raise RuntimeError("Krea2 attention instrumentation did not activate")
            expected_sparse = attention in {"rules", "jev"} or (attention == "fixed" and spec["keep"] < 100)
            if expected_sparse and not logs["attention"]["end"].get("attention_calls", {}).get("sparse"):
                raise RuntimeError("No real sparse attention calls were recorded")
        if tiles != "off" and "tiles" not in logs:
            raise RuntimeError("Krea2 tile allocation did not activate")
        for kind, selected, cadence in (
            ("attention", attention, self.args.decision_cadence),
            ("tiles", tiles, self.args.tile_cadence),
        ):
            if selected == "jev":
                bench.validate_decisions(logs[kind], replay=bool(replay), kind=kind, cadence=cadence)
        return {
            "wall_seconds": elapsed,
            "output_path": str(output),
            "artifact": artifact,
            "logs": {key: bench.public_log(value) for key, value in logs.items()},
        }

    def close(self):
        if self.restore is not None:
            self.request("/sdapi/v1/options", self.restore)


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        plan = make_plan(args)
        if plan["planned_live_generations"] and not args.allow_cloud and not args.dry_run:
            raise ValueError("--allow-cloud is required for live Jev; replay alone sends no cloud requests")
        root = bench.save_plan(args.output, plan, dry_run=args.dry_run)
    except (ValueError, OSError) as exc:
        cli.error(str(exc))
    if args.dry_run:
        return
    if not plan["groups"]:
        bench.write_report(root, {**plan, "status": "no_runnable_cases", "runs": []})
        return
    runner = KreaRunner(args, root)
    try:
        plan["runtime"] = runner.connect()
        bench.execute_plan(plan, root, runner)
    except BaseException as exc:
        if not (root / "benchmark.json").exists():
            bench.write_report(
                root, {**plan, "status": "failed", "runs": [], "error_type": type(exc).__name__, "error": str(exc)}
            )
        raise
    finally:
        runner.close()


if __name__ == "__main__":
    main()
