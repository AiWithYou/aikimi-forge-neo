"""Compare H3 Union 1/2 and native VAE modes with one fixed control video.

Example:
  python tools/benchmark_h3_union2_vae.py --control-video input.mp4 \
    --output outputs/minimax_h3/union2_vae_compare

This is an opt-in GPU benchmark. Its results are local and are never uploaded.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge import minimax_h3_bridge as bridge  # noqa: E402
from modules_forge.minimax_h3_acceleration import H3Acceleration  # noqa: E402
from modules_forge.minimax_h3_fun_control import H3FunControl  # noqa: E402

CASES = {
    "union1": ("canny", "standard"),
    "union2": ("v2_canny", "standard"),
    "union2_fp16": ("v2_canny", "fp16_accumulation"),
}
PROMPT = (
    "A single cheerful silver-haired anime dancer in a navy and white stage costume "
    "performs a gentle side-to-side dance in one continuous full-body shot. "
    "Locked camera, clean 2D animation, coherent anatomy and consistent costume. "
    "Follow the supplied motion edges. Upbeat instrumental dance music, no vocals, no text."
)


def gpu_used_mib() -> int | None:
    try:
        executable = shutil.which("nvidia-smi")
        if executable is None:
            return None
        result = subprocess.run(  # noqa: S603
            [executable, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(result.stdout.splitlines()[0].strip())
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def monitor(stop: threading.Event, values: list[dict]) -> None:
    while not stop.is_set():
        values.append({
            "gpu_used_mib": gpu_used_mib(),
            "ram_used_gib": round(psutil.virtual_memory().used / 1024**3, 2),
            "commit_free_gib": bridge._local_commit_free_gib(),
        })
        stop.wait(3)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-video", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--profile", choices=sorted(bridge.RUNTIME_PROFILES), default="low_ram")
    parser.add_argument("--quality", choices=sorted(bridge.QUALITY_DIMENSIONS), default="draft")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--max-job-minutes", type=float, default=45, help="Request a safe ComfyUI cancellation after this job time")
    parser.add_argument("--diagnostic-no-compiler", action="store_true", help="Use H3 Studio's Comfy Compiler off mode for runs that stall during model initialization")
    args = parser.parse_args()
    if not math.isfinite(args.max_job_minutes) or args.max_job_minutes <= 0:
        parser.error("--max-job-minutes must be a positive finite number")
    source = args.control_video.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runtime = bridge.managed_runtime_root()
    url = bridge.H3_SERVER_URL
    report_path = output / "benchmark.json"
    settings = {
        "schema_version": 1,
        "control_video": str(source),
        "profile": args.profile,
        "quality": args.quality,
        "steps": args.steps,
        "seed": args.seed,
        "prompt": args.prompt,
        "diagnostic_no_compiler": args.diagnostic_no_compiler,
    }
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if any(report.get(key) != value for key, value in settings.items()):
            raise ValueError("Existing benchmark settings differ; choose a new output directory")
    else:
        report = {**settings, "runs": []}
    try:
        for index, name in enumerate(args.cases):
            if any(row.get("case") == name and row.get("video") and Path(row["video"]).is_file() for row in report["runs"]):
                print(f"{name}: already complete", flush=True)  # noqa: T201
                continue
            report["runs"] = [row for row in report["runs"] if row.get("case") != name]
            control_mode, decode_mode = CASES[name]
            request = bridge.H3Request(
                mode=bridge.MODE_TEXT,
                prompt=args.prompt,
                quality=args.quality,
                duration_seconds=5,
                steps=args.steps,
                seed=args.seed,
                acceleration=H3Acceleration(
                    decode_mode=decode_mode,
                    compiler_mode="off" if args.diagnostic_no_compiler else "on",
                ),
                control=H3FunControl(mode=control_mode),
                control_video=str(source),
            )
            bridge.validate_request(request)
            stop = threading.Event()
            samples: list[dict] = []
            worker = threading.Thread(target=monitor, args=(stop, samples), daemon=True)
            started = time.perf_counter()
            row = {"case": name, "request": asdict(request)}
            last_stage = None
            timeout_requested = False
            worker.start()
            try:
                for event in bridge.run_generation(
                    request, runtime, url, output / "logs", output / name,
                    runtime_profile=args.profile,
                ):
                    if event["stage"] != last_stage:
                        print(f"{name}: {event['stage']} {event.get('message', '')}", flush=True)  # noqa: T201
                        last_stage = event["stage"]
                    if (
                        not timeout_requested
                        and event.get("elapsed", 0) >= args.max_job_minutes * 60
                        and event["stage"] != "complete"
                    ):
                        timeout_requested = True
                        bridge.cancel_generation(event["prompt_id"], url)
                        print(f"{name}: job time limit reached; cancellation requested", flush=True)  # noqa: T201
                    if event["stage"] == "complete":
                        row["job_seconds"] = round(event["elapsed"], 3)
                        row["video"] = event["path"]
                        metadata = Path(event["path"]).with_suffix(".json")
                        row["metadata"] = str(metadata)
                        row["output_validation"] = json.loads(metadata.read_text("utf-8"))["output_validation"]
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                stop.set()
                worker.join(timeout=6)
                row["wall_seconds"] = round(time.perf_counter() - started, 3)
                row["peak_gpu_used_mib"] = max((x["gpu_used_mib"] for x in samples if x["gpu_used_mib"] is not None), default=None)
                row["peak_ram_used_gib"] = max((x["ram_used_gib"] for x in samples), default=None)
                row["minimum_commit_free_gib"] = min((x["commit_free_gib"] for x in samples if x["commit_free_gib"] is not None), default=None)
                report["runs"].append(row)
                report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print("RESULT " + json.dumps({key: value for key, value in row.items() if key != "request"}, ensure_ascii=False), flush=True)  # noqa: T201
            if "video" not in row:
                raise RuntimeError(f"{name} did not produce a video")
            if index + 1 < len(args.cases) and decode_mode != CASES[args.cases[index + 1]][1]:
                bridge._release_retained_runtime(url)
    finally:
        try:
            bridge._release_retained_runtime(url)
        except bridge.H3BridgeError as exc:
            print(f"H3 runtime remains active: {exc}", file=sys.stderr, flush=True)  # noqa: T201


if __name__ == "__main__":
    main()
