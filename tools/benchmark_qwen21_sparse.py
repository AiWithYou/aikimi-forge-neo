"""Compare real Qwen 2.1 outputs with one resident model and fixed inputs.

Run with models/Qwen-Image-2.1/worker-env/Scripts/python.exe. The first run
is a separate warm-up; timed rounds alternate order to expose warm-cache bias.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.jev_sparse.credentials import enable_saved_key
from modules_forge.jev_sparse.qwen21 import OPTIONS_ENV, Options
from tools import qwen_image21_sparse_worker as worker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "dense", "fixed", "rules", "jev"),
        default=["off", "dense", "fixed", "rules", "jev"],
    )
    parser.add_argument("--allow-cloud", action="store_true")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--keep", type=float, default=25)
    parser.add_argument("--max-calls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--precision", choices=("w4a8", "int8", "bf16"), default="w4a8")
    parser.add_argument("--memory-mode", choices=("gpu", "offload"), default="gpu")
    parser.add_argument(
        "--prompt",
        default="A small red ceramic teapot on a wooden table next to a clear glass of tea, "
        "a blue linen cloth, soft window light, realistic product photograph. "
        'A cream card on the table clearly reads "TEA TIME" in dark letters.',
    )
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("repeats must be positive")
    if "jev" in args.modes and not args.allow_cloud:
        parser.error("Select --allow-cloud explicitly to send aggregate statistics to Jev")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if "jev" in args.modes:
        enable_saved_key()
    import torch

    from modules_forge.qwen_image21.core import runtime_lock

    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "protocol": "one warm-up; identical settings; forward then reverse mode order; resident weights",
        "settings": {k: v for k, v in vars(args).items() if k not in {"output", "allow_cloud"}},
        "runs": [],
    }
    lease = runtime_lock(ROOT / "models/Qwen-Image-2.1")
    try:
        schedule = [("off", 0)] + [
            (mode, n + 1)
            for n in range(args.repeats)
            for mode in (args.modes if n % 2 == 0 else list(reversed(args.modes)))
        ]
        for mode, repetition in schedule:
            label = "warmup" if not repetition else f"{mode}-{repetition}"
            directory = args.output / label
            directory.mkdir(exist_ok=False)
            options = Options(mode=mode, keep_percent=args.keep, max_calls=args.max_calls, timeout=10)
            os.environ[OPTIONS_ENV] = json.dumps(asdict(options))
            request = {
                "prompt": args.prompt,
                "width": args.width,
                "height": args.height,
                "steps": args.steps,
                "seed": args.seed,
                "precision": args.precision,
                "memory_mode": args.memory_mode,
                "input_images": [],
                "rewrite_prompt": False,
                "transparent": False,
            }
            worker.base._atomic_json(directory / "request.json", request)
            payload = {"job_dir": str(directory), "model_path": str(ROOT / "models/Qwen-Image-2.1/model")}
            print(f"BENCHMARK_START {label}", flush=True)  # noqa: T201
            started = time.perf_counter()
            result = worker.resident_run(payload)
            elapsed = time.perf_counter() - started
            metadata = result["metadata"]
            if mode == "jev" and args.max_calls:
                records = [
                    json.loads(line)
                    for line in Path(metadata["sparse_experiment"]["log_path"]).read_text(encoding="utf-8").splitlines()
                ]
                if not any(row["event"] == "decision" and row.get("diagnostics") for row in records):
                    raise RuntimeError("Qwen did not receive a valid Jev decision")
            row = {
                "label": label,
                "mode": mode,
                "repetition": repetition,
                "wall_seconds": elapsed,
                "output_path": result["output_path"],
                "timings": metadata["timings"],
                "memory": metadata["memory"],
                "sparse": metadata["sparse_experiment"],
                "reused_model": metadata["reused_model"],
            }
            report["runs"].append(row)
            worker.base._atomic_json(args.output / "benchmark.json", report)
            print("BENCHMARK_RESULT " + json.dumps(row), flush=True)  # noqa: T201
        report["medians"] = {
            mode: statistics.median(r["wall_seconds"] for r in report["runs"] if r["mode"] == mode and r["repetition"])
            for mode in args.modes
        }
        worker.base._atomic_json(args.output / "benchmark.json", report)
    finally:
        worker.base.clear_runtime()
        lease.close()


if __name__ == "__main__":
    main()
