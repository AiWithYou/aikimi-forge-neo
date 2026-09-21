"""Real H3 4-step benchmark through the existing managed ComfyUI bridge."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge import minimax_h3_bridge as bridge
from modules_forge.jev_sparse import h3_integration
from modules_forge.minimax_h3_acceleration import H3Acceleration
from tools.setup_jev_sparse import install_pack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--port", type=int, default=8192)
    parser.add_argument("--runtime-profile", choices=sorted(bridge.RUNTIME_PROFILES), default="ram")
    parser.add_argument("--clip-cache", choices=("auto", "off"), default="auto")
    parser.add_argument("--allow-cloud", action="store_true")
    args = parser.parse_args()
    if not args.allow_cloud:
        parser.error("--allow-cloud is required to send aggregate statistics to Jev")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / "repositories/minimax-h3/ComfyUI"
    logs = ROOT / "outputs/jev-sparse"
    install_pack(runtime)
    h3_integration.install()
    url = f"http://127.0.0.1:{args.port}"
    report = {"runs": [], "protocol": "5-second preview; fixed seed; separate warm-up; forward/reversed order"}
    try:
        # One owned process can execute all comparison modes, including Jev.
        bridge.start_runtime(
            runtime,
            url,
            root / "logs",
            runtime_profile=args.runtime_profile,
            acceleration=H3Acceleration(
                model_variant="fused_turbo", video_vae="int8", attention="h3_jev", clip_cache=args.clip_cache
            ),
        )
        modes = ["dense", "fixed5", "fixed10", "jev"]
        schedule = [("dense", 0)] + [
            (mode, n + 1) for n in range(args.repeats) for mode in (modes if n % 2 == 0 else list(reversed(modes)))
        ]
        for mode, repetition in schedule:
            label = "warmup" if not repetition else f"{mode}-{repetition}"
            output = root / label
            request = bridge.H3Request(
                mode=bridge.MODE_TEXT,
                prompt="A single red paper boat drifts slowly across a shallow rain puddle in a quiet stone courtyard. "
                "Locked low camera, realistic ripples and soft overcast light. Gentle rain and water sounds, "
                "no speech, no music, no people, no text.",
                aspect="16:9",
                quality="preview",
                duration_seconds=5,
                steps=4,
                seed=20260921,
                acceleration=H3Acceleration(
                    model_variant="fused_turbo", video_vae="int8", attention="h3_" + mode, clip_cache=args.clip_cache
                ),
            )
            report["settings"] = asdict(request)
            before = set(logs.glob("h3-*.jsonl"))
            print(f"BENCHMARK_START {label}", flush=True)  # noqa: T201
            started = time.perf_counter()
            final = None
            last_stage = None
            for event in bridge.run_generation(
                request, runtime, url, root / "logs", output, runtime_profile=args.runtime_profile, poll_seconds=1
            ):
                if event["stage"] != last_stage:
                    print(label, event["stage"], flush=True)  # noqa: T201
                    last_stage = event["stage"]
                if event["stage"] == "complete":
                    final = event
            if final is None:
                raise RuntimeError("H3 did not finish")
            elapsed = time.perf_counter() - started
            new_logs = set(logs.glob("h3-*.jsonl")) - before
            if len(new_logs) != 1:
                raise RuntimeError("H3 did not execute exactly one new experiment; cached output is not a timing")
            source = new_logs.pop()
            records = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
            end = next(row for row in reversed(records) if row["event"] == "end")
            steps = [row for row in records if row["event"] == "step"]
            sparse_calls = sum(
                value == "native_sla_producer" for row in steps for value in row["actual_attention"].values()
            )
            if mode != "dense" and not sparse_calls:
                raise RuntimeError("H3 selected Sparse but did not use the native producer")
            if mode == "jev" and not any(row["event"] == "decision" and row.get("source") == "jev" for row in records):
                raise RuntimeError("H3 fell back without a valid Jev decision")
            row = {
                "label": label,
                "mode": mode,
                "repetition": repetition,
                "wall_seconds": elapsed,
                "sampling_seconds": end["elapsed_seconds"],
                "api_calls": end["api_calls"],
                "api_wait_seconds": end["api_wait_seconds"],
                "sparse_calls": sparse_calls,
                "output_path": final["path"],
                "log_path": str(source),
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
        bridge._release_retained_runtime(url)
        h3_integration.uninstall()


if __name__ == "__main__":
    main()
