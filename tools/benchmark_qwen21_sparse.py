"""Compare complete Qwen 2.1 images across cases, seeds, fixed keeps and Jev replay.

Use --dry-run to save the complete matrix without loading Torch, models or Jev.
For generation use models/Qwen-Image-2.1/worker-env/Scripts/python.exe.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import sparse_benchmark as bench

REPLAY_ENV = "AIKIMI_JEV_REPLAY_LOGS"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--modes",
        nargs="+",
        choices=("off", "dense", "fixed", "rules", "jev", "replay"),
        default=["off", "dense", "fixed", "rules", "jev", "replay"],
    )
    result.add_argument("--allow-cloud", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    result.add_argument("--case-manifest", type=Path)
    result.add_argument("--cases", nargs="+", help="Selected manifest case IDs")
    result.add_argument("--reference-images", type=Path, nargs="+", default=[])
    result.add_argument("--replay-from", type=Path, help="A prior schema=2 benchmark.json with matching live logs")
    result.add_argument("--width", type=int, default=1024)
    result.add_argument("--height", type=int, default=1024)
    result.add_argument("--steps", type=int, default=20)
    result.add_argument("--repeats", type=int, default=2)
    result.add_argument(
        "--keep", type=float, help="Legacy single fixed keep, or initial adaptive keep with --fixed-keeps"
    )
    result.add_argument("--fixed-keeps", type=float, nargs="+")
    result.add_argument("--minimum-tokens", type=int, default=1024)
    result.add_argument("--block-size", type=int, choices=(64, 128, 256, 512), default=256)
    result.add_argument("--query-batch-blocks", type=int, default=4)
    result.add_argument("--max-batch-workspace-mb", type=int, default=256)
    result.add_argument("--max-calls", type=int, default=1)
    result.add_argument("--job-max-calls", type=int, default=0, help="Extra job-wide cap; 0 leaves this cap unset")
    result.add_argument("--job-max-wait-seconds", type=float, default=0)
    result.add_argument("--decision-cadence", choices=("legacy", "once", "interval", "step"), default="once")
    result.add_argument("--decision-interval", type=int, default=4)
    seeds = result.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, help="Compatibility option for a single seed")
    seeds.add_argument("--seeds", type=int, nargs="+")
    result.add_argument("--precision", choices=("w4a8", "int8", "bf16"), default="w4a8")
    result.add_argument("--memory-mode", choices=("gpu", "offload"), default="gpu")
    result.add_argument("--prompt", help="One custom text case instead of the built-in case suite")
    return result


def make_plan(args):
    args.fixed_keeps = args.fixed_keeps or ([args.keep] if args.keep is not None else [25, 50, 75])
    args.keep = 25 if args.keep is None else args.keep
    if any(size < 256 or size > 4096 or size % 32 for size in (args.width, args.height)):
        raise ValueError("Dimensions must be 256..4096 and divisible by 32")
    if not 1 <= args.steps <= 100 or not 1 <= args.decision_interval <= 100:
        raise ValueError("steps and decision-interval must be 1..100")
    if not 0 <= args.max_calls <= 8 or not 64 <= args.minimum_tokens <= 1048576:
        raise ValueError("max-calls must be 0..8 and minimum-tokens must be 64..1048576")
    if not 1 <= args.query_batch_blocks <= 16 or not 1 <= args.max_batch_workspace_mb <= 1024:
        raise ValueError("query-batch-blocks must be 1..16 and max-batch-workspace-mb must be 1..1024")
    if (
        not 0 <= args.job_max_calls <= 1000
        or not math.isfinite(args.job_max_wait_seconds)
        or not 0 <= args.job_max_wait_seconds <= 3600
    ):
        raise ValueError("job-max-calls must be 0..1000 and job-max-wait-seconds must be 0..3600")
    if any(mode in {"jev", "replay"} for mode in args.modes) and not args.max_calls:
        raise ValueError("Jev/replay comparison requires a nonzero decision budget")
    cases = bench.load_cases(
        args.case_manifest, reference_images=args.reference_images, selected=args.cases, prompt=args.prompt
    )
    settings = {
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "precision": args.precision,
        "memory_mode": args.memory_mode,
        "minimum_tokens": args.minimum_tokens,
        "block_size": args.block_size,
        "query_batch_blocks": args.query_batch_blocks,
        "max_batch_workspace_mb": args.max_batch_workspace_mb,
        "max_calls": args.max_calls,
        "job_max_calls": args.job_max_calls,
        "job_max_wait_seconds": args.job_max_wait_seconds,
        "decision_cadence": args.decision_cadence,
        "decision_interval": args.decision_interval,
        "timeout": 10,
        "rewrite_prompt": False,
        "true_cfg_scale": 1.0,
        "use_kv_cache": True,
    }
    return bench.build_plan(
        model="qwen21",
        stage="native",
        cases=cases,
        seeds=[args.seed] if args.seed is not None else (args.seeds or [20260921, 20260922]),
        dimensions=[(args.width, args.height)],
        variants=bench.build_variants(args.modes, args.fixed_keeps, args.keep),
        repeats=args.repeats,
        settings=settings,
        replay_from=args.replay_from,
    )


class QwenRunner:
    def __init__(self, args, worker, options_type, torch):
        self.args, self.worker, self.options_type, self.torch = args, worker, options_type, torch

    def prepare(self, group):
        # The first scheduled OFF image loads the resident model and warms this shape.
        return None

    def generate(self, group, spec, directory, replay):
        args, identity, case = self.args, group["identity"], group["case"]
        if [bench.digest(path) for path in case["input_images"]] != identity["input_sha256"]:
            raise RuntimeError("Reference images changed after benchmark planning")
        runtime_mode = "jev" if replay else spec["mode"]
        options = self.options_type(
            mode=runtime_mode,
            keep_percent=spec["keep"],
            max_calls=args.max_calls,
            timeout=10,
            min_tokens=args.minimum_tokens,
            block_size=args.block_size,
            query_batch_blocks=args.query_batch_blocks,
            max_batch_workspace_mb=args.max_batch_workspace_mb,
            job_max_calls=args.job_max_calls,
            job_max_wait_seconds=args.job_max_wait_seconds,
            decision_cadence=args.decision_cadence,
            update_interval=args.decision_interval,
        )
        options.validate()
        request = {
            "prompt": case["prompt"],
            "width": identity["width"],
            "height": identity["height"],
            "steps": args.steps,
            "seed": identity["seed"],
            "precision": args.precision,
            "memory_mode": args.memory_mode,
            "input_images": list(case["input_images"]),
            "rewrite_prompt": False,
            "transparent": case["transparent"],
        }
        bench.atomic_json(directory / "request.json", request)
        payload = {
            "job_dir": str(directory),
            "model_path": str(ROOT / "models/Qwen-Image-2.1/model"),
            "sparse_experiment": asdict(options),
        }
        previous_replay = os.environ.pop(REPLAY_ENV, None)
        try:
            if replay:
                os.environ[REPLAY_ENV] = json.dumps(replay["logs"])
            self.torch.cuda.synchronize()
            started = time.perf_counter()
            result = self.worker.resident_run(payload)
            self.torch.cuda.synchronize()
            artifact = bench.validate_output(
                result["output_path"],
                transparent=case["transparent"],
                expected_size=(identity["width"], identity["height"]),
            )
            elapsed = time.perf_counter() - started
        finally:
            os.environ.pop(REPLAY_ENV, None)
            if previous_replay is not None:
                os.environ[REPLAY_ENV] = previous_replay
        metadata = result["metadata"]
        if spec.get("repetition", 0) > 0 and metadata.get("reused_model") is not True:
            raise RuntimeError("The warmed resident model was reloaded during a measured run")
        fingerprint = {
            key: metadata.get(key)
            for key in ("model", "model_revision", "diffusers_revision", "compute_dtype", "versions")
        }
        if "runtime" in identity and identity["runtime"] != fingerprint:
            raise RuntimeError("Qwen model or dependency identity changed during the benchmark")
        identity["runtime"] = fingerprint
        sparse = metadata["sparse_experiment"]
        logs = {}
        if runtime_mode != "off":
            if sparse.get("status") != "completed" or not sparse.get("log_path"):
                raise RuntimeError("Qwen sparse instrumentation did not complete")
            log = bench.read_log(sparse["log_path"])
            actual_sparse = log["end"].get("attention_calls", {}).get("sparse_target", 0)
            expected_sparse = runtime_mode in {"rules", "jev"} or (runtime_mode == "fixed" and spec["keep"] < 100)
            if expected_sparse and not actual_sparse:
                raise RuntimeError("No real sparse target attention calls were recorded")
            if runtime_mode == "jev":
                bench.validate_decisions(log, replay=bool(replay), kind="attention", cadence=args.decision_cadence)
            logs["attention"] = bench.public_log(log)
        timing_keys = (
            "total_wall_seconds",
            "evaluation_cpu_wall_seconds",
            "transformer_cpu_wall_seconds",
            "controller_wall_seconds",
            "api_wait_seconds",
            "cuda_event_seconds",
            "cuda_event_evaluations",
            "timing_scope",
            "cuda_event_scope",
            "model_seconds",
            "model_seconds_kind",
        )
        return {
            "wall_seconds": elapsed,
            "output_path": result["output_path"],
            "artifact": artifact,
            "timings": metadata["timings"],
            "timing_breakdown": {key: sparse[key] for key in timing_keys if key in sparse},
            "memory": metadata["memory"],
            "sparse": sparse,
            "logs": logs,
            "reused_model": metadata["reused_model"],
        }


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
    lease, worker = None, None
    try:
        import torch

        from modules_forge.jev_sparse.credentials import enable_saved_key
        from modules_forge.jev_sparse.qwen21 import Options
        from modules_forge.qwen_image21.core import runtime_lock
        from tools import qwen_image21_sparse_worker as worker

        if plan["planned_live_generations"]:
            enable_saved_key()
        plan["runtime"] = {"device": torch.cuda.get_device_name(), "torch": torch.__version__}
        lease = runtime_lock(ROOT / "models/Qwen-Image-2.1")
        bench.execute_plan(plan, root, QwenRunner(args, worker, Options, torch))
    except BaseException as exc:
        if not (root / "benchmark.json").exists():
            bench.write_report(
                root, {**plan, "status": "failed", "runs": [], "error_type": type(exc).__name__, "error": str(exc)}
            )
        raise
    finally:
        try:
            if worker is not None:
                worker.base.clear_runtime()
        finally:
            if lease is not None:
                lease.close()


if __name__ == "__main__":
    main()
