"""Compare Qwen cached-target SDPA block batching on fixed synthetic inputs.

Run with the existing Qwen worker Python; no model weights or network calls.
CPU mode checks numerical equivalence, not GPU performance. CUDA Graph capture
is opt-in and covers this attention function only, with fixed shape/keep/budget.
It does not capture full Qwen decode, its CPU metadata operations, offload hooks,
or Jev requests. See https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.jev_sparse.qwen21 import _gather_batch_plan, block_gather_attention  # noqa: E402


def measure(call, torch, device, iterations):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    else:
        allocated = reserved = begin = end = None
    wall_started = time.perf_counter()
    if begin is not None:
        begin.record()
    host_started = time.perf_counter()
    for _ in range(iterations):
        result = call()
    host_seconds = time.perf_counter() - host_started
    cuda_seconds = None
    if end is not None:
        end.record()
        end.synchronize()
        cuda_seconds = begin.elapsed_time(end) / 1000
    wall_seconds = time.perf_counter() - wall_started
    memory = None
    if device.type == "cuda":
        memory = {
            "baseline_allocated_bytes": allocated,
            "baseline_reserved_bytes": reserved,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "peak_extra_allocated_bytes": torch.cuda.max_memory_allocated(device) - allocated,
        }
    return {
        "synchronized_wall_seconds": wall_seconds / iterations,
        "host_dispatch_seconds": host_seconds / iterations,
        "cuda_event_seconds": None if cuda_seconds is None else cuda_seconds / iterations,
        "memory": memory,
    }, result


def difference(actual, expected, torch, dtype):
    atol, rtol = (2e-3, 2e-2) if dtype != torch.float32 else (3e-6, 3e-5)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    delta = (actual.float() - expected.float()).abs()
    return {
        "max_abs_difference": delta.max().item(),
        "mean_abs_difference": delta.mean().item(),
        "atol": atol,
        "rtol": rtol,
    }


def capture(call, torch):
    # PyTorch requires warmup on a side stream before capture. Persistent input
    # tensors are retained by the caller and the captured output by this closure.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()

    def replay():
        graph.replay()
        return output

    return replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--prefix", type=int, default=512)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--block-size", type=int, choices=(64, 128, 256, 512), default=256)
    parser.add_argument("--keep", type=float, default=50)
    parser.add_argument("--query-batch-blocks", type=int, default=4)
    parser.add_argument("--workspace-mb", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if any(
        getattr(args, key) < 1
        for key in ("length", "heads", "head_dim", "batch", "repeats", "iterations", "cpu_threads")
    ):
        parser.error("Shapes, repeats, iterations and CPU threads must be positive")
    if args.prefix < 0 or args.warmup < 0:
        parser.error("Prefix and warmup must be nonnegative")
    if not 1 <= args.query_batch_blocks <= 16 or not 1 <= args.workspace_mb <= 1024:
        parser.error("Query block batch must be 1..16 and workspace must be 1..1024 MiB")
    if not math.isfinite(args.keep) or not 0 < args.keep <= 100:
        parser.error("Keep must be finite in (0, 100]")
    if args.cuda_graph and args.device != "cuda":
        parser.error("--cuda-graph requires --device cuda")
    lease = None
    if args.device == "cuda":
        from modules_forge.qwen_image21.core import runtime_lock

        lease = runtime_lock(ROOT / "models/Qwen-Image-2.1")
    try:
        benchmark(args)
    finally:
        if lease is not None:
            lease.close()


def benchmark(args):
    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(20260922)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this interpreter")
    dtype = getattr(torch, args.dtype or ("bfloat16" if device.type == "cuda" else "float32"))
    query = torch.randn(args.batch, args.length, args.heads, args.head_dim, device=device, dtype=dtype)
    key, value = [
        torch.randn(args.batch, args.prefix + args.length, args.heads, args.head_dim, device=device, dtype=dtype)
        for _ in range(2)
    ]
    valid = torch.ones(args.batch, 1, 1, args.prefix + args.length, device=device, dtype=torch.bool)
    valid[..., : args.prefix : 7] = False

    def attention(blocks):
        return block_gather_attention(
            query,
            key,
            value,
            args.keep,
            args.prefix,
            valid,
            args.block_size,
            query_batch_blocks=blocks,
            max_batch_workspace_mb=args.workspace_mb,
        )

    variants = {"one_block": lambda: attention(1), "batched": lambda: attention(args.query_batch_blocks)}
    report = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "platform": platform.platform(),
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "dtype": str(dtype),
        "settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "scope": "fixed synthetic cached-target attention; one-block vs batched at the same sparsity; not image quality or full-model speed",
        "timing_note": "Host dispatch excludes final synchronization. CUDA events include stream launch gaps; they are not summed kernel execution time.",
        "workspace_note": "Budget bounds explicit batched gather tensors; base tensors, SDPA workspace and CUDA Graph pools are separate.",
        "runs": [],
        "cuda_graph": {"status": "not_requested"},
    }
    nblocks = math.ceil(args.length / args.block_size)
    selected_tokens = max(1, math.ceil(nblocks * args.keep / 100)) * args.block_size
    report["batch_plans"] = {}
    for name, limit in (("one_block", 1), ("batched", args.query_batch_blocks)):
        if args.keep >= 100:
            report["batch_plans"][name] = {"dense": True, "sdpa_calls": 1}
            continue
        head_batch, block_batch = _gather_batch_plan(
            args.batch,
            args.heads,
            args.head_dim,
            args.block_size,
            selected_tokens,
            args.prefix,
            query.element_size(),
            min(nblocks, limit),
            args.workspace_mb,
        )
        report["batch_plans"][name] = {
            "head_batch": head_batch,
            "query_block_batch": block_batch,
            "sdpa_calls": math.ceil(args.heads / head_batch) * math.ceil(nblocks / block_batch),
        }
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        baseline = variants["one_block"]()
        report["numerical_comparison"] = difference(variants["batched"](), baseline, torch, dtype)
        if args.cuda_graph:
            try:
                replay = capture(variants["batched"], torch)
            except RuntimeError as exc:
                report["cuda_graph"] = {
                    "status": "capture_failed",
                    "error_type": type(exc).__name__,
                    "reason": str(exc)[:500],
                }
            else:
                report["cuda_graph"] = {
                    "status": "captured_attention_only",
                    "numerical_comparison": difference(replay(), baseline, torch, dtype),
                    "constraints": "shape, addresses, keep and workspace plan remain fixed; graph excludes Jev, offload and full transformer metadata",
                }
                variants["batched_cuda_graph"] = replay
        for call in variants.values():
            for _ in range(args.warmup):
                call()
        for repetition in range(args.repeats):
            names = list(variants) if repetition % 2 == 0 else list(reversed(variants))
            for name in names:
                measured, output = measure(variants[name], torch, device, args.iterations)
                difference(output, baseline, torch, dtype)
                report["runs"].append({"variant": name, "repetition": repetition + 1, **measured})
        if "batched_cuda_graph" in variants:
            # Confirm replay reads updated tensor contents (including new top-k
            # selections), instead of merely repeating the captured output.
            query.normal_()
            report["cuda_graph"]["changed_input_comparison"] = difference(
                variants["batched_cuda_graph"](), variants["batched"](), torch, dtype
            )
        if args.profile:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
                variants["batched"]()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            trace = args.output.with_suffix(".trace.json")
            profile.export_chrome_trace(str(trace))
            report["profile_trace"] = str(trace)
            kernels = [event for event in profile.events() if str(event.device_type).endswith("CUDA")]
            report["profile_kernel_summary"] = {
                "status": "captured" if kernels else "cpu_only" if device.type == "cpu" else "cuda_events_unavailable",
                "cuda_kernel_events": len(kernels),
                "summed_cuda_kernel_seconds": (
                    sum(event.self_device_time_total for event in kernels) / 1e6 if kernels else None
                ),
                "scope": "profiled single batched attention; kernel sums may double-count concurrent work",
            }
    report["medians"] = {
        name: {
            field: statistics.median(row[field] for row in report["runs"] if row["variant"] == name)
            for field in ("synchronized_wall_seconds", "host_dispatch_seconds", "cuda_event_seconds")
            if field != "cuda_event_seconds" or device.type == "cuda"
        }
        for name in variants
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(  # noqa: T201
        json.dumps(
            {
                "report": str(args.output),
                "medians": report["medians"],
                "numerical_comparison": report["numerical_comparison"],
                "cuda_graph_status": report["cuda_graph"]["status"],
            }
        )
    )


if __name__ == "__main__":
    main()
