"""Prepare persistent Qwen INT8/W4A8 checkpoints in the isolated worker env."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "models/Qwen-Image-2.1")
    parser.add_argument("--precision", nargs="+", choices=("int8", "w4a8"), default=["w4a8", "int8"])
    parser.add_argument("--verify", action="store_true", help="Verify all saved component SHA-256 values")
    parser.add_argument(
        "--reload", action="store_true", help="Release RAM and confirm the second load uses saved weights"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.dry_run:
        print(json.dumps({"root": str(root), "precision": args.precision, "save": str(root / "quantized")}, indent=2))  # noqa: T201
        return 0
    from modules_forge.qwen_image21.core import runtime_lock, runtime_manifest
    from modules_forge.qwen_image21.quantized_cache import cache_path, component_identity, manifest
    from tools import qwen_image21_worker as worker

    lease = runtime_lock(root)
    report = {"started": time.time(), "runs": []}
    job = root / "quantized" / "preparation"
    job.mkdir(parents=True, exist_ok=True)
    try:
        runtime_manifest(root)
        for precision in dict.fromkeys(args.precision):
            if args.verify:
                for component in ("transformer", "text_encoder"):
                    identity = component_identity(
                        root / "model",
                        component,
                        precision,
                        skip_modules=worker.INT8_SKIP_MODULES
                        if precision == "int8" and component == "transformer"
                        else (),
                    )
                    path = cache_path(root / "model", identity)
                    manifest(path, identity, verify_hashes=True)
                    report["runs"].append({"precision": precision, "component": component, "verified": str(path)})
                continue
            for attempt in range(2 if args.reload else 1):
                runtime = worker._load_runtime(root / "model", {"precision": precision, "memory_mode": "offload"}, job)
                row = {
                    "precision": precision,
                    "attempt": attempt + 1,
                    "load_seconds": runtime["load_seconds"],
                    "disk_cache": runtime["disk_cache"],
                    "int8_layers": runtime["int8_layers"],
                    "w4a8": runtime["w4a8"],
                }
                if attempt and any(value["status"] != "hit" for value in runtime["disk_cache"].values()):
                    raise RuntimeError("Reload unexpectedly rebuilt a quantized component")
                report["runs"].append(row)
                runtime = None
                worker.clear_runtime()
                worker._atomic_json(job / "report.json", report)
                print(json.dumps(row, ensure_ascii=False), flush=True)  # noqa: T201
    finally:
        worker.clear_runtime()
        lease.close()
    worker._atomic_json(job / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
