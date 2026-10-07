"""Resident worker entry point; publish partial decisions after every image."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.clef.core import RUNTIME, atomic_json, export_csv  # noqa: E402

_RUNNER = None
_KEY = None


def process_run(directory, runner):
    from PIL import Image, ImageOps

    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    if (directory / "loading.json").is_file():
        result["loading"] = json.loads((directory / "loading.json").read_text(encoding="utf-8"))
    start = time.perf_counter()
    for item in result["items"]:
        if item["status"] != "pending":
            continue
        if (directory / "cancel").exists():
            item["status"] = "cancelled"
        else:
            try:
                begin = time.perf_counter()
                if item.get("kind") == "record":
                    state = json.loads(Path(item["path"]).read_text(encoding="utf-8"))["state"]
                    data = runner.decide({**request, "state": state})
                elif item["path"]:
                    with Image.open(item["path"]) as image:
                        data = runner.decide(request, ImageOps.exif_transpose(image).convert("RGB"))
                        item["source_size"] = list(image.size)
                else:
                    data = runner.decide(request)
                item.update(data, status="done", seconds=time.perf_counter() - begin)
                print(f"判定済み: {item['name']} ({item['seconds']:.2f}s)", flush=True)  # noqa: T201
            except Exception as exc:
                item.update(status="error", error=str(exc))
                print(f"{item['name']}: {exc}", flush=True)  # noqa: T201
        result["seconds"] = time.perf_counter() - start
        atomic_json(directory / "result.json", result)
    result["status"] = "complete" if all(x["status"] == "done" for x in result["items"]) else "partial"
    atomic_json(directory / "result.json", result)
    (directory / "results.csv").write_text(export_csv(result), encoding="utf-8-sig")
    return result


def resident_run(payload):
    global _RUNNER, _KEY
    from modules_forge.clef.service import process_identity

    directory = Path(payload["job_dir"])
    atomic_json(directory / "worker-process.json", process_identity())
    import torch

    from modules_forge.clef.runtime import Runner

    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    key = (payload["root"], request["profile"])
    if key != _KEY:
        if _RUNNER is not None:
            _RUNNER.close()
        _RUNNER = None
        _KEY = None
        begin = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        _RUNNER = Runner(Path(payload["root"]), request["profile"])
        _KEY = key
        atomic_json(
            directory / "loading.json",
            {
                "seconds": time.perf_counter() - begin,
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            },
        )
    else:
        atomic_json(directory / "loading.json", {"seconds": 0, "reused": True})
    atomic_json(directory / "progress.json", {"message": "判定しています。"})
    result = process_run(directory, _RUNNER)
    result["quantized_cache"] = (
        {"status": "ready", "path": str(_RUNNER.saved)} if _RUNNER.saved else {"status": "not-applicable"}
    )
    atomic_json(directory / "result.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=RUNTIME)
    args = parser.parse_args()
    resident_run({"root": str(args.root), "job_dir": str(args.job_dir)})
