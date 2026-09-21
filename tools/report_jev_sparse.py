"""Summarize experiment JSONL records without claiming unpaired speedups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Experiment log exceeds 32 MiB")

    def invalid_constant(value):
        raise ValueError("Nonfinite log value")

    records = []
    incomplete = False
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line, parse_constant=invalid_constant))
        except ValueError:
            incomplete = True
    if not records or not isinstance(records[0], dict) or records[0].get("event") != "begin":
        raise ValueError("Missing begin record")
    first = records[0]
    if any(
        not isinstance(row, dict) or row.get("run_id") != first.get("run_id") or row.get("schema") != 1
        for row in records
    ):
        raise ValueError("Mixed or unsupported log records")
    ends = [row for row in records if row.get("event") == "end"]
    last = ends[-1] if ends else {}
    settings = first.get("settings", {})
    target = first.get("target")
    counts = last.get("attention_calls", {})
    if target == "h3":
        counts = {}
        for row in records:
            for reason in row.get("actual_attention", {}).values():
                counts[reason] = counts.get(reason, 0) + 1
    return {
        "file": str(path),
        "run_id": first["run_id"],
        "target": target,
        "mode": settings.get("mode"),
        "status": "incomplete_log" if incomplete else last.get("status", "incomplete"),
        "timing_scope": last.get("timing_scope", settings.get("timing_scope")),
        "measured_seconds": last.get("model_seconds") if target == "anima" else last.get("elapsed_seconds"),
        "api_calls": last.get("api_calls"),
        "api_wait_seconds": last.get("api_wait_seconds"),
        "attention_calls": counts,
        "prompt_sha256": first.get("prompt_sha256"),
        "generation_context": next((row for row in records if row.get("event") == "generation_context"), None),
        "settings": settings,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="JSONL file or outputs/jev-sparse directory")
    args = parser.parse_args()
    paths = sorted(args.path.glob("*.jsonl")) if args.path.is_dir() else [args.path]
    reports = []
    for path in paths:
        try:
            reports.append(summarize(path))
        except (OSError, ValueError, TypeError, AttributeError, KeyError) as exc:
            reports.append({"file": str(path), "status": "invalid_log", "error_type": type(exc).__name__})
    print(
        json.dumps(
            {
                "note": "Measured scopes differ: H3 sampling only; Anima summed instrumented model evaluations. Match models, prompts, media, seeds, sizes and runtime before comparing. No automatic speedup claim.",
                "runs": reports,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
