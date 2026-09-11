"""Create HyperWeave/Lanczos contact sheets, crops, maps, and metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
EXTENSION = ROOT / "extensions-builtin" / "hyperweave"
sys.path.insert(0, str(EXTENSION))

from hyperweave.comparison import build_comparison_artifacts


def parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("candidate must be LABEL=PATH")
    label, path = value.split("=", 1)
    resolved = Path(path).expanduser().resolve()
    if not label.strip() or not resolved.is_file():
        raise argparse.ArgumentTypeError(f"invalid candidate: {value}")
    return label.strip(), resolved


def parse_crop(value: str) -> tuple[str, tuple[int, int, int, int]]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("crop must be NAME=x0,y0,x1,y1")
    name, coordinates = value.split("=", 1)
    try:
        box = tuple(int(item.strip()) for item in coordinates.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
        raise argparse.ArgumentTypeError("crop box is invalid")
    return name.strip(), box


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument(
        "--candidate", action="append", required=True, type=parse_candidate
    )
    parser.add_argument("--crop", action="append", default=[], type=parse_crop)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    with Image.open(args.source) as opened:
        source = opened.copy()
    candidates = {}
    for label, path in args.candidate:
        with Image.open(path) as opened:
            candidates[label] = opened.copy()
    report = build_comparison_artifacts(
        source,
        candidates,
        args.output,
        crop_boxes=dict(args.crop),
    )
    sys.stdout.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
