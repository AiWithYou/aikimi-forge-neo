"""Run a downloaded Clef quantized bundle, without Forge or source checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["flash-int8", "clef-24gb", "clef-16gb"], required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--state", default="添付画像を評価してください。")
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    from clef_runtime.core import TEMPLATES, validate_request
    from clef_runtime.runtime import Runner

    questions = json.loads(args.questions.read_text(encoding="utf-8")) if args.questions else TEMPLATES["画像の評価"]
    request = validate_request(
        {
            "profile": args.profile,
            "questions": questions,
            "state": args.state,
            "max_pixels": args.max_pixels,
            "max_length": args.max_length,
        }
    )
    runner = Runner.from_directory(Path(__file__).resolve().parent, args.profile)
    try:
        result = runner.decide(request, str(args.image.resolve()) if args.image else None)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)  # noqa: T201
    finally:
        runner.close()


if __name__ == "__main__":
    main()
