"""Generate one real Nanosaur2 PNG through Neo's local ComfyUI integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.nanosaur2_studio import Nanosaur2Request, run_generation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default="newest, masterpiece, an explorer with a red scarf and silver goggles beside a tiny robot dinosaur, sunlit mossy forest, detailed illustration",
    )
    parser.add_argument("--negative", default="oldest, low quality, blurry, watermark, text")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--guidance", choices=("alternate", "cfg", "path_drop"), default="alternate")
    args = parser.parse_args(argv)
    request = Nanosaur2Request(
        args.prompt, args.negative, args.width, args.height, args.steps, args.cfg, args.seed, args.guidance
    )
    last_stage = ""
    try:
        for event in run_generation(request):
            stage = event["stage"]
            if stage != last_stage or stage == "complete":
                sys.stdout.write(f"{stage}: {event['message']}\n")
                sys.stdout.flush()
            last_stage = stage
            if stage == "complete":
                sys.stdout.write(
                    json.dumps(
                        {"image": event["path"], "elapsed": event["elapsed"], "metadata": event["metadata"]},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
                return 0
    except Exception as exc:
        sys.stderr.write(f"Nanosaur2 live generation failed: {exc}\n")
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
