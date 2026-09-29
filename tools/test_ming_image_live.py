"""Generate a real Ming PNG through Neo's managed, local ComfyUI integration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.ming_image_studio import SERVER_URL, MingImageRequest, _bridge, run_generation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default='A cream and forest green botanical poster. A large fern, elegant headline "BOTANICA", subtitle "A quiet collection", generous margins, minimal editorial design.',
    )
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--transparent", action="store_true")
    parser.add_argument("--precision", choices=("int8", "w4a8"), default="int8")
    args = parser.parse_args(argv)
    request = MingImageRequest(
        args.prompt,
        transparent=args.transparent,
        width=args.width,
        height=args.height,
        steps=args.steps,
        seed=args.seed,
        precision=args.precision,
    )
    last = ""
    bridge = _bridge()
    if bridge.server_runtime_root(SERVER_URL) is not None:
        sys.stderr.write("既存のComfyUIを終了してから、実生成テストを実行してください。\n")
        return 1
    try:
        for event in run_generation(request):
            if event["stage"] != last:
                sys.stdout.write(event["stage"] + ": " + event["message"] + "\n")
                sys.stdout.flush()
            last = event["stage"]
            if last == "complete":
                sys.stdout.write(json.dumps(event, ensure_ascii=False, indent=2) + "\n")
                return 0
    except Exception as exc:
        sys.stderr.write(f"Ming generation failed: {exc}\n")
    finally:
        if bridge._MANAGED_PROCESS is not None:
            try:
                bridge._release_retained_runtime(SERVER_URL)
            except Exception as exc:
                sys.stderr.write(f"実行環境の終了を確認できません: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
