"""Download or verify the small Outpaint adapter used by the existing Qwen runtime."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.outpaint_lora import install, installed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / "models" / "Qwen-Image-2.1"
    info = installed(runtime, args.version, verify=True) if args.verify else install(runtime, args.version)
    sys.stdout.write(f"Outpaint {args.version} verified: {info['path']}\n")


if __name__ == "__main__":
    main()
