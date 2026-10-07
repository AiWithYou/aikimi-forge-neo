"""Download or verify the optional Qwen Image 2.1 Consistency LoRA."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules_forge.qwen_image21.consistency_lora import install, installed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=["1500", "2000"], default="1500")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    runtime = ROOT / "models" / "Qwen-Image-2.1"
    info = installed(runtime, args.version, verify=True) if args.verify else install(runtime, args.version)
    sys.stdout.write(f"Consistency {args.version} verified: {info['path']}\n")


if __name__ == "__main__":
    main()
