"""Inspect or explicitly clear an admission stop, without accessing the GPU."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules_forge.gpu_fault import STATE  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "clear"))
    parser.add_argument("--acknowledge-recovery", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "status":
        print(json.dumps({"blocked": STATE.blocked()}))  # noqa: T201 -- CLI status.
    else:
        STATE.clear(acknowledge_recovery=args.acknowledge_recovery)
        print("停止の記録を解除しました。Forge Neoを新しく起動してください。")  # noqa: T201 -- CLI result.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
