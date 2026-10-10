"""Install or verify the fixed MiniMax H3 360° Orbit LoRA."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.minimax_h3_orbit_assets import MODEL_NAME, install, installed  # noqa: E402
from modules_forge.minimax_h3_runtime import configured_model_root  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify existing local bytes without downloading")
    args = parser.parse_args(argv)
    try:
        if args.verify:
            if not installed(ROOT):
                raise ValueError("360° Orbit LoRAが未導入、または配布版と一致しません。")
            path = configured_model_root(ROOT) / "loras" / MODEL_NAME
        else:
            path = install(ROOT)
    except (OSError, ValueError) as error:
        sys.stderr.write(f"Orbit LoRA: {error}\n")
        return 1
    sys.stdout.write(f"Orbit LoRA verified: {path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
