"""選択した背景除去モデルを事前取得する。省略時はBiRefNet標準版のみ。"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modules.background_removal import DEFAULT_MODEL, MODELS, download_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL)
    args = parser.parse_args()
    download_model(args.model, ROOT / "models" / "background_removal")


if __name__ == "__main__":
    main()
