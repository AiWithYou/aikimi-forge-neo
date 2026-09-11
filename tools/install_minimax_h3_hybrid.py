"""指定したComfyUIへHybridWindowsの固定版を導入する。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install(runtime_root: Path) -> Path:
    from modules_forge import minimax_h3_hybrid as hybrid

    runtime_root = runtime_root.resolve(strict=True)
    if not (runtime_root / "main.py").is_file() or not (runtime_root / "comfy").is_dir():
        raise ValueError("ComfyUIのフォルダーを指定してください。")
    custom = runtime_root / "custom_nodes"
    custom.mkdir(exist_ok=True)
    target = custom / hybrid.PACK
    if target.exists():
        hybrid.verify_directory(target)
        return target
    stage = Path(tempfile.mkdtemp(prefix=".hybrid-install-", dir=custom))
    try:
        for entry in hybrid.manifest()["files"]:
            url = f"https://raw.githubusercontent.com/Jalen-Brunson/ComfyUI-HybridWindows/{hybrid.REVISION}/{entry['path']}"
            with urlopen(url, timeout=45) as response:  # noqa: S310 -- 配布元・commit・ファイル名は固定。
                content = response.read(entry["bytes"] + 1)
            if not hybrid.blob_matches(content, entry):
                raise ValueError(f"配布ファイルが固定版と一致しません: {entry['path']}")
            (stage / entry["path"]).write_bytes(content)
        hybrid.verify_directory(stage)
        os.rename(stage, target)
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to(custom.resolve()):
                raise ValueError("一時フォルダーがcustom_nodesの外にあります。")
            shutil.rmtree(stage)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    args = parser.parse_args()
    sys.stdout.write(f"HybridWindowsの固定版を導入しました: {install(args.runtime_root)}\n")
