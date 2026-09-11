"""選択したComfyUIへ固定版CLIPCachedを導入する。モデルは取得しない。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def install(runtime_root: Path) -> Path:
    from modules_forge.minimax_h3_clipcache import (
        blob_matches,
        clipcache_directory,
        clipcache_manifest,
        verify_clipcache_directory,
    )

    manifest = clipcache_manifest()
    target = clipcache_directory(runtime_root)
    if target.exists():
        verify_clipcache_directory(target)
        return target
    custom = target.parent
    custom.mkdir(exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-clipcache-", dir=custom))
    origin = f"https://raw.githubusercontent.com/{manifest['repository']}/{manifest['revision']}"

    def download(entry):
        request = Request(  # noqa: S310 -- URLは固定HTTPS配布元から構築する。
            f"{origin}/{quote(entry['path'], safe='/')}", headers={"User-Agent": "Aikimi-Neo-CLIPCached-Installer"}
        )
        with urlopen(request, timeout=60) as response:  # noqa: S310 -- 固定したHTTPS配布元だけを使用する。
            content = response.read(entry["bytes"] + 1)
        if not blob_matches(content, entry):
            raise ValueError(f"CLIPCachedファイルの完全性が一致しません: {entry['path']}")
        path = stage / entry["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(download, manifest["files"]))
        verify_clipcache_directory(stage)
        os.rename(stage, target)
    finally:
        if stage.exists():
            if not stage.resolve().is_relative_to(custom.resolve()):
                raise ValueError("一時導入フォルダーがComfyUIのcustom_nodes外にあります。")
            shutil.rmtree(stage)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    arguments = parser.parse_args()
    target = install(arguments.runtime_root)
    sys.stdout.write(f"CLIPCachedの固定版を確認しました: {target}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
