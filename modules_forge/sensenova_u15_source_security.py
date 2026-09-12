"""SenseNova専用環境は監査済みの固定ソースと設定だけを読み込む。"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

from tools.aikimi_setup import SENSENOVA_RUNTIME_FILES, SENSENOVA_SOURCE_REVISION

_BYTECODE_CACHE = None


def validate_runtime_source(source: Path) -> None:
    source = source.resolve(strict=True)
    expected = {name: (digest, size) for name, digest, size in SENSENOVA_RUNTIME_FILES}
    for path in source.rglob("*"):
        if "__pycache__" in path.relative_to(source).parts:
            continue
        name = path.relative_to(source).as_posix()
        if path.is_symlink() or path.is_junction():
            raise RuntimeError("SenseNovaの推論コード・設定にリンクは使用できません。")
        if not path.is_dir() and name not in expected and name != ".sensenova_runtime_revision":
            raise RuntimeError(f"SenseNovaの固定ランタイムに未承認のファイルがあります: {name}")
    for name, (digest, size) in expected.items():
        path = source / name
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"SenseNovaの固定ファイルが不足または変更されています: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise RuntimeError(f"SenseNovaの固定ファイルのハッシュが一致しません: {name}")
    if (source / ".sensenova_runtime_revision").read_text(encoding="utf-8").strip() != SENSENOVA_SOURCE_REVISION:
        raise RuntimeError("SenseNovaの固定リビジョンが一致しません。")


def prepare_runtime_source(source: Path) -> None:
    global _BYTECODE_CACHE
    validate_runtime_source(source)
    # 配布物に紛れたpycを読まず、検証済みの.pyから読み込む。
    if _BYTECODE_CACHE is None:
        _BYTECODE_CACHE = tempfile.TemporaryDirectory(prefix="sensenova-bytecode-")
    sys.pycache_prefix = _BYTECODE_CACHE.name
    sys.dont_write_bytecode = True
    # Transformersをimportする前に設定。Hubから別の設定やコードを取得しない。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from modules.aikimi_security.model_dependencies import restrict_accelerate_checkpoint_loading

    restrict_accelerate_checkpoint_loading()
