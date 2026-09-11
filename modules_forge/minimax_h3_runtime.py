"""Neoが管理するH3実行環境とモデル保存先。GPUや外部環境には接続しない。"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRECTORY = Path("repositories/minimax-h3/ComfyUI")
MODEL_DIRECTORIES = ("diffusion_models", "text_encoders", "vae", "loras", "model_patches")
SERVER_URL = "http://127.0.0.1:8189"


def managed_runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return Path(repository_root).resolve() / RUNTIME_DIRECTORY


def local_directory(value: str | Path) -> Path:
    """共有先も含め、解決前後のローカルディスクを確認する。"""

    def check(raw):
        if "://" in raw or raw.replace("/", "\\").startswith("\\\\"):
            raise ValueError("ローカルディスクのフォルダーを指定してください。")
        path = Path(raw)
        if os.name == "nt" and path.drive:
            kind = ctypes.windll.kernel32.GetDriveTypeW(path.drive + "\\")
            if kind in {0, 1, 4}:
                raise ValueError("利用できるローカルディスクを指定してください。")

    check(str(value))
    resolved = Path(value).expanduser().resolve()
    check(str(resolved))
    if resolved.exists() and not resolved.is_dir():
        raise ValueError("ファイルではなくフォルダーを指定してください。")
    return resolved


def installed_runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path | None:
    root = managed_runtime_root(repository_root)
    if not (root.parent / "setup.json").is_file():
        return None
    if not (root / "main.py").is_file() or not (root.parent / ".venv/Scripts/python.exe").is_file():
        return None
    if not (root / "extra_model_paths.yaml").is_file():
        return None
    try:
        record = json.loads((root.parent / "setup.json").read_text(encoding="utf-8"))
        if record.get("schema_version") != 1 or record.get("fingerprint") != setup_fingerprint(repository_root):
            return None
        model_root(root)
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return root


def setup_fingerprint(repository_root: Path = REPOSITORY_ROOT) -> str:
    files = (
        "tools/minimax_h3_runtime_manifest.json",
        "tools/requirements-minimax-h3.lock",
        "patches/minimax-h3/comfyui-0.34.0-compiler.patch",
    )
    return hashlib.sha256(b"".join((Path(repository_root) / name).read_bytes() for name in files)).hexdigest()


def model_root(runtime_root: Path) -> Path:
    config = Path(runtime_root) / "extra_model_paths.yaml"
    if not config.exists():
        return Path(runtime_root) / "models"
    try:
        # JSONはYAMLのサブセット。導入側とComfyUIが同じ設定を読む。
        entry = json.loads(config.read_text(encoding="utf-8"))["aikimi_h3"]
        if any(entry.get(name) != name for name in MODEL_DIRECTORIES):
            raise ValueError("モデルの種類別フォルダーが一致しません。")
        return local_directory(entry["base_path"])
    except (OSError, UnicodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("H3のモデル保存先設定を読めません。H3のセットアップを確認してください。") from error


def configured_model_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    runtime = managed_runtime_root(repository_root)
    if (runtime / "extra_model_paths.yaml").exists():
        return model_root(runtime)
    return Path(repository_root).resolve() / "models/MiniMax-H3"


def model_config(models: Path) -> str:
    entry = {
        "base_path": str(local_directory(models)),
        "is_default": True,
        **{name: name for name in MODEL_DIRECTORIES},
    }
    return json.dumps({"aikimi_h3": entry}, ensure_ascii=False, indent=2) + "\n"


@contextmanager
def setup_lock(runtime_root: Path):
    """別ウィンドウやCLIからの同時導入・起動を防ぐ。"""
    base = Path(runtime_root).parent
    base.mkdir(parents=True, exist_ok=True)
    lock = base / "setup.lock"
    if lock.is_symlink() or lock.resolve().parent != base.resolve():
        raise ValueError("H3のセットアップロックにリンクは使用できません。")
    with lock.open("a+b") as stream:
        if stream.seek(0, os.SEEK_END) == 0:
            stream.write(b"\0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError("H3のセットアップまたは起動を別の処理で実行中です。") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
