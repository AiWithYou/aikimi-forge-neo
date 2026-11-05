"""Neoが管理するH3実行環境とモデル保存先。GPUや外部環境には接続しない。"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIRECTORY = Path("repositories/minimax-h3-4ac1c173/ComfyUI")
LEGACY_RUNTIME_DIRECTORY = Path("repositories/minimax-h3/ComfyUI")
LEGACY_FINGERPRINT = "22fc8c18d7bdf71018964277b246539b99f0861e0d115c7d6bcad9f75b420941"
MODEL_DIRECTORIES = ("diffusion_models", "text_encoders", "vae", "loras", "model_patches")
SERVER_URL = "http://127.0.0.1:8189"


def candidate_runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return Path(repository_root).resolve() / RUNTIME_DIRECTORY


def managed_runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    """Keep using the old root until the new isolated environment is complete."""
    root = Path(repository_root).resolve()
    candidate = candidate_runtime_root(root)
    if _runtime_complete(candidate, root):
        return candidate
    legacy = root / LEGACY_RUNTIME_DIRECTORY
    return legacy if (legacy / "main.py").is_file() else candidate


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


def _runtime_complete(root: Path, repository_root: Path, *, legacy: bool = False) -> bool:
    pending = root.parent / "runtime-update-in-progress.json"
    if pending.exists() or pending.is_symlink():
        return False
    if not (root.parent / "setup.json").is_file():
        return False
    if not (root / "main.py").is_file() or not (root.parent / ".venv/Scripts/python.exe").is_file():
        return False
    if not (root / "extra_model_paths.yaml").is_file():
        return False
    try:
        record = json.loads((root.parent / "setup.json").read_text(encoding="utf-8"))
        expected = LEGACY_FINGERPRINT if legacy else setup_fingerprint(repository_root)
        if record.get("schema_version") != 1 or record.get("fingerprint") != expected:
            return False
        model_root(root)
    except (OSError, ValueError, TypeError, AttributeError):
        return False
    return True


def installed_runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path | None:
    repository = Path(repository_root).resolve()
    root = managed_runtime_root(repository)
    legacy = root == repository / LEGACY_RUNTIME_DIRECTORY
    return root if _runtime_complete(root, repository, legacy=legacy) else None


def setup_fingerprint(repository_root: Path = REPOSITORY_ROOT) -> str:
    files = (
        "tools/minimax_h3_runtime_manifest.json",
        "tools/requirements-minimax-h3.lock",
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
    if Path(runtime_root).name == "ComfyUI" and base.name in {
        LEGACY_RUNTIME_DIRECTORY.parent.name,
        RUNTIME_DIRECTORY.parent.name,
    }:
        lock = base.parent / ".minimax-h3-setup.lock"
    else:
        lock = base / "setup.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    if lock.is_symlink() or lock.resolve().parent != lock.parent.resolve():
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
