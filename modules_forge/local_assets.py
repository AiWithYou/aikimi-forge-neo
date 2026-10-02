"""Local model references, content identities and LoRA rows; no model imports or downloads."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
import threading
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIBRARY = ROOT / "models" / ".local-assets" / "library.json"
HASH_CACHE = LIBRARY.with_name("hashes.json")
_LOCK = threading.RLock()


def local_path(value: str, *, directory: bool = False, suffixes=(".safetensors", ".gguf")) -> Path:
    if not isinstance(value, str) or not value.strip() or "://" in value:
        raise ValueError("このPCにあるファイル／フォルダーのパスを指定してください。")
    path = Path(value.strip().strip('"')).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if directory:
        if not path.is_dir():
            raise ValueError(f"フォルダーが見つかりません: {path}")
    elif not path.is_file() or path.suffix.lower() not in suffixes:
        raise ValueError(f"対応するローカルファイルが見つかりません: {path}")
    return path


def read_header(path: Path) -> dict:
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("safetensorsファイルが不完全です。")
        size = struct.unpack("<Q", prefix)[0]
        file_size = os.fstat(stream.fileno()).st_size
        if not 2 <= size <= 32 * 1024 * 1024 or size + 8 > file_size:
            raise ValueError("safetensorsのヘッダーが不正です。")
        header = json.loads(stream.read(size))
    if not isinstance(header, dict) or not any(key != "__metadata__" for key in header):
        raise ValueError("モデルのテンソルがありません。")
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
            or not 0 <= offsets[0] <= offsets[1] <= file_size - size - 8
        ):
            raise ValueError("safetensorsのテンソル範囲が不正、またはファイルが不完全です。")
    return header


def file_version(path: Path) -> tuple:
    """Include NTFS ChangeTime: Python's Windows ctime is creation time."""
    with Path(path).open("rb") as stream:
        stat = os.fstat(stream.fileno())
        changed = stat.st_ctime_ns
        if os.name == "nt":
            import ctypes
            import msvcrt

            class FileBasicInfo(ctypes.Structure):
                _fields_ = [(name, ctypes.c_longlong) for name in ("created", "accessed", "written", "changed")] + [
                    ("attributes", ctypes.c_ulong)
                ]

            info = FileBasicInfo()
            success = ctypes.windll.kernel32.GetFileInformationByHandleEx(
                ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno())),
                0,
                ctypes.byref(info),
                ctypes.sizeof(info),
            )
            changed = info.changed if success else None
    return stat.st_size, stat.st_mtime_ns, changed, stat.st_ino


@lru_cache(maxsize=256)
def _digest(path: str, *version) -> str:
    records = {}
    if version[2] is not None:
        try:
            records = json.loads(HASH_CACHE.read_text(encoding="utf-8"))
            if not isinstance(records, dict):
                records = {}
            entry = records.get(path, {})
            if (
                entry.get("version") == list(version)
                and isinstance(entry.get("sha256"), str)
                and len(entry["sha256"]) == 64
            ):
                return entry["sha256"]
        except (OSError, ValueError, AttributeError):
            records = {}
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if version[2] is not None and tuple(version) == file_version(Path(path)):
        records[path] = {"version": list(version), "sha256": digest}
        # Losing a concurrent cache entry costs a rehash; never reuse it under a different stamp.
        temporary = None
        try:
            HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=HASH_CACHE.parent, delete=False) as out:
                temporary = Path(out.name)
                json.dump(dict(list(records.items())[-1024:]), out)
            os.replace(temporary, HASH_CACHE)
        except OSError:
            pass  # Read-only workspaces still support local models.
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return digest


def file_identity(path: Path) -> dict:
    path = Path(path).resolve()
    before = file_version(path)
    # Filesystems without a reliable change counter are hashed on every check.
    digest_fn = _digest if before[2] is not None else _digest.__wrapped__
    digest = digest_fn(str(path), *before)
    if before != file_version(path):
        raise ValueError(f"確認中にファイルが変更されました。再実行してください: {path.name}")
    return {"path": str(path), "size": before[0], "sha256": digest}


def identity(path: Path) -> dict:
    path = Path(path).resolve()
    if path.is_file():
        return file_identity(path)
    files = [p for p in sorted(path.rglob("*")) if p.is_file() and ".cache" not in p.relative_to(path).parts]
    if not files:
        raise ValueError(f"モデルファイルがありません: {path}")
    records = [{**file_identity(p), "path": p.relative_to(path).as_posix()} for p in files]
    digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    return {"path": str(path), "sha256": digest, "files": records}


def library() -> dict:
    try:
        data = json.loads(LIBRARY.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def remember(category: str, paths: list[str]) -> None:
    """Remember only explicitly selected references, never copy or modify weights."""
    with _LOCK:
        data = library()
        data[category] = list(dict.fromkeys([*data.get(category, []), *paths]))
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        temporary = LIBRARY.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, LIBRARY)


def selection(engine: str) -> dict:
    value = library().get("selected_" + engine, {})
    return value if isinstance(value, dict) else {}


def save_selection(engine: str, value: dict) -> None:
    with _LOCK:
        data = library()
        data["selected_" + engine] = value
        LIBRARY.parent.mkdir(parents=True, exist_ok=True)
        temporary = LIBRARY.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, LIBRARY)


def choices(category: str, roots=(), *, folders: bool = False, suffixes=(".safetensors", ".gguf")) -> list:
    paths = list(library().get(category, []))
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        if folders:
            paths.extend(str(p.parent) for p in root.rglob("model_index.json"))
            paths.extend(str(p.parent) for p in root.rglob("config.json") if p.parent.name == "transformer")
        paths.extend(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in suffixes)
    return [(str(Path(p).name) + " · " + str(Path(p).parent), p) for p in sorted(set(paths))]


def lora_label(name: str) -> str:
    path = Path(name.strip().strip('"'))
    return f"{path.name} · {path.parent.name} [{path.parent}]" if path.is_absolute() else name


def lora_rows(names, rows) -> list:
    previous = {row[0]: row[1] for row in (rows or []) if len(row) == 2}
    return [[lora_label(name), previous.get(name, previous.get(lora_label(name), 1.0))] for name in (names or [])]


def lora_settings(names, rows) -> tuple[dict, ...]:
    previous = {row[0]: row[1] for row in (rows or []) if len(row) == 2}
    result = []
    seen = set()
    for name in names or []:
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("LoRAの名前が空欄、または重複しています。")
        seen.add(name)
        row_key = name if name in previous else lora_label(name)
        if row_key not in previous:
            raise ValueError("LoRAの強度欄が更新されるのを待ってください。")
        strength = previous[row_key]
        error = f"{lora_label(name)}: LoRAの強度は−2〜2の数値で指定してください。"
        if isinstance(strength, bool):
            raise ValueError(error)
        try:
            strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise ValueError(error) from exc
        if not math.isfinite(strength) or not -2 <= strength <= 2:
            raise ValueError(error)
        result.append({"name": name, "strength": strength})
    return tuple(result)
