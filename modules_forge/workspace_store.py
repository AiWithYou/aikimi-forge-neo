"""Bounded storage for gallery and prompt-library state in the data directory."""

import json
import os
import stat
import threading
from pathlib import Path, PureWindowsPath

from modules_forge.yue2_studio.core import atomic_json

_DEVICES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(f"{p}{i}" for p in ("LPT", "COM") for i in "123456789¹²³"),
}
LOCK = threading.RLock()
MAX_BYTES = 2 * 1024 * 1024


def managed_path(root, relative):
    if not isinstance(relative, str) or len(relative) > 2048:
        raise ValueError("無効な識別子です。")
    path = Path(root).resolve()
    for part in relative.split("/"):
        if (
            part in {"", ".", ".."}
            or part.endswith((" ", "."))
            or PureWindowsPath(part).drive
            or any(ord(c) < 32 or c in '<>:"\\|?*' for c in part)
            or part.partition(".")[0].upper() in _DEVICES
        ):
            raise ValueError("無効な識別子です。")
        path = path / part
        try:
            info = path.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("リンクは使用できません。")
    if path.resolve() != path or not path.is_relative_to(Path(root).resolve()):
        raise ValueError("管理対象外です。")
    return path


def open_regular(path):
    # Pin the inspected file before reading or streaming it. Re-check the
    # resolved parent and file identity after opening to catch replacements.
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.resolve(strict=True) != path:
        raise ValueError("通常ファイルを指定してください。")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    stream = os.fdopen(os.open(path, flags), "rb")
    try:
        if path.resolve(strict=True) != path or not os.path.samestat(info, os.fstat(stream.fileno())):
            raise ValueError("ファイルが変更されました。")
        return stream
    except BaseException:
        stream.close()
        raise


def read_json(path, limit=MAX_BYTES):
    with open_regular(path) as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError("ファイル容量の上限を超えています。")
    try:
        return json.loads(data)
    except (RecursionError, UnicodeError, ValueError):
        raise ValueError("JSON の形式が不正です。") from None


def write_json(path, value):
    data = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode()
    if len(data) > MAX_BYTES:
        raise ValueError("保存容量の上限を超えています。")
    atomic_json(path, value)
