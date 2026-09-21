"""Read a user-owned Windows DPAPI credential, outside the Git repository."""

from __future__ import annotations

import ctypes
import os
import re
import tempfile
from ctypes import wintypes
from pathlib import Path


def credential_path() -> Path:
    if os.name == "nt":
        if not os.environ.get("LOCALAPPDATA"):
            raise RuntimeError("Windows LOCALAPPDATA is unavailable.")
        return Path(os.environ["LOCALAPPDATA"]) / "Aikimi" / "secrets" / "jev-api-key.dpapi"
    return Path(os.environ.get("XDG_CONFIG_HOME", str(Path.home() / ".config"))) / "aikimi/secrets/jev-api-key"


def _dpapi(data: bytes, *, protect: bool) -> bytes:
    class Blob(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("data", ctypes.POINTER(ctypes.c_byte))]

    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    operation = crypt.CryptProtectData if protect else crypt.CryptUnprotectData
    operation.argtypes = [
        ctypes.POINTER(Blob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(Blob),
    ]
    operation.restype = wintypes.BOOL
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_byte)))
    output = Blob()
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(output)):
        raise RuntimeError("Cannot access the saved Jev key for this Windows user.")
    try:
        return ctypes.string_at(output.data, output.size)
    finally:
        ctypes.memset(output.data, 0, output.size)
        kernel.LocalFree(output.data)


def read_saved_key(path: Path | None = None) -> str:
    """No key is exposed through status messages, generated metadata or CLI args."""
    try:
        data = (path or credential_path()).read_text(encoding="utf-8-sig").strip()
        key = _dpapi(bytes.fromhex(data), protect=False).decode("utf-16-le") if os.name == "nt" else data
        if not key.strip():
            raise ValueError
        return key.strip()
    except (OSError, ValueError, RuntimeError):
        raise RuntimeError("Jev APIキーを設定欄で保存してください。") from None


def save_key(key: str, path: Path | None = None) -> None:
    if not isinstance(key, str) or not re.fullmatch(r"apikey_[A-Za-z0-9_-]{16,512}", key.strip()):
        raise ValueError("TypeSafeのAPIキー（apikey_で始まる値）を入力してください。")
    target = (path or credential_path()).resolve()
    if target.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("APIキーの保存先はリポジトリの外側に指定してください。")
    target.parent.mkdir(parents=True, exist_ok=True)
    if os.name != "nt":
        target.parent.chmod(0o700)
    content = _dpapi(key.strip().encode("utf-16-le"), protect=True).hex() if os.name == "nt" else key.strip()
    descriptor, temporary = tempfile.mkstemp(prefix=".jev-", dir=target.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        if os.name != "nt":
            Path(temporary).chmod(0o600)
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def cloud_source() -> dict[str, str]:
    """Only callers handling an explicitly selected Jev mode use this mapping."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip() or read_saved_key()
    return {**os.environ, "TYPESAFE_API_KEY": key, "AIKIMI_JEV_ALLOW_CLOUD": "1"}


def enable_saved_key() -> None:
    """Call only for an explicitly selected Jev launch or benchmark."""
    key = os.environ.get("TYPESAFE_API_KEY", "").strip() or read_saved_key()
    os.environ["TYPESAFE_API_KEY"] = key
    os.environ["AIKIMI_JEV_ALLOW_CLOUD"] = "1"
