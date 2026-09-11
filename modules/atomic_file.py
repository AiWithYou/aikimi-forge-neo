"""Failure-safe writes for small user-owned settings and metadata files."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from threading import RLock
from typing import Any

# Windows path resolution/stat calls can briefly deny a concurrent replacement.
# Serialize this small settings-only surface, including its metadata lookups.
_write_lock = RLock()


def atomic_write_text(filename: str | os.PathLike[str], text: str) -> None:
    """Replace a file only after its complete UTF-8 contents have been flushed.

    Resolve symlinks to preserve the behavior of open(filename, "w"). Unique
    sibling temporary files prevent simultaneous saves from mixing their bytes.
    Existing permissions are retained; new files use tempfile's private defaults.
    """
    payload = text.encode("utf-8")
    with _write_lock:
        target = Path(filename).resolve()
        try:
            mode = stat.S_IMODE(target.stat().st_mode)
        except FileNotFoundError:
            mode = None

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=".aikimi-", suffix=".tmp", dir=target.parent, delete=False
            ) as file:
                temporary = Path(file.name)
                file.write(payload)
                file.flush()
                os.fsync(file.fileno())
            if mode is not None:
                os.chmod(temporary, mode)
            os.replace(temporary, target)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def atomic_write_json(filename: str | os.PathLike[str], value: Any) -> None:
    """Serialize before touching the destination, preserving the existing format."""
    atomic_write_text(filename, json.dumps(value, indent=4, ensure_ascii=False))
