"""H3の送信前記録。応答が失われてもIDと入力の所有情報を残す。"""

from __future__ import annotations

import json
import os
import threading
import uuid
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parents[1] / "cache" / "minimax-h3" / "pending"
_LOCK = threading.RLock()


def _path(prompt_id: str) -> Path:
    if str(uuid.UUID(prompt_id)) != prompt_id:
        raise ValueError("H3のジョブIDは正規形式のUUIDで指定してください。")
    return DIRECTORY / f"{prompt_id}.json"


def read_all() -> list[dict]:
    with _LOCK:
        records = []
        for path in DIRECTORY.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["version"] != 1 or _path(record["prompt_id"]) != path:
                raise ValueError("H3の未確定ジョブ記録を読めません。")
            records.append(record)
        return records


def write(record: dict) -> None:
    with _LOCK:
        path = _path(record["prompt_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)


def update(prompt_id: str, **changes) -> None:
    with _LOCK:
        path = _path(prompt_id)
        if path.exists():
            record = json.loads(path.read_text(encoding="utf-8"))
            write({**record, **changes})


def remove(prompt_id: str) -> None:
    with _LOCK:
        _path(prompt_id).unlink(missing_ok=True)


def server_stopped(record: dict) -> bool:
    """PID再利用も区別し、送信先processの終了を確認する。照会不能は終了ではない。"""
    import psutil

    identity = record["server_process"]
    try:
        process = psutil.Process(identity["pid"])
        return process.create_time() != identity["created"] or not process.is_running()
    except psutil.NoSuchProcess:
        return True
    except psutil.AccessDenied:
        return False
