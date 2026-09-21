"""Bounded Jev IPC, strict decisions and privacy-preserving experiment records."""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

MODEL = "jev-1.13.0"
SDK_VERSION = "0.7.0"
API_ORIGIN = "https://api.typesafe.ai"
MAX_MESSAGE_BYTES = 1024 * 1024


class JevError(RuntimeError):
    pass


class ExperimentCancelled(RuntimeError):
    pass


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def cloud_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """An explicit opt-in is required; unrelated credentials never enter the SDK."""
    source = os.environ if source is None else source
    if source.get("AIKIMI_JEV_ALLOW_CLOUD") != "1":
        raise JevError("Jevの外部送信が未許可です。aikimi-jev-launch.ps1から起動してください。")
    key = source.get("TYPESAFE_API_KEY", "").strip()
    if not key:
        raise JevError("TYPESAFE_API_KEYがありません。キーは生成履歴には保存しません。")
    allowed = {"PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "COMMONPROGRAMFILES", "PATHEXT", "COMSPEC", "LANG", "LC_ALL"}
    env = {k: v for k, v in source.items() if k.upper() in allowed}
    env.update(TYPESAFE_API_KEY=key, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    return env


def sdk_python(root: Path | None = None) -> Path:
    explicit = os.environ.get("AIKIMI_JEV_PYTHON", "").strip()
    root = root or Path(__file__).resolve().parents[2]
    path = Path(explicit) if explicit else root / "repositories" / "jev-sdk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not path.is_absolute() or not path.is_file():
        raise JevError("Jev専用Pythonがありません。tools/setup_jev_sparse.py --sdk を実行してください。")
    return path.resolve()


def parse_decisions(answer: Any, allowed: Mapping[str, tuple[float, ...]], fallback: float, threshold: float = 0.3) -> dict[str, float]:
    if not isinstance(answer, dict) or not isinstance(answer.get("decisions"), dict):
        raise JevError("Jev response schema mismatch")
    decisions = answer["decisions"]
    if set(decisions) != set(allowed):
        raise JevError("Missing, extra, or non-string layer IDs")
    result = {}
    for layer, choices in allowed.items():
        entry = decisions[layer]
        if not isinstance(entry, dict):
            raise JevError("Invalid decision")
        raw, conf = entry.get("choice"), entry.get("confidence")
        if isinstance(raw, bool) or isinstance(conf, bool):
            raise JevError("Boolean decision is not a number")
        try:
            keep, confidence = float(raw), float(conf)
        except (TypeError, ValueError, OverflowError):
            raise JevError("Non-numeric decision") from None
        if not math.isfinite(keep) or keep not in choices or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise JevError("Out-of-range decision")
        result[layer] = keep if confidence >= threshold else fallback
    return result


class JevClient:
    """One SDK subprocess per batched decision, no HTTP retry or shell invocation."""
    def __init__(self, python: Path, timeout: float = 5.0, cancelled: Callable[[], bool] | None = None):
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0.5 <= timeout <= 20:
            raise ValueError("Jev timeout must be 0.5..20 seconds")
        self.python = Path(python).resolve(strict=True)
        if not self.python.is_file():
            raise ValueError("SDK Python must be a file")
        self.env = cloud_environment()
        self.timeout = float(timeout)
        self.cancelled = cancelled or (lambda: False)
        self.calls = 0
        self.wait_seconds = 0.0

    def decide(self, state: dict, allowed: Mapping[str, tuple[float, ...]], fallback: float) -> dict[str, float]:
        payload = dumps({"state": state, "allowed": allowed, "timeout": self.timeout})
        if len(payload.encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise JevError("Decision payload is too large")
        if self.cancelled():
            raise ExperimentCancelled("Generation cancelled")
        self.calls += 1
        started = time.perf_counter()
        worker = Path(__file__).with_name("sdk_worker.py")
        proc = None
        try:
            proc = subprocess.Popen([str(self.python), "-I", "-B", str(worker)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", env=self.env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            first = True
            while True:
                if self.cancelled():
                    raise ExperimentCancelled("Generation cancelled")
                remaining = self.timeout + 2.0 - (time.perf_counter() - started)
                if remaining <= 0:
                    raise JevError("SDK timeout")
                try:
                    out, _ = proc.communicate(input=payload if first else None, timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    first = False
            if proc.returncode != 0 or len(out.encode("utf-8")) > MAX_MESSAGE_BYTES:
                raise JevError("SDK worker failed; provider error body withheld")
            try:
                answer = json.loads(out)
            except (ValueError, TypeError):
                raise JevError("Invalid SDK JSON") from None
            return parse_decisions(answer, allowed, fallback)
        except OSError:
            raise JevError("SDK process could not start") from None
        finally:
            if proc is not None:
                if proc.poll() is None:
                    proc.kill()
                proc.communicate()
            self.wait_seconds += time.perf_counter() - started


class RunLog:
    """JSONL with per-run identity. Never persist prompts, credentials or raw tensors."""
    def __init__(self, root: Path, target: str, settings: dict, prompt: str = ""):
        self.id = uuid.uuid4().hex
        self.target = target
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"{target}-{self.id}.jsonl"
        self.started = time.perf_counter()
        self.closed = False
        self.write("begin", settings=settings, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())

    def write(self, event: str, **data: Any) -> None:
        record = {"schema": 1, "run_id": self.id, "target": self.target, "event": event, "elapsed_seconds": time.perf_counter() - self.started, **data}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(dumps(record) + "\n")
            stream.flush()

    def finish(self, status: str, **data: Any) -> None:
        if not self.closed:
            self.write("end", status=status, **data)
            self.closed = True


@dataclass(frozen=True)
class AnimaOptions:
    mode: str = "off"
    keep_percent: float = 75.0
    min_tokens: int = 4096
    warmup_evaluations: int = 1
    update_interval: int = 4
    max_calls: int = 4
    timeout: float = 3.0

    def validate(self) -> None:
        if self.mode not in {"off", "dense", "fixed", "rules", "jev"}:
            raise ValueError("Unknown Anima sparse mode")
        if isinstance(self.keep_percent, bool) or not isinstance(self.keep_percent, (int, float)) or not math.isfinite(self.keep_percent) or not 1 <= self.keep_percent <= 100:
            raise ValueError("Keep percentage must be 1..100")
        for name, lo, hi in (("min_tokens", 64, 1048576), ("warmup_evaluations", 0, 100), ("update_interval", 1, 100), ("max_calls", 0, 8)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in {lo}..{hi}")
        if isinstance(self.timeout, bool) or not isinstance(self.timeout, (int, float)) or not math.isfinite(self.timeout) or not 0.5 <= self.timeout <= 20:
            raise ValueError("Timeout must be 0.5..20 seconds")
