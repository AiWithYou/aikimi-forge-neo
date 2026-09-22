"""Bounded Jev IPC, strict decisions and privacy-preserving experiment records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL = "jev-1.13.0"
SDK_VERSION = "0.7.0"
API_ORIGIN = "https://api.typesafe.ai"
MAX_MESSAGE_BYTES = 1024 * 1024


class JevError(RuntimeError):
    pass


class ExperimentCancelled(RuntimeError):
    pass


class BudgetExhausted(JevError):
    pass


class ReplayError(JevError):
    """A replay is invalid, incomplete or belongs to a different request."""


@dataclass
class JevBudget:
    """Hard limits shared by every decision in one outer generation job."""

    max_calls: int = 0
    max_wait_seconds: float = 0.0
    calls: int = 0
    wait_seconds: float = 0.0

    def __post_init__(self):
        if type(self.max_calls) is not int or not 0 <= self.max_calls <= 1000:
            raise ValueError("Jev job max calls must be an integer in 0..1000")
        value = self.max_wait_seconds
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 3600
        ):
            raise ValueError("Jev job wait budget must be a finite number in 0..3600")

    @property
    def remaining_seconds(self):
        return max(0.0, self.max_wait_seconds - self.wait_seconds) if self.max_wait_seconds else math.inf

    @property
    def available(self):
        return (not self.max_calls or self.calls < self.max_calls) and self.remaining_seconds > 0

    def begin(self):
        if not self.available:
            raise BudgetExhausted("Jev generation budget exhausted")
        self.calls += 1


REPLAY_ENV = "AIKIMI_JEV_REPLAY_LOGS"


def replay_paths_from_environment(source=None):
    value = (os.environ if source is None else source).get(REPLAY_ENV, "")
    if not value:
        return []
    try:
        paths = json.loads(value)
    except (TypeError, ValueError):
        raise ReplayError("Replay logs must be a JSON array of paths") from None
    if not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths):
        raise ReplayError("Replay logs must be a nonempty JSON array of paths")
    return paths


def replay_requested():
    return bool(replay_paths_from_environment())


def decision_count(client):
    return getattr(client, "calls", 0) + getattr(client, "replay_calls", 0)


def client_available(client):
    return client is not None and getattr(client, "available", True)


def _shape(value):
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise JevError("Decision state keys must be strings")
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)) and math.isfinite(value):
        return "number"
    if isinstance(value, str):
        return "string"
    raise JevError("Invalid decision state type")


def request_contract(state, allowed):
    if not isinstance(state, dict) or not isinstance(allowed, Mapping) or not 1 <= len(allowed) <= 256:
        raise JevError("Invalid decision request")
    normalized = {}
    for key, choices in allowed.items():
        if not isinstance(key, str) or not isinstance(choices, (list, tuple)) or not choices:
            raise JevError("Invalid decision choices")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in choices):
            raise JevError("Invalid decision choices")
        normalized[key] = [float(v) for v in choices]
    # Retain request identity and tensor geometry, never prompts or tensor values.
    context_keys = {
        "target",
        "decision_kind",
        "evaluation",
        "next_model_evaluation",
        "sampling_step",
        "sampling_pass",
        "step_measured",
        "next_step",
        "initialization",
        "stage",
        "decision_cadence",
        "positive_step_bounds",
        "tensor_shapes",
        "tile_layout",
        "shape",
        "prefix_tokens",
        "target_tokens",
    }
    context = {key: state[key] for key in context_keys if key in state}
    if "constraints" in state:
        context["constraints_sha256"] = hashlib.sha256(str(state["constraints"]).encode()).hexdigest()
    return {"allowed": normalized, "state_shape": _shape(state), "context": context}


def _diagnostics(answer, result):
    return {
        layer: {
            "proposed_keep_percent": float(answer["decisions"][layer]["choice"]),
            "confidence": float(answer["decisions"][layer]["confidence"]),
            "applied_keep_percent": keep,
            "confidence_policy": "report_only",
        }
        for layer, keep in result.items()
    }


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
    allowed = {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "LOCALAPPDATA",
        "APPDATA",
        "PROGRAMFILES",
        "PROGRAMFILES(X86)",
        "COMMONPROGRAMFILES",
        "PATHEXT",
        "COMSPEC",
        "LANG",
        "LC_ALL",
    }
    env = {k: v for k, v in source.items() if k.upper() in allowed}
    env.update(TYPESAFE_API_KEY=key, PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONNOUSERSITE="1")
    return env


def sdk_python(root: Path | None = None) -> Path:
    explicit = os.environ.get("AIKIMI_JEV_PYTHON", "").strip()
    root = root or Path(__file__).resolve().parents[2]
    path = (
        Path(explicit)
        if explicit
        else root / "repositories" / "jev-sdk" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not path.is_absolute() or not path.is_file():
        raise JevError("Jev専用Pythonがありません。tools/setup_jev_sparse.py --sdk を実行してください。")
    return path.resolve()


def parse_decisions(
    answer: Any, allowed: Mapping[str, tuple[float, ...]], fallback: float, threshold: float = 0.3
) -> dict[str, float]:
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
    """One isolated SDK process and HTTP client per job, with bounded framed IPC."""

    def __init__(
        self,
        python: Path,
        timeout: float = 5.0,
        cancelled: Callable[[], bool] | None = None,
        *,
        environment=None,
        budget=None,
    ):
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or not 0.5 <= timeout <= 20
        ):
            raise ValueError("Jev timeout must be 0.5..20 seconds")
        self.python = Path(python).resolve(strict=True)
        if not self.python.is_file():
            raise ValueError("SDK Python must be a file")
        self.env = cloud_environment(environment)
        self.timeout = float(timeout)
        self.cancelled = cancelled or (lambda: False)
        self.calls = 0
        self.wait_seconds = 0.0
        self.last_diagnostics = {}
        self.last_replay = None
        self.budget = budget if budget is not None else JevBudget()
        self.job_id = uuid.uuid4().hex
        self.closed = False
        self._proc = None
        self._thread = None
        self._inflight = False
        self._requests = queue.Queue(maxsize=1)
        self._responses = queue.Queue(maxsize=1)

    @property
    def available(self):
        return not self.closed and self.budget.available

    def _start(self):
        if self._proc is not None:
            return
        worker = Path(__file__).with_name("sdk_worker.py")
        self._proc = subprocess.Popen(  # noqa: S603 -- isolated interpreter, fixed local worker, no shell
            [str(self.python), "-I", "-B", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            env=self.env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

        def exchange():
            while True:
                payload = self._requests.get()
                if payload is None:
                    return
                try:
                    self._proc.stdin.write(payload + "\n")
                    self._proc.stdin.flush()
                    out = self._proc.stdout.readline(MAX_MESSAGE_BYTES + 2)
                    if not out.endswith("\n") or len(out.encode("utf-8")) > MAX_MESSAGE_BYTES:
                        raise JevError("SDK worker failed; provider error body withheld")
                    self._responses.put((True, out))
                except Exception:
                    self._responses.put((False, None))
                    return

        self._thread = threading.Thread(target=exchange, name="jev-sdk-ipc", daemon=True)
        self._thread.start()

    def decide(self, state: dict, allowed: Mapping[str, tuple[float, ...]], fallback: float) -> dict[str, float]:
        self.last_diagnostics = {}
        self.last_replay = None
        if self.closed:
            raise JevError("Jev client is closed")
        contract = request_contract(state, allowed)
        payload = dumps({"state": state, "allowed": allowed, "timeout": self.timeout})
        if len(payload.encode("utf-8")) + 1 > MAX_MESSAGE_BYTES:
            raise JevError("Decision payload is too large")
        if self.cancelled():
            self.close("cancelled")
            raise ExperimentCancelled("Generation cancelled")
        self.budget.begin()
        self.calls += 1
        started = time.perf_counter()
        wait_limit = min(self.timeout + 2.0, self.budget.remaining_seconds)
        try:
            self._start()
            self._inflight = True
            self._requests.put_nowait(payload)
            while True:
                if self.cancelled():
                    raise ExperimentCancelled("Generation cancelled")
                remaining = wait_limit - (time.perf_counter() - started)
                if remaining <= 0:
                    raise JevError("SDK timeout or generation wait budget exhausted")
                try:
                    okay, out = self._responses.get(timeout=min(0.1, remaining))
                    break
                except queue.Empty:
                    continue
            if not okay:
                raise JevError("SDK worker failed; provider error body withheld")
            try:
                answer = json.loads(out)
            except (ValueError, TypeError):
                raise JevError("Invalid SDK JSON") from None
            # This is a reversible visual-quality experiment. Confidence measures
            # ambiguity between choices, not pixel error. Do not silently replace
            # a valid sparse choice with 100% using an uncalibrated confidence gate.
            result = parse_decisions(answer, allowed, fallback, threshold=0.0)
            self.last_diagnostics = _diagnostics(answer, result)
            self.last_replay = {
                "schema": 1,
                "job_id": self.job_id,
                "sequence": self.calls,
                "budget": {"max_calls": self.budget.max_calls, "max_wait_seconds": self.budget.max_wait_seconds},
                "request": contract,
                "answer": {
                    "decisions": {
                        layer: {
                            "choice": float(answer["decisions"][layer]["choice"]),
                            "confidence": float(answer["decisions"][layer]["confidence"]),
                        }
                        for layer in result
                    }
                },
            }
            self._inflight = False
            return result
        except OSError:
            self.close()
            raise JevError("SDK process could not start") from None
        except BaseException:
            self.close()
            raise
        finally:
            elapsed = time.perf_counter() - started
            self.wait_seconds += elapsed
            self.budget.wait_seconds += elapsed
            if self.last_replay is not None:
                self.last_replay["api_wait_seconds"] = self.wait_seconds

    def close(self, status="completed"):
        if self.closed:
            return
        self.closed = True
        proc = self._proc
        try:
            if proc is not None:
                if proc.poll() is None:
                    # EOF allows TypeSafeClient.__exit__ to close its pooled HTTP
                    # transport. An in-flight cancellation instead kills below.
                    if self._inflight:
                        proc.kill()
                    else:
                        proc.stdin.close()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
        finally:
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
            if self._thread is not None:
                self._thread.join(timeout=1.0)
            if proc is not None:
                for stream in (proc.stdin, proc.stdout):
                    if stream is not None:
                        stream.close()
            self.env.clear()

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        self.close("completed" if kind is None else "failed")


class ReplayClient:
    """Strict offline decision replay. This class never constructs a cloud client."""

    def __init__(self, log_paths, *, cancelled=None, prompt_sha256=None, replay_identity=None, budget=None):
        self.cancelled = cancelled or (lambda: False)
        self.calls, self.wait_seconds, self.replay_calls = 0, 0.0, 0
        self.last_diagnostics, self.last_replay = {}, None
        self.closed, self.failed = False, False
        self.recorded_wait_seconds = 0.0
        self.records = []
        if not isinstance(log_paths, (tuple, list)) or not log_paths:
            raise ReplayError("Replay requires recorded decision logs")
        try:
            total_bytes = 0
            for path in log_paths:
                with Path(path).open(encoding="utf-8") as stream:
                    rows = []
                    for line in stream:
                        size = len(line.encode("utf-8"))
                        total_bytes += size
                        if size > MAX_MESSAGE_BYTES or total_bytes > 64 * MAX_MESSAGE_BYTES:
                            raise ReplayError("Replay record exceeds size limit")
                        row = json.loads(line)
                        if not isinstance(row, dict) or row.get("schema") != 1:
                            raise ReplayError("Replay log schema mismatch")
                        rows.append(row)
                        if len(rows) > 100000:
                            raise ReplayError("Replay log exceeds record limit")
                if not rows or rows[0].get("event") != "begin" or rows[-1].get("event") != "end":
                    raise ReplayError("Replay log is incomplete")
                if rows[-1].get("status") not in {"completed", "sampling_pass_completed"}:
                    raise ReplayError("Replay source generation did not complete")
                if any(
                    row.get("run_id") != rows[0].get("run_id") or row.get("target") != rows[0].get("target")
                    for row in rows
                ):
                    raise ReplayError("Replay log mixes run identities")
                if prompt_sha256 is not None and rows[0].get("prompt_sha256") != prompt_sha256:
                    raise ReplayError("Replay prompt identity mismatch")
                if (
                    replay_identity is not None
                    and rows[0].get("settings", {}).get("replay_identity") != replay_identity
                ):
                    raise ReplayError("Replay generation identity mismatch")
                for row in rows:
                    if row.get("event") != "decision":
                        continue
                    if row.get("source") != "jev":
                        raise ReplayError("Replay source includes an unsuccessful or non-Jev decision")
                    item = row.get("replay")
                    if (
                        not isinstance(item, dict)
                        or item.get("schema") != 1
                        or not isinstance(item.get("request"), dict)
                    ):
                        raise ReplayError("Replay source lacks a validated request contract")
                    if not isinstance(item.get("job_id"), str) or type(item.get("sequence")) is not int:
                        raise ReplayError("Replay decision sequence is invalid")
                    self.records.append(item)
        except (OSError, ValueError, TypeError, KeyError):
            raise ReplayError("Replay log could not be read or validated") from None
        if not self.records or len({r["job_id"] for r in self.records}) != 1:
            raise ReplayError("Replay requires decisions from exactly one generation job")
        self.records.sort(key=lambda r: r["sequence"])
        if [r["sequence"] for r in self.records] != list(range(1, len(self.records) + 1)):
            raise ReplayError("Replay decisions contain missing or duplicate sequence numbers")
        limits = self.records[0].get("budget")
        if not isinstance(limits, dict) or set(limits) != {"max_calls", "max_wait_seconds"}:
            raise ReplayError("Replay source lacks the job budget contract")
        try:
            recorded_budget = JevBudget(**limits)
        except (TypeError, ValueError):
            raise ReplayError("Replay job budget is invalid") from None
        self.budget = recorded_budget if budget is None else budget
        if limits != {"max_calls": self.budget.max_calls, "max_wait_seconds": self.budget.max_wait_seconds}:
            raise ReplayError("Replay job budget differs from the source")
        previous_wait = 0.0
        for item in self.records:
            wait = item.get("api_wait_seconds")
            if (
                item.get("budget") != limits
                or isinstance(wait, bool)
                or not isinstance(wait, (int, float))
                or not math.isfinite(wait)
                or wait < previous_wait
            ):
                raise ReplayError("Replay timing or job budget contract is invalid")
            previous_wait = wait

    @property
    def available(self):
        # Do not hide exhaustion: the next requested decision must fail loudly.
        return not self.closed and self.budget.available

    def decide(self, state, allowed, fallback):
        if self.cancelled():
            raise ExperimentCancelled("Generation cancelled")
        self.last_diagnostics, self.last_replay = {}, None
        try:
            if self.closed or self.replay_calls >= len(self.records):
                raise ReplayError("Replay has no decision for this request")
            item = self.records[self.replay_calls]
            if item["request"] != request_contract(state, allowed):
                raise ReplayError("Replay request shape, schedule or allowed choices mismatch")
            result = parse_decisions(item.get("answer"), allowed, fallback, threshold=0.0)
            self.budget.begin()
            # Replay spends no API time. Its virtual budget follows the recorded
            # wait so finite live budgets stop at exactly the same decision.
            self.budget.wait_seconds += item["api_wait_seconds"] - self.recorded_wait_seconds
            self.recorded_wait_seconds = item["api_wait_seconds"]
            self.last_diagnostics = _diagnostics(item["answer"], result)
            self.last_replay = item
            self.replay_calls += 1
            return result
        except JevError as exc:
            self.failed = True
            if isinstance(exc, ReplayError):
                raise
            raise ReplayError("Replay decision schema mismatch") from None

    def close(self, status="completed"):
        if self.closed:
            return
        self.closed = True
        if status == "completed" and not self.failed and self.replay_calls != len(self.records):
            self.failed = True
            raise ReplayError("Replay finished with unused decisions")


def create_client(
    python=None,
    timeout=5.0,
    cancelled=None,
    *,
    environment=None,
    budget=None,
    replay_paths=None,
    prompt_sha256=None,
    replay_identity=None,
):
    paths = replay_paths if replay_paths is not None else replay_paths_from_environment()
    if paths:
        return ReplayClient(
            paths, cancelled=cancelled, prompt_sha256=prompt_sha256, replay_identity=replay_identity, budget=budget
        )
    return JevClient(python or sdk_python(), timeout, cancelled, environment=environment, budget=budget)


def close_client(client, status="completed"):
    close = getattr(client, "close", None)
    if close is not None:
        close(status)


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
        record = {
            "schema": 1,
            "run_id": self.id,
            "target": self.target,
            "event": event,
            "elapsed_seconds": time.perf_counter() - self.started,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(dumps(record) + "\n")
            stream.flush()

    def finish(self, status: str, **data: Any) -> None:
        if not self.closed:
            self.write("end", status=status, **data)
            self.closed = True


def cadence_interval(cadence, interval):
    return 1 if cadence == "step" else interval


def cadence_has_budget(cadence, calls, maximum):
    # Keep zero as an explicit cloud opt-out for existing API clients.
    return maximum > 0 and (cadence in {"interval", "step"} or calls < (maximum if cadence == "legacy" else 1))


@dataclass(frozen=True)
class AnimaOptions:
    mode: str = "off"
    keep_percent: float = 75.0
    min_tokens: int = 4096
    warmup_evaluations: int = 1
    update_interval: int = 4
    max_calls: int = 1
    timeout: float = 3.0
    decision_cadence: str = "legacy"
    job_max_calls: int = 0
    job_max_wait_seconds: float = 0.0

    def validate(self) -> None:
        JevBudget(self.job_max_calls, self.job_max_wait_seconds)
        if self.mode not in {"off", "dense", "fixed", "rules", "jev"}:
            raise ValueError("Unknown Anima sparse mode")
        if not isinstance(self.decision_cadence, str) or self.decision_cadence not in {
            "legacy",
            "once",
            "interval",
            "step",
        }:
            raise ValueError("Unknown Jev decision cadence")
        if (
            isinstance(self.keep_percent, bool)
            or not isinstance(self.keep_percent, (int, float))
            or not math.isfinite(self.keep_percent)
            or not 1 <= self.keep_percent <= 100
        ):
            raise ValueError("Keep percentage must be 1..100")
        for name, lo, hi in (
            ("min_tokens", 64, 1048576),
            ("warmup_evaluations", 0, 100),
            ("update_interval", 1, 100),
            ("max_calls", 0, 8),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in {lo}..{hi}")
        if (
            isinstance(self.timeout, bool)
            or not isinstance(self.timeout, (int, float))
            or not math.isfinite(self.timeout)
            or not 0.5 <= self.timeout <= 20
        ):
            raise ValueError("Timeout must be 0.5..20 seconds")
