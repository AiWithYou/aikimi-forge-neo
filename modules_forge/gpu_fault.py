"""CPU-only admission stop after an observed GPU execution fault."""

import logging
import threading
from datetime import UTC, datetime
from pathlib import Path

from modules_forge.yue2_studio.core import atomic_json

MESSAGE = "GPU異常後の停止中です。状態を確認して停止を解除してください。"
_log = logging.getLogger(__name__)


class GPUFaultError(RuntimeError):
    def __init__(self):
        super().__init__(MESSAGE)


def classify(error):
    """Classify execution failures, without importing Torch or probing a device."""
    messages, visited = [], set()
    while error is not None and id(error) not in visited:
        visited.add(id(error))
        messages.append(str(error).lower())
        error = error.__cause__ or error.__context__
    text = "\n".join(messages)
    fatal = (
        "gpu is lost",
        "device is lost",
        "device lost",
        "device-side assert",
        "illegal memory access",
        "unspecified launch failure",
        "launch timed out",
        "launch timeout",
        "context is destroyed",
        "cudnn_status_execution_failed",
        "mha_graph.execute",
    )
    if any(marker in text for marker in fatal):
        return "gpu_execution"
    if any("cuda error:" in message and "out of memory" not in message for message in messages):
        return "gpu_execution"
    return None


class FaultState:
    def __init__(self, path):
        self.path = Path(path)
        self.pending_path = self.path.with_suffix(".pending.json")
        self._latched = False
        self._pending = None
        self._guard = threading.RLock()

    def blocked(self):
        with self._guard:
            if not self._latched:
                try:
                    self.path.stat()
                except FileNotFoundError:
                    pass
                except OSError:
                    self._latched = True
                else:
                    self._latched = True
                if self._pending is None:
                    try:
                        self.pending_path.stat()
                    except FileNotFoundError:
                        pass
                    except OSError:
                        self._latched = True
                    else:
                        self._latched = True
            return self._latched

    def ensure_available(self):
        if self.blocked():
            raise GPUFaultError()

    def trip(self, reason, engine):
        with self._guard:
            self._latched = True
            try:
                atomic_json(self.path, {"reason": reason, "engine": engine, "time": datetime.now(UTC).isoformat()})
            except OSError:
                _log.exception("GPU停止状態を保存できませんでした。現在のプロセスでは停止を維持します。")

    def begin(self, identifier):
        with self._guard:
            self.ensure_available()
            if self._pending is not None:
                raise RuntimeError("GPU operation already pending")
            try:
                atomic_json(self.pending_path, {"job": identifier})
            except OSError:
                self._latched = True
                raise GPUFaultError() from None
            self._pending = identifier

    def complete(self, identifier):
        with self._guard:
            if self._pending == identifier and not self.blocked():
                try:
                    self.pending_path.unlink()
                except OSError:
                    self._latched = True
                    raise GPUFaultError() from None
                self._pending = None

    def clear(self, *, acknowledge_recovery=False):
        if acknowledge_recovery is not True:
            raise ValueError("状態を確認してから --acknowledge-recovery を指定してください。")
        with self._guard:
            for path in (self.path, self.pending_path):
                path.unlink(missing_ok=True)
            # Existing processes remain stopped; only a newly started app may run.


STATE = FaultState(Path(__file__).resolve().parents[1] / "cache" / "gpu-fault.json")
