"""専用Pythonをジョブ間で再利用する。ファイル通信で標準入力の競合を避ける。"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import errno
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import UTC, datetime
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.yue2_studio.core import atomic_json, read_json  # noqa: E402

_event_guard = threading.Lock()
_log = logging.getLogger(__name__)


def record_event(log_path, event, **fields):
    """Retain CPU lifecycle evidence without preventing a required OS stop."""
    try:
        record = {
            "event": event,
            "time_utc": datetime.now(UTC).isoformat(),
            "monotonic_ns": time.monotonic_ns(),
            **fields,
        }
        with _event_guard, Path(log_path).with_suffix(".lifecycle.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, ValueError, TypeError):
        # A failing diagnostic handler must not prevent OS termination either.
        with contextlib.suppress(Exception):
            _log.warning("ワーカーの終了・中断記録を保存できませんでした。", exc_info=True)


class WorkerExited(RuntimeError):
    def __init__(self, message, exit_code):
        super().__init__(message)
        self.exit_code = exit_code


def _synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._guard:
            return method(self, *args, **kwargs)

    return call


def _read_control_json(path, is_running=None):
    """Retry brief Windows sharing conflicts without hiding a failed worker."""
    deadline = time.monotonic() + 1.0
    while True:
        try:
            return read_json(path)
        except PermissionError as exc:
            winerror = getattr(exc, "winerror", None)
            retryable = winerror in {5, 32, 33} or (winerror is None and os.name == "nt" and exc.errno == errno.EACCES)
            if not retryable or (is_running is not None and not is_running()) or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


class ResidentWorker:
    def __init__(self, name, root, *, gpu_safety=False):
        self.name, self.root = name, Path(root)
        self.process = self.tree = self.directory = None
        self.key = None
        self.identifier = None
        self.log_path = None
        self._log_offset = 0
        self._exit_message = None
        self._exit_recorded = False
        self._stop_requested = False
        self._stop_expected_code = None
        self.gpu_safety = gpu_safety
        self._temporary = None
        self._guard = threading.RLock()
        atexit.register(self.close)

    @_synchronized
    def start(self, python, script, environment, payload, log_path):
        try:
            return self._start(python, script, environment, payload, log_path)
        except BaseException:
            # A failed spawn or command dispatch must not leave an untracked
            # process or its control files behind. close() confirms exit first.
            self.close()
            raise

    def _start(self, python, script, environment, payload, log_path):
        if self.gpu_safety:
            from modules_forge import gpu_fault

            gpu_fault.STATE.ensure_available()
            self.ensure_alive()
        key = (str(python), str(script), Path(script).stat().st_mtime_ns, tuple(sorted(environment.items())))
        reused = self.process is not None and self.process.poll() is None and key == self.key
        if not reused:
            self.close()
            if self.gpu_safety:
                gpu_fault.STATE.ensure_available()
            self.log_path = Path(log_path).resolve()
            from modules_forge.yue2_studio.service import ProcessTree

            self.root.mkdir(parents=True, exist_ok=True)
            # weakref's exit handler may run before our atexit.close callback.
            # Only close() may remove files, after the worker releases its log.
            self._temporary = tempfile.TemporaryDirectory(prefix=self.name + "-", dir=self.root, delete=False)
            self.directory = Path(self._temporary.name)
            boot_log = self.directory / "startup.log"
            with boot_log.open("wb") as stream:
                self.process = subprocess.Popen(  # noqa: S603 -- 登録済み専用Pythonと検証済みworkerをshellなしで実行。
                    [
                        str(python),
                        "-u",
                        str(Path(__file__).resolve()),
                        "--script",
                        str(Path(script).resolve()),
                        "--control",
                        str(self.directory.resolve()),
                    ],
                    stdin=subprocess.PIPE,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    cwd=ROOT,
                    close_fds=True,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            self.tree = ProcessTree(self.process)
            self.process.stdin.write(self.tree.handshake())
            self.process.stdin.flush()
            self.key = key
        self._log_offset = (self.directory / "startup.log").stat().st_size if reused else 0
        self._exit_message = None
        self.identifier = uuid.uuid4().hex
        log_path = Path(log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
        self.log_path = log_path
        atomic_json(
            self.directory / "command.json",
            {
                "id": self.identifier,
                "payload": payload,
                "log": str(log_path),
                "reused": reused,
            },
        )
        return reused

    @_synchronized
    def ensure_alive(self):
        if self.process is not None:
            exit_code = self.process.poll()
            if exit_code is not None:
                if self.gpu_safety:
                    from modules_forge import gpu_fault

                    gpu_fault.STATE.trip("worker_exit", self.name)
                raise WorkerExited(self._exit_diagnostic(exit_code), exit_code)

    @_synchronized
    def result(self):
        if self.directory is None:
            raise RuntimeError(f"{self.name} workerは終了しています。")
        if self.gpu_safety:
            self.ensure_alive()
        path = self.directory / "response.json"
        if path.exists():
            try:
                response = _read_control_json(path, lambda: self.process.poll() is None)
            except PermissionError:
                if self.process.poll() is None:
                    raise
            else:
                if response.get("id") == self.identifier:
                    if self.gpu_safety:
                        self.ensure_alive()
                    return response
        exit_code = self.process.poll()
        if exit_code is not None:
            raise WorkerExited(self._exit_diagnostic(exit_code), exit_code)
        return None

    def _exit_diagnostic(self, exit_code):
        self._record_exit(exit_code)
        if self._exit_message is not None:
            return self._exit_message
        source = self.directory / "startup.log"
        archive = ""
        if self.log_path is not None:
            destination = self.log_path.with_suffix(".native.log")
            try:
                shutil.copyfile(source, destination)
                archive = f" ネイティブログ: {destination}"
            except OSError as exc:
                archive = f" ネイティブログを保存できませんでした: {exc}"
        try:
            with source.open("rb") as stream:
                stream.seek(self._log_offset)
                detail = stream.read(3000)
                if stream.read(1):
                    stream.seek(-1500, os.SEEK_END)
                    detail = detail[:1500] + b"\n...\n" + stream.read(1500)
            detail = detail.decode("utf-8", errors="replace")
        except OSError as exc:
            detail = f"ネイティブログを読めませんでした: {exc}"
        code = f"{exit_code} / 0x{exit_code & 0xFFFFFFFF:08X}"
        self._exit_message = f"{self.name} workerが終了しました (終了コード {code})。{archive}\n{detail}"
        return self._exit_message

    def _record_exit(self, exit_code):
        if self.gpu_safety and not self._exit_recorded and self.log_path is not None:
            pid = getattr(self.process, "pid", None)
            record_event(
                self.log_path,
                "exit_observed",
                worker=self.name,
                command_id=self.identifier,
                pid=pid if type(pid) is int else None,
                exit_code=exit_code,
                stop_requested=self._stop_requested,
                expected_exit_code=self._stop_expected_code,
            )
            self._exit_recorded = True

    @_synchronized
    def close(self):
        try:
            self._close()
        except BaseException:
            if self.gpu_safety and self.process is not None and self.process.poll() is None:
                from modules_forge import gpu_fault

                gpu_fault.STATE.trip("worker_stop_unconfirmed", self.name)
            raise

    def _close(self):
        process = self.process
        if process is not None:
            exit_code = process.poll()
            if exit_code is not None and self.directory is not None:
                if self.gpu_safety and (not self._stop_requested or exit_code != self._stop_expected_code):
                    from modules_forge import gpu_fault

                    gpu_fault.STATE.trip("worker_exit", self.name)
                self._exit_diagnostic(exit_code)
            deadline = time.monotonic() + 15
            while process.poll() is None:
                if os.name == "nt":
                    expected_code = 130 if self.tree is not None and self.tree.handle is not None else 1
                else:
                    expected_code = -9
                if self.gpu_safety and not self._stop_requested and self.log_path is not None:
                    pid = getattr(process, "pid", None)
                    method = (
                        "TerminateJobObject"
                        if os.name == "nt" and expected_code == 130
                        else "kill_process_group"
                        if os.name != "nt" and self.tree is not None
                        else "kill_process"
                    )
                    record_event(
                        self.log_path,
                        "stop_requested",
                        worker=self.name,
                        command_id=self.identifier,
                        pid=pid if type(pid) is int else None,
                        method=method,
                        expected_exit_code=expected_code,
                    )
                if self.tree is not None:
                    self.tree.terminate()
                else:
                    process.kill()
                self._stop_requested = True
                self._stop_expected_code = expected_code
                try:
                    process.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    # Windowsの実Pythonが起動用Jobへ参加する前に停止を
                    # 要求した場合も、参加後の終了まで確認する。
            final_code = process.poll()
            self._record_exit(final_code)
            if self.gpu_safety and (not self._stop_requested or final_code != self._stop_expected_code):
                from modules_forge import gpu_fault

                gpu_fault.STATE.trip("worker_exit", self.name)
                self._exit_diagnostic(final_code)
            if self.gpu_safety:
                from modules_forge import gpu_fault

                if gpu_fault.STATE.blocked():
                    self._exit_diagnostic(final_code)
            if process.stdin is not None:
                process.stdin.close()
            if self.tree is not None:
                self.tree.close()
        self.process = self.tree = self.directory = self.key = None
        self.log_path = None
        self._log_offset = 0
        self._exit_message = None
        self._exit_recorded = False
        self._stop_requested = False
        self._stop_expected_code = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


class _LogStream:
    """Keep the stream captured by library handlers alive across job logs."""

    def __init__(self, destination):
        self.destination = destination

    def write(self, text):
        return self.destination.write(text)

    def flush(self):
        return self.destination.flush()

    def __getattr__(self, name):
        return getattr(self.destination, name)

    @contextlib.contextmanager
    def to(self, destination):
        previous = self.destination
        self.destination = destination
        try:
            yield
        finally:
            self.destination = previous


def serve(script, directory):
    from modules_forge.yue2_studio.worker import parent_guard

    parent_guard()
    stdout, stderr = _LogStream(sys.stdout), _LogStream(sys.stderr)
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        _serve_commands(script, directory, stdout, stderr)


def _serve_commands(script, directory, stdout, stderr):
    spec = importlib.util.spec_from_file_location("aikimi_resident_engine", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    last = None
    while True:
        command_path = directory / "command.json"
        if not command_path.exists():
            time.sleep(0.1)
            continue
        command = _read_control_json(command_path)
        if command["id"] == last:
            time.sleep(0.1)
            continue
        last = command["id"]
        result = {"id": last, "ok": False}
        fatal_exception = None
        try:
            with Path(command["log"]).open("a", encoding="utf-8", buffering=1) as log:
                with stdout.to(log), stderr.to(log):
                    try:
                        module.resident_run(command["payload"])
                        torch = sys.modules.get("torch")
                        if torch is not None and torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                        result["ok"] = True
                    except Exception as exc:
                        from modules_forge import gpu_fault

                        result["error"] = str(exc)
                        result["gpu_fault"] = gpu_fault.classify(exc)
                        if result["gpu_fault"]:
                            fatal_exception = exc
                        traceback.print_exc()
                        code = getattr(exc, "public_error_code", None)
                        if code in {"cancelled", "unsupported_attention"}:
                            result["error_code"] = code
            atomic_json(directory / "response.json", result)
        finally:
            if fatal_exception is not None:
                # The parent owns OS termination and the admission latch. Avoid
                # Python tensor destructors and any more CUDA work in this process.
                os._exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    arguments = parser.parse_args()
    serve(arguments.script, arguments.control)
