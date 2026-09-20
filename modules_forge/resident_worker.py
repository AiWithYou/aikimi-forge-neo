"""専用Pythonをジョブ間で再利用する。ファイル通信で標準入力の競合を避ける。"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import importlib.util
import os
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules_forge.yue2_studio.core import atomic_json, read_json  # noqa: E402


def _synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._guard:
            return method(self, *args, **kwargs)

    return call


class ResidentWorker:
    def __init__(self, name, root):
        self.name, self.root = name, Path(root)
        self.process = self.tree = self.directory = None
        self.key = None
        self.identifier = None
        self._temporary = None
        self._guard = threading.RLock()
        atexit.register(self.close)

    @_synchronized
    def start(self, python, script, environment, payload, log_path):
        key = (str(python), str(script), Path(script).stat().st_mtime_ns, tuple(sorted(environment.items())))
        reused = self.process is not None and self.process.poll() is None and key == self.key
        if not reused:
            self.close()
            from modules_forge.yue2_studio.service import ProcessTree

            self.root.mkdir(parents=True, exist_ok=True)
            self._temporary = tempfile.TemporaryDirectory(prefix=self.name + "-", dir=self.root)
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
            try:
                self.tree = ProcessTree(self.process)
                self.process.stdin.write(self.tree.handshake())
                self.process.stdin.flush()
            except BaseException:
                self.close()
                raise
            self.key = key
        self.identifier = uuid.uuid4().hex
        log_path = Path(log_path).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text("", encoding="utf-8")
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
    def result(self):
        if self.directory is None:
            raise RuntimeError(f"{self.name} workerは終了しています。")
        path = self.directory / "response.json"
        if path.exists():
            response = read_json(path)
            if response.get("id") == self.identifier:
                return response
        if self.process.poll() is not None:
            detail = (self.directory / "startup.log").read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"{self.name} workerが終了しました: {detail}")
        return None

    @_synchronized
    def close(self):
        process = self.process
        if process is not None:
            deadline = time.monotonic() + 15
            while process.poll() is None:
                if self.tree is not None:
                    self.tree.terminate()
                else:
                    process.kill()
                try:
                    process.wait(timeout=min(0.1, max(0, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
                    # Windowsの実Pythonが起動用Jobへ参加する前に停止を
                    # 要求した場合も、参加後の終了まで確認する。
            if process.stdin is not None:
                process.stdin.close()
            if self.tree is not None:
                self.tree.close()
        self.process = self.tree = self.directory = self.key = None
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = None


def serve(script, directory):
    from modules_forge.yue2_studio.worker import parent_guard

    parent_guard()
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
        command = read_json(command_path)
        if command["id"] == last:
            time.sleep(0.1)
            continue
        last = command["id"]
        result = {"id": last, "ok": False}
        with Path(command["log"]).open("a", encoding="utf-8", buffering=1) as log:
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    module.resident_run(command["payload"])
                    torch = sys.modules.get("torch")
                    if torch is not None and torch.cuda.is_initialized():
                        torch.cuda.synchronize()
                    result["ok"] = True
                except Exception as exc:
                    traceback.print_exc()
                    result["error"] = str(exc)
        atomic_json(directory / "response.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    arguments = parser.parse_args()
    serve(arguments.script, arguments.control)
