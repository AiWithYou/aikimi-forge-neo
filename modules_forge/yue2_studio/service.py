"""Own a worker and GPU lease independently of a browser's connection lifetime."""
from __future__ import annotations

import atexit
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .core import Request, YuE2Error, atomic_json, inside, read_json, runtime_manifest, runtime_lock, safe_environment


class ProcessTree:
    """Only the process tree created here can be terminated. Windows fails closed."""

    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.handle = None
        self.name = None
        if os.name != "nt":
            return
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [("process_time", c.c_longlong), ("job_time", c.c_longlong),
                        ("flags", w.DWORD), ("min_working", c.c_size_t),
                        ("max_working", c.c_size_t), ("active", w.DWORD),
                        ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class Extended(c.Structure):
            _fields_ = [("basic", Basic), ("io", c.c_ulonglong * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        self.kernel = c.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.name = "Local\\AikimiYuE2-" + uuid.uuid4().hex
        handle = self.kernel.CreateJobObjectW(None, self.name)
        if not handle:
            raise OSError(c.get_last_error(), "Windows Job Objectを作成できません。")
        info = Extended()
        info.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.kernel.SetInformationJobObject(handle, 9, c.byref(info), c.sizeof(info)):
            self.kernel.CloseHandle(handle)
            raise OSError(c.get_last_error(), "Windows Job Objectの保護設定に失敗しました。")
        # venvランチャーには独自のJobがあるため、ここでは空のJobを作る。
        # 実PythonがGOを受けた後、自身を登録してからモデルを読み込む。
        self.handle = handle

    def handshake(self):
        return b"GO\n" + ((self.name + "\n").encode("ascii") if self.name else b"")

    def terminate(self):
        if self.handle is not None:
            if not self.kernel.TerminateJobObject(self.handle, 130):
                raise OSError("YuE2のプロセスツリーを停止できません。")
        elif os.name != "nt":
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        elif self.process.poll() is None:
            self.process.kill()

    def close(self):
        if self.handle is not None:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


@dataclass
class Job:
    identifier: str
    owner: str
    directory: Path
    started: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    final: dict | None = None
    process: subprocess.Popen | None = None
    tree: ProcessTree | None = None
    guard: threading.Lock = field(default_factory=threading.Lock)


class Studio:
    def __init__(self, runtime: Path, outputs: Path, ownership_factory=None, release_vram=None):
        self.runtime, self.outputs = runtime, outputs
        self._ownership_factory, self._release_vram = ownership_factory, release_vram
        self._guard = threading.Lock()
        self._jobs: dict[str, Job] = {}
        self._resident = None
        self._runtime_lock = None
        atexit.register(self.shutdown)

    def start(self, request: Request, owner: str, plan_only: bool = False) -> str:
        request = request.resolved()
        if plan_only and (request.engine != "official" or request.cot == "off"):
            raise YuE2Error("楽譜だけの生成は公式Python・楽譜ありの方式で使用できます。")
        if self._ownership_factory is None:
            from modules_forge.gpu_ownership import GPUOwnership, release_forge_vram
            factory, release = GPUOwnership, release_forge_vram
        else:
            factory, release = self._ownership_factory, self._release_vram
        with self._guard:
            if any(not job.done.is_set() for job in self._jobs.values()):
                raise YuE2Error("YuE2は実行中です。完了または停止確認後に再実行してください。")
            lock = None
            lease = factory()
            lease.engine = "yue2"
            job = None
            try:
                runtime_manifest(self.runtime, request.engine)
                if not lease.acquire(blocking=False):
                    raise YuE2Error("別の生成がGPUを使用中です。完了後に再実行してください。")
                if self._runtime_lock is None:
                    from modules_forge import gpu_residency

                    # UI再構築前のStudioが保持していた待機workerを回収する。
                    gpu_residency.release_resource("yue2")
                    self._runtime_lock = runtime_lock(self.runtime)
                lock = self._runtime_lock
                directory = self.outputs / uuid.uuid4().hex
                directory.mkdir(parents=True, exist_ok=False)
                job = Job(directory.name, owner, directory)
                atomic_json(directory / "project.json", {"schema": 1, "request": asdict(request), "plan_only": plan_only})
                atomic_json(directory / "status.json", {"state": "running", "message": "実行環境を準備中"})
                self._jobs = {k: v for k, v in self._jobs.items() if not v.done.is_set()}
                self._jobs[job.identifier] = job
                thread = threading.Thread(target=self._run, args=(job, request, lease, release, lock), daemon=True)
                thread.start()
            except BaseException:
                if job is not None:
                    self._jobs.pop(job.identifier, None)
                lease.release()
                if lock is not None:
                    self._release_idle()
                raise
            return job.identifier

    def _release_idle(self):
        if self._resident is not None:
            self._resident.close()
            self._resident = None
        if self._runtime_lock is not None:
            self._runtime_lock.close()
            self._runtime_lock = None

    def _run(self, job: Job, request: Request, lease, release, lock):
        from modules_forge import gpu_residency
        from modules_forge.resident_worker import ResidentWorker

        success = False
        try:
            if job.cancel.is_set():
                raise InterruptedError("開始前に停止しました。")
            if release is not None:
                release()
            entry = runtime_manifest(self.runtime, request.engine)
            python = entry["python"] if request.engine == "official" else sys.executable
            if not Path(python).is_file():
                raise YuE2Error("専用Pythonがありません。YuE2を再セットアップしてください。")
            if self._resident is None:
                self._resident = ResidentWorker("yue2", self.runtime / "sessions")
            resident = self._resident
            reused = resident.start(python, Path(__file__).with_name("worker.py"), safe_environment(),
                                    {"runtime": str(self.runtime.resolve()), "job": str(job.directory.resolve())},
                                    job.directory / "worker.log")
            atomic_json(job.directory / "worker-session.json", {"reused": reused, "pid": resident.process.pid})
            with job.guard:
                job.process, job.tree = resident.process, resident.tree
            cancelled_at = None
            while True:
                result = resident.result()
                if result is not None:
                    break
                if job.cancel.wait(0.1):
                    if cancelled_at is None:
                        (job.directory / "cancel").touch()
                        cancelled_at = time.monotonic()
                    if time.monotonic() - cancelled_at >= 3:
                        resident.close()
                        raise InterruptedError("停止しました。完了済みの候補は履歴に残っています。")
            saved = read_json(job.directory / "status.json")
            if job.cancel.is_set():
                raise InterruptedError("停止しました。完了済みの候補は履歴に残っています。")
            if not result["ok"] or saved.get("state") != "complete":
                raise YuE2Error(result.get("error") or saved.get("message", "ログを確認してください。"))
            final = saved
            success = True
        except BaseException as exc:
            final = {"state": "cancelled" if job.cancel.is_set() else "failed", "message": str(exc)}
        finally:
            try:
                if success and request.engine == "official":
                    gpu_residency.register("yue2", self._release_idle, "yue2")
                else:
                    # 停止失敗時も実プロセスが終了するまではGPU所有権を保持する。
                    while job.process is not None and job.process.poll() is None:
                        try:
                            self._release_idle()
                        except (OSError, subprocess.TimeoutExpired):
                            atomic_json(job.directory / "status.json", {
                                "state": "running", "message": "workerの終了確認を待っています。",
                            })
                            time.sleep(0.5)
                    self._release_idle()
                if final["state"] == "failed":
                    final["message"] += f"  ログ: {job.directory / 'worker.log'}"
                job.final = final
                atomic_json(job.directory / "status.json", final)
            finally:
                try:
                    lease.release()
                finally:
                    job.done.set()

    def status(self, identifier: str, owner: str) -> dict:
        with self._guard:
            job = self._jobs.get(identifier)
        if job is None or job.owner != owner:
            raise YuE2Error("この画面の実行ジョブが見つかりません。履歴を更新してください。")
        state = job.final if job.done.is_set() else read_json(job.directory / "status.json")
        return {**state, "elapsed": time.monotonic() - job.started, "done": job.done.is_set()}

    def cancel(self, identifier: str, owner: str) -> bool:
        with self._guard:
            job = self._jobs.get(identifier)
            if job is None or job.owner != owner or job.done.is_set():
                return False
            job.cancel.set()
            return True

    def shutdown(self):
        with self._guard:
            jobs = list(self._jobs.values())
        for job in jobs:
            if not job.done.is_set():
                job.cancel.set()
                with job.guard:
                    if job.tree:
                        try:
                            job.tree.terminate()
                        except OSError:
                            pass

        self._release_idle()

    def history(self) -> list[tuple[str, str]]:
        entries = []
        if not self.outputs.exists():
            return entries
        for directory in sorted(self.outputs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not re_job(directory.name) or directory.is_symlink() or not directory.is_dir():
                continue
            for item in sorted(directory.glob("take-*")):
                try:
                    item = inside(self.outputs, item)
                    meta = read_json(item / "studio-result.json")
                    if meta.get("state") != "complete":
                        continue
                    title = str(meta.get("title") or "無題")[:160]
                    marker = " · 上限到達" if meta.get("truncated") else ""
                    entries.append((f"{title} / {item.name} / seed {meta['seed']}{marker}",
                                    f"{directory.name}/{item.name}"))
                except (OSError, ValueError, KeyError):
                    continue
            if len(entries) >= 60:
                break
        return entries[:60]

    def artifact(self, key: str) -> Path:
        import re
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{32}/take-[1-8]", key):
            raise YuE2Error("履歴の指定が不正です。")
        return inside(self.outputs, self.outputs / key)


def re_job(value: str) -> bool:
    import re
    return re.fullmatch(r"[0-9a-f]{32}", value) is not None
