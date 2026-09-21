"""Own Qwen jobs, local inputs, and GPU leases independently of UI connections."""

from __future__ import annotations

import atexit
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .annotations import snapshot_annotation
from .core import (
    QwenImage21Error,
    Request,
    atomic_json,
    copy_inputs,
    inside,
    read_json,
    runtime_lock,
    runtime_manifest,
    safe_environment,
)
from .prompt_rewriter import rewriter_manifest

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / "tools" / "qwen_image21_worker.py"
ENGINE = "qwen_image21"
MAX_FINISHED_JOBS = 64


class JobNotFound(QwenImage21Error):
    """The requesting browser does not own an available job."""


@dataclass
class Job:
    identifier: str
    owner: str
    directory: Path
    started: float = field(default_factory=time.monotonic)
    cancel: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    final: dict | None = None
    completion_committed: bool = False
    message: str = "実行環境を準備中"
    thread: threading.Thread | None = None


class Studio:
    def __init__(
        self,
        runtime: Path,
        outputs: Path,
        ownership_factory=None,
        release_vram=None,
        worker_factory=None,
        residency=None,
        poll_interval: float = 0.1,
        cancel_grace: float = 2.0,
    ):
        self.runtime, self.outputs = Path(runtime), Path(outputs)
        self._ownership_factory = ownership_factory
        self._release_vram = release_vram
        self._worker_factory = worker_factory
        self._residency = residency
        self._poll_interval, self._cancel_grace = poll_interval, cancel_grace
        self._guard = threading.RLock()
        self._jobs: dict[str, Job] = {}
        self._resident = None
        self._runtime_lock = None
        self._closed = False
        atexit.register(self.shutdown)

    def _dependencies(self):
        if self._ownership_factory is None:
            from modules_forge.gpu_ownership import GPUOwnership, release_forge_vram

            self._ownership_factory, self._release_vram = GPUOwnership, release_forge_vram
        if self._worker_factory is None:
            from modules_forge.resident_worker import ResidentWorker

            self._worker_factory = ResidentWorker
        if self._residency is None:
            from modules_forge import gpu_residency

            self._residency = gpu_residency

    def start(self, request: Request, owner: str) -> str:
        request = request.resolved()
        if not isinstance(owner, str) or not owner:
            raise QwenImage21Error("ブラウザーのQwen Image 2.1タブから操作してください。")
        runtime_manifest(self.runtime)
        if request.rewrite_prompt and not request.input_images:
            rewriter_manifest(self.runtime)
        self._dependencies()
        with self._guard:
            if self._closed:
                raise QwenImage21Error("この画面の実行環境は終了しました。画面を再読み込みしてください。")
            if any(not job.done.is_set() for job in self._jobs.values()):
                raise QwenImage21Error("Qwen Image 2.1は実行中です。完了または停止確認後に再実行してください。")
            lease = self._ownership_factory()
            lease.engine = ENGINE
            if not lease.acquire(blocking=False):
                raise QwenImage21Error("別の生成がGPUを使用中です。完了後に再実行してください。")
            job = None
            new_lock = False
            try:
                if self._runtime_lock is None:
                    # A UI rebuild may have left a different Studio's idle worker.
                    self._residency.release_resource(ENGINE)
                    self._runtime_lock = runtime_lock(self.runtime)
                    new_lock = True
                entry = runtime_manifest(self.runtime)
                if request.rewrite_prompt and not request.input_images:
                    rewriter_manifest(self.runtime)
                directory = self.outputs / uuid.uuid4().hex
                directory.mkdir(parents=True, exist_ok=False)
                job = Job(directory.name, owner, directory)
                payload = request.to_dict()
                clean_paths = copy_inputs(request.input_images, directory)
                model_paths, instruction, annotation = snapshot_annotation(
                    clean_paths, request.annotation_reference, request.annotation_layers, directory
                )
                payload["input_images"] = model_paths
                payload["user_prompt"] = request.prompt
                payload["prompt"] = request.prompt + instruction
                payload["annotation"] = annotation
                payload["annotation_layers"] = annotation.get("layer_paths", [])
                atomic_json(directory / "request.json", payload)
                atomic_json(directory / "status.json", {"state": "running", "message": job.message})
                self._jobs = dict(list(self._jobs.items())[-MAX_FINISHED_JOBS:])
                self._jobs[job.identifier] = job
                thread = threading.Thread(target=self._run, args=(job, request, entry, lease), daemon=True)
                job.thread = thread
                thread.start()
                return job.identifier
            except BaseException:
                if job is not None:
                    self._jobs.pop(job.identifier, None)
                # No process can be started until the supervisor thread starts.
                try:
                    if new_lock and self._resident is None:
                        self._runtime_lock.close()
                        self._runtime_lock = None
                finally:
                    lease.release()
                raise

    def _release_idle(self):
        """Called under the GPU queue; do not obtain another GPU lease here."""
        with self._guard:
            if self._resident is not None:
                self._resident.close()
                self._resident = None
            if self._runtime_lock is not None:
                self._runtime_lock.close()
                self._runtime_lock = None

    def _report(self, job: Job, message: str):
        job.message = message
        try:
            atomic_json(job.directory / "status.json", {"state": "running", "message": message})
        except OSError:
            # Reporting failures must never skip process termination or return the GPU.
            pass

    def _stop_confirmed(self, job: Job):
        while True:
            try:
                self._release_idle()
                return
            except Exception:
                self._report(job, "workerの終了確認を待っています。GPUの使用権は保持しています。")
                time.sleep(self._poll_interval)

    def _run(self, job: Job, request: Request, entry: dict, lease):
        success = False
        final = {"state": "failed", "message": "実行を完了できませんでした。"}
        try:
            if job.cancel.is_set():
                raise InterruptedError("開始前に停止しました。")
            if self._release_vram is not None:
                self._release_vram()
            if self._resident is None:
                self._resident = self._worker_factory(ENGINE, self.runtime / "sessions")
            resident = self._resident
            from modules_forge.jev_sparse.qwen21_integration import worker_launch

            worker, environment, sparse_payload = worker_launch(request, WORKER, safe_environment())
            reused = resident.start(
                entry["python"],
                worker,
                environment,
                {
                    "model_path": entry["model"],
                    "precision": request.precision,
                    "memory_mode": request.memory_mode,
                    "job_dir": str(job.directory.resolve()),
                    **sparse_payload,
                },
                job.directory / "worker.log",
            )
            # The resident reference is already owned before start(), even if start
            # or writing this diagnostic record fails after a process was created.
            atomic_json(job.directory / "worker-session.json", {"reused": reused, "pid": resident.process.pid})
            cancelled_at = None
            while True:
                response = resident.result()
                if response is not None:
                    break
                if job.cancel.wait(self._poll_interval):
                    if cancelled_at is None:
                        (job.directory / "cancel").touch()
                        cancelled_at = time.monotonic()
                        self._report(job, "停止を要求しました。workerの終了を確認中です。")
                    if time.monotonic() - cancelled_at >= self._cancel_grace:
                        raise InterruptedError("停止しました。")
                    # Event.wait returns immediately once set; keep cancellation
                    # polling bounded while allowing the worker to stop itself.
                    time.sleep(self._poll_interval)
            if job.cancel.is_set():
                raise InterruptedError("停止しました。")
            if not response.get("ok"):
                raise QwenImage21Error(response.get("error") or "workerでエラーが発生しました。")
            result = read_json(job.directory / "result.json")
            output = inside(job.directory, job.directory / "output.png")
            if not isinstance(result, dict) or Path(result.get("output_path", "")).resolve() != output:
                raise QwenImage21Error("生成結果の保存先を確認できません。worker.logを確認してください。")
            with Image.open(output) as image:
                if image.format != "PNG" or image.size != (request.width, request.height):
                    raise QwenImage21Error("生成画像の形式またはサイズが要求と一致しません。")
                image.verify()
            with self._guard:
                if job.cancel.is_set():
                    raise InterruptedError("停止しました。")
                self._residency.register(ENGINE, self._release_idle, ENGINE)
                metadata = result.get("metadata", {})
                rewrite = metadata.get("prompt_rewrite", {})
                rewrite_message = " · 書き換え4bit" if rewrite.get("applied") else ""
                if request.rewrite_prompt and request.input_images:
                    rewrite_message = " · 編集のため書き換え省略"
                final = {
                    "state": "complete",
                    "message": f"完了 · Seed {request.seed} · {request.width}×{request.height} · {request.precision.upper()}{rewrite_message}",
                    "output_path": str(output),
                    "seed": request.seed,
                    "progress": 1.0,
                    "effective_prompt": metadata.get("effective_prompt", request.prompt),
                    "prompt_rewrite": rewrite,
                }
                job.completion_committed = True
                success = True
        except BaseException as exc:
            final = {
                "state": "cancelled" if job.cancel.is_set() else "failed",
                "message": str(exc) or type(exc).__name__,
            }
            if final["state"] == "failed":
                final["message"] += f"  ログ: {job.directory / 'worker.log'}"
        finally:
            if not success:
                self._stop_confirmed(job)
            try:
                # On success the worker has synchronized CUDA and is safely idle;
                # on failure/cancel the entire process tree has confirmed exit.
                lease.release()
            except Exception as exc:
                final["message"] += f"  待機モデルの解放を再試行してください: {exc}"
            finally:
                job.final = final
                try:
                    atomic_json(job.directory / "status.json", final)
                except OSError:
                    pass
                job.done.set()

    def _job(self, identifier: str, owner: str) -> Job:
        with self._guard:
            job = self._jobs.get(identifier)
        if job is None or job.owner != owner:
            raise JobNotFound("この画面の実行ジョブが見つかりません。画面を再読み込みしてください。")
        return job

    def status(self, identifier: str, owner: str) -> dict:
        job = self._job(identifier, owner)
        done = job.done.is_set()
        state = job.final if done else {"state": "running", "message": job.message}
        state = dict(state)
        if not done and not job.cancel.is_set():
            try:
                progress = read_json(job.directory / "progress.json")
                if isinstance(progress, dict):
                    state.update({key: progress[key] for key in ("message", "progress", "stage") if key in progress})
            except (OSError, ValueError):
                pass
        return {**state, "elapsed": time.monotonic() - job.started, "done": done}

    def cancel(self, identifier: str, owner: str) -> bool:
        with self._guard:
            try:
                job = self._job(identifier, owner)
            except JobNotFound:
                return False
            if job.done.is_set() or job.completion_committed:
                return False
            job.cancel.set()
            return True

    def artifact(self, identifier: str, owner: str) -> Path:
        job = self._job(identifier, owner)
        if not job.done.is_set() or job.final.get("state") != "complete":
            raise QwenImage21Error("このジョブの完成画像はまだありません。")
        output = inside(self.outputs, job.directory / "output.png")
        if not output.is_file():
            raise QwenImage21Error("保存画像が見つかりません。")
        return output

    def shutdown(self):
        with self._guard:
            self._closed = True
            active = [job for job in self._jobs.values() if not job.done.is_set()]
        for job in active:
            job.cancel.set()
        for job in active:
            if job.thread is not None and job.thread is not threading.current_thread():
                job.thread.join(timeout=20)
        if all(job.done.is_set() for job in active):
            # A running supervisor retains the lease and continues its own cleanup.
            self._release_idle()
