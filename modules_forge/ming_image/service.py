"""Ming jobs retain the shared GPU lease until their worker is idle or stopped."""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from PIL import Image

from modules_forge import gpu_fault
from modules_forge.resident_worker import WorkerExited, record_event

from . import checkpoints, references
from .core import (
    MODEL_ID,
    MODEL_REVISION,
    MODEL_ROOT,
    ROOT,
    MingImageError,
    atomic_json,
    model_ready,
    model_specs,
    png_metadata,
    python_path,
    safe_model_root,
    validate_request,
)

CANCEL_GRACE_SECONDS = 5.0


def _log_error(log, error):
    try:
        with log.open("a", encoding="utf-8") as stream:
            traceback.print_exception(error, file=stream)
    except OSError:
        pass


class Service:
    def __init__(self, root=MODEL_ROOT, *, outputs=None, worker_root=ROOT / "cache/ming-image-worker"):
        from modules.paths_internal import studio_output_dir
        from modules_forge.resident_worker import ResidentWorker

        self.root = Path(root)
        self.outputs = Path(outputs) if outputs is not None else studio_output_dir("ming-image")
        self.worker_root = Path(worker_root)
        self.worker = ResidentWorker("ming-image", self.worker_root, gpu_safety=True)
        self.jobs = {}
        self._guard = threading.RLock()

    def start(self, value):
        from modules_forge.gpu_ownership import GPUOwnership

        gpu_fault.STATE.ensure_available()
        try:
            self.worker.ensure_alive()
        except WorkerExited:
            raise gpu_fault.GPUFaultError() from None
        request = validate_request(value)
        model_specs(request)
        checkpoints.bind(self.root, request)
        outputs = safe_model_root(self.outputs)
        if not model_ready(self.root, request, explain=True):
            raise MingImageError(
                f"選択した形式をセットアップしてください: aikimi-ming-image-setup.bat --profile {request['profile']}"
            )
        lease = GPUOwnership()
        lease.engine = "ming-image"
        if not lease.acquire(False):
            raise MingImageError("GPUを使用中です。処理の終了後に実行してください。")
        identifier = uuid.uuid4().hex
        directory = outputs / identifier
        work_dir = self.worker_root / "jobs" / identifier
        log = self.worker_root / "logs" / (identifier + ".log")
        created = False
        try:
            safe_model_root(work_dir).mkdir(parents=True)
            created = True
            safe_model_root(log).parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir(parents=True)
            references.snapshot(work_dir, request)
            atomic_json(work_dir / "request.json", request)
            job = {"id": identifier, "directory": str(directory), "status": "running", "error": ""}
            with self._guard:
                self.jobs[identifier] = job
            threading.Thread(target=self.execute, args=(job, lease), daemon=True).start()
        except BaseException:
            with self._guard:
                self.jobs.pop(identifier, None)
            try:
                if created:
                    shutil.rmtree(safe_model_root(work_dir))
            finally:
                lease.release()
            raise
        return identifier

    @staticmethod
    def validate_result(directory, work_dir, request):
        directory = safe_model_root(directory)
        result = json.loads(safe_model_root(directory / "result.json").read_text(encoding="utf-8"))
        references.load(work_dir, request)
        path = safe_model_root(directory / "output.png")
        if result.get("image_path") != "output.png" or not path.is_file():
            raise MingImageError("生成結果の出力パスが一致しません。")
        if any(result.get(key) != value for key, value in png_metadata(request).items()):
            raise MingImageError("生成条件が元の要求と一致しません。")
        if result.get("model_id") != MODEL_ID or result.get("model_revision") != MODEL_REVISION:
            raise MingImageError("生成モデルが元の要求と一致しません。")
        elapsed = result.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise MingImageError("生成時間の記録が不正です。")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise MingImageError("生成結果がPNGではありません。")
            if image.size != (request["width"], request["height"]):
                raise MingImageError("生成結果のサイズが生成条件と一致しません。")
            if image.mode != ("RGBA" if request["transparent"] else "RGB"):
                raise MingImageError("PNGのチャンネル数が生成条件と一致しません。")
            embedded = image.info.get("ming_image")
        if not embedded:
            raise MingImageError("PNG内の生成条件がありません。")
        metadata = json.loads(embedded)
        if metadata != png_metadata(result) or result != {**metadata, "image_path": "output.png"}:
            raise MingImageError("PNG内の生成条件が保存記録と一致しません。")
        return result

    def execute(self, job, lease):
        from modules_forge import gpu_residency
        from modules_forge.gpu_ownership import release_forge_vram

        directory = Path(job["directory"])
        work_dir = self.worker_root / "jobs" / job["id"]
        log = self.worker_root / "logs" / (job["id"] + ".log")
        environment = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1",
        )
        status = "error"
        pending = False
        try:
            if (work_dir / "cancel").exists():
                status = "cancelled"
                return
            self.worker.ensure_alive()
            gpu_fault.STATE.begin(job["id"])
            pending = True
            release_forge_vram()
            self.worker.start(
                python_path(self.root),
                ROOT / "tools" / "ming_image_worker.py",
                environment,
                {"root": str(self.root), "job_dir": str(directory), "work_dir": str(work_dir)},
                log,
            )
            gpu_residency.register("ming-image", self.worker.close, "ming-image")
            cancel_deadline = None
            while True:
                response = self.worker.result()
                if response is not None:
                    if not response["ok"]:
                        if response.get("gpu_fault"):
                            gpu_fault.STATE.trip("gpu_execution", "ming-image")
                        if response.get("error_code") == "cancelled" and (work_dir / "cancel").exists():
                            record_event(log, "cancel_acknowledged", job_id=job["id"])
                            status = "cancelled"
                            break
                        raise RuntimeError(response.get("error", "Ming Imageの生成に失敗しました。"))
                    request = json.loads(safe_model_root(work_dir / "request.json").read_text(encoding="utf-8"))
                    self.validate_result(directory, work_dir, request)
                    with self._guard:
                        status = "cancelled" if (work_dir / "cancel").exists() else "complete"
                        job["_finishing"] = True
                    break
                if (work_dir / "cancel").exists():
                    if cancel_deadline is None:
                        cancel_deadline = time.monotonic() + CANCEL_GRACE_SECONDS
                        record_event(log, "cancel_observed", job_id=job["id"])
                        with self._guard:
                            job["status"] = "stopping"
                    if time.monotonic() >= cancel_deadline:
                        status = "cancelled"
                        break
                time.sleep(0.1)
        except Exception as exc:
            reason = "worker_exit" if isinstance(exc, WorkerExited) else gpu_fault.classify(exc)
            if reason:
                gpu_fault.STATE.trip(reason, "ming-image")
            _log_error(log, exc)
            job["error"] = (
                str(exc)
                if isinstance(exc, MingImageError)
                else "生成に失敗しました。Forge Neoのログを確認してください。"
            )
        finally:
            with self._guard:
                job["_finishing"] = True

            def stop_worker():
                while True:
                    try:
                        self.worker.close()
                        break
                    except Exception as exc:
                        gpu_fault.STATE.trip("worker_stop_unconfirmed", "ming-image")
                        _log_error(log, exc)
                        with self._guard:
                            job.update(status="stopping", error="停止を確認中…")
                        time.sleep(0.5)

            def finish_cleanup(action):
                # Windows readers can briefly hold a file open. Keep ownership until cleanup succeeds.
                delay = 0.1
                while True:
                    try:
                        action()
                        return
                    except OSError as exc:
                        if delay == 0.1:
                            _log_error(log, exc)
                        with self._guard:
                            job["status"] = "stopping"
                        time.sleep(delay)
                        delay = min(2, delay * 2)

            def remove_output():
                for name in ("output.png", ".output.png.tmp", "result.json"):
                    (directory / name).unlink(missing_ok=True)
                for temporary in directory.glob(".result.json.*.tmp"):
                    temporary.unlink(missing_ok=True)

            def remove_work_dir():
                path = safe_model_root(work_dir)
                if path.exists():
                    shutil.rmtree(path)

            if status != "complete" or gpu_fault.STATE.blocked():
                stop_worker()
                finish_cleanup(remove_output)

            def complete_pending():
                self.worker.ensure_alive()
                gpu_fault.STATE.complete(job["id"])

            try:
                finish_cleanup(remove_work_dir)
            except MingImageError as exc:
                status = "error"
                job["error"] = "作業ファイルを削除できませんでした。"
                _log_error(log, exc)
                stop_worker()
                finish_cleanup(remove_output)

            try:
                if pending:
                    # Finish cleanup and the journal while FIFO ownership is held.
                    lease.release(before_unlock=complete_pending)
                else:
                    lease.release()
            except Exception as exc:
                gpu_fault.STATE.trip(gpu_fault.classify(exc) or "cleanup_failed", "ming-image")
                _log_error(log, exc)
                stop_worker()
                finish_cleanup(remove_output)
            if gpu_fault.STATE.blocked():
                status = "error"
                job.update(error=gpu_fault.MESSAGE, error_code="gpu_stopped")
            try:
                if status != "complete":
                    from modules.aikimi_security.redaction import redact_text

                    record = {"status": status, "error": redact_text(job["error"])}
                    if job.get("error_code"):
                        record["error_code"] = job["error_code"]
                    atomic_json(directory / "result.json", record)
            finally:
                with self._guard:
                    job["status"] = status

    def poll(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job is None:
                return {"status": "idle"}
            result = {key: value for key, value in job.items() if not key.startswith("_")}
        status = result["status"]
        directory = Path(result["directory"])
        terminal = status in {"complete", "error", "cancelled"}
        try:
            path = directory / "result.json" if terminal else self.worker_root / "jobs" / identifier / "progress.json"
            record = json.loads(path.read_text(encoding="utf-8"))
            result.update(
                record if terminal else {key: record[key] for key in ("stage", "current", "total") if key in record}
            )
        except (OSError, ValueError):
            pass
        result["status"] = status
        if status == "complete":
            result["image_path"] = str(safe_model_root(directory / "output.png"))
            result["files"] = [result["image_path"], str(directory / "result.json")]
        return result

    def cancel(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job and job["status"] in {"running", "stopping"} and not job.get("_finishing"):
                work_dir = self.worker_root / "jobs" / identifier
                log = self.worker_root / "logs" / (identifier + ".log")
                if not (work_dir / "cancel").exists():
                    (work_dir / "cancel").touch()
                    job["status"] = "stopping"
                    record_event(log, "cancel_requested", job_id=identifier)
                return "中断処理中…"
        return "実行中の処理はありません。"


SERVICE = Service()
