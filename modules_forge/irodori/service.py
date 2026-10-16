"""Server-owned speech jobs share Forge's GPU queue and retention policy."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import uuid
from pathlib import Path

from .core import (
    MODELS,
    ROOT,
    RUNTIME,
    IrodoriError,
    atomic_json,
    child_environment,
    python_path,
    ready,
    stage_references,
    validate_request,
    verify_result,
)


class Service:
    def __init__(self, root=RUNTIME, models=MODELS, outputs=None, worker=None):
        from modules_forge.resident_worker import ResidentWorker

        self.root, self.models = Path(root), Path(models)
        self.outputs = Path(outputs or ROOT / "outputs/irodori")
        self.worker = worker or ResidentWorker("irodori", ROOT / "cache/irodori-worker")
        self.jobs = {}
        self.guard = threading.Lock()
        self._precision = None

    @staticmethod
    def new_job(owner, directory):
        return {
            "owner": owner,
            "directory": Path(directory),
            "status": "running",
            "error": "",
            "cancel": threading.Event(),
            "accept_cancel": True,
        }

    def close_safely(self):
        # Residency calls this inside the real queue lock, including engine
        # switches and idle expiry. Never return while a worker may still run.
        while True:
            try:
                self.worker.close()
                self._precision = None
                return
            except Exception:
                logging.getLogger(__name__).warning("Irodori workerの終了確認を再試行します", exc_info=True)
                time.sleep(0.5)

    def start(self, value, references, owner):
        from modules_forge.gpu_ownership import GPUOwnership

        request = validate_request(value)
        if not owner:
            raise IrodoriError("ブラウザーを再読み込みしてから生成してください。")
        if not ready(self.root, self.models):
            raise IrodoriError("aikimi-irodori-setup.batで実行環境とモデルを準備してください。")
        with self.guard:
            if any(x["status"] in {"running", "stopping"} for x in self.jobs.values()):
                raise IrodoriError("Irodori TTSは実行中です。終了を待ってください。")
            identifier = uuid.uuid4().hex
            directory = self.outputs / identifier
            directory.mkdir(parents=True)
            lease = GPUOwnership()
            lease.engine = "irodori"
            acquired = False
            try:
                request["references"] = stage_references(references or [], directory)
                atomic_json(directory / "request.json", request)
                acquired = lease.acquire(False)
                if not acquired:
                    raise IrodoriError("GPUを使用中です。処理の終了後に実行してください。")
                job = self.new_job(owner, directory)
                self.jobs[identifier] = job
                thread = threading.Thread(target=self.execute, args=(job, lease), daemon=True)
                thread.start()
            except BaseException:
                self.jobs.pop(identifier, None)
                if acquired:
                    lease.release()
                shutil.rmtree(directory)
                raise
            return identifier

    def execute(self, job, lease):
        from modules_forge import gpu_residency
        from modules_forge.gpu_ownership import release_forge_vram

        directory = job["directory"]
        status, registered = "error", False
        try:
            release_forge_vram()
            precision = json.loads((directory / "request.json").read_text(encoding="utf-8"))["precision"]
            if self._precision is not None and self._precision != precision:
                self.close_safely()
            environment = child_environment(offline=True)
            gpu_residency.register("irodori", self.close_safely, "irodori")
            registered = True
            self.worker.start(
                python_path(self.root),
                ROOT / "tools/irodori_worker.py",
                environment,
                {"root": str(self.root), "models": str(self.models), "job_dir": str(directory)},
                directory / "run.log",
            )
            self._precision = precision
            while not job["cancel"].is_set():
                response = self.worker.result()
                if response is not None:
                    if not response.get("ok"):
                        raise IrodoriError(response.get("error", "音声生成に失敗しました。"))
                    verify_result(directory)
                    status = "complete"
                    break
                time.sleep(0.15)
        except Exception as exc:
            job["error"] = str(exc)
        finally:
            with self.guard:
                if job["cancel"].is_set():
                    status = "cancelled"
                job["accept_cancel"] = False
                job["status"] = "stopping"
            if status != "complete":
                while True:
                    try:
                        gpu_residency.release_resource("irodori") if registered else self.close_safely()
                        break
                    except Exception as exc:
                        job["error"] = f"終了確認中: {exc}"
                        time.sleep(0.5)
            try:
                atomic_json(directory / "state.json", {"status": status, "error": job["error"]})
            finally:
                try:
                    lease.release()
                finally:
                    with self.guard:
                        job["status"] = status

    def owned(self, identifier, owner):
        job = self.jobs.get(identifier)
        if job is None or not owner or job["owner"] != owner:
            raise IrodoriError("このセッションの生成履歴が見つかりません。")
        return job

    def poll(self, identifier, owner):
        job = self.owned(identifier, owner)
        result = {"status": job["status"], "error": job["error"]}
        if job["status"] == "complete":
            result["result"] = verify_result(job["directory"])
            result["audio"] = str(job["directory"] / "audio.wav")
            result["files"] = [str(job["directory"] / x) for x in ("audio.wav", "request.json", "result.json")]
        return result

    def cancel(self, identifier, owner):
        with self.guard:
            job = self.owned(identifier, owner)
            if job["accept_cancel"] and job["status"] == "running":
                job["cancel"].set()
                return "中断処理中…"
            if job["status"] == "stopping":
                return "終了処理中です。"
            return "処理は終了しています。"


SERVICE = Service()
