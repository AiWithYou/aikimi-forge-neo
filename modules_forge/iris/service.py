"""Background jobs own the GPU lease until the resident worker is safe."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .core import ROOT, RUNTIME, IrisError, atomic_json, model_ready, python_path, validate_request


class Service:
    def __init__(self, root=RUNTIME):
        from modules_forge.resident_worker import ResidentWorker

        self.root = Path(root)
        self.worker = ResidentWorker("iris", ROOT / "cache" / "iris-worker")
        self.jobs = {}

    def start(self, request):
        from modules_forge.gpu_ownership import GPUOwnership

        request = validate_request(request)
        if not model_ready(self.root, request["precision"], request["task"]):
            raise IrisError("選択したモデルを準備してください。")
        lease = GPUOwnership()
        lease.engine = "iris"
        if not lease.acquire(False):
            raise IrisError("GPUを使用中です。処理の終了後に実行してください。")
        identifier = uuid.uuid4().hex
        directory = ROOT / "outputs" / "iris" / identifier
        try:
            directory.mkdir(parents=True)
            if request["task"] != "generate":
                image = request["image"]
                if not isinstance(image, Image.Image):
                    raise IrisError("入力画像をアップロードしてください。")
                if image.width * image.height > 40_000_000:
                    raise IrisError("入力画像は40メガピクセルまでです。")
                image = ImageOps.exif_transpose(image).convert("RGBA")
                background = Image.new("RGBA", image.size, "white")
                Image.alpha_composite(background, image).convert("RGB").save(directory / "input.png")
                request["image"] = str(directory / "input.png")
            atomic_json(directory / "request.json", request)
            job = {"id": identifier, "directory": str(directory), "status": "running", "error": ""}
            self.jobs[identifier] = job
            thread = threading.Thread(target=self.execute, args=(job, lease), daemon=True)
            thread.start()
        except BaseException:
            lease.release()
            raise
        return identifier

    def execute(self, job, lease):
        from modules_forge import gpu_residency
        from modules_forge.gpu_ownership import release_forge_vram

        directory = Path(job["directory"])
        environment = dict(
            os.environ,
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            PYTHONIOENCODING="utf-8",
            PYTHONDONTWRITEBYTECODE="1",
        )
        status = "error"
        try:
            release_forge_vram()
            self.worker.start(
                python_path(self.root),
                ROOT / "tools" / "iris_worker.py",
                environment,
                {"root": str(self.root), "job_dir": str(directory)},
                directory / "run.log",
            )
            gpu_residency.register("iris", self.worker.close, "iris")
            while True:
                if (directory / "cancel").exists():
                    status = "cancelled"
                    break
                response = self.worker.result()
                if response is not None:
                    if not response["ok"]:
                        raise RuntimeError(response.get("error", "Iris処理に失敗しました。"))
                    status = "complete"
                    break
                time.sleep(0.1)
        except Exception as exc:
            job["error"] = str(exc)
        finally:
            if status != "complete":
                # Never return ownership merely because browser polling ended
                # or cleanup hit a transient Windows sharing/process error.
                while True:
                    try:
                        self.worker.close()
                        break
                    except Exception as exc:
                        job.update(status="stopping", error=f"終了を確認中: {exc}")
                        time.sleep(0.5)
                atomic_json(directory / "result.json", {"status": status, "error": job["error"]})
            lease.release()
            job["status"] = status

    def poll(self, identifier):
        job = self.jobs.get(identifier)
        if job is None:
            return {"status": "idle"}
        directory = Path(job["directory"])
        response = dict(job)
        path = directory / ("result.json" if job["status"] in {"complete", "error", "cancelled"} else "progress.json")
        if path.is_file():
            try:
                response.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return response

    def cancel(self, identifier):
        job = self.jobs.get(identifier)
        if job and job["status"] == "running":
            (Path(job["directory"]) / "cancel").touch()
            return "中断処理中…"
        return "実行中の処理はありません。"


SERVICE = Service()
