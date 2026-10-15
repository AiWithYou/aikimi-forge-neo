"""FLUX 3 jobs keep the shared GPU lease until their worker has finished."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .core import ROOT, RUNTIME, Flux3Error, atomic_json, model_ready, python_path, validate_request


class Service:
    def __init__(self, root=RUNTIME, *, outputs=None):
        from modules_forge.resident_worker import ResidentWorker

        self.root = Path(root)
        self.outputs = Path(outputs) if outputs is not None else ROOT / "outputs" / "flux3-klein"
        self.worker = ResidentWorker("flux3-klein", ROOT / "cache" / "flux3-klein-worker")
        self.jobs = {}
        self._guard = threading.RLock()

    def start(self, value):
        from modules_forge.gpu_ownership import GPUOwnership

        request = validate_request(value)
        images = request.get("images", [])
        for image in images:
            if not isinstance(image, Image.Image):
                raise Flux3Error("参照画像をアップロードしてください。")
            if image.width * image.height > 40_000_000:
                raise Flux3Error("参照画像は1枚40メガピクセルまでです。")
        if not model_ready(self.root, request):
            raise Flux3Error("選択したモデルをセットアップしてください。")
        lease = GPUOwnership()
        lease.engine = "flux3-klein"
        if not lease.acquire(False):
            raise Flux3Error("GPUを使用中です。処理の終了後に実行してください。")
        identifier = uuid.uuid4().hex
        directory = self.outputs / identifier
        try:
            directory.mkdir(parents=True)
            paths, hashes = [], []
            for index, image in enumerate(images):
                image = ImageOps.exif_transpose(image).convert("RGBA")
                path = directory / f"reference-{index:02d}.png"
                image.save(path)
                paths.append(str(path))
                hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            request.update(images=paths, reference_sha256=hashes)
            atomic_json(directory / "request.json", request)
            job = {"id": identifier, "directory": str(directory), "status": "running", "error": ""}
            self.jobs[identifier] = job
            threading.Thread(target=self.execute, args=(job, lease), daemon=True).start()
        except BaseException:
            self.jobs.pop(identifier, None)
            lease.release()
            raise
        return identifier

    @staticmethod
    def validate_result(directory):
        directory = Path(directory).resolve()
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        path = Path(result["image_path"]).resolve()
        if path != directory / "output.png" or not path.is_file():
            raise Flux3Error("生成結果の出力パスが一致しません。")
        with Image.open(path) as image:
            image.load()
            if image.format != "PNG":
                raise Flux3Error("生成結果がPNGではありません。")
            if image.size != (result["width"], result["height"]):
                raise Flux3Error("生成結果のサイズが記録と一致しません。")
            embedded = image.info.get("flux3_klein")
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        fields = ("model", "precision", "steps", "cfg", "seed", "boxes")
        if (
            any(result.get(key) != request.get(key) for key in fields)
            or result.get("input_prompt") != request["prompt"]
        ):
            raise Flux3Error("生成条件が元の要求と一致しません。")
        if result.get("reference_hashes") != request.get("reference_sha256", []):
            raise Flux3Error("参照画像の生成条件が元の要求と一致しません。")
        from .conditioning import prepare_images

        references = []
        for value in request.get("images", []):
            source = Path(value).resolve()
            if source.parent != directory:
                raise Flux3Error("参照画像の保存場所が一致しません。")
            with Image.open(source) as reference:
                references.append(reference.copy())
        prepared = prepare_images(references, request.get("width"), request.get("height"))
        expected_size = (prepared.width, prepared.height)
        if prepared.crop is not None:
            left, top, right, bottom = prepared.crop
            expected_size = (right - left, bottom - top)
        if (result["width"], result["height"]) != expected_size:
            raise Flux3Error("出力サイズが元の生成条件と一致しません。")
        if not embedded:
            raise Flux3Error("PNG内の生成条件がありません。")
        metadata = json.loads(embedded)
        if any(result.get(key) != value for key, value in metadata.items()) or any(
            key not in metadata for key in fields
        ):
            raise Flux3Error("PNG内の生成条件が保存記録と一致しません。")
        return result

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
                ROOT / "tools" / "flux3_klein_worker.py",
                environment,
                {"root": str(self.root), "job_dir": str(directory)},
                directory / "run.log",
            )
            gpu_residency.register("flux3-klein", self.worker.close, "flux3-klein")
            while True:
                if (directory / "cancel").exists():
                    status = "cancelled"
                    break
                response = self.worker.result()
                if response is not None:
                    if not response["ok"]:
                        raise RuntimeError(response.get("error", "FLUX 3の生成に失敗しました。"))
                    self.validate_result(directory)
                    with self._guard:
                        status = "cancelled" if (directory / "cancel").exists() else "complete"
                        job["_finishing"] = True
                    break
                time.sleep(0.1)
        except Exception as exc:
            job["error"] = str(exc)
        finally:
            with self._guard:
                job["_finishing"] = True
            if status != "complete":
                while True:
                    try:
                        self.worker.close()
                        break
                    except Exception as exc:
                        job.update(status="stopping", error=f"終了を確認中: {exc}")
                        time.sleep(0.5)
            try:
                if status != "complete":
                    atomic_json(directory / "result.json", {"status": status, "error": job["error"]})
            finally:
                try:
                    lease.release()
                finally:
                    with self._guard:
                        job["status"] = status

    def poll(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job is None:
                return {"status": "idle"}
            result = dict(job)
        result.pop("_finishing", None)
        status = result["status"]
        directory = Path(result["directory"])
        filename = "result.json" if status in {"complete", "error", "cancelled"} else "progress.json"
        try:
            result.update(json.loads((directory / filename).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        result["status"] = status
        if status == "complete":
            result["files"] = [str(directory / "output.png"), str(directory / "result.json")]
        return result

    def cancel(self, identifier):
        with self._guard:
            job = self.jobs.get(identifier)
            if job and job["status"] in {"running", "stopping"} and not job.get("_finishing"):
                (Path(job["directory"]) / "cancel").touch()
                return "中断処理中…"
        return "実行中の処理はありません。"


SERVICE = Service()
