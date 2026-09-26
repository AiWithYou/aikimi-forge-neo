"""Nanosaur2 text-to-image workflow for the managed local ComfyUI runtime."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from modules_forge.minimax_h3_runtime import REPOSITORY_ROOT, SERVER_URL

MANIFEST = REPOSITORY_ROOT / "tools/nanosaur2_manifest.json"
OUTPUT = REPOSITORY_ROOT / "outputs/nanosaur2"
LOGS = REPOSITORY_ROOT / "logs/nanosaur2"
NODE_NAME = "nanosaur2_support"
REQUIRED_NODES = {"Nanosaur2Loader", "CLIPTextEncode", "EmptyLatentImage", "KSampler", "VAEDecode", "SaveImage"}
_LOG = logging.getLogger(__name__)


def runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return Path(repository_root).resolve() / "repositories/nanosaur2/ComfyUI"


class Nanosaur2Error(RuntimeError):
    pass


class Nanosaur2Cancelled(Nanosaur2Error):
    pass


def _bridge():
    from modules_forge import minimax_h3_bridge

    return minimax_h3_bridge


def _entries(group: str) -> list[dict[str, Any]]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))[group]


@lru_cache(maxsize=16)
def _file_hash(path: Path, size: int, mtime_ns: int, ctime_ns: int) -> str:
    del size, mtime_ns, ctime_ns
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_ready(path: Path, entry: dict[str, Any]) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    stat = path.stat()
    if stat.st_size != entry["size"]:
        return False
    return _file_hash(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) == entry["sha256"]


def source_ready(runtime_root: Path) -> bool:
    folder = Path(runtime_root) / "custom_nodes" / NODE_NAME
    return all(_file_ready(folder / entry["path"], entry) for entry in _entries("source"))


def model_ready(runtime_root: Path) -> bool:
    folder = Path(runtime_root) / "models"
    return all(_file_ready(folder / entry["path"], entry) for entry in _entries("models"))


@dataclass(frozen=True)
class Nanosaur2Request:
    prompt: str
    negative_prompt: str = "oldest, low quality"
    width: int = 512
    height: int = 512
    steps: int = 50
    cfg: float = 4.0
    seed: int = -1
    guidance: str = "alternate"
    save_candidates: bool = False  # Common output validator only; this model emits one image.

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 20_000:
            raise Nanosaur2Error("プロンプトは1〜20,000文字で入力してください。")
        if not isinstance(self.negative_prompt, str) or len(self.negative_prompt) > 20_000:
            raise Nanosaur2Error("ネガティブプロンプトは20,000文字以内にしてください。")
        for name, value in (("幅", self.width), ("高さ", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 2048 or value % 16:
                raise Nanosaur2Error(f"{name}は256〜2048の16の倍数で指定してください。")
        if self.width * self.height > 1024 * 1536:
            raise Nanosaur2Error("画像は約1.57MP以下にしてください。")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 100:
            raise Nanosaur2Error("Stepsは1〜100で指定してください。")
        if (
            isinstance(self.cfg, bool)
            or not isinstance(self.cfg, (int, float))
            or not math.isfinite(self.cfg)
            or not 1 <= self.cfg <= 12
        ):
            raise Nanosaur2Error("CFGは1〜12で指定してください。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not -1 <= self.seed <= 2**53 - 1:
            raise Nanosaur2Error("Seedは-1〜9007199254740991で指定してください。")
        if self.guidance not in {"alternate", "cfg", "path_drop"}:
            raise Nanosaur2Error("Guidanceはalternate / cfg / path_dropから選んでください。")

    def resolved_seed(self) -> int:
        return secrets.randbelow(2**53) if self.seed == -1 else self.seed


def build_workflow(request: Nanosaur2Request, seed: int) -> dict[str, dict[str, Any]]:
    request.validate()
    if not isinstance(seed, int) or not 0 <= seed <= 2**53 - 1:
        raise Nanosaur2Error("生成Seedが不正です。")

    def node(kind: str, **inputs: Any) -> dict[str, Any]:
        return {"class_type": kind, "inputs": inputs}

    return {
        "loader": node(
            "Nanosaur2Loader",
            unet_name="nanosaur2_diffusion_model.safetensors",
            text_encoder_name="nanosaur2_text_encoder.safetensors",
            vae_name="nanosaur2_vae.safetensors",
            guidance=request.guidance,
        ),
        "positive": node("CLIPTextEncode", clip=["loader", 1], text=request.prompt.strip()),
        "negative": node("CLIPTextEncode", clip=["loader", 1], text=request.negative_prompt.strip()),
        "latent": node("EmptyLatentImage", width=request.width, height=request.height, batch_size=1),
        "sample": node(
            "KSampler",
            model=["loader", 0],
            positive=["positive", 0],
            negative=["negative", 0],
            latent_image=["latent", 0],
            seed=seed,
            steps=request.steps,
            cfg=float(request.cfg),
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        ),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["loader", 2]),
        "save": node("SaveImage", images=["decode", 0], filename_prefix="image/Aikimi_Nanosaur2"),
    }


def check_nodes(client: Any) -> None:
    schemas = client.object_info(REQUIRED_NODES, timeout=12.0)
    missing = REQUIRED_NODES - schemas.keys()
    if missing:
        raise Nanosaur2Error("ComfyUIに必要なノードがありません: " + ", ".join(sorted(missing)))
    inputs = schemas["Nanosaur2Loader"].get("input", {}).get("required", {})
    if not {"unet_name", "text_encoder_name", "vae_name", "guidance"} <= inputs.keys():
        raise Nanosaur2Error("Nanosaur2Loaderの入力仕様が変わっています。")
    for key, filename in (
        ("unet_name", "nanosaur2_diffusion_model.safetensors"),
        ("text_encoder_name", "nanosaur2_text_encoder.safetensors"),
        ("vae_name", "nanosaur2_vae.safetensors"),
    ):
        if filename not in inputs[key][0]:
            raise Nanosaur2Error(f"ComfyUIからモデルが見えません: {filename}")


def ensure_runtime(*, restart: bool = False, selected_root: Path | None = None) -> Any:
    bridge = _bridge()
    root = bridge.resolve_runtime_root(selected_root or runtime_root())
    from tools.setup_nanosaur2 import runtime_ready

    if not runtime_ready(REPOSITORY_ROOT):
        raise Nanosaur2Error("Nanosaur2専用ComfyUIのセットアップが未完了です。")
    if not source_ready(root):
        raise Nanosaur2Error("Nanosaur2ノードが未導入か破損しています。セットアップを実行してください。")
    if not model_ready(root):
        raise Nanosaur2Error("Nanosaur2モデルが未導入か破損しています。セットアップを実行してください。")
    with bridge._RUNTIME_LIFECYCLE_LOCK:
        if restart:
            readiness = bridge._restart_runtime_locked(root, SERVER_URL, LOGS, log_prefix="nanosaur2")
        else:
            readiness = bridge.inspect_readiness(root, SERVER_URL)
            if not readiness.connected:
                readiness = bridge.start_runtime(
                    root, SERVER_URL, LOGS, initial_readiness=readiness, log_prefix="nanosaur2"
                )
        client = bridge.ComfyH3Client(SERVER_URL)
        try:
            check_nodes(client)
        finally:
            client.close()
    return readiness


def save_result(
    source: Path, request: Nanosaur2Request, seed: int, prompt_id: str, readiness: Any, output_root: Path = OUTPUT
) -> dict[str, Any]:
    from PIL import Image

    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_root / f"Nanosaur2_{stamp}_{uuid.uuid4().hex[:10]}"
    stage = Path(tempfile.mkdtemp(prefix=".nanosaur2-", dir=output_root))
    metadata = {
        "model": "well9472/Nanosaur2-670M",
        "model_revision": json.loads(MANIFEST.read_text(encoding="utf-8"))["revision"],
        "prompt": request.prompt.strip(),
        "negative_prompt": request.negative_prompt.strip(),
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg": request.cfg,
        "seed": seed,
        "guidance": request.guidance,
        "sampler": "euler",
        "scheduler": "simple",
        "prompt_id": prompt_id,
        "comfyui_version": readiness.comfy_version,
        "comfyui_revision": readiness.core_revision,
    }
    try:
        destination = stage / "image.png"
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            if image.format != "PNG" or image.size != (request.width, request.height):
                raise Nanosaur2Error("生成PNGの形式・サイズが指定と一致しません。")
            image.verify()
        (stage / "parameters.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "path": str(target / "image.png"),
        "files": [str(target / "image.png"), str(target / "parameters.json")],
        "metadata": metadata,
    }


def run_generation(
    request: Nanosaur2Request, *, output_root: Path = OUTPUT, poll_seconds: float = 2.0
) -> Iterator[dict[str, Any]]:
    request.validate()
    from modules_forge.gpu_ownership import GPUOwnership

    bridge = _bridge()
    ownership = GPUOwnership()
    ownership.engine = "nanosaur2"
    prompt_id = str(uuid.uuid4())
    client = None
    submitted = False
    terminal = False
    try:
        while not ownership.acquire():
            if bridge._is_cancelled_job(prompt_id):
                raise Nanosaur2Cancelled("画像生成を停止しました。")
            yield {"stage": "queued", "message": "GPUの使用終了を待っています。", "prompt_id": prompt_id}
            time.sleep(0.1)
        bridge.release_forge_vram()
        yield {"stage": "runtime", "message": "Nanosaur2実行環境を確認しています。", "prompt_id": prompt_id}
        readiness = ensure_runtime()
        from modules_forge import gpu_residency

        gpu_residency.register("nanosaur2", lambda: bridge._release_retained_runtime(SERVER_URL), "nanosaur2")
        seed = request.resolved_seed()
        graph = build_workflow(request, seed)
        with bridge._RUNTIME_LIFECYCLE_LOCK:
            readiness = ensure_runtime()
            client = bridge.ComfyH3Client(SERVER_URL)
            if bridge._is_cancelled_job(prompt_id):
                raise Nanosaur2Cancelled("画像生成を停止しました。")
            process = bridge._loopback_server_process(SERVER_URL)
            if process is None:
                raise Nanosaur2Error("送信先のComfyUI processを確認できません。")
            bridge.pending_jobs.write(
                {
                    "version": 1,
                    "prompt_id": prompt_id,
                    "runtime_root": os.fspath(runtime_root().resolve()),
                    "server_url": SERVER_URL,
                    "server_process": {"pid": process.pid, "created": process.create_time()},
                    "prepared": {},
                    "state": "submitting",
                    "cancel_ack": False,
                }
            )
            bridge._mark_active_generation(prompt_id)
            with bridge._ACTIVE_GENERATION_LOCK:
                bridge._GPU_OWNERSHIPS[prompt_id] = ownership
            try:
                client.submit(graph, prompt_id)
                bridge.pending_jobs.update(prompt_id, state="submitted")
                submitted = True
            except bridge.H3SubmissionRejected:
                terminal = True
                raise
            except bridge.H3BridgeError:
                _LOG.warning("Nanosaur2 submission response missing; reconciling the same job ID.")
                submitted = True
        started = time.monotonic()
        yield {"stage": "queued", "message": "生成をキューに追加しました。", "prompt_id": prompt_id, "seed": seed}
        failures = 0
        while True:
            try:
                if bridge._is_cancelled_job(prompt_id) and not bridge._cancel_confirmed(prompt_id):
                    client.cancel(prompt_id)
                job = client.job(prompt_id)
            except bridge.H3JobNotFound:
                if bridge._cancel_confirmed(prompt_id):
                    terminal = True
                    raise Nanosaur2Cancelled("画像生成を停止しました。") from None
                raise
            except bridge.H3BridgeError:
                failures += 1
                if failures >= 3:
                    raise
                yield {"stage": "reconnecting", "message": "状態を再取得しています。", "prompt_id": prompt_id}
                time.sleep(poll_seconds)
                continue
            failures = 0
            status = str(job.get("status", "pending")).lower()
            if status in {"success", "completed"}:
                terminal = True
                from modules_forge.minimax_h3_images import extract_image_outputs

                primary, _ = extract_image_outputs(client.history(prompt_id), prompt_id, runtime_root(), request)
                result = save_result(primary[0], request, seed, prompt_id, readiness, output_root)
                yield {
                    "stage": "complete",
                    "message": "PNGと生成条件を保存しました。",
                    "prompt_id": prompt_id,
                    "elapsed": time.monotonic() - started,
                    "seed": seed,
                    **result,
                }
                return
            if status in {"cancelled", "canceled"}:
                terminal = True
                raise Nanosaur2Cancelled("画像生成を停止しました。")
            if status in {"failed", "error"}:
                terminal = True
                raise Nanosaur2Error(bridge._execution_error(job))
            yield {
                "stage": "running" if status in {"running", "in_progress"} else "queued",
                "message": "画像を生成しています。",
                "prompt_id": prompt_id,
                "seed": seed,
                "elapsed": time.monotonic() - started,
            }
            time.sleep(poll_seconds)
    finally:
        try:
            if client is not None and submitted and not terminal:
                try:
                    client.cancel(prompt_id)
                except bridge.H3BridgeError:
                    _LOG.warning("Nanosaur2 cancellation request failed.")
                bridge._schedule_deferred_cleanup(client, prompt_id, {}, runtime_root())
                client = None
            if terminal:
                bridge.pending_jobs.remove(prompt_id)
                bridge._clear_cancelled_job(prompt_id)
        finally:
            if terminal:
                bridge._finish_gpu_generation(prompt_id)
            bridge._clear_active_generation(prompt_id)
            if client is not None:
                client.close()
            with bridge._ACTIVE_GENERATION_LOCK:
                retained = ownership in bridge._GPU_OWNERSHIPS.values()
            if not retained:
                ownership.release()


def cancel_generation(prompt_id: str) -> None:
    _bridge().cancel_generation(prompt_id, SERVER_URL)
