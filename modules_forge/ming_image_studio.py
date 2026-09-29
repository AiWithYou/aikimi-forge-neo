"""Ming Image Design INT8/W4A8: prompts, ComfyUI graph, and lossless PNG output."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

from modules_forge.minimax_h3_runtime import REPOSITORY_ROOT, SERVER_URL

MANIFEST = REPOSITORY_ROOT / "tools/ming_image_manifest.json"
OUTPUT = REPOSITORY_ROOT / "outputs/ming-image"
LOGS = REPOSITORY_ROOT / "logs/ming-image"
MODEL = "ming_image_0.1_design_int8_convrot.safetensors"
ENCODER = "ming_image_0.1_ling_mini_2.0_w4a8.safetensors"
VAE = "ming_image_vae_bf16.safetensors"
RGBA_PREFIXES = (
    "RGBA, 4-channel, transparent background",
    "带透明通道，4通道RGBA图像",
    "透明背景，alpha通道，无底图",
    "抠图素材，背景alpha=0",
    "孤立主体，透明PNG图层",
    "不要白底，不要棋盘格，只要透明通道",
    "isolated subject, alpha matte, no background",
    "cutout PNG, alpha=0 outside the object",
    "transparent canvas, not white, not checkerboard",
    "production RGBA layer for compositing",
)
REQUIRED_NODES = {
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "ModelSamplingFlux",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
}
_LOG = logging.getLogger(__name__)


class MingImageError(RuntimeError):
    pass


class MingImageCancelled(MingImageError):
    pass


def runtime_root(repository_root: Path = REPOSITORY_ROOT) -> Path:
    return Path(repository_root).resolve() / "repositories/ming-image/ComfyUI"


def _bridge():
    from modules_forge import minimax_h3_bridge

    return minimax_h3_bridge


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def _file_hash(path: Path, size: int, modified: int, created: int) -> str:
    del size, modified, created
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def model_ready(root: Path, *, verify_hash: bool = True) -> bool:
    for entry in manifest()["models"]:
        path = root / "models" / entry["path"]
        if path.is_symlink() or not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size != entry["size"]:
            return False
        if verify_hash and _file_hash(path, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != entry["sha256"]:
            return False
    return True


def is_json_prompt(prompt: str) -> bool:
    return prompt.lstrip().startswith(("{", "["))


@dataclass(frozen=True)
class MingImageRequest:
    prompt: str
    text: str = ""
    transparent: bool = False
    width: int = 1024
    height: int = 1024
    steps: int = 12
    seed: int = -1
    parent: str = ""
    save_candidates: bool = False

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 24000:
            raise MingImageError("プロンプトは1〜24,000文字で入力してください。")
        if not isinstance(self.text, str) or len(self.text) > 2000:
            raise MingImageError("画像に載せる文字は2,000文字以内にしてください。")
        if not isinstance(self.transparent, bool):
            raise MingImageError("背景の透過指定が不正です。")
        if is_json_prompt(self.prompt):
            try:
                value = json.loads(self.prompt)
            except json.JSONDecodeError as exc:
                raise MingImageError(f"JSONの{exc.lineno}行目・{exc.colno}文字目を確認してください: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise MingImageError("構造化プロンプトにはJSONオブジェクトを指定してください。")
            if self.text.strip() or self.transparent:
                raise MingImageError("JSONはそのまま送信します。文字・透明背景はJSON内に指定してください。")
        for name, value in (("幅", self.width), ("高さ", self.height)):
            if isinstance(value, bool) or not isinstance(value, int) or not 256 <= value <= 4096 or value % 16:
                raise MingImageError(f"{name}は256〜4096の16の倍数で指定してください。")
        if self.width * self.height > 2048**2 or not 0.25 <= self.width / self.height <= 4:
            raise MingImageError("画像は2048×2048相当以下、縦横比は1:4〜4:1にしてください。")
        if isinstance(self.steps, bool) or not isinstance(self.steps, int) or not 1 <= self.steps <= 50:
            raise MingImageError("Stepsは1〜50の整数で指定してください。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not -1 <= self.seed <= 2**53 - 1:
            raise MingImageError("Seedは-1〜9007199254740991の整数で指定してください。")
        if not isinstance(self.parent, str) or len(self.parent) > 128:
            raise MingImageError("元の生成結果の識別子が不正です。")

    def resolved_seed(self) -> int:
        return secrets.randbelow(2**53) if self.seed == -1 else self.seed

    def effective_prompt(self) -> str:
        self.validate()
        if is_json_prompt(self.prompt):
            return self.prompt
        prompt = self.prompt.strip()
        if self.transparent:
            while True:
                prefix = next((p for p in RGBA_PREFIXES if prompt.startswith(p)), None)
                if prefix is None:
                    break
                prompt = prompt[len(prefix) :].lstrip(" \n\r.,，。")
            prompt = RGBA_PREFIXES[0] + ".\n" + prompt
        lines = list(dict.fromkeys(line.strip() for line in self.text.splitlines() if line.strip()))
        missing = [line for line in lines if line not in self.prompt]
        if missing:
            prompt += "\nRender each of these exact strings once, preserving spelling and language: "
            prompt += ", ".join(json.dumps(line, ensure_ascii=False) for line in missing) + "."
        return prompt


def build_workflow(request: MingImageRequest, seed: int) -> dict[str, dict[str, Any]]:
    request.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**53 - 1:
        raise MingImageError("生成Seedが不正です。")

    def node(kind: str, **inputs: Any) -> dict:
        return {"class_type": kind, "inputs": inputs}

    return {
        "model": node("UNETLoader", unet_name=MODEL, weight_dtype="default"),
        # The pinned ComfyUI detects Ming from weights; its enum has no ming_image value.
        "clip": node("CLIPLoader", clip_name=ENCODER, type="qwen_image", device="default"),
        "vae": node("VAELoader", vae_name=VAE),
        "positive": node("CLIPTextEncode", clip=["clip", 0], text=request.effective_prompt()),
        "negative": node("ConditioningZeroOut", conditioning=["positive", 0]),
        "sampling": node(
            "ModelSamplingFlux",
            model=["model", 0],
            max_shift=1.15,
            base_shift=0.5,
            width=request.width,
            height=request.height,
        ),
        "latent": node("EmptyLatentImage", width=request.width, height=request.height, batch_size=1),
        "sample": node(
            "KSampler",
            model=["sampling", 0],
            positive=["positive", 0],
            negative=["negative", 0],
            latent_image=["latent", 0],
            seed=seed,
            steps=request.steps,
            cfg=1.0,
            sampler_name="euler",
            scheduler="simple",
            denoise=1.0,
        ),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "save": node("SaveImage", images=["decode", 0], filename_prefix="image/Aikimi_Ming"),
    }


def check_nodes(client: Any) -> None:
    schemas = client.object_info(REQUIRED_NODES, timeout=12.0)
    missing = REQUIRED_NODES - schemas.keys()
    if missing:
        raise MingImageError("Ming対応ノードがありません: " + ", ".join(sorted(missing)))
    for kind, field, filename in (
        ("UNETLoader", "unet_name", MODEL),
        ("CLIPLoader", "clip_name", ENCODER),
        ("VAELoader", "vae_name", VAE),
    ):
        choices = schemas[kind].get("input", {}).get("required", {}).get(field, [[]])[0]
        if filename not in choices:
            raise MingImageError(f"ComfyUIからモデルが見えません: {filename}")


def ensure_runtime(*, restart: bool = False) -> Any:
    from tools.setup_ming_image import runtime_ready

    root = runtime_root()
    if not runtime_ready(REPOSITORY_ROOT):
        raise MingImageError("Ming専用環境が未導入です。「実行環境とモデル」から準備してください。")
    if not model_ready(root):
        raise MingImageError("Mingモデルが不足または破損しています。セットアップを確認してください。")
    bridge = _bridge()
    with bridge._RUNTIME_LIFECYCLE_LOCK:
        if restart:
            readiness = bridge._restart_runtime_locked(root, SERVER_URL, LOGS, log_prefix="ming-image")
        else:
            readiness = bridge.inspect_readiness(root, SERVER_URL)
            if not readiness.connected:
                readiness = bridge.start_runtime(
                    root, SERVER_URL, LOGS, initial_readiness=readiness, log_prefix="ming-image"
                )
        client = bridge.ComfyH3Client(SERVER_URL)
        try:
            check_nodes(client)
        finally:
            client.close()
    return readiness


def transparency_stats(image) -> dict[str, Any]:
    """Measure alpha without treating the presence of a channel as a cutout."""
    alpha = image.getchannel("A") if image.mode == "RGBA" else None
    alpha_range = alpha.getextrema() if alpha is not None else (255, 255)
    fraction = sum(alpha.histogram()[:17]) / (image.width * image.height) if alpha is not None else 0.0
    return {
        "image_mode": image.mode,
        "alpha_range": list(alpha_range),
        "has_transparency": alpha_range[0] < 255,
        "near_transparent_fraction": fraction,
        "near_transparent_alpha_threshold": 16,
    }


def save_result(
    source: Path, request: MingImageRequest, seed: int, prompt_id: str, readiness: Any, output_root: Path = OUTPUT
) -> dict[str, Any]:
    from PIL import Image

    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / f"Ming_{datetime.now():%Y%m%d-%H%M%S}_{uuid.uuid4().hex[:10]}"
    stage = Path(tempfile.mkdtemp(prefix=".ming-", dir=output_root))
    metadata = {
        "schema_version": 1,
        "model": "inclusionAI/Ming-Image-0.1-Design",
        "precision": "DiT INT8 / encoder W4A8 / VAE BF16",
        "weights_repository": manifest()["repository"],
        "weights_revision": manifest()["revision"],
        "request": asdict(request),
        "seed": seed,
        "effective_prompt": request.effective_prompt(),
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "prompt_id": prompt_id,
        "comfyui_revision": readiness.core_revision,
        "comfyui_version": readiness.comfy_version,
    }
    try:
        destination = stage / "image.png"
        shutil.copyfile(source, destination)
        with Image.open(destination) as image:
            if image.format != "PNG" or image.size != (request.width, request.height):
                raise MingImageError("生成PNGの形式・寸法が指定と一致しません。")
            image.load()
            metadata.update(transparency_stats(image))
        (stage / "parameters.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (stage / "prompt.txt").write_text(metadata["effective_prompt"], encoding="utf-8")
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "path": str(target / "image.png"),
        "files": [str(target / n) for n in ("image.png", "parameters.json", "prompt.txt")],
        "metadata": metadata,
    }


def run_generation(
    request: MingImageRequest, *, output_root: Path = OUTPUT, poll_seconds: float = 2.0
) -> Iterator[dict[str, Any]]:
    request.validate()
    from modules_forge.gpu_ownership import GPUOwnership

    bridge = _bridge()
    ownership = GPUOwnership()
    ownership.engine = "ming_image"
    prompt_id = str(uuid.uuid4())
    client = None
    submitted = False
    terminal = False
    try:
        while not ownership.acquire():
            if bridge._is_cancelled_job(prompt_id):
                raise MingImageCancelled("画像生成を停止しました。")
            yield {"stage": "queued", "message": "GPUの使用終了を待っています。", "prompt_id": prompt_id}
            time.sleep(0.1)
        bridge.release_forge_vram()
        yield {"stage": "runtime", "message": "Ming Image実行環境を確認しています。", "prompt_id": prompt_id}
        readiness = ensure_runtime()
        from modules_forge import gpu_residency

        gpu_residency.register("ming_image", lambda: bridge._release_retained_runtime(SERVER_URL), "ming_image")
        seed = request.resolved_seed()
        graph = build_workflow(request, seed)
        with bridge._RUNTIME_LIFECYCLE_LOCK:
            readiness = ensure_runtime()
            client = bridge.ComfyH3Client(SERVER_URL)
            if bridge._is_cancelled_job(prompt_id):
                raise MingImageCancelled("画像生成を停止しました。")
            process = bridge._loopback_server_process(SERVER_URL)
            if process is None:
                raise MingImageError("送信先のComfyUI processを確認できません。")
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
                _LOG.warning("MingImage submission response missing; reconciling the same job ID.")
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
                    raise MingImageCancelled("画像生成を停止しました。") from None
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
                raise MingImageCancelled("画像生成を停止しました。")
            if status in {"failed", "error"}:
                terminal = True
                raise MingImageError(bridge._execution_error(job))
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
                    _LOG.warning("MingImage cancellation request failed.")
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
