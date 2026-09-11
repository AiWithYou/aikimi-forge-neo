"""Image-only MiniMax H3 workflows using the existing local runtime boundary.

This is a short-packet adaptation, not a single-frame model: the native H3
conditioning nodes sample five frames, then SaveImage writes PNGs directly.
The audio latent stays in the sampler; no audio decoder or video encoder runs.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_negpip import H3NegPiP

FRAMES = 5
MAX_PIXELS = 2048 * 2048
MAX_PNG_BYTES = 64 * 1024 * 1024
_LOG = logging.getLogger(__name__)


class H3ImageError(RuntimeError):
    """Actionable image-workflow validation or output error."""


class H3ImageCancelled(H3ImageError):
    """Only this image job was cancelled."""


def _bridge():
    # Keep request/graph/output tests independent of Forge and its GPU imports.
    from modules_forge import minimax_h3_bridge

    return minimax_h3_bridge


def integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise H3ImageError(f"{label}は整数で指定してください。")
    if isinstance(value, str):
        if not re.fullmatch(r"-?[0-9]+", value.strip()):
            raise H3ImageError(f"{label}は整数で指定してください。")
    elif not isinstance(value, (int, float)):
        raise H3ImageError(f"{label}は整数で指定してください。")
    try:
        number = int(value)
        if not isinstance(value, str) and number != value:
            raise ValueError("fractional number")
    except (ValueError, TypeError, OverflowError) as exc:
        raise H3ImageError(f"{label}は有限の整数で指定してください。") from exc
    if not minimum <= number <= maximum:
        raise H3ImageError(f"{label}は{minimum}〜{maximum}で指定してください。")
    return number


@dataclass(frozen=True)
class H3ImageRequest:
    prompt: str
    mode: str = "text"
    width: int = 768
    height: int = 768
    steps: int = 20
    seed: int = -1
    scheduler: str = "simple"
    reference_images: tuple[str, ...] = ()
    frame_index: int = 0
    save_candidates: bool = False
    negpip: H3NegPiP = field(default_factory=H3NegPiP)

    @property
    def runtime_acceleration(self) -> H3Acceleration:
        return H3Acceleration(negpip=self.negpip)

    def validate(self) -> None:
        try:
            if not isinstance(self.negpip, H3NegPiP):
                raise ValueError("NegPiP設定の形式が不正です。")
            self.negpip.validate()
        except ValueError as exc:
            raise H3ImageError(str(exc)) from exc
        if self.mode not in {"text", "references"}:
            raise H3ImageError("テキスト生成または参照画像編集を選択してください。")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise H3ImageError("画像のプロンプトを入力してください。")
        if len(self.prompt) > 20_000:
            raise H3ImageError("プロンプトは20,000文字以内にしてください。")
        for label, value in (("幅", self.width), ("高さ", self.height)):
            integer(value, label, 256, 2048)
            if not isinstance(value, int) or value % 32:
                raise H3ImageError(f"{label}は32の倍数で指定してください。")
        if self.width * self.height > MAX_PIXELS:
            raise H3ImageError("解像度は2048×2048画素以下にしてください。")
        integer(self.steps, "Steps", 1, 100)
        integer(self.seed, "Seed", -1, 2**63 - 1)
        integer(self.frame_index, "採用フレーム", 0, FRAMES - 1)
        if not all(isinstance(v, int) for v in (self.steps, self.seed, self.frame_index)):
            raise H3ImageError("Steps・Seed・採用フレームは整数で指定してください。")
        if self.scheduler not in {"simple", "beta", "normal"}:
            raise H3ImageError("Schedulerはsimple / beta / normalから選択してください。")
        if not isinstance(self.save_candidates, bool):
            raise H3ImageError("候補保存の指定が不正です。")
        if not isinstance(self.reference_images, tuple) or any(
            not isinstance(p, str) or not p for p in self.reference_images
        ):
            raise H3ImageError("参照画像の指定が不正です。")
        if self.mode == "text" and self.reference_images:
            raise H3ImageError("参照画像を使う場合は参照画像編集へ切り替えてください。")
        if self.mode == "references" and not 1 <= len(self.reference_images) <= 9:
            raise H3ImageError("参照画像編集では1〜9枚を指定してください。")
        expected = {f"<Picture {i}>" for i in range(1, len(self.reference_images) + 1)}
        supplied = set(re.findall(r"<(?:Picture|Video|Audio)\b[^<>]*>", self.prompt, re.I))
        if supplied - expected:
            raise H3ImageError("存在しない参照タグ、または表記違いがあります。<Picture 1>形式で指定してください。")
        if expected - supplied:
            raise H3ImageError("プロンプトで全参照画像を指定してください: " + ", ".join(sorted(expected - supplied)))

    def resolved_seed(self) -> int:
        return secrets.randbelow(2**53) if self.seed == -1 else self.seed


def model_names() -> dict[str, str]:
    bridge = _bridge()
    return {
        "fl2va": bridge.H3_FL_MODEL,
        "ref2va": bridge.H3_REF_MODEL,
        "clip": bridge.H3_TEXT_ENCODER,
        "vae": bridge.H3_VIDEO_VAE,
        "audio_vae": bridge.H3_AUDIO_VAE,
    }


def build_image_workflow(
    request: H3ImageRequest,
    prepared: Mapping[str, Any],
    seed: int,
    *,
    models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    request.validate()
    seed = integer(seed, "生成Seed", 0, 2**63 - 1)
    names = model_names() if models is None else models

    def node(kind: str, **inputs: Any) -> dict[str, Any]:
        return {"class_type": kind, "inputs": inputs}

    reference_mode = request.mode == "references"
    graph = {
        "model": node("UNETLoader", unet_name=names["ref2va" if reference_mode else "fl2va"], weight_dtype="default"),
        "attention": node("ModelAttentionBackend", model=["model", 0], attention="comfy kitchen attention"),
        "clip": node("CLIPLoader", clip_name=names["clip"], type="minimax", device="default"),
        "vae": node("VAELoader", vae_name=names["vae"]),
        "noise": node("RandomNoise", noise_seed=seed),
        "sampler": node("KSamplerSelect", sampler_name="res_multistep"),
        "sigmas": node("BasicScheduler", model=["model", 0], scheduler=request.scheduler, steps=request.steps, denoise=1.0),
        "guider": node("BasicGuider", model=["attention", 0], conditioning=["condition", 0]),
        "sample": node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0], sampler=["sampler", 0], sigmas=["sigmas", 0], latent_image=["condition", 1]),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "select": node("ImageFromBatch", image=["decode", 0], batch_index=request.frame_index, length=1),
        "save": node("SaveImage", images=["select", 0], filename_prefix="image/Forge_Neo_MiniMax_H3_Image"),
    }
    inputs: dict[str, Any] = {
        "clip": ["clip", 0], "vae": ["vae", 0], "prompt": request.prompt.strip(),
        "width": request.width, "height": request.height, "length": FRAMES,
    }
    if reference_mode:
        images = prepared.get("images", [])
        if len(images) != len(request.reference_images):
            raise H3ImageError("準備済み参照画像の枚数が一致しません。")
        # Older supported native schemas require this input even without audio references.
        graph["audio_vae"] = node("VAELoader", vae_name=names["audio_vae"])
        inputs.update(audio_vae=["audio_vae", 0], ref_image_size="match")
        for index, name in enumerate(images):
            key = f"reference_{index}"
            graph[key] = node("LoadImage", image=name)
            inputs[f"ref_images.ref_image_{index}"] = [key, 0]
    graph["condition"] = node("MiniMaxH3ReferenceToVideo" if reference_mode else "MiniMaxH3ImageToVideo", **inputs)
    if request.save_candidates:
        graph["candidates"] = node("ImageFromBatch", image=["decode", 0], batch_index=0, length=FRAMES)
        graph["save_candidates"] = node("SaveImage", images=["candidates", 0], filename_prefix="image/Forge_Neo_MiniMax_H3_Candidate")
    request.negpip.apply_workflow(graph, guider="guider", condition="condition", scheduler="sigmas", node_id="negpip")
    return graph


def validate_image_runtime(request: H3ImageRequest, readiness: Any, profile: str) -> None:
    bridge = _bridge()
    # Retain all existing runtime identity, model, revision and launch-policy guards.
    bridge.validate_readiness(readiness, profile)
    if getattr(readiness, "acceleration", H3Acceleration()).negpip != request.negpip:
        raise H3ImageError("画像のNegPiP設定と実行環境の確認結果が一致しません。再確認してください。")
    if request.mode == "references" and not readiness.ready_for_ref2va:
        raise H3ImageError("参照画像編集用のRef2VAモデルを導入してください。")
    required = request.width * request.height * FRAMES * 3 * 4 / 1024**3
    required += 4.0 if profile == bridge.RUNTIME_PROFILE_FAST else 2.0
    for label, free in (("空きRAM", readiness.ram_free_gib), ("OS commit余力", readiness.commit_free_gib)):
        if free is None:
            continue
        if not isinstance(free, (int, float)) or not math.isfinite(free) or free < 0:
            raise H3ImageError(f"{label}を確認できません。環境を再確認してください。")
        if free < required:
            raise H3ImageError(f"{label}が不足しています。安全目安{required:.1f}GiBに対して{free:.1f}GiBです。")


def check_image_nodes(client: Any) -> None:
    # The shared readiness probe intentionally fetches only the video node set.
    # Query the two additional native image nodes instead of mutating that set.
    required = {"SaveImage", "ImageFromBatch"}
    schemas = client.object_info(required, timeout=8.0)
    missing = required - schemas.keys()
    if missing:
        raise H3ImageError("ComfyUIに画像用ノードがありません: " + ", ".join(sorted(missing)))
    for name, fields in (("SaveImage", {"images", "filename_prefix"}), ("ImageFromBatch", {"image", "batch_index", "length"})):
        inputs = schemas[name].get("input", {})
        available = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
        if fields - available:
            raise H3ImageError(f"{name}の入力仕様が変わっています。ComfyUIの版を確認してください。")


def extract_image_outputs(
    history: Mapping[str, Any], prompt_id: str, root: Path, request: H3ImageRequest,
) -> tuple[list[Path], list[Path]]:
    """Only accept PNGs emitted by this workflow's known SaveImage nodes."""
    item = history.get(prompt_id)
    if not isinstance(item, dict) or not isinstance(item.get("outputs"), dict):
        raise H3ImageError("完了履歴に画像の出力情報がありません。")
    base = (root / "output").resolve()

    def paths_for(key: str, expected: int) -> list[Path]:
        output = item["outputs"].get(key)
        entries = output.get("images") if isinstance(output, dict) else None
        if not isinstance(entries, list) or len(entries) != expected:
            raise H3ImageError(f"{key}のPNG出力数が不正です（必要数: {expected}）。")
        result = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("type") != "output":
                raise H3ImageError("画像の保存先がComfyUI outputではありません。")
            filename, folder = entry.get("filename"), entry.get("subfolder", "")
            if not isinstance(filename, str) or not isinstance(folder, str):
                raise H3ImageError("出力画像のパスが不正です。")
            relative = PurePosixPath(folder.replace("\\", "/")) / filename.replace("\\", "/")
            if relative.is_absolute() or ".." in relative.parts or ":" in str(relative):
                raise H3ImageError("出力フォルダー外の画像パスを拒否しました。")
            if "/" in filename or "\\" in filename or relative.suffix.lower() != ".png":
                raise H3ImageError("PNG以外、または不正なファイル名です。")
            path = base.joinpath(*relative.parts).resolve()
            if not path.is_relative_to(base) or not path.is_file():
                raise H3ImageError("出力画像が存在しないか、output外を指しています。")
            if not 0 < path.stat().st_size <= MAX_PNG_BYTES:
                raise H3ImageError("PNGファイルのサイズが不正です。")
            result.append(path)
        if len(set(result)) != expected:
            raise H3ImageError("出力画像のファイル名が重複しています。")
        return result

    primary = paths_for("save", 1)
    candidates = paths_for("save_candidates", FRAMES) if request.save_candidates else []
    return primary, candidates


def save_image_result(
    primary: list[Path], candidates: list[Path], output_directory: Path,
    request: H3ImageRequest, seed: int, prompt_id: str, readiness: Any,
    *, models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Publish the entire PNG/JSON result directory only after every file verifies."""
    from PIL import Image

    if len(primary) != 1 or len(candidates) != (FRAMES if request.save_candidates else 0):
        raise H3ImageError("保存する画像の枚数が不正です。")
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = output_directory / f"H3_Image_{stamp}_{uuid.uuid4().hex[:12]}"
    names = model_names() if models is None else models
    metadata = {
        "schema_version": 1, "output_kind": "image", "model_family": "MiniMax H3",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt_id": prompt_id, "prompt": request.prompt.strip(), "mode": request.mode,
        "width": request.width, "height": request.height, "steps": request.steps,
        "seed": seed, "sampler": "res_multistep", "scheduler": request.scheduler,
        "internal_frames": FRAMES, "selected_frame": request.frame_index,
        "save_candidates": request.save_candidates, "reference_count": len(request.reference_images),
        "models": dict(names), "diffusion_model": names["ref2va" if request.mode == "references" else "fl2va"],
        "attention": "comfy kitchen attention",
        "negpip": request.negpip.to_dict(),
        "comfyui_revision": readiness.core_revision, "comfyui_version": readiness.comfy_version,
        "comfy_kitchen_version": readiness.package_versions.get("comfy-kitchen"),
        "runtime_profile": readiness.runtime_profile,
    }
    filenames = ["image.png"] + [f"candidate_{index:02d}.png" for index in range(len(candidates))]
    stage = Path(tempfile.mkdtemp(prefix=".h3-image-", dir=output_directory))
    try:
        for source, filename in zip(primary + candidates, filenames, strict=True):
            destination = stage / filename
            shutil.copyfile(source, destination)
            with Image.open(destination) as image:
                if image.format != "PNG" or image.size != (request.width, request.height):
                    raise H3ImageError("生成PNGの形式または解像度が指定と一致しません。")
                image.verify()
        (stage / "parameters.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {
        "path": str(target / "image.png"),
        "candidates": [str(target / name) for name in filenames[1:]],
        "files": [str(target / name) for name in [*filenames, "parameters.json"]],
        "metadata": metadata,
    }


def run_image_generation(
    request: H3ImageRequest, runtime_root: Path, server_url: str,
    log_directory: Path, output_directory: Path, runtime_profile: str = "fast",
    *, poll_seconds: float = 2.0,
) -> Iterator[dict[str, Any]]:
    request.validate()
    bridge = _bridge()
    root = bridge.resolve_runtime_root(runtime_root)
    url = bridge.normalize_loopback_url(server_url)
    native_request = bridge.H3Request(
        mode=request.mode, prompt=request.prompt, steps=request.steps, seed=request.seed,
        scheduler=request.scheduler, reference_images=request.reference_images,
        acceleration=request.runtime_acceleration,
    )
    yield {"stage": "runtime", "message": "画像生成用のH3環境を確認しています。", "prompt_id": ""}
    readiness = bridge.ensure_ready(root, url, log_directory, runtime_profile=runtime_profile, acceleration=request.runtime_acceleration)
    validate_image_runtime(request, readiness, runtime_profile)
    bridge.cleanup_stale_prepared_media(root)
    prepared = bridge.prepare_media(native_request, root)
    client, prompt_id, terminal, deferred = None, "", False, False
    try:
        seed = request.resolved_seed()
        graph = build_image_workflow(request, prepared, seed)
        with bridge._RUNTIME_LIFECYCLE_LOCK:
            readiness = bridge.ensure_ready(root, url, log_directory, runtime_profile=runtime_profile, acceleration=request.runtime_acceleration)
            validate_image_runtime(request, readiness, runtime_profile)
            client = bridge.ComfyH3Client(url)
            check_image_nodes(client)
            prompt_id = client.submit(graph)
            bridge._mark_active_generation(prompt_id)
        started, failures = time.monotonic(), 0
        yield {"stage": "queued", "message": "画像生成をキューに追加しました。", "prompt_id": prompt_id, "seed": seed}
        while True:
            try:
                job = client.job(prompt_id)
            except bridge.H3JobNotFound:
                if bridge._is_cancelled_job(prompt_id):
                    terminal = True
                    raise H3ImageCancelled("画像生成を停止しました。") from None
                raise
            except bridge.H3BridgeError:
                failures += 1
                if failures >= 3:
                    raise
                yield {"stage": "reconnecting", "message": f"状態の再取得中です（{failures}/2）。", "prompt_id": prompt_id}
                time.sleep(max(0.05, poll_seconds))
                continue
            failures = 0
            status = str(job.get("status", "pending")).lower()
            elapsed = time.monotonic() - started
            if status in {"success", "completed"}:
                terminal = True
                for attempt in range(3):
                    try:
                        primary, candidates = extract_image_outputs(client.history(prompt_id), prompt_id, root, request)
                        break
                    except (bridge.H3BridgeError, H3ImageError):
                        if attempt == 2:
                            raise
                        time.sleep(max(0.05, poll_seconds))
                result = save_image_result(primary, candidates, output_directory, request, seed, prompt_id, readiness)
                yield {"stage": "complete", "message": "PNGと生成条件を保存しました。", "prompt_id": prompt_id, "seed": seed, "elapsed": elapsed, **result}
                return
            if status in {"cancelled", "canceled"}:
                terminal = True
                raise H3ImageCancelled("画像生成を停止しました。")
            if status in {"failed", "error"}:
                terminal = True
                raise H3ImageError(bridge._execution_error(job))
            running = status in {"running", "in_progress"}
            yield {
                "stage": "running" if running else "queued", "prompt_id": prompt_id,
                "seed": seed, "elapsed": elapsed,
                "message": "画像を生成しています。" if running else "キューで待機しています。",
            }
            time.sleep(max(0.05, poll_seconds))
    finally:
        try:
            if prompt_id and not terminal and client is not None:
                try:
                    client.cancel(prompt_id)
                except bridge.H3BridgeError:
                    _LOG.warning("H3 image cancellation failed; deferring input cleanup.")
                bridge._schedule_deferred_cleanup(client, prompt_id, prepared, root)
                deferred = True
            if not prompt_id or terminal:
                bridge.cleanup_prepared_media(prepared, root)
            if prompt_id and terminal:
                bridge._clear_cancelled_job(prompt_id)
        finally:
            if prompt_id:
                bridge._clear_active_generation(prompt_id)
            if client is not None and not deferred:
                try:
                    client.close()
                except Exception as exc:
                    _LOG.warning("H3 image client close failed (%s).", type(exc).__name__)
