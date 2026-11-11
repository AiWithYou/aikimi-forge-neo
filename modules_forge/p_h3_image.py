"""P-H3 Image request and native ComfyUI still-image graph."""

from __future__ import annotations

import json
import logging
import math
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from modules_forge import local_assets

SIGMAS = "1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0"
MODEL = "p-h3-image_w4a8.safetensors"
TEXT_ENCODER = "p-h3-image_text_encoder_w4a8.safetensors"
VAE = "p-h3-image_vae_int8_convrot.safetensors"
UPSAMPLER = "p-h3-image_prompt_upsampler_int8_convrot.safetensors"
OFFICIAL_MODEL = "minimax_h3_image_pruned_int8_convrot.safetensors"
OFFICIAL_TEXT_ENCODER = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
OFFICIAL_VAE = "minimax_h3_image_vae_fp16.safetensors"
_LOG = logging.getLogger(__name__)


class PH3Error(RuntimeError):
    """Invalid inputs, runtime state or image output."""


class PH3Cancelled(PH3Error):
    """The current P-H3 job was cancelled."""


def _bridge():
    from modules_forge import minimax_h3_bridge

    return minimax_h3_bridge


def compatible_checkpoint(path: Path) -> bool:
    """Check P-H3 structure; no header probe can guarantee model quality."""
    try:
        source = local_assets.local_path(str(path), suffixes=(".safetensors",))
        header = local_assets.read_header(source)
        fixed = {
            "audio_patch_proj.weight": [5376, 32],
            "video_patch_proj.weight": [5376, 96],
            "adaln_t_table": [4097, 16],
            "blocks.0.adaln_proj.linear.weight": [96768, 16],
            "blocks.0.attn.q_norm.weight": [128],
        }
        if any(header.get(key, {}).get("shape") != shape for key, shape in fixed.items()):
            return False
        for layer, output in (("blocks.0.attn.qkv_proj", 21504), ("blocks.0.mlp.fc1", 28672)):
            shape = header.get(layer + ".weight", {}).get("shape")
            if shape not in ([output, 5376], [output, 2688]):
                return False
            if shape[1] == 2688 and layer + ".comfy_quant" not in header:
                return False
        return True
    except (OSError, ValueError, TypeError, AttributeError):
        return False


@dataclass(frozen=True)
class PH3Request:
    prompt: str
    mode: str = "text"
    target_image: str = ""
    reference_images: tuple[str, ...] = ()
    width: int = 2048
    height: int = 2048
    seed: int = -1
    use_upsampler: bool = True
    checkpoint: str = ""
    lora: str = ""
    lora_strength: float = 1.0
    save_candidates: bool = False  # Shared output-path validator expects this field.
    official: bool = False

    @property
    def family(self) -> str:
        return "MiniMax-H3-Image" if self.official else "P-H3 Image"

    @property
    def model_names(self) -> dict[str, str]:
        if self.official:
            return {"model": OFFICIAL_MODEL, "text_encoder": OFFICIAL_TEXT_ENCODER, "vae": OFFICIAL_VAE}
        return {"model": MODEL, "text_encoder": TEXT_ENCODER, "vae": VAE, "upsampler": UPSAMPLER}

    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 20_000:
            raise PH3Error("プロンプトは1〜20,000文字で入力してください。")
        if self.mode not in {"text", "edit"}:
            raise PH3Error("画像生成または画像編集を選択してください。")
        if not isinstance(self.target_image, str):
            raise PH3Error("編集する画像を選択してください。")
        if not isinstance(self.reference_images, tuple) or any(
            not isinstance(value, str) or not value for value in self.reference_images
        ):
            raise PH3Error("参照画像の指定が不正です。")
        max_references = 15 if self.official else 7
        if len(self.reference_images) > max_references:
            raise PH3Error(f"追加の参照画像は{max_references}枚までです。")
        if self.mode == "text" and (self.target_image or self.reference_images):
            raise PH3Error("画像を使う場合は画像編集を選択してください。")
        if self.mode == "edit" and not self.target_image:
            raise PH3Error("編集する画像を1枚指定してください。")
        for label, number in (("幅", self.width), ("高さ", self.height)):
            if isinstance(number, bool) or not isinstance(number, int) or not 32 <= number <= 16384 or number % 32:
                raise PH3Error(f"{label}は32〜16384の32の倍数で指定してください。")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not -1 <= self.seed <= 2**63 - 1:
            raise PH3Error("Seedは-1以上の整数で指定してください。")
        if not isinstance(self.use_upsampler, bool) or self.save_candidates is not False:
            raise PH3Error("画像生成の設定が不正です。")
        if self.official:
            if self.width * self.height > 20 * 1024**2:
                raise PH3Error("MiniMax-H3-Imageは約20 MPまでです。")
            if self.use_upsampler or self.lora or self.checkpoint:
                raise PH3Error("MiniMax-H3-Imageでは固定の本体と入力プロンプトを使用します。")
        if not isinstance(self.checkpoint, str):
            raise PH3Error("チェックポイントの指定が不正です。")
        if self.checkpoint and not Path(self.checkpoint).is_absolute():
            raise PH3Error("チェックポイントはこのPCの絶対パスで指定してください。")
        if not isinstance(self.lora, str):
            raise PH3Error("LoRAの指定が不正です。")
        if self.lora:
            name = PurePosixPath(self.lora.replace("\\", "/"))
            if name.is_absolute() or ".." in name.parts or ":" in str(name) or name.suffix.lower() != ".safetensors":
                raise PH3Error("LoRAは共有モデル保存先のlorasフォルダーから選択してください。")
        if (isinstance(self.lora_strength, bool) or not isinstance(self.lora_strength, (int, float))
                or not math.isfinite(self.lora_strength) or not -2 <= self.lora_strength <= 2):
            raise PH3Error("LoRAの強度は−2〜2で指定してください。")

    @property
    def images(self) -> tuple[str, ...]:
        return (self.target_image, *self.reference_images) if self.mode == "edit" else ()

    def resolved_seed(self) -> int:
        return secrets.randbelow(2**53) if self.seed == -1 else self.seed


def _ratio(width: int, height: int) -> str:
    value = width / height
    options = ((1, 1), (4, 3), (3, 4), (3, 2), (2, 3), (16, 9), (9, 16))
    w, h = min(options, key=lambda pair: abs(math.log(value / (pair[0] / pair[1]))))
    if abs(value / (w / h) - 1) < 0.02:
        return f"{w}:{h}"
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def build_workflow(
    request: PH3Request, prepared: Mapping[str, Any], seed: int, *, models: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    request.validate()
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise PH3Error("生成Seedが不正です。")
    names = request.model_names
    if models:
        names.update(models)
    if request.checkpoint:
        names["model"] = Path(request.checkpoint).name
    images = prepared.get("images", [])
    if not isinstance(images, list) or len(images) != len(request.images) or any(
        not isinstance(image, str) or not image for image in images
    ):
        raise PH3Error("準備済み画像の数または順序が一致しません。")

    def node(kind: str, **inputs: Any) -> dict[str, Any]:
        return {"class_type": kind, "inputs": inputs}

    if request.official:
        condition = {"clip": ["clip", 0], "prompt": request.prompt.strip(),
                     "width": request.width, "height": request.height}
        graph = {
            "model": node("UNETLoader", unet_name=names["model"], weight_dtype="default"),
            "clip": node("CLIPLoader", clip_name=names["text_encoder"], type="minimax", device="default"),
            "vae": node("VAELoader", vae_name=names["vae"]),
            "negative": node("ConditioningZeroOut", conditioning=["condition", 0]),
            "sample": node("KSampler", model=["model", 0], positive=["condition", 0],
                           negative=["negative", 0], latent_image=["condition", 1], seed=seed,
                           steps=6, cfg=1.0, sampler_name="euler", scheduler="simple", denoise=1.0),
            "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
            "save": node("SaveImage", images=["decode", 0], filename_prefix="image/MiniMax_H3_Image"),
        }
        if images:
            condition["vae"] = ["vae", 0]
            for index, image in enumerate(images, 1):
                key = f"image_{index}"
                graph[key] = node("LoadImage", image=image)
                condition[f"images.{key}"] = [key, 0]
        graph["condition"] = node("MiniMaxH3ImageConditioning", **condition)
        return graph

    graph = {
        "model": node("UNETLoader", unet_name=names["model"], weight_dtype="default"),
        "shift": node("MiniMaxH3SigmaShift", model=["model", 0], shift_video=1.0, shift_audio=1.0),
        "clip": node("CLIPLoader", clip_name=names["text_encoder"], type="minimax", device="default"),
        "vae": node("VAELoader", vae_name=names["vae"]),
        "noise": node("RandomNoise", noise_seed=seed),
        "sigmas": node("ManualSigmas", sigmas=SIGMAS),
        "sampler": node("KSamplerSelect", sampler_name="euler"),
        "guider": node("BasicGuider", model=["shift", 0], conditioning=["condition", 0]),
        "sample": node("SamplerCustomAdvanced", noise=["noise", 0], guider=["guider", 0],
                       sampler=["sampler", 0], sigmas=["sigmas", 0], latent_image=["condition", 1]),
        "decode": node("VAEDecode", samples=["sample", 0], vae=["vae", 0]),
        "save": node("SaveImage", images=["decode", 0], filename_prefix="image/Aikimi_P_H3"),
    }
    if request.lora and request.lora_strength:
        graph["lora"] = node("LoraLoaderBypassModelOnly", model=["model", 0], lora_name=request.lora,
                             strength_model=request.lora_strength)
        graph["shift"]["inputs"]["model"] = ["lora", 0]
    condition = {"clip": ["clip", 0], "prompt": request.prompt.strip(), "width": request.width, "height": request.height}
    if images:
        condition["vae"] = ["vae", 0]
        for index, image in enumerate(images, 1):
            graph[f"image_{index}"] = node("LoadImage", image=image)
            condition[f"images.image_{index}"] = [f"image_{index}", 0]
        graph["edit_target"] = node("ImageScaleToTotalPixels", image=["image_1", 0],
                                    upscale_method="lanczos", megapixels=4.0, resolution_steps=32)
        condition["images.image_1"] = ["edit_target", 0]
    if request.use_upsampler:
        graph["upsampler_clip"] = node("CLIPLoader", clip_name=names["upsampler"], type="minimax", device="default")
        if images:
            image_blocks = "".join(
                f"<Picture {index}>: <|vision_start|><|image_pad|><|vision_end|>"
                for index in range(1, len(images) + 1)
            )
            prompt = f"<|im_start|>user\n{image_blocks}Edit <Picture 1>. {request.prompt.strip()}<|im_end|>\n<|im_start|>assistant\n<think>\n"
            graph["upsampler_target"] = node("ImageScaleToTotalPixels", image=["image_1", 0],
                                              upscale_method="lanczos", megapixels=1.0, resolution_steps=1)
            image_source = ["upsampler_target", 0]
            if len(images) > 1:
                graph["upsampler_size"] = node("GetImageSize", image=image_source)
                batch = {"images.image0": image_source}
                for index in range(2, len(images) + 1):
                    key = f"upsampler_ref_{index}"
                    graph[key] = node("ResizeAndPadImage", image=[f"image_{index}", 0],
                                      target_width=["upsampler_size", 0], target_height=["upsampler_size", 1],
                                      padding_color="white", interpolation="lanczos")
                    batch[f"images.image{index - 1}"] = [key, 0]
                graph["upsampler_batch"] = node("BatchImagesNode", **batch)
                image_source = ["upsampler_batch", 0]
            upsampler_inputs = {"image": image_source}
        else:
            prompt = f"Aspect ratio: {_ratio(request.width, request.height)}\nRequest: {request.prompt.strip()}"
            upsampler_inputs = {}
        graph["upsampler"] = node("TextGenerate", clip=["upsampler_clip", 0], prompt=prompt,
                                  max_length=3072 if images else 2048, sampling_mode="off",
                                  thinking=not bool(images), use_default_template=True, mtp="auto",
                                  **upsampler_inputs)
        graph["upsampler_preview"] = node("PreviewAny", source=["upsampler", 0])
        condition["prompt"] = ["upsampler", 0]
    graph["condition"] = node("PH3ImageConditioning", **condition)
    return graph


def extract_upsampled_prompt(history: Mapping[str, Any], prompt_id: str, *, enabled: bool) -> str | None:
    if not enabled:
        return None
    try:
        result = history[prompt_id]["outputs"]["upsampler_preview"]["text"]
        if not isinstance(result, list) or len(result) != 1 or not isinstance(result[0], str):
            raise ValueError("invalid upsampled prompt")
        if not result[0].strip() or len(result[0]) > 50_000:
            raise ValueError("empty or oversized upsampled prompt")
        return result[0]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise PH3Error("補助後のプロンプトを確認できません。") from exc


def maximum_png_bytes(width: int, height: int) -> int:
    """Allow uncompressed RGBA pixels, PNG overhead and a metadata allowance."""
    return width * height * 5 + 16 * 1024 * 1024


def extract_output(history: Mapping[str, Any], prompt_id: str, root: Path, request: PH3Request) -> Path:
    try:
        entries = history[prompt_id]["outputs"]["save"]["images"]
        if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
            raise ValueError("unexpected image count")
        item = entries[0]
        filename, folder = item.get("filename"), item.get("subfolder", "")
        if item.get("type") != "output" or not isinstance(filename, str) or not isinstance(folder, str):
            raise ValueError("invalid output")
        relative = PurePosixPath(folder.replace("\\", "/")) / filename.replace("\\", "/")
        if (relative.is_absolute() or ".." in relative.parts or ":" in str(relative)
                or "/" in filename or "\\" in filename or relative.suffix.lower() != ".png"):
            raise ValueError("unsafe output path")
        base = (root / "output").resolve()
        result = base.joinpath(*relative.parts).resolve()
        if not result.is_relative_to(base) or not result.is_file():
            raise ValueError("missing output")
        if not 0 < result.stat().st_size <= maximum_png_bytes(request.width, request.height):
            raise ValueError("unexpected PNG size")
        return result
    except (KeyError, TypeError, ValueError, OSError) as exc:
        raise PH3Error("生成PNGの保存先またはサイズを確認できません。") from exc


def verify_png(path: Path, width: int, height: int) -> None:
    """Isolate Pillow's pixel limit so large output does not alter global settings."""
    program = (
        "import sys; from PIL import Image; "
        "Image.MAX_IMAGE_PIXELS = 16384 * 16384; "
        "im = Image.open(sys.argv[1]); "
        "assert im.format == 'PNG' and im.size == (int(sys.argv[2]), int(sys.argv[3])); "
        "im.verify()"
    )
    try:
        subprocess.run(  # noqa: S603 -- fixed interpreter/program and separate arguments.
            [sys.executable, "-I", "-c", program, str(path), str(width), str(height)],
            check=True, capture_output=True, timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise PH3Error("生成PNGの形式または解像度が指定と一致しません。") from exc


def _checkpoint_path(request: PH3Request, runtime_root: Path) -> Path:
    if request.checkpoint:
        return Path(request.checkpoint).resolve()
    from modules_forge.minimax_h3_runtime import model_root

    return model_root(runtime_root) / "diffusion_models" / request.model_names["model"]


def check_nodes(client: Any, request: PH3Request) -> None:
    """Check the nodes and the exact model choices used by this request."""
    graph = build_workflow(request, {"images": [f"image-{i}.png" for i in range(len(request.images))]}, 0)
    required = {item["class_type"] for item in graph.values()}
    schemas = client.object_info(required, timeout=12.0)
    missing = required - schemas.keys()
    if missing:
        raise PH3Error(f"{request.family}用ノードがありません。共通実行環境を更新してください: " + ", ".join(sorted(missing)))
    names = request.model_names
    checks = [("UNETLoader", "unet_name", Path(request.checkpoint).name if request.checkpoint else names["model"]),
              ("CLIPLoader", "clip_name", names["text_encoder"]), ("VAELoader", "vae_name", names["vae"])]
    if request.use_upsampler:
        checks.append(("CLIPLoader", "clip_name", UPSAMPLER))
    if request.lora and request.lora_strength:
        checks.append(("LoraLoaderBypassModelOnly", "lora_name", request.lora))
    for kind, field, filename in checks:
        try:
            choices = schemas[kind]["input"]["required"][field][0]
            if filename not in choices:
                raise ValueError("model not registered")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise PH3Error(f"ComfyUIからモデルを確認できません: {filename}") from exc
    inputs = schemas[graph["condition"]["class_type"]].get("input", {})
    fields = set(inputs.get("required", {})) | set(inputs.get("optional", {}))
    if not {"clip", "prompt", "width", "height"} <= fields or (request.images and "vae" not in fields):
        raise PH3Error(f"{request.family}用ノードの入力仕様が一致しません。")


def ensure_runtime(
    request: PH3Request, runtime_root: Path, server_url: str, log_directory: Path, runtime_profile: str = "fast",
) -> Any:
    """Use the H3 process manager without requiring video-model files."""
    from modules_forge.minimax_h3_acceleration import H3Acceleration
    from modules_forge.minimax_h3_runtime import model_root

    bridge = _bridge()
    root = bridge.resolve_runtime_root(runtime_root)
    url = bridge.normalize_loopback_url(server_url)
    selected = _checkpoint_path(request, root)
    if request.official and not selected.is_file():
        raise PH3Error(f"MiniMax-H3-Imageのモデルが不足しています: {selected.name}")
    if not request.official and not compatible_checkpoint(selected):
        raise PH3Error("P-H3互換構造の本体がありません。モデルを準備するか、対応本体を選択してください。")
    model_directory = model_root(root)
    names = request.model_names
    needed = [("text_encoders", names["text_encoder"]), ("vae", names["vae"])]
    if request.use_upsampler:
        needed.append(("text_encoders", UPSAMPLER))
    for folder, name in needed:
        if not (model_directory / folder / name).is_file():
            raise PH3Error(f"{request.family}のモデルが不足しています: {name}")
    acceleration = H3Acceleration(compiler_mode="off", model_path=request.checkpoint)
    with bridge._RUNTIME_LIFECYCLE_LOCK:
        if bridge._RUNTIME_SETUP_ACTIVE:
            raise PH3Error("実行環境の準備が完了するまでお待ちください。")
        readiness = bridge.inspect_readiness(root, url, acceleration=acceleration)
        if readiness.connected:
            readiness = bridge.ensure_checkpoint_registration(
                readiness, root, url, log_directory, acceleration, runtime_profile, log_prefix="p-h3"
            )
            compiler_off = bridge._cli_option_values(readiness.runtime_args, "--disable-comfy-compiler") == ("",)
            # Flag options return an empty value; duplicate flags are not accepted.
            if readiness.runtime_profile != runtime_profile or not compiler_off:
                readiness = bridge._restart_runtime_locked(
                    root, url, log_directory, runtime_profile=runtime_profile,
                    acceleration=acceleration, log_prefix="p-h3",
                )
        else:
            readiness = bridge.start_runtime(
                root, url, log_directory, runtime_profile=runtime_profile,
                initial_readiness=readiness, acceleration=acceleration, log_prefix="p-h3",
            )
        bridge.require_checkpoint_registration(readiness, root, acceleration)
        if not readiness.connected:
            raise PH3Error(readiness.error or f"{request.family}の実行環境に接続できません。")
        if request.official and bridge._version_tuple(readiness.comfy_version or "") < (0, 45, 0):
            raise PH3Error("MiniMax-H3-Imageのセットアップで共通実行環境を更新してください。")
        client = bridge.ComfyH3Client(url)
        try:
            check_nodes(client, request)
        finally:
            client.close()
        return readiness


def prepare_images(request: PH3Request, runtime_root: Path) -> dict[str, Any]:
    bridge = _bridge()
    prepared: dict[str, Any] = {"images": []}
    try:
        for image in request.images:
            source = bridge._validate_media_path(image, "image")
            prepared["images"].append(bridge._copy_to_comfy_input(source, runtime_root))
        return prepared
    except BaseException:
        bridge.cleanup_prepared_media(prepared, runtime_root)
        raise


def save_result(
    source: Path, request: PH3Request, seed: int, prompt_id: str, readiness: Any,
    output_directory: Path, upsampled_prompt: str | None, snapshot: dict | None,
) -> dict[str, Any]:
    from modules_forge.studio_checkpoints import verify_snapshot

    identity = verify_snapshot(snapshot)
    output_directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    prefix = "MiniMax_H3_Image" if request.official else "P_H3_Image"
    target = output_directory / f"{prefix}_{stamp}_{uuid.uuid4().hex[:12]}"
    stage = Path(tempfile.mkdtemp(prefix=".p-h3-", dir=output_directory))
    names = request.model_names
    metadata = {
        "schema_version": 1, "model_family": request.family, "output_kind": "image",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "prompt_id": prompt_id, "prompt": request.prompt.strip(), "upsampled_prompt": upsampled_prompt,
        "mode": request.mode, "width": request.width, "height": request.height,
        "seed": seed, "steps": 6 if request.official else 8, "sampler": "euler",
        "reference_count": len(request.reference_images), "has_edit_image": bool(request.target_image),
        "use_upsampler": request.use_upsampler,
        "lora": {"name": request.lora, "strength": request.lora_strength} if request.lora else None,
        "models": {"diffusion_model": Path(request.checkpoint).name if request.checkpoint else names["model"],
                   "text_encoder": names["text_encoder"], "vae": names["vae"], "upsampler": UPSAMPLER if request.use_upsampler else None},
        "checkpoint_sha256": identity.get("sha256") if identity else None,
        "comfyui_revision": readiness.core_revision, "comfyui_version": readiness.comfy_version,
        "comfy_kitchen_version": readiness.package_versions.get("comfy-kitchen"),
        "runtime_profile": readiness.runtime_profile,
    }
    if request.official:
        metadata.update(scheduler="simple", cfg=1.0, denoise=1.0)
    else:
        metadata["sigmas"] = SIGMAS
    try:
        destination = stage / "image.png"
        shutil.copyfile(source, destination)
        verify_png(destination, request.width, request.height)
        (stage / "parameters.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        verify_snapshot(snapshot)
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {"path": str(target / "image.png"), "files": [str(target / "image.png"), str(target / "parameters.json")],
            "metadata": metadata}


def run_generation(
    request: PH3Request, runtime_root: Path, server_url: str, log_directory: Path,
    output_directory: Path, runtime_profile: str = "fast", *, poll_seconds: float = 2.0,
) -> Iterator[dict[str, Any]]:
    request.validate()
    from modules_forge import gpu_residency
    from modules_forge.gpu_ownership import GPUOwnership
    from modules_forge.minimax_h3_runtime import model_root
    from modules_forge.studio_checkpoints import snapshot_checkpoint, verify_snapshot

    bridge = _bridge()
    root = Path(runtime_root)
    original_model_root = model_root(root).resolve()
    checkpoint = str(_checkpoint_path(request, root).resolve())
    if not request.official:
        request = replace(request, checkpoint=checkpoint)

    def verify_model_root():
        if model_root(root).resolve() != original_model_root:
            raise PH3Error("モデル保存先が変更されました。生成をやり直してください。")
    ownership = GPUOwnership()
    ownership.engine = "h3"
    job_id = str(uuid.uuid4())
    client = None
    prepared = {"images": []}
    registered = False
    submitted = False
    terminal = False
    deferred = False
    try:
        def check_cancel():
            if bridge._is_cancelled_job(job_id):
                raise PH3Cancelled("画像生成を停止しました。")

        yield {"stage": "verifying", "message": "モデル本体を確認しています。初回は時間がかかります。", "prompt_id": job_id}
        check_cancel()
        if request.official and not Path(checkpoint).is_file():
            raise PH3Error("MiniMax-H3-Imageの本体がありません。「環境とモデルを準備」を実行してください。")
        snapshot = snapshot_checkpoint(checkpoint, check_cancel=check_cancel)
        check_cancel()
        while not ownership.acquire():
            if bridge._is_cancelled_job(job_id):
                raise PH3Cancelled("画像生成を停止しました。")
            yield {"stage": "queued", "message": "GPUの使用終了を待っています。", "prompt_id": job_id}
            time.sleep(0.1)
        if bridge._is_cancelled_job(job_id):
            raise PH3Cancelled("画像生成を停止しました。")
        bridge.release_forge_vram()
        yield {"stage": "runtime", "message": f"{request.family}の実行環境を確認しています。", "prompt_id": job_id}
        verify_model_root()
        verify_snapshot(snapshot)
        readiness = ensure_runtime(request, root, server_url, log_directory, runtime_profile)
        gpu_residency.register("h3", lambda: bridge._release_retained_runtime(server_url), "h3")
        bridge.cleanup_stale_prepared_media(root)
        prepared = prepare_images(request, root)
        seed = request.resolved_seed()
        graph = build_workflow(request, prepared, seed)
        with bridge._RUNTIME_LIFECYCLE_LOCK:
            verify_model_root()
            verify_snapshot(snapshot)
            readiness = ensure_runtime(request, root, server_url, log_directory, runtime_profile)
            client = bridge.ComfyH3Client(server_url)
            check_nodes(client, request)
            if bridge._is_cancelled_job(job_id):
                raise PH3Cancelled("画像生成を停止しました。")
            process = bridge._loopback_server_process(server_url)
            if process is None:
                raise PH3Error("送信先のComfyUIプロセスを確認できません。")
            verify_snapshot(snapshot)
            bridge.pending_jobs.write({
                "version": 1, "prompt_id": job_id, "runtime_root": os.fspath(root.resolve()),
                "server_url": server_url, "server_process": {"pid": process.pid, "created": process.create_time()},
                "prepared": prepared, "state": "submitting", "cancel_ack": False,
            })
            registered = True
            bridge._mark_active_generation(job_id)
            with bridge._ACTIVE_GENERATION_LOCK:
                bridge._GPU_OWNERSHIPS[job_id] = ownership
            submitted = True  # The server may accept a request even if the response is lost.
            try:
                client.submit(graph, job_id)
                bridge.pending_jobs.update(job_id, state="submitted")
            except bridge.H3SubmissionRejected:
                terminal = True
                raise
            except bridge.H3BridgeError:
                _LOG.warning("P-H3 submission response missing; reconciling the same job ID.")
        started, failures = time.monotonic(), 0
        yield {"stage": "queued", "message": "画像生成をキューに追加しました。", "prompt_id": job_id, "seed": seed}
        while True:
            try:
                if bridge._is_cancelled_job(job_id) and not bridge._cancel_confirmed(job_id):
                    client.cancel(job_id)
                job = client.job(job_id)
            except bridge.H3JobNotFound:
                if bridge._cancel_confirmed(job_id):
                    terminal = True
                    raise PH3Cancelled("画像生成を停止しました。") from None
                raise
            except bridge.H3BridgeError:
                failures += 1
                if failures >= 3:
                    raise
                yield {"stage": "reconnecting", "message": f"状態を再取得しています（{failures}/2）。", "prompt_id": job_id}
                time.sleep(max(0.05, poll_seconds))
                continue
            failures = 0
            status = str(job.get("status", "pending")).lower()
            if status in {"success", "completed"}:
                terminal = True
                for attempt in range(3):
                    try:
                        history = client.history(job_id)
                        output = extract_output(history, job_id, root, request)
                        upsampled = extract_upsampled_prompt(history, job_id, enabled=request.use_upsampler)
                        break
                    except (bridge.H3BridgeError, PH3Error, ValueError):
                        if attempt == 2:
                            raise
                        time.sleep(max(0.05, poll_seconds))
                result = save_result(output, request, seed, job_id, readiness, output_directory, upsampled, snapshot)
                yield {"stage": "complete", "message": "PNGと生成条件を保存しました。", "prompt_id": job_id,
                       "seed": seed, "elapsed": time.monotonic() - started, **result}
                return
            if status in {"cancelled", "canceled"}:
                terminal = True
                raise PH3Cancelled("画像生成を停止しました。")
            if status in {"failed", "error"}:
                terminal = True
                raise PH3Error(bridge._execution_error(job))
            yield {"stage": "running" if status in {"running", "in_progress"} else "queued",
                   "message": "画像を生成しています。", "prompt_id": job_id, "seed": seed,
                   "elapsed": time.monotonic() - started}
            time.sleep(max(0.05, poll_seconds))
    finally:
        try:
            if client is not None and submitted and not terminal:
                try:
                    client.cancel(job_id)
                except (bridge.H3BridgeError, OSError):
                    _LOG.warning("P-H3 cancellation failed; deferring image cleanup.")
                deferred = True
                bridge._schedule_deferred_cleanup(client, job_id, prepared, root)
            if not submitted or terminal:
                bridge.cleanup_prepared_media(prepared, root)
            if registered and (not submitted or terminal):
                bridge.pending_jobs.remove(job_id)
                bridge._clear_cancelled_job(job_id)
        finally:
            if terminal:
                bridge._finish_gpu_generation(job_id)
            bridge._clear_active_generation(job_id)
            if client is not None and not deferred:
                client.close()
            with bridge._ACTIVE_GENERATION_LOCK:
                retained = ownership in bridge._GPU_OWNERSHIPS.values()
            if not retained and not terminal:
                if bridge._is_cancelled_job(job_id):
                    bridge._clear_cancelled_job(job_id)
                ownership.release()


def cancel_generation(prompt_id: str, server_url: str) -> None:
    _bridge().cancel_generation(prompt_id, server_url)
