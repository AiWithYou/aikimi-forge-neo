"""Validated Qwen Image 2.1 requests and local runtime/artifact boundaries."""

from __future__ import annotations

import math
import os
import secrets
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from PIL import Image, ImageOps

from modules_forge.yue2_studio.core import atomic_json as atomic_json
from modules_forge.yue2_studio.core import read_json
from modules_forge.yue2_studio.core import runtime_lock as _runtime_lock
from modules_forge.yue2_studio.core import safe_environment as safe_environment

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_REVISION = "6256aa7666cedd47443adc8f82da9a10e110b09c"
MAX_REFERENCE_IMAGES = 10
MAX_IMAGE_PIXELS = 40_000_000
MAX_OUTPUT_PIXELS = 2400 * 1792
SETUP_COMMAND = "aikimi-qwen-image21-setup.bat"


def precision_label(precision: str) -> str:
    return {
        "base_q4_k_m": "通常版 Q4_K_M (Unsloth)",
        "turbo_bf16": "Viggle Turbo BF16",
        "turbo_q4_k_m": "Viggle Turbo Q4_K_M",
    }.get(precision, precision.upper())


class QwenImage21Error(ValueError):
    """An actionable input, runtime, or artifact error."""


def integer(value, label: str, lower: int, upper: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise QwenImage21Error(f"{label}は整数で指定してください。")
    try:
        number = int(value)
        if isinstance(value, str):
            if value.strip() != str(number):
                raise ValueError
        elif not math.isfinite(value) or number != value:
            raise ValueError
    except (ValueError, OverflowError):
        raise QwenImage21Error(f"{label}は整数で指定してください。") from None
    if not lower <= number <= upper:
        raise QwenImage21Error(f"{label}は{lower}〜{upper}の範囲です。")
    return number


def inside(root: Path, path: Path) -> Path:
    root, path = root.resolve(), path.resolve()
    if not path.is_relative_to(root):
        raise QwenImage21Error("Qwen Image 2.1の保存先の外側は参照できません。")
    return path


def validate_images(paths) -> tuple[str, ...]:
    if not isinstance(paths, (list, tuple)) or len(paths) > MAX_REFERENCE_IMAGES:
        raise QwenImage21Error(f"参照画像は最大{MAX_REFERENCE_IMAGES}枚です。")
    checked = []
    for index, value in enumerate(paths, 1):
        if not isinstance(value, (str, Path)):
            raise QwenImage21Error(f"参照画像{index}をアップロードし直してください。")
        path = Path(value)
        try:
            if not path.is_file() or path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("画像が見つからないか、64 MBを超えています。")
            with Image.open(path) as source:
                if source.width * source.height > MAX_IMAGE_PIXELS:
                    raise ValueError("画像は4000万画素以内にしてください。")
                if getattr(source, "n_frames", 1) != 1:
                    raise ValueError("アニメーションは使用できません。静止画像を指定してください。")
                source.verify()
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            raise QwenImage21Error(f"参照画像{index}: {exc}") from exc
        checked.append(str(path.resolve()))
    return tuple(checked)


def copy_inputs(paths, directory: Path) -> list[str]:
    """Snapshot ordered references before the browser can remove its uploads."""
    paths = validate_images(paths)
    result = []
    for index, value in enumerate(paths, 1):
        destination = inside(directory, directory / f"reference-{index:02d}.png")
        with Image.open(value) as source:
            source.load()
            image = ImageOps.exif_transpose(source)
            has_alpha = "A" in image.getbands() or "transparency" in image.info
            image.convert("RGBA" if has_alpha else "RGB").save(destination, format="PNG")
        result.append(str(destination))
    return result


def copy_control_image(path: str, directory: Path) -> str:
    """Keep the selected control map with the job before Gradio clears uploads."""
    validated = validate_images((path,))[0]
    target = inside(directory, directory / "control-source.png")
    with Image.open(validated) as source:
        ImageOps.exif_transpose(source).convert("RGB").save(target, format="PNG")
    return str(target)


@dataclass(frozen=True)
class Request:
    prompt: str
    width: int = 1024
    height: int = 1024
    steps: int = 40
    seed: int = -1
    transparent: bool = False
    precision: str = "int8"
    memory_mode: str = "offload"
    input_images: tuple[str, ...] = ()
    annotation_reference: int = -1
    annotation_layers: tuple[str, ...] = ()
    rewrite_prompt: bool = False
    sparse_mode: str = "off"
    sparse_keep_percent: float = 75.0
    sparse_jev_cadence: str = "legacy"
    sparse_jev_interval: int = 2
    rewrite_edit_prompt: bool = False
    preserve_unmasked: bool = False
    edit_mask_reference: int = -1
    edit_mask_path: str = ""
    mask_feather: float = 0.0
    sparse_jev_max_calls: int = 0
    sparse_jev_max_wait_seconds: float = 0.0
    control_kind: str = "off"
    control_image: str = ""
    control_strength: float = 1.0
    control_inpaint: bool = False
    fun_acc: bool = False
    operation: str = "generate"

    def resolved(self) -> Request:
        if self.operation not in {"generate", "prepare"}:
            raise QwenImage21Error("実行する操作が不正です。")
        if self.operation == "prepare" and self.precision not in {"int8", "w4a8"}:
            raise QwenImage21Error("保存する精度はINT8またはW4A8を選んでください。")
        if not isinstance(self.prompt, str) or not self.prompt.strip() or len(self.prompt) > 12000:
            raise QwenImage21Error("プロンプトを1〜12000文字で入力してください。")
        width = integer(self.width, "幅", 256, 4096)
        height = integer(self.height, "高さ", 256, 4096)
        if width % 32 or height % 32 or width * height > MAX_OUTPUT_PIXELS:
            raise QwenImage21Error("幅・高さは32の倍数、総画素数は約430万画素（2400×1792）以内で指定してください。")
        steps = integer(self.steps, "Steps", 1, 100)
        seed = integer(self.seed, "Seed", -1, 2**63 - 1)
        if self.precision not in {"int8", "bf16", "w4a8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"}:
            raise QwenImage21Error("モデル・精度の指定が不正です。")
        if self.precision.startswith("turbo_") and steps != 4:
            raise QwenImage21Error("Viggle Turboは4 stepsで生成してください。")
        if not isinstance(self.fun_acc, bool):
            raise QwenImage21Error("Fun Accの指定が不正です。")
        if self.fun_acc and (self.operation != "generate" or self.precision != "int8" or steps != 4):
            raise QwenImage21Error("Fun Accは通常版INT8・4 stepsの画像生成で使用してください。")
        if self.fun_acc and (self.sparse_mode != "off" or self.control_kind != "off"):
            raise QwenImage21Error("Fun AccではSparse AttentionとFun ControlNetをOFFにしてください。")
        if self.precision.startswith("turbo_") and self.sparse_mode != "off":
            raise QwenImage21Error("Viggle TurboではSparse AttentionをOFFにしてください。")
        if self.precision == "base_q4_k_m" and self.sparse_mode != "off":
            raise QwenImage21Error("GGUF版ではSparse AttentionをOFFにしてください。")
        if self.memory_mode not in {"offload", "gpu"}:
            raise QwenImage21Error("メモリ設定が不正です。")
        if not isinstance(self.transparent, bool):
            raise QwenImage21Error("透過背景の指定が不正です。")
        if not isinstance(self.rewrite_prompt, bool):
            raise QwenImage21Error("プロンプト書き換えの指定が不正です。")
        if not isinstance(self.rewrite_edit_prompt, bool):
            raise QwenImage21Error("編集指示の書き換え指定が不正です。")
        if not isinstance(self.preserve_unmasked, bool):
            raise QwenImage21Error("範囲外固定の指定が不正です。")
        kinds = {"off", "canny", "depth", "gray", "hed", "lineart", "mlsd", "pose", "scribble"}
        if not isinstance(self.control_kind, str) or self.control_kind not in kinds:
            raise QwenImage21Error("Fun ControlNetの条件の種類が不正です。")
        if isinstance(self.control_strength, bool) or not isinstance(self.control_strength, (int, float)) or not math.isfinite(self.control_strength) or not 0 <= self.control_strength <= 2:
            raise QwenImage21Error("Fun ControlNetの強さは0〜2で指定してください。")
        if not isinstance(self.control_inpaint, bool):
            raise QwenImage21Error("Fun ControlNetのInpainting指定が不正です。")
        if self.control_inpaint and self.control_kind == "off":
            raise QwenImage21Error("Inpainting＋Controlには制御画像と種類を指定してください。")
        control_image = ""
        if self.control_kind != "off":
            if self.operation != "generate" or self.precision != "int8":
                raise QwenImage21Error("Fun ControlNetは通常版INT8の画像生成で使用してください。")
            if self.sparse_mode != "off":
                raise QwenImage21Error("Fun ControlNetではSparse AttentionをOFFにしてください。")
            if not self.control_image:
                raise QwenImage21Error("前処理済みの制御画像を追加してください。")
            control_image = validate_images((self.control_image,))[0]
        from modules_forge.jev_sparse.qwen21 import Options

        try:
            Options(
                mode=self.sparse_mode,
                keep_percent=self.sparse_keep_percent,
                decision_cadence=self.sparse_jev_cadence,
                update_interval=integer(self.sparse_jev_interval, "Jevの再判定間隔", 1, 100),
                job_max_calls=integer(self.sparse_jev_max_calls, "Jevのジョブ内呼出上限", 0, 1000),
                job_max_wait_seconds=self.sparse_jev_max_wait_seconds,
            ).validate()
        except ValueError as exc:
            raise QwenImage21Error(str(exc)) from None
        inputs = validate_images(self.input_images)
        reference = integer(self.annotation_reference, "描画対象", -1, MAX_REFERENCE_IMAGES - 1)
        if not isinstance(self.annotation_layers, (list, tuple)):
            raise QwenImage21Error("描画レイヤーの形式が不正です。")
        if (reference == -1) != (not self.annotation_layers):
            raise QwenImage21Error("描画対象と描画レイヤーを一緒に指定してください。")
        layers = ()
        if reference != -1:
            if reference >= len(inputs):
                raise QwenImage21Error("描画対象の参照画像がありません。選択し直してください。")
            from .annotations import validate_annotation_layers

            layers, _ = validate_annotation_layers(inputs[reference], self.annotation_layers)
        from .annotations import validate_edit_mask, validate_mask_feather

        feather = validate_mask_feather(self.mask_feather)
        mask_reference = integer(self.edit_mask_reference, "マスクの編集元", -1, MAX_REFERENCE_IMAGES - 1)
        if not isinstance(self.edit_mask_path, (str, Path)):
            raise QwenImage21Error("編集マスクの形式が不正です。")
        mask_path = ""
        if self.preserve_unmasked or self.control_inpaint:
            if not 0 <= mask_reference < len(inputs) or not self.edit_mask_path:
                raise QwenImage21Error("マスク編集には編集元の参照画像と編集マスクを指定してください。")
            mask_path, _ = validate_edit_mask(
                inputs[mask_reference], str(self.edit_mask_path), output_size=(width, height)
            )
        else:
            # Disabled controls may retain browser state; never turn a guide or
            # an old mask into an edit restriction implicitly.
            mask_reference = -1
        return replace(
            self,
            prompt=self.prompt.strip(),
            width=width,
            height=height,
            steps=steps,
            seed=secrets.randbits(63) if seed == -1 else seed,
            input_images=inputs,
            annotation_reference=reference,
            annotation_layers=layers,
            sparse_jev_interval=int(self.sparse_jev_interval),
            edit_mask_reference=mask_reference,
            edit_mask_path=mask_path,
            mask_feather=feather,
            sparse_jev_max_calls=int(self.sparse_jev_max_calls),
            sparse_jev_max_wait_seconds=float(self.sparse_jev_max_wait_seconds),
            control_image=control_image,
            control_strength=float(self.control_strength),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def runtime_manifest(root: Path, precision: str = "int8") -> dict:
    """Read only installed, revision-matched local assets; never trigger downloads."""
    try:
        manifest = read_json(root / "runtime.json")
    except (OSError, ValueError) as exc:
        raise QwenImage21Error(f"未導入です。{SETUP_COMMAND}を実行してください。") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise QwenImage21Error(f"実行環境の登録が不正です。{SETUP_COMMAND}を再実行してください。")
    if manifest.get("model_revision") != MODEL_REVISION or manifest.get("diffusers_revision") != DIFFUSERS_REVISION:
        raise QwenImage21Error(f"実行環境のバージョンが一致しません。{SETUP_COMMAND}を再実行してください。")
    python = root / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    model = root / "model"
    if not isinstance(manifest.get("python"), str) or not isinstance(manifest.get("model"), str):
        raise QwenImage21Error(f"Python・モデルの登録が不正です。{SETUP_COMMAND}を再実行してください。")
    if Path(manifest["python"]).absolute() != python.absolute() or Path(manifest["model"]).resolve() != model.resolve():
        raise QwenImage21Error(f"専用環境の保存先が一致しません。{SETUP_COMMAND}を再実行してください。")
    if not python.is_file() or not model.is_dir():
        raise QwenImage21Error(f"専用Pythonまたはモデルが不足しています。{SETUP_COMMAND}を再実行してください。")
    try:
        index = read_json(model / "model_index.json")
        if not isinstance(index, dict) or index.get("_class_name") != "QwenImage21Pipeline":
            raise ValueError("QwenImage21Pipelineが見つかりません。")
    except (OSError, ValueError) as exc:
        raise QwenImage21Error(f"Qwen Image 2.1モデルが不完全です。{SETUP_COMMAND}を再実行してください。") from exc
    try:
        inventory = read_json(root / "model-files.json")
        if not isinstance(inventory, dict) or inventory.get("revision") != MODEL_REVISION:
            raise ValueError("モデルファイルの記録が一致しません。")
        files = inventory.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("モデルファイルの記録がありません。")
        seen, components = set(), set()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("モデルファイルの記録が不正です。")
            relative = Path(item["path"])
            size = item.get("size")
            if (
                relative.is_absolute()
                or relative in seen
                or isinstance(size, bool)
                or not isinstance(size, int)
                or size <= 0
            ):
                raise ValueError("モデルファイルの記録が不正です。")
            seen.add(relative)
            path = inside(model, model / relative)
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"不足またはサイズ不一致: {relative}")
            if relative.suffix in {".safetensors", ".bin"} and len(relative.parts) > 1:
                components.add(relative.parts[0])
        required = {"text_encoder", "vae"}
        if not precision.startswith("turbo_") and precision != "base_q4_k_m":
            required.add("transformer")
        if not required.issubset(components):
            raise ValueError("必要なTransformer・テキストエンコーダー・VAEの記録が不足しています。")
    except (OSError, ValueError) as exc:
        raise QwenImage21Error(f"モデルの検証に失敗しました（{exc}）。{SETUP_COMMAND}を再実行してください。") from exc
    if precision.startswith("turbo_"):
        from .turbo import turbo_manifest

        turbo_manifest(root, precision)
    elif precision == "base_q4_k_m":
        from .regular_gguf import regular_manifest

        regular_manifest(root)
    return {**manifest, "python": str(python.absolute()), "model": str(model.resolve())}


def runtime_status(root: Path) -> str:
    from .regular_gguf import regular_status
    from .turbo import turbo_status

    try:
        runtime_manifest(root, "int8")
    except QwenImage21Error as exc:
        base = str(exc)
    else:
        base = "公式フルモデルを導入済み。INT8 / W4A8 / BF16を選べます。"

    return "\n".join((regular_status(root), base, turbo_status(root, "turbo_bf16"), turbo_status(root, "turbo_q4_k_m")))


def runtime_lock(root: Path):
    try:
        return _runtime_lock(root)
    except ValueError as exc:
        raise QwenImage21Error(
            "Qwen Image 2.1のセットアップまたは実行環境が使用中です。終了後に再実行してください。"
        ) from exc
