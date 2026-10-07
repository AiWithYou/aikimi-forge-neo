"""Local Qwen 2.1 sources. Runtime, shared components and denoiser are independent."""

from __future__ import annotations

import json
import re
from pathlib import Path

from modules_forge import local_assets

TRANSFORMER_CONFIG = {
    "_class_name": "QwenImage21Transformer2DModel",
    "attention_head_dim": 128,
    "axes_dims_rope": [16, 56, 56],
    "context_in_dim": 4096,
    "in_channels": 64,
    "num_attention_heads": 32,
    "num_layers": 32,
    "out_channels": 64,
    "patch_size": 1,
    "mlp_ratio": 3,
    "eps": 1e-6,
    "causal_condition": True,
}
COMPONENTS = {
    "processor": ["transformers", "Qwen3VLProcessor"],
    "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
    "text_encoder": ["transformers", "Qwen3VLForConditionalGeneration"],
    "transformer": ["diffusers", "QwenImage21Transformer2DModel"],
    "vae": ["diffusers", "AutoencoderKLQwenImage21"],
}


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"設定ファイルが不正です: {path}")
    return value


def weight_files(folder: Path) -> list[Path]:
    indexes = list(folder.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"重みの索引が複数あります: {folder}")
    if indexes:
        mapping = read_json(indexes[0]).get("weight_map", {})
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("重みの索引が空です。")
        names = sorted(set(mapping.values()))
        paths = [(folder / name).resolve() for name in names]
        if any(not p.is_relative_to(folder.resolve()) or p.suffix != ".safetensors" or not p.is_file() for p in paths):
            raise ValueError("索引が参照する重みが不足、またはフォルダー外を参照しています。")
        return paths
    paths = sorted(folder.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"重みがありません: {folder}")
    return paths


def validate_transformer(path: Path) -> str:
    if path.suffix.lower() == ".gguf":
        with path.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("GGUFファイルが不正です。")
        return "gguf"  # The worker validates the complete converted tensor set against the model.
    files = [path] if path.is_file() else weight_files(path)
    if path.is_dir():
        config = read_json(path / "config.json")
        if config.get("_class_name") != TRANSFORMER_CONFIG["_class_name"] or any(
            config.get(k, TRANSFORMER_CONFIG[k]) != TRANSFORMER_CONFIG[k]
            for k in ("num_layers", "num_attention_heads", "attention_head_dim", "in_channels", "context_in_dim")
        ):
            raise ValueError("Qwen Image 2.1用のTransformerではありません（旧Qwen Imageとは互換性がありません）。")
        if config.get("quantization_config"):
            raise ValueError("外部の事前量子化Diffusersモデルは未対応です。BF16／FP16の元モデルを指定してください。")
    headers = [local_assets.read_header(p) for p in files]
    keys = {
        k.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
        for h in headers
        for k in h
        if k != "__metadata__"
    }
    blocks = {int(m[1]) for k in keys if (m := re.match(r"transformer_blocks\.(\d+)\.", k))}
    if blocks != set(range(32)) or not {"img_in.weight", "txt_in.in_layer.weight"}.issubset(keys):
        raise ValueError("Qwen Image 2.1本体の32ブロック構成を確認できません。LoRA・旧Qwen用モデルは指定できません。")
    if path.is_file() and any(key.endswith(".comfy_quant") for key in headers[0]):
        from .convrot_int8 import inspect_checkpoint

        return inspect_checkpoint(path, header=headers[0])["format"]
    if any(
        item.get("dtype") not in {"BF16", "F16", "F32"}
        for h in headers
        for key, item in h.items()
        if key != "__metadata__"
    ):
        raise ValueError(
            "単一safetensorsはBF16・FP16・FP32またはQwen 2.1のINT8 ConvRotに対応します。その他の事前量子化は未対応です。"
        )
    return "float"


def preferred_precision(path: Path) -> str:
    if path.suffix.lower() == ".gguf":
        return "base_q4_k_m"
    if path.suffix.lower() == ".safetensors":
        return "int8" if validate_transformer(path) == "int8_convrot" else "bf16"
    return "int8"


def resolve(runtime: Path, model: str, components: str, precision: str) -> dict:
    candidate = Path(model.strip().strip('"')).expanduser()
    if not candidate.is_absolute():
        candidate = local_assets.ROOT / candidate
    path = local_assets.local_path(model, directory=candidate.is_dir())
    full = path.is_dir() and (path / "model_index.json").is_file()
    base = path if full else local_assets.local_path(components or str(runtime / "model"), directory=True)
    transformer = path / "transformer" if full else path
    index = read_json(base / "model_index.json")
    if index.get("_class_name") != "QwenImage21Pipeline" or any(index.get(k) != v for k, v in COMPONENTS.items()):
        raise ValueError("共通部品にはQwen Image 2.1のDiffusersフォルダーを指定してください。")
    for name in ("text_encoder", "vae"):
        read_json(base / name / "config.json")
        for weights in weight_files(base / name):
            local_assets.read_header(weights)
    if read_json(base / "text_encoder/config.json").get("model_type") != "qwen3_vl":
        raise ValueError("Qwen 2.1のテキストエンコーダーではありません。")
    if read_json(base / "vae/config.json").get("_class_name") != "AutoencoderKLQwenImage21":
        raise ValueError("Qwen 2.1のVAEではありません。")
    if not (base / "processor").is_dir() or not (base / "scheduler/scheduler_config.json").is_file():
        raise ValueError("processor／schedulerが不足しています。共通部品のみのセットアップで取得できます。")
    if precision.startswith("turbo_"):
        raise ValueError("ローカルモデルは通常版の精度を選んでください。Turboプリセットは標準専用です。")
    gguf = transformer.suffix.lower() == ".gguf"
    if gguf != (precision == "base_q4_k_m"):
        raise ValueError("GGUFにはQ4の設定、safetensors／フォルダーにはBF16・INT8・W4A8を選んでください。")
    format_name = validate_transformer(transformer)
    if transformer.is_file() and not gguf:
        required = "int8" if format_name == "int8_convrot" else "bf16"
        if precision != required:
            raise ValueError(
                "INT8 ConvRot本体はINT8設定で使用してください。"
                if format_name == "int8_convrot"
                else "単一safetensorsはBF16設定で使用します。INT8／W4A8にはDiffusers Transformerフォルダーを指定してください。"
            )
    return {
        "model": str(base),
        "transformer": str(transformer),
        "format": format_name,
        "identity": local_assets.identity(transformer),
        "components": {
            name: local_assets.identity(base / name) for name in ("text_encoder", "vae", "processor", "scheduler")
        },
    }


def runtime_root(model_path: Path, request: dict) -> Path:
    return Path(request.get("runtime_root") or model_path.parent)


def transformer_path(model_path: Path, request: dict) -> Path:
    source = request.get("local_source") or {}
    return Path(source.get("transformer") or model_path / "transformer")


def validate_folder_tensors(path: Path, model_class) -> None:
    """Check every tensor before a quantizer can hide missing or extra weights."""
    from accelerate import init_empty_weights

    with init_empty_weights():
        expected = model_class.from_config(read_json(path / "config.json")).state_dict()
    actual = {}
    for file in weight_files(path):
        for name, tensor in local_assets.read_header(file).items():
            if name == "__metadata__":
                continue
            if name in actual:
                raise ValueError(f"重みのテンソルが重複しています: {name}")
            actual[name] = tensor["shape"]
    if set(actual) != set(expected) or any(tuple(actual[k]) != tuple(expected[k].shape) for k in actual):
        raise ValueError("ローカル本体のテンソル名・寸法がQwen Image 2.1と一致しません。部分読み込みは行いません。")


def load_single(path: Path, model_class, runtime: Path):
    import torch
    from accelerate import init_empty_weights

    with init_empty_weights():
        expected = model_class.from_config(TRANSFORMER_CONFIG).state_dict()

    def convert(checkpoint, **_kwargs):
        state = {}
        for name, weight in checkpoint.items():
            name = name.removeprefix("model.diffusion_model.").removeprefix("diffusion_model.")
            if name.endswith(".img_mlp.gate_up.weight"):
                a, b = weight.chunk(2, dim=0)
                state[name.replace("gate_up", "gate_layer")] = a
                state[name.replace("gate_up", "proj")] = b
            else:
                state[name] = weight
        if set(state) != set(expected):
            raise ValueError("ローカル本体のテンソル構成がQwen Image 2.1と一致しません。部分読み込みは行いません。")
        if path.suffix.lower() != ".gguf" and any(state[k].shape != expected[k].shape for k in state):
            raise ValueError("ローカル本体のテンソル寸法が一致しません。")
        return state

    if path.suffix.lower() != ".gguf":
        from safetensors.torch import load_file

        state = convert(load_file(str(path)))
        with init_empty_weights():
            model = model_class.from_config(TRANSFORMER_CONFIG)
        model.load_state_dict(state, strict=True, assign=True)
        return model.to(dtype=torch.bfloat16)
    from diffusers import GGUFQuantizationConfig
    from diffusers.loaders import single_file_model

    from .gguf import restore_non_linear_weights

    config_root = runtime / "local-config"
    folder = config_root / "transformer"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "config.json").write_text(json.dumps(TRANSFORMER_CONFIG), encoding="utf-8")
    mapping = single_file_model.SINGLE_FILE_LOADABLE_CLASSES
    previous = mapping.get("QwenImage21Transformer2DModel")
    mapping["QwenImage21Transformer2DModel"] = {"checkpoint_mapping_fn": convert, "default_subfolder": "transformer"}
    try:
        model = model_class.from_single_file(
            str(path),
            config=str(config_root),
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        return restore_non_linear_weights(model)
    finally:
        if previous is None:
            mapping.pop("QwenImage21Transformer2DModel", None)
        else:
            mapping["QwenImage21Transformer2DModel"] = previous
