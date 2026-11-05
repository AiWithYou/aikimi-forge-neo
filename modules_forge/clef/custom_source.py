"""Validate local Clef weights without executing model-supplied code."""

from __future__ import annotations

import hashlib
import json
import math
import stat
from pathlib import Path, PurePosixPath

from modules_forge import local_assets

from .core import PROFILES, ClefError

# SHA-256 over sorted {tensor_name: [dtype, shape]} JSON, not over weights.
# Established from the fixed official source and bitsandbytes bundles.
MODEL_SCHEMAS = {
    ("clef-flash", "bf16"): "3ec5aab3922c00fc40f31e4c8264af52deda3f3429893b9e75ba74dee6a91d12",
    ("clef", "bf16"): "0e423f32f4dfc5163d655fb55bfca59a769e96a6bb4e91e19eb9222865d54e39",
    ("clef-flash", "int8"): "94e178b81c4ce45b7ec8c6866b128f1bac0e878f3c4b1de45aa681cd606bae82",
    ("clef", "nf4"): "a8a8e6b9fe58fb72d4303b08e3731501da87d00791b38dce7b71c77bcbde0cb8",
}
HEAD_SCHEMAS = {
    "clef-flash": "5607b02537103dc4ca001ddc52c8587ebc07b5a41f2ce4b0d4457cafaca659dd",
    "clef": "94f9c271bbd5e54d5ab28761b04057b3f796b694388fda1bf3c53a0c47c6536c",
}
REQUIRED = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "joint_head_config.json",
    "joint_head.safetensors",
    "model.safetensors.index.json",
)
OPTIONAL = (
    "chat_template.jinja",
    "generation_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
)


def safe_member(directory, name):
    if (
        not isinstance(name, str)
        or not name
        or "\\" in name
        or ":" in name
        or name.startswith("/")
        or any(not part or part in {".", ".."} for part in name.split("/"))
        or PurePosixPath(name).is_absolute()
    ):
        raise ClefError("モデルのファイル名またはシャード索引が不正です。")
    directory = Path(directory).resolve()
    path = directory.joinpath(*name.split("/"))
    current = directory
    for part in name.split("/"):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) & 0x20000000:
            raise ClefError("モデル内のリンクや接合点は使用できません。")
    if not path.resolve().is_relative_to(directory) or not path.is_file():
        raise ClefError("モデルのファイルがフォルダー外、または存在しません。")
    return path


def read_json(directory, name):
    with safe_member(directory, name).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ClefError(f"モデル設定が不正です: {name}")
    return value


def check_config(config, model, precision):
    hidden = 4096 if model == "clef-flash" else 5120
    layers = 32 if model == "clef-flash" else 64
    text_expected = {
        "model_type": "qwen3_5_text",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "intermediate_size": 12288 if model == "clef-flash" else 17408,
        "vocab_size": 248320,
        "num_attention_heads": 16 if model == "clef-flash" else 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32 if model == "clef-flash" else 48,
        "linear_conv_kernel_dim": 4,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "tie_word_embeddings": False,
        "attn_output_gate": True,
        "full_attention_interval": 4,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 262144,
        "rope_parameters": {
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "rope_theta": 10000000,
            "rope_type": "default",
        },
    }
    text_expected["layer_types"] *= layers // 4
    vision_expected = {
        "model_type": "qwen3_5_vision",
        "depth": 27,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_heads": 16,
        "out_hidden_size": hidden,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "in_channels": 3,
        "num_position_embeddings": 2304,
        "hidden_act": "gelu_pytorch_tanh",
    }
    if config.get("model_type") != "qwen3_5" or config.get("architectures") not in (
        ["Qwen3_5Model"],
        ["Qwen3_5ForConditionalGeneration"],
    ):
        raise ClefError("Clef本体のarchitectureが一致しません。")
    if config.get("tie_word_embeddings") is not False:
        raise ClefError("Clefの入力・出力語彙設定が一致しません。")
    text, vision = config.get("text_config"), config.get("vision_config")
    if (
        not isinstance(text, dict)
        or not isinstance(vision, dict)
        or any(text.get(key) != value for key, value in text_expected.items())
        or any(vision.get(key) != value for key, value in vision_expected.items())
    ):
        raise ClefError("Clefの通常版／Flashまたはvisionの構造が一致しません。")
    if any(
        config.get(key) != value
        for key, value in {
            "image_token_id": 248056,
            "video_token_id": 248057,
            "vision_start_token_id": 248053,
            "vision_end_token_id": 248054,
        }.items()
    ):
        raise ClefError("Clef画像トークンの設定が一致しません。")
    quant = config.get("quantization_config")
    if precision == "bf16":
        if quant not in (None, {}):
            raise ClefError("BF16本体に量子化設定があります。")
    else:
        if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes":
            raise ClefError("bitsandbytes量子化設定が必要です。")
        if quant.get("load_in_8bit") is not (precision == "int8") or quant.get("load_in_4bit") is not (
            precision == "nf4"
        ):
            raise ClefError("選択した量子化方式が一致しません。")
        if quant.get("_load_in_8bit", precision == "int8") is not (precision == "int8") or quant.get(
            "_load_in_4bit", precision == "nf4"
        ) is not (precision == "nf4"):
            raise ClefError("量子化設定の保存形式が一致しません。")
        if set(quant.get("llm_int8_skip_modules") or []) != {"visual", "lm_head"}:
            raise ClefError("visionと出力語彙は量子化できません。")
        if precision == "nf4" and (
            quant.get("bnb_4bit_quant_type") != "nf4"
            or quant.get("bnb_4bit_use_double_quant") is not True
            or quant.get("bnb_4bit_compute_dtype") != "bfloat16"
            or quant.get("bnb_4bit_quant_storage") != "uint8"
        ):
            raise ClefError("対応するNF4形式ではありません。")
    return hidden


def check_processor(directory):
    processor = read_json(directory, "processor_config.json")
    image = processor.get("image_processor")
    tokenizer = read_json(directory, "tokenizer_config.json")
    if (
        processor.get("processor_class") != "Qwen3VLProcessor"
        or not isinstance(image, dict)
        or (
            image.get("image_processor_type") != "Qwen2VLImageProcessor"
            or image.get("patch_size") != 16
            or image.get("temporal_patch_size") != 2
            or image.get("merge_size") != 2
            or image.get("image_mean") != [0.5] * 3
            or image.get("image_std") != [0.5] * 3
            or image.get("do_convert_rgb") is not True
            or image.get("do_normalize") is not True
            or image.get("do_rescale") is not True
            or image.get("do_resize") is not True
            or image.get("rescale_factor") != 1 / 255
        )
    ):
        raise ClefError("Clefの画像処理設定が一致しません。")
    size = image.get("size")
    if (
        not isinstance(size, dict)
        or type(size.get("shortest_edge")) is not int
        or type(size.get("longest_edge")) is not int
        or not 0 < size["shortest_edge"] <= size["longest_edge"]
    ):
        raise ClefError("Clefの画像処理解像度の設定が不正です。")
    if tokenizer.get("tokenizer_class") != "Qwen2Tokenizer" or tokenizer.get("image_token") != "<|image_pad|>":
        raise ClefError("Clefのtokenizer設定が一致しません。")
    vocab = read_json(directory, "tokenizer.json")
    data = vocab.get("model", {})
    words, added = data.get("vocab"), vocab.get("added_tokens")
    if data.get("type") != "BPE" or not isinstance(words, dict) or not words or not isinstance(added, list):
        raise ClefError("Clefの語彙を確認できません。")
    ids = list(words.values())
    special = {}
    for item in added:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str):
            raise ClefError("Clefの特殊トークンが不正です。")
        special[item["content"]] = item.get("id")
        ids.append(item.get("id"))
    required = {
        "<|endoftext|>": 248044,
        "<|im_start|>": 248045,
        "<|im_end|>": 248046,
        "<|vision_start|>": 248053,
        "<|vision_end|>": 248054,
        "<|image_pad|>": 248056,
        "<|video_pad|>": 248057,
    }
    if (
        any(special.get(key) != value for key, value in required.items())
        or any(type(value) is not int or not 0 <= value < 248320 for value in ids)
        or len(ids) != len(set(ids))
    ):
        raise ClefError("Clefの語彙・特殊トークンが一致しません。")


def tensor_schema(path):
    header = local_assets.read_header(path)
    result = {}
    for name, value in header.items():
        if name == "__metadata__":
            continue
        shape, dtype = value.get("shape"), value.get("dtype")
        if (
            not isinstance(shape, list)
            or any(type(x) is not int or x < 0 for x in shape)
            or dtype not in {"BF16", "F32", "I8", "U8"}
        ):
            raise ClefError("未対応のsafetensorsテンソルです。")
        if (
            value["data_offsets"][1] - value["data_offsets"][0]
            != math.prod(shape) * {"BF16": 2, "F32": 4, "I8": 1, "U8": 1}[dtype]
        ):
            raise ClefError("safetensorsテンソルのサイズが一致しません。")
        result[name] = [dtype, shape]
    return result


def schema_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def custom_manifest(directory, profile):
    directory = Path(directory).resolve()
    settings = PROFILES[profile]
    model, precision = settings["model"], settings["precision"]
    if (model, precision) not in MODEL_SCHEMAS:
        raise ClefError("このモデルと保存形式の組合せには対応していません。")
    try:
        hidden = check_config(read_json(directory, "config.json"), model, precision)
        check_processor(directory)
        head_config = read_json(directory, "joint_head_config.json")
        if head_config != {
            "hidden_size": hidden,
            "width": 1024,
            "routing_layers": 2,
            "layers": 4,
            "heads": 16,
            "feedforward": 4096,
        }:
            raise ClefError("Clefの判断ヘッド設定が一致しません。")
        if schema_digest(tensor_schema(safe_member(directory, "joint_head.safetensors"))) != HEAD_SCHEMAS[model]:
            raise ClefError("Clefの判断ヘッド重みの構造が一致しません。")
        index = read_json(directory, "model.safetensors.index.json").get("weight_map")
        if (
            not isinstance(index, dict)
            or not index
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in index.items())
        ):
            raise ClefError("Clefのシャード索引が不正です。")
        paths = set(index.values())
        if any(not name.endswith(".safetensors") for name in paths):
            raise ClefError("Clefはsafetensorsシャードだけを使用します。")
        actual = {}
        for name in paths:
            for key, value in tensor_schema(safe_member(directory, name)).items():
                if index.get(key) != name:
                    raise ClefError("シャードと索引の対応が一致しません。")
                normalized = key[6:] if key.startswith(("model.visual.", "model.language_model.")) else key
                if normalized in actual:
                    raise ClefError("重複する本体テンソルがあります。")
                actual[normalized] = value
        if len(actual) != len(index) or schema_digest(actual) != MODEL_SCHEMAS[(model, precision)]:
            raise ClefError("Clefの本体・語彙・visionまたは量子化構造が一致しません。")
        names = set(REQUIRED) | paths
        names.update(name for name in OPTIONAL if (directory / name).exists())
        files = [{"path": name, "size": safe_member(directory, name).stat().st_size} for name in sorted(names)]
        return directory, {
            "source": "custom",
            "repo": None,
            "revision": None,
            "model": model,
            "precision": precision,
            "files": files,
        }
    except ClefError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ClefError("Clefモデル一式の形式またはシャード索引を確認してください。") from exc
