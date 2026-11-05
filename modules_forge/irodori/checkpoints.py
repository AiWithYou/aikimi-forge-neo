"""Content-bound, single-file Irodori v4-Large checkpoints."""

from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from modules_forge import local_assets

from .core import MODEL_REVISION, MODELS, REQUIRED_FILES, RUNTIME, IrodoriError

V4_LARGE_SCHEMA_SHA256 = "dd6ed19fc21132bb48b186e9a35246a2f12fea4e459c839cf53a1f3fdbb12e19"
CONFIG_REQUIREMENTS = {
    "flow_parameterization": "rf_velocity", "latent_dim": 32, "latent_patch_size": 1,
    "model_dim": 2048, "num_layers": 24, "num_heads": 32,
    "text_vocab_size": 262144, "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "text_encoder_revision": "dd0a2683227859151b1730ca3a63087df5b5f39b", "text_add_bos": True,
    "text_encoder_type": "pretrained", "pretrained_projector_type": "residual_mlp", "text_dim": 1280,
    "use_caption_condition": True, "use_speaker_condition": True,
    "caption_vocab_size": 262144, "caption_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "caption_add_bos": True, "caption_dim": 1280, "speaker_dim": 1280, "speaker_layers": 14,
    "speaker_heads": 20, "speaker_patch_size": 4, "use_duration_predictor": True,
}
QUANTIZATION_TYPES = {
    "int8_weight_only", "int8_dynamic_activation_int8_weight", "int4_weight_only",
    "float8_weight_only", "float8_dynamic_activation_float8_weight",
}


def _mapping(value, label):
    try:
        result = json.loads(value)
    except (TypeError, ValueError) as error:
        raise IrodoriError(f"{label}を読み取れません。") from error
    if not isinstance(result, dict):
        raise IrodoriError(f"{label}が不正です。")
    return result


def _config(metadata):
    config = _mapping(metadata.get("config_json"), "モデル設定")
    for key, expected in CONFIG_REQUIREMENTS.items():
        actual = config.get(key)
        if type(actual) is not type(expected) or actual != expected:
            raise IrodoriError(f"v4-Largeのモデル設定または公式tokenizerと一致しません: {key}")
    encoder = _mapping(metadata.get("text_encoder_config_json"), "テキストエンコーダー設定")
    section = encoder.get("encoder", {})
    text = section.get("text_config", {}) if isinstance(section, dict) else {}
    if not isinstance(text, dict):
        text = {}
    if (
        encoder.get("model_type") != "t5gemma2"
        or encoder.get("vocab_size") != 262144
        or (encoder.get("bos_token_id"), encoder.get("eos_token_id"), encoder.get("pad_token_id")) != (2, 1, 0)
        or text.get("model_type") != "t5gemma2_text"
        or text.get("vocab_size") != 262144
        or text.get("hidden_size") != 1152
        or text.get("num_hidden_layers") != 26
    ):
        raise IrodoriError("公式tokenizerと互換のテキストエンコーダーではありません。")


def _tensor(header, key, dtype=None, shape=None):
    entry = header.get(key)
    if not isinstance(entry, dict):
        raise IrodoriError(f"モデルのテンソルがありません: {key}")
    actual_shape, actual_dtype = entry.get("shape"), entry.get("dtype")
    if (
        not isinstance(actual_shape, list)
        or any(type(x) is not int or x <= 0 for x in actual_shape)
        or (shape is not None and actual_shape != shape)
        or (dtype is not None and actual_dtype not in dtype)
    ):
        raise IrodoriError(f"モデルのテンソル形式が不正です: {key}")
    return actual_shape


def inspect_header(header):
    if not isinstance(header, dict) or not isinstance(header.get("__metadata__"), dict):
        raise IrodoriError("モデルの設定メタデータがありません。")
    metadata = header["__metadata__"]
    _config(metadata)
    raw_quant = metadata.get("irodori_quantization_json")
    qtype = "float"
    schema, consumed = [], set()
    if raw_quant is None:
        for name in set(header) - {"__metadata__"}:
            schema.append((name, _tensor(header, name, {"F32", "F16", "BF16"})))
    else:
        quant = _mapping(raw_quant, "量子化設定")
        qtype = quant.get("quantization_type")
        if (
            type(quant.get("format_version")) is not int or quant.get("format_version") != 1
            or quant.get("backend") != "torchao" or quant.get("backend_version") != "0.16.0"
            or quant.get("compute_dtype") != "bfloat16" or quant.get("profile") not in {"core", "all-linear"}
            or qtype not in QUANTIZATION_TYPES
            or type(quant.get("quantized_modules")) is not int or quant["quantized_modules"] <= 0
        ):
            raise IrodoriError("未対応のIrodori量子化形式です。")
        if qtype == "int4_weight_only" and (
            type(quant.get("group_size")) is not int or quant["group_size"] not in {32, 64, 128, 256}
            or quant.get("packing_format") != "tile_packed_to_4d"
        ):
            raise IrodoriError("未対応のINT4保存形式です。")
        try:
            names = json.loads(metadata["tensor_names"])
        except (TypeError, ValueError, KeyError) as error:
            raise IrodoriError("量子化テンソル一覧が不正です。") from error
        if not isinstance(names, list) or not all(isinstance(x, str) and x for x in names) or len(names) != len(set(names)):
            raise IrodoriError("量子化テンソル一覧が不正です。")
        count = 0
        for name in names:
            item = _mapping(metadata.get(name), "量子化テンソル情報")
            kind = item.get("_type")
            if kind == "Tensor":
                schema.append((name, _tensor(header, name, {"F32", "F16", "BF16"})))
                consumed.add(name)
                continue
            expected_kind = (
                "Int8Tensor" if qtype.startswith("int8_") else
                "Int4TilePackedTo4dTensor" if qtype == "int4_weight_only" else "Float8Tensor"
            )
            if kind != expected_kind or not name.endswith(".weight") or not isinstance(item.get("_data"), dict):
                raise IrodoriError("量子化テンソルの方式が一致しません。")
            data, parts = item["_data"], item.get("_tensor_data_names")
            prefix = name.removesuffix("weight") + "_weight_"
            qkey = prefix + "qdata"
            dynamic = qtype in {"int8_dynamic_activation_int8_weight", "float8_dynamic_activation_float8_weight"}
            if qtype == "int4_weight_only":
                shape = data.get("shape")
                group = quant["group_size"]
                if (
                    parts != ["qdata", "scale_and_zero"] or not isinstance(shape, list) or len(shape) != 2
                    or any(type(x) is not int or x <= 0 for x in shape)
                    or data.get("block_size") != [1, group]
                ):
                    raise IrodoriError("INT4のテンソル形状が不正です。")
                padded_rows = (shape[0] + 7) // 8 * 8
                padded_columns = (shape[1] + 1023) // 1024 * 1024
                _tensor(header, qkey, {"I32"}, [padded_rows // 8, padded_columns // 128, 32, 4])
                skey = prefix + "scale_and_zero"
                _tensor(header, skey, {"BF16"}, [padded_columns // group, padded_rows, 2])
            else:
                shape = _tensor(header, qkey, {"I8"} if kind == "Int8Tensor" else {"F8_E4M3"})
                if len(shape) != 2 or parts != ["qdata", "scale"]:
                    raise IrodoriError("量子化テンソルの形状が不正です。")
                block_shape = shape if qtype == "float8_dynamic_activation_float8_weight" else [1, shape[1]]
                activation = data.get("act_quant_kwargs")
                activation_type = "QuantizeTensorToInt8Kwargs" if kind == "Int8Tensor" else "QuantizeTensorToFloat8Kwargs"
                if (
                    data.get("block_size") != block_shape
                    or data.get("dtype") != {"_type": "torch.dtype", "_data": "bfloat16"}
                    or (activation is not None) != dynamic
                    or (dynamic and (not isinstance(activation, dict) or activation.get("_type") != activation_type))
                ):
                    raise IrodoriError("量子化の活性値設定が一致しません。")
                skey = prefix + "scale"
                scale_shape = [1, 1] if qtype == "float8_dynamic_activation_float8_weight" else [shape[0], 1]
                _tensor(header, skey, {"BF16", "F32"} if kind == "Int8Tensor" else {"F32"}, scale_shape)
            schema.append((name, shape))
            consumed.update((qkey, skey))
            count += 1
        if count != quant["quantized_modules"] or consumed != set(header) - {"__metadata__"}:
            raise IrodoriError("量子化本体の対象が不完全です。")
    digest = hashlib.sha256(json.dumps(sorted(schema), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
    if digest != V4_LARGE_SCHEMA_SHA256:
        raise IrodoriError("Irodori v4-Largeの本体構造と一致しません。")
    return qtype


def _no_links(path):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_reparse_tag", 0) & 0x20000000:
            raise IrodoriError("モデルの保存先にリンクは使用できません。")
    return path.resolve()


def validate_name(name):
    if name == "official":
        return name
    if (
        not isinstance(name, str) or not name or len(name) > 255 or name.startswith(".")
        or any(x in name for x in ("/", "\\", ":")) or not name.lower().endswith(".safetensors")
    ):
        raise IrodoriError("独自本体はcheckpoints直下のsafetensors名で選択してください。")
    return name


def resolve(models, name):
    root = _no_links(models)
    validate_name(name)
    if name == "official":
        path = root / "v4-Large/model.safetensors"
    else:
        path = root / "v4-Large/checkpoints" / name
    path = _no_links(path)
    if not path.is_relative_to(root) or not path.is_file():
        raise IrodoriError("選択した本体が見つかりません。一覧を更新してください。")
    return path


def ensure_gpu_compatibility(kind, capability):
    if kind == "int4_weight_only" and capability < (8, 0):
        raise IrodoriError("INT4 weight-onlyにはCompute Capability 8.0以上が必要です。")
    if kind == "float8_dynamic_activation_float8_weight" and capability < (8, 9):
        raise IrodoriError("FP8 dynamicにはCompute Capability 8.9以上が必要です。")


def verify_components(models, *, runtime=RUNTIME):
    try:
        files = json.loads((Path(runtime) / "model-files.json").read_text(encoding="utf-8"))
        records = {item["file"]: item["sha256"] for item in files}
        for name in REQUIRED_FILES[1:]:
            path = _no_links(Path(models) / name)
            if local_assets.file_identity(path)["sha256"] != records[name]:
                raise IrodoriError(f"公式tokenizer・codecなどの共有部品が変更されました: {name}")
    except (OSError, KeyError, TypeError) as error:
        raise IrodoriError("公式tokenizer・codecなどの共有部品を確認できません。") from error


def inventory(models=MODELS):
    choices = [("公式 v4-Large", "official")]
    directory = Path(models) / "v4-Large/checkpoints"
    for candidate in sorted(directory.glob("*.safetensors"), key=lambda x: x.name.casefold()):
        try:
            path = resolve(models, candidate.name)
            kind = inspect_header(local_assets.read_header(path))
        except (OSError, ValueError, TypeError, KeyError):
            continue
        choices.append((f"{candidate.stem} · {kind}", candidate.name))
    return choices


def bind(models, request, *, runtime=RUNTIME):
    name = request.get("checkpoint", "official")
    path = resolve(models, name)
    before = local_assets.file_version(path)
    kind = inspect_header(local_assets.read_header(path))
    identity = local_assets.file_identity(path)
    after = local_assets.file_version(path)
    if before != after:
        raise IrodoriError("確認中に本体が変更されました。")
    if kind != "float" and request["precision"] != "bf16":
        raise IrodoriError("量子化本体の計算精度はBF16を選択してください。")
    record = {"name": name, "sha256": identity["sha256"], "format": kind, "size": after[0]}
    if name == "official":
        try:
            files = json.loads((Path(runtime) / "model-files.json").read_text(encoding="utf-8"))
            expected = next(item["sha256"] for item in files if item["file"] == "v4-Large/model.safetensors")
        except (OSError, ValueError, TypeError, KeyError, StopIteration) as error:
            raise IrodoriError("公式本体の記録を確認できません。") from error
        if record["sha256"] != expected:
            raise IrodoriError("公式本体が変更されています。再セットアップしてください。")
        record["model_revision"] = MODEL_REVISION
    if "checkpoint_identity" in request and request["checkpoint_identity"] != record:
        raise IrodoriError("選択した本体が変更されました。選び直してください。")
    request["checkpoint_identity"] = record
    return record
