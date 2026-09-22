"""Optional, local NF4 text and image-edit prompt rewriting for Qwen Image 2.1."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

from .core import QwenImage21Error, inside, read_json

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
MODEL_REVISION = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"
DIRECTORY = "prompt-rewriter-nf4"
SETUP_COMMAND = "aikimi-qwen-image21-setup.bat --prompt-rewriter-only"
MAX_NEW_TOKENS = 4096
EDIT_MODEL_ID = "Qwen/Qwen-Image-2.1-PE-I2I"
EDIT_MODEL_REVISION = "72927bc08afc99b7888ceb7d7d51a12db3700bbd"
EDIT_DIRECTORY = "edit-prompt-rewriter-nf4"
EDIT_SETUP_COMMAND = "aikimi-qwen-image21-setup.bat --edit-prompt-rewriter-only"
RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "1:2", "2:1", "21:9", "9:21", "4:5", "5:4", "3:1", "1:3")
_LITERALS = re.compile(r'「([^」]+)」|『([^』]+)』|"([^"\n]+)"|“([^”]+)”')
_REFERENCES = re.compile(
    r"<image(\d+)>|(?<![A-Za-z0-9_])image[ \t]*(\d+)(?![A-Za-z0-9_:：×/])"
    r"|画像[ \t]*(\d+)(?![\d:：xX×/枚個つ])|(?<!\d)(\d+)[ \t]*枚目",
    re.IGNORECASE,
)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def profile(editing=False):
    return (EDIT_MODEL_ID, EDIT_MODEL_REVISION, EDIT_DIRECTORY) if editing else (MODEL_ID, MODEL_REVISION, DIRECTORY)


def rewriter_manifest(root: Path, *, verify_hashes: bool = False, editing: bool = False) -> dict:
    """Validate only local files; an OFF request never needs this optional model."""
    model_id, revision, directory = profile(editing)
    model = inside(root, root / directory)
    try:
        record = read_json(model / "rewriter-files.json")
        if (
            not isinstance(record, dict)
            or record.get("schema") != 1
            or record.get("model") != model_id
            or record.get("revision") != revision
            or record.get("quantization") != "nf4-double"
        ):
            raise ValueError("モデルの登録が一致しません")
        files = record.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("ファイルの記録がありません")
        seen = set()
        for item in files:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                raise ValueError("ファイルの記録が不正です")
            relative, size = Path(item["path"]), item.get("size")
            if relative.is_absolute() or relative in seen or type(size) is not int or size <= 0:
                raise ValueError("ファイルの記録が不正です")
            seen.add(relative)
            path = inside(model, model / relative)
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"不足またはサイズ不一致: {relative}")
            if verify_hashes and file_hash(path) != item.get("sha256"):
                raise ValueError(f"SHA-256不一致: {relative}")
        if not {Path(name) for name in ("config.json", "tokenizer.json", "system_prompt.txt")}.issubset(seen):
            raise ValueError("設定・トークナイザー・公式指示文が不足しています")
        if not any(path.suffix == ".safetensors" for path in seen):
            raise ValueError("量子化した重みがありません")
        config = read_json(model / "config.json")
        quantization = config.get("quantization_config", {})
        if (
            config.get("model_type") != ("qwen3_5" if editing else "qwen3_5_text")
            or quantization.get("quant_method") != "bitsandbytes"
            or quantization.get("load_in_4bit", quantization.get("_load_in_4bit")) is not True
            or quantization.get("bnb_4bit_quant_type") != "nf4"
            or quantization.get("bnb_4bit_use_double_quant") is not True
        ):
            raise ValueError("要求されたNF4・二重量子化モデルではありません")
        if editing:
            processor_file = next(
                (name for name in ("processor_config.json", "preprocessor_config.json") if Path(name) in seen),
                None,
            )
            if processor_file is None:
                raise ValueError("画像入力の設定がありません")
            processor_config = read_json(model / processor_file)
            image_config = (
                processor_config.get("image_processor", {})
                if processor_file == "processor_config.json"
                else processor_config
            )
            if not image_config.get("image_processor_type"):
                raise ValueError("画像入力の設定が不正です")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise QwenImage21Error(
            f"プロンプト書き換えモデルが未導入または不完全です（{exc}）。"
            f"{EDIT_SETUP_COMMAND if editing else SETUP_COMMAND} "
            "を実行するか、書き換えをOFFにしてください。"
        ) from exc
    return {**record, "path": str(model)}


def rewriter_status(root: Path, *, editing=False) -> str:
    label = "編集補助" if editing else "書き換え"
    try:
        rewriter_manifest(root, editing=editing)
    except QwenImage21Error:
        return f"{label}: 未導入。追加するには {EDIT_SETUP_COMMAND if editing else SETUP_COMMAND}"
    return f"{label}: 4bit NF4・二重量子化を導入済み。"


def parse_rewrite(text: str, *, editing=False) -> dict:
    """Accept the official answer with or without thinking/JSON code fences."""
    answer = text.rsplit("</think>", 1)[-1].strip()
    if answer.startswith("```") and answer.endswith("```"):
        lines = answer.splitlines()
        if lines[0].strip() in {"```json", "```"}:
            answer = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(answer)
        prompt = result["rewritten_prompt"]
        ratio = result["wh_ratio"]
        follow = result["ratio_follow"] if editing else ""
        ratio_valid = ratio in RATIOS
        if editing:
            # The I2I system prompt also permits ratios such as 18:39, 9:20,
            # and custom output dimensions; it does not use the T2I shortlist.
            ratio_valid = (
                isinstance(ratio, str)
                and re.fullmatch(r"[0-9]{1,5}(?:\.[0-9]{1,4})?:[0-9]{1,5}(?:\.[0-9]{1,4})?", ratio) is not None
                and all(float(part) > 0 for part in ratio.split(":"))
            )
            ratio_valid = (ratio_valid and follow == "") or (
                ratio == "" and isinstance(follow, str) and re.fullmatch(r"<image(?:[1-9]|10)>", follow) is not None
            )
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000 or not ratio_valid:
            raise ValueError
    except (ValueError, TypeError, KeyError) as exc:
        raise QwenImage21Error(
            "書き換えモデルから有効なプロンプトを取得できませんでした。入力を短くするか、書き換えをOFFにして再実行してください。"
        ) from exc
    return {"rewritten_prompt": prompt.strip(), "wh_ratio": ratio, **({"ratio_follow": follow} if editing else {})}


def quantization_config(*, editing=False):
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # The output head is untied and large (248320 x 4096). Quantize it too.
        llm_int8_skip_modules=["model.visual"] if editing else [],
    )


def check_quantized_model(model, *, editing=False) -> int:
    from bitsandbytes.nn import Linear4bit

    layers = sum(isinstance(layer, Linear4bit) for layer in model.modules())
    if (
        not layers
        or not getattr(model, "is_loaded_in_4bit", False)
        or not isinstance(model.lm_head, Linear4bit)
        or model.config.model_type != ("qwen3_5" if editing else "qwen3_5_text")
    ):
        raise RuntimeError("書き換えモデルの4bit量子化と出力層を確認できません。")
    if editing:
        import torch

        visual = getattr(getattr(model, "model", None), "visual", None)
        if (
            visual is None
            or any(isinstance(layer, Linear4bit) for layer in visual.modules())
            or any(parameter.dtype != torch.bfloat16 for parameter in visual.parameters())
        ):
            raise RuntimeError("編集補助モデルの画像認識部分をBF16で確認できません。再セットアップしてください。")
    return layers


def _reference_ids(text):
    # Quoted image text such as 「画像9」 is content, not an input-image reference.
    unquoted = _LITERALS.sub(" ", text)
    numbers = (next(value for value in group if value) for group in _REFERENCES.findall(unquoted))
    return {f"<image{number.lstrip('0') or '0'}>" for number in numbers}


def generation_eos_ids(tokenizer, model):
    """Stop on the chat turn terminator as well as the checkpoint's document EOS."""
    configured = model.generation_config.eos_token_id
    candidates = [tokenizer.eos_token_id, *(configured if isinstance(configured, (tuple, list)) else [configured])]
    tokens = list(dict.fromkeys(token for token in candidates if type(token) is int and token >= 0))
    if not tokens:
        raise QwenImage21Error("書き換えモデルの終了トークンを確認できません。再セットアップしてください。")
    return tokens


def validate_edit_rewrite(result, prompt, image_count):
    """Reject lost literal text or changed reference IDs; retain original constraints."""
    rewritten = result["rewritten_prompt"]
    literals = _LITERALS.findall(prompt)
    refs = _reference_ids(prompt)
    generated_refs = _reference_ids(rewritten)
    allowed = {f"<image{i}>" for i in range(1, image_count + 1)}
    # The official I2I prompt explicitly omits tags for a single input. Only
    # this unambiguous, ratio-confirmed case can safely regain its explicit ID.
    if image_count == 1 and refs == {"<image1>"} and not generated_refs and result.get("ratio_follow") == "<image1>":
        rewritten = "<image1>: " + rewritten
        generated_refs = {"<image1>"}
    if not refs.issubset(generated_refs) or not generated_refs.issubset(allowed):
        raise QwenImage21Error("編集補助が参照番号を変更しました。補助をOFFにして再実行してください。")
    if result.get("ratio_follow") and result["ratio_follow"] not in allowed:
        raise QwenImage21Error("編集補助が存在しない参照画像を指定しました。")
    if any(next(value for value in group if value) not in rewritten for group in literals):
        raise QwenImage21Error("編集補助が指定した文字を変更しました。補助をOFFにして再実行してください。")
    effective = rewritten + "\n\nOriginal editing instruction and preservation constraints (authoritative):\n" + prompt
    if len(effective) > 12000:
        raise QwenImage21Error("編集補助後の指示が長すぎます。元の指示を短くしてください。")
    return {**result, "rewritten_prompt": effective}


def rewrite_prompt(
    root: Path, prompt: str, width: int, height: int, seed: int, check_cancel, progress, *, images=()
) -> dict:
    """Run under the image job's GPU lease, then release all rewriter tensors."""
    editing = bool(images)
    if editing:
        if not 1 <= len(images) <= 10:
            raise QwenImage21Error("編集補助には1〜10枚の参照画像が必要です。")
        allowed = {f"<image{i}>" for i in range(1, len(images) + 1)}
        if not _reference_ids(prompt).issubset(allowed):
            raise QwenImage21Error("編集指示の参照番号に対応する画像がありません。参照画像と番号を確認してください。")
    entry = rewriter_manifest(root, editing=editing)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    from modules_forge.qwen_image21_environment import validate_running_versions

    validate_running_versions()
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoProcessor,
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
    )

    class CancelRewrite(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check_cancel()
            return False

    started = time.monotonic()
    model = inputs = output = generated = None
    try:
        check_cancel()
        progress("書き換えモデルを4bitで読み込み中")
        model_class = AutoModelForImageTextToText if editing else AutoModelForCausalLM
        model = model_class.from_pretrained(
            entry["path"],
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        ).eval()
        layers = check_quantized_model(model, editing=editing)
        check_cancel()
        processor = (AutoProcessor if editing else AutoTokenizer).from_pretrained(
            entry["path"], local_files_only=True, trust_remote_code=False
        )
        tokenizer = processor.tokenizer if editing else processor
        system = (Path(entry["path"]) / "system_prompt.txt").read_text(encoding="utf-8").strip()
        ratio = min(
            RATIOS,
            key=lambda value: abs(math.log((int(value.split(":")[0]) / int(value.split(":")[1])) / (width / height))),
        )
        if editing:
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system + "\nPreserve quoted text verbatim and all reference image numbers. "
                            "Keep explicit image references and indices (Image 1, 画像1, 1枚目, <image1>, etc.) "
                            "as <imageN> tags even for a single image; quoted image text is not a reference. "
                            "Keep every explicit preservation constraint and do not introduce extra edits.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        *({"type": "image", "image": image.convert("RGB")} for image in images),
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            ).to(model.device)
        else:
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{prompt}\n\nOutput aspect ratio: {ratio}."},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
        check_cancel()
        progress("プロンプトを書き換え中")
        generation_started = time.monotonic()
        eos_tokens = generation_eos_ids(tokenizer, model)
        # Isolate sampling from image generation and repeated jobs in this worker.
        with torch.inference_mode(), torch.random.fork_rng(devices=[0]):
            torch.manual_seed(seed)
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                eos_token_id=eos_tokens,
                stopping_criteria=StoppingCriteriaList([CancelRewrite()]),
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            )
        check_cancel()
        generated = output[0, inputs["input_ids"].shape[1] :]
        result = parse_rewrite(tokenizer.decode(generated, skip_special_tokens=True), editing=editing)
        if editing:
            result = validate_edit_rewrite(result, prompt, len(images))
        result.update(
            {
                "enabled": True,
                "applied": True,
                "model": entry["model"],
                "revision": entry["revision"],
                "quantization": "nf4-double",
                "linear4bit_layers": layers,
                "thinking": False,
                "requested_ratio": ratio,
                "size_changed": False,
                "generated_tokens": int(generated.shape[0]),
                "stop_token_id": int(generated[-1]),
                "eos_token_ids": eos_tokens,
                "load_seconds": round(generation_started - started, 3),
                "generation_seconds": round(time.monotonic() - generation_started, 3),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        )
        generated = None
        return result
    finally:
        model = inputs = output = generated = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
