"""Optional, local NF4 text-only prompt rewriting for Qwen Image 2.1."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

from .core import QwenImage21Error, inside, read_json

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
MODEL_REVISION = "f3ed7985c788ad75b3ab7223e0c4c51e2a43545b"
DIRECTORY = "prompt-rewriter-nf4"
SETUP_COMMAND = "aikimi-qwen-image21-setup.bat --prompt-rewriter-only"
MAX_NEW_TOKENS = 4096
RATIOS = ("1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "1:2", "2:1", "21:9", "9:21", "4:5", "5:4", "3:1", "1:3")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rewriter_manifest(root: Path, *, verify_hashes: bool = False) -> dict:
    """Validate only local files; an OFF request never needs this optional model."""
    model = inside(root, root / DIRECTORY)
    try:
        record = read_json(model / "rewriter-files.json")
        if (
            not isinstance(record, dict)
            or record.get("schema") != 1
            or record.get("model") != MODEL_ID
            or record.get("revision") != MODEL_REVISION
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
            config.get("model_type") != "qwen3_5_text"
            or quantization.get("quant_method") != "bitsandbytes"
            or quantization.get("load_in_4bit", quantization.get("_load_in_4bit")) is not True
            or quantization.get("bnb_4bit_quant_type") != "nf4"
            or quantization.get("bnb_4bit_use_double_quant") is not True
        ):
            raise ValueError("テキスト専用NF4・二重量子化モデルではありません")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise QwenImage21Error(
            f"プロンプト書き換えモデルが未導入または不完全です（{exc}）。{SETUP_COMMAND} を実行するか、書き換えをOFFにしてください。"
        ) from exc
    return {**record, "path": str(model)}


def rewriter_status(root: Path) -> str:
    try:
        rewriter_manifest(root)
    except QwenImage21Error:
        return f"書き換え: 未導入。追加するには {SETUP_COMMAND}"
    return "書き換え: 4bit NF4・二重量子化を導入済み。"


def parse_rewrite(text: str) -> dict:
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
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 12000 or ratio not in RATIOS:
            raise ValueError
    except (ValueError, TypeError, KeyError) as exc:
        raise QwenImage21Error(
            "書き換えモデルから有効なプロンプトを取得できませんでした。入力を短くするか、書き換えをOFFにして再実行してください。"
        ) from exc
    return {"rewritten_prompt": prompt.strip(), "wh_ratio": ratio}


def quantization_config():
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        # The output head is untied and large (248320 x 4096). Quantize it too.
        llm_int8_skip_modules=[],
    )


def check_quantized_model(model) -> int:
    from bitsandbytes.nn import Linear4bit

    layers = sum(isinstance(layer, Linear4bit) for layer in model.modules())
    if (
        not layers
        or not getattr(model, "is_loaded_in_4bit", False)
        or not isinstance(model.lm_head, Linear4bit)
        or model.config.model_type != "qwen3_5_text"
    ):
        raise RuntimeError("書き換えモデルのテキスト専用4bit量子化を確認できません。")
    return layers


def rewrite_prompt(root: Path, prompt: str, width: int, height: int, seed: int, check_cancel, progress) -> dict:
    """Run under the image job's GPU lease, then release all rewriter tensors."""
    entry = rewriter_manifest(root)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    from modules_forge.qwen_image21_environment import validate_running_versions

    validate_running_versions()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    class CancelRewrite(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check_cancel()
            return False

    started = time.monotonic()
    model = inputs = output = generated = None
    try:
        check_cancel()
        progress("書き換えモデルを4bitで読み込み中")
        model = AutoModelForCausalLM.from_pretrained(
            entry["path"],
            dtype=torch.bfloat16,
            device_map={"": "cuda:0"},
            local_files_only=True,
            use_safetensors=True,
            trust_remote_code=False,
        ).eval()
        layers = check_quantized_model(model)
        tokenizer = AutoTokenizer.from_pretrained(entry["path"], local_files_only=True, trust_remote_code=False)
        system = (Path(entry["path"]) / "system_prompt.txt").read_text(encoding="utf-8").strip()
        ratio = min(
            RATIOS,
            key=lambda value: abs(math.log((int(value.split(":")[0]) / int(value.split(":")[1])) / (width / height))),
        )
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
                stopping_criteria=StoppingCriteriaList([CancelRewrite()]),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        check_cancel()
        generated = output[0, inputs["input_ids"].shape[1] :]
        result = parse_rewrite(tokenizer.decode(generated, skip_special_tokens=True))
        result.update(
            {
                "enabled": True,
                "applied": True,
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "quantization": "nf4-double",
                "linear4bit_layers": layers,
                "thinking": False,
                "requested_ratio": ratio,
                "size_changed": False,
                "generated_tokens": int(generated.shape[0]),
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
