"""Keep the official joint head while moving only vocabulary rows over PCIe."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
from contextlib import contextmanager
from functools import partial

import torch

from .bundle import bundle_manifest, read_bundle
from .core import PREPROCESSING_VERSION, PROFILES


class LexicalRows:
    def __init__(self, weight, *, device, dtype):
        if weight.device.type != "cpu" or weight.is_meta:
            raise ValueError("語彙行列には実体のあるCPUテンソルが必要です。")
        self.weight, self.device, self.dtype = weight, device, dtype

    def __getitem__(self, token_ids):
        shape = (*token_ids.shape, self.weight.shape[1])
        return self.weight.index_select(0, token_ids.reshape(-1).cpu()).reshape(shape).to(self.device, dtype=self.dtype)


class CPUEmbedding(torch.nn.Module):
    def __init__(self, weight, *, device, dtype):
        super().__init__()
        self.rows = LexicalRows(weight, device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(weight, requires_grad=False)

    def forward(self, tokens):
        return self.rows[tokens]


def load_options(profile):
    settings = PROFILES[profile]
    result = {"device_map": {"": 0}}
    if settings["cpu_embeddings"]:
        result["device_map"] = {
            "visual": 0,
            "language_model.embed_tokens": "cpu",
            "language_model.layers": 0,
            "language_model.norm": 0,
            "language_model.rotary_emb": 0,
        }
    if settings["precision"] != "bf16":
        quantization = {
            "llm_int8_skip_modules": ["visual", "lm_head"],
            "llm_int8_enable_fp32_cpu_offload": settings["cpu_embeddings"],
        }
        if settings["precision"] == "int8":
            quantization["load_in_8bit"] = True
        else:
            quantization.update(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        result["quantization"] = quantization
    return result


def image_options(processor, max_pixels):
    # Transformers 5 applies per-call pixel bounds through size, rather than
    # the legacy max_pixels constructor argument. Never change processor defaults.
    return {
        "images_kwargs": {
            "size": {
                "shortest_edge": min(processor.image_processor.size["shortest_edge"], max_pixels),
                "longest_edge": max_pixels,
            }
        }
    }


def image_measurement(processor, encoded):
    grid = (encoded.media or {}).get("image_grid_thw")
    if grid is None:
        return None, 0
    frames, height, width = map(int, grid[0])
    image_processor = processor.image_processor
    patch, merge = image_processor.patch_size, image_processor.merge_size
    return [width * patch, height * patch], frames * height * width // merge**2


def encode_complete(official, processor, record, max_length):
    # The release silently truncates state. Encode once without truncation and
    # refuse overflow before collating or allocating decoder activations.
    encoded = official.encode_record(processor.tokenizer, record, processor=processor, max_length=1_000_000)
    if len(encoded.input_ids) > max_length:
        raise ValueError(
            f"入力は{len(encoded.input_ids)}トークンです。上限{max_length}を超えます。補足や項目を短くするか処理解像度を下げてください。"
        )
    return encoded


def tensor_from_source(source, key):
    from safetensors import safe_open

    index = json.loads((source / "model.safetensors.index.json").read_text())
    with safe_open(source / index["weight_map"][key], framework="pt", device="cpu", backend="pread") as stream:
        return stream.get_tensor(key)


@contextmanager
def stream_weights():
    import transformers.modeling_utils as loading

    # Windows reserves commit for every copy-on-write mmap, including unread
    # pages. Keep TF's lazy tensor loading without retaining all shard maps.
    original = loading.safe_open
    loading.safe_open = partial(original, backend="pread")
    try:
        yield
    finally:
        loading.safe_open = original


class Runner:
    @classmethod
    def from_directory(cls, directory, profile):
        return cls(None, profile, directory=directory)

    def __init__(self, root, profile, *, directory=None):
        from safetensors.torch import load_file
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3_5Model

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("ClefにはBF16対応のNVIDIA CUDA GPUが必要です。")
        self.settings, self.profile = PROFILES[profile], profile
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if total < self.settings["min_vram_gib"]:
            raise RuntimeError(
                f"{self.settings['label']}のGPUメモリが不足しています。16GB設定またはFlashを選んでください。"
            )
        source, _ = read_bundle(directory, profile) if directory is not None else bundle_manifest(root, profile)
        spec = importlib.util.spec_from_file_location(
            "clef_release_" + self.settings["model"].replace("-", "_"), source / "joint_schema_model.py"
        )
        self.official = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.official
        spec.loader.exec_module(self.official)
        options = load_options(profile)
        saved = source if self.settings["precision"] != "bf16" else None
        if "quantization" in options:
            options["quantization_config"] = BitsAndBytesConfig(**options.pop("quantization"))
        print(f"{self.settings['label']}: {'保存済み' if saved else '公式重み'}を読み込み中", flush=True)  # noqa: T201
        with stream_weights():
            self.backbone, loading = Qwen3_5Model.from_pretrained(
                saved or source,
                dtype=torch.bfloat16,
                local_files_only=True,
                attn_implementation="sdpa",
                output_loading_info=True,
                **options,
            )
        expected_extra = {"lm_head.weight"}
        if loading["missing_keys"] or set(loading["unexpected_keys"]) != expected_extra:
            raise RuntimeError(f"Clef本体のキーが一致しません: {loading}")
        self.backbone.eval()
        self.backbone.config.use_cache = False
        if self.settings["cpu_embeddings"]:
            # Read the bundle's BF16 rows directly; CPU offload may cast to FP32.
            # Discard the old embedding together with its hook. Detaching a
            # hook can temporarily restore the entire vocabulary onto CUDA.
            embedding = tensor_from_source(source, "language_model.embed_tokens.weight")
            self.backbone.language_model.embed_tokens = CPUEmbedding(embedding, device="cuda:0", dtype=torch.bfloat16)
            gc.collect()
        self.lexical = LexicalRows(tensor_from_source(source, "lm_head.weight"), device="cuda:0", dtype=torch.bfloat16)
        self.head = self.official.JointSchemaHead(**json.loads((source / "joint_head_config.json").read_text()))
        self.head.load_state_dict(load_file(source / "joint_head.safetensors", backend="pread"), strict=True)
        self.head = self.head.to(device="cuda:0", dtype=torch.bfloat16).eval()
        self.processor = AutoProcessor.from_pretrained(source, local_files_only=True)
        self.source = source
        self.root, self.saved = root, saved

    @torch.inference_mode()
    def decide(self, request, image=None):
        record = {
            "model": self.settings["model"],
            "state": request["state"],
            "questions": request["questions"],
            "media_kwargs": image_options(self.processor, request["max_pixels"]),
        }
        if image is not None:
            record["images"] = [image]
        encoded = encode_complete(self.official, self.processor, record, request["max_length"])
        processing_size, image_tokens = image_measurement(self.processor, encoded)
        batch = self.official.collate_records([encoded], self.processor.tokenizer.pad_token_id, torch.device("cuda:0"))
        torch.cuda.reset_peak_memory_stats()
        media = batch.get("media") or {}
        model = self.backbone if media else self.backbone.language_model
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
            return_dict=True,
            **media,
        )
        logits = self.head(
            outputs.last_hidden_state, batch["input_ids"], batch["attention_mask"], batch["records"], self.lexical
        )[0]
        answers = {
            q.question_id: self.official.systemone_answer(
                request["questions"][q.question_id],
                dict(zip(q.option_ids, logit.float().softmax(-1).tolist(), strict=True)),
            )
            for q, logit in zip(encoded.questions, logits, strict=True)
        }
        torch.cuda.synchronize()
        return {
            "model": self.settings["model"],
            "answers": answers,
            "preprocessing": PREPROCESSING_VERSION,
            "processing_size": processing_size,
            "usage": {"input_tokens": len(encoded.input_ids), "image_tokens": image_tokens, "output_tokens": 0},
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        }

    def close(self):
        self.backbone = self.head = self.lexical = self.processor = None
        gc.collect()
        torch.cuda.empty_cache()
