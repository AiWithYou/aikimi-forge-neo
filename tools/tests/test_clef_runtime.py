"""Real tiny tensors verify CPU row gathering without downloading any model."""

from types import SimpleNamespace

import pytest
import torch

from modules_forge.clef.runtime import CPUEmbedding, LexicalRows, encode_complete, load_options


def test_lexical_rows_keep_dense_values_and_index_on_cpu():
    weight = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    rows = LexicalRows(weight, device="cpu", dtype=torch.bfloat16)
    tokens = torch.tensor([4, 1, 4])
    assert torch.equal(rows[tokens], weight[tokens])
    assert torch.equal(rows[tokens].mean(0), weight[tokens].mean(0))
    assert rows.weight.device.type == "cpu"


def test_cpu_embedding_does_not_transfer_entire_vocabulary():
    weight = torch.arange(40, dtype=torch.bfloat16).reshape(10, 4)
    embedding = CPUEmbedding(weight, device="cpu", dtype=torch.bfloat16)
    tokens = torch.tensor([[3, 2], [1, 9]])
    assert torch.equal(embedding(tokens), weight[tokens])
    assert embedding.weight.device.type == "cpu"


def test_meta_weights_are_refused():
    with pytest.raises(ValueError, match="CPU"):
        LexicalRows(torch.empty(6, 4, device="meta"), device="cpu", dtype=torch.bfloat16)


def test_quantization_excludes_vision_and_output_and_maps_base_class():
    flash = load_options("flash-int8")
    normal = load_options("clef-16gb")
    assert flash["quantization"]["load_in_8bit"]
    assert "visual" in flash["quantization"]["llm_int8_skip_modules"]
    assert normal["quantization"]["load_in_4bit"]
    assert normal["quantization"]["bnb_4bit_use_double_quant"]
    assert normal["device_map"]["language_model.embed_tokens"] == "cpu"
    assert "" not in normal["device_map"]  # A recursive parent hook would move embeddings to CUDA first.
    assert all(not k.startswith("model.") for k in normal["device_map"])
    assert "device_map" not in load_options("flash-bf16").get("quantization", {})


def test_input_overflow_is_reported_without_silent_truncation():
    calls = []

    def encode(tokenizer, record, **kw):
        calls.append(kw)
        return SimpleNamespace(input_ids=tuple(range(50)))

    official = SimpleNamespace(encode_record=encode)
    processor = SimpleNamespace(tokenizer=object())
    with pytest.raises(ValueError, match="50"):
        encode_complete(official, processor, {"state": "long", "questions": {}}, 40)
    assert calls[0]["max_length"] > 50
    assert len(encode_complete(official, processor, {}, 60).input_ids) == 50


def test_requested_pixel_budget_reaches_real_image_preprocessing():
    from PIL import Image
    from transformers import Qwen2VLImageProcessor

    from modules_forge.clef.runtime import image_measurement, image_options

    image_processor = Qwen2VLImageProcessor(
        size={"shortest_edge": 65536, "longest_edge": 16777216}, patch_size=16, merge_size=2
    )
    processor = SimpleNamespace(image_processor=image_processor)
    image = Image.new("RGB", (1344, 1728))
    counts = []
    for budget in (262144, 1048576):
        options = image_options(processor, budget)
        data = image_processor.preprocess(image, return_tensors="pt", **options["images_kwargs"])
        grid = data["image_grid_thw"][0]
        pixels = int(grid[1]) * int(grid[2]) * 16**2
        assert pixels <= budget
        size, image_tokens = image_measurement(processor, SimpleNamespace(media=data))
        assert size[0] * size[1] == pixels
        assert image_tokens == pixels // 32**2
        counts.append(pixels)
    assert counts[1] > counts[0] * 3
    assert image_processor.size["longest_edge"] == 16777216
    assert image_measurement(processor, SimpleNamespace(media=None)) == (None, 0)


def test_conditional_checkpoint_loads_base_without_losing_backbone(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

    from modules_forge.clef.runtime import stream_weights

    config = Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 128,
            "layer_types": ["full_attention"],
            "tie_word_embeddings": False,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_heads": 4,
            "out_hidden_size": 64,
            "num_position_embeddings": 16,
        },
    )
    original = Qwen3_5ForConditionalGeneration(config)
    original.save_pretrained(tmp_path)
    with stream_weights():
        base, info = Qwen3_5Model.from_pretrained(tmp_path, output_loading_info=True)
    assert not info["missing_keys"]
    assert set(info["unexpected_keys"]) == {"lm_head.weight"}
    assert torch.equal(base.language_model.embed_tokens.weight, original.model.language_model.embed_tokens.weight)


def test_stream_weights_reads_lazy_slices_with_pread_and_restores_on_failure(tmp_path, monkeypatch):
    import transformers.modeling_utils as loading
    from safetensors.torch import save_file

    from modules_forge.clef.runtime import stream_weights

    expected = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    path = tmp_path / "weights.safetensors"
    save_file({"weight": expected}, path)
    original = loading.safe_open
    backends = []

    def record_open(*args, **kwargs):
        backends.append(kwargs.get("backend"))
        return original(*args, **kwargs)

    monkeypatch.setattr(loading, "safe_open", record_open)
    with pytest.raises(RuntimeError, match="load interrupted"):
        with stream_weights():
            with loading.safe_open(path, framework="pt", device="cpu") as reader:
                assert torch.equal(reader.get_slice("weight")[...], expected)
            raise RuntimeError("load interrupted")
    assert backends == ["pread"]
    assert loading.safe_open is record_open


def test_source_tensor_is_read_once_with_pread_and_survives_file_removal(tmp_path, monkeypatch):
    import json

    import safetensors
    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    expected = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    path = tmp_path / "weights.safetensors"
    save_file({"lm_head.weight": expected}, path)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": path.name}}), encoding="utf-8"
    )
    original = safetensors.safe_open
    backends = []

    def record_open(*args, **kwargs):
        backends.append(kwargs.get("backend"))
        return original(*args, **kwargs)

    monkeypatch.setattr(safetensors, "safe_open", record_open)
    weight = tensor_from_source(tmp_path, "lm_head.weight")
    assert backends == ["pread"]
    path.unlink()
    assert torch.equal(weight, expected)
