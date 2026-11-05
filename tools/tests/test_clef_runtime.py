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
    import json

    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5Model

    from modules_forge.clef.runtime import safe_pretrained_options, stream_weights

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
    marker = tmp_path / "dynamic-code-ran"
    (tmp_path / "evil.py").write_text(f"open({str(marker)!r}, 'w').write('bad')", encoding="utf-8")
    config_path = tmp_path / "config.json"
    data = json.loads(config_path.read_text(encoding="utf-8"))
    data["auto_map"] = {"AutoModel": "evil.Untrusted"}
    config_path.write_text(json.dumps(data), encoding="utf-8")
    with stream_weights():
        base, info = Qwen3_5Model.from_pretrained(tmp_path, output_loading_info=True, **safe_pretrained_options()[0])
    assert not marker.exists()
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


@pytest.mark.parametrize("canonical", ["language_model.embed_tokens.weight", "visual.patch_embed.proj.weight"])
@pytest.mark.parametrize("indexed_prefix", ["", "model."])
@pytest.mark.parametrize("requested_prefix", ["", "model."])
def test_source_tensor_resolves_known_prefix_uniquely(tmp_path, canonical, indexed_prefix, requested_prefix):
    import json

    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    name = indexed_prefix + canonical
    expected = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    save_file({name: expected}, tmp_path / "weights.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "weights.safetensors"}}), encoding="utf-8"
    )
    assert torch.equal(tensor_from_source(tmp_path, requested_prefix + canonical), expected)


@pytest.mark.parametrize("same_shard", [True, False])
def test_source_tensor_rejects_duplicate_prefix_aliases_before_open(tmp_path, monkeypatch, same_shard):
    import json

    import safetensors

    from modules_forge.clef.runtime import tensor_from_source

    canonical = "language_model.embed_tokens.weight"
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {canonical: "one.safetensors", "model." + canonical:
                                   "one.safetensors" if same_shard else "two.safetensors"}}), encoding="utf-8"
    )
    monkeypatch.setattr(safetensors, "safe_open", lambda *_args, **_kwargs: pytest.fail("ambiguous shard opened"))
    with pytest.raises(ValueError, match="重複"):
        tensor_from_source(tmp_path, canonical)


def test_source_tensor_does_not_resolve_unknown_prefix(tmp_path):
    import json

    from modules_forge.clef.runtime import tensor_from_source

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"wrapper.lm_head.weight": "weights.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="テンソル"):
        tensor_from_source(tmp_path, "lm_head.weight")


def test_source_tensor_rejects_prefixed_alias_with_unsafe_shard_path(tmp_path):
    import json

    from modules_forge.clef.runtime import tensor_from_source

    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.embed_tokens.weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="シャード"):
        tensor_from_source(tmp_path, "language_model.embed_tokens.weight")


def test_source_tensor_rejects_shard_outside_model_directory(tmp_path):
    import json

    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    source = tmp_path / "model"
    source.mkdir()
    save_file({"lm_head.weight": torch.zeros(2, 2)}, tmp_path / "outside.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "../outside.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="シャード"):
        tensor_from_source(source, "lm_head.weight")


def test_source_tensor_rejects_linked_shard(tmp_path, monkeypatch):
    import json
    import stat
    from pathlib import Path

    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    shard = tmp_path / "shard.safetensors"
    save_file({"lm_head.weight": torch.zeros(2, 2)}, shard)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"lm_head.weight": "shard.safetensors"}}), encoding="utf-8"
    )
    original = Path.lstat
    monkeypatch.setattr(
        Path, "lstat", lambda path: SimpleNamespace(st_mode=stat.S_IFLNK) if path == shard else original(path)
    )
    with pytest.raises(ValueError, match="シャード"):
        tensor_from_source(tmp_path, "lm_head.weight")


def test_source_tensor_rejects_linked_shard_index(tmp_path, monkeypatch):
    import json
    import stat
    from pathlib import Path

    from safetensors.torch import save_file

    from modules_forge.clef.runtime import tensor_from_source

    save_file({"lm_head.weight": torch.zeros(2, 2)}, tmp_path / "shard.safetensors")
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"lm_head.weight": "shard.safetensors"}}), encoding="utf-8")
    original = Path.lstat
    monkeypatch.setattr(
        Path, "lstat", lambda path: SimpleNamespace(st_mode=stat.S_IFLNK) if path == index else original(path)
    )
    with pytest.raises(ValueError, match="シャード"):
        tensor_from_source(tmp_path, "lm_head.weight")
