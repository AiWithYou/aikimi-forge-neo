"""Local H3 checkpoints are selected by use, not by their filenames."""

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from modules_forge import local_assets, studio_checkpoints
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_bridge import MODEL_FILES, H3Request, build_workflow, model_file_status
from modules_forge.minimax_h3_images import H3ImageRequest, build_image_workflow


@pytest.fixture(autouse=True)
def isolate_hash_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(local_assets, "HASH_CACHE", tmp_path / "hashes.json")


def checkpoint(path, *, storage="int8", adaln=8, broken=None, extra_w4=False, extra_int8=False, codebooks=None):
    """Tiny safetensors header with the shapes of all H3 block layers."""
    header = {}
    data = bytearray()
    codebooks = codebooks or {}

    def tensor(name, dtype, shape, payload=b""):
        start = len(data)
        data.extend(payload)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [start, len(data)]}

    for name, shape in {
        "audio_patch_proj.weight": [5376, 32], "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [1025 if adaln == 8 else 4097, adaln],
    }.items():
        tensor(name, "F32", shape)
    layers = {}
    for block in range(50):
        tensor(f"blocks.{block}.adaln_proj.linear.weight", "F16", [96768, adaln])
        for layer, out_size, in_size in (
            ("attn.out_proj", 5376, 7168), ("attn.qkv_proj", 21504, 5376),
            ("mlp.fc1", 28672, 5376), ("mlp.fc2", 5376, 14336),
        ):
            stem = f"blocks.{block}.{layer}"
            tensor(stem + ".weight", "I8" if storage != "bf16" else "BF16", [out_size, in_size // 2 if storage.startswith("w4a8") else in_size])
            if storage.startswith("int8"):
                tensor(stem + ".weight_scale", "F32", [out_size, 1])
                config = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
                if storage == "int8-legacy":
                    layers[stem] = config
                else:
                    payload = json.dumps(config).encode()
                    tensor(stem + ".comfy_quant", "U8", [len(payload)], payload)
            elif storage.startswith("w4a8"):
                tensor(stem + ".weight_s_rel", "F8_E4M3", [out_size, in_size // 16])
                tensor(stem + ".weight_s_channel", "F32", [out_size])
                tensor(stem + ".weight_codebook", "F32", [16],
                       struct.pack("<16f", *codebooks.get(stem, range(16))))
                config = {"format": "asym_w4a8_int8", "group_size": 16}
                if storage == "w4a8-comfy":
                    payload = json.dumps(config).encode()
                    tensor(stem + ".comfy_quant", "U8", [len(payload)], payload)
                else:
                    layers[stem] = config
    if extra_w4:
        stem = "blocks.12.attn.out_proj"
        header.pop(stem + ".comfy_quant", None)
        header.pop(stem + ".weight_scale", None)
        tensor(stem + ".weight", "I8", [5376, 3584])
        tensor(stem + ".weight_s_rel", "F8_E4M3", [5376, 448])
        tensor(stem + ".weight_s_channel", "F32", [5376])
        tensor(stem + ".weight_codebook", "F32", [16],
               struct.pack("<16f", *codebooks.get(stem, range(16))))
        layers[stem] = {"format": "asym_w4a8_int8", "group_size": 16}
    if extra_int8:
        stem = "blocks.12.attn.out_proj"
        header.pop(stem + ".comfy_quant", None)
        tensor(stem + ".weight", "I8", [5376, 7168])
        tensor(stem + ".weight_scale", "F32", [5376, 1])
        layers[stem] = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    if layers:
        header["__metadata__"] = {"_quantization_metadata": json.dumps({"layers": layers})}
    if broken:
        broken(header)
    encoded = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)
    return str(path.resolve())


@pytest.mark.parametrize("storage", ["int8", "int8-legacy", "w4a8", "w4a8-comfy", "bf16"])
def test_h3_accepts_renamed_full_model_by_tensors_and_detects_storage(tmp_path, storage):
    path = checkpoint(tmp_path / "arbitrary_fused_turbo.safetensors", storage=storage)
    assert studio_checkpoints.h3_compatible(Path(path))
    assert studio_checkpoints.h3_variant(path) == ("w4a8" if storage.startswith("w4a8") else "base")
    renamed = Path(path).with_name("renamed.safetensors")
    Path(path).rename(renamed)
    assert studio_checkpoints.h3_compatible(renamed)
    assert studio_checkpoints.h3_variant(str(renamed)) == ("w4a8" if storage.startswith("w4a8") else "base")


@pytest.mark.parametrize("broken", [
    lambda h: h.pop("blocks.49.attn.out_proj.weight"),
    lambda h: h["blocks.0.attn.out_proj.weight"].update(shape=[5376, 3584]),
    lambda h: h.pop("blocks.0.attn.out_proj.weight_scale"),
    lambda h: h.pop("blocks.0.attn.out_proj.comfy_quant"),
])
def test_h3_rejects_incomplete_or_malformed_weights(tmp_path, broken):
    path = checkpoint(tmp_path / "fl2va.safetensors", broken=broken)
    assert not studio_checkpoints.h3_compatible(Path(path))


def test_h3_rejects_p_h3_and_lora(tmp_path):
    assert not studio_checkpoints.h3_compatible(Path(checkpoint(tmp_path / "p.safetensors", adaln=16)))
    path = checkpoint(tmp_path / "lora.safetensors", broken=lambda h: h.pop("blocks.49.adaln_proj.linear.weight"))
    assert not studio_checkpoints.h3_compatible(Path(path))


def test_mixed_w4a8_in_an_interior_block_still_requires_its_runtime(tmp_path):
    path = checkpoint(tmp_path / "mixed.safetensors", extra_w4=True)
    assert studio_checkpoints.h3_variant(path) == "w4a8"
    bad = checkpoint(tmp_path / "bad.safetensors", extra_w4=True,
                     broken=lambda h: h.pop("blocks.12.attn.out_proj.weight_s_rel"))
    assert not studio_checkpoints.h3_compatible(Path(bad))


def test_unknown_quantization_in_an_interior_block_is_rejected(tmp_path):
    def corrupt(header):
        quant = json.loads(header["__metadata__"]["_quantization_metadata"])
        quant["layers"]["blocks.12.attn.out_proj"]["format"] = "unknown_packed_format"
        header["__metadata__"]["_quantization_metadata"] = json.dumps(quant)

    path = checkpoint(tmp_path / "unknown.safetensors", extra_w4=True, broken=corrupt)
    assert not studio_checkpoints.h3_compatible(Path(path))


@pytest.mark.parametrize("storage", ["int8", "int8-legacy"])
def test_int8_marker_on_a_float_weight_is_rejected(tmp_path, storage):
    path = checkpoint(tmp_path / "contradictory.safetensors", storage=storage,
                      broken=lambda h: h["blocks.0.attn.out_proj.weight"].update(dtype="BF16"))
    with pytest.raises(ValueError, match="blocks.0.attn.out_proj"):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("corrupt", ["float_weight", "missing_weight", "missing_scale", "wrong_scale_dtype", "wrong_scale_shape", "bad_convrot", "missing_descriptor"])
def test_all_int8_descriptors_are_validated_including_middle_blocks(tmp_path, corrupt):
    stem = "blocks.12.attn.out_proj"

    def break_layer(header):
        if corrupt == "float_weight":
            header[stem + ".weight"]["dtype"] = "BF16"
        elif corrupt == "missing_weight":
            header.pop(stem + ".weight")
        elif corrupt == "missing_scale":
            header.pop(stem + ".weight_scale")
        elif corrupt == "wrong_scale_dtype":
            header[stem + ".weight_scale"]["dtype"] = "I8"
        elif corrupt == "wrong_scale_shape":
            header[stem + ".weight_scale"]["shape"] = [1, 5376]
        else:
            metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
            if corrupt == "missing_descriptor":
                metadata["layers"].pop(stem)
            else:
                metadata["layers"][stem]["convrot_groupsize"] = 15
            header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)

    path = checkpoint(tmp_path / "bad-middle.safetensors", extra_int8=True, broken=break_layer)
    with pytest.raises(ValueError, match="blocks.12.attn.out_proj"):
        studio_checkpoints.h3_variant(path)


def test_valid_mixed_float_and_int8_without_a_float_marker_is_supported(tmp_path):
    def make_one_float(header):
        stem = "blocks.0.attn.out_proj"
        header[stem + ".weight"]["dtype"] = "BF16"
        header.pop(stem + ".weight_scale")
        header.pop(stem + ".comfy_quant")

    path = checkpoint(tmp_path / "mixed.safetensors", broken=make_one_float)
    assert studio_checkpoints.h3_variant(path) == "base"


def test_int8_nonrotated_scalar_scale_is_valid(tmp_path):
    def make_scalar(header):
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        config = metadata["layers"]["blocks.0.attn.out_proj"]
        config.pop("convrot")
        header["blocks.0.attn.out_proj.weight_scale"]["shape"] = []
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)

    path = checkpoint(tmp_path / "scalar.safetensors", storage="int8-legacy", broken=make_scalar)
    assert studio_checkpoints.h3_variant(path) == "base"


def test_w4a8_loader_defaults_float_scale_and_optional_codebook_are_valid(tmp_path):
    def use_defaults(header):
        stem = "blocks.0.attn.out_proj"
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"][stem].pop("group_size")
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        header[stem + ".weight_s_rel"]["dtype"] = "F32"
        header.pop(stem + ".weight_codebook")

    path = checkpoint(tmp_path / "w4-defaults.safetensors", storage="w4a8", broken=use_defaults)
    assert studio_checkpoints.h3_variant(path) == "w4a8"


def test_w4a8_invalid_convrot_group_is_rejected(tmp_path):
    def bad_group(header):
        stem = "blocks.0.attn.out_proj"
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"][stem]["convrot_groupsize"] = 15
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)

    path = checkpoint(tmp_path / "bad-rot.safetensors", storage="w4a8", broken=bad_group)
    with pytest.raises(ValueError, match="blocks.0.attn.out_proj"):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("block", [1, 12, 25, 48])
@pytest.mark.parametrize("layer", ["adaln_proj.linear", "attn.out_proj", "attn.qkv_proj", "mlp.fc1", "mlp.fc2"])
def test_every_interior_block_requires_its_expected_layer(tmp_path, block, layer):
    stem = f"blocks.{block}.{layer}"
    path = checkpoint(tmp_path / "missing.safetensors", storage="bf16",
                      broken=lambda header: header.pop(stem + ".weight"))
    with pytest.raises(ValueError, match=stem.replace(".", r"\.")):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("storage", ["bf16", "int8-legacy", "w4a8"])
def test_interior_weight_and_scale_cannot_share_a_wrong_logical_shape(tmp_path, storage):
    stem = "blocks.12.attn.out_proj"

    def corrupt(header):
        header[stem + ".weight"]["shape"] = [5376, 3328 if storage == "w4a8" else 6656]
        if storage == "w4a8":
            header[stem + ".weight_s_rel"]["shape"] = [5376, 416]

    path = checkpoint(tmp_path / "wrong-shape.safetensors", storage=storage, broken=corrupt)
    with pytest.raises(ValueError, match="blocks.12.attn.out_proj"):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("convrot", [True, False])
@pytest.mark.parametrize("scale_shape", [[], [1], [1, 1], [5376, 1]])
def test_int8_tensorwise_scale_supports_scalar_and_per_channel_with_or_without_convrot(tmp_path, convrot, scale_shape):
    stem = "blocks.12.attn.out_proj"

    def configure(header):
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"][stem]["convrot"] = convrot
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        header[stem + ".weight_scale"]["shape"] = scale_shape

    path = checkpoint(tmp_path / "scale.safetensors", storage="int8-legacy", broken=configure)
    assert studio_checkpoints.h3_variant(path) == "base"


@pytest.mark.parametrize("dtype", ["F8_E4M3", "F32", "U8"])
@pytest.mark.parametrize("group", [4, 8, 16, 32, 48, 64, 96])
def test_w4a8_supported_group_sizes_and_relative_scale_types(tmp_path, dtype, group):
    stem = "blocks.12.attn.qkv_proj"

    def configure(header):
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"][stem]["group_size"] = group
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        header[stem + ".weight_s_rel"].update(dtype=dtype, shape=[21504, 5376 // group])

    path = checkpoint(tmp_path / "w4.safetensors", storage="w4a8", broken=configure)
    assert studio_checkpoints.h3_variant(path) == "w4a8"


@pytest.mark.parametrize("group", [0, 3, 6, 7, 12, 24, 5120])
def test_w4a8_invalid_group_sizes_are_rejected(tmp_path, group):
    stem = "blocks.12.attn.qkv_proj"

    def configure(header):
        metadata = json.loads(header["__metadata__"]["_quantization_metadata"])
        metadata["layers"][stem]["group_size"] = group
        header["__metadata__"]["_quantization_metadata"] = json.dumps(metadata)
        header[stem + ".weight_s_rel"]["shape"] = [21504, 5376 // max(group, 1)]

    path = checkpoint(tmp_path / "bad-group.safetensors", storage="w4a8", broken=configure)
    with pytest.raises(ValueError, match="blocks.12.attn.qkv_proj"):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("suffix", ["weight_s_channel", "weight_codebook"])
@pytest.mark.parametrize("dtype", ["F16", "BF16", "U8"])
def test_w4a8_channel_scale_and_codebook_require_fp32(tmp_path, suffix, dtype):
    stem = "blocks.12.attn.out_proj"
    path = checkpoint(tmp_path / "bad-type.safetensors", storage="w4a8",
                      broken=lambda header: header[stem + "." + suffix].update(dtype=dtype))
    with pytest.raises(ValueError, match="blocks.12.attn.out_proj"):
        studio_checkpoints.h3_variant(path)


def test_declared_legacy_metadata_requires_layers(tmp_path):
    def corrupt(header):
        header["__metadata__"] = {"_quantization_metadata": "{}"}

    path = checkpoint(tmp_path / "missing-layers.safetensors", storage="bf16", broken=corrupt)
    with pytest.raises(ValueError, match="メタデータ"):
        studio_checkpoints.h3_variant(path)


@pytest.mark.parametrize("version", ["0.2.31", "0.2.33", "0.2.40", "0.2.41rc1"])
def test_custom_w4a8_requires_validated_kitchen_version(tmp_path, version):
    from modules_forge import minimax_h3_bridge as bridge
    from tools.tests.test_h3_acceleration import H3AccelerationTests

    path = checkpoint(tmp_path / "local.safetensors", storage="w4a8")
    harness = H3AccelerationTests()
    harness.bridge = bridge
    ready = replace(harness.ready(), acceleration=H3Acceleration(checkpoint=path),
                    package_versions={"comfy-kitchen": version})
    with pytest.raises(bridge.H3BridgeError, match="0.2.41"):
        bridge.validate_readiness(ready)


def test_custom_w4a8_accepts_validated_kitchen_version(tmp_path):
    from modules_forge import minimax_h3_bridge as bridge
    from tools.tests.test_h3_acceleration import H3AccelerationTests

    path = checkpoint(tmp_path / "local.safetensors", storage="w4a8")
    harness = H3AccelerationTests()
    harness.bridge = bridge
    ready = replace(harness.ready(), acceleration=H3Acceleration(checkpoint=path),
                    package_versions={"comfy-kitchen": "0.2.41"})
    bridge.validate_readiness(ready)


@pytest.mark.parametrize("values", [
    list(reversed(range(16))),
    [*range(15), float("nan")],
    [*range(15), float("inf")],
])
def test_w4a8_lora_checks_only_applied_codebook_values(tmp_path, values):
    stem = "blocks.12.attn.out_proj"
    path = checkpoint(tmp_path / "w4.safetensors", storage="w4a8", codebooks={stem: values})
    assert studio_checkpoints.h3_variant(path) == "w4a8"
    assert studio_checkpoints.h3_variant(path, lora_targets=("diffusion_model.blocks.11.attn.out_proj.weight",)) == "w4a8"
    with pytest.raises(ValueError, match="codebook"):
        studio_checkpoints.h3_variant(path, lora_targets=(f"diffusion_model.{stem}.weight",))


def test_w4a8_lora_accepts_sorted_or_absent_codebook(tmp_path):
    stem = "blocks.12.attn.out_proj"
    target = (f"diffusion_model.{stem}.weight",)
    path = checkpoint(tmp_path / "equal.safetensors", storage="w4a8", codebooks={stem: [1.0] * 16})
    assert studio_checkpoints.h3_variant(path, lora_targets=target) == "w4a8"
    path = checkpoint(tmp_path / "default.safetensors", storage="w4a8",
                      broken=lambda header: header.pop(stem + ".weight_codebook"))
    assert studio_checkpoints.h3_variant(path, lora_targets=target) == "w4a8"


def test_orbit_rejects_bad_w4a8_codebook_before_gpu_ownership(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from modules_forge import minimax_h3_bridge as bridge

    stem = "blocks.12.attn.out_proj"
    path = checkpoint(tmp_path / "bad.safetensors", storage="w4a8", codebooks={stem: list(reversed(range(16)))})
    request = H3Request(mode="orbit", prompt="test", first_frame="image.png",
                        acceleration=H3Acceleration(checkpoint=path))
    monkeypatch.setattr(bridge.orbit_assets, "snapshot", lambda _: {"targets": [f"diffusion_model.{stem}.weight"]})
    monkeypatch.setattr(bridge.orbit_assets, "verify", lambda _: None)
    ownership = Mock()
    monkeypatch.setattr(bridge, "GPUOwnership", ownership)
    monkeypatch.setattr(bridge, "release_forge_vram", Mock())
    monkeypatch.setattr(bridge, "_run_generation", Mock(side_effect=AssertionError("must not start")))
    with pytest.raises(ValueError, match="codebook"):
        next(bridge.run_generation(request, tmp_path, "http://127.0.0.1:8189", tmp_path, tmp_path))
    ownership.assert_not_called()


def test_orbit_standard_w4a8_detects_model_replacement_while_queued(tmp_path, monkeypatch):
    from unittest.mock import Mock

    from modules_forge import minimax_h3_bridge as bridge

    acceleration = H3Acceleration(model_variant="w4a8")
    directory, filename = acceleration.model_files(MODEL_FILES)["FL2VA"]
    model = Path(checkpoint(tmp_path / "models" / directory / filename, storage="w4a8"))
    request = H3Request(mode="orbit", prompt="test", first_frame="image.png", acceleration=acceleration)
    monkeypatch.setattr(bridge, "model_root", lambda _: tmp_path / "models")
    monkeypatch.setattr(bridge.orbit_assets, "snapshot", lambda _: {"targets": ["diffusion_model.blocks.12.attn.out_proj.weight"]})
    monkeypatch.setattr(bridge.orbit_assets, "verify", lambda _: None)
    lease = Mock()
    lease.acquire.side_effect = [False, True]
    monkeypatch.setattr(bridge, "GPUOwnership", lambda: lease)
    monkeypatch.setattr(bridge, "release_forge_vram", Mock())
    monkeypatch.setattr(bridge.time, "sleep", lambda _: None)
    ready = Mock()
    monkeypatch.setattr(bridge, "ensure_ready", ready)
    run = bridge.run_generation(request, tmp_path, "http://127.0.0.1:8189", tmp_path, tmp_path)
    assert next(run)["stage"] == "queued"
    with model.open("ab") as stream:
        stream.write(b"replacement")
    with pytest.raises(ValueError, match="変更"):
        next(run)
    ready.assert_not_called()
    lease.release.assert_called_once()


@pytest.mark.parametrize("legacy_format", ["int8_tensorwise", "asym_w4a8_int8"])
def test_conflicting_legacy_and_comfy_quant_for_same_layer_are_rejected(tmp_path, legacy_format):
    def add_legacy(header):
        header["__metadata__"] = {"_quantization_metadata": json.dumps({"layers": {
            "blocks.0.attn.out_proj": {"format": legacy_format},
        }})}

    path = checkpoint(tmp_path / "mixed-metadata.safetensors", storage="w4a8-comfy", broken=add_legacy)
    assert not studio_checkpoints.h3_compatible(Path(path))


def test_mode_selects_and_freezes_only_its_explicit_slot(tmp_path):
    first = checkpoint(tmp_path / "one" / "same.safetensors")
    second = checkpoint(tmp_path / "two" / "same.safetensors", storage="w4a8")
    values = (*H3Acceleration().values()[:-2], first, second)
    fl = H3Acceleration.from_values(values, mode="text")
    ref = H3Acceleration.from_values(values, mode="references")
    assert (fl.checkpoint, ref.checkpoint) == (first, second)
    assert fl.checkpoint_mode == "FL2VA"
    assert ref.checkpoint_mode == "Ref2VA"
    assert fl.model_files(MODEL_FILES)["FL2VA"][1] == "same.safetensors"
    assert ref.model_files(MODEL_FILES)["Ref2VA"][1] == "same.safetensors"
    assert studio_checkpoints.config_path(tmp_path, first) != studio_checkpoints.config_path(tmp_path, second)
    assert build_workflow(H3Request(mode="text", prompt="test", acceleration=fl), {}, seed=1)["1"]["inputs"]["unet_name"] == "same.safetensors"
    assert build_workflow(H3Request(mode="references", prompt="<Picture 1>", reference_images=("p.png",), acceleration=ref), {"images": ["p.png"]}, seed=1)["1"]["inputs"]["unet_name"] == "same.safetensors"
    snap = studio_checkpoints.snapshot_checkpoint(first)
    Path(second).write_bytes(b"unused changed")
    studio_checkpoints.verify_snapshot(snap)
    Path(first).write_bytes(b"changed while waiting")
    with pytest.raises(ValueError, match="変更"):
        studio_checkpoints.verify_snapshot(snap)


def test_missing_unused_official_or_inactive_custom_does_not_block_active_mode(tmp_path):
    first = checkpoint(tmp_path / "custom.safetensors")
    values = (*H3Acceleration().values()[:-2], first, str(tmp_path / "missing.safetensors"))
    fl = H3Acceleration.from_values(values, mode="text")
    fl.checkpoint_for_mode("text")
    assert model_file_status(tmp_path, fl)["FL2VA"]
    assert not model_file_status(tmp_path, fl)["Ref2VA"]
    request = H3ImageRequest(mode="text", prompt="test", checkpoint=first)
    assert build_image_workflow(request, {}, 1)["model"]["inputs"]["unet_name"] == "custom.safetensors"
    with pytest.raises(ValueError):
        H3Acceleration.from_values(values, mode="references").checkpoint_for_mode("references")


def test_same_model_can_be_explicitly_selected_for_both_modes(tmp_path):
    path = checkpoint(tmp_path / "both.safetensors")
    values = (*H3Acceleration().values()[:-2], path, path)
    fl = H3Acceleration.from_values(values, mode="text")
    ref = H3Acceleration.from_values(values, mode="references")
    assert fl.checkpoint_for_mode("text") == path
    assert ref.checkpoint_for_mode("references") == path


def test_turbo_name_does_not_override_format_or_steps(tmp_path):
    path = checkpoint(tmp_path / "fused_turbo_w4a8.safetensors", storage="bf16")
    acceleration = replace(H3Acceleration(), checkpoint=path, checkpoint_mode="FL2VA")
    assert acceleration.storage_variant == "base"
    assert acceleration.model_variant == "base"
    graph = build_workflow(H3Request(mode="text", prompt="test", steps=20, acceleration=acceleration), {}, seed=1)
    assert graph["8"]["inputs"]["steps"] == 20


def test_custom_w4a8_enforces_runtime_requirement_from_actual_storage(tmp_path):
    from modules_forge import minimax_h3_bridge as bridge
    from tools.tests.test_h3_acceleration import H3AccelerationTests

    path = checkpoint(tmp_path / "plain-name.safetensors", storage="w4a8-comfy")
    harness = H3AccelerationTests()
    harness.bridge = bridge
    ready = replace(harness.ready(), acceleration=H3Acceleration(checkpoint=path),
                    package_versions={"comfy-kitchen": "0.2.30"})
    with pytest.raises(bridge.H3BridgeError, match="W4A8"):
        bridge.validate_readiness(ready)


def test_video_ui_passes_both_slots_and_resolves_the_current_mode(tmp_path):
    from tools.tests.test_h3_acceleration_ui import load_ui

    ui = load_ui(str(tmp_path))
    app = ui._build_ui()[0][0]
    try:
        generate = next(fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "_generate_ui")
        assert [item.label for item in generate.inputs[-2:]] == ["FL2VA本体", "Ref2VA本体"]
        fl = checkpoint(tmp_path / "fl.safetensors")
        ref = checkpoint(tmp_path / "ref.safetensors")
        values = (*H3Acceleration().values()[:-2], fl, ref)
        args = ["references", "<Picture 1>", None, None, ["photo.png"], None, None,
                "16:9", "preview", 5, 20, 1, "simple", "match", "off", None, 1.0]
        request = ui._request_from_ui(*args, *values)
        assert request.acceleration.checkpoint == ref
        assert request.acceleration.checkpoint_mode == "Ref2VA"
    finally:
        app.close()


def test_image_ui_uses_slot_corresponding_to_mode(tmp_path):
    from tools.tests.test_minimax_h3_images import load_image_ui

    ui, _ = load_image_ui()
    fl = checkpoint(tmp_path / "fl.safetensors")
    ref = checkpoint(tmp_path / "ref.safetensors")
    values = (*ui.H3NegPiP().values(), fl, ref)
    request = ui._request("references", "<Picture 1>", ["photo.png"], 768, 768, 20, "1", "simple", 0, False, *values)
    assert request.checkpoint == ref


def test_turbo_button_does_not_replace_local_checkpoint(tmp_path):
    from tools.tests.test_h3_acceleration_ui import load_ui

    ui = load_ui(str(tmp_path))
    update = ui._turbo_four_updates("16:9", "preview", 5, "match", "chosen.safetensors")
    assert update[0].get("__type__") == "update"
    assert update[1:3] == (4, "simple")


def test_history_restore_preserves_the_other_mode_selection(tmp_path):
    from tools.tests.test_h3_acceleration_ui import load_ui

    ui = load_ui(str(tmp_path))
    request = ui.H3Request(mode="references", prompt="test",
                           acceleration=H3Acceleration(checkpoint="chosen.safetensors", checkpoint_mode="Ref2VA"))
    updates = ui._history_settings_updates(request, include_acceleration=True)
    fl_update, ref_update = updates[22 + len(H3Acceleration().values()) - 2:22 + len(H3Acceleration().values())]
    assert "value" not in fl_update
    assert ref_update["value"] == "chosen.safetensors"


@pytest.mark.parametrize("feature", ["minimax_h3", "minimax_h3_image"])
def test_launcher_status_considers_both_selected_slots_independently(tmp_path, feature):
    from unittest.mock import patch

    from modules import aikimi_capabilities as capabilities
    from modules.aikimi_diagnostics import DiagnosticPaths

    for name in ("modules_forge/minimax_h3_bridge.py",
                 "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_studio.py",
                 "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    chosen = checkpoint(tmp_path / "reference.safetensors")
    paths = DiagnosticPaths(root=tmp_path, models_root=tmp_path / "models", output_root=tmp_path / "output")

    def status(runtime, acceleration=None):
        ready = acceleration is not None and acceleration.checkpoint == chosen and acceleration.checkpoint_mode == "Ref2VA"
        return {"FL2VA": False, "Ref2VA": ready, "Qwen3-VL 32B": True, "Video VAE": True, "Audio VAE": True}

    with (patch("modules_forge.minimax_h3_runtime.installed_runtime_root", return_value=tmp_path),
          patch("modules_forge.minimax_h3_runtime.candidate_runtime_root", return_value=tmp_path),
          patch("modules_forge.minimax_h3_bridge.model_file_status", side_effect=status),
          patch("modules_forge.local_assets.selection", side_effect=lambda engine: {"checkpoint": chosen} if engine == "h3_ref2va" else {})):
        result = capabilities.feature_check(feature, paths)
    assert result.available
