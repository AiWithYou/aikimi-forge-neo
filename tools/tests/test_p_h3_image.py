"""Offline contracts for the P-H3 still-image workflow."""

import json
import struct
from dataclasses import replace

import pytest

from modules_forge import p_h3_image as ph3


def safetensors_header(tmp_path, shapes, name="model.safetensors"):
    header = {
        key: {"dtype": "BF16", "shape": shape, "data_offsets": [0, 0]}
        for key, shape in shapes.items()
    }
    data = json.dumps(header).encode()
    path = tmp_path / name
    path.write_bytes(struct.pack("<Q", len(data)) + data)
    return path


def test_structural_probe_accepts_renamed_p_h3_and_both_quantization_packings(tmp_path):
    common = {
        "audio_patch_proj.weight": [5376, 32],
        "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [4097, 16],
        "blocks.0.adaln_proj.linear.weight": [96768, 16],
        "blocks.0.attn.q_norm.weight": [128],
    }
    for packed, width in ((False, 5376), (True, 2688)):
        shapes = dict(common)
        for layer, out in (("blocks.0.attn.qkv_proj", 21504), ("blocks.0.mlp.fc1", 28672)):
            shapes[layer + ".weight"] = [out, width]
            if packed:
                shapes[layer + ".comfy_quant"] = []
        assert ph3.compatible_checkpoint(safetensors_header(tmp_path, shapes, f"different-name-{width}.safetensors"))
    old = dict(common, adaln_t_table=[1025, 8])
    assert not ph3.compatible_checkpoint(safetensors_header(tmp_path, old, "old-h3.safetensors"))
    assert not ph3.compatible_checkpoint(safetensors_header(tmp_path, {"decoder.weight": [1]}, "vae.safetensors"))
    assert not ph3.compatible_checkpoint(safetensors_header(tmp_path, {"lora_A.weight": [1]}, "lora.safetensors"))


def test_request_accepts_official_sizes_and_distinguishes_target_from_references():
    ph3.PH3Request("A photograph", width=2912, height=2912).validate()
    ph3.PH3Request("A photograph", width=3872, height=2176).validate()
    edit = ph3.PH3Request(
        "Give the person the hat", mode="edit", target_image="target.png",
        reference_images=tuple(f"reference-{i}.png" for i in range(7)),
    )
    edit.validate()
    for invalid in (
        replace(edit, target_image=""),
        replace(edit, reference_images=edit.reference_images + ("ninth.png",)),
        replace(edit, mode="text"),
        replace(edit, reference_images=("",)),
        ph3.PH3Request(" "),
        ph3.PH3Request("x", width=2913),
        ph3.PH3Request("x", width=16416),
        ph3.PH3Request("x", use_upsampler="false"),
    ):
        with pytest.raises(ph3.PH3Error):
            invalid.validate()


def test_custom_resolution_follows_native_node_limits_without_an_area_cap():
    ph3.PH3Request("test", width=32, height=16384).validate()
    ph3.PH3Request("test", width=8192, height=8192).validate()
    ph3.PH3Request("test", width=16384, height=16384).validate()
    assert ph3.maximum_png_bytes(8192, 8192) > 64 * 1024 * 1024


def test_text_workflow_uses_the_official_p_h3_sampler_and_no_image_inputs():
    request = ph3.PH3Request("A photograph", use_upsampler=False)
    graph = ph3.build_workflow(request, {"images": []}, 42)
    assert graph["condition"]["class_type"] == "PH3ImageConditioning"
    assert not any(k.startswith("images.") for k in graph["condition"]["inputs"])
    assert graph["shift"]["class_type"] == "MiniMaxH3SigmaShift"
    assert graph["shift"]["inputs"]["shift_video"] == 1.0
    assert graph["shift"]["inputs"]["shift_audio"] == 1.0
    assert graph["sigmas"]["inputs"]["sigmas"] == ph3.SIGMAS
    assert graph["sampler"]["inputs"]["sampler_name"] == "euler"
    assert graph["guider"]["class_type"] == "BasicGuider"
    assert graph["noise"]["inputs"]["noise_seed"] == 42
    assert graph["save"]["class_type"] == "SaveImage"
    assert {n["class_type"] for n in graph.values()}.isdisjoint(
        {"TextGenerate", "MiniMaxH3ReferenceToVideo", "MiniMaxH3ImageToVideo", "CFGGuider"}
    )
    assert all("upsampler" not in key for key in graph)


def test_edit_workflow_preserves_eight_image_order_in_both_branches():
    request = ph3.PH3Request(
        "Arrange the subjects together", mode="edit", target_image="target.png",
        reference_images=tuple(f"reference-{i}.png" for i in range(7)),
    )
    names = [f"forge_h3/{i}.png" for i in range(8)]
    graph = ph3.build_workflow(request, {"images": names}, 123)
    condition = graph["condition"]["inputs"]
    for i, name in enumerate(names, 1):
        assert graph[f"image_{i}"]["inputs"]["image"] == name
        expected = ["edit_target", 0] if i == 1 else [f"image_{i}", 0]
        assert condition[f"images.image_{i}"] == expected
    assert graph["edit_target"]["class_type"] == "ImageScaleToTotalPixels"
    assert graph["edit_target"]["inputs"] == {
        "image": ["image_1", 0], "upscale_method": "lanczos", "megapixels": 4.0, "resolution_steps": 32,
    }
    assert graph["upsampler_target"]["inputs"]["image"] == ["image_1", 0]
    assert graph["upsampler_batch"]["inputs"]["images.image0"] == ["upsampler_target", 0]
    for i in range(1, 8):
        assert graph["upsampler_batch"]["inputs"][f"images.image{i}"] == [f"upsampler_ref_{i + 1}", 0]
    assert graph["upsampler"]["class_type"] == "TextGenerate"
    assert graph["upsampler"]["inputs"]["thinking"] is False
    assert graph["upsampler"]["inputs"]["use_default_template"] is True
    assert graph["condition"]["inputs"]["prompt"] == ["upsampler", 0]
    assert graph["upsampler_preview"]["inputs"]["source"] == ["upsampler", 0]
    with pytest.raises(ph3.PH3Error):
        ph3.build_workflow(request, {"images": names[:-1]}, 123)


def test_every_workflow_connection_points_to_an_existing_node():
    for mode, target, references in (("text", "", ()), ("edit", "a", ()), ("edit", "a", ("b", "c"))):
        request = ph3.PH3Request("A photograph", mode=mode, target_image=target, reference_images=references)
        names = [f"i-{i}.png" for i in range(bool(target) + len(references))]
        graph = ph3.build_workflow(request, {"images": names}, 3)
        for node in graph.values():
            for value in node["inputs"].values():
                if isinstance(value, list):
                    assert value[0] in graph
                    assert isinstance(value[1], int)


def test_upsampled_prompt_is_captured_separately_from_the_request():
    history = {"job-1": {"outputs": {"upsampler_preview": {"text": ["Detailed prompt"]}}}}
    assert ph3.extract_upsampled_prompt(history, "job-1", enabled=True) == "Detailed prompt"
    assert ph3.extract_upsampled_prompt({}, "job-1", enabled=False) is None
    with pytest.raises(ph3.PH3Error):
        ph3.extract_upsampled_prompt({"job-1": {"outputs": {}}}, "job-1", enabled=True)
