"""Input and checkpoint boundaries for optional Qwen 2.1 Fun ControlNet."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from PIL import Image

from modules_forge.qwen_image21 import core
from modules_forge.qwen_image21.fun_controlnet import installed, status
from tools import qwen_image21_worker as worker


def test_control_image_is_snapshotted_without_changing_the_reference_list(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (48, 32), (12, 34, 56, 70)).save(source)
    request = core.Request(
        "A dancer", control_kind="pose", control_image=str(source),
        precision="int8", seed=43,
    ).resolved()
    job = tmp_path / "job"
    job.mkdir()
    copied = core.copy_control_image(request.control_image, job)
    source.unlink()
    assert Path(copied).is_file()
    assert Image.open(copied).mode == "RGB"
    assert request.input_images == ()


@pytest.mark.parametrize("fields", [
    {"control_kind": "pose", "control_image": ""},
    {"control_kind": "pose", "precision": "bf16"},
    {"control_kind": "pose", "sparse_mode": "fixed"},
    {"control_kind": "pose", "control_strength": float("nan")},
    {"control_kind": "pose", "control_strength": 2.1},
    {"control_kind": "unknown"},
])
def test_invalid_control_requests_stop_before_gpu_or_download(tmp_path, fields):
    image = tmp_path / "pose.png"
    Image.new("RGB", (32, 32)).save(image)
    fields = {"control_image": str(image), **fields}
    with pytest.raises(core.QwenImage21Error):
        core.Request("A dancer", **fields).resolved()


def test_worker_rejects_missing_control_and_incompatible_base(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    job = tmp_path / "job"
    job.mkdir()
    request = {
        "prompt": "A dancer", "width": 512, "height": 512, "steps": 2, "seed": 43,
        "control_image": str(tmp_path / "missing.png"), "control_strength": 1.0,
        "input_images": [], "precision": "int8", "memory_mode": "offload",
    }
    (job / "request.json").write_text(json.dumps(request))
    payload = {"job_dir": str(job), "model_path": str(model)}
    with pytest.raises(ValueError, match="ControlNet"):
        worker._read_request(payload)
    image = tmp_path / "pose.png"
    Image.new("RGB", (32, 32)).save(image)
    request["control_image"] = str(image)
    request["precision"] = "bf16"
    (job / "request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="INT8"):
        worker._read_request(payload)
    request["precision"] = "int8"
    request["sparse_mode"] = "fixed"
    (job / "request.json").write_text(json.dumps(request))
    with pytest.raises(ValueError, match="Sparse Attention"):
        worker._read_request(payload)


def test_missing_optional_patch_has_actionable_status(tmp_path):
    assert "未導入" in status(tmp_path)
    with pytest.raises(ValueError, match="導入コマンド"):
        installed(tmp_path)


def test_inpainting_request_requires_a_control_and_a_matching_edit_mask(tmp_path):
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    control = tmp_path / "control.png"
    for path in (source, mask, control):
        Image.new("RGB", (512, 512), "white").save(path)
    fields = dict(
        control_kind="pose", control_image=str(control), control_inpaint=True,
        input_images=(str(source),), edit_mask_reference=0, edit_mask_path=str(mask),
        width=512, height=512, precision="int8",
    )
    resolved = core.Request("Change the jacket", **fields).resolved()
    assert resolved.control_inpaint and resolved.edit_mask_path == str(mask)
    with pytest.raises(core.QwenImage21Error, match="制御画像"):
        core.Request("Change the jacket", **(fields | {"control_kind": "off"})).resolved()
    with pytest.raises(core.QwenImage21Error, match="マスク"):
        core.Request("Change the jacket", **(fields | {"edit_mask_path": ""})).resolved()
    with pytest.raises(core.QwenImage21Error):
        core.Request("Change the jacket", **(fields | {"width": 768})).resolved()


def test_worker_inpainting_requires_the_source_mask_and_output_size(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model_index.json").write_text(json.dumps({"_class_name": "QwenImage21Pipeline"}))
    job = tmp_path / "job"
    job.mkdir()
    control = tmp_path / "control.png"
    source = tmp_path / "source.png"
    mask = tmp_path / "mask.png"
    for path in (control, source, mask):
        Image.new("RGB", (512, 512), "white").save(path)
    request = {
        "prompt": "Change the jacket", "width": 512, "height": 512,
        "steps": 2, "seed": 1, "control_image": str(control),
        "control_inpaint": True, "precision": "int8", "memory_mode": "offload",
        "edit_mask": {"original_path": str(source), "mask_path": str(mask)},
    }
    payload = {"job_dir": str(job), "model_path": str(model)}
    (job / "request.json").write_text(json.dumps(request))
    assert worker._read_request(payload)[2]["control_inpaint"] is True
    (job / "request.json").write_text(json.dumps(request | {"edit_mask": None}))
    with pytest.raises(ValueError, match="source and mask"):
        worker._read_request(payload)
    (job / "request.json").write_text(json.dumps(request | {"width": 768}))
    with pytest.raises(ValueError):
        worker._read_request(payload)


def test_inpaint_condition_packs_control_keep_mask_and_masked_source():
    torch = pytest.importorskip("torch")
    # The app venv has an older Diffusers; the installed worker uses the
    # pinned Qwen 2.1 revision. Only the conditioning method is exercised here.
    module_name = "diffusers.models.transformers.transformer_qwenimage21"
    qwen_stub = ModuleType(module_name)
    qwen_stub.QwenImage21TransformerBlock = object
    with patch.dict(sys.modules, {module_name: qwen_stub}):
        from modules_forge.qwen_image21.fun_controlnet_runtime import FunUnion

    class Processor:
        def preprocess(self, image, **_kwargs):
            pixels = torch.as_tensor(__import__("numpy").array(image.convert("RGB"))).float()
            return pixels.permute(2, 0, 1).unsqueeze(0) / 255

    class Pipe:
        image_processor = Processor()
        vae = SimpleNamespace(dtype=torch.float32)
        _execution_device = torch.device("cpu")

        def _encode_vae_image(self, image, _generator):
            # A small deterministic stand-in for the VAE preserves the alpha
            # and spatial values so the channel order and masking are visible.
            return image[:, :1].repeat(1, 64, 1, 1, 1)

        def _pack_latents(self, tensor, batch, channels, height, width):
            assert (batch, channels, height, width) == (1, 129, 2, 2)
            return tensor

    subject = SimpleNamespace(context=None, strength=0)
    control = Image.new("RGB", (2, 2), "white")
    source = Image.new("RGB", (2, 2), "white")
    mask = Image.new("L", (2, 2), 0)
    mask.putpixel((1, 0), 255)
    FunUnion.set_control(
        subject, Pipe(), control, 0.8, torch.Generator(),
        inpaint_image=source, mask_image=mask,
    )
    assert subject.context.shape == (1, 129, 1, 2, 2)
    assert torch.all(subject.context[:, :64] == 1)
    assert subject.context[0, 64, 0].tolist() == [[1, 0], [1, 1]]
    assert subject.context[0, 65, 0].tolist() == [[1, 0], [1, 1]]
    assert subject.strength == 0.8
    FunUnion.set_control(subject, Pipe(), control, 1, torch.Generator())
    assert torch.count_nonzero(subject.context[:, 64:]) == 0
