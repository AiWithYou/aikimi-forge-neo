"""Real residual arithmetic and the packed projection boundary for shared adapters."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file
from torch import nn

from modules_forge.qwen_image21.adapter_stack import materialize_pdd_projection
from modules_forge.qwen_image21.capabilities import PRECISIONS, recommended_steps, validate_sampling
from modules_forge.qwen_image21.core import Request
from modules_forge.qwen_image21.outpaint_runtime import load_adapter
from modules_forge.qwen_image21.pdd_vendor.lora_utils_pdd import PDDLoRALinear
from modules_forge.qwen_image21.style_lora_runtime import load_adapters


def test_pdd_outpaint_and_multiple_style_residuals_add_without_changing_base(tmp_path):
    model = nn.Module()
    block = nn.Module()
    block.attn = nn.Module()
    base = nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
    original = base.weight.detach().clone()
    pdd = PDDLoRALinear(base, rank=2, alpha=1)
    with torch.no_grad():
        pdd.lora_down.fill_(1)
        pdd.lora_up.fill_(1)
    block.attn.to_q = pdd
    model.transformer_blocks = nn.ModuleList([block])
    mapped = {"transformer_blocks.0.attn.to_q": (torch.ones(2, 3), torch.ones(4, 2))}
    with (
        patch("modules_forge.qwen_image21.outpaint_runtime.load_file", return_value={}),
        patch("modules_forge.qwen_image21.outpaint_runtime.map_adapter", return_value=mapped),
    ):
        load_adapter(model, "pinned-adapter")
    (tmp_path / "loras").mkdir()
    prefix = "lora_unet_transformer_blocks__0__attn__to_q"
    tensors = {
        prefix + ".lora_down.weight": torch.ones(2, 3),
        prefix + ".lora_up.weight": torch.ones(4, 2),
        prefix + ".alpha": torch.tensor(1.0),
    }
    for name in ("a", "b"):
        save_file(tensors, tmp_path / f"loras/{name}.safetensors", metadata={"model_type": "qwen_image_21"})
    load_adapters(
        model,
        tmp_path,
        {
            "precision": "base_q4_k_m",
            "steps": 4,
            "fun_acc": True,
            "style_loras": [{"name": "a.safetensors", "strength": 0.5}, {"name": "b.safetensors", "strength": -0.25}],
        },
    )
    x = torch.ones(1, 3, dtype=torch.bfloat16)
    torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(x), base(x) + 3 + 6 + 0.75, rtol=0.02, atol=0.02)
    torch.testing.assert_close(base.weight, original)


@pytest.mark.parametrize("precision", sorted(PRECISIONS))
@pytest.mark.parametrize("fun_acc", [False, True])
@pytest.mark.parametrize("control", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_sampling_matrix_accepts_executable_combinations(precision, fun_acc, control, sparse):
    values = {
        "precision": precision,
        "steps": 4 if fun_acc else 17,
        "fun_acc": fun_acc,
        "control_kind": "canny" if control else "off",
        "sparse_mode": "fixed" if sparse else "off",
    }
    validate_sampling(values)


@pytest.mark.parametrize("precision", sorted(PRECISIONS))
@pytest.mark.parametrize("strength", [-0.5, 2.5])
def test_requests_preserve_steps_and_control_strength_outside_recommendations(tmp_path, precision, strength):
    control = tmp_path / "control.png"
    Image.new("RGB", (256, 256), "white").save(control)
    request = Request(
        "generate",
        precision=precision,
        steps=120,
        sparse_mode="fixed",
        control_kind="canny",
        control_image=str(control),
        control_strength=strength,
    ).resolved()
    assert request.steps == 120
    assert request.control_strength == strength
    assert request.sparse_mode == "fixed"


@pytest.mark.parametrize("steps", [0, -1, 4.5, True, float("inf")])
def test_invalid_step_input_still_fails(steps):
    with pytest.raises(ValueError, match="Steps"):
        Request("generate", precision="turbo_official_int8", steps=steps).resolved()


@pytest.mark.parametrize("strength", [True, float("inf"), float("nan")])
def test_nonfinite_or_boolean_control_strength_still_fails(strength):
    with pytest.raises(ValueError, match="強さ"):
        Request("generate", control_strength=strength).resolved()


@pytest.mark.parametrize("precision", sorted(PRECISIONS))
def test_current_fun_acc_sampler_requires_four_steps(precision):
    with pytest.raises(ValueError, match="4 steps"):
        validate_sampling({"precision": precision, "fun_acc": True, "steps": 8})


def test_recommendations_are_nonblocking_and_only_describe_selected_options():
    from modules_forge.qwen_image21.capabilities import sampling_recommendations

    assert sampling_recommendations({"precision": "turbo_official_int8", "steps": 8}) == []
    notices = sampling_recommendations(
        {
            "precision": "turbo_official_int8",
            "steps": 12,
            "sparse_mode": "fixed",
            "control_kind": "canny",
            "control_strength": 2.5,
        }
    )
    assert any("8 steps" in notice for notice in notices)
    assert any("通常Attention" in notice for notice in notices)
    assert any("0〜2" in notice for notice in notices)


@pytest.mark.parametrize("precision", sorted(PRECISIONS))
def test_outpaint_shared_profile_preserves_custom_steps(precision):
    from modules_forge.qwen_image21.outpaint_profile import FIELDS, resolve

    profile = Request("generate", precision=precision).to_dict()
    profile["lora_strengths"] = []
    resolved = resolve([profile[name] for name in FIELDS], 120)
    assert resolved["steps"] == 120


def test_outpaint_fun_acc_uses_the_four_step_sampler_control():
    from modules_forge.qwen_image21.outpaint_profile import FIELDS, resolve

    profile = Request("generate", precision="turbo_official_int8", fun_acc=True).to_dict()
    profile["lora_strengths"] = []
    resolved = resolve([profile[name] for name in FIELDS], 25)
    assert resolved["steps"] == 4


def test_only_gguf_decoder_is_dequantized_and_logical_dimensions_are_checked():
    packed = nn.Parameter(torch.ones(2, 7, dtype=torch.uint8), requires_grad=False)
    packed.quant_type = "Q4_K"
    source = SimpleNamespace(weight=packed, in_features=3, out_features=4, bias=torch.ones(4))
    untouched = object()
    transformer = SimpleNamespace(proj_out=source, transformer_blocks=untouched)
    fake = ModuleType("diffusers.quantizers.gguf.utils")
    fake.dequantize_gguf_tensor = lambda _: torch.full((4, 3), 2.0)
    with patch.dict(sys.modules, {"diffusers.quantizers.gguf.utils": fake}):
        materialize_pdd_projection(transformer)
        assert isinstance(transformer.proj_out, nn.Linear)
        assert transformer.proj_out.weight.dtype == torch.bfloat16
        torch.testing.assert_close(
            transformer.proj_out(torch.ones(1, 3, dtype=torch.bfloat16)), torch.full((1, 4), 7.0, dtype=torch.bfloat16)
        )
        assert transformer.transformer_blocks is untouched
        assert source.weight is packed
        transformer.proj_out = source
        fake.dequantize_gguf_tensor = lambda _: torch.ones(2, 7)
        with pytest.raises(ValueError, match="復元"):
            materialize_pdd_projection(transformer)
        assert transformer.proj_out is source


def test_local_model_sampling_contract_preserves_selected_file_and_rejects_turbo(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGBA", (256, 256), (15, 30, 45, 70)).save(source)
    model = str(tmp_path / "external-model")
    for precision in ("int8", "bf16", "w4a8", "base_q4_k_m"):
        request = Request(
            "extend",
            local_model=model,
            precision=precision,
            fun_acc=True,
            steps=4,
            input_images=(str(source),),
            width=320,
            height=256,
            outpaint_version="v2",
            outpaint_margins=(32, 0, 32, 0),
        ).resolved()
        assert request.local_model == model
        assert request.precision == precision
    for precision in sorted(item for item in PRECISIONS if recommended_steps(item) is not None):
        with pytest.raises(ValueError, match="外部モデル"):
            Request("extend", local_model=model, precision=precision, steps=recommended_steps(precision)).resolved()


def test_turning_control_off_does_not_keep_a_hidden_upload_active_for_sparse(tmp_path):
    source = tmp_path / "old-control.png"
    Image.new("RGB", (256, 256), "white").save(source)
    request = Request("generate", sparse_mode="fixed", control_kind="off", control_image=str(source)).resolved()
    assert request.control_image == ""
    assert request.sparse_mode == "fixed"
