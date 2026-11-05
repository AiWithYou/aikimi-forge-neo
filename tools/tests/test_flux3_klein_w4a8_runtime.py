"""CPU integration against FLUX 3's pinned Comfy and comfy-kitchen."""

import os
import subprocess
import unittest
from pathlib import Path

from modules_forge.flux3_klein import core

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    core.python_path().is_file() and (core.COMFY_ROOT / "comfy/sd.py").is_file(), "FLUX runtime is not installed"
)
class FluxW4RuntimeTests(unittest.TestCase):
    def test_fused_slice_lora_changes_and_restores_real_cpu_output(self):
        script = r"""
import importlib.metadata, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path("repositories/flux3-klein/ComfyUI").resolve()))
sys.argv = ["flux3_w4_cpu", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora, comfy.utils
from comfy import ops
from comfy.model_patcher import ModelPatcher
from comfy_kitchen.tensor import AsymW4A8Int8Layout
from safetensors.torch import save_file
from modules_forge.flux3_klein import inference
assert importlib.metadata.version("comfy-kitchen") == "0.2.37"
torch.manual_seed(7)
weight = torch.randn((48, 256), dtype=torch.bfloat16) * 0.1
packed, params = AsymW4A8Int8Layout.quantize(weight)
config = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
state = {"weight": packed, "weight_s_rel": params.scale, "weight_s_channel": params.s_channel,
         "weight_codebook": params.codebook, "comfy_quant": torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8)}
class Toy(comfy.model_base.Flux3):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.model_config = SimpleNamespace(unet_config={})
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.single_blocks = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.single_blocks[0].qkv = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(256, 48, bias=False)
    def forward(self, value):
        return self.diffusion_model.single_blocks[0].qkv(value)
model = Toy()
model.diffusion_model.single_blocks[0].qkv.load_state_dict(state, strict=True)
patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
target = ("diffusion_model.single_blocks.0.qkv.weight", (0, 16, 16))
assert comfy.lora.model_lora_keys_unet(model, {})["diffusion_model.single_blocks.0.k_proj"] == target
runtime = inference.ComfyRuntime.__new__(inference.ComfyRuntime)
runtime.torch, runtime.utils = torch, comfy.utils
inputs = torch.ones((1, 256), dtype=torch.bfloat16)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "toy.safetensors"
    prefix = "diffusion_model.single_blocks.0.k_proj"
    save_file({prefix + ".lora_A.weight": torch.ones((4, 256)) * 0.1,
               prefix + ".lora_B.weight": torch.ones((16, 4)) * 0.1}, str(path))
    first = patcher.model(inputs).float()
    patched, count = runtime.apply_lora(patcher, path, 1.0)
    assert count == 1
    patched.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    changed = patched.model(inputs).float()
    patched.unpatch_model(device_to=torch.device("cpu"))
    last = patcher.model(inputs).float()
    assert bool(torch.isfinite(changed).all())
    assert not torch.equal(first, changed)
    assert torch.equal(first, last)
print("FLUX_W4_CPU_LORA_OK")
"""
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
        result = subprocess.run(  # noqa: S603 -- fixed local interpreter and embedded CPU-only test
            [str(core.python_path()), "-X", "utf8", "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=150,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FLUX_W4_CPU_LORA_OK", result.stdout)

    def test_converted_fused_and_separate_files_load_in_pinned_comfy_on_cpu(self):
        script = r'''
import sys, tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path("repositories/flux3-klein/ComfyUI").resolve()))
sys.argv = ["flux3_w4_conversion_cpu", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy import ops
from comfy_kitchen.tensor import AsymW4A8Int8Layout
from safetensors import safe_open
from safetensors.torch import save_file
from modules_forge import local_assets
from modules_forge.flux3_klein import checkpoints
from tools import convert_flux3_klein_w4a8 as converter
from tools.tests.test_flux3_klein_w4a8 import FUSED, shapes

torch.set_num_threads(1)
torch.manual_seed(42)
def quantize_cpu(weight, device):
    assert device.type == "cuda"  # The production CLI checks for an Ampere GPU.
    packed, params = AsymW4A8Int8Layout.quantize(weight.cpu())
    assert params.correction is None and params.codebook is not None
    return packed, params.scale, params.s_channel, params.codebook

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    local_assets.LIBRARY = root / "library.json"
    local_assets.HASH_CACHE = root / "hashes.json"
    for fused in (True, False):
        dimensions = shapes("base")
        if not fused:
            for block in ("single_blocks.0", "single_blocks.1"):
                dimensions.pop(block + ".qkv.weight")
                for part in ("q", "k", "v"):
                    dimensions[block + "." + part + "_proj.weight"] = (16, 256)
        src, dst = root / f"source-{fused}.safetensors", root / f"converted-{fused}.safetensors"
        save_file({key: torch.randn(shape, dtype=torch.bfloat16) * 0.1 for key, shape in dimensions.items()}, str(src))
        before = src.read_bytes()
        with (patch.object(checkpoints, "expected_shapes", side_effect=shapes),
              patch.object(checkpoints, "quantized_keys", return_value=FUSED),
              patch.object(converter, "quantize_weight", side_effect=quantize_cpu),
              patch.object(converter, "validate_conversion_resources"),
              patch.object(converter.gpu_fault.STATE, "ensure_available"),
              patch.object(torch.cuda, "is_available", return_value=True),
              patch.object(torch.cuda, "get_device_capability", return_value=(8, 6))):
            converter.convert(src, dst, torch.device("cuda:0"))
            assert checkpoints.header_info(dst)["precision"] == "w4a8"
        assert src.read_bytes() == before
        prefix = "single_blocks.0.qkv" if fused else "single_blocks.0.q_proj"
        with safe_open(dst, framework="pt", device="cpu") as stored:
            state = {name[len(prefix)+1:]: stored.get_tensor(name) for name in stored.keys() if name.startswith(prefix + ".")}
        linear = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(256, 48 if fused else 16, bias=False)
        linear.load_state_dict(state, strict=True)
        inputs = torch.ones((1, 256), dtype=torch.bfloat16)
        baseline = linear(inputs).float()
        assert bool(torch.isfinite(baseline).all())
        state["weight_s_channel"] = state["weight_s_channel"].clone()
        state["weight_s_rel"] = state["weight_s_rel"].clone()
        state["weight_s_channel"][0] = 0
        state["weight_s_channel"][1] *= -1
        state["weight_s_channel"][3] = -0.0
        rel_bytes = state["weight_s_rel"].view(torch.uint8)
        rel_bytes[2] = torch.bitwise_xor(rel_bytes[2], 0x80)
        rel_bytes[3, 0] = 0x80
        zero = ops.mixed_precision_ops({}, compute_dtype=torch.bfloat16).Linear(256, 48 if fused else 16, bias=False)
        zero.load_state_dict(state, strict=True)
        values = zero(inputs).float()
        assert bool(torch.isfinite(values).all()) and values[0, 0].item() == values[0, 3].item() == 0
        assert torch.allclose(values[0, 1:3], -baseline[0, 1:3], atol=0.05, rtol=0.05)
print("FLUX_W4_CPU_CONVERSION_OK")
'''
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONUTF8="1")
        result = subprocess.run(  # noqa: S603 -- fixed local interpreter and embedded CPU-only test
            [str(core.python_path()), "-X", "utf8", "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=150,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("FLUX_W4_CPU_CONVERSION_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
