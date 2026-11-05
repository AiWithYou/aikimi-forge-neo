"""Small CPU integration test against the installed MOSS runtime."""

import subprocess
import unittest
from pathlib import Path

from modules_forge.moss_image import core

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    core.python_path().is_file() and (core.runtime_paths()["comfy"] / "comfy/sd.py").is_file(),
    "The MOSS runtime is not installed",
)
class MossW4A8RuntimeTests(unittest.TestCase):
    def test_packed_linear_and_lora_zero_one_zero(self):
        script = r'''
import importlib.metadata, json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path("repositories/moss-image/ComfyUI").resolve()))
sys.argv = ["moss_w4_cpu", "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.lora
from comfy import ops
from comfy.model_patcher import ModelPatcher
from comfy_kitchen.tensor import AsymW4A8Int8Layout
from safetensors.torch import save_file
from modules_forge.moss_image import lora

assert importlib.metadata.version("comfy-kitchen") == "0.2.39"
torch.manual_seed(7)
weight = torch.randn((48, 256), dtype=torch.bfloat16) * 0.1
packed, params = AsymW4A8Int8Layout.quantize(weight)
assert params.correction is None
config = {"format": "asym_w4a8_int8", "group_size": 16, "convrot_groupsize": 256}
state = {
    "weight": packed, "weight_s_rel": params.scale, "weight_s_channel": params.s_channel,
    "weight_codebook": params.codebook,
    "comfy_quant": torch.tensor(list(json.dumps(config).encode()), dtype=torch.uint8),
}

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_model = torch.nn.Module()
        self.diffusion_model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.diffusion_model.layers[0].proj = torch.nn.Module()
        self.diffusion_model.layers[0].proj.qkv = ops.mixed_precision_ops(
            {}, compute_dtype=torch.bfloat16
        ).Linear(256, 48, bias=False)
    def forward(self, value):
        return self.diffusion_model.layers[0].proj.qkv(value)

model = Toy()
linear = model.diffusion_model.layers[0].proj.qkv
linear.load_state_dict(state, strict=True)
assert tuple(linear.state_dict()["weight"].shape) == (48, 128)
assert tuple(linear.weight.shape) == (48, 256)
patcher = ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
target = ("diffusion_model.layers.0.proj.qkv.weight", (0, 0, 16))
mapper = SimpleNamespace(
    model_lora_keys_unet=lambda model, keys: {"diffusion_model.layers.0.proj.q": target},
    load_lora=comfy.lora.load_lora,
)
inputs = torch.ones((1, 256), dtype=torch.bfloat16)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "loras").mkdir()
    prefix = "layers.0.proj.q"
    save_file({
        prefix + ".lora_A.weight": torch.ones((4, 256)) * 0.1,
        prefix + ".lora_B.weight": torch.ones((16, 4)) * 0.1,
    }, str(root / "loras" / "toy.safetensors"))
    def request(strength):
        return {"loras": [{"name": "toy.safetensors", "strength": strength}]}
    initial, count = lora.apply(patcher, root, request(0), mapper, torch)
    assert count == 0
    first = initial.model(inputs).float()
    patched, count = lora.apply(patcher, root, request(1), mapper, torch)
    assert count == 1
    patched.patch_model(device_to=torch.device("cpu"), force_patch_weights=True)
    changed = patched.model(inputs).float()
    patched.unpatch_model(device_to=torch.device("cpu"))
    restored, count = lora.apply(patcher, root, request(0), mapper, torch)
    last = restored.model(inputs).float()
    assert count == 0 and bool(torch.isfinite(changed).all())
    assert not torch.equal(first, changed)
    assert torch.equal(first, last)
print("W4A8_CPU_LORA_OK")
'''
        result = subprocess.run(  # noqa: S603 -- Fixed local interpreter and embedded CPU-only test program.
            [str(core.python_path()), "-X", "utf8", "-c", script],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("W4A8_CPU_LORA_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
