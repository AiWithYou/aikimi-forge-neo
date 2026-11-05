"""Verify CPU-safe serialization against the installed official Irodori runtime."""

import os
import subprocess
import unittest
from pathlib import Path

from modules_forge.irodori.core import python_path

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(python_path().is_file(), "Irodori runtime is not installed")
class QuantRuntimeTests(unittest.TestCase):
    def test_upstream_int8_and_fp8_weight_only_roundtrip_on_cpu(self):
        script = r'''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path("repositories/irodori/source").resolve()))
import torch
torch.cuda.is_available = lambda: False
from safetensors.torch import save_file
from irodori_tts.inference_runtime import _load_checkpoint_from_safetensors
from irodori_tts.quantization import _build_torchao_config, flatten_quantized_state_dict
for mode in ("int8_weight_only", "int8_dynamic_activation_int8_weight", "float8_weight_only"):
    linear = torch.nn.Sequential(torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16))
    quantize, config = _build_torchao_config(mode)
    quantize(linear, config)
    tensors, metadata = flatten_quantized_state_dict(
        linear.state_dict(), base_metadata={"config_json": json.dumps({})},
        quantization_type=mode, profile="core", compute_dtype=torch.bfloat16, quantized_modules=1,
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "model.safetensors"
        save_file(tensors, str(path), metadata=metadata)
        loaded, _, _, _ = _load_checkpoint_from_safetensors(path)
        assert type(loaded["0.weight"]).__module__.startswith("torchao.")
        restored = torch.nn.Sequential(torch.nn.Linear(256, 64, bias=False, dtype=torch.bfloat16))
        restored.load_state_dict(loaded, assign=True)
        value = restored(torch.ones((1, 256), dtype=torch.bfloat16))
        assert bool(torch.isfinite(value).all())
print("IRODORI_CPU_QUANT_OK")
'''
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        result = subprocess.run(  # noqa: S603 -- Fixed local interpreter and embedded CPU-only program.
            [str(python_path()), "-X", "utf8", "-c", script],
            cwd=ROOT, env=environment, capture_output=True, text=True, encoding="utf-8", timeout=120, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("IRODORI_CPU_QUANT_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
