"""Root and nested layers keep equivalent patching and quantization metadata."""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from backend import operations
from backend.patcher.base import ModelPatcher
from backend.quant_ops import QuantizedTensor


class LayerContainer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, values):
        return self.layer(values)


class RootQuantizedPatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.filename = Path(self.temporary.name) / "量子化 重み.pt"
        self.cpu = torch.device("cpu")
        self.quantized_weight = torch.tensor([[127, 0, -127, 127], [-127, 127, 0, -127]], dtype=torch.int8)
        self.original_weight = self.quantized_weight.float() * 0.25
        self.values = torch.tensor([[0.2, 0.5, -0.75, 1.1], [-0.15, 1.2, 0.3, -0.2]])

    def saved_state(self, state):
        torch.save(state, self.filename)
        return torch.load(self.filename, map_location="cpu", weights_only=True)

    def make_quantized_patcher(self, nested=False, online=False):
        factory = operations.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True)
        layer = factory.Linear(4, 2, bias=False, device=self.cpu)
        layer.load_state_dict(
            self.saved_state(
                {
                    "weight": self.quantized_weight,
                    "weight_scale": torch.tensor(0.25),
                    "comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8),
                }
            )
        )
        model = LayerContainer(layer) if nested else layer
        patcher = ModelPatcher(model, load_device=self.cpu, offload_device=self.cpu)
        key = "layer.weight" if nested else "weight"
        self.assertEqual(
            patcher.add_patches({key: (self.original_weight.clone(),)}, strength_patch=0.5, online_mode=online), [key]
        )
        return patcher, layer, key

    def assert_forward(self, model, multiplier):
        with torch.no_grad():
            torch.testing.assert_close(
                model(self.values), torch.nn.functional.linear(self.values, self.original_weight * multiplier)
            )

    def test_offline_root_and_nested_patches_keep_quantized_storage_and_reloadable_metadata(self):
        for nested in (False, True):
            with self.subTest(nested=nested):
                patcher, layer, _ = self.make_quantized_patcher(nested)
                try:
                    patcher.patch_model(device_to=self.cpu)
                    self.assertIsInstance(layer.weight, QuantizedTensor)
                    self.assert_forward(patcher.model, 1.5)
                    saved = self.saved_state(layer.state_dict())
                    self.assertEqual(json.loads(saved["comfy_quant"].numpy().tobytes())["format"], "int8_tensorwise")
                    torch.testing.assert_close(saved["weight"], self.quantized_weight)
                    torch.testing.assert_close(saved["weight_scale"], torch.tensor(0.375))
                    restored = operations.mixed_precision_ops(
                        compute_dtype=torch.float32, full_precision_mm=True
                    ).Linear(4, 2, bias=False, device=self.cpu)
                    restored.load_state_dict(saved)
                    self.assert_forward(restored, 1.5)
                finally:
                    patcher.unpatch_model()
                self.assert_forward(patcher.model, 1.0)
                torch.testing.assert_close(layer.state_dict()["weight_scale"], torch.tensor(0.25))

    def test_root_and_nested_online_or_low_memory_patches_restore_original_quantized_weight(self):
        for nested in (False, True):
            for online in (False, True):
                with self.subTest(nested=nested, online=online):
                    patcher, layer, _ = self.make_quantized_patcher(nested, online)
                    try:
                        patcher.patch_model(device_to=self.cpu, lowvram_model_memory=0 if online else 1)
                        self.assert_forward(patcher.model, 1.5)
                        self.assertIsInstance(layer.weight, QuantizedTensor)
                        torch.testing.assert_close(layer.state_dict()["weight_scale"], torch.tensor(0.25))
                        torch.testing.assert_close(layer.weight._qdata, self.quantized_weight)
                    finally:
                        patcher.unpatch_model()
                    self.assert_forward(patcher.model, 1.0)
                    self.assertEqual(layer.weight_function, [])

    def test_nested_native_and_forge_layers_still_patch_and_restore(self):
        native_linear = torch.nn.Linear
        for forge in (False, True):
            with self.subTest(forge=forge):
                if forge:
                    with operations.using_forge_operations(
                        device=self.cpu, dtype=torch.float32, manual_cast_enabled=True
                    ):
                        layer = torch.nn.Linear(4, 2, bias=False)
                else:
                    layer = native_linear(4, 2, bias=False)
                layer.load_state_dict(self.saved_state({"weight": self.original_weight.clone()}))
                model = LayerContainer(layer)
                patcher = ModelPatcher(model, load_device=self.cpu, offload_device=self.cpu)
                patcher.add_patches({"layer.weight": (self.original_weight.clone(),)}, strength_patch=0.5)
                try:
                    patcher.patch_model(device_to=self.cpu)
                    self.assert_forward(model, 1.5)
                finally:
                    patcher.unpatch_model()
                self.assert_forward(model, 1.0)


if __name__ == "__main__":
    unittest.main()
