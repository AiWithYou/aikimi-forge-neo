"""Root layers use the same patch keys as their PyTorch state dictionaries."""

import unittest

import torch

from backend import operations
from backend.patcher.base import ModelPatcher


class RootLayerPatchTests(unittest.TestCase):
    def make_patcher(self, online=False):
        cpu = torch.device("cpu")
        with operations.using_forge_operations(device=cpu, dtype=torch.float32, manual_cast_enabled=True):
            model = torch.nn.Linear(3, 2)
        self.original_weight = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.original_bias = torch.tensor([1.0, 2.0])
        model.load_state_dict({"weight": self.original_weight.clone(), "bias": self.original_bias.clone()})
        self.delta_weight = torch.tensor([[0.5, -0.5, 1.0], [1.0, 0.5, -1.0]])
        self.delta_bias = torch.tensor([0.5, -0.5])
        patcher = ModelPatcher(model, load_device=cpu, offload_device=cpu)
        self.assertEqual(
            set(patcher.add_patches({"weight": (self.delta_weight,), "bias": (self.delta_bias,)}, online_mode=online)),
            {"weight", "bias"},
        )
        return patcher

    def assert_output(self, model, patched):
        inputs = torch.tensor([[1.0, 0.5, -1.0], [-2.0, 1.0, 0.25]])
        weight = self.original_weight + self.delta_weight if patched else self.original_weight
        bias = self.original_bias + self.delta_bias if patched else self.original_bias
        with torch.no_grad():
            torch.testing.assert_close(model(inputs), torch.nn.functional.linear(inputs, weight, bias))

    def test_full_load_then_partial_offload_keeps_root_weight_and_bias_patches(self):
        patcher = self.make_patcher()
        patcher.patch_model(device_to=torch.device("cpu"))
        self.assert_output(patcher.model, patched=True)
        self.assertGreater(patcher.partially_unload(torch.device("cpu"), memory_to_free=1), 0)
        self.assertEqual(len(patcher.model.weight_function), 1)
        self.assertEqual(len(patcher.model.bias_function), 1)
        self.assert_output(patcher.model, patched=True)
        patcher.unpatch_model()
        self.assert_output(patcher.model, patched=False)

    def test_low_memory_load_applies_root_patches_during_forward(self):
        patcher = self.make_patcher()
        patcher.patch_model(device_to=torch.device("cpu"), lowvram_model_memory=1)
        self.assertEqual(len(patcher.model.weight_function), 1)
        self.assertEqual(len(patcher.model.bias_function), 1)
        self.assert_output(patcher.model, patched=True)
        patcher.unpatch_model()
        self.assert_output(patcher.model, patched=False)

    def test_online_root_patches_keep_storage_unmodified(self):
        patcher = self.make_patcher(online=True)
        patcher.patch_model(device_to=torch.device("cpu"))
        self.assert_output(patcher.model, patched=True)
        torch.testing.assert_close(patcher.model.weight, self.original_weight)
        torch.testing.assert_close(patcher.model.bias, self.original_bias)
        patcher.unpatch_model()
        self.assert_output(patcher.model, patched=False)


if __name__ == "__main__":
    unittest.main()
