"""Actual CPU loading, clone fallback and online patch lifetime integration."""

from __future__ import annotations

import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F

from backend import memory_management, operations
from backend.patcher.base import ModelPatcher
from backend.quant_ops import QuantizedTensor
from modules_forge.packages.comfy.weight_adapter.lora import LoRAAdapter


class PatcherCloneIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(torch.no_grad())
        self.enterContext(torch.random.fork_rng(devices=[]))
        self.enterContext(patch.object(memory_management, "current_loaded_models", []))
        self.enterContext(patch("modules_forge.gpu_residency.release_resource"))
        self.cpu = torch.device("cpu")
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.filename = Path(self.directory.name) / "複製 bias 重み.pt"
        self.qdata = torch.tensor([[127, 0, -127, 127], [-127, 127, 0, -127]], dtype=torch.int8)
        self.weight = self.qdata.float() * 0.25
        self.bias = torch.tensor([0.5, -1.0])
        self.inputs = torch.tensor([[1.0, 0.0, -0.5, 0.25], [-0.25, 1.0, 0.5, -1.0]])
        self.addCleanup(self.release_models)

    @staticmethod
    def release_models():
        for loaded in list(memory_management.current_loaded_models):
            if loaded.model is not None:
                loaded.model_unload()
        memory_management.current_loaded_models.clear()

    def make_patcher(self, quantized=False):
        if quantized:
            factory = operations.mixed_precision_ops(compute_dtype=torch.float32, full_precision_mm=True)
            model = factory.Linear(4, 2, bias=True, device=self.cpu)
            state = {
                "weight": self.qdata.clone(),
                "bias": self.bias.clone(),
                "weight_scale": torch.tensor(0.25),
                "comfy_quant": torch.tensor(list(b'{"format":"int8_tensorwise"}'), dtype=torch.uint8),
            }
        else:
            with operations.using_forge_operations(device=self.cpu, dtype=torch.float32, manual_cast_enabled=True):
                model = torch.nn.Linear(4, 2)
            state = {"weight": self.weight.clone(), "bias": self.bias.clone()}
        torch.save(state, self.filename)
        model.load_state_dict(torch.load(self.filename, map_location="cpu", weights_only=True))
        return ModelPatcher(model, load_device=self.cpu, offload_device=self.cpu)

    def add_online(self, model, scale):
        weight = self.weight * scale
        bias = self.bias * scale
        model.add_patches({"weight": (weight,), "bias": (bias,)}, online_mode=True)
        return weakref.ref(weight), weakref.ref(bias)

    def assert_forward(self, patcher, multiplier, dtype=torch.float32):
        actual = patcher.model(self.inputs.to(dtype))
        expected = F.linear(
            self.inputs.to(dtype), (self.weight * multiplier).to(dtype), (self.bias * multiplier).to(dtype)
        )
        torch.testing.assert_close(actual, expected)

    def test_parent_child_grandchild_reloads_keep_online_weight_and_bias_isolated(self):
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                parent = self.make_patcher(quantized)
                child = parent.clone()
                self.add_online(child, 0.5)
                grandchild = child.clone()
                self.add_online(grandchild, -0.25)
                for patcher, multiplier in ((child, 1.5), (grandchild, 1.25), (child, 1.5), (parent, 1.0)):
                    memory_management.load_models_gpu([patcher], force_full_load=True)
                    self.assertEqual(len(memory_management.current_loaded_models), 1)
                    self.assertIs(memory_management.current_loaded_models[0].model, patcher)
                    self.assert_forward(patcher, multiplier)
                    self.assert_forward(patcher, multiplier, torch.bfloat16)
                    torch.testing.assert_close(patcher.model.bias, self.bias)
                    if quantized:
                        self.assertIsInstance(patcher.model.weight, QuantizedTensor)
                        torch.testing.assert_close(patcher.model.weight._qdata, self.qdata)
                        torch.testing.assert_close(patcher.model.state_dict()["weight_scale"], torch.tensor(0.25))
                    else:
                        torch.testing.assert_close(patcher.model.weight, self.weight)
                self.release_models()

    def test_collected_clone_chain_falls_back_and_releases_online_tensors_after_parent_reload(self):
        for quantized in (False, True):
            with self.subTest(quantized=quantized):
                parent = self.make_patcher(quantized)
                child = parent.clone()
                child_tensors = self.add_online(child, 0.5)
                grandchild = child.clone()
                grandchild_tensors = self.add_online(grandchild, 0.25)
                references = weakref.ref(child), weakref.ref(grandchild)
                memory_management.load_models_gpu([grandchild], force_full_load=True)
                self.assert_forward(grandchild, 1.75)
                loaded = memory_management.current_loaded_models[0]
                del child, grandchild
                gc.collect()

                self.assertTrue(all(reference() is None for reference in references))
                self.assertIs(loaded.model, parent)
                memory_management.load_models_gpu([parent], force_full_load=True)
                self.assert_forward(parent, 1.0)
                self.assertTrue(all(reference() is None for reference in child_tensors + grandchild_tensors))
                self.assertEqual(parent.model.weight_function, [])
                self.assertEqual(parent.model.bias_function, [])
                self.release_models()

    def test_removed_loaded_wrapper_dies_while_its_patcher_and_real_model_are_alive(self):
        parent = self.make_patcher()
        child = parent.clone()
        self.add_online(child, 0.5)
        memory_management.load_models_gpu([child], force_full_load=True)
        reference = weakref.ref(memory_management.current_loaded_models[0])
        self.assertTrue(memory_management.unload_model(child))
        gc.collect()

        self.assertIsNone(reference())
        self.assert_forward(child, 1.0)
        self.assertEqual(child.model.weight_function, [])
        self.assertEqual(child.model.bias_function, [])
        memory_management.load_models_gpu([child], force_full_load=True)
        self.assert_forward(child, 1.5)

    def test_partial_unload_preserves_offline_then_online_patch_order(self):
        for quantized, dora in ((False, False), (False, True), (True, False)):
            with self.subTest(quantized=quantized, dora=dora):
                parent = self.make_patcher(quantized)
                if dora:
                    up = torch.tensor([[1.0], [2.0]])
                    down = torch.tensor([[0.5, -0.25, 0.25, 0.5]])
                    scale = self.weight.norm(dim=1, keepdim=True)
                    adapter = LoRAAdapter(set(), (up, down, 1.0, None, scale, None))
                    parent.add_patches({"weight": adapter})
                    adapted = self.weight + up @ down
                    offline_weight = adapted * (
                        scale / (adapted.norm(dim=1, keepdim=True) + torch.finfo(torch.float32).eps)
                    )
                else:
                    parent.add_patches({"weight": (self.weight * 0.5,), "bias": (self.bias * 0.5,)}, strength_model=0.5)
                    offline_weight = self.weight
                child = parent.clone()
                self.add_online(child, 0.25)
                memory_management.load_models_gpu([child], force_full_load=True)
                expected = F.linear(self.inputs, offline_weight + self.weight * 0.25, self.bias * 1.25)
                torch.testing.assert_close(child.model(self.inputs), expected)
                child.partially_unload(self.cpu, memory_to_free=1)
                torch.testing.assert_close(child.model(self.inputs), expected)
                child.partially_load(self.cpu, extra_memory=1e9)
                torch.testing.assert_close(child.model(self.inputs), expected)
                self.release_models()


if __name__ == "__main__":
    unittest.main()
