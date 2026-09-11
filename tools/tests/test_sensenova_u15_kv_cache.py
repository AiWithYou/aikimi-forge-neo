import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from modules_forge.sensenova_u15_kv_cache import LayerwiseKVOffload


class Cache:
    def __init__(self, count):
        self.layers = [SimpleNamespace(keys=None, values=None) for _ in range(count)]


class Layer(nn.Module):
    def __init__(self, index, check=None):
        super().__init__()
        self.index, self.check = index, check

    def forward(self, x, past_key_values=None, update_cache=True, attention_mask=None):
        if self.check:
            self.check(past_key_values, self.index)
        layer = past_key_values.layers[self.index]
        if update_cache:
            layer.keys = x.clone() if layer.keys is None else torch.cat([layer.keys, x], dim=2)
            layer.values = x.clone() if layer.values is None else torch.cat([layer.values, x], dim=2)
            k, v = layer.keys, layer.values
        else:
            prefix = layer.flash_prefix_len
            layer.flash_k_cache[:, prefix:].copy_(x.transpose(1, 2))
            layer.flash_v_cache[:, prefix:].copy_(x.transpose(1, 2))
            k, v = layer.flash_k_cache.transpose(1, 2), layer.flash_v_cache.transpose(1, 2)
        return torch.nn.functional.scaled_dot_product_attention(x, k, v)


def native_prepare(cache, current_len, batch_size):
    for layer in cache.layers:
        prefix = layer.keys.shape[2]
        shape = (batch_size, prefix + current_len, layer.keys.shape[1], layer.keys.shape[3])
        layer.flash_prefix_len = prefix
        layer.flash_total_len = prefix + current_len
        layer.flash_k_cache = torch.empty(shape, dtype=layer.keys.dtype)
        layer.flash_v_cache = torch.empty(shape, dtype=layer.values.dtype)
        layer.flash_k_cache[:, :prefix].copy_(layer.keys.transpose(1, 2))
        layer.flash_v_cache[:, :prefix].copy_(layer.values.transpose(1, 2))


class KVOffloadTests(unittest.TestCase):
    def test_prefix_and_denoising_are_identical_to_full_cache(self):
        torch.manual_seed(42)
        prefix = torch.randn(1, 2, 7, 4)
        current = torch.randn(1, 2, 3, 4)
        baseline = Cache(3)
        layers = nn.ModuleList([Layer(i) for i in range(3)])
        for layer in layers:
            layer(prefix, past_key_values=baseline)
        native_prepare(baseline, 3, 1)
        expected = [layer(current, past_key_values=baseline, update_cache=False) for layer in layers]

        module = SimpleNamespace(prepare_flash_kv_cache=native_prepare)
        cache = Cache(3)
        manager = LayerwiseKVOffload(layers, module, "cpu")
        try:
            for layer in layers:
                layer(prefix, past_key_values=cache)
            original = [item.keys for item in cache.layers]
            module.prepare_flash_kv_cache(cache, 3, 1)
            self.assertTrue(all(item.flash_k_cache is None for item in cache.layers))
            for _ in range(2):
                for i, layer in enumerate(layers):
                    actual = layer(current, past_key_values=cache, update_cache=False)
                    self.assertTrue(torch.equal(actual, expected[i]))
                    self.assertTrue(all(item.flash_k_cache is None for item in cache.layers))
                    self.assertIs(cache.layers[i].keys, original[i])
            self.assertEqual(manager.flash_bytes_peak, 2 * 1 * 10 * 2 * 4 * 4)
        finally:
            manager.close()
        self.assertIs(module.prepare_flash_kv_cache, native_prepare)

    def test_only_current_layer_gets_flash_buffers(self):
        def check(cache, index):
            for i, layer in enumerate(cache.layers):
                self.assertEqual(layer.flash_k_cache is not None, i == index)

        layers = nn.ModuleList([Layer(i, check) for i in range(3)])
        cache = Cache(3)
        for layer in cache.layers:
            layer.keys = torch.ones(1, 2, 5, 4)
            layer.values = torch.ones(1, 2, 5, 4)
        module = SimpleNamespace(prepare_flash_kv_cache=native_prepare)
        manager = LayerwiseKVOffload(layers, module, "cpu")
        try:
            module.prepare_flash_kv_cache(cache, 3, 1)
            for layer in layers:
                layer(torch.ones(1, 2, 3, 4), past_key_values=cache, update_cache=False)
        finally:
            manager.close()

    def test_allocation_failure_releases_partial_workspace(self):
        layers = nn.ModuleList([Layer(0)])
        cache = Cache(1)
        cache.layers[0].keys = torch.ones(1, 2, 5, 4)
        cache.layers[0].values = torch.ones(1, 2, 5, 4)
        saved = copy.deepcopy(cache.layers[0].keys)
        module = SimpleNamespace(prepare_flash_kv_cache=native_prepare)
        manager = LayerwiseKVOffload(layers, module, "cpu")
        try:
            module.prepare_flash_kv_cache(cache, 3, 1)
            with mock.patch(
                "modules_forge.sensenova_u15_kv_cache.torch.empty",
                side_effect=[torch.empty(1, 8, 2, 4), RuntimeError("allocation failed")],
            ):
                with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                    layers[0](torch.ones(1, 2, 3, 4), past_key_values=cache, update_cache=False)
            self.assertIsNone(cache.layers[0].flash_k_cache)
            self.assertIsNone(cache.layers[0].flash_v_cache)
            self.assertTrue(torch.equal(cache.layers[0].keys, saved))
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
