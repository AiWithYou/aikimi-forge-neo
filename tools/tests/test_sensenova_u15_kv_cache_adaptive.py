import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from modules_forge.sensenova_u15_kv_cache import LayerwiseKVOffload


class Cache:
    def __init__(self):
        self.layers = [SimpleNamespace(keys=None, values=None)]


class Layer(nn.Module):
    def forward(self, x, past_key_values=None, update_cache=True, attention_mask=None):
        layer = past_key_values.layers[0]
        if update_cache:
            layer.keys = x.clone()
            layer.values = x.clone()
            return x
        prefix = layer.flash_prefix_len
        layer.flash_k_cache[:, prefix:].copy_(x.transpose(1, 2))
        layer.flash_v_cache[:, prefix:].copy_(x.transpose(1, 2))
        return x


def native_prepare(cache, current_len, batch_size):
    layer = cache.layers[0]
    prefix = layer.keys.shape[2]
    shape = (batch_size, prefix + current_len, layer.keys.shape[1], layer.keys.shape[3])
    layer.flash_prefix_len = prefix
    layer.flash_total_len = prefix + current_len
    layer.flash_k_cache = torch.empty(shape, dtype=layer.keys.dtype)
    layer.flash_v_cache = torch.empty(shape, dtype=layer.values.dtype)


class AdaptiveKVPolicyTests(unittest.TestCase):
    def test_cpu_keeps_legacy_layerwise_policy(self):
        layer = Layer()
        module = SimpleNamespace(prepare_flash_kv_cache=native_prepare)
        cache = Cache()
        manager = LayerwiseKVOffload(nn.ModuleList([layer]), module, "cpu")
        try:
            prefix = torch.randn(1, 2, 4, 3)
            current = torch.randn(1, 2, 2, 3)
            layer(prefix, past_key_values=cache)
            module.prepare_flash_kv_cache(cache, 2, 1)
            layer(current, past_key_values=cache, update_cache=False)
            telemetry = manager.telemetry
            self.assertEqual(telemetry["mode"], "layerwise_cpu")
            self.assertFalse(telemetry["adaptive_enabled"])
            self.assertFalse(telemetry["prefetch_enabled"])
        finally:
            manager.close()
        self.assertIs(module.prepare_flash_kv_cache, native_prepare)

    def test_environment_switches_support_ab_measurement(self):
        layer = Layer()
        module = SimpleNamespace(prepare_flash_kv_cache=native_prepare)
        with mock.patch.dict(
            os.environ,
            {
                "AIKIMI_SENSENOVA_KV_ADAPTIVE": "0",
                "AIKIMI_SENSENOVA_KV_PREFETCH": "0",
            },
            clear=False,
        ):
            manager = LayerwiseKVOffload(nn.ModuleList([layer]), module, "cpu")
        try:
            telemetry = manager.telemetry
            self.assertFalse(telemetry["adaptive_requested"])
            self.assertFalse(telemetry["prefetch_requested"])
        finally:
            manager.close()


if __name__ == "__main__":
    unittest.main()
