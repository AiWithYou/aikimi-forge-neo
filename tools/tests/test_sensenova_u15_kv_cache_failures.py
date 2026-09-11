"""CUDAを模擬し、転送と後始末の失敗時の所有権を検証する。"""

import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from modules_forge.sensenova_u15_kv_cache import LayerwiseKVOffload, _env_float, _Visit


class Cache:
    def __init__(self, keys, values):
        self.layers = [SimpleNamespace(keys=keys, values=values)]


def tensor():
    value = mock.Mock()
    value.device.type = "cpu"
    value.numel.return_value = 16
    value.element_size.return_value = 4
    value.is_pinned.return_value = True
    value.to.return_value = value
    return value


class FailureTests(unittest.TestCase):
    def setUp(self):
        self.layer = nn.Identity()
        self.original = mock.Mock()
        self.module = SimpleNamespace(prepare_flash_kv_cache=self.original)
        self.manager = LayerwiseKVOffload([self.layer], self.module, "cpu")
        self.addCleanup(self.manager.close)
        self.cache = Cache(tensor(), tensor())

    def promote(self):
        self.manager._adaptive_enabled = True
        with mock.patch.object(torch.cuda, "mem_get_info", return_value=(16 * 1024**3, 24 * 1024**3)):
            self.manager._promote(self.cache)

    def test_promotion_is_atomic_and_not_double_counted(self):
        keys, values = self.cache.layers[0].keys, self.cache.layers[0].values
        keys.to.return_value = tensor()
        values.to.return_value = tensor()
        values.to.side_effect = torch.OutOfMemoryError("test OOM")
        self.promote()
        self.assertIs(self.cache.layers[0].keys, keys)
        self.assertIs(self.cache.layers[0].values, values)
        self.assertEqual(self.manager._resident_bytes, 0)
        self.assertEqual(self.manager.transferred_bytes, 0)
        values.to.side_effect = None
        self.promote()
        self.promote()
        self.assertEqual(self.manager._resident_bytes, 128)
        self.assertEqual(self.manager.transferred_bytes, 128)

    def test_promotion_unexpected_error_propagates(self):
        self.cache.layers[0].values.to.side_effect = RuntimeError("bad shape")
        with self.assertRaisesRegex(RuntimeError, "bad shape"):
            self.promote()
        self.assertEqual(self.manager._resident_bytes, 0)

    def schedule_failure(self, error, sync_error=None):
        manager = self.manager
        manager._prefetch_enabled = True
        manager._prefetch_stream = mock.Mock()
        manager._prefetch_stream.synchronize.side_effect = sync_error
        self.cache.layers[0].values.to.side_effect = error
        with mock.patch.object(torch.cuda, "stream", return_value=nullcontext()):
            manager._schedule_prefetch(self.cache, 0)

    def test_prefetch_oom_disables_retries_and_sync_transfer_recovers(self):
        keys, values = self.cache.layers[0].keys, self.cache.layers[0].values
        self.schedule_failure(torch.OutOfMemoryError("test OOM"))
        self.manager._prefetch_stream.synchronize.assert_called_once()
        self.assertFalse(self.manager._prefetch_enabled)
        self.assertFalse(self.manager._pending_sources)
        self.assertFalse(self.manager._prefetched)
        self.assertEqual(self.manager.prefetch_fallbacks, 1)
        self.assertIn("test OOM", self.manager.telemetry["prefetch_fallback_reasons"][0])
        self.manager._schedule_prefetch(self.cache, 0)
        self.assertEqual(values.to.call_count, 1)
        values.to.side_effect = None
        self.assertEqual(self.manager._prefix_on_device(self.cache, 0, self.cache.layers[0]), (keys, values))
        self.assertIs(self.cache.layers[0].keys, keys)
        self.assertIs(self.cache.layers[0].values, values)

    def test_prefetch_non_oom_propagates_after_sync(self):
        with self.assertRaisesRegex(RuntimeError, "bad copy"):
            self.schedule_failure(RuntimeError("bad copy"))
        self.manager._prefetch_stream.synchronize.assert_called_once()

    def test_failed_sync_retains_sources_until_close_retry(self):
        with self.assertRaises(ExceptionGroup) as caught:
            self.schedule_failure(torch.OutOfMemoryError("OOM"), RuntimeError("sync"))
        self.assertEqual(len(caught.exception.exceptions), 2)
        self.assertEqual(len(self.manager._pending_sources), 2)
        self.manager._prefetch_stream.synchronize.side_effect = None
        self.manager.close()
        self.assertFalse(self.manager._pending_sources)

    def test_take_registers_both_consumers(self):
        keys, values, event, stream = tensor(), tensor(), mock.Mock(), mock.Mock()
        self.manager._prefetched[(id(self.cache), 0)] = SimpleNamespace(keys=keys, values=values, event=event)
        with mock.patch.object(torch.cuda, "current_stream", return_value=stream):
            self.assertEqual(self.manager._take_prefetch(self.cache, 0), (keys, values))
        event.synchronize.assert_called_once()
        keys.record_stream.assert_called_once_with(stream)
        values.record_stream.assert_called_once_with(stream)
        self.assertFalse(self.manager._prefetched)

    def test_cleanup_collects_errors_restores_function_and_retries(self):
        handle = mock.Mock()
        handle.remove.side_effect = RuntimeError("hook")
        self.manager._hooks.insert(0, handle)
        self.manager._prefetch_stream = mock.Mock()
        self.manager._prefetched[(id(self.cache), 0)] = object()
        entry = self.cache.layers[0]
        entry.keys.device.type = "cuda"
        entry.keys.to.side_effect = RuntimeError("offload")
        entry.flash_k_cache = entry.flash_v_cache = object()
        self.manager._active[id(self.layer)].append(_Visit(self.cache, 0, True, {}))
        with self.assertRaises(ExceptionGroup) as caught:
            self.manager.close()
        self.assertEqual(len(caught.exception.exceptions), 2)
        self.assertIs(self.module.prepare_flash_kv_cache, self.original)
        self.assertFalse(self.layer._forward_hooks)
        self.assertFalse(self.layer._forward_pre_hooks)
        self.assertIsNone(entry.flash_k_cache)
        self.assertIsNone(entry.flash_v_cache)
        self.assertFalse(self.manager._prefetched)
        self.manager._prefetch_stream.synchronize.assert_called_once()
        handle.remove.side_effect = None
        entry.keys.device.type = "cpu"
        self.manager.close()
        self.manager.close()
        self.assertFalse(self.manager._hooks)
        self.assertFalse(any(self.manager._active.values()))

    def test_demotion_continues_with_value_and_retries_failed_key(self):
        entry = self.cache.layers[0]
        keys, values = entry.keys, entry.values
        keys.device.type = values.device.type = "cuda"
        keys.to.side_effect = RuntimeError("key offload")
        values.to.return_value = tensor()
        self.manager._remember_size(self.cache, 0, entry)
        self.manager._resident_indices(self.cache).add(0)
        self.manager._resident_bytes = 128
        with self.assertRaises(ExceptionGroup):
            self.manager.close()
        self.assertIs(entry.keys, keys)
        self.assertIs(entry.values, values.to.return_value)
        self.assertEqual(self.manager._resident_bytes, 128)
        self.assertIs(self.module.prepare_flash_kv_cache, self.original)
        keys.to.side_effect = None
        keys.to.return_value = tensor()
        self.manager.close()
        self.assertEqual(self.manager._resident_bytes, 0)

    def test_invalid_and_bounded_config(self):
        for raw, expected in [("nan", 2), ("inf", 2), ("-inf", 2), ("bad", 2), ("-1", 0), ("99", 8), ("3.5", 3.5)]:
            with self.subTest(raw=raw), mock.patch.dict(os.environ, {"KV_TEST": raw}):
                self.assertEqual(_env_float("KV_TEST", 2, 0, 8), expected)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA実機が必要")
class CudaLifetimeTests(unittest.TestCase):
    def test_cross_stream_allocator_reuse_and_close(self):
        module = SimpleNamespace(prepare_flash_kv_cache=lambda *args: None)
        original = module.prepare_flash_kv_cache
        manager = LayerwiseKVOffload([nn.Identity()], module, "cuda")
        cache = Cache(torch.arange(262144, dtype=torch.float32), torch.full((262144,), 7.0))
        consumer = torch.cuda.Stream()
        try:
            for _ in range(12):
                manager._schedule_prefetch(cache, 0)
                with torch.cuda.stream(consumer):
                    keys, values = manager._take_prefetch(cache, 0)
                    torch.cuda._sleep(2_000_000)
                    actual_k, actual_v = keys.clone(), values.clone()
                    del keys, values
                with torch.cuda.stream(manager._prefetch_stream):
                    trash = [torch.empty_like(actual_k).fill_(-123) for _ in range(8)]
                consumer.synchronize()
                self.assertTrue(torch.equal(actual_k.cpu(), cache.layers[0].keys))
                self.assertTrue(torch.equal(actual_v.cpu(), cache.layers[0].values))
                del trash
            manager._schedule_prefetch(cache, 0)
        finally:
            manager.close()
        manager.close()
        self.assertFalse(manager._prefetched)
        self.assertIs(module.prepare_flash_kv_cache, original)
