"""参照KVをCPU主体で保持し、余剰VRAM常駐と1層先読みを適応的に使う。"""

from __future__ import annotations

import math
import os
import time
import weakref
from dataclasses import dataclass
from typing import Any

import torch

_GIB = 1024**3


@dataclass
class _Visit:
    cache: Any
    index: int
    update: bool
    original: dict[str, torch.Tensor]


@dataclass
class _Prefetch:
    keys: torch.Tensor
    values: torch.Tensor
    source_keys: torch.Tensor
    source_values: torch.Tensor
    event: Any
    size_bytes: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else raw.strip().casefold() not in {"", "0", "false", "no", "off"}


def _env_float(name: str, default: float, low: float, high: float) -> float:
    try:
        value = float(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = default
    if not math.isfinite(value):
        value = default
    return max(low, min(high, value))


class LayerwiseKVOffload:
    """固定版SenseNovaのKV寿命を管理し、referenceモードだけ適応高速化する。

    A/B/C実測用:
      A: AIKIMI_SENSENOVA_KV_ADAPTIVE=0
      B: AIKIMI_SENSENOVA_KV_ADAPTIVE=1, AIKIMI_SENSENOVA_KV_PREFETCH=0
      C: 両方1（既定）
    """

    def __init__(self, layers, prepare_module, device):
        self._layers = layers
        self._module = prepare_module
        self._device = torch.device(device)
        self._original_prepare = prepare_module.prepare_flash_kv_cache
        self._hooks = []
        self._active = {id(layer): [] for layer in layers}
        self._sizes = weakref.WeakKeyDictionary()
        self._resident: dict[int, set[int]] = {}
        self._cache_refs: dict[int, Any] = {}
        self._prefetched: dict[tuple[int, int], _Prefetch] = {}
        self._pending_sources = []
        self.prefetch_fallback_reasons = []
        self._closed = False

        self._adaptive_requested = _env_bool("AIKIMI_SENSENOVA_KV_ADAPTIVE", True)
        self._prefetch_requested = _env_bool("AIKIMI_SENSENOVA_KV_PREFETCH", True)
        self._adaptive_enabled = bool(
            self._adaptive_requested and self._device.type == "cuda" and torch.cuda.is_available()
        )
        self._prefetch_enabled = bool(self._adaptive_enabled and self._prefetch_requested)
        self._resident_fraction = _env_float("AIKIMI_SENSENOVA_KV_RESIDENT_FRACTION", 0.25, 0.0, 0.8)
        self._headroom = int(_env_float("AIKIMI_SENSENOVA_KV_HEADROOM_GIB", 6.0, 1.0, 32.0) * _GIB)
        self._resident_cap = int(_env_float("AIKIMI_SENSENOVA_KV_MAX_RESIDENT_GIB", 2.0, 0.0, 32.0) * _GIB)
        self._prefetch_stream = torch.cuda.Stream(device=self._device) if self._prefetch_enabled else None

        self.prefix_bytes_peak = 0
        self.flash_bytes_peak = 0
        self.transferred_bytes = 0
        self.synchronous_h2d_bytes = 0
        self.prefetch_h2d_bytes = 0
        self.resident_h2d_bytes = 0
        self.resident_budget_bytes = 0
        self.resident_bytes_peak = 0
        self.prefetch_bytes_peak = 0
        self.prefetch_scheduled = 0
        self.prefetch_hits = 0
        self.prefetch_misses = 0
        self.prefetch_fallbacks = 0
        self.residency_fallbacks = 0
        self.prefetch_wait_seconds = 0.0
        self.visits = 0
        self._resident_bytes = 0
        self._last_free_bytes = 0
        self._last_total_bytes = 0

        self._module.prepare_flash_kv_cache = self.prepare_flash_cache
        try:
            for index, layer in enumerate(layers):
                self._hooks.append(layer.register_forward_pre_hook(self._pre_hook(index), with_kwargs=True))
                self._hooks.append(layer.register_forward_hook(self._post_hook, with_kwargs=True, always_call=True))
        except Exception as error:
            try:
                self.close()
            except Exception as cleanup_error:
                raise ExceptionGroup("KVフック登録と後始末に失敗しました", [error, cleanup_error]) from None
            raise

    @staticmethod
    def _bytes(tensor):
        return tensor.numel() * tensor.element_size()

    def _remember_size(self, cache, index, layer):
        sizes = self._sizes.setdefault(cache, {})
        sizes[index] = sum(self._bytes(value) for value in (layer.keys, layer.values) if value is not None)
        self.prefix_bytes_peak = max(
            self.prefix_bytes_peak,
            sum(sum(entries.values()) for entries in self._sizes.values()),
        )

    def _resident_indices(self, cache) -> set[int]:
        cache_id = id(cache)
        self._cache_refs[cache_id] = cache
        return self._resident.setdefault(cache_id, set())

    def _clear_prefetch(self):
        if self._prefetch_stream is not None:
            self._prefetch_stream.synchronize()
        self._prefetched.clear()
        self._pending_sources.clear()

    def _demote(self, cache):
        cache_id = id(cache)
        indices = self._resident.get(cache_id, set())
        if not indices:
            return
        errors = []
        for index in tuple(indices):
            if index >= len(cache.layers):
                continue
            layer = cache.layers[index]
            for name in ("keys", "values"):
                try:
                    value = getattr(layer, name, None)
                    if value is not None and value.device.type != "cpu":
                        setattr(layer, name, value.to("cpu"))
                except Exception as error:
                    errors.append(error)
            if all(value is None or value.device.type == "cpu" for value in (layer.keys, layer.values)):
                self._resident_bytes = max(0, self._resident_bytes - self._sizes.get(cache, {}).get(index, 0))
                indices.remove(index)
        if errors:
            raise ExceptionGroup("KVキャッシュのCPU退避に失敗しました", errors)

    def _promote(self, cache):
        if not self._adaptive_enabled or self._resident_cap <= 0:
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._device)
        except torch.OutOfMemoryError:
            self.residency_fallbacks += 1
            return
        self._last_free_bytes = int(free_bytes)
        self._last_total_bytes = int(total_bytes)
        usable = max(0, int(free_bytes) - self._headroom)
        self.resident_budget_bytes = min(self._resident_cap, int(usable * self._resident_fraction))
        remaining = max(0, self.resident_budget_bytes - self._resident_bytes)
        resident = self._resident_indices(cache)
        for index, layer in enumerate(cache.layers):
            if index in resident:
                continue
            if remaining <= 0:
                break
            if layer.keys is None or layer.values is None:
                continue
            layer_bytes = self._bytes(layer.keys) + self._bytes(layer.values)
            if layer_bytes > remaining:
                continue
            try:
                keys = layer.keys.to(self._device)
                values = layer.values.to(self._device)
            except torch.OutOfMemoryError:
                self.residency_fallbacks += 1
                break
            layer.keys, layer.values = keys, values
            resident.add(index)
            remaining -= layer_bytes
            self._resident_bytes += layer_bytes
            self.resident_h2d_bytes += layer_bytes
            self.transferred_bytes += layer_bytes
            self.resident_bytes_peak = max(self.resident_bytes_peak, self._resident_bytes)

    def _schedule_prefetch(self, cache, index: int):
        if not self._prefetch_enabled or self._prefetch_stream is None:
            return
        if index < 0 or index >= len(cache.layers) or index in self._resident_indices(cache):
            return
        key = (id(cache), index)
        if key in self._prefetched:
            return
        layer = cache.layers[index]
        if layer.keys is None or layer.values is None:
            return
        if layer.keys.device.type != "cpu" or layer.values.device.type != "cpu":
            return
        source_keys = source_values = None
        try:
            source_keys = layer.keys if layer.keys.is_pinned() else layer.keys.pin_memory()
            source_values = layer.values if layer.values.is_pinned() else layer.values.pin_memory()
            with torch.cuda.stream(self._prefetch_stream):
                keys = source_keys.to(self._device, non_blocking=True)
                values = source_values.to(self._device, non_blocking=True)
                event = torch.cuda.Event()
                event.record(self._prefetch_stream)
            size_bytes = self._bytes(keys) + self._bytes(values)
            self._prefetched[key] = _Prefetch(keys, values, source_keys, source_values, event, size_bytes)
            self.prefetch_scheduled += 1
            self.prefetch_h2d_bytes += size_bytes
            self.transferred_bytes += size_bytes
            self.prefetch_bytes_peak = max(
                self.prefetch_bytes_peak,
                sum(item.size_bytes for item in self._prefetched.values()),
            )
        except Exception as error:
            # 同期に失敗しても転送元を保持し、closeで再試行できるようにする。
            self._pending_sources.extend(value for value in (source_keys, source_values) if value is not None)
            self._prefetch_enabled = False
            self.prefetch_fallback_reasons.append(f"{type(error).__name__}: {error}")
            try:
                self._clear_prefetch()
            except Exception as cleanup_error:
                raise ExceptionGroup("KV先読みと同期に失敗しました", [error, cleanup_error]) from None
            if not isinstance(error, (torch.OutOfMemoryError, MemoryError)):
                raise
            self.prefetch_fallbacks += 1

    def _take_prefetch(self, cache, index: int):
        key = (id(cache), index)
        item = self._prefetched.get(key)
        if item is None:
            self.prefetch_misses += 1
            return None
        started = time.perf_counter()
        item.event.synchronize()
        stream = torch.cuda.current_stream(self._device)
        item.keys.record_stream(stream)
        item.values.record_stream(stream)
        del self._prefetched[key]
        self.prefetch_wait_seconds += time.perf_counter() - started
        self.prefetch_hits += 1
        return item.keys, item.values

    def _prefix_on_device(self, cache, index, layer):
        if index in self._resident_indices(cache):
            return layer.keys, layer.values
        prefetched = self._take_prefetch(cache, index) if self._prefetch_enabled else None
        if prefetched is not None:
            return prefetched
        keys = layer.keys.to(self._device)
        values = layer.values.to(self._device)
        copied = self._bytes(keys) + self._bytes(values)
        self.synchronous_h2d_bytes += copied
        self.transferred_bytes += copied
        return keys, values

    def prepare_flash_cache(self, past_key_values, current_len, batch_size):
        if past_key_values is None:
            return
        self._clear_prefetch()
        self._demote(past_key_values)
        for index, layer in enumerate(past_key_values.layers):
            for name in ("keys", "values"):
                value = getattr(layer, name)
                if value is not None and value.device.type != "cpu":
                    setattr(layer, name, value.to("cpu"))
            prefix = layer.keys.shape[2] if layer.keys is not None and layer.keys.numel() else 0
            if prefix and layer.keys.shape[0] != batch_size:
                raise RuntimeError("参照KVキャッシュのバッチ数が一致しません。")
            layer.flash_prefix_len = prefix
            layer.flash_total_len = prefix + current_len
            layer.flash_k_cache = None
            layer.flash_v_cache = None
            self._remember_size(past_key_values, index, layer)
        self._promote(past_key_values)
        self._schedule_prefetch(past_key_values, 0)

    def _pre_hook(self, index):
        def prepare(module, args, kwargs):
            cache = kwargs.get("past_key_values")
            if cache is None:
                self._active[id(module)].append(None)
                return
            visit = _Visit(cache, index, bool(kwargs.get("update_cache", True)), {})
            self._active[id(module)].append(visit)
            self.visits += 1
            if index >= len(cache.layers):
                return
            layer = cache.layers[index]
            flash_ready = (
                not visit.update
                and hasattr(layer, "flash_total_len")
                and layer.keys is not None
                and layer.keys.numel() > 0
            )
            try:
                device_prefix = None
                if flash_ready:
                    device_prefix = self._prefix_on_device(cache, index, layer)
                    created = []
                    try:
                        for name, prefix in zip(("flash_k_cache", "flash_v_cache"), device_prefix, strict=True):
                            batch, heads, _, dimension = prefix.shape
                            buffer = torch.empty(
                                (batch, layer.flash_total_len, heads, dimension),
                                device=self._device,
                                dtype=prefix.dtype,
                            )
                            buffer[:, : layer.flash_prefix_len].copy_(prefix.transpose(1, 2))
                            setattr(layer, name, buffer)
                            created.append(name)
                    except Exception:
                        for name in created:
                            setattr(layer, name, None)
                        layer.flash_k_cache = None
                        layer.flash_v_cache = None
                        raise
                    self.flash_bytes_peak = max(
                        self.flash_bytes_peak,
                        self._bytes(layer.flash_k_cache) + self._bytes(layer.flash_v_cache),
                    )

                if not flash_ready or kwargs.get("attention_mask") is not None:
                    if index not in self._resident_indices(cache):
                        prefix = device_prefix
                        if prefix is None and self._prefetch_enabled:
                            prefix = self._take_prefetch(cache, index)
                        if prefix is None:
                            prefix = tuple(
                                value.to(self._device) if value is not None else None
                                for value in (layer.keys, layer.values)
                            )
                            copied = sum(self._bytes(value) for value in prefix if value is not None)
                            self.synchronous_h2d_bytes += copied
                            self.transferred_bytes += copied
                        for name, value in zip(("keys", "values"), prefix, strict=True):
                            if value is not None:
                                visit.original[name] = getattr(layer, name)
                                setattr(layer, name, value)

                if not visit.update:
                    self._schedule_prefetch(cache, index + 1)
            except Exception:
                for name in ("flash_k_cache", "flash_v_cache"):
                    if hasattr(layer, name):
                        setattr(layer, name, None)
                raise

        return prepare

    def _finish(self, visit):
        if visit is None or visit.index >= len(visit.cache.layers):
            return
        layer = visit.cache.layers[visit.index]
        resident = visit.index in self._resident_indices(visit.cache)
        errors = []
        for name in ("keys", "values"):
            try:
                value = getattr(layer, name)
                if not visit.update and name in visit.original:
                    setattr(layer, name, visit.original[name])
                elif value is not None and not resident and value.device.type != "cpu":
                    setattr(layer, name, value.to("cpu"))
            except Exception as error:
                errors.append(error)
        for name in ("flash_k_cache", "flash_v_cache"):
            if hasattr(layer, name):
                setattr(layer, name, None)
        if errors:
            raise ExceptionGroup("KVレイヤーの後始末に失敗しました", errors)
        self._remember_size(visit.cache, visit.index, layer)

    def _post_hook(self, module, args, kwargs, output):
        stack = self._active[id(module)]
        if stack:
            self._finish(stack[-1])
            stack.pop()

    @property
    def telemetry(self):
        return {
            "mode": "adaptive_layerwise_cpu" if self._adaptive_enabled else "layerwise_cpu",
            "adaptive_requested": self._adaptive_requested,
            "adaptive_enabled": self._adaptive_enabled,
            "prefetch_requested": self._prefetch_requested,
            "prefetch_enabled": self._prefetch_enabled,
            "prefix_cache_bytes_peak": self.prefix_bytes_peak,
            "flash_workspace_bytes_peak": self.flash_bytes_peak,
            "prefix_h2d_bytes": self.transferred_bytes,
            "synchronous_h2d_bytes": self.synchronous_h2d_bytes,
            "prefetch_h2d_bytes": self.prefetch_h2d_bytes,
            "resident_h2d_bytes": self.resident_h2d_bytes,
            "resident_budget_bytes": self.resident_budget_bytes,
            "resident_bytes_peak": self.resident_bytes_peak,
            "prefetch_bytes_peak": self.prefetch_bytes_peak,
            "prefetch_scheduled": self.prefetch_scheduled,
            "prefetch_hits": self.prefetch_hits,
            "prefetch_misses": self.prefetch_misses,
            "prefetch_fallbacks": self.prefetch_fallbacks,
            "prefetch_fallback_reasons": list(self.prefetch_fallback_reasons),
            "prefetch_wait_seconds": round(self.prefetch_wait_seconds, 6),
            "residency_fallbacks": self.residency_fallbacks,
            "cuda_free_bytes_at_budget": self._last_free_bytes,
            "cuda_total_bytes": self._last_total_bytes,
            "layer_visits": self.visits,
        }

    def close(self):
        if self._closed:
            return
        errors = []
        try:
            for handle in tuple(self._hooks):
                try:
                    handle.remove()
                    self._hooks.remove(handle)
                except Exception as error:
                    errors.append(error)
            for stack in self._active.values():
                for index in range(len(stack) - 1, -1, -1):
                    try:
                        self._finish(stack[index])
                        del stack[index]
                    except Exception as error:
                        errors.append(error)
            try:
                self._clear_prefetch()
            except Exception as error:
                errors.append(error)
            for cache in list(self._cache_refs.values()):
                try:
                    self._demote(cache)
                except Exception as error:
                    errors.append(error)
        finally:
            self._module.prepare_flash_kv_cache = self._original_prepare
        if errors:
            raise ExceptionGroup("KVキャッシュの後始末に失敗しました", errors)
        self._sizes.clear()
        self._resident.clear()
        self._cache_refs.clear()
        self._closed = True
