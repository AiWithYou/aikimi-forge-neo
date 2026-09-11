"""GPU-free regressions for file caching, LoRA loading and memory reporting."""

from __future__ import annotations

import ast
import itertools
import os
import tempfile
import threading
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[2]


def load_definitions(path, names, namespace):
    """Execute selected real definitions without importing the GPU application."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    body.extend(node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, path, "exec"), namespace)  # noqa: S102 - only repository source is executed
    return namespace


class FileCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "model.safetensors"
        self.path.write_bytes(b"old")
        self.entries = {}
        self.compute = Mock(return_value="metadata")
        self.ns = load_definitions("modules/cache.py", {"cached_data_for_file"}, {
            "os": os, "cache": lambda _: self.entries, "dump_cache": lambda: None,
        })

    def get(self, path=None):
        return self.ns["cached_data_for_file"]("metadata", "same-title", path or self.path, self.compute)

    def test_unchanged_file_is_cached(self):
        self.assertEqual(self.get(), "metadata")
        self.assertEqual(self.get(), "metadata")
        self.compute.assert_called_once()

    def test_older_timestamp_and_preserved_timestamp_replacements_are_refreshed(self):
        self.get()
        old = self.path.stat()
        os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns - 1_000_000_000))
        self.get()
        current = self.path.stat()
        self.path.write_bytes(b"different-size")
        os.utime(self.path, ns=(current.st_atime_ns, current.st_mtime_ns))
        self.get()
        self.assertEqual(self.compute.call_count, 3)

    def test_title_collision_and_legacy_entry_are_refreshed(self):
        self.entries["same-title"] = {"mtime": self.path.stat().st_mtime, "value": "legacy"}
        self.get()
        other = self.path.with_name("other.safetensors")
        other.write_bytes(self.path.read_bytes())
        old = self.path.stat()
        os.utime(other, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.get(other)
        self.assertEqual(self.compute.call_count, 2)


class Objects(SimpleNamespace):
    def shallow_copy(self):
        return Objects(**vars(self))


class LoraTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "a.safetensors"
        self.path.write_bytes(b"fixture")
        original = Objects(unet=object(), clip=object())
        self.model = Objects(current_lora_hash=None, forge_objects_original=original,
                             forge_objects=original.shallow_copy())
        entry = Objects(filename=str(self.path), read_hash=Mock())
        self.reader = Mock(return_value={"weight": object()})
        self.patcher = Mock(side_effect=lambda model, clip, *_args, **_kwargs: (model, clip))
        self.ns = load_definitions("extensions-builtin/sd_forge_lora/networks.py", {"load_networks"}, {
            "os": os, "sd_models": Objects(model_data=Objects(get_sd_model=lambda: self.model)),
            "loaded_networks": [], "forbidden_network_aliases": set(),
            "available_networks": {"a": entry}, "available_network_aliases": {"a": entry},
            "load_network": lambda *_: Objects(), "logger": Mock(),
            "dynamic_args": Objects(online_lora=False, nunchaku=False),
            "load_lora_state_dict": self.reader, "load_lora_for_models": self.patcher,
        })

    def load(self, strength=1.0):
        self.ns["load_networks"](["a"], [strength], [strength])

    def test_unchanged_lora_skips_read_and_strength_change_reloads(self):
        self.load()
        self.load()
        self.assertEqual(self.reader.call_count, 1)
        self.load(0.5)
        self.assertEqual(self.reader.call_count, 2)

    def test_failed_load_is_retried(self):
        self.reader.side_effect = [OSError("fixture read failure"), {"weight": object()}]
        with self.assertRaises(OSError):
            self.load()
        self.assertIsNone(self.model.current_lora_hash)
        self.load()
        self.assertEqual(self.reader.call_count, 2)
        self.assertIsNotNone(self.model.current_lora_hash)

    def test_failed_strength_change_does_not_reuse_previous_success_key(self):
        self.load()
        self.patcher.side_effect = RuntimeError("fixture patch failure")
        with self.assertRaises(RuntimeError):
            self.load(0.5)
        self.assertIsNone(self.model.current_lora_hash)
        self.patcher.side_effect = lambda model, clip, *_args, **_kwargs: (model, clip)
        self.load()
        self.assertEqual(self.reader.call_count, 3)

    def test_replaced_file_reloads_without_strength_change(self):
        self.load()
        old = self.path.stat()
        self.path.write_bytes(b"larger fixture")
        os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns))
        self.load()
        os.utime(self.path, ns=(old.st_atime_ns, old.st_mtime_ns - 1_000_000_000))
        self.load()
        self.assertEqual(self.reader.call_count, 3)

    def test_patch_key_checks_do_not_scan_loaded_key_lists(self):
        class Keys(list):
            def __contains__(self, _value):
                raise AssertionError("quadratic membership lookup")

        def patcher():
            result = Objects(model=Objects(diffusion_model=object()), cond_stage_model=object())
            result.clone = lambda: result
            result.add_patches = lambda **_kwargs: Keys(["a", "b"])
            return result

        ns = load_definitions("extensions-builtin/sd_forge_lora/networks.py", {"load_lora_for_models"}, {
            "os": os, "dynamic_args": Objects(nunchaku=False), "logger": Mock(),
            "model_lora_keys_unet": lambda _: {}, "model_lora_keys_clip": lambda _: {},
            "load_lora": Mock(side_effect=[({"a": 1, "b": 2}, {}), ({"a": 1, "b": 2}, {})]),
        })
        unet, clip = patcher(), patcher()
        self.assertEqual(ns["load_lora_for_models"](unet, clip, {"a": 1, "b": 2}, 1.0, 1.0), (unet, clip))

    def test_directory_scan_processes_files_lazily(self):
        events = []

        def walk(_directory, **_kwargs):
            for name in ("a", "b"):
                events.append("yield-" + name)
                yield str(self.path.with_name(name + ".safetensors"))

        def build(name, _filename):
            events.append("build-" + name)
            return Objects(alias=name)

        ns = load_definitions("extensions-builtin/sd_forge_lora/networks.py", {"process_network_files"}, {
            "os": os, "itertools": itertools,
            "shared": Objects(cmd_opts=Objects(lora_dir=self.directory.name, lora_dirs=[]), walk_files=walk),
            "network": Objects(NetworkOnDisk=build), "errors": Mock(),
            "available_networks": {}, "available_network_aliases": {}, "forbidden_network_aliases": set(),
        })
        ns["process_network_files"]()
        self.assertEqual(events, ["yield-a", "build-a", "yield-b", "build-b"])


class StopPolling(BaseException):
    pass


class RunOnce:
    def __init__(self):
        self.started = False
        self.running = True

    def wait(self):
        if self.started:
            raise StopPolling
        self.started = True

    def is_set(self):
        return self.running

    def clear(self):
        self.running = False


class MemmonTests(unittest.TestCase):
    def setUp(self):
        self.backend = Mock()
        self.backend.mem_get_info.return_value = (600, 1000)
        self.backend.memory_stats.return_value = {
            "reserved_bytes.all.current": 100, "reserved_bytes.all.peak": 200,
        }
        self.sleep = Mock()
        ns = load_definitions("modules/memmon.py", {"MemUsageMonitor"}, {
            "threading": threading, "time": Objects(sleep=self.sleep), "defaultdict": defaultdict,
            "torch": Objects(cuda=self.backend),
            "memory_management": Objects(is_intel_xpu=lambda: False, logger=Mock()),
        })
        self.device = Objects(index=1)
        self.opts = Objects(memmon_poll_rate=0)
        self.monitor = ns["MemUsageMonitor"]("fixture", self.device, self.opts)
        self.monitor.run_flag = RunOnce()

    def run_once(self):
        with self.assertRaises(StopPolling):
            self.monitor.run()

    def test_read_before_first_poll_reports_observed_usage(self):
        self.assertEqual(self.monitor.read()["system_peak"], 400)

    def test_disabled_polling_keeps_baseline_and_resets_selected_device(self):
        self.run_once()
        self.assertEqual(self.monitor.read()["system_peak"], 400)
        self.backend.reset_peak_memory_stats.assert_called_once_with(self.device)

    def test_changing_poll_rate_to_zero_during_query_does_not_divide_by_zero(self):
        self.opts.memmon_poll_rate = 2
        calls = 0

        def memory(_index):
            nonlocal calls
            calls += 1
            if calls == 2:
                self.opts.memmon_poll_rate = 0
            return 600, 1000

        self.backend.mem_get_info.side_effect = memory
        self.run_once()
        self.sleep.assert_called_once_with(0.5)

    def test_read_preserves_larger_previously_sampled_peak(self):
        self.monitor.data["min_free"] = 300
        self.assertEqual(self.monitor.read()["system_peak"], 700)


if __name__ == "__main__":
    unittest.main()
