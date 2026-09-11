"""CPU-only regressions for model hashes and live-preview responses."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import os
import sys
import tempfile
import threading
import time
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load_module(filename, shared, caches):
    """Import the real module with only application/GPU dependencies stubbed."""
    package = ModuleType("modules")
    package.__path__ = []
    package.shared = shared
    package.cache = ModuleType("modules.cache")
    package.cache.cache = lambda name: caches[name]
    package.cache.dump_cache = Mock()
    name = "_efficiency_" + Path(filename).stem
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        name: module,
        "modules": package,
        "modules.shared": shared,
        "modules.cache": package.cache,
        "gradio": SimpleNamespace(skip=lambda: None),
    }
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class HashCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "model.safetensors"
        self.payload = b"tensor-data"
        self.path.write_bytes((2).to_bytes(8, "little") + b"{}" + self.payload)
        self.caches = defaultdict(dict)
        self.shared = SimpleNamespace(cmd_opts=SimpleNamespace(no_hashing=False))
        self.module = load_module("modules/hashes.py", self.shared, self.caches)
        self.output = redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_addnet_and_full_file_hashes_are_distinct_and_cached(self):
        full = self.module.sha256(self.path, "model")
        addnet = self.module.sha256(self.path, "model", use_addnet_hash=True)
        self.assertEqual(full, hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(addnet, hashlib.sha256(self.payload).hexdigest())
        self.assertNotEqual(full, addnet)
        self.assertEqual(self.module.sha256_from_cache(self.path, "model"), full)
        self.assertEqual(self.module.sha256_from_cache(self.path, "model", True), addnet)

    def test_replacements_with_older_or_preserved_mtime_invalidate(self):
        self.module.sha256(self.path, "model")
        previous = self.path.stat()
        self.path.write_bytes(b"replacement-with-different-size")
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertIsNone(self.module.sha256_from_cache(self.path, "model"))
        self.module.sha256(self.path, "model")
        os.utime(self.path, ns=(previous.st_atime_ns, previous.st_mtime_ns - 1_000_000_000))
        self.assertIsNone(self.module.sha256_from_cache(self.path, "model"))

    def test_same_title_on_another_path_is_not_a_cache_hit(self):
        self.module.sha256(self.path, "model")
        other = self.path.with_name("other.safetensors")
        other.write_bytes(b"x" * self.path.stat().st_size)
        stat = self.path.stat()
        os.utime(other, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertIsNone(self.module.sha256_from_cache(other, "model"))

    def test_legacy_entry_is_invalidated(self):
        self.caches["hashes"]["model"] = {"mtime": self.path.stat().st_mtime, "sha256": "stale"}
        self.assertIsNone(self.module.sha256_from_cache(self.path, "model"))

    def test_no_hashing_and_missing_files(self):
        self.shared.cmd_opts.no_hashing = True
        self.assertIsNone(self.module.sha256(self.path, "model"))
        self.shared.cmd_opts.no_hashing = False
        expected = self.module.sha256(self.path, "model")
        self.shared.cmd_opts.no_hashing = True
        self.assertEqual(self.module.sha256(self.path, "model"), expected)
        self.path.unlink()
        self.assertIsNone(self.module.sha256_from_cache(self.path, "model"))

    def test_file_changed_during_hash_is_not_cached(self):
        def change_file(_filename):
            self.path.write_bytes(b"changed-during-read")
            return "mixed-data-digest"

        with patch.object(self.module, "calculate_sha256_real", side_effect=change_file):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.module.sha256(self.path, "model")
        self.assertNotIn("model", self.caches["hashes"])

    def test_cache_hit_uses_one_database_read(self):
        self.module.sha256(self.path, "model")
        entries = self.caches["hashes"]
        spy = Mock(wraps=entries)
        self.caches["hashes"] = spy
        self.assertIsNotNone(self.module.sha256_from_cache(self.path, "model"))
        spy.get.assert_called_once_with("model")

    def test_concurrent_misses_read_file_once_for_each_hash_mode(self):
        for addnet in (False, True):
            with self.subTest(addnet=addnet):
                barrier = threading.Barrier(4)
                local = threading.local()
                lookup = self.module.sha256_from_cache

                def simultaneous_miss(*args, lookup=lookup, local=local, barrier=barrier, **kwargs):
                    value = lookup(*args, **kwargs)
                    if not getattr(local, "started", False):
                        local.started = True
                        barrier.wait(timeout=5)
                    return value

                name = "addnet_hash_safetensors" if addnet else "calculate_sha256_real"
                with (
                    patch.object(self.module, "sha256_from_cache", side_effect=simultaneous_miss),
                    patch.object(self.module, name, wraps=getattr(self.module, name)) as calculate,
                    ThreadPoolExecutor(max_workers=4) as pool,
                ):
                    results = list(pool.map(lambda _, addnet=addnet: self.module.sha256(self.path, "model", addnet), range(4)))
                self.assertEqual(len(set(results)), 1)
                self.assertEqual(calculate.call_count, 1)
                self.assertFalse(self.module._hash_locks)

    def test_different_files_hash_in_parallel(self):
        other = self.path.with_name("other.safetensors")
        other.write_bytes(b"other model")
        barrier = threading.Barrier(2)
        calculate = self.module.calculate_sha256_real

        def overlap(filename):
            barrier.wait(timeout=5)
            return calculate(filename)

        with (
            patch.object(self.module, "calculate_sha256_real", side_effect=overlap),
            ThreadPoolExecutor(max_workers=2) as pool,
        ):
            results = list(pool.map(lambda p: self.module.sha256(p, p.name), (self.path, other)))
        self.assertNotEqual(*results)
        self.assertFalse(self.module._hash_locks)

    def test_failed_hash_releases_lock_and_can_retry(self):
        with patch.object(self.module, "calculate_sha256_real", side_effect=OSError("read failed")):
            with self.assertRaisesRegex(OSError, "read failed"):
                self.module.sha256(self.path, "model")
        self.assertFalse(self.module._hash_locks)
        self.assertNotIn("model", self.caches["hashes"])
        self.assertEqual(self.module.sha256(self.path, "model"), hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertFalse(self.module._hash_locks)

    def test_warm_cache_does_not_allocate_request_lock(self):
        expected = self.module.sha256(self.path, "model")
        with patch.object(self.module, "_hash_request_lock", side_effect=AssertionError("warm cache locked")):
            self.assertEqual(self.module.sha256(self.path, "model"), expected)


class PreviewTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (32, 32))
        self.state = SimpleNamespace(
            job_count=1, job_no=0, sampling_steps=10, sampling_step=2,
            time_start=time.time(), id_live_preview=1, current_image=self.image,
            textinfo="Sampling", set_current_image=Mock(),
        )
        self.opts = SimpleNamespace(live_previews_enable=True, live_previews_image_format="png")
        shared = SimpleNamespace(state=self.state, opts=self.opts)
        self.module = load_module("modules/progress.py", shared, defaultdict(dict))
        self.module.start_task("job-a")

    def request(self, **kwargs):
        return self.module.progressapi(self.module.ProgressRequest(id_task="job-a", **kwargs))

    def test_same_preview_is_encoded_once_for_multiple_clients(self):
        with patch.object(self.image, "save", wraps=self.image.save) as save:
            with ThreadPoolExecutor(max_workers=4) as pool:
                responses = list(pool.map(lambda _: self.request(), range(4)))
            self.assertEqual(save.call_count, 1)
        self.assertEqual(len({response.live_preview for response in responses}), 1)
        raw = base64.b64decode(responses[0].live_preview.split(",", 1)[1])
        with Image.open(io.BytesIO(raw)) as decoded:
            self.assertEqual(decoded.size, self.image.size)

    def test_same_client_and_disabled_previews_do_not_encode(self):
        with patch.object(self.image, "save", wraps=self.image.save) as save:
            self.assertIsNone(self.request(id_live_preview=1).live_preview)
            self.assertIsNone(self.request(live_preview=False).live_preview)
            self.opts.live_previews_enable = False
            self.assertIsNone(self.request().live_preview)
            save.assert_not_called()

    def test_response_id_is_captured_before_image_encoding(self):
        original_save = self.image.save

        def save_and_update(*args, **kwargs):
            original_save(*args, **kwargs)
            self.state.current_image = Image.new("RGB", (40, 40))
            self.state.id_live_preview = 2

        with patch.object(self.image, "save", side_effect=save_and_update):
            response = self.request()
        self.assertEqual(response.id_live_preview, 1)
        self.assertEqual(self.request(id_live_preview=1).id_live_preview, 2)

    def test_format_and_image_changes_invalidate_preview_cache(self):
        png = self.request().live_preview
        self.opts.live_previews_image_format = "jpeg"
        jpeg = self.request().live_preview
        self.assertTrue(jpeg.startswith("data:image/jpeg;"))
        self.assertNotEqual(png, jpeg)
        self.state.current_image = Image.new("RGB", (45, 45))
        self.assertNotEqual(self.request().live_preview, jpeg)

    def test_idle_request_without_task_id_is_not_active(self):
        self.module.finish_task("job-a")
        self.state.set_current_image.reset_mock()
        response = self.module.progressapi(self.module.ProgressRequest())
        self.assertFalse(response.active)
        self.assertIsNone(response.live_preview)
        self.state.set_current_image.assert_not_called()

    def test_task_finish_releases_cache_and_restart_encodes_again(self):
        self.request()
        self.module.finish_task("job-a")
        self.assertFalse(self.module._preview_cache)
        self.module.start_task("job-a")
        with patch.object(self.image, "save", wraps=self.image.save) as save:
            self.request()
            self.module.finish_task("another-job")
            self.request()
            save.assert_called_once()

    def test_old_inflight_encoding_cannot_fill_new_task_cache(self):
        original_save = self.image.save

        def save_and_switch(*args, **kwargs):
            original_save(*args, **kwargs)
            self.module.finish_task("job-a")
            self.module.start_task("job-b")

        with patch.object(self.image, "save", side_effect=save_and_switch):
            self.request()
        self.assertFalse(self.module._preview_cache)


if __name__ == "__main__":
    unittest.main()
