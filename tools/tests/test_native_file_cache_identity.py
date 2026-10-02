"""Real model/hash caches refresh after timestamp-preserving Windows file writes."""

from __future__ import annotations

import ctypes
import hashlib
import io
import json
import os
import stat
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from safetensors import safe_open

from modules.metadata_cache import MetadataCache
from modules_forge import local_assets
from tools.tests import test_hash_preview_efficiency as hash_fixtures
from tools.tests import test_resident_worker_read_races as lock_fixtures
from tools.tests.test_runtime_efficiency import load_definitions


class NativeFileCacheIdentityTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="file-version-日本語-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / "adapter.safetensors"
        header = {
            "__metadata__": {"ss_output_name": "old-alias"},
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        }
        encoded = json.dumps(header).encode()
        self.path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 1))
        self.caches = {name: MetadataCache(self.root / name) for name in ("hashes", "hashes-addnet")}
        self.hashes = hash_fixtures.load_module(
            "modules/hashes.py", SimpleNamespace(cmd_opts=SimpleNamespace(no_hashing=False)), self.caches
        )
        self.metadata = MetadataCache(self.root / "metadata")
        self.cached = load_definitions(
            "modules/cache.py",
            {"cached_data_for_file"},
            {"os": os, "cache": lambda _: self.metadata, "dump_cache": lambda: None},
        )["cached_data_for_file"]
        self.enterContext(redirect_stdout(io.StringIO()))
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "assets/library.json"))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "assets/hashes.json"))

    def mutate_in_place(self, old_alias="old-alias", new_alias="new-alias", value=2):
        before = self.path.stat()
        data = self.path.read_bytes().replace(old_alias.encode(), new_alias.encode())
        self.path.write_bytes(data[:-4] + struct.pack("<f", value))
        os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = self.path.stat()
        self.assertEqual(
            (before.st_size, before.st_mtime_ns, before.st_ino), (after.st_size, after.st_mtime_ns, after.st_ino)
        )
        if os.name == "nt":
            self.assertEqual(before.st_ctime_ns, after.st_ctime_ns)

    def expected_hash(self, addnet=False):
        data = self.path.read_bytes()
        if addnet:
            data = data[8 + struct.unpack("<Q", data[:8])[0] :]
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def read_metadata(filename):
        with safe_open(str(filename), framework="numpy") as tensors:
            return tensors.metadata()

    def test_same_size_and_mtime_inplace_write_rehashes_full_and_addnet(self):
        original = [self.hashes.sha256(self.path, "model", addnet) for addnet in (False, True)]
        self.mutate_in_place()
        for addnet, old in zip((False, True), original, strict=True):
            with self.subTest(addnet=addnet):
                self.assertNotEqual(self.expected_hash(addnet), old)
                self.assertIsNone(self.hashes.sha256_from_cache(self.path, "model", addnet))
                self.assertEqual(self.hashes.sha256(self.path, "model", addnet), self.expected_hash(addnet))

    def test_real_network_alias_refreshes_after_persistent_cache_reopen(self):
        reads = Mock(side_effect=self.read_metadata)
        network = load_definitions(
            "extensions-builtin/sd_forge_lora/network.py",
            {"NetworkOnDisk"},
            {
                "os": os,
                "sd_models": SimpleNamespace(read_metadata_from_safetensors=reads),
                "cache": SimpleNamespace(cached_data_for_file=self.cached),
                "hashes": SimpleNamespace(sha256_from_cache=Mock(return_value=None)),
                "errors": Mock(),
            },
        )["NetworkOnDisk"]
        self.assertEqual(network("adapter", str(self.path)).alias, "old-alias")
        self.assertEqual(network("adapter", str(self.path)).alias, "old-alias")
        self.assertEqual(reads.call_count, 1)
        self.mutate_in_place()
        self.metadata.close()
        self.metadata = MetadataCache(self.root / "metadata")
        self.assertEqual(self.read_metadata(self.path)["ss_output_name"], "new-alias")
        self.assertEqual(network("adapter", str(self.path)).alias, "new-alias")
        self.assertEqual(reads.call_count, 2)

    def test_directory_caller_keeps_cached_data_without_opening_it_as_a_model(self):
        directory = self.root / ".git"
        directory.mkdir()
        compute = Mock(return_value={"commit": "fixture"})
        with patch.object(local_assets, "file_version", side_effect=AssertionError("opened a directory")):
            for _ in range(2):
                self.assertEqual(self.cached("extensions-git", "extension", directory, compute), {"commit": "fixture"})
        compute.assert_called_once()

    def test_legacy_six_field_entries_are_recomputed_once(self):
        identity = self.hashes._hash_file_identity(self.path)[:6]
        self.caches["hashes"]["model"] = {"file_identity": identity, "sha256": "legacy hash"}
        with patch.object(self.hashes, "calculate_sha256_real", wraps=self.hashes.calculate_sha256_real) as calculate:
            for _ in range(2):
                self.assertEqual(self.hashes.sha256(self.path, "model"), self.expected_hash())
        calculate.assert_called_once()
        self.metadata["model"] = {"file_identity": identity, "value": "legacy metadata"}
        compute = Mock(return_value="fresh metadata")
        for _ in range(2):
            self.assertEqual(self.cached("metadata", "model", self.path, compute), "fresh metadata")
        compute.assert_called_once()

    def test_non_windows_identity_uses_stat_change_time_without_opening_file(self):
        from modules import file_identity

        info = SimpleNamespace(st_mode=stat.S_IFREG, st_size=4, st_mtime_ns=10, st_ctime_ns=20, st_ino=30, st_dev=40)
        fake_os = SimpleNamespace(name="posix", stat=Mock(return_value=info), path=os.path)
        with (
            patch.object(file_identity, "os", fake_os),
            patch.object(local_assets, "file_version", side_effect=AssertionError("native Windows helper used")),
        ):
            identity = file_identity.cache_file_identity(self.path)
        self.assertEqual(identity[1:], (10, 4, 20, 30, 40, 20))
        fake_os.stat.assert_called_once_with(self.path)

    def test_initial_stat_permission_error_keeps_existing_contract(self):
        from modules import file_identity

        error = PermissionError("fixture stat denied")
        fake_os = SimpleNamespace(name="nt", stat=Mock(side_effect=error), path=os.path)
        with patch.object(file_identity, "os", fake_os), patch.object(local_assets, "file_version") as version:
            with self.assertRaises(PermissionError) as raised:
                file_identity.cache_file_identity(self.path)
        self.assertIs(raised.exception, error)
        version.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "requires Windows file-sharing semantics")
    def test_exclusive_handle_bypasses_cache_and_preserves_no_hashing_listing(self):
        self.hashes.sha256(self.path, "model")
        original = self.hashes._hash_file_identity(self.path)
        compute = Mock(return_value="cached metadata")
        self.cached("metadata", "model", self.path, compute)
        compute.return_value = "uncached metadata"
        with lock_fixtures.exclusive_file_read(self.path):
            self.assertEqual(self.path.stat().st_size, original[2])
            with self.assertRaises(PermissionError), self.path.open("rb"):
                pass
            identity = self.hashes._hash_file_identity(self.path)
            self.assertEqual(identity[:6], original[:6])
            self.assertIsNone(identity[-1])
            self.assertIsNone(self.hashes.sha256_from_cache(self.path, "model"))
            self.hashes.shared.cmd_opts.no_hashing = True
            self.assertIsNone(self.hashes.sha256(self.path, "model"))
            self.hashes.shared.cmd_opts.no_hashing = False
            with self.assertRaises(PermissionError):
                self.hashes.sha256(self.path, "model")
            for _ in range(2):
                self.assertEqual(self.cached("metadata", "model", self.path, compute), "uncached metadata")
        self.assertEqual(compute.call_count, 3)
        self.assertEqual(self.metadata["model"]["value"], "cached metadata")

    @unittest.skipUnless(os.name == "nt", "requires the Windows file-information API")
    def test_native_information_failure_never_reuses_or_persists_weak_cache_stamp(self):
        self.hashes.sha256(self.path, "model")
        compute = Mock(side_effect=lambda: self.read_metadata(self.path))
        self.cached("metadata", "model", self.path, compute)
        with patch.object(ctypes.windll.kernel32, "GetFileInformationByHandleEx", return_value=0):
            for old_alias, new_alias, value in (("old-alias", "new-alias", 2), ("new-alias", "far-alias", 3)):
                self.mutate_in_place(old_alias, new_alias, value)
                self.assertIsNone(self.hashes.sha256_from_cache(self.path, "model"))
                self.assertEqual(self.hashes.sha256(self.path, "model"), self.expected_hash())
                self.assertEqual(self.cached("metadata", "model", self.path, compute)["ss_output_name"], new_alias)
                self.assertIsNotNone(self.caches["hashes"]["model"]["file_identity"][-1])
                self.assertIsNotNone(self.metadata["model"]["file_identity"][-1])
        self.assertEqual(compute.call_count, 3)

    @unittest.skipUnless(os.name == "nt", "requires the Windows file-information API")
    def test_native_information_recovery_during_hash_returns_fresh_uncached_value(self):
        get_info = ctypes.windll.kernel32.GetFileInformationByHandleEx
        calls = 0

        def recover(*args):
            nonlocal calls
            calls += 1
            return 0 if calls <= 3 else get_info(*args)

        with patch.object(ctypes.windll.kernel32, "GetFileInformationByHandleEx", side_effect=recover):
            self.assertEqual(self.hashes.sha256(self.path, "model"), self.expected_hash())
        self.assertNotIn("model", self.caches["hashes"])

    @unittest.skipUnless(os.name == "nt", "requires native ChangeTime for restored modification times")
    def test_timestamp_preserving_write_during_hash_does_not_commit_mixed_digest(self):
        def change_file(_filename):
            self.mutate_in_place()
            return "mixed digest"

        with patch.object(self.hashes, "calculate_sha256_real", side_effect=change_file):
            with self.assertRaisesRegex(RuntimeError, "changed"):
                self.hashes.sha256(self.path, "model")
        self.assertNotIn("model", self.caches["hashes"])


if __name__ == "__main__":
    unittest.main()
