from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from modules.metadata_cache import MetadataCache
from tools.tests.test_runtime_efficiency import load_definitions


class LoraMetadataReplacementTests(unittest.TestCase):
    def test_persistent_lora_alias_cache_refreshes_after_same_size_same_time_replacement(self):
        with tempfile.TemporaryDirectory(prefix="LoRA 日本語 ") as temporary:
            root = Path(temporary)
            filename = root / "adapter.safetensors"
            replacement = root / "replacement.safetensors"
            tensor = {"layer.lora_up.weight": torch.ones((1, 1))}
            save_file(tensor, str(filename), metadata={"ss_output_name": "old-alias"})
            cache = MetadataCache(root / "cache")
            reads = Mock(side_effect=self.read_metadata)
            cache_function = load_definitions(
                "modules/cache.py",
                {"cached_data_for_file"},
                {"os": os, "cache": lambda _name: cache, "dump_cache": lambda: None},
            )["cached_data_for_file"]
            network = load_definitions(
                "extensions-builtin/sd_forge_lora/network.py",
                {"NetworkOnDisk"},
                {
                    "os": os,
                    "sd_models": SimpleNamespace(read_metadata_from_safetensors=reads),
                    "cache": SimpleNamespace(cached_data_for_file=cache_function),
                    "hashes": SimpleNamespace(sha256_from_cache=Mock(return_value=None)),
                    "errors": Mock(),
                },
            )["NetworkOnDisk"]

            self.assertEqual(network("adapter", str(filename)).alias, "old-alias")
            self.assertEqual(network("adapter", str(filename)).alias, "old-alias")
            self.assertEqual(reads.call_count, 1)
            old = filename.stat()
            save_file(tensor, str(replacement), metadata={"ss_output_name": "new-alias"})
            self.assertEqual(replacement.stat().st_size, old.st_size)
            os.utime(replacement, ns=(old.st_atime_ns, old.st_mtime_ns))
            replacement.replace(filename)
            self.assertNotEqual(filename.stat().st_ino, old.st_ino)
            cache.close()
            cache = MetadataCache(root / "cache")

            self.assertEqual(network("adapter", str(filename)).alias, "new-alias")
            self.assertEqual(reads.call_count, 2)

    @staticmethod
    def read_metadata(filename):
        with safe_open(filename, framework="pt", device="cpu") as handle:
            return handle.metadata()


if __name__ == "__main__":
    unittest.main()
