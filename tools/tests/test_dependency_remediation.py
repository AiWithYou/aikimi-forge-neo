"""Behavioral security regressions for the locally maintained dependencies."""

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import chdir
from pathlib import Path

import torch
from safetensors.torch import save_file

from modules.metadata_cache import MetadataCache
from modules_forge.sensenova_transformers_compat import create_causal_mask


class CheckpointShardTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.checkpoint = self.root / "snapshot"
        self.checkpoint.mkdir()
        self.outside = self.root / "outside.safetensors"
        self.weights = {"weight": torch.full((2, 2), 7.0)}
        save_file(self.weights, self.outside, metadata={"format": "pt"})

    def loaders(self):
        from accelerate import load_checkpoint_and_dispatch
        from accelerate.utils import load_checkpoint_in_model

        return (load_checkpoint_in_model, load_checkpoint_and_dispatch)

    def index(self, shard):
        index = self.checkpoint / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": shard}}), encoding="utf-8")
        return index

    def test_both_public_apis_reject_escape_before_loading_weights(self):
        for name in (
            "../outside.safetensors",
            "..\\outside.safetensors",
            str(self.outside),
            "C:/outside.safetensors",
            "C:outside.safetensors",
            "\\\\server\\share\\weights",
            "weights:stream",
            "nested/../../outside.safetensors",
            ".",
            "missing.safetensors",
        ):
            for loader in self.loaders():
                with self.subTest(shard=name, api=loader.__name__):
                    model = torch.nn.Linear(2, 2, bias=False)
                    original = model.weight.detach().clone()
                    with self.assertRaises(ValueError):
                        loader(model, str(self.index(name)), device_map={"": "cpu"})
                    self.assertTrue(torch.equal(model.weight, original))

    def test_nested_shards_and_relative_checkpoint_directories_load(self):
        shard = self.checkpoint / "nested" / "weights.safetensors"
        shard.parent.mkdir()
        save_file(self.weights, shard, metadata={"format": "pt"})
        index = self.index("nested/weights.safetensors")
        for loader in self.loaders():
            model = torch.nn.Linear(2, 2, bias=False)
            with chdir(self.root):
                loader(model, str(index.relative_to(self.root)), device_map={"": "cpu"})
            self.assertTrue(torch.equal(model.weight, self.weights["weight"]))

    def test_snapshot_links_to_hub_blobs_remain_supported(self):
        link = self.checkpoint / "weights.safetensors"
        try:
            link.symlink_to(self.outside)
        except OSError as exc:
            self.skipTest(f"Symlink creation unavailable: {exc}")
        for loader in self.loaders():
            model = torch.nn.Linear(2, 2, bias=False)
            loader(model, str(self.index(link.name)), device_map={"": "cpu"})
            self.assertTrue(torch.equal(model.weight, self.weights["weight"]))


class MetadataCacheTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)
        self.cache = MetadataCache(self.path)

    def test_nested_metadata_and_file_identity_survive_reopen(self):
        value = {
            "file_identity": ("モデル.safetensors", 1234567890123456789, 400),
            "value": {"tags": ["blue", None, True], "tuple": {"dict": "literal keys"}},
        }
        self.cache["metadata"] = value
        self.assertEqual(MetadataCache(self.path)["metadata"], value)
        self.assertEqual(self.cache.get("absent", 42), 42)

    def test_legacy_cache_is_never_opened_or_modified(self):
        legacy = self.path / "cache.db"
        legacy.write_bytes(b"not a database or a trusted Python object")
        self.cache["safe"] = {"hash": "abc"}
        self.assertEqual(MetadataCache(self.path)["safe"], {"hash": "abc"})
        self.assertEqual(legacy.read_bytes(), b"not a database or a trusted Python object")

    def test_objects_cannot_supply_executable_serializers(self):
        class Executable:
            def __reduce__(self):
                raise AssertionError("Python serialization must not run")

        with self.assertRaises(TypeError):
            self.cache["bad"] = Executable()
        self.assertNotIn("bad", self.cache)

    def test_concurrent_instances_commit_without_lost_entries(self):
        def write(index):
            MetadataCache(self.path)[str(index)] = {"value": index}

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write, range(40)))
        self.assertEqual(len(self.cache), 40)
        for index in range(40):
            self.assertEqual(self.cache[str(index)], {"value": index})
        del self.cache["0"]
        with self.assertRaises(KeyError):
            del self.cache["0"]
        self.cache.clear()
        self.assertEqual(list(self.cache), [])


class SenseNovaMaskCompatibilityTests(unittest.TestCase):
    def config(self):
        from transformers import Qwen3Config

        config = Qwen3Config()
        config._attn_implementation = "eager"
        return config

    def test_explicit_cache_positions_are_preserved(self):
        embeds = torch.zeros(1, 3, 4)
        mask = create_causal_mask(self.config(), embeds, None, torch.arange(3), None)
        self.assertTrue(torch.equal(mask[0, 0] == 0, torch.ones(3, 3, dtype=torch.bool).tril()))
        shifted = create_causal_mask(self.config(), embeds, None, torch.arange(5, 8), None)
        self.assertTrue(torch.all(shifted == 0))

    def test_padding_and_custom_4d_masks_remain_supported(self):
        embeds = torch.zeros(1, 3, 4)
        mask = create_causal_mask(self.config(), embeds, torch.tensor([[0, 1, 1]]), torch.arange(3), None)
        self.assertTrue(torch.all(mask[:, :, :, 0] < -1e20))
        custom = torch.zeros(1, 1, 3, 3)
        self.assertIs(create_causal_mask(self.config(), embeds, custom, torch.arange(3), None), custom)

    def test_dynamic_cache_prefix_uses_new_cache_size_api(self):
        from transformers import DynamicCache

        config = self.config()
        cache = DynamicCache(config=config)
        keys = torch.zeros(1, 1, 4, 8)
        cache.update(keys, keys, 0)
        result = create_causal_mask(config, torch.zeros(1, 2, 4), None, torch.arange(4, 6), cache)
        self.assertEqual(result.shape, (1, 1, 2, 6))
        self.assertEqual(int((result[0, 0, 0] == 0).sum()), 5)
        self.assertEqual(int((result[0, 0, 1] == 0).sum()), 6)


if __name__ == "__main__":
    unittest.main()
