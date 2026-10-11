"""Forge LoRA cards must address the chosen nested file and honor disabled weights."""

import hashlib
import itertools
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import local_assets
from tools.tests import test_runtime_efficiency as runtime_tests


class ForgeLoraSelectionTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.library = patch.object(local_assets, "LIBRARY", self.root / "library.json")
        self.library.start()
        self.addCleanup(self.library.stop)
        cache = patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json")
        cache.start()
        self.addCleanup(cache.stop)

    def scan(self, aliases=None, extra_roots=()):
        aliases = aliases or {}

        def build(name, filename):
            return SimpleNamespace(name=name, filename=filename, alias=aliases.get(name, "shared-style"))

        def walk(directory, **_kwargs):
            return (str(path) for path in sorted(Path(directory).rglob("*.safetensors")))

        scope = runtime_tests.load_definitions(
            "extensions-builtin/sd_forge_lora/networks.py",
            {"process_network_files"},
            {
                "os": os,
                "itertools": itertools,
                "shared": SimpleNamespace(
                    cmd_opts=SimpleNamespace(lora_dir=str(self.root), lora_dirs=list(extra_roots)), walk_files=walk
                ),
                "network": SimpleNamespace(NetworkOnDisk=build),
                "errors": Mock(),
                "available_networks": {},
                "available_network_aliases": {},
                "forbidden_network_aliases": set(),
            },
        )
        scope["process_network_files"]()
        return scope

    def create(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return str(path)

    def test_nested_same_basename_cards_keep_both_exact_files(self):
        first = self.create("Anima/portraits/style.safetensors")
        second = self.create("Anima/landscapes/style.safetensors")
        scope = self.scan()
        entries = scope["available_networks"]
        self.assertEqual(
            {name: entry.filename for name, entry in entries.items()},
            {"Anima/portraits/style": first, "Anima/landscapes/style": second},
        )
        self.assertIn("shared-style", scope["forbidden_network_aliases"])
        for name, entry in entries.items():
            self.assertIs(scope["available_network_aliases"][name], entry)

    def test_metadata_alias_cannot_override_another_canonical_file(self):
        first = self.create("Anima/one.safetensors")
        second = self.create("Anima/two.safetensors")
        scope = self.scan({"Anima/one": "Anima/two", "Anima/two": "another-alias"})
        self.assertEqual(scope["available_network_aliases"]["Anima/one"].filename, first)
        self.assertEqual(scope["available_network_aliases"]["Anima/two"].filename, second)
        self.assertIn("anima/two", scope["forbidden_network_aliases"])

    def test_refreshing_the_same_file_does_not_make_its_alias_ambiguous(self):
        self.create("Anima/one.safetensors")
        scope = self.scan({"Anima/one": "unique-style"})
        scope["process_network_files"](["Anima/one"])
        self.assertEqual(scope["forbidden_network_aliases"], set())
        self.assertIs(scope["available_network_aliases"]["unique-style"], scope["available_networks"]["Anima/one"])

    def test_equal_names_in_explicit_roots_do_not_replace_the_default_file(self):
        first = self.create("style.safetensors")
        extra = TemporaryDirectory()
        self.addCleanup(extra.cleanup)
        second = Path(extra.name) / "style.safetensors"
        second.touch()
        scope = self.scan(extra_roots=(extra.name,))
        self.assertEqual({entry.filename for entry in scope["available_networks"].values()}, {first, str(second)})
        self.assertEqual(scope["available_networks"]["style"].filename, first)

    def test_zero_strength_does_not_read_and_removes_the_previous_patch(self):
        fixture = runtime_tests.LoraTests(methodName="test_unchanged_lora_skips_read_and_strength_change_reloads")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.load()
        fixture.reader.reset_mock()
        fixture.patcher.reset_mock()
        fixture.load(0.0)
        fixture.reader.assert_not_called()
        fixture.patcher.assert_not_called()
        self.assertIs(fixture.model.forge_objects.unet, fixture.model.forge_objects_original.unet)
        self.assertEqual(fixture.ns["loaded_networks"], [])

    def test_missing_enabled_lora_stops_generation_before_reading_weights(self):
        fixture = runtime_tests.LoraTests(methodName="test_unchanged_lora_skips_read_and_strength_change_reloads")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.ns["available_networks"].clear()
        fixture.ns["available_network_aliases"].clear()
        fixture.ns["update_available_networks_by_names"] = Mock()
        fixture.ns["list_available_networks"] = Mock()
        with self.assertRaisesRegex(ValueError, "a"):
            fixture.load()
        fixture.reader.assert_not_called()
        fixture.patcher.assert_not_called()

    def test_text_encoder_only_strength_keeps_the_lora_enabled(self):
        fixture = runtime_tests.LoraTests(methodName="test_unchanged_lora_skips_read_and_strength_change_reloads")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.ns["load_networks"](["a"], [0.5], [0.0])
        fixture.reader.assert_called_once()
        self.assertEqual(fixture.patcher.call_args.args[3:5], (0.0, 0.5))

    def test_past_external_loras_are_not_added_to_the_registry(self):
        extra = TemporaryDirectory()
        self.addCleanup(extra.cleanup)
        old = Path(extra.name) / "old.safetensors"
        old.touch()
        local_assets.remember("forge_lora", [str(old)])
        self.assertEqual(self.scan()["available_networks"], {})

    def test_only_active_selected_external_loras_are_registered(self):
        extra = TemporaryDirectory()
        self.addCleanup(extra.cleanup)
        active = Path(extra.name) / "active.safetensors"
        disabled = Path(extra.name) / "disabled.safetensors"
        active.touch()
        disabled.touch()
        local_assets.save_selection(
            "forge_lora",
            {"loras": [{"name": str(active), "strength": 0.5}, {"name": str(disabled), "strength": 0}]},
        )
        entries = self.scan()["available_networks"]
        expected = "local_" + hashlib.sha256(os.path.normcase(os.path.realpath(active)).encode()).hexdigest()[:16]
        self.assertEqual({name: entry.filename for name, entry in entries.items()}, {expected: str(active.resolve())})


if __name__ == "__main__":
    unittest.main()
