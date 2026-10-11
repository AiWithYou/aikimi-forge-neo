"""Checkpoint discovery is inexpensive, unambiguous and keeps missing selections."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge import local_assets, model_catalog


class ModelCatalogTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def touch(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_managed_legacy_and_external_roots_are_canonical_and_unique(self):
        legacy = self.root / "model/Iris-3B"
        legacy.mkdir(parents=True)
        external = self.root / "外部 モデル"
        with patch.object(model_catalog, "ROOT", self.root):
            roots = model_catalog.checkpoint_roots("Iris-3B", (external,), str(external))
        self.assertEqual(roots, (self.root / "models/Iris-3B/checkpoints", legacy, external))
        with patch.object(model_catalog, "ROOT", self.root), self.assertRaises(ValueError):
            model_catalog.checkpoint_roots("../outside")

    def test_folder_bundle_hides_internal_shards_and_shared_parts(self):
        bundle = self.touch("models/Full/model_index.json").parent
        self.touch("models/Full/transformer/diffusion_pytorch_model-00001.safetensors")
        self.touch("models/Full/vae/model.safetensors")
        single = self.touch("models/日本語 空白/body.gguf")
        self.touch("models/.cache/ignored.safetensors")
        self.touch("models/worker-env/ignored.safetensors")
        actual = model_catalog.discover_files((self.root / "models",), folders=True)
        self.assertEqual(set(actual), {str(bundle), str(single)})

    def test_adapter_validator_rejects_components_and_accepts_its_bundle(self):
        self.touch("models/component.safetensors")
        single = self.touch("models/body.safetensors")
        bundle = self.touch("models/custom/config.yaml").parent
        self.touch("models/custom/part.safetensors")
        actual = model_catalog.discover_files(
            (self.root / "models",),
            folders=True,
            validator=lambda path: path.name == "body.safetensors" or (path / "config.yaml").is_file(),
        )
        self.assertEqual(set(actual), {str(single), str(bundle)})

    def test_labels_keep_relative_subfolders_and_disambiguate_equal_roots(self):
        first = self.touch("one/checkpoints/FT/body.safetensors")
        second = self.touch("two/checkpoints/FT/body.safetensors")
        roots = (self.root / "one/checkpoints", self.root / "two/checkpoints")
        choices = model_catalog.choice_labels((str(first), str(second), str(first)), roots)
        self.assertEqual(len(choices), 2)
        self.assertEqual(len({label for label, _ in choices}), 2)
        self.assertTrue(all(label.startswith("FT/body.safetensors") for label, _ in choices))
        self.assertTrue(all(str(self.root) not in label for label, _ in choices))
        self.assertEqual({value for _, value in choices}, {str(first), str(second)})

    def test_missing_current_is_kept_without_writing_or_substituting(self):
        missing = self.root / "deleted/body.safetensors"
        choices = model_catalog.choice_labels((), (self.root,), current=str(missing))
        self.assertEqual(choices, [("deleted/body.safetensors (見つかりません)", str(missing))])
        self.assertFalse(missing.exists())

    def test_component_uses_library_instead_of_forge_config(self):
        component = SimpleNamespace()
        self.assertIs(model_catalog.configure_component(component), component)
        self.assertTrue(component.do_not_save_to_config)

    def test_overlapping_roots_do_not_reintroduce_bundle_shards(self):
        bundle = self.touch("models/Full/model_index.json").parent
        self.touch("models/Full/transformer/shard.safetensors")
        actual = model_catalog.discover_files((bundle.parent, bundle / "transformer"), folders=True)
        self.assertEqual(actual, [str(bundle)])

    def test_existing_local_asset_choices_use_relative_labels(self):
        path = self.touch("checkpoints/illustration/style.safetensors")
        with patch.object(local_assets, "LIBRARY", self.root / "library.json"):
            choices = local_assets.choices("lora", (self.root / "checkpoints",))
        self.assertEqual(choices, [("illustration/style.safetensors", str(path))])


if __name__ == "__main__":
    unittest.main()
