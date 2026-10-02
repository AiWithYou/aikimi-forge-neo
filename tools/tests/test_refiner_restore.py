"""Refiner restoration must work after the UNet has already been offloaded."""

import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]


def load_postprocess():
    path = ROOT / "modules/processing_scripts/refiner.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ScriptRefiner")
    method = next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == "postprocess")
    method.decorator_list = []
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace["postprocess"]


class RefinerRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.postprocess = staticmethod(load_postprocess())

    def run_restore(self, residents, *, checkpoint="original.safetensors"):
        original = SimpleNamespace(ORIGINAL_CHECKPOINT=checkpoint)
        model = object()
        unet = SimpleNamespace(model=SimpleNamespace(diffusion_model=model, gguf_baked=True))
        shared = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
        weights, metadata = {"weight": "original"}, {"format": "fixture"}
        restore = Mock()
        load = Mock(return_value=(weights, metadata))
        bake = Mock(side_effect=lambda value: value)
        dependencies = {
            "modules": SimpleNamespace(sd_samplers_common=original, shared=shared),
            "huggingface_guess": SimpleNamespace(guess=lambda value: SimpleNamespace(unet_key_prefix="unet.")),
            "backend.loader": SimpleNamespace(preprocess_state_dict=lambda value: value),
            "backend.memory_management": SimpleNamespace(
                LoadedModel=object, current_loaded_models=residents, bake_gguf_model=bake
            ),
            "backend.patcher.unet": SimpleNamespace(UnetPatcher=FixtureUnet),
            "backend.state_dict": SimpleNamespace(
                convert_quantization=lambda state, meta: (state, meta),
                load_state_dict=restore,
                try_filter_state_dict=lambda state, prefix: state,
            ),
            "backend.utils": SimpleNamespace(load_torch_file=load),
            "modules_forge.main_entry": SimpleNamespace(logger=Mock()),
        }
        with patch.dict(sys.modules, dependencies):
            self.postprocess(SimpleNamespace())
        return original, model, weights, restore, load, bake

    def test_restore_without_any_resident_model(self):
        original, model, weights, restore, _, _ = self.run_restore([])
        restore.assert_called_once_with(model, weights, ignore_start="llm")
        self.assertIsNone(original.ORIGINAL_CHECKPOINT)

    def test_non_unet_resident_is_not_removed_or_unloaded(self):
        other = SimpleNamespace(model=object(), model_unload=Mock())
        residents = [other]
        _, _, _, restore, _, _ = self.run_restore(residents)
        self.assertEqual(residents, [other])
        other.model_unload.assert_not_called()
        restore.assert_called_once()

    def test_resident_unet_is_unloaded_before_restoration(self):
        other = SimpleNamespace(model=object(), model_unload=Mock())
        unet = SimpleNamespace(model=FixtureUnet(), model_unload=Mock())
        residents = [other, unet]
        original, _, _, restore, _, _ = self.run_restore(residents)
        self.assertEqual(residents, [other])
        unet.model_unload.assert_called_once_with()
        other.model_unload.assert_not_called()
        restore.assert_called_once()
        self.assertIsNone(original.ORIGINAL_CHECKPOINT)

    def test_offloaded_gguf_is_restored_and_rebaked(self):
        original, _, _, restore, _, bake = self.run_restore([], checkpoint="original.GGUF")
        restore.assert_called_once()
        bake.assert_called_once()
        self.assertFalse(bake.call_args.args[0].gguf_baked)
        self.assertIsNone(original.ORIGINAL_CHECKPOINT)

    def test_no_refiner_switch_does_not_load_weights(self):
        _, _, _, restore, load, _ = self.run_restore([], checkpoint=None)
        restore.assert_not_called()
        load.assert_not_called()


class FixtureUnet:
    pass


if __name__ == "__main__":
    unittest.main()
