"""NegPiPキャッシュの配線と、通常キャッシュとの分離を検証する。"""

import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_negpip_cache as integration
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_acceleration_ui import acceleration_note
from modules_forge.minimax_h3_negpip import H3NegPiP
from tools.tests.test_hako_integrations import schema


def combined_schemas():
    result = schema()
    required = copy.deepcopy(result["ApplyMiniMaxH3NegPiP"]["input"]["required"])
    required.pop("clip")
    required.update(clip_name=[["encoder.safetensors"]], cache_mode=[["auto", "refresh"]])
    result[integration.NODE] = {"input": {"required": required}, "output": ["MODEL", "CLIP"]}
    return result


def adapter():
    spec = importlib.util.spec_from_file_location("negpip_cache_test", integration.BUNDLE / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"folder_paths": SimpleNamespace(), "nodes": SimpleNamespace()}):
        spec.loader.exec_module(module)
    return module


class NegPiPCacheTests(unittest.TestCase):
    def test_graph_keeps_native_conditioning_and_both_model_routes(self):
        for mode in ("text", "keyframes", "references"):
            with self.subTest(mode=mode):
                option = H3Acceleration(clip_cache="auto", negpip=H3NegPiP(enabled=True, value_strength=1.25))
                request = bridge.H3Request(
                    mode=mode, prompt=("<Picture 1> " if mode == "references" else "") + "(shake:-0.7@v0-2)", acceleration=option,
                    first_frame="first.png" if mode == "keyframes" else None,
                    reference_images=("ref.png",) if mode == "references" else (),
                )
                graph = bridge.build_workflow(request, {"first_frame": "first.png", "images": ["ref.png"]}, seed=123)
                self.assertNotIn("2", graph)
                self.assertEqual(graph["17"]["class_type"], integration.NODE)
                self.assertEqual(graph["17"]["inputs"]["value_strength"], 1.25)
                self.assertEqual(graph["17"]["inputs"]["clip_name"], bridge.H3_TEXT_ENCODER)
                self.assertEqual(graph["5"]["inputs"]["clip"], ["17", 1])
                self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
                self.assertEqual(graph["8"]["inputs"]["model"], ["17", 0])
                self.assertIn(graph["5"]["class_type"], {"MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"})

    def test_new_pack_requires_both_options(self):
        for cache, enabled in (("off", True), ("auto", False), ("auto", True)):
            option = H3Acceleration(clip_cache=cache, negpip=H3NegPiP(enabled=enabled))
            self.assertEqual(integration.PACK in option.runtime_packs(), cache != "off" and enabled)
            command = bridge._runtime_command(Path("python"), 8199, acceleration=option)
            self.assertTrue(bridge._runtime_arguments_are_allowed(command[1:]))

    def test_combined_schema_rejects_missing_controls_and_wrong_outputs(self):
        valid = combined_schemas()
        H3Acceleration(clip_cache="auto", negpip=H3NegPiP(enabled=True)).validate_nodes(valid)
        for mutate in (
            lambda n: n[integration.NODE]["input"]["required"].pop("block_stride"),
            lambda n: n[integration.NODE].update(output=["CLIP"]),
            lambda n: n[integration.NODE]["input"]["required"].update(cache_mode=[["auto"]]),
        ):
            changed = copy.deepcopy(valid)
            mutate(changed)
            with self.assertRaises(ValueError):
                integration.validate_nodes(changed)

    def test_model_patch_is_enabled_even_on_cache_hit(self):
        module = adapter()
        negpip = SimpleNamespace(patch_model=mock.Mock(return_value="patched"))
        proxy = object()
        with mock.patch.object(module, "negpip_module", return_value=negpip), mock.patch.object(module, "create_proxy", return_value=proxy):
            result = module.NegPiPCachedCLIP().apply("model", "encoder", "auto", value_strength=1.5)
        self.assertEqual(result, ("patched", proxy))
        negpip.patch_model.assert_called_once_with("model", {"value_strength": 1.5, "enabled": True})

    def test_cache_identity_changes_with_settings_and_negpip_source(self):
        module = adapter()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "negpip.py"
            source.write_text("version one", encoding="utf-8")
            with mock.patch.object(module, "negpip_module", return_value=SimpleNamespace(__file__=str(source))), mock.patch.dict(
                sys.modules, {"minimaxh3_clipcache.encoder_abi": SimpleNamespace(get_encoder_abi_id=lambda: ("abi", True))}
            ):
                original = module.cache_identity({"value_strength": 1.0})
                self.assertNotEqual(original, "abi")
                self.assertEqual(original, module.cache_identity({"value_strength": 1.0}))
                self.assertNotEqual(original, module.cache_identity({"value_strength": 2.0}))
                source.write_text("version two", encoding="utf-8")
                self.assertNotEqual(original, module.cache_identity({"value_strength": 1.0}))

    def test_missing_encoder_abi_is_an_error(self):
        module = adapter()
        with mock.patch.dict(sys.modules, {"minimaxh3_clipcache.encoder_abi": SimpleNamespace(get_encoder_abi_id=lambda: (None, False))}):
            with self.assertRaisesRegex(RuntimeError, "識別子"):
                module.cache_identity({})

    def test_install_is_idempotent_and_preserves_cache_and_user_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").touch()
            (root / "models").mkdir()
            target = integration.install(root)
            (target / "cache").mkdir()
            saved = target / "cache" / "keep.json"
            saved.write_text("{}", encoding="utf-8")
            self.assertEqual(integration.install(root), target)
            (target / "__init__.py").write_text("user edit", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "同梱版"):
                integration.install(root)
            self.assertEqual(saved.read_text(), "{}")
            self.assertEqual((target / "__init__.py").read_text(), "user edit")

    def test_ui_note_describes_combined_cache_and_history_roundtrip(self):
        option = H3Acceleration(clip_cache="auto", negpip=H3NegPiP(enabled=True, block_start=3))
        self.assertEqual(H3Acceleration.from_values(option.values()), option)
        self.assertIn("専用キャッシュ", acceleration_note(*option.values()))
        self.assertNotIn('role="alert"', acceleration_note(*option.values()))


if __name__ == "__main__":
    unittest.main()
