"""Forge's real LoRA controls preserve selection, strengths and physical file identity."""

import importlib.util
import json
import struct
import sys
import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import gradio as gr

from modules_forge import local_assets

ROOT = Path(__file__).resolve().parents[2]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ForgeLoraUITests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.standard = self.root / "Lora"
        self.standard.mkdir()
        for name, value in (("LIBRARY", self.root / "library.json"), ("HASH_CACHE", self.root / "hashes.json")):
            self.enterContext(patch.object(local_assets, name, value))
        package = ModuleType("modules")
        package.scripts = SimpleNamespace(Script=object, AlwaysVisible=True)
        package.shared = SimpleNamespace(cmd_opts=SimpleNamespace(lora_dir=str(self.standard), lora_dirs=[]))
        with patch.dict(sys.modules, {"modules": package}):
            self.picker = load_module("_forge_lora_ui", ROOT / "scripts/local_lora_stack.py")
        self.network = SimpleNamespace(
            NetworkOnDisk=Mock(side_effect=lambda name, path: SimpleNamespace(name=name, filename=path))
        )
        self.networks = SimpleNamespace(available_networks={}, available_network_aliases={})
        self.enterContext(patch.dict(sys.modules, {"network": self.network, "networks": self.networks}))
        self.anima = load_module(
            "_anima_standard_lora_ui", ROOT / "extensions-builtin/anima-3-8b/anima3b/standard_lora.py"
        )

    def create(self, relative):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"0000")
        return str(path)

    def save(self, *settings):
        local_assets.save_selection(
            "forge_lora", {"loras": [{"name": name, "strength": strength} for name, strength in settings]}
        )

    def build(self):
        with gr.Blocks(analytics_enabled=False) as self.demo:
            self.script = self.picker.Script()
            self.names, self.rows = self.script.ui(False)
        self.addCleanup(self.demo.close)

    def test_png_information_clears_previous_form_selection_without_saving(self):
        path = self.create("Lora/style.safetensors")
        self.save((path, 0.7))
        self.build()
        fields = dict(getattr(self.script, "infotext_fields", ()) or ())
        self.assertIn(self.names, fields, "PNG import must reset the form before using saved prompt tags")
        self.assertIn(self.rows, fields)
        params = {"Prompt": "portrait <lora:style:0.7>", "Local LoRA stack": "style.safetensors:0.7"}
        with patch.object(local_assets, "save_selection", side_effect=AssertionError("PNG restore must not save")):
            names = fields[self.names](params)["value"]
            rows = fields[self.rows](params)
            self.assertEqual(names, [])
            self.assertEqual(rows["value"], [])
            self.assertEqual(rows["visible"], "hidden")
            processing = self.processing()
            processing.prompt = params["Prompt"]
            processing.all_prompts = [params["Prompt"], params["Prompt"]]
            self.picker.apply_selections(processing, names, rows["value"])
        self.network.NetworkOnDisk.assert_not_called()
        self.assertTrue(all(value.count("<lora:") == 1 for value in processing.all_prompts))
        self.assertEqual(local_assets.selection("forge_lora")["loras"], [{"name": path, "strength": 0.7}])

    def test_batch_prompts_receive_one_tag_per_selected_file(self):
        path = self.create("Lora/style.safetensors")
        self.networks.available_networks["style"] = SimpleNamespace(filename=path)
        processing = self.processing()
        processing.prompt = ["portrait", "landscape"]
        processing.all_prompts = ["portrait", "landscape", "portrait", "landscape"]
        self.picker.apply_selections(processing, [path], [[path, 0.7]])
        self.assertTrue(all(value.count("<lora:style:0.7>") == 1 for value in processing.prompt))
        self.assertTrue(all(value.count("<lora:style:0.7>") == 1 for value in processing.all_prompts))

    def test_gallery_send_to_includes_reset_fields_across_script_instances(self):
        self.build()
        source_keys = getattr(self.script, "paste_field_names", ()) or ()
        with gr.Blocks(analytics_enabled=False) as target_demo:
            target_script = self.picker.Script()
            target_names, target_rows = target_script.ui(True)
        self.addCleanup(target_demo.close)
        fields = [(field, key) for field, key in target_script.infotext_fields if key in source_keys]
        self.assertEqual([field for field, _ in fields], [target_names, target_rows])
        params = {"Prompt": "portrait <lora:style:0.7>"}
        updates = [key(params) for _, key in fields]
        self.assertEqual(updates[0]["value"], [])
        self.assertEqual(updates[1]["value"], [])
        self.assertEqual(updates[1]["visible"], "hidden")

    def callback(self, component, event):
        function = next(
            (
                self.demo.fns[dependency["id"]]
                for dependency in self.demo.get_config_file()["dependencies"]
                if any(tuple(target) == (component._id, event) for target in dependency["targets"])
            ),
            None,
        )
        self.assertIsNotNone(function, f"LoRA {event} must be wired to the actual control")
        return function

    @staticmethod
    def processing():
        return SimpleNamespace(prompt="portrait", all_prompts=["styled portrait"], extra_generation_params={})

    def test_empty_strength_table_stays_mounted_and_programmatic_updates_do_not_save(self):
        self.build()
        self.assertEqual(self.names.label, "LoRA（複数選択・フルパスも入力可）")
        self.assertEqual(self.rows.headers, ["LoRA", "強度"])
        self.assertEqual(self.rows.visible, "hidden")
        self.assertTrue(self.names.do_not_save_to_config and self.rows.do_not_save_to_config)
        path = self.create("Lora/style.safetensors")
        change = self.callback(self.names, "change")
        with patch.object(
            local_assets, "save_selection", side_effect=AssertionError("readonly callback must not save")
        ):
            update = change.fn([path], [])
            self.assertTrue(update["visible"])
            self.assertEqual(update["value"], [[local_assets.lora_label(path), 1.0]])
            self.assertEqual(change.fn([], update["value"])["visible"], "hidden")
        self.assertEqual(change.show_progress, "hidden")

    def test_user_inputs_save_new_selection_and_individual_strengths(self):
        first = self.create("Lora/first.safetensors")
        second = self.create("Lora/second.safetensors")
        self.save((first, 0.35))
        self.build()
        select = self.callback(self.names, "input")
        self.assertEqual(select.inputs, [self.names, self.rows])
        select.fn([first, second], [[local_assets.lora_label(first), 0.35]])
        self.assertEqual(
            local_assets.selection("forge_lora")["loras"],
            [{"name": first, "strength": 0.35}, {"name": second, "strength": 1.0}],
        )
        strength = self.callback(self.rows, "input")
        self.assertEqual(strength.inputs, [self.names, self.rows])
        strength.fn([first, second], [[local_assets.lora_label(first), -0.7], [local_assets.lora_label(second), 0]])
        self.assertEqual(
            local_assets.selection("forge_lora")["loras"],
            [{"name": first, "strength": -0.7}, {"name": second, "strength": 0.0}],
        )

    def test_new_browser_load_reads_latest_library_without_writing_or_opening_weights(self):
        self.build()
        first = self.create("Lora/first.safetensors")
        second = self.create("outside/second.safetensors")
        self.save((first, -0.5), (second, 0.7))
        loader = next(
            (
                function
                for function in self.demo.fns.values()
                if function.inputs == [] and function.outputs == [self.names, self.rows]
            ),
            None,
        )
        self.assertIsNotNone(loader, "New browser sessions must restore the latest library selection")
        with (
            patch.object(local_assets, "save_selection", side_effect=AssertionError("initial load must not save")),
            patch.object(local_assets, "read_header", side_effect=AssertionError("UI restore must not open weights")),
        ):
            names, rows = loader.fn()
        self.assertEqual(names["value"], [first, second])
        self.assertEqual(
            rows["value"], [[local_assets.lora_label(first), -0.5], [local_assets.lora_label(second), 0.7]]
        )
        self.assertTrue(rows["visible"])
        self.assertIn(second, {value for _, value in names["choices"]})
        self.assertEqual(loader.show_progress, "hidden")

    def test_catalog_scans_standard_and_cli_roots_and_keeps_only_current_external_paths(self):
        primary = self.create("Lora/sub/style.safetensors")
        additional = self.create("cli/sub/style.safetensors")
        old = self.create("old/style.safetensors")
        current = self.create("current/style.safetensors")
        self.picker.shared.cmd_opts.lora_dirs = [str(self.root / "cli")]
        local_assets.remember("forge_lora", [old])
        self.assertEqual({value for _, value in self.picker.available()}, {primary, additional})
        self.assertEqual({value for _, value in self.picker.available([current])}, {primary, additional, current})

    def test_selection_injects_exact_distinct_files_and_individual_strengths(self):
        first = self.create("Lora/portrait/style.safetensors")
        second = self.create("Lora/landscape/style.safetensors")
        entries = {
            "portrait/style": SimpleNamespace(filename=first),
            "landscape/style": SimpleNamespace(filename=second),
        }
        self.networks.available_networks.update(entries)
        self.networks.available_network_aliases.update(entries)
        processing = self.processing()
        self.picker.apply_selections(processing, [first, second], [[first, 0.35], [second, -0.7]])
        self.assertIn("<lora:portrait/style:0.35>", processing.prompt)
        self.assertIn("<lora:landscape/style:-0.7>", processing.prompt)
        self.assertTrue(all(prompt.count("<lora:") == 2 for prompt in processing.all_prompts))
        self.assertEqual({entry.filename for entry in self.networks.available_networks.values()}, {first, second})

    def test_zero_skips_missing_weights_and_old_job_does_not_overwrite_latest_selection(self):
        current = self.create("Lora/current.safetensors")
        self.save((current, 0.7))
        previous = local_assets.selection("forge_lora")
        with patch.object(local_assets, "read_header", side_effect=AssertionError("zero must not open weights")):
            self.picker.apply_selections(self.processing(), ["missing.safetensors"], [["missing.safetensors", 0]])
        self.network.NetworkOnDisk.assert_not_called()
        self.assertEqual(local_assets.selection("forge_lora"), previous)

    def test_duplicate_in_expanded_prompt_is_rejected_by_canonical_file_identity(self):
        path = self.create("Lora/style.safetensors")
        entry = SimpleNamespace(filename=path)
        self.networks.available_networks["Anima/style"] = entry
        self.networks.available_network_aliases["Anima/style"] = SimpleNamespace(
            filename=self.create("other/style.safetensors")
        )
        processing = self.processing()
        processing.all_prompts = ["styled <lora:Anima/style:0.5>"]
        with self.assertRaisesRegex(ValueError, "同じファイル"):
            self.picker.apply_selections(processing, [path], [[path, 0.7]])
        self.assertEqual(processing.prompt, "portrait")

    def test_anima_and_stack_duplicate_file_is_rejected_in_either_callback_order(self):
        path = self.create("Lora/style.safetensors")
        entry = SimpleNamespace(filename=path)
        self.networks.available_networks["Anima/style"] = entry
        self.networks.available_network_aliases["Anima/style"] = entry
        for stack_first in (False, True):
            with self.subTest(stack_first=stack_first):
                processing = self.processing()
                stack = partial(self.picker.apply_selections, processing, [path], [[path, 0.7]])
                anima = partial(
                    self.anima.apply_standard_lora_selection, processing, "Anima/style", 0.5, {"Anima/style": path}
                )
                first, second = (stack, anima) if stack_first else (anima, stack)
                first()
                with self.assertRaisesRegex(ValueError, "同じファイル"):
                    second()
                self.assertEqual(processing.prompt.count("<lora:"), 1)


if __name__ == "__main__":
    unittest.main()
