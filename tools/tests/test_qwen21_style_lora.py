"""LoRA compatibility boundaries, residual math, and multiple adapter contracts."""

import asyncio
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from modules_forge.qwen_image21 import style_lora, style_lora_runtime
from modules_forge.qwen_image21.core import Request
from modules_forge.qwen_image21.style_lora_runtime import load_adapters


class StyleLoraTests(unittest.TestCase):
    def setUp(self):
        from tools.tests.test_qwen_image21_service import isolate_local_assets

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        isolate_local_assets(self, self.root)
        (self.root / "loras").mkdir()
        self.path = self.root / "loras/test.safetensors"
        self.prefix = "lora_unet_transformer_blocks__0__attn__to_q"
        self.tensors = {
            self.prefix + ".lora_down.weight": torch.ones(2, 3),
            self.prefix + ".lora_up.weight": torch.ones(4, 2),
            self.prefix + ".alpha": torch.tensor(1.0),
        }
        self.write()

    def write(self, metadata=None):
        save_file(self.tensors, self.path, metadata=metadata or {"model_type": "qwen_image_21"})

    def test_pair_and_unknown_keys_are_not_silently_skipped(self):
        self.assertEqual(style_lora.inspect(self.root, self.path.name)["linear_layers"], 1)
        self.tensors["unrecognised.weight"] = torch.ones(1)
        self.write()
        with self.assertRaisesRegex(ValueError, "未対応"):
            style_lora.inspect(self.root, self.path.name)
        del self.tensors["unrecognised.weight"]
        del self.tensors[self.prefix + ".lora_up.weight"]
        self.write()
        with self.assertRaisesRegex(ValueError, "ペア"):
            style_lora.inspect(self.root, self.path.name)

    def test_consistency_is_identified_by_content_after_rename_or_external_path(self):
        from modules_forge.qwen_image21 import consistency_lora

        self.write({"ss_output_name": "qwen21_consistency_v1"})
        digest = style_lora.fingerprint(self.path)
        external = self.root / "renamed.safetensors"
        external.write_bytes(self.path.read_bytes())
        with patch.dict(consistency_lora.HASHES, {"1500": digest}):
            for name in (self.path.name, str(external)):
                info = style_lora.inspect(self.root, name)
                self.assertEqual(info["consistency_version"], "1500")
                self.assertIn("ausboss", info["source"])
                self.assertEqual(info["training_base_forward"], "")
        self.write({"ss_output_name": "qwen21_consistency_v1"})
        self.assertNotIn("consistency_version", style_lora.inspect(self.root, self.path.name))

    def test_training_auxiliaries_require_known_metadata(self):
        self.tensors["qwen_partition_global_adapter.alpha"] = torch.tensor(1.0)
        self.write()
        with self.assertRaisesRegex(ValueError, "未対応"):
            style_lora.inspect(self.root, self.path.name)
        self.write(
            {"qwen_partition_global_adapter": "latent_summary_v1", "qwen_base_forward": "convrot_int8_bf16_backward_v1"}
        )
        values = Request(prompt="test", style_loras=({"name": self.path.name, "strength": 1.0},)).to_dict()
        info = style_lora.validate_installed(self.root, values)[0]
        self.assertEqual(info["training_base_forward"], "convrot_int8_bf16_backward_v1")
        self.assertEqual(info["training_only_keys"], ["qwen_partition_global_adapter.alpha"])

    def options(self, strength=1.0, **extra):
        return {"style_loras": [{"name": self.path.name, "strength": strength}], "precision": "int8", **extra}

    def model(self):
        base = torch.nn.Linear(3, 4, bias=False, dtype=torch.bfloat16)
        model = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.to_q = base
        model.transformer_blocks = torch.nn.ModuleList([block])
        return model, base

    def test_path_escape_and_incompatible_modes(self):
        with self.assertRaises(ValueError):
            style_lora.resolve(self.root, "../outside.safetensors")
        for changes in ({"fun_acc": True}, {"sparse_mode": "static"}):
            with self.assertRaises(ValueError):
                Request(prompt="test", **self.options(**changes)).resolved()
        for changes in (
            {"fun_acc": True, "steps": 4},
            {"precision": "turbo_bf16"},
            {"precision": "turbo_bf16", "steps": 4},
            {"sparse_mode": "fixed"},
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    Request(prompt="test", **self.options(**changes)).resolved().style_loras[0]["strength"], 1
                )
        for strength in (float("nan"), float("inf"), 2.01, True, "1", None):
            with self.assertRaisesRegex(ValueError, self.path.name):
                style_lora.validate_options(self.options(strength))
        duplicate = self.options()
        duplicate["style_loras"] *= 2
        with self.assertRaisesRegex(ValueError, "重複"):
            style_lora.validate_options(duplicate)

    def test_alpha_strength_residual_and_base_unchanged(self):
        model, base = self.model()
        original = base.weight.detach().clone()
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        before = base(x)
        result = load_adapters(model, self.root, self.options(0.5))
        self.assertEqual(result[0]["applied_layers"], 1)
        torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(x), before + 1.5)
        torch.testing.assert_close(base.weight, original)

    def test_two_adapters_on_same_layer_add_independent_residuals(self):
        save_file(self.tensors, self.root / "loras/second.safetensors")
        model, base = self.model()
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        before = base(x)
        values = self.options(0.5)
        values["style_loras"].append({"name": "second.safetensors", "strength": -0.25})
        result = load_adapters(model, self.root, values)
        self.assertEqual([item["strength"] for item in result], [0.5, -0.25])
        self.assertEqual([item["applied_layers"] for item in result], [1, 1])
        torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(x), before + 0.75, atol=0.016, rtol=0.016)
        self.assertIs(model.transformer_blocks[0].attn.to_q.base.base, base)

    def test_invalid_second_adapter_never_partially_mutates_model(self):
        bad = {key: value for key, value in self.tensors.items()}
        bad[self.prefix + ".lora_up.weight"] = torch.ones(5, 2)
        save_file(bad, self.root / "loras/bad.safetensors")
        model, base = self.model()
        values = self.options()
        values["style_loras"].append({"name": "bad.safetensors", "strength": 1})
        with self.assertRaisesRegex(ValueError, "bad.safetensors"):
            load_adapters(model, self.root, values)
        self.assertIs(model.transformer_blocks[0].attn.to_q, base)

    def test_file_changed_during_load_never_mutates_model_or_records_new_hash(self):
        model, base = self.model()
        original_load = style_lora_runtime.load_file

        def changed_load(*args, **kwargs):
            state = original_load(*args, **kwargs)
            self.tensors[self.prefix + ".lora_up.weight"] *= 2
            self.write()
            return state

        with patch.object(style_lora_runtime, "load_file", side_effect=changed_load):
            with self.assertRaisesRegex(ValueError, "LoRA.*changed"):
                load_adapters(model, self.root, self.options())
        self.assertIs(model.transformer_blocks[0].attn.to_q, base)

    def test_first_adapter_changed_while_loading_second_never_mutates_model(self):
        second = self.root / "loras/second.safetensors"
        save_file(self.tensors, second)
        values = self.options()
        values["style_loras"].append({"name": second.name, "strength": 1})
        model, base = self.model()
        original_load = style_lora_runtime.load_file
        mutations = []

        def changed_load(path, **kwargs):
            state = original_load(path, **kwargs)
            if Path(path).resolve() == second.resolve():
                self.tensors[self.prefix + ".lora_up.weight"] *= 2
                self.write()
                mutations.append(Path(path).resolve())
            return state

        with patch.object(style_lora_runtime, "load_file", side_effect=changed_load):
            with self.assertRaisesRegex(ValueError, "LoRA.*changed"):
                load_adapters(model, self.root, values)
        self.assertIs(model.transformer_blocks[0].attn.to_q, base)
        self.assertEqual(mutations, [second.resolve()])

    def test_zero_skips_loading_and_cache_tracks_files_instead_of_strength(self):
        model, base = self.model()
        with patch("modules_forge.qwen_image21.style_lora_runtime.load_file") as load:
            self.assertEqual(load_adapters(model, self.root, self.options(0)), [])
        load.assert_not_called()
        self.assertIs(model.transformer_blocks[0].attn.to_q, base)
        key = style_lora.cache_key(self.root, self.options())
        self.assertEqual(key, style_lora.cache_key(self.root, self.options(0.5)))
        self.assertNotEqual(key, style_lora.cache_key(self.root, {"style_loras": []}))
        self.tensors[self.prefix + ".alpha"] = torch.ones(1, 1)
        self.write()
        self.assertNotEqual(key, style_lora.cache_key(self.root, self.options()))

    def test_reused_adapters_update_independent_alpha_scales_and_zero_metadata(self):
        second = self.root / "loras/second.safetensors"
        save_file(self.tensors | {self.prefix + ".alpha": torch.tensor(2.0)}, second)
        model, base = self.model()
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        before = base(x)
        values = self.options(0.5)
        values["style_loras"].append({"name": second.name, "strength": -0.25})
        loaded = load_adapters(model, self.root, values)
        wrapper = model.transformer_blocks[0].attn.to_q
        values["style_loras"] = [{"name": second.name, "strength": 0.75}, {"name": self.path.name, "strength": -1}]
        with patch("modules_forge.qwen_image21.style_lora_runtime.load_file") as load:
            current = style_lora_runtime.update_strengths(model, loaded, self.root, values)
            self.assertEqual([item["name"] for item in current], [second.name, self.path.name])
            self.assertEqual([item["strength"] for item in current], [0.75, -1])
            torch.testing.assert_close(wrapper(x), before + 1.5, atol=0.016, rtol=0.016)
            values["style_loras"] = [{"name": self.path.name, "strength": 0}, {"name": second.name, "strength": 0}]
            self.assertEqual(style_lora_runtime.update_strengths(model, loaded, self.root, values), [])
            torch.testing.assert_close(wrapper(x), before)
            values["style_loras"] = [{"name": self.path.name, "strength": 1}]
            current = style_lora_runtime.update_strengths(model, loaded, self.root, values)
            torch.testing.assert_close(wrapper(x), before + 3, atol=0.016, rtol=0.016)
            self.assertEqual(current[0]["strength"], 1)
        load.assert_not_called()
        self.assertIs(model.transformer_blocks[0].attn.to_q, wrapper)
        self.assertIs(wrapper.base.base, base)

    def test_unloaded_adapter_update_fails_before_changing_any_scale(self):
        model, base = self.model()
        loaded = load_adapters(model, self.root, self.options(0.5))
        save_file(self.tensors, self.root / "loras/second.safetensors")
        values = self.options(-1)
        values["style_loras"].append({"name": "second.safetensors", "strength": 1})
        with self.assertRaisesRegex(ValueError, "読み込み"):
            style_lora_runtime.update_strengths(model, loaded, self.root, values)
        self.assertEqual(model.transformer_blocks[0].attn.to_q.scale, 0.25)
        self.assertIs(model.transformer_blocks[0].attn.to_q.base, base)

    def test_updating_style_strength_preserves_sampling_and_outpaint_residuals(self):
        from modules_forge.qwen_image21.outpaint_runtime import OutpaintLinear
        from modules_forge.qwen_image21.pdd_vendor.lora_utils_pdd import PDDLoRALinear

        model, base = self.model()
        sampling = PDDLoRALinear(base, rank=2, alpha=1)
        with torch.no_grad():
            sampling.lora_down.fill_(1)
            sampling.lora_up.fill_(1)
        outpaint = OutpaintLinear(sampling, torch.ones(2, 3), torch.ones(4, 2))
        model.transformer_blocks[0].attn.to_q = outpaint
        x = torch.ones(1, 3, dtype=torch.bfloat16)
        before = outpaint(x)
        loaded = load_adapters(model, self.root, self.options())
        style_lora_runtime.update_strengths(model, loaded, self.root, self.options(0))
        torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(x), before)
        self.assertIs(model.transformer_blocks[0].attn.to_q.base, outpaint)
        style_lora_runtime.update_strengths(model, loaded, self.root, self.options(-0.5))
        torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(x), before - 1.5, rtol=0.016, atol=0.016)

    def test_training_base_metadata_does_not_restrict_application_or_reuse(self):
        for training_base in ("convrot_int8_bf16_backward_v1", "bf16", "future_distilled_base_v3"):
            for precision in ("int8", "base_q4_k_m", "turbo_bf16"):
                with self.subTest(training_base=training_base, precision=precision):
                    self.write({"qwen_base_forward": training_base})
                    model, base = self.model()
                    options = self.options(precision=precision, steps=17, local_model="selected.safetensors")
                    original = dict(options)
                    x = torch.ones(1, 3, dtype=torch.bfloat16)
                    before = base(x)
                    loaded = load_adapters(model, self.root, options)
                    wrapper = model.transformer_blocks[0].attn.to_q
                    torch.testing.assert_close(wrapper(x), before + 3, rtol=0.016, atol=0.016)
                    current = style_lora_runtime.update_strengths(
                        model, loaded, self.root, self.options(0.25, precision=precision)
                    )
                    self.assertEqual(current[0]["training_base_forward"], training_base)
                    self.assertNotIn("experimental_base_mismatch", current[0])
                    self.assertEqual(wrapper.scale, 0.125)
                    torch.testing.assert_close(wrapper(x), before + 0.75, rtol=0.016, atol=0.016)
                    self.assertEqual(options, original)
                    self.assertEqual(style_lora_runtime.update_strengths(model, loaded, self.root, self.options(0)), [])
                    self.assertEqual(wrapper.scale, 0)

    def test_prepare_command_records_training_base_as_information(self):
        from types import SimpleNamespace

        from tools import prepare_qwen21_style_lora as prepare

        self.write({"qwen_base_forward": "future_distilled_base_v3"})
        downloaded = self.root / "models/Qwen-Image-2.1/loras/owner--repo/weights.safetensors"
        downloaded.parent.mkdir(parents=True)
        downloaded.write_bytes(self.path.read_bytes())
        item = SimpleNamespace(
            rfilename=downloaded.name, lfs=SimpleNamespace(sha256=style_lora.fingerprint(downloaded))
        )
        repository = SimpleNamespace(sha="fixed-revision", siblings=[item])
        with (
            patch.object(prepare, "ROOT", self.root),
            patch("sys.argv", ["prepare_qwen21_style_lora.py", "owner/repo"]),
            patch("huggingface_hub.HfApi.model_info", return_value=repository),
            patch("huggingface_hub.hf_hub_download", return_value=str(downloaded)),
            patch("sys.stdout", io.StringIO()),
        ):
            prepare.main()
        receipt = json.loads(downloaded.with_suffix(".source.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["training_base_forward"], "future_distilled_base_v3")
        self.assertNotIn("base_mismatch", receipt)

    def test_ui_accepts_training_base_information_without_confirmation(self):
        from types import SimpleNamespace

        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        self.write({"qwen_base_forward": "convrot_int8_bf16_backward_v1"})
        with patch.object(ui, "RUNTIME", self.root):
            enabled = ui.lora_readiness([self.path.name], [[self.path.name, 1]], "", SimpleNamespace())
        self.assertTrue(enabled[2]["interactive"])
        self.assertEqual(enabled[0]["label"], "LoRA · 1件")
        self.assertEqual(enabled[1], "")

    def test_ui_add_remove_preserves_strength_by_name_not_row(self):
        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        with patch.object(ui, "RUNTIME", self.root):
            table, _ = ui.lora_selection(["other.safetensors", self.path.name], [[self.path.name, 0.65]])
            self.assertEqual(table["value"], [["other.safetensors", 1.0], [self.path.name, 0.65]])
            table, _ = ui.lora_selection([self.path.name], table["value"])
            self.assertEqual(table["value"], [[self.path.name, 0.65]])
            settings = ui.lora_settings([self.path.name], [["irrelevant", 0.2], [self.path.name, 0.65]])
            self.assertEqual(settings, ({"name": self.path.name, "strength": 0.65},))
            self.assertEqual(ui.lora_settings([self.path.name], [[self.path.name, "0.65"]]), settings)
            with self.assertRaisesRegex(ValueError, self.path.name):
                ui.lora_settings([self.path.name], [[self.path.name, "bad"]])
            with self.assertRaisesRegex(ValueError, "更新"):
                ui.lora_settings([self.path.name], [])

    def test_reopening_settings_preserves_strength_without_training_base_gate(self):
        from tools.tests.test_qwen_image21_service import load_ui

        ui = load_ui()
        self.write({"qwen_base_forward": "convrot_int8_bf16_backward_v1"})
        names, rows = [self.path.name], [[self.path.name, 0.65]]
        with patch.object(ui, "RUNTIME", self.root):
            demo = ui.on_ui_tabs()[0][0]
            self.addCleanup(demo.close)
            config = demo.get_config_file()
            section = next(
                item["id"] for item in config["components"] if item["props"].get("elem_id") == "qwen21-lora-section"
            )
            expand = next(
                item
                for item in config["dependencies"]
                if any(target[0] == section and target[1] == "expand" for target in item["targets"])
            )
            self.assertEqual(len(expand["inputs"]), 2)
            reopened = asyncio.run(demo.call_function(expand["id"], [names, rows]))["prediction"]
            self.assertEqual(len(reopened), 2)
            self.assertEqual(reopened[0]["value"], rows)
            self.assertFalse(
                any(item["props"].get("label") == "異なる量子化で試す（実験）" for item in config["components"])
            )
            self.assertNotIn("allow_lora_base_mismatch", Request.__dataclass_fields__)


if __name__ == "__main__":
    unittest.main()
