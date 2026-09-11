import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

from modules_forge.krea2_upscale import (
    KREA2_DEFAULT_DENOISE,
    KREA2_DEFAULT_SAMPLER,
    KREA2_DEFAULT_SCHEDULER,
    KREA2_I2I_STEPS,
    KREA2_T2I_STEPS,
)
from modules_forge.presets import (
    DEFAULT_ADDITIONAL_MODULES,
    DEFAULT_CHECKPOINTS,
    DEFAULT_UNET_STORAGE_DTYPES,
    HIRES_DENOISE,
    HIRES_STEPS,
    I2I_DENOISE,
    I2I_STEPS,
    SAMPLERS,
    SCHEDULERS,
    STEPS,
    T2I_DIMENSIONS,
    PresetArch,
    register,
)

ROOT = Path(__file__).resolve().parents[2]


class Krea2PresetDefaultTests(unittest.TestCase):
    def test_krea_defaults_to_native_int8_convrot(self):
        self.assertEqual(
            DEFAULT_CHECKPOINTS[PresetArch.krea],
            "krea2_turbo_int8_convrot.safetensors",
        )
        self.assertEqual(
            DEFAULT_ADDITIONAL_MODULES[PresetArch.krea],
            (
                "qwen_image_vae.safetensors",
                "qwen3vl_4b_fp8_scaled.safetensors",
            ),
        )
        self.assertEqual(DEFAULT_UNET_STORAGE_DTYPES[PresetArch.krea], "Automatic")

    def test_registered_krea_options_use_int8_defaults(self):
        class FakeOptionInfo:
            def __init__(self, default=None, *_args, **_kwargs):
                self.default = default

        fake_gradio = ModuleType("gradio")
        fake_gradio.Dropdown = object
        fake_gradio.Slider = object
        fake_options = ModuleType("modules.options")
        fake_options.OptionInfo = FakeOptionInfo
        fake_options.OptionRow = FakeOptionInfo
        fake_options.options_section = lambda _section, values: values
        fake_shared_items = ModuleType("modules.shared_items")
        fake_shared_items.list_samplers = lambda: []
        fake_shared_items.list_schedulers = lambda: []
        options_templates = {}

        with mock.patch.dict(
            sys.modules,
            {
                "gradio": fake_gradio,
                "modules.options": fake_options,
                "modules.shared_items": fake_shared_items,
            },
        ):
            register(options_templates)

        self.assertEqual(
            options_templates["forge_checkpoint_krea"].default,
            "krea2_turbo_int8_convrot.safetensors",
        )
        self.assertEqual(
            options_templates["forge_additional_modules_krea"].default,
            [
                "qwen_image_vae.safetensors",
                "qwen3vl_4b_fp8_scaled.safetensors",
            ],
        )
        self.assertEqual(
            options_templates["forge_unet_storage_dtype_krea"].default,
            "Automatic",
        )

    def test_t2i_defaults_match_measured_turbo_baseline(self):
        self.assertEqual(SAMPLERS[PresetArch.krea], KREA2_DEFAULT_SAMPLER)
        self.assertEqual(SCHEDULERS[PresetArch.krea], KREA2_DEFAULT_SCHEDULER)
        self.assertEqual(STEPS[PresetArch.krea], KREA2_T2I_STEPS)
        self.assertEqual(T2I_DIMENSIONS[PresetArch.krea], (1024, 1024))

    def test_i2i_and_hires_keep_enough_steps_for_low_denoise(self):
        self.assertEqual(I2I_STEPS[PresetArch.krea], KREA2_I2I_STEPS)
        self.assertEqual(HIRES_STEPS[PresetArch.krea], KREA2_I2I_STEPS)
        self.assertEqual(I2I_DENOISE[PresetArch.krea], KREA2_DEFAULT_DENOISE)
        self.assertEqual(HIRES_DENOISE[PresetArch.krea], KREA2_DEFAULT_DENOISE)

    def test_switching_away_from_krea_restores_standard_denoise(self):
        for arch in PresetArch:
            if arch is PresetArch.krea:
                continue
            with self.subTest(arch=arch.name):
                self.assertEqual(I2I_DENOISE[arch], 0.60)
                self.assertEqual(HIRES_DENOISE[arch], 0.60)

    def test_preset_callback_output_layout_stays_in_sync(self):
        tree = ast.parse(
            (ROOT / "modules_forge" / "main_entry.py").read_text(encoding="utf-8")
        )
        forge_main_entry = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "forge_main_entry"
        )
        on_preset_change = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "on_preset_change"
        )
        output_targets = next(
            node.value
            for node in forge_main_entry.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "output_targets"
                for target in node.targets
            )
        )
        callback_outputs = next(
            node.value
            for node in on_preset_change.body
            if isinstance(node, ast.Return)
        )

        self.assertEqual(len(output_targets.elts), len(callback_outputs.elts))
        target_names = [element.id for element in output_targets.elts]
        self.assertIn("ui_txt2img_hr_denoise", target_names)
        self.assertIn("ui_img2img_denoise", target_names)

    def test_preset_load_saves_and_refreshes_once_when_checkpoint_is_unchanged(self):
        tree = ast.parse((ROOT / "modules_forge" / "main_entry.py").read_text(encoding="utf-8"))
        load_presets = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_presets"
        )
        namespace = {}
        exec(  # noqa: S102 - execute only the extracted local callback in isolation
            compile(
                ast.fix_missing_locations(ast.Module(body=[load_presets], type_ignores=[])),
                filename="modules_forge/main_entry.py",
                mode="exec",
            ),
            namespace,
        )

        calls = mock.Mock()
        calls.resolve_module_values.return_value = ["resolved-vae.safetensors"]
        calls.sd_models.get_closet_checkpoint_match.return_value = object()
        calls.checkpoint_change.return_value = False
        namespace.update(
            resolve_module_values=calls.resolve_module_values,
            dtype_change=calls.dtype_change,
            modules_change=calls.modules_change,
            checkpoint_change=calls.checkpoint_change,
            refresh_model_loading_parameters=calls.refresh_model_loading_parameters,
            sd_models=calls.sd_models,
            shared=mock.Mock(
                config_filename="config.json",
                opts=mock.Mock(set=calls.set, save=calls.save),
            ),
        )

        namespace["_load_presets"](
            "unchanged.safetensors",
            ["vae.safetensors"],
            "Automatic",
            "krea",
        )

        self.assertEqual(
            calls.mock_calls,
            [
                mock.call.resolve_module_values(["vae.safetensors"]),
                mock.call.sd_models.get_closet_checkpoint_match("unchanged.safetensors"),
                mock.call.set("forge_preset", "krea"),
                mock.call.dtype_change("Automatic", "krea", save=False, refresh=False),
                mock.call.modules_change(
                    ["resolved-vae.safetensors"],
                    "krea",
                    save=False,
                    refresh=False,
                ),
                mock.call.checkpoint_change("unchanged.safetensors", "krea", save=False, refresh=False),
                mock.call.save("config.json"),
                mock.call.refresh_model_loading_parameters(),
            ],
        )


if __name__ == "__main__":
    unittest.main()
