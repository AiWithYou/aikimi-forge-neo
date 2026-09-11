from __future__ import annotations

import ast
import importlib.util
import os
import sys
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
MAIN_ENTRY = ROOT / "modules_forge" / "main_entry.py"
INFOTEXT_UTILS = ROOT / "modules" / "infotext_utils.py"
PROCESSING = ROOT / "modules" / "processing.py"
PID_SCRIPT = ROOT / "extensions-builtin" / "sd_forge_pid" / "scripts" / "pid.py"
CHECKPOINT_METADATA = ROOT / "modules" / "ui_extra_networks_checkpoints_user_metadata.py"


class FakeOptions:
    def __init__(self):
        self.data = {
            "forge_additional_modules": [],
            "forge_additional_modules_sd": [],
            "forge_additional_modules_krea": [],
        }
        self.set_calls = []
        self.save_calls = []
        self.sd_model_checkpoint = "model.safetensors"
        self.forge_preset = "sd"
        self.sd_checkpoint_dropdown_use_short = False

    def __getattr__(self, name):
        try:
            return self.data[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def set(self, name, value, **kwargs):
        self.set_calls.append((name, value, kwargs))
        changed = self.data.get(name) != value
        self.data[name] = value
        return changed

    def save(self, filename):
        self.save_calls.append(filename)


def load_main_entry():
    backend = ModuleType("backend")
    memory_management = ModuleType("backend.memory_management")
    memory_management.bnb_enabled = lambda: False
    backend.memory_management = memory_management

    backend_args = ModuleType("backend.args")
    backend_args.dynamic_args = SimpleNamespace()
    backend_logging = ModuleType("backend.logging")
    backend_logging.setup_logger = lambda _logger: None

    modules = ModuleType("modules")
    modules.__path__ = []
    module_stubs = {}
    for name in (
        "infotext_utils",
        "paths",
        "processing",
        "sd_models",
        "shared",
        "shared_items",
        "ui_common",
    ):
        stub = ModuleType(f"modules.{name}")
        setattr(modules, name, stub)
        module_stubs[f"modules.{name}"] = stub

    opts = FakeOptions()
    shared = module_stubs["modules.shared"]
    shared.opts = opts
    shared.cmd_opts = SimpleNamespace(
        vae_dirs=[],
        text_encoder_dirs=[],
        freeze_settings=False,
    )
    shared.config_filename = "config.json"
    module_stubs["modules.paths"].models_path = "models"
    module_stubs["modules.sd_models"].get_closet_checkpoint_match = lambda value: (
        value if value in {"model.safetensors", "other.safetensors"} else None
    )

    presets = ModuleType("modules_forge.presets")

    class PresetArch:
        krea = SimpleNamespace(name="krea")

        @staticmethod
        def choices():
            return ["sd", "krea"]

    presets.PresetArch = PresetArch
    presets.is_video = lambda _preset: 0
    presets.use_distill = lambda _preset: False
    presets.use_shift = lambda _preset: False

    stubs = {
        "backend": backend,
        "backend.memory_management": memory_management,
        "backend.args": backend_args,
        "backend.logging": backend_logging,
        "modules": modules,
        "modules_forge.presets": presets,
        **module_stubs,
    }
    module_name = "_test_additional_module_main_entry"
    spec = importlib.util.spec_from_file_location(module_name, MAIN_ENTRY)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module, opts, shared


def load_infotext_module_helper(main_entry, opts):
    tree = ast.parse(INFOTEXT_UTILS.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_additional_module_parameters"
    )
    namespace = {
        "Any": object,
        "main_entry": main_entry,
        "shared": SimpleNamespace(opts=opts),
    }
    exec(  # noqa: S102 - execute only the extracted repository helper in isolation
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            filename=str(INFOTEXT_UTILS),
            mode="exec",
        ),
        namespace,
    )
    return namespace["_resolve_additional_module_parameters"]


def load_checkpoint_metadata_save(main_entry):
    tree = ast.parse(CHECKPOINT_METADATA.read_text(encoding="utf-8"))
    editor = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CheckpointUserMetadataEditor"
    )
    method = next(
        node for node in editor.body if isinstance(node, ast.FunctionDef) and node.name == "save_user_metadata"
    )
    namespace = {"main_entry": main_entry}
    exec(  # noqa: S102 - execute only the extracted repository method in isolation
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            filename=str(CHECKPOINT_METADATA),
            mode="exec",
        ),
        namespace,
    )
    return namespace["save_user_metadata"]


class AdditionalModuleIdentityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.main_entry, cls.opts, cls.shared = load_main_entry()

    def setUp(self):
        self.opts.data = {
            "forge_additional_modules": [],
            "forge_additional_modules_sd": [],
            "forge_additional_modules_krea": [],
        }
        self.opts.set_calls.clear()
        self.opts.save_calls.clear()
        self.opts.sd_model_checkpoint = "model.safetensors"
        self.opts.forge_preset = "sd"
        self.main_entry.module_list.clear()
        self.main_entry._module_path_to_selector.clear()
        self.main_entry._module_selector_to_kind.clear()
        self.main_entry._module_selector_to_infotext.clear()
        self.main_entry._module_basename_to_selectors.clear()
        self.main_entry._module_infotext_to_selectors.clear()
        self.main_entry._unresolved_module_choices.clear()
        self.main_entry._module_registry_initialized = False
        self.main_entry._legacy_module_migration_complete = False

    def build_registry(self, root_a: Path, root_b: Path):
        self.main_entry._rebuild_module_registry([("VAE", str(root_a)), ("text_encoder", str(root_b))])

    def test_duplicate_basenames_keep_distinct_safe_selectors_and_exact_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            vae_module = vae / "shared.safetensors"
            encoder_module = encoder / "shared.safetensors"
            vae_module.touch()
            encoder_module.touch()

            self.build_registry(vae, encoder)

            selectors = sorted(self.main_entry.module_list)
            self.assertEqual(len(selectors), 2)
            self.assertNotEqual(selectors[0], selectors[1])
            self.assertTrue(all(selector.startswith("shared [") for selector in selectors))
            self.assertNotIn(str(root), repr(self.main_entry.module_dropdown_choices()))
            self.assertEqual(
                {self.main_entry.resolve_module_value(selector) for selector in selectors},
                {str(vae_module.resolve()), str(encoder_module.resolve())},
            )
            self.assertEqual(
                {self.main_entry.module_kind(selector) for selector in selectors},
                {"VAE", "text_encoder"},
            )
            with self.assertRaisesRegex(
                self.main_entry.ModuleResolutionError,
                "matches multiple files",
            ):
                self.main_entry.resolve_module_value("shared.safetensors")

    def test_discovery_is_repeatable_and_uses_real_casefolded_suffixes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            (vae / "z").mkdir(parents=True)
            encoder.mkdir()
            (vae / "UPPER.SAFETENSORS").touch()
            (vae / "not-a-real-safetensors").touch()
            (vae / "z" / "same.safetensors").touch()
            (encoder / "same.safetensors").touch()

            self.build_registry(vae, encoder)
            first = dict(self.main_entry.module_list)
            self.main_entry._rebuild_module_registry([("VAE", str(vae)), ("text_encoder", str(encoder))])

            self.assertEqual(first, self.main_entry.module_list)
            self.assertTrue(any(selector.casefold() == "upper.safetensors" for selector in first))
            self.assertFalse(any("not-a-real-safetensors" in selector for selector in first))

    def test_registry_persists_the_canonical_path_instead_of_a_short_alias(self):
        short_root = os.fspath(Path("RUNNER~1") / "modules")
        canonical_root = os.fspath(Path("runneradmin") / "modules")
        short_file = os.fspath(Path(short_root) / "encoder.safetensors")
        canonical_file = os.fspath(Path(canonical_root) / "encoder.safetensors")

        def canonicalize(value):
            return {
                short_root: canonical_root,
                short_file: canonical_file,
            }.get(os.fspath(value), os.fspath(value))

        with (
            mock.patch.object(
                self.main_entry,
                "find_files_with_extensions",
                return_value=[short_file],
            ),
            mock.patch.object(
                self.main_entry,
                "_canonical_module_path",
                side_effect=canonicalize,
            ),
        ):
            self.main_entry._rebuild_module_registry([("text_encoder", short_root)])

        self.assertEqual(
            self.main_entry.module_list,
            {"encoder.safetensors": canonical_file},
        )

    def test_exact_saved_path_is_remapped_when_a_new_duplicate_appears(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            saved_path = vae / "shared.safetensors"
            saved_path.touch()
            self.build_registry(vae, encoder)
            self.assertEqual(
                self.main_entry.module_values_to_ui_selectors([str(saved_path)]),
                ["shared.safetensors"],
            )

            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            remapped = self.main_entry.module_values_to_ui_selectors([str(saved_path)])

            self.assertEqual(len(remapped), 1)
            self.assertNotEqual(remapped, ["shared.safetensors"])
            self.assertEqual(
                self.main_entry.resolve_module_values(remapped),
                [str(saved_path.resolve())],
            )

    def test_modules_change_persists_exact_path_and_fails_atomically(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            selected_path = vae / "shared.safetensors"
            selected_path.touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            selected = self.main_entry.module_values_to_ui_selectors([str(selected_path)])[0]

            changed = self.main_entry.modules_change([selected], "krea", save=False, refresh=False)

            expected = [str(selected_path.resolve())]
            self.assertTrue(changed)
            self.assertEqual(self.opts.data["forge_additional_modules"], expected)
            self.assertEqual(self.opts.data["forge_additional_modules_krea"], expected)

            before = dict(self.opts.data)
            calls_before = list(self.opts.set_calls)
            with self.assertRaises(self.main_entry.ModuleResolutionError):
                self.main_entry.modules_change(["shared.safetensors"], "krea", save=True, refresh=True)
            self.assertEqual(self.opts.data, before)
            self.assertEqual(self.opts.set_calls, calls_before)
            self.assertEqual(self.opts.save_calls, [])

    def test_generation_override_accepts_only_safe_selectors_or_unique_basenames(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            duplicate = vae / "shared.safetensors"
            unique = vae / "unique.safetensors"
            duplicate.touch()
            unique.touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            duplicate_selector = self.main_entry.module_values_to_ui_selectors([str(duplicate)])[0]

            resolved = self.main_entry.resolve_generation_module_values([duplicate_selector, "unique.safetensors"])

            self.assertEqual(
                resolved,
                sorted(
                    [str(duplicate.resolve()), str(unique.resolve())],
                    key=self.main_entry._normalize_module_path,
                ),
            )

    def test_generation_override_rejects_paths_ambiguous_unknown_and_unresolved_values(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            unique = vae / "unique.safetensors"
            unique.touch()
            (vae / "shared.safetensors").touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            unresolved = self.main_entry.module_values_to_ui_selectors(["missing.safetensors"])[0]

            for value, reason in (
                (str(unique.resolve()), "path"),
                ("vae/unique.safetensors", "path"),
                ("vae\\unique.safetensors", "path"),
                ("shared.safetensors", "ambiguous"),
                ("missing.safetensors", "missing"),
                (unresolved, "missing"),
            ):
                with self.subTest(value=value, reason=reason):
                    with self.assertRaises(self.main_entry.ModuleResolutionError) as caught:
                        self.main_entry.resolve_generation_module_values([value])
                    self.assertEqual(caught.exception.reason, reason)

    def test_generation_override_lazily_builds_registry_for_api_only_startup_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            vae = models / "VAE"
            encoder = models / "text_encoder"
            vae.mkdir(parents=True)
            encoder.mkdir()
            selected = encoder / "api-only.safetensors"
            selected.touch()

            with (
                mock.patch.object(self.main_entry.paths, "models_path", str(models)),
                mock.patch.object(
                    self.main_entry,
                    "_rebuild_module_registry",
                    wraps=self.main_entry._rebuild_module_registry,
                ) as rebuild,
                mock.patch.object(
                    self.main_entry.shared_items,
                    "refresh_checkpoints",
                    create=True,
                ) as refresh_checkpoints,
            ):
                first = self.main_entry.resolve_generation_module_values([selected.name])
                second = self.main_entry.resolve_generation_module_values([selected.name])

            self.assertEqual(first, [str(selected.resolve())])
            self.assertEqual(second, first)
            rebuild.assert_called_once()
            refresh_checkpoints.assert_not_called()
            self.assertEqual(self.opts.save_calls, [])

    def test_preset_load_validates_modules_before_changing_any_setting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            (vae / "shared.safetensors").touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            before = dict(self.opts.data)

            with self.assertRaises(self.main_entry.ModuleResolutionError):
                self.main_entry._load_presets(
                    "other.safetensors",
                    ["shared.safetensors"],
                    "float8-e4m3fn",
                    "krea",
                )

            self.assertEqual(self.opts.data, before)
            self.assertEqual(self.opts.set_calls, [])
            self.assertEqual(self.opts.save_calls, [])

    def test_preset_load_validates_checkpoint_before_changing_any_setting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            (vae / "unique.safetensors").touch()
            self.build_registry(vae, encoder)
            before = dict(self.opts.data)

            with self.assertRaisesRegex(ValueError, "checkpoint is not available"):
                self.main_entry._load_presets(
                    "missing.safetensors",
                    ["unique.safetensors"],
                    "float8-e4m3fn",
                    "krea",
                )

            self.assertEqual(self.opts.data, before)
            self.assertEqual(self.opts.set_calls, [])
            self.assertEqual(self.opts.save_calls, [])

    def test_preset_load_passes_prevalidated_exact_module_paths(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            selected = vae / "unique.safetensors"
            selected.touch()
            self.build_registry(vae, encoder)

            with (
                mock.patch.object(
                    self.main_entry,
                    "modules_change",
                    wraps=self.main_entry.modules_change,
                ) as modules_change,
                mock.patch.object(self.main_entry, "refresh_model_loading_parameters") as refresh,
            ):
                self.main_entry._load_presets(
                    "other.safetensors",
                    ["unique.safetensors"],
                    "float8-e4m3fn",
                    "krea",
                )

            modules_change.assert_called_once_with(
                [str(selected.resolve())],
                "krea",
                save=False,
                refresh=False,
            )
            self.assertEqual(
                self.opts.data["forge_additional_modules"],
                [str(selected.resolve())],
            )
            self.assertEqual(self.opts.data["forge_unet_storage_dtype"], "float8-e4m3fn")
            self.assertEqual(self.opts.data["forge_preset"], "krea")
            self.assertEqual(self.opts.data["sd_model_checkpoint"], "other.safetensors")
            self.assertEqual(self.opts.save_calls, ["config.json"])
            self.assertEqual(
                refresh.call_args_list,
                [mock.call(refresh=False), mock.call(refresh=False), mock.call(refresh=False), mock.call()],
            )

    def test_preset_ui_update_does_not_persist_before_validated_load(self):
        tree = ast.parse(MAIN_ENTRY.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "on_preset_change"
        )
        mutations = [
            node.attr for node in ast.walk(function) if isinstance(node, ast.Attribute) and node.attr in {"set", "save"}
        ]

        self.assertEqual(mutations, [])

    def test_unset_preset_checkpoint_uses_the_current_checkpoint(self):
        self.opts.data["forge_checkpoint_krea"] = None

        updates = self.main_entry.on_preset_change("krea")

        self.assertEqual(updates[0]["value"], "model.safetensors")
        source = MAIN_ENTRY.read_text(encoding="utf-8")
        self.assertEqual(
            source.count(") or shared.opts.sd_model_checkpoint"),
            2,
        )

    def test_failed_preset_load_recovers_the_committed_preset_ui(self):
        self.opts.forge_preset = "anima"
        self.opts.data["forge_checkpoint_anima"] = None

        updates = self.main_entry.recover_preset_ui()

        self.assertEqual(updates[0]["value"], "anima")
        self.assertEqual(updates[1]["value"], "model.safetensors")
        source = MAIN_ENTRY.read_text(encoding="utf-8")
        self.assertIn("preset_load.failure(", source)
        self.assertIn("outputs=[ui_forge_preset, *output_targets]", source)

    def test_unique_legacy_values_migrate_once_but_ambiguous_values_do_not(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            unique = vae / "unique.safetensors"
            unique.touch()
            (vae / "shared.safetensors").touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            self.opts.data["forge_additional_modules_krea"] = ["unique.safetensors"]
            self.opts.data["forge_additional_modules_sd"] = ["shared.safetensors"]

            self.main_entry._migrate_legacy_module_options_once()
            self.main_entry._migrate_legacy_module_options_once()

            self.assertEqual(
                self.opts.data["forge_additional_modules_krea"],
                [str(unique.resolve())],
            )
            self.assertEqual(
                self.opts.data["forge_additional_modules_sd"],
                ["shared.safetensors"],
            )
            self.assertEqual(self.opts.save_calls, ["config.json"])

            unresolved = self.main_entry.module_values_to_ui_selectors(["shared.safetensors"])
            self.assertTrue(unresolved[0].startswith(self.main_entry._UNRESOLVED_MODULE_PREFIX))
            choices = self.main_entry.module_dropdown_choices()
            self.assertIn("ambiguous; reselect", repr(choices))
            self.assertNotIn(str(root), repr(choices))
            with warnings.catch_warnings(record=True) as caught:
                dropdown = self.main_entry.gr.Dropdown(
                    choices=choices,
                    value=unresolved,
                    multiselect=True,
                )
            self.assertEqual(dropdown.value, unresolved)
            self.assertFalse(
                any("not in the list of choices" in str(item.message) for item in caught),
                [str(item.message) for item in caught],
            )

    def test_active_preset_uses_exact_current_paths_to_migrate_legacy_names(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            active = vae / "shared.safetensors"
            active.touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            self.opts.data["forge_additional_modules"] = [str(active.resolve())]
            self.opts.data["forge_additional_modules_sd"] = ["shared.safetensors"]

            self.main_entry._migrate_legacy_module_options_once()

            self.assertEqual(
                self.opts.data["forge_additional_modules_sd"],
                [str(active.resolve())],
            )
            self.assertEqual(self.opts.save_calls, ["config.json"])

    def test_failed_legacy_migration_save_rolls_back_without_breaking_startup(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            (vae / "unique.safetensors").touch()
            self.build_registry(vae, encoder)
            self.opts.data["forge_additional_modules_krea"] = ["unique.safetensors"]

            with (
                mock.patch.object(
                    self.opts,
                    "save",
                    side_effect=OSError("read-only config"),
                ),
                mock.patch.object(self.main_entry.logger, "exception") as reported,
            ):
                self.main_entry._migrate_legacy_module_options_once()

            self.assertEqual(
                self.opts.data["forge_additional_modules_krea"],
                ["unique.safetensors"],
            )
            reported.assert_called_once()

    def test_duplicate_infotext_and_hires_references_round_trip_exactly(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            vae_module = vae / "shared.safetensors"
            encoder_module = encoder / "shared.safetensors"
            vae_module.touch()
            encoder_module.touch()
            self.build_registry(vae, encoder)

            vae_ref = self.main_entry.module_infotext_reference(str(vae_module))
            encoder_ref = self.main_entry.module_infotext_reference(str(encoder_module))
            self.assertNotEqual(vae_ref, encoder_ref)
            self.assertIsNone(self.main_entry.module_selector_from_infotext("shared"))

            resolve_parameters = load_infotext_module_helper(self.main_entry, self.opts)
            parameters = {
                "Module 1": vae_ref,
                "Hires Module 1": encoder_ref,
            }
            resolve_parameters(parameters)

            self.assertEqual(
                self.main_entry.resolve_module_values(parameters["VAE/TE"]),
                [str(vae_module.resolve())],
            )
            self.assertEqual(
                self.main_entry.resolve_module_values(parameters["Hires VAE/TE"]),
                [str(encoder_module.resolve())],
            )

    def test_infotext_distinguishes_different_extensions_with_the_same_stem(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            bin_module = vae / "encoder.bin"
            safetensors_module = encoder / "encoder.safetensors"
            bin_module.touch()
            safetensors_module.touch()
            self.build_registry(vae, encoder)

            bin_reference = self.main_entry.module_infotext_reference(str(bin_module))
            safetensors_reference = self.main_entry.module_infotext_reference(str(safetensors_module))

            self.assertEqual(bin_reference, "encoder.bin")
            self.assertEqual(safetensors_reference, "encoder.safetensors")
            self.assertEqual(
                self.main_entry.module_selector_from_infotext(bin_reference),
                "encoder.bin",
            )
            self.assertEqual(
                self.main_entry.module_selector_from_infotext(safetensors_reference),
                "encoder.safetensors",
            )
            self.assertIsNone(self.main_entry.module_selector_from_infotext("encoder"))

    def test_hires_built_in_marker_survives_parameter_consumption(self):
        resolve_parameters = load_infotext_module_helper(self.main_entry, self.opts)
        parameters = {"Hires Module 1": "Built-in"}

        resolve_parameters(parameters)

        self.assertEqual(parameters["Hires VAE/TE"], [])
        self.assertNotIn("Hires Module 1", parameters)

    def test_processing_writes_safe_module_references_for_both_passes(self):
        source = PROCESSING.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("main_entry.module_infotext_reference(m)"),
            2,
        )

    def test_downstream_selectors_preserve_module_kind_and_metadata_paths(self):
        pid_source = PID_SCRIPT.read_text(encoding="utf-8")
        metadata_source = CHECKPOINT_METADATA.read_text(encoding="utf-8")

        self.assertIn('module_kind(m) == "text_encoder"', pid_source)
        self.assertIn("distilled_cfg_scale=1.5", pid_source)
        self.assertIn("main_entry.resolve_module_values(vae)", metadata_source)
        self.assertIn("main_entry.module_values_to_ui_selectors(vae)", metadata_source)
        self.assertIn("main_entry.module_dropdown_choices()", metadata_source)

    def test_checkpoint_metadata_saves_the_exact_selected_module_path(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            vae = root / "vae"
            encoder = root / "encoder"
            vae.mkdir()
            encoder.mkdir()
            vae_module = vae / "shared.safetensors"
            vae_module.touch()
            (encoder / "shared.safetensors").touch()
            self.build_registry(vae, encoder)
            selector = self.main_entry.module_values_to_ui_selectors([str(vae_module)])[0]
            metadata = {}

            class Editor:
                def get_user_metadata(self, _name):
                    return metadata

                def write_user_metadata(self, _name, _metadata):
                    self.written = _metadata

            editor = Editor()
            save_metadata = load_checkpoint_metadata_save(self.main_entry)
            save_metadata(editor, "checkpoint", "", "", [selector], "Unknown")

            self.assertEqual(
                editor.written["vae_te"],
                [str(vae_module.resolve())],
            )


if __name__ == "__main__":
    unittest.main()
