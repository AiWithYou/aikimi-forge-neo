"""Selected checkpoint contracts for the video and music Studio adapters."""

import json
import os
import struct
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from modules_forge import local_assets, studio_checkpoints
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.nanosaur2_studio import Nanosaur2Request, build_workflow
from modules_forge.yue2_studio.core import Request, runtime_manifest


def checkpoint(path, names):
    header = {name: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for name in names}
    header["__metadata__"] = {"h3_variant": "base"}
    for name, shape in {
        "audio_patch_proj.weight": [5376, 32],
        "video_patch_proj.weight": [5376, 96],
        "adaln_t_table": [1025, 8],
        "s_embedder.proj.weight": [1536, 64],
        "final_layer.linear.weight": [64, 1536],
        "blocks.0.attn.q_norm.weight": [96],
        "blocks.0.mlp.w12.weight": [8192, 1536],
    }.items():
        if name in header:
            header[name]["shape"] = shape
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
    return str(path)


class SelectedCheckpointRoutes(unittest.TestCase):
    def test_h3_rejects_ambiguous_roles_unless_metadata_or_fused(self):
        with TemporaryDirectory() as folder:
            path = Path(
                checkpoint(
                    Path(folder) / "h3_fl2va_ft_from_ref2va.safetensors",
                    ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
                )
            )
            with self.assertRaisesRegex(ValueError, "役割"):
                studio_checkpoints.h3_role(path)
            self.assertFalse(studio_checkpoints.h3_compatible(path))
            header = local_assets.read_header(path)
            header["__metadata__"]["h3_role"] = "fl2va"
            with patch.object(local_assets, "read_header", return_value=header):
                self.assertEqual(studio_checkpoints.h3_role(path), "fl2va")
            fused = path.with_name("h3_fl2va_ref2va_fused_turbo.safetensors")
            path.rename(fused)
            self.assertEqual(studio_checkpoints.h3_role(fused), "both")

    def test_compatibility_scans_reject_non_mapping_tensor_entries(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "custom_fl2va.safetensors"
            path.write_bytes(b"exists")
            for validator, names in (
                (
                    studio_checkpoints.h3_compatible,
                    ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
                ),
                (
                    studio_checkpoints.nanosaur_compatible,
                    [
                        "s_embedder.proj.weight",
                        "final_layer.linear.weight",
                        "blocks.0.attn.q_norm.weight",
                        "blocks.0.mlp.w12.weight",
                    ],
                ),
            ):
                with (
                    self.subTest(validator=validator.__name__),
                    patch.object(local_assets, "read_header", return_value=dict.fromkeys(names, [])),
                ):
                    self.assertFalse(validator(path))

    def test_nanosaur_standard_metadata_uses_diffusion_manifest_entry_by_path(self):
        from types import SimpleNamespace

        from PIL import Image

        from modules_forge import nanosaur2_studio as nano

        with TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = json.loads(nano.MANIFEST.read_text(encoding="utf-8"))
            expected = next(
                entry["sha256"]
                for entry in manifest["models"]
                if entry["path"] == "diffusion_models/nanosaur2_diffusion_model.safetensors"
            )
            manifest["models"].reverse()
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            source = root / "image.png"
            Image.new("RGB", (256, 256)).save(source)
            with patch.object(nano, "MANIFEST", manifest_path):
                result = nano.save_result(
                    source,
                    Nanosaur2Request("test", width=256, height=256),
                    42,
                    "job",
                    SimpleNamespace(core_revision="test", comfy_version="test"),
                    root / "output",
                )
            self.assertEqual(result["metadata"]["checkpoint"]["sha256"], expected)

    def test_new_browser_session_restores_latest_library_without_writing(self):
        import gradio as gr

        from modules_forge import local_assets, model_catalog, studio_checkpoint_ui

        with TemporaryDirectory() as folder:
            root = Path(folder)
            with (
                patch.object(local_assets, "LIBRARY", root / "library.json"),
                patch.object(model_catalog, "ROOT", root),
            ):
                with gr.Blocks() as app:
                    checkpoint_ui, _ = studio_checkpoint_ui.create(
                        "nanosaur2", "Nanosaur2", (), lambda _: False, gradio=gr
                    )
                selected = str(root / "missing.safetensors")
                local_assets.save_selection("nanosaur2", {"checkpoint": selected})
                before = local_assets.LIBRARY.read_bytes()
                restore = [fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "restore"]
                self.assertEqual(len(restore), 1)
                self.assertEqual(restore[0].outputs, [checkpoint_ui])
                with patch.object(local_assets, "save_selection", side_effect=AssertionError("restore must only read")):
                    values = restore[0].fn()
                self.assertEqual(values["value"], selected)
                self.assertEqual(local_assets.LIBRARY.read_bytes(), before)
                app.close()

    def test_h3_and_nanosaur_rescan_only_managed_nested_tree_and_keep_current(self):
        from modules_forge import model_catalog, studio_checkpoint_ui

        with TemporaryDirectory() as folder:
            root = Path(folder)
            with (
                patch.object(model_catalog, "ROOT", root),
                patch.object(local_assets, "LIBRARY", root / "library.json"),
            ):
                for family, validator, names in (
                    (
                        "MiniMax-H3",
                        studio_checkpoints.h3_compatible,
                        ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
                    ),
                    (
                        "Nanosaur2",
                        studio_checkpoints.nanosaur_compatible,
                        [
                            "s_embedder.proj.weight",
                            "final_layer.linear.weight",
                            "blocks.0.attn.q_norm.weight",
                            "blocks.0.mlp.w12.weight",
                        ],
                    ),
                ):
                    with self.subTest(family=family):
                        managed = root / "models" / family / "checkpoints" / "finetunes"
                        managed.mkdir(parents=True)
                        first = checkpoint(managed / "one_fl2va.safetensors", names)
                        old = root / "model" / family
                        old.mkdir(parents=True)
                        outside = checkpoint(old / "outside_fl2va.safetensors", names)
                        local_assets.remember(family, [outside])
                        initial = studio_checkpoint_ui.inventory(family, (), validator)
                        self.assertEqual([value for _, value in initial], ["", str(Path(first).resolve())])
                        second = checkpoint(managed / "two_fl2va.safetensors", names)
                        with patch.object(
                            local_assets, "file_identity", side_effect=AssertionError("rescan must not hash")
                        ):
                            values = studio_checkpoint_ui.inventory(family, (), validator, current=outside)
                        self.assertEqual({value for _, value in values}, {"", first, second, outside})

    def test_yue2_engine_refresh_binds_only_engine_and_selected_package(self):
        import importlib.util
        from types import ModuleType, SimpleNamespace

        from modules_forge import model_catalog

        with TemporaryDirectory() as folder:
            root = Path(folder)
            modules = ModuleType("modules")
            modules.script_callbacks = SimpleNamespace(on_ui_tabs=lambda callback: None)
            paths = ModuleType("modules.paths")
            paths.data_path = paths.script_path = str(root)
            repository = Path(__file__).resolve().parents[2]
            spec = importlib.util.spec_from_file_location(
                "test_yue2_checkpoint_ui", repository / "extensions-builtin/yue2-studio/scripts/yue2_studio.py"
            )
            ui = importlib.util.module_from_spec(spec)
            with patch.dict("sys.modules", {"modules": modules, "modules.paths": paths}):
                spec.loader.exec_module(ui)
            self.addCleanup(ui.STUDIO.shutdown)
            with (
                patch.object(model_catalog, "ROOT", root),
                patch.object(local_assets, "LIBRARY", root / "library.json"),
            ):
                app = ui.on_ui_tabs()[0][0]
                self.addCleanup(app.close)
                refresh = next(fn for fn in app.fns.values() if fn.fn and fn.fn.__name__ == "refresh_checkpoint_engine")
                self.assertEqual([component.label for component in refresh.inputs], ["エンジン", "チェックポイント"])
                self.assertEqual(refresh.outputs, [refresh.inputs[1]])
                package = root / "models/YuE2/checkpoints/native/selected"
                package.mkdir(parents=True)
                (package / "config.json").write_text('{"model_type":"yue2"}')
                for name in ("model.safetensors", "qwen.tiktoken", "weights_manifest.json"):
                    (package / name).write_text("{}")
                selected = str(package.resolve())
                values = refresh.fn("official", selected)
                self.assertIn(selected, [value for _, value in values["choices"]])
                self.assertEqual(values["value"], selected)
                values = refresh.fn("cpp", selected)
                self.assertEqual(values["value"], selected)
                self.assertFalse(
                    any(
                        component.label == "探索フォルダー"
                        for component in app.blocks.values()
                        if hasattr(component, "label")
                    )
                )

    def test_renamed_h3_w4a8_keeps_runtime_precision_requirement(self):
        with TemporaryDirectory() as folder:
            path = Path(
                checkpoint(
                    Path(folder) / "custom_fl2va.safetensors",
                    ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
                )
            )
            header = local_assets.read_header(path)
            header["__metadata__"] = {
                "_quantization_metadata": json.dumps(
                    {"layers": {"blocks.0.attn.out_proj": {"format": "asym_w4a8_int8"}}}
                )
            }
            encoded = json.dumps(header).encode()
            path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 4)
            self.assertEqual(studio_checkpoints.h3_variant(str(path)), "w4a8")

    def test_comfy_config_changes_for_equal_names_and_overwritten_file(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            first, second = root / "one", root / "two"
            first.mkdir()
            second.mkdir()
            a = Path(
                checkpoint(
                    first / "custom_fl2va.safetensors",
                    ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
                )
            )
            b = Path(
                checkpoint(second / a.name, ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"])
            )
            config_a = studio_checkpoints.write_comfy_config(root, str(a))
            config_b = studio_checkpoints.write_comfy_config(root, str(b))
            self.assertNotEqual(config_a, config_b)
            for config, selected in ((config_a, a), (config_b, b)):
                entry = json.loads(config.read_text())["neo_checkpoint"]
                self.assertTrue(entry["is_default"])
                resolved = Path(entry["base_path"]) / entry["diffusion_models"] / selected.name
                self.assertEqual(resolved.resolve(), selected.resolve())
                self.assertEqual(local_assets.file_identity(resolved), local_assets.file_identity(selected))
            old = studio_checkpoints.config_path(root, str(a))
            stamp = a.stat().st_mtime_ns
            os.utime(a, ns=(stamp + 2_000_000_000, stamp + 2_000_000_000))
            self.assertNotEqual(old, studio_checkpoints.config_path(root, str(a)))

    def test_external_comfy_is_not_restarted_for_unregistered_selection(self):
        from types import SimpleNamespace

        from modules_forge import minimax_h3_bridge as bridge

        with TemporaryDirectory() as folder:
            root = Path(folder)
            selected = checkpoint(
                root / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            readiness = SimpleNamespace(runtime_args=())
            with (
                patch.object(bridge, "_MANAGED_PROCESS", None),
                patch.object(bridge, "_restart_runtime_locked") as restart,
            ):
                with self.assertRaisesRegex(bridge.H3BridgeError, "外部ComfyUI"):
                    bridge.ensure_checkpoint_registration(
                        readiness, root, "http://127.0.0.1:8189", root, H3Acceleration(checkpoint=selected)
                    )
                restart.assert_not_called()

    def test_h3_rejects_unknown_role_and_wrong_structure(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            unknown = checkpoint(
                root / "custom.safetensors", ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"]
            )
            wrong = checkpoint(root / "other_fl2va.safetensors", ["other.weight"])
            self.assertFalse(studio_checkpoints.h3_compatible(Path(unknown)))
            self.assertFalse(studio_checkpoints.h3_compatible(Path(wrong)))

    def test_h3_acceleration_uses_selected_file_after_profile(self):
        with TemporaryDirectory() as folder:
            path = checkpoint(
                Path(folder) / "custom_fl2va.safetensors",
                ["audio_patch_proj.weight", "video_patch_proj.weight", "adaln_t_table"],
            )
            option = H3Acceleration(model_variant="w4a8", checkpoint=path)
            self.assertEqual(option.model_files({})["FL2VA"], ("diffusion_models", Path(path).name))
            with self.assertRaisesRegex(ValueError, "Ref2VA"):
                option.checkpoint_for_mode("references")

    def test_nanosaur_workflow_keeps_selected_checkpoint(self):
        with TemporaryDirectory() as folder:
            path = checkpoint(
                Path(folder) / "my_nanosaur.safetensors",
                [
                    "s_embedder.proj.weight",
                    "final_layer.linear.weight",
                    "blocks.0.attn.q_norm.weight",
                    "blocks.0.mlp.w12.weight",
                ],
            )
            graph = build_workflow(Nanosaur2Request(prompt="forest", checkpoint=path), 42)
            self.assertEqual(graph["loader"]["inputs"]["unet_name"], "my_nanosaur.safetensors")

    def test_yue2_runtime_selection_reaches_worker_manifest(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            models = root / "selected"
            models.mkdir()
            (models / "config.json").write_text('{"model_type":"yue2"}')
            (models / "model.safetensors").write_bytes(b"weights")
            (models / "qwen.tiktoken").write_bytes(b"tokenizer")
            (models / "weights_manifest.json").write_text("{}")
            (root / "runtime.json").write_text(
                json.dumps({"schema": 1, "official": {"model": "old", "vae": "shared", "python": "python"}})
            )
            selected = Request(style="piano", lyrics="lyrics", checkpoint=str(models))
            with patch("modules_forge.local_assets.identity", return_value={"sha256": "selected"}):
                entry = runtime_manifest(root, selected.engine, selected.checkpoint)
            self.assertEqual(entry["model"], str(models.resolve()))
            self.assertEqual(entry["vae"], "shared")
            self.assertEqual(entry["checkpoint_identity"], {"sha256": "selected"})


if __name__ == "__main__":
    unittest.main()
