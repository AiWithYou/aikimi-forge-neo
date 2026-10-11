"""Local full fine-tunes stay separate from the original Iris checkpoint."""

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from modules_forge.iris import core, runtime
from modules_forge.iris.quantization import convert_linears

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = "illustration-ft300"


def ready_root(root):
    python = core.python_path(root)
    python.parent.mkdir(parents=True)
    python.touch()
    core.atomic_json(root / "runtime.json", {"code_revision": core.CODE_REVISION, "packaging": core.PACKAGING_REVISION})
    encoder = root / "text-encoder"
    encoder.mkdir()
    (encoder / "config.json").write_text("{}", encoding="utf-8")
    (encoder / "model.safetensors").write_bytes(b"text")
    core.atomic_json(
        encoder / "download.json",
        {"revision": core.TEXT_REVISION, "files": [{"path": "model.safetensors", "size": 4}]},
    )


def ready_model(directory):
    directory.mkdir(parents=True)
    weights = {
        key: torch.ones(2, 2)
        for key in ("s_embedder.proj.weight", "t_embedder.mlp.0.weight", "final_layer.linear.weight", "blocks.0.weight")
    }
    if directory.name == "int8":
        for key in list(weights):
            if key == "final_layer.linear.weight":
                continue
            weights[key.removesuffix(".weight") + ".qweight"] = weights.pop(key).to(torch.int8)
            weights[key.removesuffix(".weight") + ".scale"] = torch.ones(2, 1)
    save_file(weights, directory / "model.safetensors")
    (directory / "config.yaml").write_text("model:\n  hidden_size: 2\n", encoding="utf-8")
    core.atomic_json(
        directory / "manifest.json",
        {
            "files": [
                {"path": name, "size": (directory / name).stat().st_size}
                for name in ("model.safetensors", "config.yaml")
            ]
        },
    )


class CheckpointTests(unittest.TestCase):
    def test_selected_model_uses_its_own_directory(self):
        root = Path("models/Iris-3B")
        self.assertEqual(core.model_directory(root, "normal", "generate"), root / "official")
        with tempfile.TemporaryDirectory() as folder:
            selected = Path(folder) / "int8"
            ready_model(selected)
            self.assertEqual(core.model_directory(root, "int8", "generate", str(selected)), selected)

    def test_local_checkpoint_paths_are_allowed_but_urls_and_empty_values_are_not(self):
        for value in ("", " ", "https://example.test/model.safetensors", None):
            with self.subTest(value=value), self.assertRaises(core.IrisError):
                core.validate_request({"prompt": "a fox", "checkpoint": value})
        request = core.validate_request({"prompt": "a fox", "checkpoint": CHECKPOINT})
        self.assertEqual(request["checkpoint"], CHECKPOINT)
        self.assertEqual(
            core.validate_request({"prompt": "a fox", "checkpoint": "models/日本語 FT/model.safetensors"})[
                "checkpoint"
            ],
            "models/日本語 FT/model.safetensors",
        )

    def test_local_models_share_the_encoder_and_incomplete_models_are_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ready_root(root)
            ready_model(root / "official")
            selected = root / "checkpoints" / CHECKPOINT
            ready_model(selected / "int8")
            core.atomic_json(selected / "checkpoint.json", {"label": "学習済みイラスト"})
            partial = root / "checkpoints/partial/official"
            ready_model(partial)
            (partial / "model.safetensors").write_bytes(b"truncated")
            self.assertTrue(core.model_ready(root, "normal", "generate"))
            self.assertTrue(core.model_ready(root, "int8", "generate", str(selected / "int8/model.safetensors")))
            self.assertFalse(core.model_ready(root, "normal", "generate", str(selected / "int8/model.safetensors")))
            self.assertEqual(
                [value for _, value in core.checkpoint_choices(root)],
                ["base", str((selected / "int8/model.safetensors").resolve())],
            )
            self.assertTrue((root / "official/model.safetensors").is_file())

    def test_downstream_tasks_discard_the_generation_checkpoint(self):
        request = core.validate_request({"task": "depth", "image": "input.png", "checkpoint": CHECKPOINT})
        self.assertNotIn("checkpoint", request)

    def test_a_different_checkpoint_replaces_the_resident_runner(self):
        source = ROOT / "tools/iris_worker.py"
        spec = importlib.util.spec_from_file_location("test_iris_checkpoint_worker", source)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        events = []

        class FakeRunner:
            def __init__(self, root, precision, task, checkpoint="base"):
                self.checkpoint = checkpoint
                events.append(("load", checkpoint))

            def close(self):
                events.append(("close", self.checkpoint))

            def set_loras(self, loras):
                self.loras = loras

            def run(self, directory, request=None):
                return self.checkpoint

        with tempfile.TemporaryDirectory() as temporary, patch.object(runtime, "Runner", FakeRunner):
            root = Path(temporary)
            ready_model(root / "int8")
            ready_model(root / "checkpoints" / CHECKPOINT / "int8")
            selected = str(root / "checkpoints" / CHECKPOINT / "int8/model.safetensors")
            directory = root / "job"
            directory.mkdir()
            payload = {"root": str(root), "job_dir": str(directory)}
            for reference in ("base", selected, selected):
                core.atomic_json(
                    directory / "request.json", {"task": "generate", "precision": "int8", "checkpoint": reference}
                )
                self.assertEqual(worker.resident_run(payload), reference)
        self.assertEqual(events, [("load", "base"), ("close", "base"), ("load", selected)])


class TinyDiT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = torch.nn.Linear(64, 64)
        self.s_embedder = torch.nn.Module()
        self.s_embedder.proj = torch.nn.Linear(64, 64)
        self.t_embedder = torch.nn.Module()
        self.t_embedder.mlp = torch.nn.Sequential(torch.nn.Linear(64, 64))
        self.final_layer = torch.nn.Module()
        self.final_layer.linear = torch.nn.Linear(2, 2)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(64, 64)])


class CheckpointDtypeTests(unittest.TestCase):
    def test_bf16_and_int8_reload_keep_the_runtime_dtype_contract(self):
        config_module = types.ModuleType("iris3b.config")
        config_module.inference_config = lambda raw, options: SimpleNamespace(model=SimpleNamespace(hidden_size=2))
        dit_module = types.ModuleType("iris3b.models.dit")
        dit_module.IrisDiT = TinyDiT
        source = TinyDiT(None).bfloat16()
        normal_state = {key: value.clone() for key, value in source.state_dict().items()}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normal = root / "checkpoints" / CHECKPOINT / "official"
            normal.mkdir(parents=True)
            (normal / "config.yaml").write_text("model:\n  hidden_size: 2\n", encoding="utf-8")
            save_file(normal_state, normal / "model.safetensors")
            with patch.dict(sys.modules, {"iris3b.config": config_module, "iris3b.models.dit": dit_module}):
                loaded, _, _ = runtime.load_model(root, "normal", "generate", str(normal / "model.safetensors"))
                for key, value in loaded.state_dict().items():
                    self.assertEqual(value.dtype, torch.float32)
                    self.assertTrue(torch.equal(value, normal_state[key].float()))
                convert_linears(source.float())
                int8_state = {key: value.clone() for key, value in source.state_dict().items()}
                quantized = root / "checkpoints" / CHECKPOINT / "int8"
                quantized.mkdir()
                (quantized / "config.yaml").write_text("model:\n  hidden_size: 2\n", encoding="utf-8")
                save_file(int8_state, quantized / "model.safetensors")
                loaded, _, _ = runtime.load_model(root, "int8", "generate", str(quantized / "model.safetensors"))
                self.assertEqual(loaded.layer.qweight.dtype, torch.int8)
                self.assertEqual(loaded.layer.scale.dtype, torch.float32)
                for key, value in loaded.state_dict().items():
                    self.assertTrue(torch.equal(value, int8_state[key]))


class CheckpointUITests(unittest.TestCase):
    def test_the_screen_passes_the_selected_checkpoint_and_never_downloads_base_instead(self):
        spec = importlib.util.spec_from_file_location(
            "test_iris_checkpoint_screen", ROOT / "extensions-builtin/iris-studio/scripts/iris_studio.py"
        )
        screen = importlib.util.module_from_spec(spec)
        callbacks = types.ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = lambda callback: None
        with patch.dict(sys.modules, {"modules.script_callbacks": callbacks}):
            spec.loader.exec_module(screen)
        blocks, _, _ = screen.on_ui_tabs()[0]
        choices = [
            component
            for component in blocks.get_config_file()["components"]
            if component.get("props", {}).get("elem_id") == "iris-checkpoint"
        ]
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["props"]["label"], "チェックポイント")
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / "int8"
            ready_model(directory)
            selected = str(directory / "model.safetensors")
            with (
                patch.object(screen, "ready", return_value=True),
                patch.object(screen.SERVICE, "start", return_value="ft-job") as start,
            ):
                screen.start({}, "generate", "int8", "a fox", None, 42, "1024×1024", 50, 3, "", selected)
            self.assertEqual(start.call_args.args[0]["checkpoint"], selected)
            (directory / "model.safetensors").unlink()
            with (
                patch.object(screen, "ready", return_value=False),
                patch.object(screen.PREPARATION, "start") as download,
            ):
                response = screen.start({}, "generate", "w4a8", "a fox", None, 42, "1024×1024", 50, 3, "", selected)
            download.assert_not_called()
            self.assertIn("チェックポイント", response[4])


if __name__ == "__main__":
    unittest.main()
