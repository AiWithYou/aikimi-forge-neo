"""Local quantization destinations and provenance with tiny CPU-only fixtures."""

import io
import json
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors import safe_open

from modules_forge import local_assets
from modules_forge.iris import w4a8
from tools import quantize_iris
from tools.tests.test_iris_local_source import checkpoint


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)
        self.modulation_cores = torch.nn.Sequential(torch.nn.Linear(64, 64))


def contents(directory):
    return {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}


class IrisLocalQuantizeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name)
        self.root = self.project / "models/Iris-3B"
        self.weights = checkpoint(self.project / "model/日本語 学習", "水彩 300.safetensors")
        self.config = self.weights.with_name("config.yaml")
        self.patch = patch.object(local_assets, "ROOT", self.project)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        cache = patch.object(local_assets, "HASH_CACHE", self.project / "hash-cache.json")
        cache.start()
        self.addCleanup(cache.stop)

    def test_custom_export_uses_exact_selected_file_and_explicit_destination(self):
        destination = self.project / "model/圧縮 重み/int8"
        before = contents(self.weights.parent)
        model = Tiny()
        with (
            patch.object(quantize_iris, "load_model", return_value=(model, None, {})) as load,
            redirect_stdout(io.StringIO()),
        ):
            quantize_iris.export(
                self.root, "generate", "int8", str(self.weights.parent), output_dir="model/圧縮 重み/int8"
            )
        load.assert_called_once_with(self.root, "normal", "generate", str(self.weights))
        self.assertEqual(contents(self.weights.parent), before)
        self.assertEqual((destination / "config.yaml").read_bytes(), self.config.read_bytes())
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_path"], str(self.weights))
        self.assertEqual(manifest["source_config_path"], str(self.config))
        self.assertEqual(manifest["source_sha256"], quantize_iris.sha256(self.weights))
        self.assertEqual(manifest["source_config_sha256"], quantize_iris.sha256(self.config))
        self.assertNotIn("source_repo", manifest)
        self.assertNotIn("source_revision", manifest)
        self.assertEqual({record["path"] for record in manifest["files"]}, {"model.safetensors", "config.yaml"})
        with safe_open(destination / "model.safetensors", framework="pt") as source:
            metadata = source.metadata()
            self.assertEqual(metadata["format"], "iris-rowwise-int8-v1")
            for key in ("source_path", "source_config_path", "source_sha256", "source_config_sha256"):
                self.assertEqual(metadata[key], manifest[key])
            self.assertNotIn("source_repo", metadata)
            self.assertNotIn("source_revision", metadata)
            self.assertIn("linear.qweight", source.keys())
            self.assertIn("modulation_cores.0.weight", source.keys())

    def test_custom_w4a8_uses_source_size_without_requiring_sibling_int8(self):
        destination = self.project / "outside/w4a8"
        before = contents(self.weights.parent)
        self.assertFalse((self.root / "int8/model.safetensors").exists())
        with (
            patch.object(quantize_iris, "load_model", return_value=(Tiny(), None, {})),
            patch.object(
                w4a8, "convert_linears", return_value={"format": w4a8.FORMAT, "linear_count": 1, "bytes": 1}
            ) as convert,
            redirect_stdout(io.StringIO()),
        ):
            quantize_iris.export(self.root, "generate", "w4a8", str(self.weights), output_dir=destination)
        convert.assert_called_once()
        self.assertEqual(contents(self.weights.parent), before)
        self.assertEqual(
            json.loads((destination / "manifest.json").read_text(encoding="utf-8"))["source_sha256"],
            quantize_iris.sha256(self.weights),
        )
        with safe_open(destination / "model.safetensors", framework="pt") as source:
            self.assertEqual(source.metadata()["format"], w4a8.FORMAT)

    def test_custom_export_requires_a_distinct_new_output_before_loading(self):
        existing = self.project / "existing"
        existing.mkdir()
        (existing / "model.safetensors").write_bytes(b"keep existing output")
        for output in (None, self.weights.parent, self.weights.parent / "child/..", existing):
            with self.subTest(output=output), patch.object(quantize_iris, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "output-dir|出力|上書き"):
                    quantize_iris.export(self.root, "generate", "int8", str(self.weights), output_dir=output)
                load.assert_not_called()
        self.assertEqual((existing / "model.safetensors").read_bytes(), b"keep existing output")

    def test_remote_or_file_output_and_non_generation_custom_are_rejected(self):
        for output in ("https://example.test/output", self.weights):
            with self.subTest(output=output), patch.object(quantize_iris, "load_model") as load:
                with self.assertRaisesRegex(ValueError, "出力|フォルダー|このPC"):
                    quantize_iris.export(self.root, "generate", "int8", str(self.weights), output_dir=output)
                load.assert_not_called()
        with patch.object(quantize_iris, "load_model") as load:
            with self.assertRaisesRegex(ValueError, "画像生成|generate"):
                quantize_iris.export(self.root, "depth", "int8", str(self.weights), output_dir=self.project / "output")
            load.assert_not_called()

    def test_source_change_during_conversion_is_rejected_before_writing_output(self):
        destination = self.project / "output"

        def change_source(_model):
            self.config.write_text("model: {hidden_size: 3}\n", encoding="utf-8")
            return {"format": "iris-rowwise-int8-v1", "linear_count": 1, "bytes": 1}

        with (
            patch.object(quantize_iris, "load_model", return_value=(Tiny(), None, {})),
            patch.object(quantize_iris, "convert_linears", side_effect=change_source),
        ):
            with self.assertRaisesRegex(ValueError, "変更|変換元"):
                quantize_iris.export(self.root, "generate", "int8", str(self.weights), output_dir=destination)
        self.assertFalse((destination / "model.safetensors").exists())

    def test_base_export_retains_managed_destinations_and_official_provenance(self):
        source = checkpoint(self.root / "official")
        destination = self.root / "int8"
        with (
            patch.object(quantize_iris, "load_model", return_value=(Tiny(), None, {})) as load,
            redirect_stdout(io.StringIO()),
        ):
            quantize_iris.export(self.root, "generate")
        load.assert_called_once_with(self.root, "normal", "generate", "base")
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["source_repo"], quantize_iris.SOURCE_REPO)
        self.assertEqual(manifest["source_revision"], quantize_iris.SOURCE_REVISION)
        self.assertEqual(manifest["source_sha256"], quantize_iris.sha256(source))

    def test_cli_forwards_actual_checkpoint_and_output_directory(self):
        destination = self.project / "output"
        with (
            patch.object(
                sys,
                "argv",
                [
                    "quantize_iris.py",
                    "--task",
                    "generate",
                    "--root",
                    str(self.root),
                    "--checkpoint",
                    str(self.weights),
                    "--output-dir",
                    str(destination),
                ],
            ),
            patch.object(quantize_iris, "export") as export,
            patch.object(quantize_iris, "export_text") as text,
        ):
            quantize_iris.main()
        export.assert_called_once_with(self.root, "generate", "int8", str(self.weights), output_dir=destination)
        text.assert_not_called()

    def test_cli_rejects_custom_without_output_and_custom_non_generation(self):
        for extra in (["--task", "generate"], ["--task", "depth", "--output-dir", str(self.project / "output")]):
            with (
                self.subTest(extra=extra),
                patch.object(sys, "argv", ["quantize_iris.py", "--checkpoint", str(self.weights), *extra]),
                patch.object(quantize_iris, "export") as export,
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as error,
            ):
                quantize_iris.main()
            self.assertEqual(error.exception.code, 2)
            export.assert_not_called()


if __name__ == "__main__":
    unittest.main()
