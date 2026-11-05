"""Irodori full checkpoint selection and quantization metadata contracts."""

import copy
import hashlib
import importlib
import json
import sys
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

CONFIG = {
    "flow_parameterization": "rf_velocity", "latent_dim": 32, "latent_patch_size": 1,
    "model_dim": 2048, "num_layers": 24, "num_heads": 32,
    "text_vocab_size": 262144, "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "text_encoder_revision": "dd0a2683227859151b1730ca3a63087df5b5f39b", "text_add_bos": True,
    "text_encoder_type": "pretrained", "pretrained_projector_type": "residual_mlp", "text_dim": 1280,
    "use_caption_condition": True, "use_speaker_condition": True,
    "caption_vocab_size": 262144, "caption_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "caption_add_bos": True, "caption_dim": 1280, "speaker_dim": 1280, "speaker_layers": 14,
    "speaker_heads": 20, "speaker_patch_size": 4, "use_duration_predictor": True,
}
ENCODER = {"model_type": "t5gemma2", "vocab_size": 262144, "bos_token_id": 2,
           "eos_token_id": 1, "pad_token_id": 0,
           "encoder": {"text_config": {"model_type": "t5gemma2_text", "vocab_size": 262144,
                                       "hidden_size": 1152, "num_hidden_layers": 26}}}
WEIGHT = "blocks.0.attention.wq.weight"
BIAS = "test.bias"
SHAPES = [(WEIGHT, [16, 256]), (BIAS, [16])]
SCHEMA = hashlib.sha256(json.dumps(sorted(SHAPES), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def header(quant=None):
    meta = {"config_json": json.dumps(CONFIG), "text_encoder_config_json": json.dumps(ENCODER)}
    result = {"__metadata__": meta, BIAS: {"dtype": "BF16", "shape": [16]}}
    if quant is None:
        result[WEIGHT] = {"dtype": "F32", "shape": [16, 256]}
        return result
    names = [BIAS, WEIGHT]
    meta["tensor_names"] = json.dumps(names)
    meta[BIAS] = json.dumps({"_type": "Tensor"})
    dynamic = quant in ("int8_dynamic_activation_int8_weight", "float8_dynamic_activation_float8_weight")
    if quant == "int4_weight_only":
        typ, data, parts = "Int4TilePackedTo4dTensor", {"shape": [16, 256], "block_size": [1, 128]}, ["qdata", "scale_and_zero"]
        result["blocks.0.attention.wq._weight_qdata"] = {"dtype": "I32", "shape": [2, 8, 32, 4]}
        result["blocks.0.attention.wq._weight_scale_and_zero"] = {"dtype": "BF16", "shape": [8, 16, 2]}
    else:
        typ = "Int8Tensor" if quant.startswith("int8") else "Float8Tensor"
        block = [16, 256] if quant == "float8_dynamic_activation_float8_weight" else [1, 256]
        data, parts = {
            "block_size": block, "act_quant_kwargs": {"_type": "QuantizeTensorToInt8Kwargs"} if dynamic else None,
            "dtype": {"_type": "torch.dtype", "_data": "bfloat16"},
        }, ["qdata", "scale"]
        if quant == "float8_dynamic_activation_float8_weight":
            data["act_quant_kwargs"] = {"_type": "QuantizeTensorToFloat8Kwargs"}
        dtype = "I8" if typ == "Int8Tensor" else "F8_E4M3"
        result["blocks.0.attention.wq._weight_qdata"] = {"dtype": dtype, "shape": [16, 256]}
        scale_shape = [1, 1] if quant == "float8_dynamic_activation_float8_weight" else [16, 1]
        result["blocks.0.attention.wq._weight_scale"] = {"dtype": "BF16" if typ == "Int8Tensor" else "F32", "shape": scale_shape}
    meta[WEIGHT] = json.dumps({"_type": typ, "_data": data, "_tensor_data_names": parts})
    q = {"format_version": 1, "backend": "torchao", "backend_version": "0.16.0", "quantization_type": quant,
         "compute_dtype": "bfloat16", "profile": "core", "quantized_modules": 1}
    if quant == "int4_weight_only":
        q.update(group_size=128, packing_format="tile_packed_to_4d")
    meta["irodori_quantization_json"] = json.dumps(q)
    return result


class CheckpointTests(unittest.TestCase):
    def test_status_accepts_custom_only_without_claiming_unready_models(self):
        from modules import aikimi_capabilities
        from modules.aikimi_diagnostics import DiagnosticPaths

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            paths = DiagnosticPaths(root=root, models_root=root / "models", output_root=root / "output")
            choices = [("公式", "official"), ("Voice", "Voice.safetensors")]
            with (
                patch("modules_forge.irodori.checkpoints.inventory", return_value=choices),
                patch("modules_forge.irodori.core.ready", side_effect=lambda root, models, checkpoint="official":
                      checkpoint == "Voice.safetensors"),
            ):
                self.assertTrue(aikimi_capabilities.feature_check("irodori", paths).available)
            with (
                patch("modules_forge.irodori.checkpoints.inventory", return_value=choices),
                patch("modules_forge.irodori.core.ready", return_value=False),
            ):
                self.assertFalse(aikimi_capabilities.feature_check("irodori", paths).available)

    def test_model_choice_callbacks_are_wired_in_irodori_ui(self):
        import ast

        path = Path(__file__).resolve().parents[2] / "modules_forge/irodori/ui.py"
        module = ast.parse(path.read_text(encoding="utf-8"))
        functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
        self.assertTrue({"refresh_models", "select_model"} <= functions)

    def test_common_tokenizer_and_codec_are_hash_checked(self):
        from modules_forge.irodori import checkpoints, core

        with tempfile.TemporaryDirectory() as folder:
            root, models = Path(folder) / "runtime", Path(folder) / "models"
            root.mkdir()
            records = []
            for name in core.REQUIRED_FILES[1:]:
                file = models / name
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b"good")
                records.append({"file": name, "sha256": hashlib.sha256(b"good").hexdigest()})
            (root / "model-files.json").write_text(json.dumps(records), encoding="utf-8")
            checkpoints.verify_components(models, runtime=root)
            (models / "v4-Large/tokenizer/tokenizer.json").write_bytes(b"evil")
            with self.assertRaisesRegex(ValueError, "tokenizer"):
                checkpoints.verify_components(models, runtime=root)

    def test_quantized_gpu_requirements_distinguish_fp8_weight_only_and_dynamic(self):
        from modules_forge.irodori import checkpoints

        for kind in ("float", "int8_weight_only", "int8_dynamic_activation_int8_weight"):
            checkpoints.ensure_gpu_compatibility(kind, (8, 6))
        checkpoints.ensure_gpu_compatibility("int4_weight_only", (8, 0))
        with self.assertRaisesRegex(ValueError, "8.0"):
            checkpoints.ensure_gpu_compatibility("int4_weight_only", (7, 5))
        for capability in ((7, 5), (8, 0), (8, 6), (8, 9)):
            with self.subTest(kind="float8_weight_only", capability=capability):
                checkpoints.ensure_gpu_compatibility("float8_weight_only", capability)
        for capability in ((8, 0), (8, 6)):
            with self.subTest(kind="float8_dynamic", capability=capability), self.assertRaisesRegex(ValueError, "8.9"):
                checkpoints.ensure_gpu_compatibility("float8_dynamic_activation_float8_weight", capability)
        checkpoints.ensure_gpu_compatibility("float8_dynamic_activation_float8_weight", (8, 9))

    def test_fp8_weight_only_still_requires_the_worker_bf16_check(self):
        from modules_forge.irodori import checkpoints
        from tools import irodori_worker

        # Keep the worker's imports outside the sys.modules patch, so later tests can reuse them.
        importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        upstream = types.ModuleType("irodori_tts.inference_runtime")
        upstream.InferenceRuntime = upstream.RuntimeKey = upstream.SamplingRequest = object
        upstream.save_wav = lambda *args: None
        package = types.ModuleType("irodori_tts")
        package.__path__ = []
        package.inference_runtime = upstream
        identity = {"name": "Voice.safetensors", "format": "float8_weight_only"}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "request.json").write_text(json.dumps({
                "text": "文", "precision": "bf16", "checkpoint": identity["name"], "checkpoint_identity": identity,
            }), encoding="utf-8")
            payload = {"root": str(directory), "job_dir": str(directory), "models": str(directory)}
            with (
                patch.dict(sys.modules, {"irodori_tts": package, "irodori_tts.inference_runtime": upstream}),
                patch.object(sys, "path", list(sys.path)),
                patch.object(checkpoints, "bind", return_value=identity),
                patch.object(checkpoints, "verify_components"),
                patch.object(checkpoints, "resolve", return_value=directory / identity["name"]),
                patch.object(checkpoints, "ensure_gpu_compatibility") as compatibility,
                patch.object(torch.cuda, "is_bf16_supported", return_value=False),
                patch.object(torch.cuda, "get_device_capability") as capability,
                self.assertRaisesRegex(ValueError, "BF16"),
            ):
                irodori_worker.resident_run(payload)
            compatibility.assert_not_called()
            capability.assert_not_called()

    def test_fp8_weight_only_reaches_runtime_on_bf16_capable_sm86(self):
        from modules_forge.irodori import checkpoints
        from tools import irodori_worker

        class ReachedRuntime(Exception):
            pass

        def reached_runtime(**kwargs):
            raise ReachedRuntime()

        importlib.import_module("numpy")
        torch = importlib.import_module("torch")
        upstream = types.ModuleType("irodori_tts.inference_runtime")
        upstream.InferenceRuntime = upstream.SamplingRequest = object
        upstream.RuntimeKey = reached_runtime
        upstream.save_wav = lambda *args: None
        package = types.ModuleType("irodori_tts")
        package.__path__ = []
        package.inference_runtime = upstream
        identity = {"name": "Voice.safetensors", "format": "float8_weight_only"}
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "request.json").write_text(json.dumps({
                "text": "文", "precision": "bf16", "checkpoint": identity["name"], "checkpoint_identity": identity,
            }), encoding="utf-8")
            payload = {"root": str(directory), "job_dir": str(directory), "models": str(directory)}
            with (
                patch.dict(sys.modules, {"irodori_tts": package, "irodori_tts.inference_runtime": upstream}),
                patch.object(sys, "path", list(sys.path)),
                patch.object(checkpoints, "bind", return_value=identity),
                patch.object(checkpoints, "verify_components"),
                patch.object(checkpoints, "resolve", return_value=directory / identity["name"]),
                patch.object(torch.cuda, "is_bf16_supported", return_value=True),
                patch.object(torch.cuda, "get_device_capability", return_value=(8, 6)) as capability,
                patch.object(irodori_worker, "local_watermark", return_value=nullcontext()),
                patch.object(irodori_worker, "local_tokenizer", return_value=nullcontext()),
                self.assertRaises(ReachedRuntime),
            ):
                irodori_worker.resident_run(payload)
            capability.assert_called_once_with()

    def test_int4_supports_upstream_padding_of_input_dimension(self):
        from modules_forge.irodori import checkpoints

        model = header("int4_weight_only")
        entry = json.loads(model["__metadata__"][WEIGHT])
        entry["_data"]["shape"] = [1024, 1152]
        model["__metadata__"][WEIGHT] = json.dumps(entry)
        model["blocks.0.attention.wq._weight_qdata"]["shape"] = [128, 16, 32, 4]
        model["blocks.0.attention.wq._weight_scale_and_zero"]["shape"] = [16, 1024, 2]
        schema = [(WEIGHT, [1024, 1152]), (BIAS, [16])]
        digest = hashlib.sha256(json.dumps(sorted(schema), separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        with patch.object(checkpoints, "V4_LARGE_SCHEMA_SHA256", digest):
            self.assertEqual(checkpoints.inspect_header(model), "int4_weight_only")

    def test_all_upstream_precisions_use_the_same_logical_model_structure(self):
        from modules_forge.irodori import checkpoints

        quantizations = (None, "int8_weight_only", "int8_dynamic_activation_int8_weight", "int4_weight_only",
                         "float8_weight_only", "float8_dynamic_activation_float8_weight")
        with patch.object(checkpoints, "V4_LARGE_SCHEMA_SHA256", SCHEMA):
            for quant in quantizations:
                with self.subTest(quant=quant):
                    self.assertEqual(checkpoints.inspect_header(header(quant)), quant or "float")

    def test_missing_metadata_wrong_tokenizer_codec_and_unknown_quantization_are_rejected(self):
        from modules_forge.irodori import checkpoints

        bad = []
        missing = header()
        missing["__metadata__"].pop("config_json")
        bad.append(missing)
        for field, value in (("latent_dim", 128), ("text_tokenizer_repo", "other/tokenizer"),
                             ("caption_tokenizer_repo", "other/tokenizer")):
            invalid = header()
            cfg = copy.deepcopy(CONFIG)
            cfg[field] = value
            invalid["__metadata__"]["config_json"] = json.dumps(cfg)
            bad.append(invalid)
        incomplete = header()
        incomplete.pop(WEIGHT)
        bad.append(incomplete)
        unknown = header("int8_weight_only")
        q = json.loads(unknown["__metadata__"]["irodori_quantization_json"])
        q["quantization_type"] = "w4a8"
        unknown["__metadata__"]["irodori_quantization_json"] = json.dumps(q)
        bad.append(unknown)
        missing_parts = header("int8_weight_only")
        missing_parts.pop("blocks.0.attention.wq._weight_scale")
        bad.append(missing_parts)
        wrong_block = header("int8_weight_only")
        item = json.loads(wrong_block["__metadata__"][WEIGHT])
        item["_data"]["block_size"] = [8, 8]
        wrong_block["__metadata__"][WEIGHT] = json.dumps(item)
        bad.append(wrong_block)
        for invalid in bad:
            with self.subTest(metadata=invalid.get("__metadata__", {}).keys()), self.assertRaises(ValueError), \
                 patch.object(checkpoints, "V4_LARGE_SCHEMA_SHA256", SCHEMA):
                checkpoints.inspect_header(invalid)

    def test_custom_path_is_confined_and_bound_to_content(self):
        from modules_forge.irodori import checkpoints

        with tempfile.TemporaryDirectory() as folder:
            models = Path(folder)
            directory = models / "v4-Large/checkpoints"
            directory.mkdir(parents=True)
            path = directory / "MyVoice.safetensors"
            path.write_bytes(b"first")
            with patch.object(checkpoints.local_assets, "read_header", return_value=header()), \
                 patch.object(checkpoints, "V4_LARGE_SCHEMA_SHA256", SCHEMA):
                request = {"checkpoint": path.name, "precision": "bf16"}
                identity = checkpoints.bind(models, request)
                self.assertEqual(identity["name"], path.name)
                self.assertEqual(len(identity["sha256"]), 64)
                path.write_bytes(b"other")
                with self.assertRaises(ValueError):
                    checkpoints.bind(models, request)
            for name in ("../secret.safetensors", "nested/model.safetensors", "C:/file.safetensors", "model.pt"):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    checkpoints.resolve(models, name)

    def test_replaced_after_selection_is_rejected_before_gpu_ownership(self):
        from modules_forge.irodori import checkpoints
        from modules_forge.irodori.service import Service

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            directory = root / "models/v4-Large/checkpoints"
            directory.mkdir(parents=True)
            path = directory / "MyVoice.safetensors"
            path.write_bytes(b"first")
            with (
                patch.object(checkpoints.local_assets, "read_header", return_value=header()),
                patch.object(checkpoints, "V4_LARGE_SCHEMA_SHA256", SCHEMA),
                patch("modules_forge.irodori.service.ready", return_value=True),
                patch("modules_forge.gpu_ownership.GPUOwnership") as ownership,
            ):
                request = {"text": "文", "checkpoint": path.name, "precision": "bf16"}
                identity = checkpoints.bind(root / "models", request)
                path.write_bytes(b"other")
                with self.assertRaisesRegex(ValueError, "変更"):
                    Service(root=root, models=root / "models").start(
                        {**request, "checkpoint_identity": identity}, [], "owner"
                    )
                ownership.assert_not_called()


if __name__ == "__main__":
    unittest.main()
