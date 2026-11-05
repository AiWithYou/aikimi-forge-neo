"""Full FT selection keeps the file, architecture and provenance bound to a job."""

import json
import stat
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge.flux3_klein import checkpoints, core
from modules_forge.flux3_klein.history import History
from modules_forge.flux3_klein.inference import ComfyRuntime, Runner
from modules_forge.flux3_klein.service import Service
from tools import setup_flux3_klein as setup
from tools.tests.test_flux3_klein_inference import FakeRuntime


def write_fixture(path, *, model="distilled", precision="bf16", descriptor=None):
    """Tiny valid safetensors; tests replace the architecture's expected dimensions."""
    tensors = {
        "txt_in.weight": ("BF16", [2, 4], bytes(16)),
        "single_blocks.0.qkv.weight": ("BF16", [4, 2], bytes(16)),
    }
    if model == "distilled":
        tensors["ref_time_in.in_layer.weight"] = ("BF16", [2, 2], bytes(8))
    if precision == "int8":
        name = "single_blocks.0.qkv"
        config = descriptor or {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
        config = json.dumps(config).encode()
        tensors[name + ".weight"] = ("I8", [4, 2], bytes(8))
        tensors[name + ".weight_scale"] = ("F32", [4, 1], bytes(16))
        tensors[name + ".comfy_quant"] = ("U8", [len(config)], config)
    offset, header, payload = 0, {}, bytearray()
    for key, (dtype, shape, data) in tensors.items():
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def tiny_shapes(model):
    shapes = {"txt_in.weight": (2, 4), "single_blocks.0.qkv.weight": (4, 2)}
    if model == "distilled":
        shapes["ref_time_in.in_layer.weight"] = (2, 2)
    return shapes


def write_separate_qkv(path, *, model="base", change=None):
    tensors = {"txt_in.weight": ("BF16", [2, 4], bytes(16))}
    if model == "distilled":
        tensors["ref_time_in.in_layer.weight"] = ("BF16", [2, 2], bytes(8))
    for block in range(2):
        prefix = f"single_blocks.{block}"
        for projection in ("q", "k", "v"):
            tensors[f"{prefix}.{projection}_proj.weight"] = ("BF16", [2, 2], bytes(8))
    if change == "missing":
        tensors.pop("single_blocks.1.v_proj.weight")
    elif change == "mixed":
        for projection in ("q", "k", "v"):
            tensors.pop(f"single_blocks.1.{projection}_proj.weight")
        tensors["single_blocks.1.qkv.weight"] = ("BF16", [6, 2], bytes(24))
    elif change == "both":
        tensors["single_blocks.0.qkv.weight"] = ("BF16", [6, 2], bytes(24))
    elif change == "wrong_shape":
        tensors["single_blocks.1.k_proj.weight"] = ("BF16", [1, 2], bytes(4))
    elif change == "int8":
        tensors["single_blocks.0.q_proj.weight"] = ("I8", [2, 2], bytes(4))
    offset, header, payload = 0, {}, bytearray()
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def qkv_test_shapes(model):
    shapes = {"txt_in.weight": (2, 4), **{f"single_blocks.{n}.qkv.weight": (6, 2) for n in range(2)}}
    if model == "distilled":
        shapes["ref_time_in.in_layer.weight"] = (2, 2)
    return shapes


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "checkpoints/sub/custom.safetensors"
        self.shapes = patch.object(checkpoints, "expected_shapes", side_effect=tiny_shapes)
        self.quantized = patch.object(checkpoints, "quantized_keys", return_value={"single_blocks.0.qkv.weight"})
        self.shapes.start()
        self.quantized.start()
        self.addCleanup(self.shapes.stop)
        self.addCleanup(self.quantized.stop)

    def test_bf16_and_real_int8_convrot_signatures_are_distinguished(self):
        for variant in ("base", "distilled"):
            for precision in ("bf16", "int8"):
                with self.subTest(model=variant, precision=precision):
                    write_fixture(self.path, model=variant, precision=precision)
                    info = checkpoints.inspect(self.root, "sub/custom.safetensors")
                    self.assertEqual((info["model"], info["precision"]), (variant, precision))
                    self.assertEqual(len(info["sha256"]), 64)
                    self.assertNotIn(str(self.root), json.dumps(info))

    def test_bf16_accepts_separate_q_k_v_for_base_and_distilled(self):
        with (
            patch.object(checkpoints, "expected_shapes", side_effect=qkv_test_shapes),
            patch.object(checkpoints, "quantized_keys", return_value={f"single_blocks.{n}.qkv.weight" for n in range(2)}),
        ):
            for variant in ("base", "distilled"):
                with self.subTest(model=variant):
                    write_separate_qkv(self.path, model=variant)
                    info = checkpoints.inspect(self.root, "sub/custom.safetensors")
                    self.assertEqual((info["model"], info["precision"]), (variant, "bf16"))

    def test_bf16_rejects_mixed_missing_duplicated_or_wrong_projections(self):
        with (
            patch.object(checkpoints, "expected_shapes", side_effect=qkv_test_shapes),
            patch.object(checkpoints, "quantized_keys", return_value={f"single_blocks.{n}.qkv.weight" for n in range(2)}),
        ):
            for change in ("missing", "mixed", "both", "wrong_shape", "int8"):
                with self.subTest(change=change):
                    write_separate_qkv(self.path, change=change)
                    with self.assertRaises(ValueError):
                        checkpoints.inspect(self.root, "sub/custom.safetensors")

    def test_unrecognized_quantization_and_incomplete_model_are_rejected(self):
        write_fixture(self.path, precision="int8", descriptor={"format": "asym_w4a8_int8"})
        with self.assertRaisesRegex(ValueError, "量子化"):
            checkpoints.inspect(self.root, "sub/custom.safetensors")
        write_fixture(
            self.path,
            precision="int8",
            descriptor={"format": "int8_tensorwise", "convrot": 1, "convrot_groupsize": 256},
        )
        with self.assertRaisesRegex(ValueError, "量子化"):
            checkpoints.inspect(self.root, "sub/custom.safetensors")
        write_fixture(self.path)
        with patch.object(
            checkpoints, "expected_shapes", return_value={**tiny_shapes("distilled"), "missing.weight": (4, 4)}
        ):
            with self.assertRaisesRegex(ValueError, "本体"):
                checkpoints.inspect(self.root, "sub/custom.safetensors")

    def test_identity_is_bound_at_selection_and_rechecked_before_gpu(self):
        write_fixture(self.path)
        identity = checkpoints.inspect(self.root, "sub/custom.safetensors")
        self.path.write_bytes(self.path.read_bytes() + b"changed")
        request = {"prompt": "cat", "checkpoint": "sub/custom.safetensors", "model": "distilled", "precision": "bf16"}
        service = Service(self.root, outputs=self.root / "outputs")
        with patch("modules_forge.gpu_ownership.GPUOwnership") as ownership:
            with self.assertRaisesRegex(ValueError, "変更"):
                service.start(request, checkpoint_identity=identity)
            ownership.assert_not_called()

    def test_mismatched_selected_format_and_paths_are_rejected(self):
        write_fixture(self.path)
        request = {"checkpoint": "sub/custom.safetensors", "model": "base", "precision": "int8"}
        with self.assertRaisesRegex(ValueError, "一致"):
            checkpoints.bind(self.root, request)
        for name in ("../custom.safetensors", "C:/private.safetensors", "/absolute.safetensors"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                checkpoints.resolve(self.root, name)
        original = Path.lstat

        def redirected(path):
            return SimpleNamespace(st_mode=stat.S_IFLNK) if path == self.path else original(path)

        with patch.object(Path, "lstat", redirected), self.assertRaises(ValueError):
            checkpoints.resolve(self.root, "sub/custom.safetensors")

    def test_custom_model_does_not_require_official_dit_and_records_its_own_source(self):
        write_fixture(self.path)
        request = core.validate_request({"prompt": "cat", "checkpoint": "sub/custom.safetensors", "precision": "bf16"})
        checkpoints.bind(self.root, request)
        with (
            patch.object(core, "runtime_ready", return_value=True),
            patch.object(core, "asset_ready", return_value=True) as ready,
        ):
            self.assertTrue(core.model_ready(self.root, request))
        self.assertTrue(all("diffusion_models/" not in call.args[1]["path"] for call in ready.call_args_list))
        record = checkpoints.record(request)
        self.assertEqual(record["source"], "custom")
        self.assertEqual(record["name"], "sub/custom.safetensors")
        self.assertNotIn("revision", record)
        self.assertEqual(core.model_paths(self.root, request)["dit"], self.path.resolve())

    def test_an_exact_official_copy_is_recorded_as_official_origin(self):
        write_fixture(self.path)
        request = core.validate_request({"prompt": "cat", "checkpoint": "sub/custom.safetensors", "precision": "bf16"})
        identity = checkpoints.bind(self.root, request)
        with patch.object(
            core, "model_specs", return_value={"dit": {"size": identity["size"], "sha256": identity["sha256"]}}
        ):
            record = checkpoints.record(request)
        self.assertEqual(record["source"], "official")
        self.assertEqual(record["revision"], core.MODEL_REVISION)
        self.assertEqual(record["sha256"], identity["sha256"])

    def test_components_only_plan_excludes_official_dit(self):
        plan = setup.download_plan(precision="bf16", components_only=True)
        self.assertEqual(
            {spec["path"] for spec in plan}, {"text_encoders/qwen3vl_4b_bf16.safetensors", "vae/flux2-vae.safetensors"}
        )
        with patch.object(setup, "run", return_value={"ok": True}) as run, patch("builtins.print"):
            self.assertEqual(setup.main(["--components-only"]), 0)
        self.assertTrue(run.call_args.kwargs["components_only"])

    def test_exact_selected_path_is_passed_to_comfy_even_if_names_collide(self):
        model = type("Flux3", (), {"__module__": "comfy.ldm.flux3.model"})()
        model.flow_map = True
        loaded = SimpleNamespace(model=SimpleNamespace(diffusion_model=model))
        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.sd = SimpleNamespace(load_diffusion_model=Mock(return_value=loaded))
        runtime.nodes = SimpleNamespace(
            NODE_CLASS_MAPPINGS={"UNETLoader": Mock(side_effect=AssertionError("basename loader"))}
        )
        target = self.root / "checkpoints/sub/same-name.safetensors"
        self.assertIs(runtime.load_dit(target, "distilled"), loaded)
        runtime.sd.load_diffusion_model.assert_called_once_with(str(target), model_options={})

    def test_custom_provenance_survives_png_and_result_validation(self):
        write_fixture(self.path)
        request = core.validate_request(
            {
                "prompt": "cat",
                "checkpoint": "sub/custom.safetensors",
                "precision": "bf16",
                "width": 64,
                "height": 64,
                "seed": 7,
            }
        )
        checkpoints.bind(self.root, request)
        directory = self.root / ("a" * 32)
        directory.mkdir()
        request["reference_sha256"] = []
        core.atomic_json(directory / "request.json", request)
        paths = {name: self.root / name for name in ("dit", "text_encoder", "vae", "pup", "pup_prompts")}
        paths["dit"] = self.path
        with patch("modules_forge.flux3_klein.inference.model_paths", return_value=paths):
            result = Runner(self.root, FakeRuntime()).run(directory, request)
        self.assertIsNone(result["model_revision"])
        self.assertEqual(result["checkpoint"], checkpoints.record(request))
        self.assertEqual(Service.validate_result(directory)["checkpoint"], result["checkpoint"])
        self.assertEqual(History(self.root).load("a" * 32)["checkpoint"], result["checkpoint"])
        result["checkpoint"]["source"] = "official"
        core.atomic_json(directory / "result.json", result)
        with self.assertRaisesRegex(ValueError, "出所"):
            Service.validate_result(directory)
        with self.assertRaises(ValueError):
            History(self.root).load("a" * 32)

    def test_replacement_during_text_encoding_is_rejected_before_model_load(self):
        write_fixture(self.path)
        request = core.validate_request(
            {
                "prompt": "cat",
                "checkpoint": "sub/custom.safetensors",
                "precision": "bf16",
                "width": 64,
                "height": 64,
                "seed": 7,
            }
        )
        checkpoints.bind(self.root, request)
        runtime = FakeRuntime()

        def replace(*args):
            self.path.write_bytes(self.path.read_bytes() + b"changed")
            return "encoded"

        runtime.encode_text = replace
        paths = {name: self.root / name for name in ("dit", "text_encoder", "vae", "pup", "pup_prompts")}
        paths["dit"] = self.path
        with patch("modules_forge.flux3_klein.inference.model_paths", return_value=paths):
            with self.assertRaisesRegex(ValueError, "変更"):
                Runner(self.root, runtime).run(self.root / "job", request)
        self.assertFalse(any(event[0] == "dit" for event in runtime.events))
        self.assertFalse((self.root / "job/output.png").exists())


class RealHeaderTests(unittest.TestCase):
    def test_architecture_table_has_the_expected_full_size_layers(self):
        distilled = checkpoints.expected_shapes("distilled")
        base = checkpoints.expected_shapes("base")
        self.assertEqual(len(distilled), 273)
        self.assertEqual(len(base), 271)
        self.assertEqual(len(checkpoints.quantized_keys()), 172)
        self.assertEqual(distilled["txt_in.weight"], (3072, 20480))
        self.assertEqual(distilled["single_blocks.27.mlp_in.weight"], (18432, 3072))
        self.assertIn("ref_time_in.in_layer.weight", distilled)
        self.assertNotIn("ref_time_in.in_layer.weight", base)

    def test_installed_official_int8_headers_match_supported_structure(self):
        root = core.RUNTIME / "diffusion_models"
        for name, model in (
            ("flux3_image_klein_sd_int8_convrot.safetensors", "distilled"),
            ("flux3_image_klein_int8_convrot.safetensors", "base"),
        ):
            path = root / name
            if not path.is_file():
                self.skipTest("official INT8 weights are not installed")
            self.assertEqual(checkpoints.header_info(path), {"model": model, "precision": "int8"})

    def test_installed_official_bf16_separate_qkv_header(self):
        path = core.RUNTIME / "base-bf16-training/model.safetensors"
        if not path.is_file():
            self.skipTest("official Base BF16 training weights are not installed")
        self.assertEqual(checkpoints.header_info(path), {"model": "base", "precision": "bf16"})


if __name__ == "__main__":
    unittest.main()
