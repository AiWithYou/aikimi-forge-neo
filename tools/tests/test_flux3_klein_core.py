"""CPU-only contracts for FLUX 3 requests and pinned local assets."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge.flux3_klein import core


class Flux3RequestTests(unittest.TestCase):
    def request(self, **values):
        return core.validate_request({"prompt": " a red balloon ", **values})

    def test_distilled_and_base_defaults_preserve_prompt(self):
        distilled = self.request()
        self.assertEqual(
            (distilled["model"], distilled["precision"], distilled["steps"], distilled["cfg"]),
            ("distilled", "int8", 16, 1.0),
        )
        self.assertEqual((distilled["width"], distilled["height"]), (1024, 1024))
        self.assertEqual(distilled["prompt"], " a red balloon ")
        self.assertIs(distilled["pup"], False)
        self.assertTrue(0 <= distilled["seed"] <= 2**32 - 1)
        base = self.request(model="base", seed=0)
        self.assertEqual((base["steps"], base["cfg"], base["seed"]), (50, 3.5, 0))

    def test_reference_order_and_automatic_dimensions_are_preserved(self):
        first, second = object(), object()
        images = [first, second]
        result = self.request(images=images, pup=False)
        self.assertEqual(result["images"], images)
        self.assertIsNot(result["images"], images)
        self.assertIsNone(result["width"])
        self.assertIsNone(result["height"])

    def test_finite_strict_numeric_bounds_and_known_keys(self):
        invalid = [
            {"model": "klein"},
            {"precision": "gguf"},
            {"prompt": " "},
            {"prompt": "x" * 12001},
            {"steps": True},
            {"steps": 1.0},
            {"steps": 0},
            {"steps": 201},
            {"seed": True},
            {"seed": -2},
            {"seed": 2**32},
            {"pup": 1},
            {"cfg": float("nan")},
            {"cfg": float("inf")},
            {"cfg": True},
            {"cfg": 1.1},
            {"model": "base", "cfg": 0},
            {"model": "base", "cfg": 100.1},
            {"width": 512},
            {"width": None, "height": 512},
            {"width": 65, "height": 512},
            {"width": True, "height": 512},
            {"width": 4097, "height": 512},
            {"images": ()},
            {"images": [None] * 11},
            {"unexpected": "value"},
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(core.Flux3Error):
                self.request(**changes)
        for shape in ((64, 64), (4096, 4096)):
            self.assertEqual(self.request(width=shape[0], height=shape[1])["width"], shape[0])

    def test_request_must_be_a_dictionary(self):
        for value in (None, [], "prompt"):
            with self.assertRaises(core.Flux3Error):
                core.validate_request(value)

    def test_boxes_validate_both_layout_and_edit_schemas_without_mutation(self):
        rows = [
            {"id": "balloon", "desc": "red", "bbox": [0.0, 0.1, 0.5, 0.6]},
            {"id": "remove", "desc": "old bird", "from": "ref_image_0", "src_bbox": [0, 0, 1, 1], "tgt_bbox": None},
            {"id": "new", "desc": "cloud", "from": "target", "src_bbox": None, "tgt_bbox": [0.2, 0.2, 0.4, 0.4]},
        ]
        before = json.dumps(rows)
        result = self.request(images=["reference"], boxes=rows)
        self.assertEqual(result["boxes"], rows)
        self.assertEqual(json.dumps(rows), before)
        self.assertIsNot(result["boxes"][0], rows[0])

    def test_boxes_reject_bad_coordinate_identity_provenance_and_schema(self):
        valid = {"id": "obj", "desc": "balloon", "bbox": [0.1, 0.1, 0.9, 0.9]}
        bad = [
            {**valid, "bbox": [0, 0, 0, 1]},
            {**valid, "bbox": [-0.1, 0, 1, 1]},
            {**valid, "bbox": [0, 0, 1000, 1000]},
            {**valid, "bbox": [False, 0, 1, 1]},
            {**valid, "bbox": [0, 0, float("nan"), 1]},
            {**valid, "id": "bad-id"},
            {**valid, "desc": ""},
            {**valid, "from": "target"},
            {**valid, "unknown": 1},
            {"id": "obj", "desc": "x", "from": "ref_image_1", "src_bbox": [0, 0, 1, 1], "tgt_bbox": None},
            {"id": "obj", "desc": "x", "from": "target", "src_bbox": [0, 0, 1, 1], "tgt_bbox": None},
            {"id": "obj", "desc": "x", "from": "ref_image_0", "src_bbox": None, "tgt_bbox": [0, 0, 1, 1]},
        ]
        for row in bad:
            with self.subTest(row=row), self.assertRaises(core.Flux3Error):
                self.request(images=["reference"], boxes=[row])
        with self.assertRaises(core.Flux3Error):
            self.request(boxes=[valid, valid])
        with self.assertRaises(core.Flux3Error):
            self.request(boxes={})


class Flux3AssetTests(unittest.TestCase):
    def test_runtime_receipt_requires_matching_git_pin_and_clean_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models/Flux-3-Klein"
            python = core.python_path(root)
            python.parent.mkdir(parents=True)
            python.write_bytes(b"python")
            runtime = core.comfy_path(root)
            runtime.mkdir(parents=True)
            (runtime / "main.py").write_text("# source", encoding="utf-8")
            core.atomic_json(
                root / ".runtime.json",
                {"schema_version": 1, "fingerprint": core.runtime_fingerprint(), "comfy_revision": core.COMFY_REVISION},
            )

            def result(output):
                return SimpleNamespace(returncode=0, stdout=output)

            with patch.object(core.subprocess, "run", side_effect=[result(core.COMFY_REVISION), result("")]):
                self.assertTrue(core.runtime_ready(root))
            with patch.object(core.subprocess, "run", side_effect=[result(core.COMFY_REVISION), result(" M main.py")]):
                self.assertFalse(core.runtime_ready(root))
            with patch.object(core.subprocess, "run", return_value=result("wrong pin")):
                self.assertFalse(core.runtime_ready(root))

    def test_variant_names_and_managed_python_are_explicit(self):
        root = Path("isolated")
        paths = core.model_paths(root, {"model": "distilled", "precision": "int8"})
        self.assertEqual(paths["dit"], root / "diffusion_models/flux3_image_klein_sd_int8_convrot.safetensors")
        self.assertEqual(paths["pup_prompts"], root / "text_encoders/system_prompts.json")
        base = core.model_paths(root, {"model": "base", "precision": "bf16"})
        self.assertEqual(base["dit"].name, "flux3_image_klein_bf16.safetensors")
        self.assertEqual(base["text_encoder"].name, "qwen3vl_4b_bf16.safetensors")
        self.assertEqual(
            core.model_paths(root, {"precision": "nvfp4"})["text_encoder"].name, "qwen3vl_4b_int8_convrot.safetensors"
        )
        self.assertEqual(core.python_path(root), root / "worker-env/Scripts/python.exe")
        self.assertEqual(core.COMFY_REVISION, "20b46065ef7f34fa21e8abebe8d0cca90a904d6d")

    def test_readiness_checks_exact_hash_and_optional_pup(self):
        import hashlib

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = {
                name: {"path": f"{name}/asset", "size": 4, "sha256": hashlib.sha256(b"good").hexdigest()}
                for name in ("dit", "text_encoder", "vae", "pup", "pup_prompts")
            }
            with (
                patch.object(core, "model_specs", return_value=specs),
                patch.object(core, "runtime_ready", return_value=True),
            ):
                for name in ("dit", "text_encoder", "vae"):
                    path = root / specs[name]["path"]
                    path.parent.mkdir(parents=True)
                    path.write_bytes(b"good")
                self.assertTrue(core.model_ready(root, {"pup": False}))
                self.assertTrue(core.model_ready(root, {}))
                self.assertFalse(core.model_ready(root, {"pup": True}))
                (root / specs["dit"]["path"]).write_bytes(b"evil")
                self.assertFalse(core.model_ready(root, {"pup": False}))

    def test_model_readiness_requires_verified_runtime(self):
        with patch.object(core, "runtime_ready", return_value=False), patch.object(core, "model_specs") as specs:
            self.assertFalse(core.model_ready(Path("missing"), {"pup": False}))
        specs.assert_not_called()

    def test_atomic_json_rejects_nonfinite_values_and_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            core.atomic_json(path, {"message": "日本語"})
            with self.assertRaises(ValueError):
                core.atomic_json(path, {"bad": float("nan")})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"message": "日本語"})
            self.assertEqual(list(path.parent.glob("*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
