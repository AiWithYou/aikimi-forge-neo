"""MOSS precision selection and verified component/runtime boundaries."""

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules_forge.moss_image import core


class CoreTests(unittest.TestCase):
    def test_image_free_requests_validate_before_site_packages_are_installed(self):
        source = (
            "from modules_forge.moss_image.core import validate_request; "
            "value=validate_request({'prompt':'setup','seed':0}); "
            "assert value['images']==[]; "
            "value=validate_request({'prompt':'cat','width':1536,'height':1536,"
            "'hires':{'base_width':1024,'base_height':1024}}); "
            "assert value['hires']['steps']==20"
        )
        result = subprocess.run(  # noqa: S603 -- Fixed source, current interpreter, no site packages.
            [sys.executable, "-S", "-c", source],
            cwd=core.ROOT,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_int8_defaults_preserve_generation_settings_and_loras(self):
        value = core.validate_request({"prompt": " anime ", "seed": 7})
        self.assertEqual((value["precision"], value["text_precision"]), ("int8", "int8"))
        self.assertEqual(value["loras"], [])
        self.assertEqual((value["prompt"], value["steps"], value["cfg"]), (" anime ", 50, 3))
        value = core.validate_request(
            {
                "prompt": "cat",
                "precision": "bf16",
                "text_precision": "w4a8",
                "steps": 23,
                "cfg": 4.5,
                "loras": [{"name": "style.safetensors", "strength": -0.5}],
            }
        )
        self.assertEqual(
            (value["precision"], value["text_precision"], value["steps"], value["cfg"]), ("bf16", "w4a8", 23, 4.5)
        )
        self.assertEqual(value["loras"][0]["strength"], -0.5)

    def test_invalid_numbers_precisions_and_lora_values_are_rejected(self):
        for change in (
            {"cfg": 10**1000},
            {"shift": float("nan")},
            {"width": True},
            {"width": 1000},
            {"precision": "fp8"},
            {"text_precision": "nvfp4"},
            {"skip_layer": 1},
            {"loras": [{"name": "a.safetensors", "strength": True}]},
            {"loras": [{"name": "../a.safetensors", "strength": 1}]},
            {"loras": [{"name": "a.safetensors", "strength": 3}]},
            {"loras": [{"name": "a.safetensors", "strength": 1}] * 2},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", **change})

    def test_internal_identity_cannot_be_supplied_via_public_request(self):
        value = {"prompt": "cat", "lora_identities": []}
        with self.assertRaises(ValueError):
            core.validate_request(value)
        self.assertEqual(core.validate_request(value, internal=True)["lora_identities"], [])

    def test_resolution_limit_allows_4k_dimensions_and_rejects_over_4096(self):
        for width, height in ((3840, 2160), (4096, 4096)):
            with self.subTest(width=width, height=height):
                value = core.validate_request({"prompt": "cat", "width": width, "height": height})
                self.assertEqual((value["width"], value["height"]), (width, height))
        for width, height in ((4112, 512), (512, 4112)):
            with self.subTest(width=width, height=height), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", "width": width, "height": height})

    def test_download_selection_keeps_fp32_vae_and_does_not_fetch_unused_weights(self):
        plan = core.model_specs(core.validate_request({"prompt": "cat"}))
        self.assertTrue(any("dit-int8" in x["path"] for x in plan))
        self.assertTrue(any("qwen35-int8" in x["path"] for x in plan))
        self.assertFalse(any("model-00001" in x["path"] for x in plan))
        self.assertTrue(any(x["path"].endswith("vae/diffusion_pytorch_model.safetensors") for x in plan))
        w4 = core.model_specs(core.validate_request({"prompt": "cat", "text_precision": "w4a8"}))
        self.assertTrue(any("w4a8" in x["path"] for x in w4))
        self.assertFalse(any("qwen35-int8" in x["path"] for x in w4))

    def test_component_paths_follow_the_selected_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "alias").mkdir()
            request = core.validate_request({"prompt": "cat", "precision": "bf16", "text_precision": "int8"})
            root = root / "alias" / ".."
            paths = core.asset_paths(root, request)
            self.assertEqual(paths["dit"], root.resolve() / "checkpoints/MOSS-Image-SFT")
            self.assertTrue(paths["text_encoder"].name.endswith("int8-convrot.safetensors"))
            self.assertTrue(paths["vae"].is_relative_to(root.resolve()))

    def test_verified_model_receipt_requires_the_selected_components_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = b"verified"
            spec = {"path": "part.safetensors", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            path = root / spec["path"]
            path.write_bytes(data)
            receipt = {
                "schema_version": 2,
                "model_id": core.MODEL_ID,
                "model_revision": core.MODEL_REVISION,
                "manifest_sha256": hashlib.sha256(core.MANIFEST.read_bytes()).hexdigest(),
                "files": {
                    spec["path"]: {
                        "size": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns,
                        "sha256": spec["sha256"],
                    }
                },
            }
            core.atomic_json(root / core.RECEIPT, receipt)
            with (
                patch.object(core, "model_specs", return_value=[spec]),
                patch.object(core, "runtime_ready", return_value=True),
            ):
                self.assertTrue(core.model_ready(root))
                path.write_bytes(b"tampered")
                self.assertFalse(core.model_ready(root))
            with patch.object(core, "runtime_ready", return_value=False):
                self.assertFalse(core.model_ready(root))

    def test_runtime_paths_are_separate_from_other_studios(self):
        paths = core.runtime_paths()
        self.assertEqual(paths["comfy"], core.ROOT / "repositories/moss-image/ComfyUI")
        self.assertEqual(set(paths), {"base", "comfy", "python"})
        self.assertEqual(core.python_path(), paths["python"])
        self.assertEqual(len(core.runtime_fingerprint()), 64)

    def test_runtime_ready_rejects_missing_pinned_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(core.runtime_ready(Path(directory) / "models/MOSS-Image"))
