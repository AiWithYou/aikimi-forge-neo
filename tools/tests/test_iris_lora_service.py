"""Service acceptance persists authoritative LoRA identity and sampling conditions."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from modules_forge import local_assets
from modules_forge.iris import lora
from modules_forge.iris import service as service_module
from modules_forge.iris.service import Service
from tools.tests.test_iris_lora import adapter


class LoRAServiceTests(unittest.TestCase):
    def generation(self, **changes):
        return {
            "task": "generate",
            "precision": "int8",
            "prompt": "A red kite at sunset",
            "steps": 37,
            "cfg": 7.0,
            "negative_prompt": "blurry",
            "seed": 123,
            **changes,
        }

    def test_turbo_acceptance_saves_manual_settings_and_source_identity_before_dispatch(self):
        for precision in ("normal", "int8", "w4a8"):
            with self.subTest(precision=precision), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                turbo = adapter(
                    root,
                    "turbo.safetensors",
                    meta={"rank": "1", "alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})},
                )
                style = adapter(root, "style.safetensors")
                selection = [{"name": turbo.name, "strength": 1}, {"name": style.name, "strength": -0.5}]
                request = self.generation(
                    precision=precision, loras=selection, sampling={"mode": "standard"}, steps=8, cfg=2.0
                )
                service = Service(root)
                lease = Mock()
                lease.acquire.return_value = True
                dispatched = []

                def observe_dispatch(service=service, dispatched=dispatched):
                    job = next(iter(service.jobs.values()))
                    dispatched.append(json.loads((Path(job["directory"]) / "request.json").read_text(encoding="utf-8")))

                with (
                    patch.object(service_module, "ROOT", root),
                    patch.object(service_module, "model_ready", return_value=True),
                    patch.object(local_assets, "HASH_CACHE", root / "hashes.json"),
                    patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease) as ownership,
                    patch.object(service_module.threading, "Thread") as thread,
                ):
                    thread.return_value.start.side_effect = observe_dispatch
                    identifier = service.start(request)
                saved = json.loads((root / "outputs/iris" / identifier / "request.json").read_text(encoding="utf-8"))
                self.assertEqual(dispatched, [saved])
                self.assertEqual((saved["steps"], saved["cfg"], saved["negative_prompt"]), (8, 2.0, "blurry"))
                self.assertEqual(saved["sampling"], {"mode": "turbo", "default_steps": 4, "shift": 4.0})
                self.assertEqual(saved["precision"], precision)
                self.assertEqual(saved["seed"], 123)
                self.assertEqual(
                    saved["loras"][0]["metadata"], {"rank": 1, "alpha": 1.0, "turbo_steps": 4, "shift": 4.0}
                )
                for record, path, strength in zip(saved["loras"], (turbo, style), (1.0, -0.5), strict=True):
                    self.assertEqual(
                        (record["name"], record["path"], record["strength"]), (path.name, str(path.resolve()), strength)
                    )
                    self.assertEqual(
                        record["source"],
                        {
                            "path": str(path.resolve()),
                            "size": path.stat().st_size,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        },
                    )
                self.assertEqual((request["steps"], request["cfg"], request["negative_prompt"]), (8, 2.0, "blurry"))
                self.assertEqual(request["loras"], selection)
                ownership.assert_called_once_with()
                lease.acquire.assert_called_once_with(False)
                self.assertEqual(lease.engine, "iris")
                self.assertEqual(thread.call_args.kwargs["target"], service.execute)
                self.assertEqual(thread.call_args.kwargs["args"], (service.jobs[identifier], lease))
                thread.return_value.start.assert_called_once_with()

    def test_turbo_fills_only_omitted_steps_and_cfg_from_recommended_defaults(self):
        for omitted in (("steps", "cfg"), ("steps",), ("cfg",)):
            with self.subTest(omitted=omitted), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                adapter(
                    root,
                    meta={"rank": "1", "alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})},
                )
                request = self.generation(loras=[{"name": "test.safetensors", "strength": 1}], steps=8, cfg=2.0)
                for name in omitted:
                    request.pop(name)
                service = Service(root)
                lease = Mock()
                lease.acquire.return_value = True
                with (
                    patch.object(service_module, "ROOT", root),
                    patch.object(service_module, "model_ready", return_value=True),
                    patch.object(local_assets, "HASH_CACHE", root / "hashes.json"),
                    patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                    patch.object(service_module.threading, "Thread"),
                ):
                    identifier = service.start(request)
                saved = json.loads((root / "outputs/iris" / identifier / "request.json").read_text(encoding="utf-8"))
                self.assertEqual(
                    (saved["steps"], saved["cfg"], saved["negative_prompt"]),
                    (4 if "steps" in omitted else 8, 1.0 if "cfg" in omitted else 2.0, "blurry"),
                )
                self.assertEqual(saved["sampling"], {"mode": "turbo", "default_steps": 4, "shift": 4.0})
                for name in omitted:
                    self.assertNotIn(name, request)

    def test_invalid_names_and_strengths_are_rejected_before_gpu_ownership_or_thread(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter(root, "normal.safetensors")
            adapter(
                root,
                "turbo.safetensors",
                meta={"rank": "1", "alpha": "1", "extra": json.dumps({"turbo_steps": 4, "shift": 4})},
            )
            selections = (
                [{"name": "missing.safetensors", "strength": 1}],
                [{"name": "normal.bin", "strength": 1}],
                [{"name": "../escape.safetensors", "strength": 1}],
                [{"name": "https://example.invalid/normal.safetensors", "strength": 1}],
                [{"name": "normal.safetensors", "strength": 3}],
                [{"name": "normal.safetensors", "strength": True}],
                [{"name": "normal.safetensors", "strength": float("nan")}],
                [{"name": "normal.safetensors", "strength": "invalid"}],
                [{"name": "turbo.safetensors", "strength": 0.5}],
            )
            service = Service(root)
            for selection in selections:
                with (
                    self.subTest(selection=selection),
                    patch.object(service_module, "ROOT", root),
                    patch.object(service_module, "model_ready", return_value=True) as ready,
                    patch.object(local_assets, "HASH_CACHE", root / "hashes.json"),
                    patch("modules_forge.gpu_ownership.GPUOwnership") as ownership,
                    patch.object(service_module.threading, "Thread") as thread,
                    self.assertRaises(ValueError),
                ):
                    service.start(self.generation(loras=selection))
                ready.assert_not_called()
                ownership.assert_not_called()
                thread.assert_not_called()
                self.assertEqual(service.jobs, {})
                self.assertFalse((root / "outputs").exists())

    def test_depth_and_upscale_discard_generation_loras_and_sampling_before_dispatch(self):
        for task in ("depth", "upscale"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                service = Service(root)
                lease = Mock()
                lease.acquire.return_value = True
                request = self.generation(
                    task=task,
                    image=Image.new("RGB", (4, 3)),
                    loras=[{"name": "missing.safetensors", "strength": "invalid"}],
                    sampling={"mode": "turbo", "steps": 4, "shift": 4},
                )
                with (
                    patch.object(service_module, "ROOT", root),
                    patch.object(service_module, "model_ready", return_value=True),
                    patch.object(lora, "snapshot") as snapshot,
                    patch.object(lora, "sampling_settings") as sampling,
                    patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                    patch.object(service_module.threading, "Thread") as thread,
                ):
                    identifier = service.start(request)
                saved = json.loads((root / "outputs/iris" / identifier / "request.json").read_text(encoding="utf-8"))
                self.assertEqual(set(saved), {"task", "precision", "image"})
                self.assertEqual((saved["task"], saved["precision"]), (task, "int8"))
                snapshot.assert_not_called()
                sampling.assert_not_called()
                lease.acquire.assert_called_once_with(False)
                thread.return_value.start.assert_called_once_with()

    def test_zero_strength_missing_lora_saves_standard_settings_without_path_lookup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            service = Service(root)
            lease = Mock()
            lease.acquire.return_value = True
            request = self.generation(
                loras=[{"name": "missing.safetensors", "strength": 0}],
                sampling={"mode": "turbo", "steps": 4, "shift": 4},
            )
            with (
                patch.object(service_module, "ROOT", root),
                patch.object(service_module, "model_ready", return_value=True),
                patch.object(lora, "resolve") as resolve,
                patch("modules_forge.gpu_ownership.GPUOwnership", return_value=lease),
                patch.object(service_module.threading, "Thread") as thread,
            ):
                identifier = service.start(request)
            saved = json.loads((root / "outputs/iris" / identifier / "request.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["loras"], [])
            self.assertEqual(saved["sampling"], {"mode": "standard"})
            self.assertEqual((saved["steps"], saved["cfg"], saved["negative_prompt"]), (37, 7.0, "blurry"))
            resolve.assert_not_called()
            lease.acquire.assert_called_once_with(False)
            thread.return_value.start.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
