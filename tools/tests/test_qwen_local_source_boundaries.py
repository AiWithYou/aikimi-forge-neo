"""External source validation preserves selection and reaches the resident model key."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from modules_forge import local_assets
from modules_forge.qwen_image21 import core, local_source, service
from tools.tests import test_local_model_sources as fixtures
from tools.tests import test_qwen_image21_service as service_fixtures
from tools.tests import test_qwen_image21_worker as worker_fixtures


class QwenLocalSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.enterContext(patch.object(local_assets, "LIBRARY", self.root / "assets/library.json"))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "assets/hashes.json"))
        self.runtime = self.root / "runtime"
        python = self.runtime / "worker-env" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.write_bytes(b"test interpreter")
        core.atomic_json(self.runtime / "runtime.json", {"schema": 1, "diffusers_revision": core.DIFFUSERS_REVISION})
        self.components = fixtures.qwen_fixture(self.root / "共通部品", transformer=False)
        self.transformer = fixtures.qwen_transformer(self.root / "外部本体")

    def test_shared_components_reject_partial_weights_before_identity(self):
        for component in ("text_encoder", "vae"):
            with self.subTest(component=component):
                weights = self.components / component / "model.safetensors"
                original = weights.read_bytes()
                weights.write_bytes(original[:-1])
                try:
                    with (
                        patch.object(
                            local_assets, "identity", side_effect=AssertionError("partial component identified")
                        ),
                        self.assertRaises(ValueError),
                    ):
                        local_source.resolve(self.runtime, str(self.transformer), str(self.components), "int8")
                finally:
                    weights.write_bytes(original)

    def test_invalid_model_preflight_keeps_saved_selection_and_does_not_start_worker(self):
        selected = {
            "local_model": str(self.transformer),
            "local_components": str(self.components),
            "style_loras": [],
            "precision": "int8",
        }
        local_assets.remember("qwen21_model", [str(self.transformer)])
        local_assets.save_selection("qwen21", selected)
        local_assets.HASH_CACHE.write_bytes(b'{"sentinel": "unchanged"}')
        library_before = local_assets.LIBRARY.read_bytes()
        hashes_before = local_assets.HASH_CACHE.read_bytes()
        weights = self.transformer / "diffusion_pytorch_model.safetensors"
        weights.write_bytes(weights.read_bytes()[:-1])
        ownership, workers = Mock(), Mock()
        studio = service.Studio(
            self.runtime,
            self.root / "outputs",
            ownership_factory=ownership,
            release_vram=Mock(),
            worker_factory=workers,
            residency=service_fixtures.Residency(),
        )
        self.addCleanup(studio.shutdown)
        request = core.Request(prompt="a bird", width=256, height=256, **selected)
        with self.assertRaises(core.QwenImage21Error):
            studio.start(request, "owner")
        self.assertEqual(local_assets.selection("qwen21"), selected)
        self.assertEqual(local_assets.LIBRARY.read_bytes(), library_before)
        self.assertEqual(local_assets.HASH_CACHE.read_bytes(), hashes_before)
        ownership.assert_not_called()
        workers.assert_not_called()

    def test_external_components_and_transformer_selection_reach_resident_key(self):
        worker = worker_fixtures.worker
        self.enterContext(patch.object(worker, "_RESIDENT_RUNTIME", None))
        self.enterContext(patch.object(worker, "_RESIDENT_KEY", None))
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))
        self.enterContext(patch.dict(sys.modules, {"torch": fake_torch}))
        job = self.root / "job"
        job.mkdir()
        request = {
            "precision": "int8",
            "memory_mode": "offload",
            "runtime_root": str(self.runtime),
            "local_model": str(self.transformer),
            "local_components": str(self.components),
        }

        def resolve_and_load():
            request["local_source"] = local_source.resolve(
                self.runtime, request["local_model"], request["local_components"], request["precision"]
            )
            return worker._runtime_for_request(Path(request["local_source"]["model"]), request, job)

        with patch.object(worker, "_load_runtime", side_effect=[{}, {}, {}]) as loader, redirect_stdout(io.StringIO()):
            self.assertFalse(resolve_and_load()[1])
            self.assertTrue(resolve_and_load()[1])
            request["local_components"] = str(fixtures.qwen_fixture(self.root / "別の共通部品", transformer=False))
            self.assertFalse(resolve_and_load()[1])
            request["local_model"] = str(fixtures.qwen_transformer(self.root / "別の本体"))
            self.assertFalse(resolve_and_load()[1])
            self.assertEqual(loader.call_count, 3)
        self.assertFalse((self.runtime / "model").exists())


if __name__ == "__main__":
    unittest.main()
