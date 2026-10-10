"""H3 handoff validates selected control models and Orbit LoRA before export."""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_handoff as handoff
from modules_forge import minimax_h3_handoff_store as store
from modules_forge import minimax_h3_union2_vae as union2_vae
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_fun_control import H3FunControl


class H3HandoffIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        self.control = H3FunControl("v2_canny")
        self.acceleration = H3Acceleration()
        self.request = types.SimpleNamespace(
            mode="text",
            dimensions=(864, 480),
            frame_count=124,
            steps=20,
            scheduler="simple",
            resolved_seed=42,
            control=self.control,
            acceleration=self.acceleration,
        )
        self.readiness = types.SimpleNamespace(
            runtime_root=self.root,
            runtime_args=(),
            comfy_version="0.36.0",
            core_revision=union2_vae.VAE_FIXED_COMMIT,
            package_versions={},
        )
        self.client = mock.Mock()
        self.control_module = types.SimpleNamespace(
            NODES=("ModelPatchLoader",),
            validate_nodes=mock.Mock(),
            validate_model=mock.Mock(),
        )
        self.bridge = bridge
        self.enterContext(mock.patch.object(bridge, "resolve_runtime_root", return_value=self.root))
        self.enterContext(mock.patch.object(bridge, "validate_request"))
        self.enterContext(mock.patch.object(bridge, "ensure_ready", return_value=self.readiness))
        self.enterContext(mock.patch.object(bridge, "prepare_media", return_value={}))
        self.enterContext(mock.patch.object(bridge, "cleanup_prepared_media"))
        self.enterContext(
            mock.patch.object(
                bridge,
                "build_workflow",
                return_value={"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}},
            )
        )
        self.enterContext(mock.patch.object(bridge, "fun_control", self.control_module))
        self.enterContext(mock.patch.object(bridge, "ComfyH3Client", return_value=self.client))

    def test_current_uses_selected_control_without_submitting(self):
        with mock.patch.object(union2_vae, "check_runtime") as runtime_check:
            record, url = handoff.export_current(self.request, self.root, "http://127.0.0.1:8188", self.root, "fast")
        runtime_check.assert_called_once_with(self.readiness, decode_mode="standard", union2=True)
        self.control_module.validate_nodes.assert_called_once_with(self.client.object_info.return_value, self.control)
        self.control_module.validate_model.assert_called_once_with(self.root, self.control)
        self.client.close.assert_called_once()
        self.bridge.build_workflow.assert_called_once()
        self.assertEqual(record["metadata"]["control"]["mode"], "v2_canny")
        self.assertTrue(url.endswith(record["token"]))

    def test_history_revalidates_recorded_control(self):
        metadata = handoff._metadata(self.request, 42, self.readiness, "fast", source="generation")
        record = store.create_snapshot(
            self.root / "input",
            {"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}},
            {},
            metadata,
        )
        video = self.root / "completed.mp4"
        video.touch()
        store.write_receipt(video, record)
        item = types.SimpleNamespace(public_id="history-id", path=video)
        with mock.patch.object(union2_vae, "check_runtime") as runtime_check:
            restored, _ = handoff.export_history("history-id", [item], self.root, "http://127.0.0.1:8188", self.root)
        runtime_check.assert_called_once_with(self.readiness, decode_mode="standard", union2=True)
        self.control_module.validate_nodes.assert_called_once_with(self.client.object_info.return_value, self.control)
        self.control_module.validate_model.assert_called_once_with(self.root, self.control)
        self.assertEqual(restored["token"], record["token"])

    def test_optional_snapshot_bug_does_not_cancel_video_generation(self):
        with mock.patch.object(handoff, "_metadata", side_effect=AttributeError("missing runtime field")):
            record, warning = handoff.capture_generation(
                self.request, {}, {}, 42, self.readiness, self.root, "fast", "job-id"
            )
        self.assertIsNone(record)
        self.assertIn("保存に失敗", warning)


class H3OrbitHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input").mkdir()
        self.request = bridge.H3Request(
            mode=bridge.MODE_ORBIT,
            prompt="Camera orbits a still subject.",
            first_frame="photo.png",
            aspect="1:1",
            quality="native",
            duration_seconds=3,
            steps=28,
            seed=42,
            orbit_strength=0.75,
        )
        self.readiness = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)
        self.graph = {"1": {"class_type": "RandomNoise", "inputs": {"noise_seed": 42}}}
        self.enterContext(mock.patch.object(bridge, "resolve_runtime_root", return_value=self.root))
        self.ready = self.enterContext(mock.patch.object(bridge, "ensure_ready", return_value=self.readiness))
        self.prepare = self.enterContext(mock.patch.object(bridge, "prepare_media", return_value={}))
        self.enterContext(mock.patch.object(bridge, "cleanup_prepared_media"))
        self.build = self.enterContext(mock.patch.object(bridge, "build_workflow", return_value=self.graph))

    def history(self):
        record = store.create_snapshot(
            self.root / "input",
            self.graph,
            {},
            handoff._metadata(self.request, 42, self.readiness, "fast", source="generation"),
        )
        video = self.root / "completed.mp4"
        video.touch()
        store.write_receipt(video, record)
        return record, types.SimpleNamespace(public_id="orbit-history-id", path=video)

    def test_current_missing_orbit_weights_rejects_before_copying_media(self):
        with self.assertRaisesRegex(bridge.H3BridgeError, "Orbit LoRAが未導入"):
            handoff.export_current(self.request, self.root, bridge.H3_SERVER_URL, self.root, "fast")
        self.ready.assert_called_once()
        self.prepare.assert_not_called()
        self.build.assert_not_called()

    def test_current_validates_orbit_runtime_and_records_adapter_identity(self):
        with mock.patch.object(bridge, "validate_orbit_runtime") as validate:
            record, url = handoff.export_current(self.request, self.root, bridge.H3_SERVER_URL, self.root, "fast")
        validate.assert_called_once_with(self.readiness)
        self.assertEqual(record["metadata"]["orbit"], bridge.orbit_metadata(self.request))
        self.assertEqual(record["metadata"]["orbit"]["strength"], 0.75)
        self.assertEqual(record["metadata"]["frames"], 73)
        self.assertTrue(url.endswith(record["token"]))

    def test_history_missing_orbit_weights_rejects_recorded_workflow(self):
        _, item = self.history()
        with self.assertRaisesRegex(bridge.H3BridgeError, "Orbit LoRAが未導入"):
            handoff.export_history(item.public_id, [item], self.root, bridge.H3_SERVER_URL, self.root)
        self.ready.assert_called_once()
        self.prepare.assert_not_called()

    def test_history_revalidates_orbit_and_preserves_saved_graph_and_metadata(self):
        original, item = self.history()
        with mock.patch.object(bridge, "validate_orbit_runtime") as validate:
            restored, _ = handoff.export_history(item.public_id, [item], self.root, bridge.H3_SERVER_URL, self.root)
        validate.assert_called_once_with(self.readiness)
        self.assertEqual(restored["prompt"], original["prompt"])
        self.assertEqual(restored["metadata"]["orbit"], bridge.orbit_metadata(self.request))
        self.build.assert_not_called()
        self.prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
