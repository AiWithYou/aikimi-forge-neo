"""W4A8 model selection, pinned download and per-mode readiness contracts."""

import unittest
from dataclasses import replace

from modules_forge.minimax_h3_acceleration import W4A8_MODELS, H3Acceleration
from tools.setup_minimax_h3_w4a8 import model_profile
from tools.tests import test_h3_acceleration as h3_tests


class H3W4A8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = h3_tests.load_bridge()

    def test_each_mode_selects_its_packed_weight_and_preserves_sampling(self):
        option = H3Acceleration(model_variant="w4a8")
        for mode in ("text", "keyframes", "references"):
            request = self.bridge.H3Request(
                mode=mode,
                prompt="<Picture 1> test" if mode == "references" else "test",
                acceleration=option,
                first_frame="image.png" if mode == "keyframes" else None,
                reference_images=("image.png",) if mode == "references" else (),
            )
            graph = self.bridge.build_workflow(request, {"first_frame": "image.png", "images": ["image.png"]}, seed=42)
            self.assertEqual(
                graph["1"]["inputs"]["unet_name"], W4A8_MODELS["Ref2VA" if mode == "references" else "FL2VA"]
            )
            self.assertEqual(graph["3"]["inputs"]["vae_name"], self.bridge.H3_VIDEO_VAE)
            self.assertEqual(graph["8"]["inputs"]["steps"], 20)
            self.assertEqual(graph["7"]["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(H3Acceleration.from_dict(option.to_dict()), option)
        self.assertEqual(H3Acceleration.from_values(option.values()), option)

    def test_ref_only_install_is_ready_and_text_generation_is_rejected(self):
        option = H3Acceleration(model_variant="w4a8")
        readiness = h3_tests.H3AccelerationTests.ready(self, option)
        readiness.model_files["FL2VA"] = False
        readiness.server_model_files["FL2VA"] = False
        self.assertTrue(readiness.ready_for_ref2va)
        self.assertFalse(readiness.ready_for_fl2va)
        self.bridge.validate_readiness(readiness)
        request = self.bridge.H3Request(mode="references", prompt="test", acceleration=option)
        self.bridge._validate_request_runtime_constraints(request, readiness, "fast")
        with self.assertRaisesRegex(self.bridge.H3BridgeError, "FL2VA"):
            self.bridge._validate_request_runtime_constraints(replace(request, mode="text"), readiness, "fast")

    def test_fl_only_install_does_not_claim_reference_readiness(self):
        readiness = h3_tests.H3AccelerationTests.ready(self, H3Acceleration(model_variant="w4a8"))
        readiness.model_files["Ref2VA"] = False
        self.assertTrue(readiness.ready_for_fl2va)
        self.assertFalse(readiness.ready_for_ref2va)

    def test_w4a8_requires_verified_kitchen_version_without_changing_standard_gate(self):
        readiness = h3_tests.H3AccelerationTests.ready(self, H3Acceleration())
        readiness.package_versions["comfy-kitchen"] = "0.2.30"
        self.bridge.validate_readiness(readiness)
        readiness = replace(readiness, acceleration=H3Acceleration(model_variant="w4a8"))
        with self.assertRaisesRegex(self.bridge.H3BridgeError, "W4A8"):
            self.bridge.validate_readiness(readiness)

    def test_optional_downloads_match_selection_and_pin_hashes(self):
        all_files = model_profile().artifacts
        self.assertEqual({x.relative_path.split("/")[-1] for x in all_files}, set(W4A8_MODELS.values()))
        self.assertEqual(sum(x.size for x in all_files), 24311515056)
        for mode in ("fl2va", "ref2va"):
            profile = model_profile(mode)
            self.assertEqual(len(profile.artifacts), 1)
            artifact = profile.artifacts[0]
            self.assertEqual(artifact.artifact_id, mode)
            self.assertRegex(artifact.sha256, "^[0-9a-f]{64}$")
            self.assertIn("/resolve/f4cac997f880e93cf6940af61ee8d58ef31ff7f3/", artifact.url)
        with self.assertRaises(ValueError):
            model_profile("invalid")


if __name__ == "__main__":
    unittest.main()
