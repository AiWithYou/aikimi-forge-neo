"""CPU-only regression coverage for the opt-in H3 acceleration integration."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules_forge.minimax_h3_acceleration import (  # noqa: E402
    FAST_VAE_NODE, FAST_VAE_PACK, INT8_VIDEO_VAE, SPARSE_NODE, TURBO_MODEL,
    H3Acceleration,
)


def load_bridge():
    name = "_h3_acceleration_bridge_test"
    spec = importlib.util.spec_from_file_location(name, ROOT / "modules_forge/minimax_h3_bridge.py")
    module = importlib.util.module_from_spec(spec)
    # Isolate the application environment, not the real HTTP/serialization code.
    redaction = ModuleType("modules.aikimi_security.redaction")
    redaction.sanitized_subprocess_environment = lambda values: dict(values)
    with patch.dict(sys.modules, {name: module, redaction.__name__: redaction}):
        spec.loader.exec_module(module)
    return module


def optional_nodes():
    return {
        FAST_VAE_NODE: {"input": {"required": {"samples": ["LATENT"], "vae": ["VAE"], "tile_batch_size": ["INT", {"min": 1, "max": 8}]}}},
        SPARSE_NODE: {"input": {"required": {
            "model": ["MODEL"],
            "selection": ["COMFY_DYNAMICCOMBO_V3", {"options": [
                {"key": "Sol-Attn (adaptive tau)", "inputs": {"required": {"tau": ["FLOAT"]}}},
                {"key": "top-k (SLA)", "inputs": {"required": {"keep_percent": ["FLOAT"]}}},
            ]}],
            "start_percent": ["FLOAT"], "end_percent": ["FLOAT"],
            "dense_blocks": ["STRING"], "min_tokens": ["INT"], "extra_tokens": ["INT"],
            "sink_conditioning": [["exact_kv", "exact_kv_and_rows", "off"]], "verbose": ["BOOLEAN"],
        }}},
    }


class H3AccelerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bridge = load_bridge()

    def request(self, **kwargs):
        return self.bridge.H3Request(mode="text", prompt="test prompt", **kwargs)

    def ready(self, option=H3Acceleration()):
        return self.bridge.RuntimeReadiness(
            runtime_root=Path("."), server_url="http://127.0.0.1:8188", connected=True,
            ck_attention_available=True, h3_core_optimized=True, core_revision="test-revision",
            package_versions={"comfy-kitchen": "0.2.33"}, runtime_profile="fast",
            runtime_args=tuple(self.bridge._runtime_command(Path("python"), 8188, acceleration=option)[1:]),
            model_files={name: True for name in self.bridge.MODEL_FILES},
            server_model_files={name: True for name in self.bridge.MODEL_FILES},
            acceleration=option, node_schemas=optional_nodes(),
        )

    def test_default_graph_keeps_standard_models_and_all_sampling_defaults(self):
        graph = self.bridge.build_workflow(self.request(), {}, seed=42)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], self.bridge.H3_FL_MODEL)
        self.assertEqual(graph["3"]["inputs"]["vae_name"], self.bridge.H3_VIDEO_VAE)
        self.assertEqual(graph["11"]["class_type"], "VAEDecode")
        self.assertEqual(graph["9"]["inputs"]["model"], ["15", 0])
        self.assertNotIn("16", graph)
        self.assertEqual(graph["8"]["inputs"]["steps"], 20)
        self.assertEqual(graph["7"]["inputs"]["sampler_name"], "res_multistep")

    def test_all_four_axes_are_independent(self):
        for model in ("base", "fused_turbo"):
            for vae in ("fp16", "int8"):
                for decode in ("standard", "fast"):
                    for attention in ("dense", "sol", "sla"):
                        with self.subTest(model=model, vae=vae, decode=decode, attention=attention):
                            option = H3Acceleration(model, vae, decode, 3, attention)
                            graph = self.bridge.build_workflow(self.request(acceleration=option), {}, seed=42)
                            self.assertEqual(graph["1"]["inputs"]["unet_name"], TURBO_MODEL if model == "fused_turbo" else self.bridge.H3_FL_MODEL)
                            self.assertEqual(graph["3"]["inputs"]["vae_name"], INT8_VIDEO_VAE if vae == "int8" else self.bridge.H3_VIDEO_VAE)
                            self.assertEqual(graph["11"]["class_type"], FAST_VAE_NODE if decode == "fast" else "VAEDecode")
                            self.assertEqual("16" in graph, attention != "dense")
                            self.assertEqual(graph["8"]["inputs"]["steps"], 20)  # never silently change steps

    def test_sparse_uses_flat_dynamic_combo_after_kitchen_and_protects_audio(self):
        for mode, key, value in (("sol", "tau", 1.3), ("sla", "keep_percent", 10.0)):
            graph = self.bridge.build_workflow(self.request(acceleration=H3Acceleration(attention=mode)), {}, seed=42)
            inputs = graph["16"]["inputs"]
            self.assertIsInstance(inputs["selection"], str)
            self.assertEqual(inputs[f"selection.{key}"], value)
            self.assertEqual(inputs["model"], ["15", 0])
            self.assertEqual(graph["9"]["inputs"]["model"], ["16", 0])
            self.assertEqual(inputs["sink_conditioning"], "exact_kv_and_rows")
            self.assertEqual(inputs["extra_tokens"], 256)
            self.assertEqual(inputs["min_tokens"], 12288)

    def test_reference_mode_uses_the_chosen_model_without_changing_media_links(self):
        request = self.bridge.H3Request(mode="references", prompt="<Picture 1>", reference_images=("photo.png",), acceleration=H3Acceleration(model_variant="fused_turbo"))
        graph = self.bridge.build_workflow(request, {"images": ["forge_h3/photo.png"]}, seed=1)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], TURBO_MODEL)
        self.assertEqual(graph["5"]["class_type"], "MiniMaxH3ReferenceToVideo")
        self.assertEqual(graph["5"]["inputs"]["ref_images.ref_image_0"], ["20", 0])

    def test_selected_models_are_checked_without_requiring_unused_base_files(self):
        option = H3Acceleration(model_variant="fused_turbo", video_vae="int8")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            files = option.model_files(self.bridge.MODEL_FILES)
            for directory, filename in files.values():
                path = root / "models" / directory / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            self.assertTrue(all(self.bridge.model_file_status(root, option).values()))
            self.assertFalse(self.bridge.model_file_status(root)["FL2VA"])
            nodes = {loader: {"input": {"required": {key: [[file for folder, file in files.values() if folder == directory]]}}}
                     for loader, key, directory in (("UNETLoader", "unet_name", "diffusion_models"), ("CLIPLoader", "clip_name", "text_encoders"), ("VAELoader", "vae_name", "vae"))}
            self.assertTrue(all(self.bridge.server_model_file_status(nodes, option).values()))

    def test_only_explicit_fast_vae_allows_exactly_one_custom_pack(self):
        for profile in ("fast", "low_ram"):
            base = self.bridge._runtime_command(Path("python"), 8188, profile)[1:]
            self.assertNotIn("--whitelist-custom-nodes", base)
            fast = self.bridge._runtime_command(Path("python"), 8188, profile, H3Acceleration(decode_mode="fast"))[1:]
            self.assertIn("--disable-all-custom-nodes", fast)
            self.assertIn("--disable-api-nodes", fast)
            self.assertEqual(fast[-2:], ["--whitelist-custom-nodes", FAST_VAE_PACK])
            self.assertEqual(self.bridge.runtime_profile_from_args(fast, 8188), profile)
            for tail in (["another-pack"], ["--whitelist-custom-nodes", FAST_VAE_PACK], ["--enable-manager"]):
                self.assertIsNone(self.bridge.runtime_profile_from_args(fast + tail, 8188))
            self.assertIsNone(self.bridge.runtime_profile_from_args(base + ["--whitelist-custom-nodes", "another-pack"], 8188))

    def test_incompatible_node_kitchen_or_whitelist_is_rejected(self):
        option = H3Acceleration(decode_mode="fast", attention="sla")
        ready = self.ready(option)
        self.bridge.validate_readiness(ready)
        for changed in (
            replace(ready, node_schemas={}),
            replace(ready, package_versions={"comfy-kitchen": "0.2.30"}),
            replace(ready, runtime_args=self.ready().runtime_args),
            replace(ready, h3_core_optimized=False),
        ):
            with self.assertRaises(self.bridge.H3BridgeError):
                self.bridge.validate_readiness(changed)
        self.bridge.validate_readiness(replace(self.ready(), node_schemas={}))

    def test_policy_rejects_nonfinite_and_unrecognized_choices(self):
        for changed in ({"tile_batch_size": 0}, {"tile_batch_size": 2.5}, {"tile_batch_size": True}, {"sparse_tau": float("nan")}, {"sparse_keep_percent": float("inf")}, {"attention": "vsa"}, {"decode_mode": "other"}, {"model_variant": "../../model"}, {"unexpected": 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                H3Acceleration.from_dict(changed)

    def test_mirror_and_history_roundtrip_and_legacy_defaults(self):
        from tools.tests.media_fixtures import write_video

        option = H3Acceleration("fused_turbo", "int8", "fast", 2, "sla", 1.5, 15, 0.25)
        request = self.request(acceleration=option, steps=4)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source.mp4"
            write_video(source, width=864, height=480, frames=124)
            output = root / "output"
            target = self.bridge.mirror_result(source, output, request, "job-test", 42, self.ready(option))
            items = self.bridge.list_history(None, output)
            restored = self.bridge.load_history_request(items[0].public_id, items, output)
            self.assertEqual(restored.acceleration, option)
            self.assertEqual(restored.steps, 4)
            metadata = json.loads(target.with_suffix('.json').read_text())
            self.assertEqual(metadata["selected_models"]["Video VAE"], INT8_VIDEO_VAE)
            metadata.pop("acceleration")
            metadata["schema_version"] = 1
            target.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertEqual(self.bridge.load_history_request(items[0].public_id, items, output).acceleration, H3Acceleration())

    def test_generation_passes_same_options_through_both_preflights(self):
        option = H3Acceleration(model_variant="fused_turbo", video_vae="int8")
        ready = self.ready(option)
        client = Mock()
        client.submit.return_value = "test-job"
        client.job.return_value = {"status": "cancelled"}
        with (
            patch.object(self.bridge, "ensure_ready", return_value=ready) as ensure,
            patch.object(self.bridge, "prepare_media", return_value={}),
            patch.object(self.bridge, "cleanup_stale_prepared_media"),
            patch.object(self.bridge, "cleanup_prepared_media"),
            patch.object(self.bridge, "ComfyH3Client", return_value=client),
        ):
            with self.assertRaises(self.bridge.H3GenerationCancelled):
                list(self.bridge.run_generation(self.request(acceleration=option), Path('.'), ready.server_url, Path('.'), Path('.')))
            self.assertEqual(ensure.call_count, 2)
            self.assertTrue(all(call.kwargs["acceleration"] == option for call in ensure.call_args_list))
            self.assertEqual(client.submit.call_args.args[0]["1"]["inputs"]["unet_name"], TURBO_MODEL)
        self.assertEqual(self.bridge._active_generation_count(), 0)


if __name__ == "__main__":
    unittest.main()
