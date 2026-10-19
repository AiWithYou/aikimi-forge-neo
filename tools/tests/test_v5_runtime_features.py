"""Public controls must reach native sampling and both H3 VAE directions."""

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.moss_image import core
from modules_forge.moss_image.api import public_result
from modules_forge.moss_image.inference import ComfyRuntime
from tools.tests import test_moss_image_ui as ui_tests

ROOT = Path(__file__).resolve().parents[2]


class MossRopeTests(unittest.TestCase):
    def test_request_defaults_bounds_and_private_metadata(self):
        self.assertEqual(core.validate_request({"prompt": "cat"})["rope"], {"scale_x": 1.0, "scale_y": 1.0})
        for value in (None, [], {"scale_x": 0}, {"scale_y": 4.01}, {"scale_x": True},
                      {"scale_x": float("nan")}, {"scale_x": float("inf")}, {"private_path": "secret"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                core.validate_request({"prompt": "cat", "rope": value})
        request = core.validate_request({"prompt": "cat", "rope": {"scale_x": 0.25, "scale_y": 4}})
        self.assertEqual(core.validate_request(request, internal=True), request)
        result = public_result({**request, "rope": {**request["rope"], "private_path": "secret"}})
        self.assertEqual(result["rope"], request["rope"])
        result = public_result({"sampling_passes": [{"rope": {"scale_x": .5, "scale_y": .75, "private_path": "secret"}}]})
        self.assertEqual(result["sampling_passes"][0]["rope"], {"scale_x": .5, "scale_y": .75})

    def test_scales_reach_sampler_and_do_not_leak_between_jobs(self):
        import torch

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.torch, runtime.last_metadata = torch, {}
        original = SimpleNamespace(model_options={"transformer_options": {"other": 7}})
        patched = SimpleNamespace(model_options={"transformer_options": {"other": 7}}, get_model_object=lambda key: None)
        runtime.advanced = SimpleNamespace(ModelSamplingSD3=lambda: SimpleNamespace(patch=lambda *a, **k: (patched,)))
        runtime.samplers = SimpleNamespace(calculate_sigmas=lambda *a: torch.tensor([1., 0.]), sampler_object=lambda key: key)
        runtime.sample_module = SimpleNamespace(sample_custom=lambda *a, **k: a[7])
        request = core.validate_request({"prompt": "cat", "steps": 1, "width": 512, "height": 512,
                                         "skip_layer": False, "rope": {"scale_x": .75, "scale_y": .5}})
        runtime.sample(original, [], [], request, lambda *a: None, lambda: None)
        self.assertEqual(patched.model_options["transformer_options"]["rope_options"], request["rope"])
        self.assertEqual(patched.model_options["transformer_options"]["other"], 7)
        self.assertNotIn("rope_options", original.model_options["transformer_options"])

    def test_ui_passes_both_axes_without_changing_dimensions_or_hires(self):
        screen = ui_tests.UITests().load_screen()
        with patch.object(screen.SERVICE, "start", return_value="job") as start:
            screen.start({}, "cat", "", 1024, 768, 8, 3, 5, 7, False, rope_x=.75, rope_y=.5,
                         hires_enabled=True, hires_scale=1.5)
        request = start.call_args.args[0]
        self.assertEqual(request["rope"], {"scale_x": .75, "scale_y": .5})
        self.assertEqual((request["width"], request["height"]), (1536, 1152))
        blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        ids = {component.elem_id for component in blocks.blocks.values()}
        self.assertTrue({"moss-image-rope-x", "moss-image-rope-y"} <= ids)

    def test_reference_positions_use_new_rope_axes_and_keep_reference_identity(self):
        import torch

        tree = ast.parse((ROOT / "vendor/moss_image_ostris_edit.py").read_text("utf-8"))
        funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_pe", "_image_ids", "_text_ids"}]
        calls = []

        def rope(positions, dimension, theta):
            calls.append((positions.clone(), dimension, theta))
            return torch.zeros(*positions.shape, dimension // 2, 2, 2)

        namespace = {"torch": torch, "comfy": SimpleNamespace(ldm=SimpleNamespace(flux=SimpleNamespace(math=SimpleNamespace(rope=rope))))}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "reference-rope", "exec"), namespace)  # noqa: S102
        ids = torch.cat((namespace["_image_ids"](0, 3, 2, 2, "cpu"), namespace["_text_ids"](3, "cpu"),
                         namespace["_image_ids"](1, 3, 2, 2, "cpu")))
        namespace["_pe"](SimpleNamespace(rope_axes_dim=[32] * 4), ids, {"rope_options": {"scale_x": .5, "scale_y": .75}})
        self.assertEqual(len(calls), 4)
        self.assertTrue(torch.equal(calls[0][0][0], ids[:, 0]))
        self.assertEqual(calls[0][2], 10000)
        self.assertEqual(calls[1][2], 10000)
        self.assertGreater(calls[2][2], 10000)
        self.assertGreater(calls[3][2], calls[2][2])


class H3LightVAETests(unittest.TestCase):
    def test_lightvae_graphs_keep_the_numeric_ids_required_for_comfy_handoff(self):
        from modules_forge.minimax_h3_bridge import H3Request, build_workflow
        from modules_forge.minimax_h3_handoff_store import validate_prompt

        for variant in ("light_int8", "light_fp32", "lynnreal"):
            request = H3Request(mode="text", prompt="A quiet room", acceleration=H3Acceleration(video_vae=variant))
            validate_prompt(build_workflow(request, {}, seed=42))

    def test_all_download_dry_run_accepts_shared_encoder_without_writes(self):
        from tools import setup_h3_lightvae

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "not-created"
            with patch.object(setup_h3_lightvae, "configured_model_root", return_value=target), patch.object(setup_h3_lightvae, "emit"):
                self.assertEqual(setup_h3_lightvae.main(["--all", "--dry-run"]), 0)
            self.assertFalse(target.exists())

    def test_light_core_gate_and_missing_decoder_block_readiness(self):
        from modules_forge import minimax_h3_bridge as bridge

        option = H3Acceleration(video_vae="light_int8")
        self.assertEqual(bridge.required_core_commit(option), "1d9cec9e8cab32f26360254031c967d25e3fbc6b")
        files = {name: True for name in option.model_files(bridge.MODEL_FILES)}
        ready = bridge.RuntimeReadiness(runtime_root=None, server_url="http://127.0.0.1:8189",
                                        connected=True, ck_attention_available=True, h3_core_optimized=True,
                                        model_files=dict(files), server_model_files=dict(files), acceleration=option)
        self.assertTrue(ready.ready_for_fl2va)
        ready.model_files["Video Decoder VAE"] = False
        self.assertFalse(ready.ready_for_fl2va)
        ready.model_files["Video Decoder VAE"] = True
        ready.server_model_files["Video Encoder VAE"] = False
        self.assertFalse(ready.ready_for_ref2va)

    def test_separate_encoder_decoder_and_readiness_require_both(self):
        from modules_forge.minimax_h3_bridge import MODEL_FILES

        for variant in ("light_int8", "light_fp32", "lynnreal"):
            option = H3Acceleration(video_vae=variant)
            option.validate()
            files = option.model_files(MODEL_FILES)
            graph = {key: {"inputs": {}} for key in ("1", "3", "11", "17")}
            graph["17"]["inputs"]["retained"] = True
            option.apply_workflow(graph, MODEL_FILES, "references")
            if variant != "lynnreal":
                self.assertIn("Video Decoder VAE", files)
                self.assertEqual(graph["3"]["inputs"]["vae_name"], files["Video Encoder VAE"][1])
                self.assertEqual(graph["18"]["inputs"]["vae_name"], files["Video Decoder VAE"][1])
                self.assertEqual(graph["11"]["inputs"]["vae"], ["18", 0])
                self.assertTrue(graph["17"]["inputs"]["retained"])
                option.apply_workflow(graph, MODEL_FILES, "text")
                self.assertEqual(graph["3"]["inputs"]["vae_name"], MODEL_FILES["Video VAE"][1])
            with self.assertRaises(ValueError):
                H3Acceleration(video_vae=variant, decode_mode="fast").validate()

    def test_optional_download_profiles_have_fixed_integrity_and_no_implicit_core_change(self):
        from tools import setup_h3_lightvae

        for variant in ("light_int8", "light_fp32", "lynnreal"):
            profile = setup_h3_lightvae.profile(variant)
            self.assertTrue(profile.artifacts)
            for item in profile.artifacts:
                self.assertEqual(len(item.sha256), 64)
                self.assertGreater(item.size, 0)
                self.assertTrue(item.relative_path.startswith("vae/"))
                self.assertRegex(item.url, r"/resolve/[0-9a-f]{40}/")

    def test_runtime_pins_and_dependency_locks_cover_the_new_native_features(self):
        moss = json.loads((ROOT / "tools/moss_image_manifest.json").read_text("utf-8"))
        h3 = json.loads((ROOT / "tools/minimax_h3_runtime_manifest.json").read_text("utf-8"))
        self.assertEqual(moss["runtime"]["comfy_revision"], "b9b290e43f1eb379946dfeb013faf9bbefa1a618")
        self.assertEqual(h3["comfy_revision"], moss["runtime"]["comfy_revision"])
        for name in ("moss-image", "minimax-h3"):
            self.assertIn("comfy-kitchen==0.2.39", (ROOT / f"tools/requirements-{name}.lock").read_text("utf-8"))
        h3_lock = (ROOT / "tools/requirements-minimax-h3.lock").read_text("utf-8")
        self.assertIn("torch==2.13.0+cu130", h3_lock)
        self.assertNotIn("\ntorchaudio==", h3_lock)


if __name__ == "__main__":
    unittest.main()
