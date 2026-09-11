"""Offline contract/rollback tests; no model downloads or CUDA required."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import gradio as gr
import torch

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_images as images
from modules_forge.cd_tuner_state import TensorEditLedger
from modules_forge.minimax_h3_acceleration import FAST_VAE_PACK, H3Acceleration
from modules_forge.minimax_h3_negpip import (
    BUNDLE_BLOBS,
    NEGPIP_NODE,
    NEGPIP_PACK,
    H3NegPiP,
    custom_node_whitelist,
    install_bundled_negpip,
)
from modules_forge.minimax_h3_negpip_ui import create_negpip_controls

ROOT = Path(__file__).resolve().parents[2]
CD_PATH = ROOT / "extensions-builtin/sd-webui-cd-tuner/scripts/cdtuner.py"
IMAGE_UI_PATH = ROOT / "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"
MODELS = {name: name + ".safetensors" for name in ("fl2va", "ref2va", "clip", "vae", "audio_vae")}


def schema():
    types = {
        "model": "MODEL",
        "clip": "CLIP",
        "value_strength": "FLOAT",
        "apply_positive_weights": "BOOLEAN",
        "block_start": "INT",
        "block_end": "INT",
        "block_stride": "INT",
        "protect_text_rows": "BOOLEAN",
        "measure_attention_mass": "BOOLEAN",
    }
    return {NEGPIP_NODE: {"input": {"required": {k: [v] for k, v in types.items()}}, "output": ["MODEL", "CLIP"]}}


def import_path(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {**stubs, name: module}):
        spec.loader.exec_module(module)
    return module


class NegPiPPolicyTests(unittest.TestCase):
    def test_default_disabled_and_backward_compatible_runtime_values(self):
        defaults = H3Acceleration()
        self.assertFalse(defaults.negpip.enabled)
        self.assertEqual(H3Acceleration.from_values(defaults.values()[:8]), defaults)
        legacy = defaults.to_dict()
        legacy.pop("negpip")
        self.assertEqual(H3Acceleration.from_dict(legacy), defaults)
        self.assertEqual(H3Acceleration.from_values(()), defaults)
        with self.assertRaises(ValueError):
            H3Acceleration.from_values(defaults.values()[:15])

    def test_round_trip_nested_history_settings(self):
        option = H3Acceleration(
            negpip=H3NegPiP(
                enabled=True, value_strength=1.75, block_start=2, block_end=20, block_stride=2, protect_text_rows=True
            )
        )
        self.assertEqual(H3Acceleration.from_dict(option.to_dict()), option)
        self.assertEqual(H3Acceleration.from_values(option.values()), option)
        self.assertEqual(H3NegPiP.from_dict(option.negpip.to_dict()), option.negpip)

    def test_invalid_options_fail_before_graph_or_runtime_changes(self):
        for kwargs in (
            {"enabled": 1},
            {"apply_positive_weights": "false"},
            {"value_strength": float("nan")},
            {"value_strength": float("inf")},
            {"value_strength": True},
            {"value_strength": -1},
            {"value_strength": 9},
            {"block_start": 1.5},
            {"block_end": 1000},
            {"block_start": 5, "block_end": 4},
            {"block_stride": 0},
            {"block_stride": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                H3NegPiP(**kwargs).validate()
        with self.assertRaises(ValueError):
            H3NegPiP.from_dict({"unknown": True})
        with self.assertRaises(ValueError):
            H3Acceleration(negpip=True).validate()

    def test_integer_ui_values_are_normalized_for_comfy_inputs(self):
        option = H3NegPiP(enabled=True, block_start=2.0, block_end=12.0, block_stride=2.0)
        self.assertIsInstance(option.node_inputs()["block_start"], int)
        option.validate()

    def test_all_video_modes_wire_model_and_clip(self):
        for mode, prompt, media, kwargs in (
            (bridge.MODE_TEXT, "an apple", {}, {}),
            (bridge.MODE_KEYFRAMES, "an apple", {"first_frame": "first.png"}, {"first_frame": "first.png"}),
            (
                bridge.MODE_REFERENCES,
                "<Picture 1> an apple",
                {"images": ["ref.png"]},
                {"reference_images": ("ref.png",)},
            ),
        ):
            with self.subTest(mode=mode):
                request = bridge.H3Request(
                    mode=mode, prompt=prompt, acceleration=H3Acceleration(negpip=H3NegPiP(enabled=True)), **kwargs
                )
                graph = bridge.build_workflow(request, media, seed=42)
                self.assertEqual(graph["17"]["class_type"], NEGPIP_NODE)
                self.assertEqual(graph["17"]["inputs"]["model"], ["15", 0])
                self.assertEqual(graph["17"]["inputs"]["clip"], ["2", 0])
                self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
                self.assertEqual(graph["8"]["inputs"]["model"], ["17", 0])
                self.assertEqual(graph["5"]["inputs"]["clip"], ["17", 1])
                self.assertEqual(graph["12"]["class_type"], "VAEDecodeAudio")

    def test_disabled_graph_is_identical_even_with_nondefault_disabled_values(self):
        request = bridge.H3Request(mode=bridge.MODE_TEXT, prompt="an apple")
        expected = bridge.build_workflow(request, {}, seed=7)
        altered = replace(
            request, acceleration=H3Acceleration(negpip=H3NegPiP(value_strength=4.0, protect_text_rows=True))
        )
        self.assertEqual(bridge.build_workflow(altered, {}, seed=7), expected)
        self.assertNotIn("17", expected)
        self.assertNotIn(NEGPIP_NODE, altered.acceleration.extra_nodes())
        self.assertEqual(altered.acceleration.runtime_packs(), ())

    def test_fast_vae_and_negpip_keep_independent_graph_stages(self):
        option = H3Acceleration(decode_mode="fast", negpip=H3NegPiP(enabled=True))
        request = bridge.H3Request(mode=bridge.MODE_TEXT, prompt="an apple", acceleration=option)
        graph = bridge.build_workflow(request, {}, seed=3)
        self.assertEqual(graph["11"]["class_type"], "MiniMaxH3FastVAEDecode")
        self.assertEqual(graph["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(option.runtime_packs(), (FAST_VAE_PACK, NEGPIP_PACK))

    def test_sparse_combination_is_rejected_explicitly(self):
        for attention in ("sol", "sla"):
            with self.subTest(attention=attention), self.assertRaisesRegex(ValueError, "Sparse"):
                H3Acceleration(attention=attention, negpip=H3NegPiP(enabled=True)).validate()

    def test_schema_requires_both_outputs_and_known_input_contract(self):
        option = H3NegPiP(enabled=True)
        option.validate_nodes(schema())
        for change in ("missing_clip", "wrong_output", "new_required", "missing_node"):
            nodes = schema()
            if change == "missing_clip":
                del nodes[NEGPIP_NODE]["input"]["required"]["clip"]
            elif change == "wrong_output":
                nodes[NEGPIP_NODE]["output"] = ["MODEL"]
            elif change == "new_required":
                nodes[NEGPIP_NODE]["input"]["required"]["unhandled"] = ["STRING"]
            else:
                nodes = {}
            with self.subTest(change=change), self.assertRaises(ValueError):
                option.validate_nodes(nodes)
        H3NegPiP().validate_nodes({})

    def test_image_text_and_reference_graphs_wire_both_outputs(self):
        for mode, prompt, refs, prepared in (
            ("text", "an apple", (), {}),
            ("references", "<Picture 1> an apple", ("a.png",), {"images": ["a.png"]}),
        ):
            request = images.H3ImageRequest(prompt, mode=mode, reference_images=refs, negpip=H3NegPiP(enabled=True))
            graph = images.build_image_workflow(request, prepared, 10, models=MODELS)
            self.assertEqual(graph["negpip"]["inputs"]["model"], ["attention", 0])
            self.assertEqual(graph["negpip"]["inputs"]["clip"], ["clip", 0])
            self.assertEqual(graph["guider"]["inputs"]["model"], ["negpip", 0])
            self.assertEqual(graph["condition"]["inputs"]["clip"], ["negpip", 1])
            self.assertEqual(graph["sigmas"]["inputs"]["model"], ["negpip", 0])
            self.assertFalse(any(node["class_type"] in {"SaveVideo", "VAEDecodeAudio"} for node in graph.values()))
            plain = replace(request, negpip=H3NegPiP())
            self.assertNotIn("negpip", images.build_image_workflow(plain, prepared, 10, models=MODELS))

    def test_existing_graph_node_is_not_overwritten(self):
        graph = {"17": {"class_type": "Existing"}}
        before = copy.deepcopy(graph)
        with self.assertRaises(ValueError):
            H3NegPiP(enabled=True).apply_workflow(graph)
        self.assertEqual(graph, before)


class NegPiPRuntimeTests(unittest.TestCase):
    def test_all_launch_profiles_and_pack_combinations(self):
        for profile in ("fast", "low_ram"):
            for fast in (False, True):
                for enabled in (False, True):
                    with self.subTest(profile=profile, fast=fast, enabled=enabled):
                        option = H3Acceleration(
                            decode_mode="fast" if fast else "standard", negpip=H3NegPiP(enabled=enabled)
                        )
                        command = bridge._runtime_command(Path("python"), 8188, profile, acceleration=option)
                        self.assertIn("--disable-all-custom-nodes", command)
                        self.assertIn("--disable-api-nodes", command)
                        self.assertEqual(bridge.runtime_profile_from_args(command[1:], 8188), profile)
                        self.assertEqual(custom_node_whitelist(command), option.runtime_packs())

    def test_default_command_uses_h3_memory_policy(self):
        self.assertEqual(
            bridge._runtime_command(Path("python"), 8188),
            [
                "python",
                "main.py",
                "--listen",
                "127.0.0.1",
                "--port",
                "8188",
                "--disable-all-custom-nodes",
                "--disable-api-nodes",
                "--reserve-vram",
                "2",
                "--preview-method",
                "none",
                "--async-offload",
                "2",
            ],
        )

    def test_unknown_or_repeated_whitelist_is_rejected(self):
        base = bridge._runtime_command(Path("python"), 8188)[1:]
        for suffix in (
            ["--whitelist-custom-nodes", "evil"],
            ["--whitelist-custom-nodes", NEGPIP_PACK, "evil"],
            ["--whitelist-custom-nodes", NEGPIP_PACK, NEGPIP_PACK],
            ["--whitelist-custom-nodes", NEGPIP_PACK, "--whitelist-custom-nodes", FAST_VAE_PACK],
            ["--whitelist-custom-nodes"],
        ):
            with self.subTest(suffix=suffix):
                self.assertIsNone(bridge.runtime_profile_from_args([*base, *suffix], 8188))

    def test_exact_equals_form_is_supported_without_widening_allowlist(self):
        base = bridge._runtime_command(Path("python"), 8188)[1:]
        self.assertEqual(
            bridge.runtime_profile_from_args([*base, f"--whitelist-custom-nodes={NEGPIP_PACK}"], 8188), "fast"
        )
        self.assertIsNone(bridge.runtime_profile_from_args([*base, "--whitelist-custom-nodes=../other"], 8188))

    def test_bundled_install_is_pinned_idempotent_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("")
            (root / "models").mkdir()
            target = install_bundled_negpip(root)
            self.assertEqual({p.name for p in target.iterdir()}, set(BUNDLE_BLOBS))
            stamps = {p.name: p.stat().st_mtime_ns for p in target.iterdir()}
            self.assertEqual(install_bundled_negpip(root), target)
            self.assertEqual({p.name: p.stat().st_mtime_ns for p in target.iterdir()}, stamps)
            source = target / "minimax_h3_negpip.py"
            source.write_bytes(source.read_bytes() + b"\n# local edit\n")
            with self.assertRaises(ValueError):
                install_bundled_negpip(root)
            self.assertTrue(source.read_bytes().endswith(b"# local edit\n"))

    def test_install_refuses_linked_custom_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "runtime"
            root.mkdir()
            (root / "main.py").write_text("")
            (root / "models").mkdir()
            outside = Path(tmp) / "outside"
            outside.mkdir()
            try:
                (root / "custom_nodes").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("Symlinks unavailable")
            with self.assertRaises(ValueError):
                install_bundled_negpip(root)
            self.assertEqual(list(outside.iterdir()), [])

    def test_install_refuses_unknown_files_in_managed_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "main.py").write_text("")
            (root / "models").mkdir()
            target = install_bundled_negpip(root)
            (target / "custom.py").write_text("# user code")
            with self.assertRaises(ValueError):
                install_bundled_negpip(root)
            self.assertEqual((target / "custom.py").read_text(), "# user code")


class LedgerTests(unittest.TestCase):
    def test_restore_keeps_parameter_identity_and_dtype(self):
        for dtype in (torch.float32, torch.bfloat16):
            layer = torch.nn.Linear(4, 4).to(dtype)
            parameter = layer.weight
            original = parameter.detach().clone()
            ledger = TensorEditLedger()
            ledger.capture(layer, "weight")
            with torch.inference_mode():
                layer.weight.mul_(2)
            ledger.capture(layer, "weight")
            ledger.restore()
            self.assertIs(layer.weight, parameter)
            self.assertEqual(layer.weight.dtype, dtype)
            self.assertTrue(torch.equal(layer.weight, original))
            self.assertFalse(ledger.pending)
            ledger.restore()

    def test_original_module_is_restored_after_model_switch(self):
        old = torch.nn.Linear(2, 2)
        new = torch.nn.Linear(2, 2)
        a, b = old.weight.detach().clone(), new.weight.detach().clone()
        ledger = TensorEditLedger()
        ledger.capture(old, "weight")
        with torch.inference_mode():
            old.weight.zero_()
        ledger.restore()
        self.assertTrue(torch.equal(old.weight, a))
        self.assertTrue(torch.equal(new.weight, b))

    def test_inference_tensor_can_be_restored(self):
        with torch.inference_mode():
            module = SimpleNamespace(weight=torch.ones(2))
        ledger = TensorEditLedger()
        ledger.capture(module, "weight")
        with torch.inference_mode():
            module.weight.fill_(9)
        ledger.restore()
        self.assertTrue(torch.equal(module.weight, torch.ones(2)))


class CDTunerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = ModuleType("modules")
        callbacks = ModuleType("modules.script_callbacks")
        for name in ("CFGDenoiserParams", "CFGDenoisedParams", "AfterCFGCallbackParams"):
            setattr(callbacks, name, object)
        for name in ("on_cfg_denoiser", "on_cfg_denoised", "on_cfg_after_cfg", "on_script_unloaded"):
            setattr(callbacks, name, mock.Mock())
        scripts = ModuleType("modules.scripts")
        scripts.Script = type("ScriptBase", (), {})
        scripts.AlwaysVisible = True
        components = ModuleType("modules.ui_components")

        class AccordionInput:
            def __init__(self, value=False, **kwargs):
                self.checkbox = gr.Checkbox(value=value, label=kwargs.get("label"))
                self.accordion = gr.Accordion(label=kwargs.get("label"), open=bool(value))

            def __enter__(self):
                self.accordion.__enter__()
                return self.checkbox

            def __exit__(self, *args):
                return self.accordion.__exit__(*args)

        components.InputAccordion = AccordionInput
        components.ToolButton = gr.Button
        devices = ModuleType("modules.devices")
        devices.device, devices.dtype = torch.device("cpu"), torch.float32
        shared = ModuleType("modules.shared")
        shared.opts = SimpleNamespace(sd_model_checkpoint="test")
        networks = ModuleType("modules.extra_networks")
        networks.parse_prompts = mock.Mock(side_effect=lambda prompts: (prompts, {}))
        stubs = {
            "modules": package,
            "modules.ui": ModuleType("modules.ui"),
            "modules.scripts": scripts,
            "modules.script_callbacks": callbacks,
            "modules.ui_components": components,
            "modules.devices": devices,
            "modules.shared": shared,
            "modules.extra_networks": networks,
        }
        for name, value in stubs.items():
            if name.startswith("modules."):
                setattr(package, name.split(".", 1)[1], value)
        cls.cd = import_path("_test_aikimi_cdtuner", CD_PATH, stubs)
        cls.shared, cls.networks = shared, networks

    def setUp(self):
        class TinyUNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_blocks = torch.nn.ModuleList([torch.nn.ModuleList([torch.nn.Conv2d(4, 4, 1)])])
                self.out = torch.nn.Sequential(torch.nn.GroupNorm(1, 4), torch.nn.Identity(), torch.nn.Conv2d(4, 4, 1))

        parent = torch.nn.Module()
        parent.diffusion_model = TinyUNet()
        self.dm = parent.diffusion_model
        self.shared.sd_model = SimpleNamespace(
            is_sdxl=False,
            forge_objects_after_applying_lora=SimpleNamespace(unet=SimpleNamespace(model=parent)),
            model_config=None,
        )
        self.p = SimpleNamespace(
            all_prompts=["whole job prompt"],
            prompts=["current batch prompt"],
            batch_size=1,
            extra_generation_params={},
            refiner_switch_at=None,
            enable_hr=False,
        )
        self.script = self.cd.Script()
        self.settings = [True, 2.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, -1, -1, 0, 0, "1,1", "Horizontal", "", 2, 1, []]

    def tearDown(self):
        self.cd._cleanup_active()

    def params(self, shape=(1, 4, 8, 8)):
        return SimpleNamespace(x=torch.randn(shape), sampling_step=0, total_sampling_steps=20)

    def test_cleanup_restores_weights_and_disables_callbacks(self):
        original = self.dm.input_blocks[0][0].weight.detach().clone()
        self.script.process_batch(self.p, *self.settings)
        self.script.denoiser_callback(self.params())
        self.assertFalse(torch.equal(self.dm.input_blocks[0][0].weight, original))
        self.script.on_process_cleanup(self.p)
        self.assertTrue(torch.equal(self.dm.input_blocks[0][0].weight, original))
        self.assertFalse(self.script.active)
        self.assertFalse(self.script.activec)
        self.assertFalse(self.script._edits.pending)
        self.script.denoiser_callback(self.params())
        self.assertTrue(torch.equal(self.dm.input_blocks[0][0].weight, original))

    def test_current_batch_prompt_is_used(self):
        self.script.process_batch(self.p, *self.settings)
        self.networks.parse_prompts.assert_called_with(["current batch prompt"])

    def test_disabled_other_tab_rolls_back_previous_active_instance(self):
        original = self.dm.input_blocks[0][0].weight.detach().clone()
        self.script.process_batch(self.p, *self.settings)
        self.script.denoiser_callback(self.params())
        other = self.cd.Script()
        other.process_batch(self.p, False, *self.settings[1:])
        self.assertTrue(torch.equal(self.dm.input_blocks[0][0].weight, original))
        self.assertFalse(self.script.active)
        self.assertFalse(other.active)

    def test_callback_failure_rolls_back_partial_edits(self):
        original = self.dm.input_blocks[0][0].weight.detach().clone()
        self.script.process_batch(self.p, *self.settings)
        setter = self.cd.getset_nested_module_tensor
        writes = 0

        def fail_second_write(clone, model, path, new_tensor=None):
            nonlocal writes
            if not clone:
                writes += 1
                if writes == 2:
                    raise RuntimeError("simulated error")
            return setter(clone, model, path, new_tensor)

        with (
            mock.patch.object(self.cd, "getset_nested_module_tensor", side_effect=fail_second_write),
            self.assertRaisesRegex(RuntimeError, "simulated"),
        ):
            self.script.denoiser_callback(self.params())
        self.assertTrue(torch.equal(self.dm.input_blocks[0][0].weight, original))
        self.assertFalse(self.script.active)
        self.assertNotIn("CDT", self.p.extra_generation_params)

    def test_color_only_does_not_rewrite_model_weights(self):
        self.settings[1], self.settings[5] = 0, 1.0
        self.script.process_batch(self.p, *self.settings)
        with mock.patch.object(
            self.cd, "getset_nested_module_tensor", side_effect=AssertionError("unnecessary weight edit")
        ):
            self.script.denoiser_callback(self.params())
        self.assertFalse(self.script._edits.pending)

    def test_hires_uses_last_two_dimensions_for_5d_latents(self):
        self.settings[1], self.settings[9] = 0, 3.0
        self.p.enable_hr = True
        self.script.process_batch(self.p, *self.settings)
        self.script.denoiser_callback(self.params((1, 4, 1, 8, 8)))
        self.script.denoiser_callback(self.params((1, 4, 1, 8, 16)))
        self.assertEqual(self.script.pas, 1)
        self.assertTrue(self.script._edits.pending)

    def test_apply_once_is_per_request_not_a_persistent_disable(self):
        self.settings[-1] = ["Apply once"]
        self.script.process_batch(self.p, *self.settings)
        self.assertIn("CDT", self.p.extra_generation_params)
        self.script.postprocess_batch(self.p)
        self.script.process_batch(self.p, *self.settings)
        self.assertFalse(self.script.active)
        self.assertNotIn("CDT", self.p.extra_generation_params)
        self.assertNotIn("CDTC", self.p.extra_generation_params)
        self.script.on_process_cleanup(self.p)
        self.script.process_batch(self.p, *self.settings)
        self.assertTrue(self.script.active)

    def test_quantized_weights_are_not_corrupted(self):
        layer = torch.nn.Module()
        layer.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.int8), requires_grad=False)
        with self.assertRaisesRegex(ValueError, "量子化"):
            self.cd.getset_nested_module_tensor(False, layer, "weight", torch.zeros(2, dtype=torch.int8))
        self.assertTrue(torch.equal(layer.weight, torch.ones(2, dtype=torch.int8)))

    def test_color_map_validation_and_no_runtime_preview_allocation(self):
        with mock.patch.object(self.cd.PIL.Image, "fromarray", side_effect=AssertionError("preview allocation")):
            outer, inner = self.cd.makecells(" 1,  1 ", "Horizontal", False)
        self.assertEqual(len(outer), 1)
        self.assertEqual(len(inner[0]), 2)
        for ratio in ("0,0", "-1,1", "nan,1", "inf,1", ""):
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.cd.makecells(ratio, "Horizontal", False)
        self.assertEqual(len(self.cd._parse_colors("1,2,3", 2)), 2)
        with self.assertRaises(ValueError):
            self.cd._parse_colors("1,2,3;4,5,6", 3)
        with self.assertRaises(ValueError):
            self.cd._parse_colors("nan,2,3", 1)

    def test_gradio_controls_keep_original_parameter_contract(self):
        with gr.Blocks(analytics_enabled=False):
            a = self.cd.Script().ui(False)
            b = self.cd.Script().ui(True)
        self.assertEqual(len(a), 22)
        self.assertEqual(len(b), 22)
        self.assertFalse(a[0].value)
        self.assertFalse(b[0].value)
        self.assertNotEqual(a[11].elem_id, b[11].elem_id)

    def test_saved_infotext_round_trip_preserves_saturation_and_output_count(self):
        self.settings[14:16] = [1.5, -2.5]
        with mock.patch.object(self.cd, "vaedealer"), mock.patch.object(self.cd, "vaedealer2"):
            self.script.process_batch(self.p, *self.settings)
        with gr.Blocks(analytics_enabled=False) as ui:
            controls = self.cd.Script().ui(False)
        event = next(event for event in ui.fns.values() if event.fn.__name__ == "infotexter")
        updates = event.fn(self.p.extra_generation_params["CDT"])
        self.assertEqual(len(updates), len(event.outputs))
        self.assertEqual([update["value"] for update in updates[1:]], self.settings[:16])
        self.assertEqual(len(controls), 22)

    def test_color_only_hires_applies_at_the_final_pass_even_without_resize(self):
        self.settings[1], self.settings[5] = 0, 1.0
        self.p.enable_hr = True
        self.p.is_hr_pass = False
        self.script.process_batch(self.p, *self.settings)
        params = self.params((1, 4, 1, 8, 8))
        params.sampling_step = params.total_sampling_steps - 2
        before = params.x.clone()
        self.script.denoised_callback(params)
        self.assertTrue(torch.equal(params.x, before))
        self.p.is_hr_pass = True
        self.script.denoised_callback(params)
        self.assertFalse(torch.equal(params.x, before))

    def test_color_map_applies_each_forge_step_and_restarts_once_for_hires(self):
        self.settings[1] = 0
        self.settings[18:20] = ["1,0,0", 3]
        self.p.enable_hr = True
        self.p.is_hr_pass = False
        self.script.process_batch(self.p, *self.settings)
        for hires, shape in ((False, (1, 4, 1, 8, 8)), (True, (1, 4, 1, 16, 16))):
            self.p.is_hr_pass = hires
            for step in range(4):
                with self.subTest(hires=hires, step=step):
                    params = self.params(shape)
                    params.sampling_step = step
                    before = params.x.clone()
                    self.script.denoiser_callback(params)
                    self.assertEqual(torch.equal(params.x, before), step >= 3)


class NegPiPUITests(unittest.TestCase):
    def test_shared_control_values_match_policy_order(self):
        with gr.Blocks(analytics_enabled=False):
            controls = create_negpip_controls(prefix="test")
        self.assertEqual(H3NegPiP.from_values([c.value for c in controls]), H3NegPiP())

    def test_video_controls_match_runtime_option_arity(self):
        from modules_forge.minimax_h3_acceleration_ui import create_acceleration_controls

        with gr.Blocks(analytics_enabled=False):
            duration = gr.Slider(5, 15, value=5, step=1)
            controls, buttons = create_acceleration_controls(duration)
        self.assertEqual(len(controls), len(H3Acceleration().values()))
        self.assertEqual(len(buttons), 3)
        self.assertEqual(H3Acceleration.from_values([c.value for c in controls]), H3Acceleration())

    def test_image_ui_binds_negpip_to_generation_and_runtime_controls(self):
        package = ModuleType("modules")
        callbacks = ModuleType("modules.script_callbacks")
        callbacks.on_ui_tabs = mock.Mock()
        paths = ModuleType("modules.paths")
        paths.data_path = paths.script_path = str(ROOT)
        package.script_callbacks = callbacks
        module = import_path(
            "_test_h3_negpip_image_ui",
            IMAGE_UI_PATH,
            {"modules": package, "modules.script_callbacks": callbacks, "modules.paths": paths},
        )
        with mock.patch.object(module, "_initial_runtime", return_value=""):
            tabs = module.on_ui_tabs()
        blocks = tabs[0][0]
        functions = {entry.fn.__name__: entry for entry in blocks.fns.values() if entry.fn is not None}
        self.assertEqual(len(functions["_generate"].inputs), 21)
        self.assertEqual(len(functions["_runtime_action"].inputs), 11)
        self.assertEqual(len(functions["_restart"].inputs), 11)
        request = module._request(
            "text", "apple", None, 768, 768, 20, "-1", "simple", 0, False, *H3NegPiP(enabled=True).values()
        )
        self.assertTrue(request.negpip.enabled)


if __name__ == "__main__":
    unittest.main()


class AdditionalRollbackTests(unittest.TestCase):
    def test_stop_step_restores_diffusion_without_undoing_vae(self):
        diffusion, vae = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        before_diffusion = diffusion.weight.detach().clone()
        before_vae = vae.weight.detach().clone()
        ledger = TensorEditLedger()
        ledger.capture(diffusion, "weight", group="diffusion")
        ledger.capture(vae, "weight", group="vae")
        with torch.inference_mode():
            diffusion.weight.add_(1)
            vae.weight.add_(2)
        ledger.restore(group="diffusion")
        self.assertTrue(torch.equal(diffusion.weight, before_diffusion))
        self.assertFalse(torch.equal(vae.weight, before_vae))
        self.assertTrue(ledger.pending)
        ledger.restore()
        self.assertTrue(torch.equal(vae.weight, before_vae))
        self.assertFalse(ledger.pending)

    def test_bundle_has_explicit_windows_line_ending_policy(self):
        from modules_forge.minimax_h3_negpip import BUNDLE_ROOT

        policy = (BUNDLE_ROOT / ".gitattributes").read_text().splitlines()
        for name in BUNDLE_BLOBS:
            self.assertIn(f"{name} -text", policy)
