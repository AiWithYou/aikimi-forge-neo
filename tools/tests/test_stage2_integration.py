import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def load(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageTwoTests(unittest.TestCase):
    def test_degrid_alpha_and_single_pixel_on_cpu(self):
        from modules import shared  # noqa: F401 - normal Forge import order

        mod = load("scripts/degrid.py")
        for size in (1, 16):
            rgba = np.arange(size * size * 4, dtype=np.uint8).reshape(size, size, 4)
            pp = SimpleNamespace(image=Image.fromarray(rgba))
            with patch("backend.memory_management.get_torch_device", return_value=torch.device("cpu")):
                mod.ScriptPostprocessingDegrid().process(pp, True, 0)
            np.testing.assert_array_equal(np.asarray(pp.image)[:, :, 3], rgba[:, :, 3])

    def test_rgba_loader_and_infotext(self):
        self.assertIn('config["conv_out_channels"]', (ROOT / "backend/loader.py").read_text("utf8"))
        self.assertIn('img.mode not in ("RGB", "RGBA")', (ROOT / "modules/infotext_utils.py").read_text("utf8"))
        self.assertIn('img2img_background_color', (ROOT / "modules/shared_options.py").read_text("utf8"))

    def test_sparse_attention_cpu(self):
        import gradio as gr

        sparse = load("extensions-builtin/sd_forge_sparse/scripts/sparse.py")
        q = torch.ones(1, 2, 1, 128)
        self.assertIn("CUDA", sparse.SparseAttnPatch.ineligible(q, q, q, 128))
        with patch("modules.shared.opts", SimpleNamespace(), create=True), gr.Blocks():
            controls = sparse.SparseAttentionForForge().ui()
        self.assertIsInstance(controls[2], gr.Slider)
        self.assertIsInstance(controls[3], gr.Slider)

    def test_lora_schedule(self):
        from unittest.mock import Mock

        with patch.dict("sys.modules", networks=SimpleNamespace(available_network_aliases={})):
            mod = load("extensions-builtin/sd_forge_lora_schedule/scripts/control.py")
        script = mod.LoRAControl()
        schedule = script._parse_schedule("0@0.25, 1@0.75")
        for t, expected in ((0, 0), (0.5, 0.5), (1, 1)):
            self.assertEqual(mod.schedule_weight(schedule, t), expected)
        with self.assertRaises(ValueError):
            script._parse_schedule("1@0.5,2@0.5")
        script.postprocess(Mock(), Mock())

    def test_schedule_batch_and_cleanup(self):
        aliases = {"a": SimpleNamespace(filename="a.safetensors")}
        with patch.dict("sys.modules", networks=SimpleNamespace(available_network_aliases=aliases)):
            mod = load("extensions-builtin/sd_forge_lora_schedule/scripts/control.py")
        script = mod.LoRAControl()
        tag = "<lora:a:[0@0,1@1]>"
        with patch.object(mod.dynamic_args, "online_lora", True):
            for original, expected in (([tag, "x" + tag], ["<lora:a:1>", "x<lora:a:1>"]),
                                       ([tag, "x"], ["", "x"])):
                p = SimpleNamespace(prompt=original)
                try:
                    script.before_process(p)
                    self.assertEqual(p.prompt, expected)
                    self.assertEqual(bool(script.mapping), expected[0] != "")
                finally:
                    script.on_process_cleanup(p)
                self.assertIs(p.prompt, original)

    def test_extension_directory_and_access_policy(self):
        from modules import ui_extensions

        with patch.object(ui_extensions.shared, "cmd_opts", SimpleNamespace(disable_extension_access=True)):
            with self.assertRaises(PermissionError):
                ui_extensions.check_access()
        self.assertEqual(ui_extensions.extension_directory_name("My Extension"), "My Extension")
        with self.assertRaises(ValueError):
            ui_extensions.extension_directory_name("category/example")

    def test_xyz_schedule_label(self):
        import ast
        import re

        tree = ast.parse((ROOT / "scripts/xyz_grid.py").read_text("utf8"))
        tree.body = [node for node in tree.body if getattr(node, "name", "") == "format_lora_label"]
        context = {"re": re, "StableDiffusionProcessing": object}
        exec(compile(tree, "xyz_grid.py", "exec"), context)  # noqa: S102 - isolate the local formatter from UI initialization
        self.assertEqual(context["format_lora_label"](None, None, "<lora:a:[0@0,1@1]> <lora:b:0.12501>"),
                         "a:[0@0,1@1], b:0.125")

    def test_reference_rounding(self):
        from backend.misc.image_resize import resize_reference

        with patch("modules.shared.opts", SimpleNamespace(reference_max_side=0)):
            x = torch.zeros(1, 3, 33, 62)
            self.assertEqual(resize_reference(x, 16).shape[-2:], (32, 64))

    def test_triton_cpu_fallback(self):
        from modules_forge.packages.gguf.kquant_triton import try_dequantize

        self.assertIsNone(try_dequantize(None, torch.ones(2)))
        self.assertFalse((ROOT / "modules_forge/packages/gguf/gguf_reader.py").exists())

    def test_preset_choices(self):
        from modules_forge.presets import PresetArch

        self.assertEqual(PresetArch.choices(disabled=["krea"], sort=True), ["anima", "xl"])

    def test_spectrum_gradio_entry_point(self):
        import sys
        from collections import namedtuple

        import gradio as gr

        extension = ROOT / "extensions-builtin/sd_forge_spectrum"
        with patch.object(sys, "path", [str(extension), *sys.path]):
            from lib_spectrum.presets import PresetManager

            infotext = SimpleNamespace(PasteField=namedtuple("PasteField", ["component", "label"]))
            with patch.dict(sys.modules, {"modules.infotext_utils": infotext}), patch.object(PresetManager, "presets", {}):
                mod = load("extensions-builtin/sd_forge_spectrum/scripts/spectrum.py")
                with patch("modules.shared.opts", SimpleNamespace(), create=True), gr.Blocks():
                    controls = mod.SpectrumForForge().ui()
        self.assertEqual(len(controls), 8)

    def test_spectrum_window(self):
        mod = load("extensions-builtin/sd_forge_spectrum/lib_spectrum/forecaster.py")
        model = mod.ChebyshevForecaster(1, 3, 0.01, 5)
        for step in range(5):
            model.update(step, torch.tensor([1., 2.]))
        self.assertEqual(model.t_buf.numel(), 3)
        self.assertTrue(torch.isfinite(model.predict(5)).all())

    def test_moss_slg_request_bounds(self):
        from modules_forge.moss_image.core import MossImageError, validate_request

        settings = {"layers": [22, 24], "start": 0.2, "end": 0.8}
        self.assertEqual(validate_request({"prompt": "tree", "slg": settings})["slg"], settings)
        for invalid in (dict(settings, end=0.1), dict(settings, layers=[42]), dict(settings, start=-0.1)):
            with self.assertRaises(MossImageError):
                validate_request({"prompt": "tree", "slg": invalid})

    def test_slg_patch_lifetime(self):
        from unittest.mock import Mock

        settings = SimpleNamespace(slg_enable=True, slg_single_layers="22", slg_start_percent=0, slg_end_percent=1)
        with patch("modules.shared.opts", settings, create=True):
            mod = load("modules/processing_scripts/slg.py")
        script = mod.SkipLayerGuidance()
        original = Mock()
        p = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=original)),
                            extra_generation_params={})
        with patch.object(mod.dynamic_args, "moss", False, create=True):
            script.process_before_every_sampling(p)
        self.assertIs(p.sd_model.forge_objects.unet, original)
        with patch.object(mod.dynamic_args, "moss", True, create=True):
            script.process_before_every_sampling(p)
        self.assertIs(p.sd_model.forge_objects.unet, original.clone.return_value)
        script.postprocess(p, None)
        self.assertIs(p.sd_model.forge_objects.unet, original)

    def test_scheduled_weight_restores_even_after_last_step(self):
        from unittest.mock import Mock

        with patch.dict("sys.modules", networks=SimpleNamespace(available_network_aliases={})):
            mod = load("extensions-builtin/sd_forge_lora_schedule/scripts/control.py")
        wrap = mod.OnlineLoRAPatch("w", [0.25, None], name="a")
        unet = SimpleNamespace(weight_wrapper_patches={"w": [wrap]})
        fake = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
        mod.LoRAControl.mapping["a"] = [(1, 0), (0, 1)]
        with patch.object(mod, "shared", fake):
            mod.LoRAControl.adjust_lora(SimpleNamespace(sampling_step=5, total_sampling_steps=10))
        self.assertEqual(wrap.patch[0][0], 0.5)
        mod.LoRAControl().postprocess(Mock(), None)
        self.assertEqual(wrap.patch[0][0], 0.25)

    def test_real_builder_route_requires_remote_authentication(self):
        import ast
        from unittest.mock import Mock

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from modules.aikimi_security.auth import install_remote_auth_middleware
        from tools.tests.test_remote_auth_middleware import options

        path = ROOT / "extensions-builtin/sd_forge_json_builder/scripts/builder.py"
        fn = next(node for node in ast.parse(path.read_text("utf8")).body if getattr(node, "name", "") == "json_builder_api")
        shared = Mock()
        shared.prompt_styles.styles = {"example": SimpleNamespace(prompt="tree")}
        namespace = {"shared": shared}
        exec(compile(ast.Module([fn], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        app = FastAPI()
        namespace["json_builder_api"](None, app)
        install_remote_auth_middleware(app, options(remote=True, api_only=True, api_auth="reader:example"))
        with TestClient(app) as client:
            route = "/internal/json_builder_prompt_styles"
            self.assertEqual(client.get(route).status_code, 401)
            self.assertEqual(client.get(route, auth=("reader", "example")).json(), [{"name": "example", "prompt": "tree"}])

    def test_rgba_encoder_preserves_alpha_and_supplies_opaque_rgb(self):
        from unittest.mock import Mock

        from backend.nn.wan_vae import WanVAE

        encoder = Mock(side_effect=lambda image, **kwargs: image)
        encoder.conv1.in_channels = 4
        model = SimpleNamespace(encoder=encoder, conv1=lambda image: torch.cat([image, image], dim=1))
        rgb = torch.rand(1, 3, 1, 8, 8)
        result = WanVAE.encode(model, rgb)
        torch.testing.assert_close(result[:, :3], rgb)
        torch.testing.assert_close(result[:, 3:], torch.ones_like(rgb[:, :1]))
        rgba = torch.rand(1, 4, 1, 8, 8)
        torch.testing.assert_close(WanVAE.encode(model, rgba), rgba)

    def test_nvfp4_and_k_quant_decode_without_cuda(self):
        from gguf import GGMLQuantizationType as Q

        from backend.loader_gguf import dequantize
        from backend.operations_gguf import ParameterGGUF, dequantize_tensor

        # NVFP4 uses 4 E4M3 scales followed by E2M1 pairs per 64 values.
        data = torch.full((1, 36), 0x22, dtype=torch.uint8)
        data[:, :4] = 0x38
        packed = ParameterGGUF(data, tensor_type=Q.NVFP4, tensor_shape=torch.Size([1, 64]))
        torch.testing.assert_close(dequantize(packed, torch.float32), torch.ones(1, 64))
        packed = ParameterGGUF(torch.zeros(1, 144, dtype=torch.uint8), tensor_type=Q.Q4_K, tensor_shape=torch.Size([1, 256]))
        packed.gguf_cls.bake(packed)
        decoded = dequantize_tensor(packed, input=torch.zeros(1, 256))
        self.assertEqual(decoded.shape, (1, 256))
        self.assertEqual(decoded.count_nonzero(), 0)

    def test_sparse_attention_without_kernel_skips_before_accessing_model(self):
        with patch.dict("sys.modules", comfy_kitchen=None):
            mod = load("extensions-builtin/sd_forge_sparse/scripts/sparse.py")
        self.assertIsNone(mod.ck)
        mod.SparseAttentionForForge().process_before_every_sampling(SimpleNamespace(), True)

    def test_mixed_length_conditions_preserve_image_indices_and_and_weights(self):
        from backend.sampling.condition import compile_weighted_conditions
        from modules.prompt_parser import stack_conditioning

        rows = [torch.ones(2, 4), torch.ones(5, 4), torch.ones(2, 4)]
        heterogeneous = stack_conditioning(rows)
        compiled = compile_weighted_conditions(heterogeneous, [[(0, 1), (1, -1)], [(2, 1)]])
        self.assertEqual([(c["batch_indices"], c["strength"]) for c in compiled], [([0, 1], 1), ([0], -1)])
        self.assertEqual(compiled[1]["cross_attn"].shape, (1, 5, 4))

    def test_extra_networks_and_sampling_contracts(self):
        for path, marker in (
            ("extensions-builtin/sd_forge_json_builder/scripts/builder.py", "/internal/json_builder_prompt_styles"),
            ("modules/processing_scripts/slg.py", "SLG Single Layers"),
            ("scripts/xyz_grid.py", 'AxisOption("Text Encoder"'),
            ("scripts/xyz_grid.py", "format_lora_label"),
            ("modules_forge/main_entry.py", "on_refresh_checkpoint"),
            ("modules/ui_extensions.py", "new_only"),
            ("backend/args.py", "--force-w4a8"),
            ("backend/operations_gguf.py", "NVFP4"),
            ("backend/attention.py", "optimized_attention_override"),
            ("backend/sampling/condition.py", "repeat_to_batch_size"),
            ("scripts/prompts_from_file.py", "all_negative_prompts"),
        ):
            with self.subTest(path=path):
                self.assertIn(marker, (ROOT / path).read_text("utf8"))
