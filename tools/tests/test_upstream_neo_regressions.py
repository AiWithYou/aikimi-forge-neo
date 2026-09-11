import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import torch
from PIL import Image

from backend import attention
from backend.nn import anima, vae
from backend.nn.vae import (
    AutoencoderKLFlux2,
    DiagonalGaussianDistribution,
    IntegratedAutoencoderKL,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_upscaler_module():
    modules_package = ModuleType("modules")
    devices = ModuleType("modules.devices")
    devices.device_esrgan = "cpu"
    modelloader = ModuleType("modules.modelloader")
    modelloader.load_models = mock.Mock(return_value=[])
    shared = ModuleType("modules.shared")
    shared.models_path = str(ROOT / "models")
    shared.opts = SimpleNamespace(ESRGAN_tile=0, ESRGAN_tile_overlap=0)
    shared.state = SimpleNamespace(interrupted=False)
    images = ModuleType("modules.images")
    images.LANCZOS = Image.Resampling.LANCZOS
    images.NEAREST = Image.Resampling.NEAREST
    modules_package.devices = devices
    modules_package.modelloader = modelloader
    modules_package.shared = shared

    path = ROOT / "modules" / "upscaler.py"
    spec = importlib.util.spec_from_file_location("_test_upstream_upscaler", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.devices": devices,
        "modules.modelloader": modelloader,
        "modules.shared": shared,
        "modules.images": images,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


def _load_util_module():
    modules_package = ModuleType("modules")
    shared = ModuleType("modules.shared")
    shared.opts = SimpleNamespace(list_hidden_files=False)
    paths = ModuleType("modules.paths_internal")
    paths.cwd = str(ROOT)
    paths.script_path = str(ROOT)
    modules_package.shared = shared

    path = ROOT / "modules" / "util.py"
    spec = importlib.util.spec_from_file_location("_test_upstream_util", path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "modules": modules_package,
        "modules.shared": shared,
        "modules.paths_internal": paths,
    }
    with mock.patch.dict(sys.modules, stubs):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


upscaler = _load_upscaler_module()
util = _load_util_module()


class UpstreamNeoAttentionTests(unittest.TestCase):
    def test_forced_attention_precision_uses_mapping_and_fallback(self):
        with mock.patch.object(
            attention.memory_management,
            "force_upcast_attention_dtype",
            return_value={torch.float16: torch.float32},
        ):
            self.assertIs(
                attention.get_attn_precision(torch.bfloat16, torch.float16),
                torch.float32,
            )
            self.assertIs(
                attention.get_attn_precision(torch.bfloat16, torch.float64),
                torch.bfloat16,
            )


class UpstreamNeoHiresOutputTests(unittest.TestCase):
    def test_hires_output_directory_is_restricted_like_other_local_paths(self):
        tree = ast.parse((ROOT / "modules" / "shared_options.py").read_text(encoding="utf-8"))
        restricted = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "restricted_opts" for target in node.targets)
        )

        self.assertIn("outdir_hires_samples", {element.value for element in restricted.elts})

    def test_global_output_precedes_hires_specific_and_current_paths(self):
        tree = ast.parse((ROOT / "modules" / "processing.py").read_text(encoding="utf-8"))
        resolver = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "_resolve_hires_output_path"
        )
        namespace = {}
        exec(  # noqa: S102 - execute only the extracted pure local helper
            compile(
                ast.fix_missing_locations(ast.Module(body=[resolver], type_ignores=[])),
                filename="modules/processing.py",
                mode="exec",
            ),
            namespace,
        )
        resolve = namespace["_resolve_hires_output_path"]

        self.assertEqual(resolve("global", "hires", "txt2img"), "global")
        self.assertEqual(resolve("", "hires", "txt2img"), "hires")
        self.assertEqual(resolve("", "", "txt2img"), "txt2img")


class _Ones(torch.nn.Module):
    def forward(self, tensor, *args, **kwargs):
        del args, kwargs
        return torch.ones_like(tensor)


class _RecordingUpscaler(upscaler.Upscaler):
    def __init__(self, transform):
        self.calls = 0
        self._transform = transform

    def do_upscale(self, image: Image.Image, selected_model: str):
        del selected_model
        self.calls += 1
        return self._transform(image)

    def load_model(self, path: str):
        del path
        return None


class UpstreamNeoAnimaTests(unittest.TestCase):
    def test_fused_adaln_math_is_only_used_when_explicitly_optimized(self):
        tensor = torch.tensor([[[1.0, -2.0], [0.5, 3.0]]])
        scale = torch.tensor([[[0.25, -0.5], [1.0, 0.125]]])
        shift = torch.tensor([[[2.0, 1.0], [-1.0, 0.25]]])
        norm = torch.nn.Identity()

        with mock.patch.object(
            anima.torch,
            "addcmul",
            side_effect=AssertionError("legacy variants must not use the fused path"),
        ):
            legacy = anima._fn(tensor, norm, scale, shift, optimized=False)

        with mock.patch.object(anima.torch, "addcmul", wraps=torch.addcmul) as addcmul:
            optimized = anima._fn(tensor, norm, scale, shift, optimized=True)

        addcmul.assert_called_once()
        torch.testing.assert_close(optimized, legacy)

    def test_rotary_addcmul_matches_the_reference_equation(self):
        tensor = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [-2.0, 0.5, 1.5, -3.0]]])
        cosine = torch.tensor([[0.75, 0.5, -0.25, 1.0]])
        sine = torch.tensor([[0.25, -0.5, 0.75, 0.125]])

        expected = tensor * cosine.unsqueeze(1) + anima.rotate_half(tensor) * sine.unsqueeze(1)
        actual = anima.apply_rotary_pos_emb(tensor, cosine, sine)

        torch.testing.assert_close(actual, expected)

    def test_adapter_residuals_are_inplace_only_when_the_anima38_flag_is_set(self):
        def block():
            value = anima.TransformerBlock(
                source_dim=2,
                model_dim=2,
                num_heads=1,
                mlp_ratio=1.0,
                use_self_attn=False,
                layer_norm=True,
            )
            value.norm_cross_attn = torch.nn.Identity()
            value.cross_attn = _Ones()
            value.norm_mlp = torch.nn.Identity()
            value.mlp = _Ones()
            return value

        context = torch.zeros(1, 1, 2)
        legacy_input = torch.zeros(1, 1, 2)
        legacy_pointer = legacy_input.data_ptr()
        legacy_output = block()(legacy_input, context)

        self.assertEqual(legacy_input.data_ptr(), legacy_pointer)
        self.assertTrue(torch.equal(legacy_input, torch.zeros_like(legacy_input)))
        self.assertNotEqual(legacy_output.data_ptr(), legacy_pointer)

        optimized_block = block()
        optimized_block._anima38_inplace = True
        optimized_input = torch.zeros(1, 1, 2)
        optimized_pointer = optimized_input.data_ptr()
        optimized_output = optimized_block(optimized_input, context)

        self.assertEqual(optimized_output.data_ptr(), optimized_pointer)
        torch.testing.assert_close(optimized_output, legacy_output)
        torch.testing.assert_close(optimized_output, torch.full_like(optimized_output, 2.0))


class UpstreamNeoVaeTests(unittest.TestCase):
    def test_gaussian_sample_addcmul_matches_the_reference_equation(self):
        mean = torch.tensor([[[[1.0, -2.0]], [[0.5, 3.0]]]])
        log_variance = torch.tensor([[[[0.0, 0.5]], [[-0.5, 1.0]]]])
        distribution = DiagonalGaussianDistribution(torch.cat((mean, log_variance), dim=1))

        torch.manual_seed(20260826)
        expected = distribution.mean + distribution.std * torch.randn(distribution.mean.shape)
        torch.manual_seed(20260826)
        actual = distribution.sample()

        torch.testing.assert_close(actual, expected)

    def test_flux2_decode_denormalization_matches_multiply_then_add(self):
        model = AutoencoderKLFlux2.__new__(AutoencoderKLFlux2)
        torch.nn.Module.__init__(model)
        model.bn_eps = 1e-4
        model.ps = [1, 1]
        model.bn = torch.nn.BatchNorm2d(2, affine=False, track_running_stats=True)
        model.bn.eval()
        model.preprocess_decode = lambda latent: latent
        with torch.no_grad():
            model.bn.running_mean.copy_(torch.tensor([0.5, -1.0]))
            model.bn.running_var.copy_(torch.tensor([4.0, 9.0]))

        latent = torch.tensor([[[[1.0, -2.0]], [[0.25, 0.5]]]])
        scale = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.bn_eps)
        mean = model.bn.running_mean.view(1, -1, 1, 1)
        expected = latent * scale + mean

        with mock.patch.object(
            IntegratedAutoencoderKL,
            "decode",
            autospec=True,
            side_effect=lambda _model, value: value,
        ):
            actual = model.decode(latent)

        torch.testing.assert_close(actual, expected)

    def test_flux2_denormalization_avoids_addcmul_on_cpu(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                latent = torch.tensor([[[[1.0, -2.0]]]], dtype=dtype)
                scale = torch.tensor([[[[2.0]]]], dtype=dtype)
                mean = torch.tensor([[[[0.5]]]], dtype=dtype)

                with mock.patch.object(
                    vae.torch,
                    "addcmul",
                    side_effect=AssertionError("CPU tensors must use the multiply-add path"),
                ):
                    actual = vae._flux2_denormalize(latent, mean, scale)

                self.assertTrue(torch.equal(actual, latent * scale + mean))

    def test_flux2_denormalization_keeps_addcmul_for_accelerators(self):
        latent = SimpleNamespace(device=SimpleNamespace(type="cuda"))
        scale = object()
        mean = object()
        expected = object()

        with mock.patch.object(vae.torch, "addcmul", return_value=expected) as addcmul:
            actual = vae._flux2_denormalize(latent, mean, scale)

        addcmul.assert_called_once_with(mean, latent, scale)
        self.assertIs(actual, expected)


class UpstreamNeoUpscalerTests(unittest.TestCase):
    def setUp(self):
        self.interrupted = mock.patch.object(upscaler.shared.state, "interrupted", False)
        self.interrupted.start()

    def tearDown(self):
        self.interrupted.stop()

    def test_fractional_scale_rounds_to_the_nearest_multiple_of_eight(self):
        scaler = _RecordingUpscaler(lambda image: image.resize((image.width * 2, image.height * 2)))

        result = scaler.upscale(Image.new("RGB", (10, 10)), 1.5)

        self.assertEqual(result.size, (16, 16))
        self.assertEqual(scaler.calls, 1)

    def test_builtin_nearest_receives_the_requested_scale(self):
        pixels = [(255, 255, 255) if (x + y) % 2 else (0, 0, 0) for y in range(8) for x in range(8)]
        source = Image.new("RGB", (8, 8))
        source.putdata(pixels)
        scaler = upscaler.UpscalerNearest()

        actual = scaler.upscale(source, 2.0)
        expected = source.resize((16, 16), Image.Resampling.NEAREST)

        self.assertEqual(scaler.scale, 2.0)
        self.assertEqual(actual.tobytes(), expected.tobytes())

    def test_no_growth_stops_iteration_before_the_four_pass_limit(self):
        scaler = _RecordingUpscaler(lambda image: image)

        result = scaler.upscale(Image.new("RGB", (8, 8)), 4.0)

        self.assertEqual(result.size, (32, 32))
        self.assertEqual(scaler.calls, 1)

    def test_growth_stops_when_both_target_dimensions_are_reached(self):
        scaler = _RecordingUpscaler(lambda image: image.resize((image.width * 2, image.height * 2)))

        result = scaler.upscale(Image.new("RGB", (8, 8)), 4.0)

        self.assertEqual(result.size, (32, 32))
        self.assertEqual(scaler.calls, 2)

    def test_iteration_count_remains_bounded_for_a_distant_target(self):
        scaler = _RecordingUpscaler(lambda image: image.resize((image.width + 1, image.height + 1)))

        result = scaler.upscale(Image.new("RGB", (8, 8)), 100.0)

        self.assertEqual(result.size, (800, 800))
        self.assertEqual(scaler.calls, upscaler.UPSCALE_ITERATIONS)

    def test_interruption_skips_model_calls_but_keeps_exact_output_size(self):
        scaler = _RecordingUpscaler(lambda image: image.resize((image.width * 2, image.height * 2)))

        with mock.patch.object(upscaler.shared.state, "interrupted", True):
            result = scaler.upscale(Image.new("RGB", (10, 10)), 1.5)

        self.assertEqual(result.size, (16, 16))
        self.assertEqual(scaler.calls, 0)


class UpstreamNeoNaturalSortTests(unittest.TestCase):
    def test_natural_sort_orders_stems_numerically_then_extensions(self):
        filenames = [
            "model10.safetensors",
            "model2.safetensors",
            "model2.ckpt",
            "model1.SAFETENSORS",
        ]

        actual = sorted(filenames, key=util.natural_sort_key)

        self.assertEqual(
            actual,
            [
                "model1.SAFETENSORS",
                "model2.ckpt",
                "model2.safetensors",
                "model10.safetensors",
            ],
        )


if __name__ == "__main__":
    unittest.main()
