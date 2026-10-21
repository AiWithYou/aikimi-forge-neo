"""Conservative inference policy contracts, exercised with CPU tensors only."""

import ast
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from modules_forge import gpu_fault
from modules_forge.moss_image.safety import InferencePolicy, PolicyError


class InferencePolicyTests(unittest.TestCase):
    def setUp(self):
        self.previous = {
            name: getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
            for name in ("flash", "cudnn", "math", "mem_efficient")
        }
        self.addCleanup(self.restore)
        self.enterContext(patch.object(torch.cuda, "_lazy_init", side_effect=AssertionError("CUDA forbidden")))
        self.ops = SimpleNamespace(
            SDPA_BACKEND_PRIORITY=[
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.MATH,
            ],
            linear_input_act=Mock(side_effect=lambda linear, x, *a, **k: linear(x)),
        )
        self.original = self.ops.linear_input_act
        self.policy = InferencePolicy(torch, self.ops)

    def restore(self):
        for name, value in self.previous.items():
            getattr(torch.backends.cuda, f"enable_{name}_sdp")(value)

    def test_effective_policy_and_per_call_guard_cover_aliases(self):
        alias = torch.nn.functional.scaled_dot_product_attention
        for shape in ((1, 24, 8, 128), (1, 24, 36864, 128)):
            with self.subTest(shape=shape), self.policy.enforce():
                query = torch.empty(shape, device="meta")
                with sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]):
                    with self.assertRaises(PolicyError):
                        alias(query, query, query)
        self.assertEqual(self.ops.SDPA_BACKEND_PRIORITY, [SDPBackend.EFFICIENT_ATTENTION])
        self.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
        self.assertFalse(torch.backends.cuda.math_sdp_enabled())

    def test_policy_tampering_between_jobs_is_rejected(self):
        self.ops.SDPA_BACKEND_PRIORITY[:] = [SDPBackend.MATH]
        with self.assertRaises(PolicyError):
            with self.policy.enforce():
                self.fail("policy mismatch must stop before model execution")

    def test_runtime_recreation_reuses_policy_without_hook_stacking(self):
        again = InferencePolicy.install(torch, self.ops)
        self.assertIs(again, self.policy)
        self.assertIs(again.original_linear, self.original)

    def test_real_comfy_wrapper_uses_only_efficient_for_small_and_large_calls(self):
        from torch.overrides import TorchFunctionMode

        root = Path(__file__).resolve().parents[2]
        fixture = json.loads((root / "tools/tests/fixtures/moss_sdpa_wrapper.json").read_text(encoding="utf-8"))
        manifest = json.loads((root / "tools/moss_image_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(fixture["revision"], manifest["runtime"]["comfy_revision"])
        self.assertEqual(hashlib.sha256(fixture["source"].encode("utf-8")).hexdigest(), fixture["source_sha256"])
        tree = ast.parse(fixture["source"])
        node = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef)
            and n.name == "scaled_dot_product_attention"
            and "SDPA_BACKEND_PRIORITY" in ast.unparse(n)
        )
        namespace = {
            "torch": torch,
            "sdpa_kernel": sdpa_kernel,
            "SDPA_BACKEND_PRIORITY": self.ops.SDPA_BACKEND_PRIORITY,
            "SDPA_CONTIGUOUS_KV_L2_SIZE": 0,
        }
        exec(  # noqa: S102 -- Fixed upstream fixture under a CPU-only guard.
            compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), "comfy-wrapper", "exec"),
            namespace,
        )
        flags = []

        class Observe(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                if func is torch.nn.functional.scaled_dot_product_attention:
                    flags.append(
                        tuple(
                            getattr(torch.backends.cuda, f"{name}_sdp_enabled")()
                            for name in ("flash", "cudnn", "math", "mem_efficient")
                        )
                    )
                    return args[0]
                return func(*args, **(kwargs or {}))

        with Observe(), self.policy.enforce():
            for shape in ((1, 24, 8, 128), (1, 24, 36864, 128), (1, 24, 65536, 128)):
                q = torch.empty(shape, device="meta")
                self.assertIs(namespace["scaled_dot_product_attention"](q, q, q), q)
        self.assertEqual(flags, [(False, False, False, True)] * 3)
        self.assertEqual(self.policy.snapshot()["sdpa_calls_checked"], 3)

    def test_bfloat16_situ_matches_fp32_activation_with_single_rounding(self):
        x = torch.linspace(-40, 40, 2 * 2 * 16, dtype=torch.bfloat16).reshape(2, 2, 16)
        h, g = x.float().chunk(2, dim=-1)
        expected = ((4 * torch.tanh(g / 4) * torch.sigmoid(g)) * (25 * torch.tanh(h / 25))).to(x.dtype)
        result = self.ops.linear_input_act(lambda value: value, x, "situ")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)

    def test_gqa_expands_shared_heads_before_efficient_sdpa(self):
        from torch.overrides import TorchFunctionMode

        q = torch.randn(1, 16, 10, 32)
        k, v = torch.randn(1, 4, 10, 32), torch.randn(1, 4, 10, 32)
        mask = torch.ones(10, 10, dtype=torch.bool).tril()
        with sdpa_kernel([SDPBackend.MATH]):
            expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, enable_gqa=True, scale=0.25)
        observed = []

        class Observe(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is torch.nn.functional.scaled_dot_product_attention:
                    query, key, value = tuple(args[:3]) or tuple(kwargs[n] for n in ("query", "key", "value"))
                    self_case.assertEqual(query.shape[-3], key.shape[-3])
                    self_case.assertEqual(query.shape[-3], value.shape[-3])
                    self_case.assertFalse(kwargs.get("enable_gqa", False))
                    self_case.assertFalse(torch.backends.cuda.math_sdp_enabled())
                    self_case.assertFalse(torch.backends.cuda.cudnn_sdp_enabled())
                    torch.testing.assert_close(key, k.repeat_interleave(4, dim=-3))
                    torch.testing.assert_close(value, v.repeat_interleave(4, dim=-3))
                    observed.append(True)
                    # Execute only a CPU reference after observing the guarded call.
                    with sdpa_kernel([SDPBackend.MATH]):
                        return func(*args, **kwargs)
                return func(*args, **kwargs)

        self_case = self
        with Observe(), self.policy.enforce():
            positional = torch.nn.functional.scaled_dot_product_attention(q, k, v, mask, enable_gqa=True, scale=0.25)
            keyword = torch.nn.functional.scaled_dot_product_attention(
                query=q, key=k, value=v, attn_mask=mask, enable_gqa=True, scale=0.25
            )
        self.assertEqual(len(observed), 2)
        torch.testing.assert_close(positional, expected)
        torch.testing.assert_close(keyword, expected)

    def test_invalid_gqa_heads_stop_before_attention(self):
        q = torch.empty(1, 5, 10, 32, device="meta")
        k = torch.empty(1, 4, 10, 32, device="meta")
        with self.policy.enforce(), self.assertRaisesRegex(PolicyError, "GQA"):
            torch.nn.functional.scaled_dot_product_attention(q, k, k, enable_gqa=True)

    def test_situ_is_eager_chunked_and_never_reaches_fused_quantizer(self):
        x = torch.linspace(-1000, 1000, 1025 * 8).reshape(1025, 8)
        original = x.clone()
        hidden, gate = x.chunk(2, dim=-1)
        expected = (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (25 * torch.tanh(hidden / 25))
        observed = []

        def linear(y):
            observed.append(y)
            return y

        result = self.ops.linear_input_act(linear, x, "situ")
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(x, original))
        self.assertLessEqual(result.abs().max().item(), 100)
        self.assertIsNone(self.original.call_args.args[2])
        self.assertEqual(self.policy.snapshot()["situ_eager_calls"], 1)
        self.assertEqual(len(observed), 1)

    def test_other_activations_and_residual_arguments_are_preserved(self):
        x, residual, scale = torch.ones(2, 8), torch.ones(2, 4), torch.ones(4)
        self.ops.linear_input_act(lambda value: value, x, "swiglu", None, 0, residual, scale)
        self.assertEqual(self.original.call_args.args[2], "swiglu")
        self.assertIs(self.original.call_args.args[5], residual)
        self.assertIs(self.original.call_args.args[6], scale)


class RuntimeFatalCleanupTests(unittest.TestCase):
    def test_worker_keeps_runner_alive_after_fatal_without_cleanup(self):
        from tools import moss_image_worker

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "request.json").write_text("{}", encoding="utf-8")
            runner = Mock()
            runner.run.side_effect = RuntimeError("CUDA error: unknown error")
            with (
                patch.object(moss_image_worker, "_runner", runner),
                patch.object(moss_image_worker, "_root", root),
                patch("modules_forge.moss_image.core.safe_model_root", return_value=root),
            ):
                with self.assertRaisesRegex(RuntimeError, "unknown error"):
                    moss_image_worker.resident_run({"root": str(root), "job_dir": str(root)})
                self.assertIs(moss_image_worker._runner, runner)
                runner.close.assert_not_called()

    def test_generate_fault_skips_offload_and_keeps_failed_state(self):
        from modules_forge.moss_image.inference import ComfyRuntime

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.root = Path("unused")
        runtime.failed = False
        runtime.offload = Mock()
        request = {"precision": "int8", "text_precision": "int8"}
        with patch("modules_forge.moss_image.inference.core.asset_paths", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "unknown error"):
                runtime.generate(request, Mock(), Mock(side_effect=RuntimeError("CUDA error: unknown error")))
        self.assertTrue(runtime.failed)
        runtime.offload.assert_not_called()

    def test_offload_is_skipped_after_fatal_gpu_error(self):
        from modules_forge.moss_image.inference import ComfyRuntime

        runtime = ComfyRuntime.__new__(ComfyRuntime)
        runtime.failed = True
        runtime.management = Mock()
        runtime.torch = SimpleNamespace(cuda=Mock())
        runtime.cpu = False
        runtime.offload()
        runtime.close()
        runtime.management.unload_all_models.assert_not_called()
        runtime.torch.cuda.is_available.assert_not_called()

    def test_pending_record_survives_parent_crash_but_normal_completion_removes_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = gpu_fault.FaultState(Path(temporary) / "fault.json")
            state.begin("job")
            self.assertFalse(state.blocked())
            self.assertTrue(gpu_fault.FaultState(state.path).blocked())
            state.complete("job")
            self.assertFalse(gpu_fault.FaultState(state.path).blocked())


if __name__ == "__main__":
    unittest.main()
