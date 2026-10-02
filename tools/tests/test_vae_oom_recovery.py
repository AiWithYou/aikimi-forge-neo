"""Check VAE batch OOM recovery releases failed-attempt CPU tensors."""

import gc
import types
import unittest
import weakref
from unittest.mock import patch

import torch
import torch.nn.functional as F

from backend import memory_management
from backend.patcher.vae import VAE


class BatchFailingAutoencoder(torch.nn.Module):
    def __init__(self, operation):
        super().__init__()
        self.config = types.SimpleNamespace(latent_channels=3)
        self.operation = operation
        self.calls = 0
        self.failed_input = None
        self.full_output = None

    def run(self, samples, operation):
        self.calls += 1
        if self.calls == 2 and operation == self.operation:
            self.failed_input = weakref.ref(samples)
            raise torch.OutOfMemoryError("injected second-batch OOM")
        output = (
            F.avg_pool2d(samples, 8)
            if operation == "encode"
            else F.interpolate(samples, scale_factor=8, mode="nearest")
        )
        if self.calls == 1:
            self.full_output = weakref.ref(output)
        return output

    def encode(self, samples):
        return self.run(samples, "encode")

    def decode(self, samples):
        return self.run(samples, "decode")


class VaeOomRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.cpu = torch.device("cpu")
        self.enterContext(patch.object(memory_management, "load_models_gpu"))
        self.enterContext(patch.object(memory_management, "vae_offload_device", return_value=self.cpu))
        self.enterContext(patch.object(memory_management, "intermediate_device", return_value=self.cpu))
        self.enterContext(patch.object(memory_management, "get_free_memory", return_value=1))
        self.enterContext(patch.object(memory_management, "VAE_ALWAYS_TILED", False))

    @torch.inference_mode()
    def test_encode_batch_oom_releases_attempt_tensors_before_tiled_retry(self):
        self.check_recovery("encode")

    @torch.inference_mode()
    def test_decode_batch_oom_releases_attempt_tensors_before_tiled_retry(self):
        self.check_recovery("decode")

    @torch.inference_mode()
    def test_model_loading_oom_retries_without_uninitialized_batch_variables(self):
        for operation in ("encode", "decode"):
            with self.subTest(operation=operation):
                vae = VAE(model=BatchFailingAutoencoder(None), device=self.cpu, dtype=torch.float32)
                vae.memory_used_encode = vae.memory_used_decode = lambda shape, dtype: 1
                if operation == "encode":
                    source = torch.linspace(0, 1, 2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
                    expected = F.avg_pool2d(VAE.process_input(source.movedim(-1, 1)), 8)
                else:
                    source = torch.linspace(-1, 1, 24).reshape(2, 3, 2, 2)
                    expected = VAE.process_output(F.interpolate(source, scale_factor=8, mode="nearest")).movedim(1, -1)
                with (
                    patch.object(
                        memory_management, "load_models_gpu", side_effect=[torch.OutOfMemoryError("load OOM"), None]
                    ),
                    patch.object(memory_management, "soft_empty_cache") as empty_cache,
                ):
                    actual = getattr(vae, operation)(source)
                empty_cache.assert_called_once_with()
                torch.testing.assert_close(actual, expected)

    def check_recovery(self, operation):
        model = BatchFailingAutoencoder(operation)
        vae = VAE(model=model, device=self.cpu, dtype=torch.float32)
        vae.memory_used_encode = vae.memory_used_decode = lambda shape, dtype: 1
        allocated = []
        real_empty = torch.empty
        real_process_output = vae.process_output

        def capture_empty(*args, **kwargs):
            output = real_empty(*args, **kwargs)
            allocated.append(weakref.ref(output))
            return output

        def capture_output(output):
            allocated.append(weakref.ref(output))
            return real_process_output(output)

        def assert_attempt_released():
            gc.collect()
            self.assertTrue(model.failed_input() is None, "failed batch input remains referenced at fallback")
            self.assertTrue(model.full_output() is None, "previous batch model output remains referenced at fallback")
            self.assertTrue(allocated)
            self.assertTrue(all(reference() is None for reference in allocated), "full-path outputs remain at fallback")

        if operation == "encode":
            source = torch.linspace(0, 1, 2 * 16 * 16 * 3).reshape(2, 16, 16, 3)
            expected = F.avg_pool2d(VAE.process_input(source.movedim(-1, 1)), 8)
        else:
            source = torch.linspace(-1, 1, 24).reshape(2, 3, 2, 2)
            expected = VAE.process_output(F.interpolate(source, scale_factor=8, mode="nearest")).movedim(1, -1)
        with (
            patch("backend.patcher.vae.torch.empty", side_effect=capture_empty),
            patch.object(vae, "process_output", new=capture_output),
            patch.object(memory_management, "soft_empty_cache", side_effect=assert_attempt_released) as empty_cache,
        ):
            actual = getattr(vae, operation)(source)
        empty_cache.assert_called_once_with()
        self.assertGreater(model.calls, 2)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
