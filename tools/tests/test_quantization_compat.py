import unittest

import torch

from backend import memory_management, operations, utils
from modules_forge.packages.huggingface_guess.detection import detect_unet_config


class _MemoryModel:
    parent = None
    load_device = torch.device("cuda")
    current_device = torch.device("cuda")

    def model_size(self):
        return 100

    def loaded_size(self):
        return 40


class QuantizationCompatibilityTests(unittest.TestCase):
    def test_loaded_model_uses_current_device_attribute(self):
        model = _MemoryModel()
        loaded = memory_management.LoadedModel(model)

        self.assertEqual(loaded.model_memory_required(torch.device("cuda")), 60)
        self.assertEqual(loaded.model_memory_required(torch.device("cpu")), 100)

    def test_prequantized_bnb_dtype_markers_are_detected(self):
        for quant_type in ("nf4", "fp4"):
            with self.subTest(quant_type=quant_type):
                state_dict = {
                    "layer.weight": torch.zeros((8, 1), dtype=torch.uint8),
                    f"layer.weight.quant_state.bitsandbytes__{quant_type}": torch.zeros(1, dtype=torch.uint8),
                }
                self.assertEqual(utils.weight_dtype(state_dict), quant_type)

    def test_gguf_import_and_operation_picker(self):
        from backend.operations_gguf import ParameterGGUF
        from modules_forge.packages import gguf

        self.assertTrue(callable(gguf.GGUFReader))
        self.assertTrue(issubclass(ParameterGGUF, torch.nn.Parameter))

        original_linear = torch.nn.Linear
        with operations.using_forge_operations(bnb_dtype="gguf"):
            self.assertIs(torch.nn.Linear, operations.ForgeOperationsGGUF.Linear)
        self.assertIs(torch.nn.Linear, original_linear)

    @unittest.skipUnless(memory_management.bnb_enabled(), "bitsandbytes is not installed")
    def test_bnb_import_and_operation_picker(self):
        from backend.operations_bnb import ForgeLoader4Bit

        self.assertTrue(issubclass(ForgeLoader4Bit, torch.nn.Module))

        original_linear = torch.nn.Linear
        with operations.using_forge_operations(bnb_dtype="nf4"):
            self.assertIs(torch.nn.Linear, operations.ForgeOperationsBNB4bits.Linear)
        self.assertIs(torch.nn.Linear, original_linear)

    @unittest.skipUnless(
        memory_management.bnb_enabled() and torch.cuda.is_available(),
        "bitsandbytes and CUDA are required for NF4 execution",
    )
    def test_bnb_nf4_cuda_forward(self):
        torch.manual_seed(123)
        weight = torch.randn((16, 64), dtype=torch.float16) * 0.1
        with operations.using_forge_operations(
            device=torch.device("cpu"),
            dtype=torch.float16,
            manual_cast_enabled=True,
            bnb_dtype="nf4",
        ):
            layer = torch.nn.Linear(64, 16, bias=False)
        layer.load_state_dict({"weight": weight}, strict=True)
        inputs = torch.randn((2, 64), device="cuda", dtype=torch.float16)

        actual = layer(inputs)
        expected = torch.nn.functional.linear(inputs, weight.cuda())

        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, torch.float16)
        self.assertLess(float((actual - expected).abs().mean()), 0.15)

    @unittest.skipUnless(
        memory_management.bnb_enabled() and torch.cuda.is_available(),
        "bitsandbytes and CUDA are required for NF4 shape recovery",
    )
    def test_krea2_nf4_detection_uses_logical_tensor_shapes(self):
        from backend.operations_bnb import ForgeParams4bit

        state_dict = {}

        def add_nf4_weight(key, shape):
            weight = ForgeParams4bit(
                torch.randn(shape, dtype=torch.float16),
                requires_grad=False,
                compress_statistics=False,
                blocksize=64,
                quant_type="nf4",
                quant_storage=torch.uint8,
            ).cuda()
            state_dict[key] = weight
            for suffix, value in weight.quant_state.as_dict(packed=True).items():
                state_dict[f"{key}.{suffix}"] = value

        add_nf4_weight("first.weight", (8, 16))
        add_nf4_weight("blocks.0.attn.wq.weight", (256, 8))
        add_nf4_weight("blocks.0.attn.wk.weight", (128, 8))
        add_nf4_weight("txtfusion.projector.weight", (8, 5))
        state_dict["txtfusion.layerwise_blocks.0.prenorm.scale"] = torch.zeros(32)

        config = detect_unet_config(state_dict, "")

        self.assertEqual(
            config,
            {
                "image_model": "krea2",
                "features": 8,
                "channels": 4,
                "patch": 2,
                "layers": 1,
                "heads": 2,
                "kvheads": 1,
                "txtlayers": 5,
                "txtdim": 32,
            },
        )


if __name__ == "__main__":
    unittest.main()
