import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from modules_forge.model_runtime_status import (
    ForgeModelStatus,
    describe_loaded_model,
    quantization_summary,
)


ROOT = Path(__file__).resolve().parents[2]


class _QuantizedLayer:
    quant_format = "int8_tensorwise"

    def __init__(self, *, convrot: bool):
        params = SimpleNamespace(
            convrot=convrot,
            convrot_groupsize=256 if convrot else None,
        )
        self.weight = SimpleNamespace(_params=params)


class _Transformer:
    def __init__(self):
        self.layers = [_QuantizedLayer(convrot=True), _QuantizedLayer(convrot=False)]

    def modules(self):
        return [self, *self.layers]


class ModelRuntimeStatusTests(unittest.TestCase):
    def test_api_schema_imports_and_accepts_runtime_payload(self):
        payload = {
            "loaded": False,
            "architecture": None,
            "configuration": None,
            "transformer": None,
            "checkpoint": None,
            "checkpoint_sha256": None,
            "additional_modules": [],
            "quantization": {},
            "inspection_errors": [],
        }

        status = ForgeModelStatus(**payload)

        self.assertFalse(status.loaded)
        self.assertIsNone(status.configuration)

    def test_quantization_summary_reports_loaded_formats_and_convrot(self):
        summary = quantization_summary(_Transformer())

        self.assertEqual(summary["quantized_layer_count"], 2)
        self.assertEqual(summary["formats"], {"int8_tensorwise": 2})
        self.assertEqual(summary["convrot_layer_count"], 1)
        self.assertEqual(summary["convrot_group_sizes"], [256])

    def test_loaded_model_description_uses_runtime_types_not_filename_guessing(self):
        transformer = _Transformer()
        model = SimpleNamespace(
            forge_objects=SimpleNamespace(
                unet=SimpleNamespace(
                    model=SimpleNamespace(diffusion_model=transformer)
                )
            ),
            model_config=SimpleNamespace(),
        )
        loading = {
            "checkpoint_info": SimpleNamespace(
                filename=r"C:\models\turbo_krea2_int8.safetensors",
                sha256="0123",
            ),
            "additional_modules": [r"C:\models\qwen_image_vae.safetensors"],
        }

        status = describe_loaded_model(model, loading)

        self.assertTrue(status["loaded"])
        self.assertEqual(status["architecture"], "types.SimpleNamespace")
        self.assertEqual(status["transformer"], f"{__name__}._Transformer")
        self.assertEqual(status["checkpoint_sha256"], "0123")
        self.assertEqual(status["quantization"]["convrot_layer_count"], 1)
        self.assertEqual(status["inspection_errors"], [])

    def test_unloaded_placeholder_has_no_architecture(self):
        status = describe_loaded_model(SimpleNamespace(), {})

        self.assertFalse(status["loaded"])
        self.assertIsNone(status["architecture"])
        self.assertIsNone(status["transformer"])

    def test_krea2_engine_explicitly_enables_runtime_shift(self):
        tree = ast.parse(
            (ROOT / "backend" / "diffusion_engine" / "krea.py").read_text(
                encoding="utf-8"
            )
        )
        krea_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Krea2"
        )
        constructor = next(
            node
            for node in krea_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assignments = [
            node
            for node in ast.walk(constructor)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "use_shift"
                for target in node.targets
            )
        ]

        self.assertEqual(len(assignments), 1)
        self.assertIs(assignments[0].value.value, True)


if __name__ == "__main__":
    unittest.main()
