import ast
import hashlib
import io
import os
import unittest
from pathlib import Path


class AddnetHashHeaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "modules/hashes.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "addnet_hash_safetensors")
        namespace = {"hashlib": hashlib, "os": os}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.calculate = staticmethod(namespace["addnet_hash_safetensors"])

    def test_truncated_length_or_header_is_rejected(self):
        for content in (b"", b"short", (100).to_bytes(8, "little") + b"{}", (2**64 - 1).to_bytes(8, "little")):
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    self.calculate(io.BytesIO(content))

    def test_only_tensor_payload_is_hashed(self):
        header = b'{"metadata": {}}'
        payload = b"tensor payload" * 100000
        source = io.BytesIO(len(header).to_bytes(8, "little") + header + payload)
        source.seek(5)
        self.assertEqual(self.calculate(source), hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
