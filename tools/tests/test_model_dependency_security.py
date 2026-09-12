"""未修正APIと改変ランタイムがモデル読込へ到達しないことを確認する。"""

import hashlib
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from modules.aikimi_security import model_dependencies
from modules_forge import sensenova_u15_source_security as source_security


class AccelerateBoundaryTests(unittest.TestCase):
    def test_public_checkpoint_apis_reject_before_any_model_or_file_access(self):
        import accelerate
        import accelerate.big_modeling
        import accelerate.utils
        import accelerate.utils.modeling
        import torch

        modules = (
            (accelerate, ("load_checkpoint_in_model", "load_checkpoint_and_dispatch")),
            (accelerate.big_modeling, ("load_checkpoint_in_model", "load_checkpoint_and_dispatch")),
            (accelerate.utils, ("load_checkpoint_in_model",)),
            (accelerate.utils.modeling, ("load_checkpoint_in_model",)),
        )
        with ExitStack() as stack:
            for module, names in modules:
                for name in names:
                    stack.enter_context(mock.patch.object(module, name, getattr(module, name)))
            model_dependencies.restrict_accelerate_checkpoint_loading()
            model_dependencies.restrict_accelerate_checkpoint_loading()
            for module, names in modules:
                for name in names:
                    with self.subTest(module=module.__name__, name=name):
                        with self.assertRaisesRegex(RuntimeError, "使用できません"):
                            getattr(module, name)(object(), "../untrusted/model.index.json")
            with accelerate.init_empty_weights():
                self.assertTrue(torch.nn.Linear(2, 2).weight.is_meta)


class SenseNovaSourceBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.config = self.root / "config.json"
        content = b'{"model_type":"neo_unify"}'
        self.config.write_bytes(content)
        (self.root / ".sensenova_runtime_revision").write_text("reviewed", encoding="utf-8")
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(
            mock.patch.object(
                source_security,
                "SENSENOVA_RUNTIME_FILES",
                (("config.json", hashlib.sha256(content).hexdigest(), len(content)),),
            )
        )
        self.stack.enter_context(mock.patch.object(source_security, "SENSENOVA_SOURCE_REVISION", "reviewed"))

    def test_reviewed_source_is_accepted(self):
        source_security.validate_runtime_source(self.root)

    def test_same_size_config_replacement_is_rejected(self):
        self.config.write_bytes(self.config.read_bytes().replace(b"neo_unify", b"evil_code"))
        with self.assertRaisesRegex(RuntimeError, "ハッシュ"):
            source_security.validate_runtime_source(self.root)

    def test_added_python_or_tokenizer_file_is_rejected(self):
        for name in ("__init__.py", "tokenizer.json", "chat_template.jinja", "injected.pyc"):
            with self.subTest(name=name):
                extra = self.root / name
                extra.write_text("malicious", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "未承認"):
                    source_security.validate_runtime_source(self.root)
                extra.unlink()

    def test_missing_file_is_rejected(self):
        self.config.unlink()
        with self.assertRaisesRegex(RuntimeError, "不足"):
            source_security.validate_runtime_source(self.root)

    def test_revision_mismatch_is_rejected(self):
        (self.root / ".sensenova_runtime_revision").write_text("different", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "リビジョン"):
            source_security.validate_runtime_source(self.root)

    def test_worker_ignores_bundled_bytecode_and_forces_offline_loading(self):
        cache = self.root / "__pycache__"
        cache.mkdir()
        (cache / "config.cpython-313.pyc").write_bytes(b"untrusted bytecode")
        with (
            mock.patch.dict(source_security.os.environ, {}, clear=False),
            mock.patch.object(source_security.sys, "pycache_prefix", None),
            mock.patch.object(source_security.sys, "dont_write_bytecode", False),
            mock.patch.object(model_dependencies, "restrict_accelerate_checkpoint_loading") as restrict,
        ):
            source_security.prepare_runtime_source(self.root)
            self.assertNotEqual(source_security.sys.pycache_prefix, str(cache))
            self.assertTrue(Path(source_security.sys.pycache_prefix).is_dir())
            self.assertTrue(source_security.sys.dont_write_bytecode)
            self.assertEqual(source_security.os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(source_security.os.environ["TRANSFORMERS_OFFLINE"], "1")
            restrict.assert_called_once()
