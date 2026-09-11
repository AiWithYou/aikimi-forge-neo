import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import yaml

ROOT = Path(__file__).resolve().parents[2]
HANDLER = (
    ROOT
    / "extensions-builtin"
    / "forge_legacy_preprocessors"
    / "annotator"
    / "mmpkg"
    / "mmcv"
    / "fileio"
    / "handlers"
    / "yaml_handler.py"
)


def load_handler_module():
    package_name = "_aikimi_yaml_handlers"
    module_name = f"{package_name}.yaml_handler"
    package = types.ModuleType(package_name)
    package.__path__ = []
    base = types.ModuleType(f"{package_name}.base")
    base.BaseFileHandler = object
    spec = importlib.util.spec_from_file_location(module_name, HANDLER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the vendored YAML handler")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            package_name: package,
            f"{package_name}.base": base,
            module_name: module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class LegacyYamlSecurityTests(unittest.TestCase):
    def test_handler_loads_plain_yaml_with_safe_loader(self):
        handler = load_handler_module().YamlHandler()
        self.assertEqual(handler.load_from_fileobj(io.StringIO("value: 3\n")), {"value": 3})

    def test_handler_rejects_python_object_constructors(self):
        handler = load_handler_module().YamlHandler()
        payload = "!!python/object/apply:builtins.str ['unsafe']\n"
        with self.assertRaises(yaml.constructor.ConstructorError):
            handler.load_from_fileobj(io.StringIO(payload))


if __name__ == "__main__":
    unittest.main()
