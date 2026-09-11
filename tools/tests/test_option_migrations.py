import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
INITIALIZE_UTIL = ROOT / "modules" / "initialize_util.py"


def load_migration():
    tree = ast.parse(INITIALIZE_UTIL.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "migrate_renamed_options"
    )
    namespace = {}
    exec(  # noqa: S102 - execute only the extracted local migration
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            filename=str(INITIALIZE_UTIL),
            mode="exec",
        ),
        namespace,
    )
    return namespace["migrate_renamed_options"]


class RenamedOptionMigrationTests(unittest.TestCase):
    def run_migration(self, data, *, frozen=False, save_side_effect=None):
        options = SimpleNamespace(
            data=dict(data),
            save=mock.Mock(side_effect=save_side_effect),
        )
        shared = SimpleNamespace(
            opts=options,
            cmd_opts=SimpleNamespace(freeze_settings=frozen),
            config_filename="config.json",
        )
        modules = ModuleType("modules")
        modules.shared = shared
        with mock.patch.dict(sys.modules, {"modules": modules}):
            changed = load_migration()()
        return changed, shared

    def test_old_false_value_becomes_positive_true_value(self):
        changed, shared = self.run_migration({"klein_no_reference": False})

        self.assertTrue(changed)
        self.assertEqual(shared.opts.data, {"klein_do_reference": True})
        shared.opts.save.assert_called_once_with("config.json")

    def test_old_true_value_becomes_positive_false_value(self):
        changed, shared = self.run_migration({"klein_no_reference": True})

        self.assertTrue(changed)
        self.assertEqual(shared.opts.data, {"klein_do_reference": False})

    def test_new_value_wins_and_frozen_settings_are_not_saved(self):
        changed, shared = self.run_migration(
            {"klein_no_reference": False, "klein_do_reference": False},
            frozen=True,
        )

        self.assertTrue(changed)
        self.assertEqual(shared.opts.data, {"klein_do_reference": False})
        shared.opts.save.assert_not_called()

    def test_save_failure_rolls_back_both_old_and_new_keys(self):
        changed, shared = self.run_migration(
            {"klein_no_reference": False, "klein_do_reference": False},
            save_side_effect=OSError("read-only config"),
        )

        self.assertFalse(changed)
        self.assertEqual(
            shared.opts.data,
            {"klein_no_reference": False, "klein_do_reference": False},
        )


if __name__ == "__main__":
    unittest.main()
