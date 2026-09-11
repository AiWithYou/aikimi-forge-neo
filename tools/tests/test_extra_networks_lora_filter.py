import ast
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXTRA_NETWORKS_JS = ROOT / "javascript" / "extraNetworks.js"
LORA_PAGE_PY = ROOT / "extensions-builtin" / "sd_forge_lora" / "ui_extra_networks_lora.py"
EXTRA_NETWORKS_PY = ROOT / "modules" / "ui_extra_networks.py"


class ExtraNetworksLoraFilterTests(unittest.TestCase):
    def test_server_keeps_every_preset_available_for_client_side_switches(self):
        tree = ast.parse(LORA_PAGE_PY.read_text(encoding="utf-8"))
        page_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ExtraNetworksPageLora"
        )
        list_items = next(
            node for node in page_class.body if isinstance(node, ast.FunctionDef) and node.name == "list_items"
        )
        create_item = next(
            node
            for node in ast.walk(list_items)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "create_item"
        )
        enable_filter = next(keyword.value for keyword in create_item.keywords if keyword.arg == "enable_filter")

        self.assertIsInstance(enable_filter, ast.Constant)
        self.assertIs(enable_filter.value, False)

    def test_card_renderer_exposes_the_canonical_preset_name(self):
        tree = ast.parse(EXTRA_NETWORKS_PY.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].slice, ast.Constant)
            and node.targets[0].slice.value == "SDversion"
        )
        canonical_get = assignment.value
        fallback_get = canonical_get.args[1]

        self.assertEqual(canonical_get.args[0].value, "sd_version")
        self.assertEqual(fallback_get.args[0].value, "sd_version_str")
        self.assertEqual(fallback_get.args[1].value, "SdVersion.Unknown")
        self.assertIn('data-sort-sdversion="{sd_version}"', EXTRA_NETWORKS_PY.read_text(encoding="utf-8"))

    def test_current_dropdown_value_controls_lora_compatibility_filter(self):
        source = EXTRA_NETWORKS_JS.read_text(encoding="utf-8")
        apply_filter = source.split("let applyFilter = function", 1)[1].split("let applySort = function", 1)[0]

        self.assertIn('querySelector("#forge_ui_preset input")?.value', apply_filter)
        self.assertIn("opts.lora_preset_filter === true", apply_filter)
        self.assertIn("extraNetworksLoraCardMatchesPreset", apply_filter)
        self.assertIn("extraNetworksApplyLoraTreePresetFilter", apply_filter)
        self.assertNotIn("radioButtons", apply_filter)
        self.assertNotIn("== True", apply_filter)

    def test_sd_to_krea_switch_matches_only_the_current_preset(self):
        source = EXTRA_NETWORKS_JS.read_text(encoding="utf-8")
        helper = (
            "function extraNetworksLoraCardMatchesPreset"
            + source.split("function extraNetworksLoraCardMatchesPreset", 1)[1].split(
                "function setupExtraNetworksForTab", 1
            )[0]
        )
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for the JavaScript regression test")

        harness = (
            helper
            + r"""
const assert = require("node:assert/strict");
const cards = ["sd", "krea", "Unknown", null];
function visible(preset, enabled = true) {
    return cards.filter((version) => extraNetworksLoraCardMatchesPreset(version, preset, enabled));
}
assert.deepEqual(visible("sd"), ["sd", "Unknown", null]);
assert.deepEqual(visible("krea"), ["krea", "Unknown", null]);
assert.deepEqual(visible("krea", false), cards);
"""
        )
        result = subprocess.run(  # noqa: S603
            [node, "-e", harness],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
