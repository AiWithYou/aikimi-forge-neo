import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STUDIO_STYLES = {
    "MiniMax H3": (
        REPOSITORY_ROOT / "extensions-builtin" / "minimax-h3-studio" / "style.css",
        "#h3-studio",
        "h3",
    ),
    "SenseNova U1.5": (
        REPOSITORY_ROOT / "extensions-builtin" / "sensenova-u15-studio" / "style.css",
        "#sensenova-u15-studio",
        "sn",
    ),
}


def _selector_blocks(source: str, selector: str) -> list[str]:
    pattern = re.compile(rf"{re.escape(selector)}\s*\{{([^{{}}]*)\}}", re.DOTALL)
    return pattern.findall(source)


class StudioCssBoundaryTests(unittest.TestCase):
    def test_studio_styles_do_not_target_forge_global_chrome(self):
        forbidden_selectors = ("#quicksettings", "#tabs", ".tab-nav")
        for name, (path, _root, _prefix) in STUDIO_STYLES.items():
            source = path.read_text(encoding="utf-8")
            with self.subTest(studio=name):
                for selector in forbidden_selectors:
                    self.assertNotIn(selector, source)

    def test_studio_heroes_do_not_take_layout_space(self):
        for name, (path, root, prefix) in STUDIO_STYLES.items():
            source = path.read_text(encoding="utf-8")
            hero_selector = f"{root} .{prefix}-hero"
            hero_blocks = _selector_blocks(source, hero_selector)

            with self.subTest(studio=name):
                self.assertTrue(hero_blocks)
                self.assertIn("display: none", hero_blocks[0])
                self.assertIn("min-height: 0", hero_blocks[0])
                self.assertTrue(
                    all("min-height:" not in block or "min-height: 0" in block for block in hero_blocks),
                    f"{name} must not reserve decorative hero height",
                )
                self.assertTrue(
                    all("display:" not in block or "display: none" in block for block in hero_blocks),
                    f"{name} must not restore the redundant Studio hero",
                )

    def test_compact_layout_preserves_accessibility_and_mobile_contracts(self):
        for name, (path, root, _prefix) in STUDIO_STYLES.items():
            source = path.read_text(encoding="utf-8")
            with self.subTest(studio=name):
                self.assertIn(":focus-visible", source)
                self.assertIn("prefers-reduced-motion: reduce", source)
                self.assertRegex(source, r"@media \(max-width: (?:620|640)px\)")
                self.assertIn(f"{root} *::before", source)
                self.assertIn(f"{root} *::after", source)


if __name__ == "__main__":
    unittest.main()
