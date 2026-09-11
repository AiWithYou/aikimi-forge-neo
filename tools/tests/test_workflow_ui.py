import unittest

from modules_forge.workflow_ui import (
    workflow_hero,
    workflow_section,
    workflow_summary,
)


class WorkflowUiTests(unittest.TestCase):
    def test_hero_escapes_content_and_keeps_step_order(self):
        markup = workflow_hero(
            "A < B",
            "safe & clear",
            badges=("img2img", "Batch 1×1"),
            steps=("入力", "設定", "生成"),
        )

        self.assertIn("A &lt; B", markup)
        self.assertIn("safe &amp; clear", markup)
        self.assertLess(markup.index("入力"), markup.index("設定"))
        self.assertLess(markup.index("設定"), markup.index("生成"))
        self.assertEqual(markup.count("<li>"), 3)

    def test_section_and_summary_expose_accessible_structure(self):
        section = workflow_section(2, "出力", "長辺または幅と高さ")
        summary = workflow_summary(
            "4K Smart",
            (("出力", "長辺 4096 px"), ("Tile", "自動")),
            status="推奨設定",
            note="Generate前に入力を確認",
            tone="caution",
        )

        self.assertIn(">02<", section)
        self.assertIn('aria-live="polite"', summary)
        self.assertIn("is-caution", summary)
        self.assertIn("<dt>出力</dt>", summary)
        self.assertIn("<dd>長辺 4096 px</dd>", summary)

    def test_unknown_summary_tone_falls_back_to_ready(self):
        summary = workflow_summary("Test", (), tone="unknown")
        self.assertIn("is-ready", summary)


if __name__ == "__main__":
    unittest.main()
