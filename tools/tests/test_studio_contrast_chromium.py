"""明るいGradioテーマ内でも、Studioの説明と設定を読めることを確認する。"""

from __future__ import annotations

import unittest
from pathlib import Path

import gradio as gr

from tools.tests.chromium_helpers import find_chromium, reserve_local_port
from tools.tests.test_gradio_frontend_compat_chromium import _wait_expression, cdp_page

ROOT = Path(__file__).resolve().parents[2]


class StudioContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chromium = find_chromium()
        if not cls.chromium:
            raise unittest.SkipTest("Chromium is required for rendered contrast checks")
        styles = "\n".join(
            (ROOT / f"extensions-builtin/{name}/style.css").read_text(encoding="utf-8")
            for name in ("minimax-h3-studio", "sensenova-u15-studio")
        )
        with gr.Blocks(analytics_enabled=False) as cls.demo:
            for root, prefix in (("sensenova-u15-studio", "sn"), ("h3-studio", "h3")):
                with gr.Column(elem_id=root):
                    gr.HTML(
                        f'<div class="{prefix}-mode-note"><span id="{prefix}-probe-note">設定を選んでください。</span></div>'
                        f'<details><summary id="{prefix}-probe-summary">詳細を開く</summary>入力設定</details>'
                        f'<code id="{prefix}-probe-code">Image-1</code>'
                    )
                    gr.Textbox(label="プロンプト", info="画像の指示", elem_id=f"{prefix}-probe-field")
                    if prefix == "sn":
                        with gr.Row(elem_classes=["sn-actions"]):
                            gr.Button("停止", elem_id="sn-cancel")
                            gr.Button("生成", elem_id="sn-generate")
        cls.port = reserve_local_port()
        cls.demo.launch(
            server_name="127.0.0.1",
            server_port=cls.port,
            prevent_thread_lock=True,
            quiet=True,
            inbrowser=False,
            ssr_mode=False,
            css=styles,
        )

    @classmethod
    def tearDownClass(cls):
        cls.demo.close()

    def test_light_theme_text_contrast_reaches_4_5(self):
        with cdp_page(self.chromium, f"http://127.0.0.1:{self.port}/?__theme=light") as page:
            self.assertTrue(page.evaluate(_wait_expression("#h3-probe-summary", "true")))
            values = page.evaluate("""(() => {
                const luminance = color => {
                    const c = color.match(/[0-9.]+/g).slice(0, 3).map(Number).map(v => {
                        v /= 255; return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
                    });
                    return c[0] * 0.2126 + c[1] * 0.7152 + c[2] * 0.0722;
                };
                return ['sn', 'h3'].flatMap(prefix => ['note', 'summary', 'code'].map(kind => {
                    const node = document.getElementById(`${prefix}-probe-${kind}`);
                    const color = getComputedStyle(node).color;
                    let parent = node, background;
                    while (parent) {
                        background = getComputedStyle(parent).backgroundColor;
                        if (background !== 'rgba(0, 0, 0, 0)' && background !== 'transparent') break;
                        parent = parent.parentElement;
                    }
                    const a = luminance(color), b = luminance(background);
                    return {prefix, kind, color, background, ratio:(Math.max(a,b)+0.05)/(Math.min(a,b)+0.05)};
                }));
            })()""")
        self.assertEqual(len(values), 6)
        for value in values:
            with self.subTest(studio=value["prefix"], text=value["kind"]):
                self.assertGreaterEqual(value["ratio"], 4.5, value)

    def test_mobile_action_bar_stays_inside_the_viewport(self):
        with cdp_page(self.chromium, f"http://127.0.0.1:{self.port}/") as page:
            page.send(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 390,
                    "height": 844,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            self.assertTrue(page.evaluate(_wait_expression("#sn-generate", "true")))
            bounds = page.evaluate("""(() => {
                const button = document.querySelector('#sn-generate').getBoundingClientRect();
                const bar = document.querySelector('.sn-actions').getBoundingClientRect();
                return {left:bar.left, right:bar.right, buttonRight:button.right, width:innerWidth};
            })()""")
        self.assertGreaterEqual(bounds["left"], 8, bounds)
        self.assertLessEqual(bounds["right"], bounds["width"] - 8, bounds)
        self.assertLess(bounds["buttonRight"], bounds["width"], bounds)


if __name__ == "__main__":
    unittest.main()
