"""Real Gradio controls remain inside the shared image settings column."""

import os
import unittest
from pathlib import Path

import gradio as gr

from tools.tests import test_moss_image_ui
from tools.tests.chromium_helpers import cdp_page, find_chromium, reserve_local_port

ROOT = Path(__file__).resolve().parents[2]


class ImageWorkspaceLayoutChromiumTests(unittest.TestCase):
    def test_full_forge_workspace_controls_are_not_clipped(self):
        url = os.environ.get("AIKIMI_FULL_UI_URL")
        if not url:
            self.skipTest("set AIKIMI_FULL_UI_URL to probe the full Forge page")
        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chrome or Chromium is not installed")
        with cdp_page(chromium, url) as page:
            self.assertTrue(page.wait_for("#aikimi-tab-moss-image", "!element.disabled", timeout_ms=45_000))
            page.evaluate("document.querySelector('#aikimi-tab-moss-image').click()")
            self.assertTrue(page.wait_for("#moss-image-settings", "element.getBoundingClientRect().width > 0"))
            self.assert_controls_fit(page)

    def assert_controls_fit(self, page):
        for width in (1280, 1024, 768, 390):
            with self.subTest(width=width):
                page.send(
                    "Emulation.setDeviceMetricsOverride",
                    {"width": width, "height": 900, "deviceScaleFactor": 1, "mobile": False},
                )
                failures = page.evaluate("""new Promise(resolve => requestAnimationFrame(() => {
                    const column=document.querySelector('#moss-image-settings').getBoundingClientRect();
                    resolve([...document.querySelectorAll('#moss-image-settings .block.auto-margin')]
                        .filter(e => {const r=e.getBoundingClientRect(); return r.width && (r.left<column.left-1 || r.right>column.right+1);})
                        .map(e => ({text:e.textContent.trim().slice(0,50),width:e.getBoundingClientRect().width})));
                }))""")
                self.assertEqual(failures, [])

    def test_number_and_slider_controls_fit_desktop_tablet_and_mobile(self):
        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chrome or Chromium is not installed")
        screen = test_moss_image_ui.UITests().load_screen()
        blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        with blocks:
            hydrated = gr.Markdown("loading", elem_id="workspace-test-ready")
            blocks.load(lambda: "ready", outputs=hydrated, api_visibility="private")
        css = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in (
                "style.css",
                "extensions-builtin/aikimi-ui/style.css",
                "extensions-builtin/moss-image-studio/style.css",
            )
        )
        _, url, _ = blocks.launch(
            server_name="127.0.0.1",
            server_port=reserve_local_port(),
            prevent_thread_lock=True,
            quiet=True,
            css=css,
        )
        with cdp_page(chromium, url) as page:
            self.assertTrue(page.wait_for("#workspace-test-ready", "element.textContent.trim() === 'ready'"))
            ready = page.wait_for(
                "#moss-image-settings",
                "element.getBoundingClientRect().width > 0 && [...element.querySelectorAll('input')].some(e => e.getBoundingClientRect().width > 0)",
            )
            self.assertTrue(
                ready,
                page.evaluate(
                    "[...document.querySelectorAll('[id^=moss-image]')].map(e=>({id:e.id,width:e.getBoundingClientRect().width,display:getComputedStyle(e).display}))"
                ),
            )
            self.assertEqual(
                page.evaluate("document.querySelector('#moss-image-hires-settings').getBoundingClientRect().height"),
                0,
                "Hires settings must stay collapsed until enabled",
            )
            page.evaluate("document.querySelector('#moss-image-hires-enabled input').click()")
            self.assertTrue(page.wait_for("#moss-image-hires-settings", "element.getBoundingClientRect().height > 0"))
            page.evaluate("document.querySelector('#moss-image-img2img-button').click()")
            self.assertTrue(page.wait_for("#moss-image-hires-controls", "element.getBoundingClientRect().height === 0"))
            page.evaluate("document.querySelector('#moss-image-txt2img-button').click()")
            self.assertTrue(page.wait_for("#moss-image-hires-controls", "element.getBoundingClientRect().height > 0"))
            self.assertTrue(page.evaluate("document.querySelector('#moss-image-hires-enabled input').checked"))
            self.assert_controls_fit(page)
            self.assertEqual(page.exceptions, [])


if __name__ == "__main__":
    unittest.main()
