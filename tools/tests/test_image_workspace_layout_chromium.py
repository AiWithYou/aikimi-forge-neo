"""Real Gradio controls remain inside the shared image settings column."""

import os
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import gradio as gr

from modules_forge.moss_image import core
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
        test_moss_image_ui.restore_defaults(
            blocks, {"moss_image_studio/拡大して仕上げる（Hires.fix）/value": False}
        )
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

    def test_saved_hires_on_is_visible_and_submits_the_displayed_size_after_load(self):
        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chrome or Chromium is not installed")
        screen = test_moss_image_ui.UITests().load_screen()
        blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        test_moss_image_ui.restore_defaults(blocks, {
            "moss_image_studio/拡大して仕上げる（Hires.fix）/value": True,
            "moss_image_studio/幅/value": 1024,
            "moss_image_studio/高さ/value": 768,
            "moss_image_studio/倍率/value": 1.5,
            "moss_image_studio/プロンプト/value": "cat",
        })
        with blocks:
            hydrated = gr.Markdown("loading", elem_id="workspace-test-ready")
            blocks.load(lambda: "ready", outputs=hydrated, api_visibility="private")
        css = "\n".join((ROOT / path).read_text(encoding="utf-8") for path in (
            "style.css", "extensions-builtin/aikimi-ui/style.css", "extensions-builtin/moss-image-studio/style.css"
        ))
        _, url, _ = blocks.launch(server_name="127.0.0.1", server_port=reserve_local_port(),
                                  prevent_thread_lock=True, quiet=True, css=css)
        captured = []
        submitted = threading.Event()

        def capture(value):
            captured.append(core.validate_request(value))
            submitted.set()
            return "test-job"

        with patch.object(screen.SERVICE, "start", side_effect=capture), cdp_page(chromium, url) as page:
            self.assertTrue(page.wait_for("#workspace-test-ready", "element.textContent.trim() === 'ready'"))
            self.assertTrue(page.evaluate("document.querySelector('#moss-image-hires-enabled input').checked"))
            self.assertTrue(page.wait_for("#moss-image-hires-settings", "element.getBoundingClientRect().height > 0"))
            self.assertTrue(page.wait_for("#moss-image-hires-size", "element.textContent.includes('1536×1152')"))
            page.evaluate("[...document.querySelectorAll('button')].find(e => e.textContent.trim() === '生成').click()")
            self.assertTrue(submitted.wait(10), "Restored controls did not submit a generation request")
            self.assertEqual((captured[0]["width"], captured[0]["height"]), (1536, 1152))
            self.assertEqual(captured[0]["hires"]["base_height"], 768)
            self.assertEqual(page.exceptions, [])

    def test_delayed_hires_load_converges_to_changes_made_while_loading(self):
        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chrome or Chromium is not installed")
        for case in ("disable", "width", "invalid"):
            with self.subTest(case=case):
                self.assert_delayed_hires_load(chromium, case)

    def assert_delayed_hires_load(self, chromium, case):
        screen = test_moss_image_ui.UITests().load_screen()
        original = screen.hires_controls
        entered, release, independent, submitted = (threading.Event() for _ in range(4))
        guard, calls, captured = threading.Lock(), [], []

        def controlled(*values):
            with guard:
                calls.append(values)
                number = len(calls)
            if number == 1:
                entered.set()
                if not release.wait(30):
                    raise RuntimeError("Initial Hires response was not released")
            return (*original(*values), str(number))

        with patch.object(screen, "hires_controls", controlled):
            blocks = screen.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        test_moss_image_ui.restore_defaults(blocks, {
            "moss_image_studio/拡大して仕上げる（Hires.fix）/value": True,
            "moss_image_studio/幅/value": 511 if case == "invalid" else 1024,
            "moss_image_studio/高さ/value": 768,
            "moss_image_studio/倍率/value": 1.5,
            "moss_image_studio/プロンプト/value": "cat",
        })
        width_selector = '#moss-image-settings input[aria-label="幅"]'
        with blocks:
            load_reply = gr.Markdown("waiting", elem_id="hires-test-load-reply")
            change_reply = gr.Markdown("waiting", elem_id="hires-test-change-reply")
        for event in blocks.fns.values():
            if event.fn is controlled:
                event.outputs.append(load_reply)
            elif event.fn is screen.hires_preview:
                original_preview = event.fn

                def changed(*values, preview=original_preview):
                    result = preview(*values)
                    independent.set()
                    return result, "changed"

                event.fn = changed
                event.outputs.append(change_reply)
        _, url, _ = blocks.launch(server_name="127.0.0.1", server_port=reserve_local_port(),
                                  prevent_thread_lock=True, quiet=True,
                                  css=(ROOT / "extensions-builtin/moss-image-studio/style.css").read_text(encoding="utf-8"))

        def capture(value):
            captured.append(core.validate_request(value))
            submitted.set()
            return "test-job"

        try:
            with patch.object(screen.SERVICE, "start", side_effect=capture), cdp_page(chromium, url) as page:
                self.assertTrue(entered.wait(10), "Initial Hires load did not start")
                self.assertTrue(page.wait_for("#moss-image-hires-enabled input", "element.checked"))
                if case == "disable":
                    page.evaluate("document.querySelector('#moss-image-hires-enabled input').click()")
                    self.assertTrue(page.wait_for("#moss-image-hires-enabled input", "!element.checked"))
                elif case == "invalid":
                    self.assertTrue(page.wait_for(width_selector, "Number(element.value) === 511"))
                    page.evaluate(f"""(() => {{const input=document.querySelector({width_selector!r});
                        input.value=512;input.dispatchEvent(new Event('input',{{bubbles:true}}));
                        input.dispatchEvent(new Event('change',{{bubbles:true}}));
                        document.querySelector('#moss-image-hires-enabled input').click();}})()""")
                    self.assertTrue(page.wait_for("#moss-image-hires-enabled input", "!element.checked"))
                else:
                    self.assertTrue(page.wait_for(width_selector, "Number(element.value) === 1024"))
                    page.evaluate(f"""(() => {{const input=document.querySelector({width_selector!r});
                        for(const value of [768,512]) {{input.value=value;input.dispatchEvent(new Event('input',{{bubbles:true}}));
                        input.dispatchEvent(new Event('change',{{bubbles:true}}));}} }})()""")
                    self.assertTrue(page.wait_for(width_selector, "Number(element.value) === 512"))
                # In the old implementation, wait for the independent change
                # response to be applied before releasing the older load.
                # A single always_last event instead retains this change locally.
                if independent.wait(2):
                    self.assertTrue(page.wait_for("#hires-test-change-reply", "element.textContent.trim() === 'changed'"))
                release.set()
                self.assertTrue(page.wait_for("#hires-test-load-reply", "element.textContent.trim() !== 'waiting'"))
                page.evaluate("new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))")
                disabled = case in ("disable", "invalid")
                expected = "" if disabled else "768×1152"
                self.assertTrue(page.wait_for("#moss-image-hires-size",
                    "element.textContent.trim() === ''" if not expected else f"element.textContent.includes('{expected}')"), str(calls))
                self.assertTrue(page.wait_for("#moss-image-hires-settings",
                    "element.getBoundingClientRect().height === 0" if disabled else "element.getBoundingClientRect().height > 0"))
                page.evaluate("[...document.querySelectorAll('button')].find(e=>e.textContent.trim()==='生成').click()")
                self.assertTrue(submitted.wait(10))
                self.assertEqual((captured[0]["width"], captured[0]["height"]),
                                 (512,768) if case == "invalid" else (1024,768) if case == "disable" else (768,1152))
                self.assertEqual("hires" in captured[0], not disabled)
                self.assertEqual(page.exceptions, [])
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()
