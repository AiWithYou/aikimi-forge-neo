"""Optional real-browser regressions. Uses existing Playwright/Chromium, never downloads."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[2]
HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Forge keyboard regression fixture</title>
<style>
body { font: 16px sans-serif; margin: 32px; }
.css-hidden { display: none; }
button, input { padding: 10px; margin: 8px; }
</style></head><body>
<h1>Forge keyboard regression fixture</h1>
<p>Isolated UI controls. No model or generation backend is running.</p>
<div id="tabs"><div class="tab-nav" role="tablist">
<button role="tab" aria-controls="tab_txt2img">txt2img</button>
<button role="tab" aria-controls="tab_img2img" aria-selected="true">img2img</button></div>
<div id="tab_txt2img" class="tabitem css-hidden"><button id="txt2img_generate">Hidden generate</button></div>
<div id="tab_img2img" class="tabitem">
<label>Prompt <input id="prompt"></label><button id="img2img_generate">Generate</button>
<button id="img2img_interrupt" style="display: none">Interrupt</button>
<button id="img2img_skip">Skip</button><output id="result">Ready</output>
</div></div></body></html>"""


@unittest.skipUnless(sync_playwright, "Playwright is not installed")
class ShortcutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pw = sync_playwright().start()
        cls.addClassCleanup(cls.pw.stop)
        executable = shutil.which("chromium") or shutil.which("chromium-browser")
        if executable is None:
            candidate = Path(cls.pw.chromium.executable_path)
            if not candidate.is_file():
                raise unittest.SkipTest("No existing Chromium installation")
            executable = str(candidate)
        cls.browser = cls.pw.chromium.launch(executable_path=executable, headless=True)
        cls.addClassCleanup(cls.browser.close)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1100, "height": 750})
        self.addCleanup(self.page.close)
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.set_content(HTML)
        self.page.add_script_tag(content=(ROOT / "script.js").read_text(encoding="utf-8"))
        self.page.evaluate("""() => {
            window.generated = 0;
            window.interrupted = 0;
            window.skipped = 0;
            document.getElementById('img2img_generate').onclick = () => {
                generated++;
                document.getElementById('result').textContent = 'Generated ' + generated;
            };
            document.getElementById('img2img_interrupt').onclick = () => interrupted++;
            document.getElementById('img2img_skip').onclick = () => skipped++;
        }""")
        self.assertEqual(self.page.title(), "Forge keyboard regression fixture")
        self.assertTrue(self.page.locator("#prompt").is_visible())

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def inline_hidden_tab(self):
        self.page.evaluate("document.getElementById('tab_txt2img').style.display = 'none'")

    def test_css_hidden_tab_is_not_targeted_by_shortcut(self):
        self.assertEqual(self.page.evaluate("get_uiCurrentTabContent().id"), "tab_img2img")
        self.page.locator("#prompt").press("Control+Enter")
        self.assertEqual(self.page.evaluate("generated"), 1)

    def test_null_detached_and_shadow_nodes_are_handled(self):
        self.assertEqual(self.page.evaluate("""() => {
            const host = document.createElement('div');
            const root = host.attachShadow({mode: 'open'});
            const button = document.createElement('button');
            root.append(button);
            document.body.append(host);
            const shown = uiElementIsVisible(button);
            host.style.display = 'none';
            const hidden = uiElementIsVisible(button);
            host.remove();
            return [shown, hidden, uiElementIsVisible(button), uiElementIsVisible(null)];
        }"""), [True, False, False, False])

    def test_ordinary_typing_does_not_scan_tabs(self):
        self.page.evaluate("""() => {
            window.tabScans = 0;
            const original = get_uiCurrentTabContent;
            window.get_uiCurrentTabContent = () => { tabScans++; return original(); };
        }""")
        self.page.locator("#prompt").press_sequentially("a prompt")
        self.assertEqual(self.page.evaluate("tabScans"), 0)

    def test_held_shortcut_generates_once(self):
        self.inline_hidden_tab()
        self.page.keyboard.down("Control")
        self.page.keyboard.down("Enter")
        self.page.keyboard.down("Enter")
        self.page.keyboard.up("Enter")
        self.page.keyboard.up("Control")
        self.assertEqual(self.page.evaluate("generated"), 1)

    def test_prevented_and_composing_keys_do_not_generate(self):
        self.inline_hidden_tab()
        self.page.evaluate("""() => {
            const input = document.getElementById('prompt');
            input.addEventListener('keydown', event => event.preventDefault());
            input.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', ctrlKey: true, bubbles: true, cancelable: true
            }));
            document.dispatchEvent(new KeyboardEvent('keydown', {
                key: 'Enter', ctrlKey: true, isComposing: true, bubbles: true
            }));
        }""")
        self.assertEqual(self.page.evaluate("generated"), 0)

    def test_synchronous_interrupt_is_followed_by_one_restart(self):
        self.inline_hidden_tab()
        self.page.evaluate("""() => {
            const button = document.getElementById('img2img_interrupt');
            button.style.display = 'block';
            button.onclick = () => { interrupted++; button.style.display = 'none'; };
        }""")
        self.page.keyboard.press("Control+Enter")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("[interrupted, generated]"), [1, 1])

    def test_batched_style_mutations_only_restart_once(self):
        self.inline_hidden_tab()
        self.page.evaluate("""() => {
            const button = document.getElementById('img2img_interrupt');
            button.style.display = 'block';
            button.onclick = () => {
                interrupted++;
                queueMicrotask(() => {
                    button.style.opacity = '0';
                    button.style.display = 'none';
                });
            };
        }""")
        self.page.keyboard.press("Control+Enter")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("[interrupted, generated]"), [1, 1])

    def test_interrupt_only_option_and_skip_still_work(self):
        self.inline_hidden_tab()
        self.page.evaluate("""() => {
            opts.ctrl_enter_interrupt = true;
            document.getElementById('img2img_interrupt').style.display = 'block';
        }""")
        self.page.keyboard.press("Control+Enter")
        self.page.keyboard.press("Alt+Enter")
        self.assertEqual(self.page.evaluate("[interrupted, generated, skipped]"), [1, 0, 1])


    def test_repeated_distinct_presses_schedule_one_restart(self):
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'block'")
        self.page.keyboard.press("Control+Enter")
        self.page.keyboard.press("Control+Enter")
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'none'")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("[interrupted, generated]"), [1, 1])

    def test_escape_cancels_pending_restart(self):
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'block'")
        self.page.keyboard.press("Control+Enter")
        self.page.keyboard.press("Escape")
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'none'")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("generated"), 0)

    def test_manual_interrupt_cancels_pending_restart(self):
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'block'")
        self.page.keyboard.press("Control+Enter")
        self.page.locator("#img2img_interrupt").click()
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'none'")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("generated"), 0)

    def test_combined_modifiers_do_not_generate_and_skip_together(self):
        self.page.keyboard.press("Control+Alt+Enter")
        self.assertEqual(self.page.evaluate("[generated, skipped]"), [1, 0])

    def test_hidden_tab_does_not_restart_after_interrupt(self):
        self.page.evaluate("document.getElementById('img2img_interrupt').style.display = 'block'")
        self.page.keyboard.press("Control+Enter")
        self.page.evaluate("""() => {
            document.getElementById('tab_img2img').style.display = 'none';
            document.getElementById('img2img_interrupt').style.display = 'none';
        }""")
        self.page.wait_for_timeout(30)
        self.assertEqual(self.page.evaluate("generated"), 0)


if __name__ == "__main__":
    unittest.main()
