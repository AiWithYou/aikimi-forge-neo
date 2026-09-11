"""Browser regressions for new feature navigation; no model or downloads required."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "extensions-builtin/aikimi-ui/javascript/aikimi_tabs.js"
HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<title>Aikimi feature navigation regression</title>
<style>
body{font:16px sans-serif;margin:24px;max-width:1100px} button,input{padding:12px;margin:4px}
[hidden]{display:none!important} .tabitem{border:1px solid #aaa;padding:20px}
.aikimi-feature-active{outline:2px solid #456} nav{margin-bottom:16px}
</style></head><body><h1>Aikimi feature navigation regression</h1>
<p>Isolated controls. No model or generation backend.</p><main id="app">
<div id="forge_ui_preset"><label>UI Preset <input value="sd" readonly aria-controls="preset-options"></label>
<ul id="preset-options" role="listbox" hidden><li role="option" data-index="0">krea</li>
<li role="option" data-index="1">anima</li></ul></div>
<div id="tabs"><div id="native-nav" class="tab-nav" role="tablist">
<button role="tab" aria-controls="tab_txt2img" aria-selected="true">txt2img</button>
<button role="tab" aria-controls="tab_img2img">img2img</button>
<button role="tab" aria-controls="tab_extras">Extras</button></div>
<div id="tab_txt2img" class="tabitem">Text to image</div>
<div id="tab_img2img" class="tabitem" hidden>Image to image</div>
<div id="tab_extras" class="tabitem" hidden>Extras</div></div></main></body></html>"""
BOOT = """() => {
    window.gradioApp = () => document.getElementById('app');
    window.updates = []; window.tabChanges = []; window.presetWrites = []; window.accordionWrites = [];
    window.onUiLoaded = fn => fn();
    window.onUiUpdate = fn => updates.push(fn);
    window.onUiTabChange = fn => tabChanges.push(fn);
    window.addNative = (id, title) => {
        const panel = document.createElement('div'); panel.id = id; panel.className = 'tabitem';
        panel.hidden = true; panel.textContent = title;
        const button = document.createElement('button'); button.setAttribute('role', 'tab');
        button.setAttribute('aria-controls', id); button.textContent = title;
        document.getElementById('tabs').append(panel);
        document.getElementById('native-nav').append(button);
    };
    document.getElementById('native-nav').addEventListener('click', event => {
        if (!event.target.matches('button')) return;
        for (const button of event.currentTarget.children)
            button.setAttribute('aria-selected', String(button === event.target));
        for (const panel of document.querySelectorAll('#tabs > .tabitem'))
            panel.hidden = panel.id !== event.target.getAttribute('aria-controls');
        tabChanges.forEach(fn => fn());
    });
    document.querySelector('#forge_ui_preset input').addEventListener('keydown', event => {
        if (event.key === 'ArrowDown') document.getElementById('preset-options').hidden = false;
    });
    document.getElementById('preset-options').addEventListener('mousedown', event => {
        if (!event.target.matches('[role=option]')) return;
        const input = document.querySelector('#forge_ui_preset input');
        input.value = event.target.textContent; presetWrites.push(input.value);
        event.currentTarget.hidden = true;
        input.dispatchEvent(new Event('input', {bubbles:true}));
    });
    window.mountAccordion = (tab='txt2img') => {
        const id = 'aikimi-' + tab + '-anima38';
        const root = document.createElement('div'); root.id = id;
        root.innerHTML = '<div class="label-wrap">Anima 3.8B</div><input type="checkbox" id="' +
            id + '-visible-checkbox"><div id="' + id + '-checkbox"><input type="checkbox"></div>';
        document.getElementById('tab_' + tab).append(root);
        root.visibleCheckbox = root.querySelector('input'); root.onVisibleCheckboxChange = () => {};
    };
    window.inputAccordionChecked = (id, value) => {
        const root = document.getElementById(id); accordionWrites.push([id, value]);
        root.querySelector('.label-wrap').classList.toggle('open', value);
        root.querySelectorAll('input').forEach(input => input.checked = value);
    };
    new MutationObserver(records => updates.forEach(fn => fn(records))).observe(gradioApp(),
        {childList:true,subtree:true});
}"""


@unittest.skipUnless(sync_playwright, "Playwright is not installed")
class FeatureNavigationLifecycleTests(unittest.TestCase):
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
        self.page.evaluate(BOOT)
        self.page.add_script_tag(content=SCRIPT.read_text(encoding="utf-8"))
        self.assertEqual(self.page.title(), "Aikimi feature navigation regression")
        self.assertTrue(self.page.locator("#aikimi-tab-anima38").is_visible())

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def test_cancelled_activation_does_not_change_a_late_preset(self):
        self.page.evaluate("window.detachedPreset = document.getElementById('forge_ui_preset'); detachedPreset.remove()")
        self.page.locator("#aikimi-tab-anima38").click()
        self.page.locator("#native-nav button[aria-controls='tab_extras']").click()
        self.page.evaluate("gradioApp().prepend(detachedPreset); mountAccordion()")
        self.page.wait_for_timeout(180)
        self.assertEqual(self.page.evaluate("presetWrites"), [])
        self.assertEqual(self.page.evaluate("accordionWrites"), [])
        self.assertTrue(self.page.locator("#tab_extras").is_visible())

    def test_cancelled_activation_does_not_enable_a_late_accordion(self):
        self.page.locator("#aikimi-tab-anima38").click()
        self.page.wait_for_function("presetWrites.length === 1")
        self.page.locator("#native-nav button[aria-controls='tab_extras']").click()
        self.page.evaluate("mountAccordion()")
        self.page.wait_for_timeout(180)
        self.assertEqual(self.page.evaluate("accordionWrites"), [])

    def test_latest_click_does_not_wait_for_obsolete_activation_timeout(self):
        self.page.locator("#aikimi-tab-anima38").click()
        self.page.wait_for_function("presetWrites.length === 1")
        self.page.locator("#aikimi-tab-krea2").click()
        self.page.wait_for_function("AikimiTabs.getActiveFeature() === 'krea2'", timeout=1500)
        self.assertEqual(self.page.locator("#forge_ui_preset input").input_value(), "krea")

    def test_late_native_studios_appear_without_manual_refresh(self):
        self.assertTrue(self.page.locator("#aikimi-tab-sensenova").is_hidden())
        self.page.evaluate("addNative('tab_sensenova_u15_studio', 'SenseNova')")
        self.page.wait_for_timeout(120)
        self.assertTrue(self.page.locator("#aikimi-tab-sensenova").is_visible())
        self.page.locator("#aikimi-tab-sensenova").click()
        self.page.wait_for_function("AikimiTabs.getActiveFeature() === 'sensenova'")
        self.assertTrue(self.page.locator("#tab_sensenova_u15_studio").is_visible())

    def test_busy_feedback_clears_on_success_and_cancellation(self):
        self.page.locator("#aikimi-tab-anima38").click()
        self.assertEqual(self.page.locator("#aikimi-tab-anima38").get_attribute("aria-busy"), "true")
        self.page.evaluate("mountAccordion()")
        self.page.wait_for_function("AikimiTabs.getActiveFeature() === 'anima38'")
        self.assertEqual(self.page.locator("#aikimi-tab-anima38").get_attribute("aria-busy"), "false")
        self.assertEqual(self.page.locator("#aikimi-tab-anima38").inner_text(), "Anima")

    def test_manual_click_on_same_native_tab_cancels_pending_alias(self):
        self.page.locator("#aikimi-tab-anima38").click()
        self.page.wait_for_function("presetWrites.length === 1")
        self.page.locator("#native-nav button[aria-controls='tab_txt2img']").click()
        self.page.evaluate("mountAccordion()")
        self.page.wait_for_timeout(180)
        self.assertEqual(self.page.evaluate("accordionWrites"), [])
        self.assertEqual(self.page.locator("#aikimi-tab-anima38").get_attribute("aria-busy"), "false")

    def test_removed_native_studio_is_no_longer_offered(self):
        self.page.evaluate("addNative('tab_minimax_h3_studio', 'MiniMax H3')")
        self.page.wait_for_function("!document.getElementById('aikimi-tab-minimax-h3').hidden")
        self.page.evaluate("""() => {
            document.getElementById('tab_minimax_h3_studio').remove();
            document.querySelector('#native-nav button[aria-controls=tab_minimax_h3_studio]').remove();
        }""")
        self.page.wait_for_function("document.getElementById('aikimi-tab-minimax-h3').hidden")
        self.assertTrue(self.page.locator("#aikimi-tab-krea2").is_visible())

    def test_anima_uses_current_img2img_panel_and_keyboard_navigation(self):
        self.page.evaluate("mountAccordion('img2img')")
        self.page.locator("#native-nav button[aria-controls='tab_img2img']").click()
        self.page.locator("#aikimi-tab-krea2").focus()
        self.page.keyboard.press("ArrowRight")
        self.assertEqual(self.page.evaluate("document.activeElement.id"), "aikimi-tab-anima38")
        self.page.keyboard.press("Enter")
        self.page.wait_for_function("AikimiTabs.getActiveFeature() === 'anima38'")
        self.assertEqual(self.page.evaluate("AikimiTabs.getActiveContainer().id"), "tab_img2img")
        self.assertEqual(self.page.evaluate("accordionWrites"), [["aikimi-img2img-anima38", True]])


if __name__ == "__main__":
    unittest.main()
