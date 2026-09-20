import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")
STUDIOS = {
    "h3": ROOT / "extensions-builtin/minimax-h3-studio/javascript/minimax_h3_studio.js",
    "sn": ROOT / "extensions-builtin/sensenova-u15-studio/javascript/sensenova_u15_studio.js",
}

FIXTURE = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const kind = process.argv[2];
const calls = [], callbacks = {}, keyHandlers = [], documentListeners = new Map();
let active = false, lookups = 0, clicks = 0, statusEnabled = true;
let percent = 10, error = '', message = 'Generating';
const attributes = new Map();
const button = {
  disabled: false, textContent: 'Generate',
  getAttribute: key => attributes.get(key) ?? null,
  setAttribute: (key, value) => attributes.set(key, value),
  click: () => clicks++
};
const announcer = {...button, textContent: ''};
const progress = {
  dataset: {stage: 'running'},
  querySelector(selector) {
    if (selector === 'strong') return {textContent: message};
    if (selector === "[role='progressbar']") return {getAttribute: () => String(percent)};
    if (selector === '.sn-progress-head span') return {textContent: `${percent}%`};
    return {textContent: 'Generating'};
  }
};
const studio = {dataset: {}, get offsetParent() {return active ? {} : null;}};
const app = {
  dataset: {}, classList: {toggle() {}}, addEventListener() {},
  querySelector(selector) {
    lookups++;
    if (selector === '#h3-progress .h3-progress[data-stage]' ||
        selector === '#sn-progress [data-stage]') return progress;
    if (selector === '#h3-progress-announcer .h3-sr-only') return announcer;
    if (selector === '#sn-validation .sn-inline-error') return {textContent: error};
    if (selector === '#sensenova-u15-studio') return studio;
    if (selector.includes('generate button') || selector.includes('cancel button')) return button;
    return null;
  }
};
const api = {
  publish: (source, value) => {if (statusEnabled) calls.push({source, ...value});},
  clear: source => calls.push({source, clear: true})
};
const window = {AikimiStatus: api, addEventListener() {}};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  gradioApp: () => app,
  get_uiCurrentTabContent: () => {lookups++; return {id: active ? 'tab_minimax_h3_studio' : ''};},
  onUiLoaded: fn => {callbacks.loaded = fn;},
  onUiTabChange: fn => {callbacks.tab = fn;},
  onAfterUiUpdate: fn => {callbacks.updated = fn;},
  document: {addEventListener: (name, fn) => {
    if (name === 'keydown') keyHandlers.push(fn);
    else documentListeners.set(name, fn);
  }},
  window
});
function update() {callbacks.updated();}
function key(overrides = {}) {
  const event = {key: 'Enter', ctrlKey: true, target: {closest: () => null},
    preventDefault() {this.defaultPrevented = true;}, ...overrides};
  for (const handler of keyHandlers) handler(event);
}
"""


@unittest.skipUnless(NODE, "Node.js is required for studio status event regressions")
class StudioStatusEventTests(unittest.TestCase):
    def run_fixture(self, kind, script):
        result = subprocess.run(  # noqa: S603 -- Fixed local JavaScript fixture and repository script.
            [NODE, "-e", FIXTURE + script, str(STUDIOS[kind]), kind],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=True,
        )
        return json.loads(result.stdout)

    def test_progress_changes_publish_without_repeating_unchanged_status(self):
        for kind in STUDIOS:
            with self.subTest(studio=kind):
                calls = self.run_fixture(
                    kind,
                    """
update(); update(); percent = 25; update(); update();
progress.dataset.stage = 'complete'; percent = 100; update(); update();
progress.dataset.stage = 'cancelled'; update(); update();
process.stdout.write(JSON.stringify(calls));
""",
                )
                self.assertEqual(len(calls), 4)
                self.assertEqual(calls[0]["progress"], 0.1)
                self.assertEqual(calls[1]["progress"], 0.25)
                self.assertEqual(calls[2]["state"], "completed")
                self.assertTrue(calls[3]["clear"])

    def test_late_status_api_initialization_does_not_lose_current_progress(self):
        for kind in STUDIOS:
            with self.subTest(studio=kind):
                calls = self.run_fixture(
                    kind,
                    """
window.AikimiStatus = null; update();
window.AikimiStatus = api; update(); update();
process.stdout.write(JSON.stringify(calls));
""",
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["progress"], 0.1)

    def test_sensenova_changed_error_detail_is_published(self):
        calls = self.run_fixture(
            "sn",
            """
progress.dataset.stage = 'error'; message = 'Generation failed'; update();
error = 'The runtime was disconnected'; update(); update();
process.stdout.write(JSON.stringify(calls));
""",
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["errorDetails"], "Generation failed")
        self.assertEqual(calls[1]["errorDetails"], "The runtime was disconnected")

    def test_showing_status_again_restores_unchanged_studio_state(self):
        for kind in STUDIOS:
            with self.subTest(studio=kind):
                calls = self.run_fixture(
                    kind,
                    """
progress.dataset.stage = 'error'; message = 'Generation failed'; update();
calls.length = 0; statusEnabled = false;
documentListeners.get('aikimi:status-visibility-change')?.(); update();
statusEnabled = true;
documentListeners.get('aikimi:status-visibility-change')?.(); update(); update();
process.stdout.write(JSON.stringify(calls));
""",
                )
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["state"], "error")
                self.assertEqual(calls[0]["errorDetails"], "Generation failed")

    def test_shortcut_repeats_and_unrelated_keys_do_not_inspect_or_trigger_studio(self):
        for kind in STUDIOS:
            with self.subTest(studio=kind):
                result = self.run_fixture(
                    kind,
                    """
active = true;
key({key: 'a', ctrlKey: false}); key({repeat: true});
key({key: 'Escape', repeat: true}); key({isComposing: true});
key({keyCode: 229}); key({defaultPrevented: true});
const ignored = {lookups, clicks};
key(); key({key: 'Escape', ctrlKey: false});
active = false; key();
process.stdout.write(JSON.stringify({ignored, clicks}));
""",
                )
                self.assertEqual(result["ignored"], {"lookups": 0, "clicks": 0})
                self.assertEqual(result["clicks"], 2)
