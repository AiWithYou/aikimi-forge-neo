import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NODE = shutil.which("node")


@unittest.skipUnless(NODE, "Node.js is required for the draft event regression")
class SenseNovaDraftTests(unittest.TestCase):
    def test_programmatic_prompt_updates_are_saved_without_erasing_initial_draft(self):
        script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const key = 'forge-neo:sensenova-u15:prompt-draft:v1';
const storage = new Map([[key, 'Saved draft']]);
const prompt = {value: '', setAttribute() {}};
const counter = {dataset: {}};
const status = {textContent: ''};
const timers = new Map();
let timerId = 0, loaded, updated, writes = 0;
const app = {
  querySelector(selector) {
    if (selector === '#sn-prompt textarea') return prompt;
    if (selector === '#sn-prompt-count') return counter;
    if (selector === '#sn-draft-status') return status;
    return null;
  },
  addEventListener() {}, classList: {toggle() {}}
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
  gradioApp: () => app, onUiLoaded: fn => {loaded = fn;},
  onUiTabChange() {}, onAfterUiUpdate: fn => {updated = fn;},
  document: {addEventListener() {}},
  window: {
    addEventListener() {},
    setTimeout(fn) {timers.set(++timerId, fn); return timerId;},
    clearTimeout(id) {timers.delete(id);},
    localStorage: {
      setItem(k, value) {storage.set(k, value); writes++;},
      removeItem(k) {storage.delete(k); writes++;}
    }
  }
});
function flush() {
  const callbacks = Array.from(timers.values()); timers.clear();
  for (const callback of callbacks) callback();
}
loaded(); flush();
const initial = storage.get(key);
prompt.value = 'Saved draft'; updated(); flush();
prompt.value = 'Saved draft\n\nUse Image-1 as the primary reference.';
updated(); flush();
const appended = storage.get(key), count = writes;
updated(); flush();
const stable = writes === count;
prompt.value = ''; updated(); flush();
process.stdout.write(JSON.stringify({initial, appended, stable, cleared: !storage.has(key)}));
"""
        result = subprocess.run(  # noqa: S603 -- 固定したローカルJS fixtureをNodeで実行する。
            [
                NODE,
                "-e",
                script,
                str(ROOT / "extensions-builtin/sensenova-u15-studio/javascript/sensenova_u15_studio.js"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
            check=True,
        )
        data = json.loads(result.stdout)
        self.assertEqual(data["initial"], "Saved draft")
        self.assertEqual(data["appended"], "Saved draft\n\nUse Image-1 as the primary reference.")
        self.assertTrue(data["stable"])
        self.assertTrue(data["cleared"])
