import json
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_JS = ROOT / "extensions-builtin" / "aikimi-ui" / "javascript" / "aikimi_image_workspace.js"

FIXTURE = r"""
const fs = require("fs");
const vm = require("vm");
const listeners = new Map();
const frames = [];
let current;
let feature;
let forgeClicks = 0;
function surface(key, editable = true) {
    const buttons = ["txt2img", "img2img"].map((mode, i) => ({
        id: `${key}-${mode}-button`, disabled: i === 1 && !editable,
        selected: i === 0, clicks: 0,
        getAttribute(name) { return name === "aria-selected" ? String(this.selected) : null; },
        closest(selector) { return selector === ".aikimi-image-modes" ? tabs : null; },
        click() {
            this.clicks++;
            buttons.forEach(button => button.selected = button === this);
        }
    }));
    const list = {querySelectorAll: () => buttons};
    const tabs = {querySelector: () => list};
    const ui = {querySelector: () => tabs};
    return {id: key, buttons, querySelector: () => ui};
}
const qwen = surface("qwen21-common");
const ming = surface("ming-common", false);
const forge = {id: "tab_txt2img", querySelector: () => null};
const root = {querySelectorAll: () => [], querySelector: () => null};
const context = {
    document: {
        ...root,
        addEventListener(name, callback) {
            const group = listeners.get(name) || [];
            group.push(callback); listeners.set(name, group);
        }
    },
    window: {AikimiTabs: {getActiveFeature: () => feature, getActiveContainer: () => current}},
    gradioApp: () => root,
    MutationObserver: class {observe() {}},
    requestAnimationFrame: callback => frames.push(callback),
    onUiLoaded() {}, onUiUpdate() {}, onAfterUiUpdate() {},
    get_uiTopTabButton: id => ({
        getAttribute: () => String(id === current.id),
        click() { forgeClicks++; feature = null; }
    })
};
vm.runInNewContext(fs.readFileSync(process.argv[1], "utf8"), context);
function emit(name, event) { (listeners.get(name) || []).forEach(callback => callback(event)); }
function select(container, key) {
    current = container; feature = key;
    emit("aikimi:feature-tab-change", {detail: {feature: key, ready: true}});
    while (frames.length) frames.shift()();
}
current = qwen; feature = "qwen_image21";
qwen.buttons[1].click();
emit("click", {target: {closest: selector => selector === 'button[role="tab"]' ? qwen.buttons[1] : null}});
select(forge, "anima38");
const alias = {feature, container: current.id, forgeClicks};
select(ming, "ming_image");
const textOnlyMode = ming.buttons.map(button => button.selected);
qwen.buttons[0].click();
select(qwen, "qwen_image21");
process.stdout.write(JSON.stringify({alias, textOnlyMode, qwenMode: qwen.buttons.map(button => button.selected)}));
"""


@unittest.skipUnless(shutil.which("node"), "Node.js is required for the frontend lifecycle fixture")
class ImageWorkspaceJavascriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        result = subprocess.run(  # noqa: S603 - fixed local fixture and repository script
            [shutil.which("node"), "-e", FIXTURE, str(WORKSPACE_JS)],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
        cls.result = json.loads(result.stdout)

    def test_studio_mode_memory_does_not_move_the_activated_forge_host(self):
        self.assertEqual(self.result["alias"], {"feature": "anima38", "container": "tab_txt2img", "forgeClicks": 0})

    def test_text_only_studio_does_not_erase_the_desired_edit_mode(self):
        self.assertEqual(self.result["textOnlyMode"], [True, False])
        self.assertEqual(self.result["qwenMode"], [False, True])

    def test_model_help_waits_for_lazy_settings_and_survives_remount(self):
        script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
let loaded, updated, modules = null, token = false;
const preset = {value: 'sd'};
const titles = [];
function mount() {
    const title = {links: [], appendChild(link) { this.links.push(link); link.isConnected = true; }};
    titles.push(title);
    modules = {querySelector(selector) {
        if (selector === 'span') return title;
        if (selector === 'div.wrap-inner') return {querySelector: () => token};
        return null;
    }};
    return title;
}
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8'), {
    onUiLoaded: fn => loaded = fn, onAfterUiUpdate: fn => updated = fn,
    document: {
        getElementById(id) {
            return id === 'setting_sd_modules' ? modules : {querySelector: () => preset};
        },
        createElement: () => ({style: {}, isConnected: false})
    }
});
loaded(); updated(); // lazy settings have not mounted yet
const first = mount(); updated(); updated();
assert.equal(first.links.length, 1);
assert.equal(first.links[0].style.visibility, 'hidden');
preset.value = 'krea'; updated();
assert.equal(first.links[0].style.visibility, 'visible');
token = true; updated();
assert.equal(first.links[0].style.visibility, 'hidden');
first.links[0].isConnected = false;
const second = mount(); updated(); updated();
assert.equal(second.links.length, 1);
assert.ok(second.links[0].href.endsWith('/wiki/Download-Models'));
"""
        result = subprocess.run(  # noqa: S603 -- Fixed local lifecycle fixture and repository JavaScript.
            [shutil.which("node"), "-e", script, str(ROOT / "javascript/modelHelp.js")],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
