import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {runInNewContext} from "node:vm";
import test from "node:test";

test("studio insertion updates the bound Gradio input", () => {
    const input = {value: "", dispatchEvent(event) { this.event = event.type; }};
    const context = {window: {}, Event, onUiUpdate() {}, onUiLoaded() {}, document: {addEventListener() {}}};
    runInNewContext(readFileSync("extensions-builtin/aikimi-ui/javascript/aikimi_workspace.js", "utf8"), context);
    context.window.AikimiWorkspace.setInput(input, "夕焼け");
    assert.equal(input.value, "夕焼け");
    assert.equal(input.event, "input");
});
