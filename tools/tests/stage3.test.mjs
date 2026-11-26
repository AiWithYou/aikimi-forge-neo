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

test("MOSS transfer overwrites and clears both bound fields", () => {
    const field = () => ({value: "old", getClientRects: () => [1], dispatchEvent() {}});
    const positive = field(), negative = field();
    const document = {addEventListener() {}, querySelectorAll(selector) {
        if (selector.includes("#moss-image-negative-prompt")) return [negative];
        if (selector.includes("#moss-image-prompt")) return [positive];
        return [];
    }};
    const context = {window: {AikimiTabs: {getActiveFeature: () => "moss_image", getActiveContainer: () => document}},
        document, Event, onUiUpdate() {}, onUiLoaded() {}};
    runInNewContext(readFileSync("extensions-builtin/aikimi-ui/javascript/aikimi_workspace.js", "utf8"), context);
    for (const value of ["rain", ""]) {
        context.window.AikimiWorkspace.insert("clouds", value);
        assert.equal(positive.value, "clouds");
        assert.equal(negative.value, value);
    }
});
