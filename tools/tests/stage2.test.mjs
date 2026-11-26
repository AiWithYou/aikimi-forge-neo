import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

function script(path) {
    return readFileSync(new URL("../../" + path, import.meta.url), "utf8");
}

test("resolution rounds half steps up", () => {
    const ctx = vm.createContext({onOptionsAvailable() {}});
    vm.runInContext(script("javascript/dimRounding.js"), ctx);
    assert.equal(ctx.sRound(511, 16), 512);
    assert.equal(ctx.sRound(520, 16), 528);
});

test("layout compiles and restores pixel coordinates and captions", () => {
    const dir = "extensions-builtin/sd_forge_json_builder/javascript/";
    const ctx = vm.createContext({});
    for (const name of ["compile", "decompile"]) vm.runInContext(script(dir + name + ".js"), ctx);
    const draft = {width: 512, height: 256, bbox_coord_space: "abs", bbox_format: "xyxy",
        summary: {caption: "park", canvas: "single"}, items: [{label:"object", caption:"tree", bbox:[32,16,256,128]}]};
    const result = ctx.decompilePrompt(ctx.compilePrompt(JSON.stringify(draft)), 512, 256);
    assert.deepEqual(Array.from(result.items[0].bbox), draft.items[0].bbox);
    assert.equal(result.summary.caption, "park");
    for (const label of ["JSONをコピー", "MOSS Image のプロンプト欄へ送る", "現在のプロンプト欄へ挿入"]) {
        assert.ok(script(dir + "main.js").includes(label));
    }
});
test("MOSS transfer and copy use native centered JSON, insertion preserves neighboring text", () => {
    const dir = "extensions-builtin/sd_forge_json_builder/javascript/";
    const ctx = vm.createContext({onUiLoaded() {}, onUiTabChange() {}});
    for (const name of ["compile", "main"]) vm.runInContext(script(dir + name + ".js"), ctx);
    const draft = {width: 512, height: 768, summary: {caption: "tree (green)", canvas:"single"},
        bbox_coord_space:"abs", bbox_format:"xyxy", items:[{label:"object",caption:"tree",bbox:[32,16,256,128]}]};
    const text = ctx.compilePrompt(JSON.stringify(draft), false);
    const compiled = JSON.parse(text);
    assert.deepEqual(compiled.items[0].bbox, [-368,-224,-256,0]);
    assert.equal(compiled.summary.caption, "tree (green)");
    const field = {value:"abc", selectionStart:1, selectionEnd:2, setRangeText(v, s, e) {
        this.value = this.value.slice(0,s) + v + this.value.slice(e);
    }};
    ctx.writeBuilderPrompt(field, text, true, () => {});
    assert.equal(field.value, "a" + text + "c");
    ctx.writeBuilderPrompt(field, text, false, () => {});
    assert.equal(field.value, text);
});

test("MOSS transfer opens the tab and waits for Gradio to mount its input", async () => {
    let active = false;
    let field;
    const ctx = vm.createContext({onUiLoaded() {}, onAfterUiUpdate() {}, onUiTabChange() {},
        get_uiTopTabButton: () => ({click() { active = true; }}),
        document: {querySelector: () => field},
        setTimeout(fn) { field = {value: ""}; fn(); }});
    vm.runInContext(script("extensions-builtin/sd_forge_json_builder/javascript/main.js"), ctx);
    assert.equal(await ctx.openMossBuilderPrompt(), field);
    assert.ok(active);
    assert.ok(field);
});

test("JSON Builder requests styles under the current WebUI subpath", async () => {
    for (const suffix of ["/subpath/", "/subpath?view=styles#editor", "/"]) {
        let requested;
        const location = new URL("https://example.invalid" + suffix);
        const ctx = vm.createContext({onUiLoaded() {}, onUiTabChange() {}, URL, window: {location},
            fetch: url => { requested = new URL(url, location); return {json: () => []}; }});
        vm.runInContext(script("extensions-builtin/sd_forge_json_builder/javascript/main.js"), ctx);
        await ctx.fetchBuilderStyles();
        const prefix = suffix === "/" ? "" : "/subpath";
        assert.equal(requested.href, "https://example.invalid" + prefix + "/internal/json_builder_prompt_styles");
    }
});
