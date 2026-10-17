import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(new URL("../../javascript/dragdrop.js", import.meta.url), "utf8");

function fixture({ ok = true } = {}) {
    const listeners = {};
    const requests = [];
    const changes = [];
    const fileInput = { dispatchEvent: event => changes.push(event.type) };
    const context = {
        URL,
        Event,
        File,
        DataTransfer: class {
            files = [];
            items = { add: file => this.files.push(file) };
        },
        window: {
            document: { addEventListener: (name, handler) => { listeners[name] = handler; } },
            addEventListener: () => {},
        },
        document: { addEventListener: () => {} },
        get_tab_index: () => 0,
        gradioApp: () => ({ getElementById: () => ({ querySelector: () => fileInput }) }),
        fetch: async (url, options) => {
            requests.push({ url, options });
            return { ok, blob: async () => new Blob(["image bytes"], { type: "image/png" }) };
        },
        console: { error: () => {} },
    };
    vm.runInNewContext(source, context);
    async function drop(url, files = []) {
        await listeners.drop({
            composedPath: () => [{ placeholder: "Prompt", closest: () => null }],
            dataTransfer: {
                files,
                items: [],
                getData: type => type === "text/uri-list" ? url : "",
            },
            preventDefault() {},
            stopPropagation() {},
        });
    }
    return { drop, requests, changes, fileInput };
}

test("dropped image URLs use the fixed POST endpoint without fetching the supplied URL", async () => {
    const state = fixture();
    const url = "https://images.example/photo.png";
    await state.drop(url);
    assert.equal(state.requests.length, 1);
    assert.equal(state.requests[0].url, "internal/remote-image");
    assert.equal(state.requests[0].options.method, "POST");
    assert.equal(state.requests[0].options.credentials, "same-origin");
    assert.equal(JSON.parse(state.requests[0].options.body).url, url);
    assert.deepEqual(state.changes, ["change"]);
    assert.equal(state.fileInput.files[0].name, "image.png");
});

test("rejected remote image requests do not upload a file", async () => {
    const state = fixture({ ok: false });
    await state.drop("http://127.0.0.1:7861/internal/private");
    assert.equal(state.requests[0].url, "internal/remote-image");
    assert.deepEqual(state.changes, []);
    assert.equal(state.fileInput.files, undefined);
});

test("dropping an image file does not make any network request", async () => {
    const state = fixture();
    const files = [new File(["image"], "local.png", { type: "image/png" })];
    await state.drop("", files);
    assert.deepEqual(state.requests, []);
    assert.deepEqual(state.changes, ["change"]);
    assert.equal(state.fileInput.files, files);
});
