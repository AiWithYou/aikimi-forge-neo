import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { runInNewContext } from 'node:vm';

const source = readFileSync(new URL('../../extensions-builtin/clef-studio/assets/clipboard.js', import.meta.url), 'utf8');

function setup() {
    const listeners = new Map();
    const deliveries = [];
    const hint = { textContent: '' };
    const input = {
        disabled: false,
        value: '',
        files: [],
        dispatchEvent(event) { deliveries.push({ type: event.type, files: [...this.files] }); },
    };
    const area = { getClientRects: () => area.visible ? [{}] : [], visible: true };
    const scope = {
        dataset: {},
        focus() {},
        querySelector(selector) {
            if (selector === '#clef-images') return area;
            if (selector === '#clef-images input[type="file"]') return input;
            return null;
        },
        addEventListener(name, handler) {
            listeners.set(name, [...(listeners.get(name) || []), handler]);
        },
    };
    const element = {
        closest: () => scope,
        querySelector: () => hint,
    };
    class DataTransfer {
        files = [];
        items = { add: file => this.files.push(file) };
    }
    const context = { element, DataTransfer, Event, File };
    const initialize = () => runInNewContext(`(() => {${source}\n})()`, context);
    initialize();
    const paste = (files, editable = false) => {
        const event = {
            target: { closest: () => editable ? {} : null },
            clipboardData: { files },
            defaultPrevented: false,
            stopped: false,
            preventDefault() { this.defaultPrevented = true; },
            stopPropagation() { this.stopped = true; },
        };
        for (const handler of listeners.get('paste') || []) handler(event);
        return event;
    };
    return { input, area, deliveries, listeners, hint, initialize, paste };
}

const png = () => new File(['pixels'], 'image.png', { type: 'image/png' });

test('one paste sends all supported images once through the existing file input', () => {
    const state = setup();
    const image = png();
    const jpeg = new File(['pixels'], 'photo.jpg', { type: 'image/jpeg' });
    const event = state.paste([image, new File(['text'], 'note.txt', { type: 'text/plain' }), jpeg]);
    assert.equal(state.deliveries.length, 1);
    assert.equal(state.deliveries[0].type, 'change');
    assert.deepEqual(state.deliveries[0].files, [image, jpeg]);
    assert.equal(event.defaultPrevented, true);
    assert.equal(event.stopped, true);
});

test('normal text paste and editable targets are untouched', () => {
    const state = setup();
    assert.equal(state.paste([]).defaultPrevented, false);
    assert.equal(state.paste([png()], true).defaultPrevented, false);
    assert.equal(state.deliveries.length, 0);
});

test('closed or disabled image input cannot receive a paste', () => {
    const state = setup();
    state.area.visible = false;
    assert.equal(state.paste([png()]).defaultPrevented, false);
    state.area.visible = true;
    state.input.disabled = true;
    assert.equal(state.paste([png()]).defaultPrevented, false);
    assert.equal(state.deliveries.length, 0);
});

test('reinitialization never doubles a paste and successive pastes stay usable', () => {
    const state = setup();
    state.initialize();
    state.paste([png()]);
    state.paste([png()]);
    assert.equal(state.deliveries.length, 2);
    assert.equal(state.listeners.get('paste').length, 1);
});

test('oversize or unsupported images are rejected before upload with a visible reason', () => {
    const state = setup();
    state.paste([{ name: 'large.png', type: 'image/png', size: 33 * 1024 ** 2 }]);
    assert.equal(state.deliveries.length, 0);
    assert.match(state.hint.textContent, /32MiB/);
    state.paste([new File(['pixels'], 'image.gif', { type: 'image/gif' })]);
    assert.equal(state.deliveries.length, 0);
    assert.match(state.hint.textContent, /PNG/);
});
