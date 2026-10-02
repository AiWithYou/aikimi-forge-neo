import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const script = fs.readFileSync(new URL('../../extensions-builtin/qwen-image21-studio/javascript/qwen_image21_outpaint_canvas.js', import.meta.url), 'utf8');

// Exercise the shipped event handlers without launching a browser or a model.
function editor(pads = [128, 128, 128, 128]) {
    const events = {}, hooks = {}, input = {}, output = {textContent: ''};
    const classes = {toggle() {}, add() {}, remove() {}, contains() {return false;}};
    const ratio = {dataset: {ratio: '1'}, setAttribute() {}};
    const frame = {style: {}}, picture = {style: {}}, view = {clientWidth: 1208, clientHeight: 824};
    const root = {
        dataset: {layout: JSON.stringify({w: 736, h: 512, pads})}, classList: classes, isConnected: true,
        contains: () => false,
        querySelector: selector => ({'.qoc-viewport': view, '.qoc-frame': frame, '.qoc-picture': picture, '.qoc-size': output})[selector],
        querySelectorAll: selector => selector === '[data-ratio]' ? [ratio] : [],
    };
    const control = {
        dataset: {}, classList: classes, closest: selector => selector === '.qoc' ? root : control,
        setPointerCapture() {}, hasPointerCapture() {return true;}, releasePointerCapture() {},
    };
    const target = {closest: selector => selector.includes('.qoc-h,') ? control : root};
    const app = {querySelectorAll: () => [root], querySelector: () => input};
    vm.runInNewContext(script, {
        console, document: {activeElement: null, addEventListener: (name, fn) => {events[name] = fn;}},
        window: {addEventListener: (name, fn) => {hooks[name] = fn;}},
        ResizeObserver: class {observe() {} disconnect() {}}, MutationObserver: class {observe() {}},
        gradioApp: () => app, updateInput() {}, onUiLoaded: fn => fn(), onAfterUiUpdate: fn => {hooks.update = fn;},
    });
    const pointer = (name, x = 0, y = 0) => events[name]({target, button: 0, pointerId: 1, clientX: x, clientY: y, preventDefault() {}});
    return {pointer, output, input, root, hooks, control, frame,
        pads: () => JSON.parse(input.value).pads,
        key: key => events.keydown({target: {closest: () => control}, key, shiftKey: false, preventDefault() {}}),
    };
}

test('half-pixel source movement cannot enlarge the output by one grid cell', () => {
    const e = editor(); // fitted scale is exactly 1 px per source pixel
    const before = e.output.textContent;
    e.pointer('pointerdown'); e.pointer('pointermove', 0.5, 0.5); e.pointer('pointerup', 0.5, 0.5);
    assert.equal(e.output.textContent, before);
    const [l, t, r, b] = e.pads();
    assert.equal(l + r, 256); assert.equal(t + b, 256);
});

test('moving an unaligned canvas to its edge consumes all displayed padding', () => {
    const e = editor([128, 128, 268, 128]); // aligned width: 1152, horizontal padding: 416
    e.pointer('pointerdown'); e.pointer('pointermove', -500, 0); e.pointer('pointerup', -500, 0);
    assert.deepEqual(e.pads(), [0, 128, 416, 128]);
    assert.equal(e.output.textContent, '1152 × 768 px');
});

test('keyboard source movement uses the same aligned boundary as dragging', () => {
    const e = editor([128, 128, 268, 128]);
    e.key('ArrowLeft');
    assert.deepEqual(e.pads(), [106, 128, 310, 128]);
    assert.equal(e.output.textContent, '1152 × 768 px');
});

test('cancelling a source drag restores the original four margins', () => {
    const e = editor([128, 128, 268, 128]);
    e.pointer('pointerdown'); e.pointer('pointermove', 50, 20); e.pointer('pointercancel', 50, 20);
    assert.deepEqual(e.pads(), [128, 128, 268, 128]);
});

test('recycled Gradio HTML is initialized again even with unchanged geometry', () => {
    const e = editor();
    e.frame.style = {}; e.output.textContent = ''; e.hooks.update();
    assert.equal(e.output.textContent, '992 × 768 px');
    assert.ok(e.frame.style.width);
});
