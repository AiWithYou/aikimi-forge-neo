import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(new URL('../../extensions-builtin/flux3-klein-studio/javascript/flux3_layout.js', import.meta.url), 'utf8');
const context = {module: {exports: {}}, onUiLoaded() {}, onUiUpdate() {}};
vm.runInNewContext(source, context);
const editor = context.module.exports;
const plain = value => JSON.parse(JSON.stringify(value));

test('reverse pointer drags become clamped normalized yxyx rectangles', () => {
    assert.deepEqual(plain(editor.rectangle([0.8, 0.7], [0.1, 0.2])), [0.2, 0.1, 0.7, 0.8]);
    assert.deepEqual(plain(editor.rectangle([-1, -0.5], [2, 1.5])), [0, 0, 1, 1]);
    assert.equal(editor.rectangle([0.3, 0.3], [0.301, 0.8]), null);
});

test('adding a normal rectangle preserves advanced edit provenance and null deletion', () => {
    const rows = [{id: 'obj_1', desc: 'remove', from: 'ref_image_0', src_bbox: [0, 0, 1, 1], tgt_bbox: null}];
    const before = JSON.stringify(rows);
    const updated = editor.appendRectangle(rows, [0.1, 0.2, 0.3, 0.4]);
    assert.equal(JSON.stringify(rows), before);
    assert.deepEqual(plain(updated[0]), rows[0]);
    assert.equal(updated[1].id, 'obj_2');
    assert.deepEqual(plain(updated[1].bbox), [0.1, 0.2, 0.3, 0.4]);
    assert.ok(updated[1].desc);
});

test('clear removes drawn rectangles while keeping manual edit schemas', () => {
    const edit = {id: 'edit', desc: 'new', from: 'target', src_bbox: null, tgt_bbox: [0, 0, 1, 1]};
    const rows = [{id: 'obj_1', desc: 'balloon', bbox: [0, 0, 1, 1]}, edit];
    assert.deepEqual(plain(editor.clearRectangles(rows)), [edit]);
});

test('invalid manual JSON is rejected without silently replacing it', () => {
    assert.equal(editor.parseRows('{broken'), null);
    assert.equal(editor.parseRows('{}'), null);
    assert.equal(editor.parseRows('[null]'), null);
    assert.deepEqual(plain(editor.parseRows('[]')), []);
});
