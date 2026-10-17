import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

class Element {
    children = [];
    classes = new Set();
    classList = {
        add: name => this.classes.add(name),
        remove: name => this.classes.delete(name),
        contains: name => this.classes.has(name),
    };
    appendChild(child) {
        child.parentNode = child.parentElement = this;
        this.children.push(child);
    }
    insertBefore(child, before) {
        child.parentNode = child.parentElement = this;
        const index = this.children.indexOf(before);
        this.children.splice(index < 0 ? this.children.length : index, 0, child);
    }
    addEventListener() {}
    removeEventListener() {}
}

function load(file, extra = {}) {
    const context = {document: {createElement: () => new Element(), createTextNode: () => new Element()}, ...extra};
    vm.runInNewContext(fs.readFileSync(new URL(`../../javascript/${file}`, import.meta.url), 'utf8'), context);
    return context;
}

test('profiler rows and grouped rows do not overwrite a global loop index', () => {
    const context = load('profilerVisualization.js', {i: 'extension-owned'});
    const table = context.createVisualizationTable({root: 1, 'root/child': 0.2, 'root/small': 0.01}, 0.05);
    assert.ok(table.children.length > 3);
    assert.equal(context.i, 'extension-owned');
});

test('comma-containing profiler paths remain separate branches', () => {
    const context = load('profilerVisualization.js');
    const table = context.createVisualizationTable({
        'a,b': 1, 'a,b/c': 1, 'a,b/c/left': 0.5,
        a: 1, 'a/b,c': 1, 'a/b,c/right': 0.5,
    });
    const cells = table.children.flatMap(row => row.children.map(cell => cell.textContent));
    assert.equal(cells.filter(value => value === 'left').length, 1);
    assert.equal(cells.filter(value => value === 'right').length, 1);
    assert.equal(table.children.length, 7);
});

test('settings comments preserve globals and insert both before and after hints', () => {
    const parent = new Element();
    const span = new Element();
    parent.appendChild(span);
    const setting = new Element();
    setting.id = 'setting_example';
    setting.classes.add('gradio-checkbox');
    setting.querySelector = () => span;
    let callback;
    const context = load('ui_settings_hints.js', {
        comment: 'extension-owned',
        opts: {_comments_before: {example: 'before'}, _comments_after: {example: 'after'}},
        onOptionsChanged: fn => { callback = fn; },
        gradioApp: () => ({querySelectorAll: () => [setting]}),
    });
    callback();
    assert.equal(context.comment, 'extension-owned');
    assert.deepEqual(parent.children.filter(child => child.className === 'settings-comment').map(child => child.innerHTML), ['before', 'after']);
});

test('metadata hierarchy preserves false-like values and prototype-named keys', () => {
    const context = load('extraNetworks.js', {
        window: {addEventListener() {}}, onUiLoaded() {}, onUiUpdate() {},
    });
    const input = JSON.parse('{"zero":0,"zero/child":1,"flag":false,"flag/child":2,"__proto__":"text","constructor":{"child":3},"toString":{"child":4}}');
    const flattened = context.extraNetworksFlattenMetadata(input);
    assert.equal(flattened.zero, 0);
    assert.equal(flattened.flag, false);
    assert.equal(flattened.__proto__, 'text');
    assert.equal(Object.hasOwn(flattened, 'constructor'), true);
    assert.equal(Object.hasOwn(flattened, 'toString'), true);
    assert.equal(flattened['constructor/child'], 3);
    assert.equal(flattened['toString/child'], 4);
});
