"""Run the shipped canvas loaders with deterministic asynchronous Image completion."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "modules_forge/forge_canvas/canvas.js"

HARNESS = r"""
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const pending = [];
let marks = null;
const ctx = {
    clearRect() { marks = null; },
    drawImage(image) { marks = image.src; },
};
const canvas = {width: 256, height: 256, getContext: () => ctx};
const input = {value: null};
const hint = {style: {}};
const context = {
    URL,
    window: {location: {href: 'http://fixture.test/', origin: 'http://fixture.test'}},
    document: {
        querySelector: () => null,
        createElement: () => ({}),
        getElementById: id => id.startsWith('drawingCanvas_') ? canvas : id.startsWith('imageInput_') ? input : hint,
    },
    Image: class {
        constructor() { this.width = 256; this.height = 256; pending.push(this); }
    },
};
vm.runInNewContext(fs.readFileSync(process.argv[1], 'utf8') + ';globalThis.ForgeCanvas = ForgeCanvas;', context);
// Use the production constructor and loaders, without unrelated DOM listeners/layout.
context.ForgeCanvas.prototype.init = () => {};
const editor = new context.ForgeCanvas('fixture');
editor.adjustInitialPositionAndScale = () => {};
editor.drawImage = () => {};
editor.saveState = () => {};
editor.updateUndoRedoButtons = () => {};
const backgroundWrites = [];
editor.background_gradio_bind.set_value = value => backgroundWrites.push(value);
const source = name => 'forge-file:' + JSON.stringify({path: '/unused', url: '/' + name + '.png'});
const finish = image => image.onload();

switch (process.argv[2]) {
case 'background_order':
    editor.loadImage(source('first'));
    editor.loadImage(source('latest'));
    finish(pending[1]);
    finish(pending[0]);
    assert.equal(editor.img, 'http://fixture.test/latest.png');
    assert.equal(backgroundWrites.length, 0); // The Qwen server value remains the latest file.
    break;
case 'background_clear':
    editor.loadImage(source('first'));
    editor.loadImage(null);
    finish(pending[0]);
    assert.equal(editor.img, null);
    break;
case 'drawing_order':
    editor.loadDrawing('old strokes');
    editor.loadDrawing('new strokes');
    finish(pending[1]);
    finish(pending[0]);
    assert.equal(marks, 'new strokes');
    break;
case 'drawing_clear':
    editor.loadDrawing('old strokes');
    editor.loadDrawing(null);
    finish(pending[0]);
    assert.equal(marks, null);
    break;
case 'drawing_source_changed':
    editor.loadImage(source('first'));
    finish(pending[0]);
    editor.loadDrawing('strokes for first');
    editor.loadImage(source('latest'));
    finish(pending[2]);
    finish(pending[1]);
    assert.equal(editor.img, 'http://fixture.test/latest.png');
    assert.equal(marks, null);
    break;
case 'latest_drawing_for_latest_source':
    editor.loadImage(source('first'));
    editor.loadImage(source('latest'));
    finish(pending[1]);
    editor.loadDrawing('strokes for latest');
    finish(pending[2]);
    finish(pending[0]);
    assert.equal(editor.img, 'http://fixture.test/latest.png');
    assert.equal(marks, 'strokes for latest');
    break;
case 'same_source_preserves_pending_drawing':
    editor.loadImage(source('first'));
    finish(pending[0]);
    editor.loadDrawing('strokes for first');
    editor.loadImage(source('first'));
    finish(pending[2]);
    finish(pending[1]);
    assert.equal(marks, 'strokes for first');
    break;
default:
    throw new Error('Unknown regression case');
}
"""


class CanvasLoadOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.node = shutil.which("node")
        if cls.node is None:
            raise unittest.SkipTest("Node.js is required for the JavaScript regression")

    def run_case(self, case):
        with tempfile.TemporaryDirectory(prefix="canvas-load-order-") as directory:
            result = subprocess.run(  # noqa: S603
                [self.node, "-e", HARNESS, str(SCRIPT), case],
                cwd=directory,
                capture_output=True,
                check=False,
                text=True,
                timeout=10,
            )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_latest_background_wins_out_of_order_decode(self):
        self.run_case("background_order")

    def test_clearing_background_invalidates_pending_decode(self):
        self.run_case("background_clear")

    def test_latest_drawing_wins_out_of_order_decode(self):
        self.run_case("drawing_order")

    def test_clearing_drawing_invalidates_pending_decode(self):
        self.run_case("drawing_clear")

    def test_new_source_invalidates_pending_drawing_for_previous_source(self):
        self.run_case("drawing_source_changed")

    def test_new_drawing_for_latest_source_survives_obsolete_background_decode(self):
        self.run_case("latest_drawing_for_latest_source")

    def test_same_source_preserves_pending_drawing(self):
        self.run_case("same_source_preserves_pending_drawing")


if __name__ == "__main__":
    unittest.main()
