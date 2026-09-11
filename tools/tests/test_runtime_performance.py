"""Focused CPU/offline checks for allocation and extra-network UI changes."""

import ast
import json
import shutil
import subprocess
import unittest
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parents[2]


def load_nodes(path, names, namespace, class_name=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = tree.body
    if class_name:
        nodes = next(node for node in nodes if isinstance(node, ast.ClassDef) and node.name == class_name).body
    selected = [node for node in nodes if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert len(selected) == len(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


class RuntimePerformanceTests(unittest.TestCase):
    def gc_fixture(self, **overrides):
        states = Enum("VRAMState", "NORMAL_VRAM HIGH_VRAM LOW_VRAM NO_VRAM DISABLED SHARED")
        memory = SimpleNamespace(
            is_nvidia=Mock(return_value=True),
            DISABLE_SMART_MEMORY=False,
            signal_empty_cache=False,
            VRAMState=states,
            vram_state=states.NORMAL_VRAM,
            soft_empty_cache=Mock(),
        )
        for key, value in overrides.items():
            setattr(memory, key, value)
        ns = load_nodes("modules/devices.py", {"torch_gc"}, {"memory_management": memory, "device": SimpleNamespace(type="cuda")})
        return ns, memory

    def test_default_cleanup_preserves_existing_callers(self):
        ns, memory = self.gc_fixture()
        ns["torch_gc"]()
        memory.soft_empty_cache.assert_called_once_with()

    def test_routine_cleanup_reuses_normal_and_high_vram_cache(self):
        ns, memory = self.gc_fixture()
        for state in (memory.VRAMState.NORMAL_VRAM, memory.VRAMState.HIGH_VRAM):
            memory.vram_state = state
            ns["torch_gc"](force=False)
        memory.soft_empty_cache.assert_not_called()

    def test_low_vram_and_explicit_memory_pressure_still_clean_up(self):
        ns, memory = self.gc_fixture()
        for state in (memory.VRAMState.LOW_VRAM, memory.VRAMState.NO_VRAM):
            memory.vram_state = state
            ns["torch_gc"](force=False)
        memory.vram_state = memory.VRAMState.NORMAL_VRAM
        memory.signal_empty_cache = True
        ns["torch_gc"](force=False)
        memory.signal_empty_cache = False
        memory.DISABLE_SMART_MEMORY = True
        ns["torch_gc"](force=False)
        self.assertEqual(memory.soft_empty_cache.call_count, 4)

    def test_non_nvidia_and_non_cuda_cleanup_is_unchanged(self):
        ns, memory = self.gc_fixture()
        memory.is_nvidia.return_value = False
        ns["torch_gc"](force=False)
        memory.is_nvidia.return_value = True
        for device_type in ("cpu", "mps", "xpu"):
            ns["device"].type = device_type
            ns["torch_gc"](force=False)
        self.assertEqual(memory.soft_empty_cache.call_count, 4)

    def test_only_four_generation_boundaries_opt_out(self):
        tree = ast.parse((ROOT / "modules/processing.py").read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == "devices.torch_gc"]
        routine = [node for node in calls if any(k.arg == "force" and isinstance(k.value, ast.Constant) and k.value.value is False for k in node.keywords)]
        self.assertEqual(len(routine), 4)
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "process_images_inner")
        self.assertTrue(all(main.lineno <= node.lineno <= main.end_lineno for node in routine))
        self.assertGreater(len(calls), len(routine))

    def test_empty_disk_cache_is_reused_without_truthiness_queries(self):
        class EmptyCache:
            def __bool__(self):
                raise AssertionError("Cache existence must not query its length")

        instance = EmptyCache()
        factory = Mock(return_value=instance)
        import os
        import threading

        ns = load_nodes("modules/cache.py", {"cache"}, {"caches": {}, "cache_lock": threading.Lock(), "cache_dir": "cache", "os": os, "diskcache": SimpleNamespace(Cache=factory)})
        self.assertIs(ns["cache"]("metadata"), instance)
        self.assertIs(ns["cache"]("metadata"), instance)
        factory.assert_called_once()

    def test_close_drops_own_intermediates_without_mutating_retained_lists(self):
        ns = load_nodes("modules/processing.py", {"close"}, {"opts": SimpleNamespace(persistent_cond_cache=False)}, "StableDiffusionProcessing")
        latents, pixels = [object()], [object()]
        p = SimpleNamespace(sampler=object(), c=object(), uc=object(), latents_after_sampling=latents, pixels_after_sampling=pixels, modified_noise=object(), clear_prompt_cache=Mock())
        ns["close"](p)
        self.assertEqual(p.latents_after_sampling, [])
        self.assertEqual(p.pixels_after_sampling, [])
        self.assertEqual(len(latents), 1)
        self.assertEqual(len(pixels), 1)
        self.assertIsNone(p.modified_noise)
        p.clear_prompt_cache.assert_called_once_with()
        ns["opts"].persistent_cond_cache = True
        ns["close"](p)
        p.clear_prompt_cache.assert_called_once_with()

    @unittest.skipIf(torch is None, "PyTorch is needed for CPU tensor checks")
    def test_normalization_is_exact_contiguous_and_does_not_modify_inputs(self):
        ns = load_nodes("modules/processing.py", {"_normalize_decoded_batch"}, {"torch": torch})
        values = [-2.0, -1.0, -0.5, 0.0, 0.25, 1.0, 2.0, float("nan"), float("inf"), -float("inf"), 0.12345, 0.9999]
        with torch.inference_mode():
            for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                for shape in ((2, 3, 1, 2), (2, 1, 3, 1, 2)):
                    original = torch.tensor(values, dtype=dtype).reshape(shape).transpose(-1, -2)
                    saved = original.clone()
                    reference = torch.clamp((torch.stack(list(original)).float() + 1.0) / 2.0, 0.0, 1.0)
                    for samples in (original, list(original)):
                        result = ns["_normalize_decoded_batch"](samples)
                        torch.testing.assert_close(result, reference, rtol=0, atol=0, equal_nan=True)
                        torch.testing.assert_close(original, saved, rtol=0, atol=0, equal_nan=True)
                        self.assertTrue(result.is_contiguous())
                        self.assertNotEqual(result.data_ptr(), original.data_ptr())

    @unittest.skipIf(torch is None, "PyTorch is needed for CPU tensor checks")
    def test_tensor_decode_fast_path_keeps_default_list_api(self):
        decoded = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        decode = Mock(return_value=decoded)
        ns = load_nodes("modules/processing.py", {"DecodedSamples", "decode_latent_batch"}, {"decode_first_stage": decode})
        result = ns["decode_latent_batch"]("model", "latent", target_device="cpu")
        self.assertIsInstance(result, list)
        self.assertTrue(result.already_decoded)
        torch.testing.assert_close(torch.stack(result), decoded)
        fast = ns["decode_latent_batch"]("model", "latent", target_device="cpu", return_tensor=True)
        self.assertIs(fast, decoded)

    def test_extra_networks_runtime(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required for JavaScript checks")
        source = (ROOT / "javascript/extraNetworks.js").read_text(encoding="utf-8")
        harness = r'''
const assert = require("node:assert/strict");
const vm = require("node:vm");
const timers = new Map(); let timerId = 0; let scans = 0;
const styles = new Map();
const document = {
    getElementById: id => styles.get(id),
    createElement: () => ({textContent: "", appendChild(node) { this.textContent += node.textContent; }}),
    createTextNode: text => ({textContent: text}),
    createDocumentFragment: () => ({children: [], appendChild(c) { this.children.push(c); }}),
    head: {appendChild(s) { styles.set(s.id, s); }, removeChild(s) { styles.delete(s.id); }},
};
function card(text, version, searchOnly = false) {
    const classes = new Set();
    return {text, version, searchOnly, dataset: {sortDefault: text},
        querySelector: () => searchOnly ? {} : null,
        querySelectorAll() { return [{textContent: this.text}]; },
        getAttribute() { return this.version; },
        classList: {add: c => classes.add(c), remove: c => classes.delete(c), contains: c => classes.has(c), toggle(c, force) { force ? classes.add(c) : classes.delete(c); }},
    };
}
let cards = [card("alpha cat", "sd"), card("beta cat", "krea"), card("hidden alpha", "Unknown", true)];
const handlers = {};
const search = {value: "", dataset: {}, isConnected: true, addEventListener(name, fn) { assert(!handlers[name]); handlers[name] = fn; }};
const preset = {value: "sd"};
const controlsDiv = {insertBefore() {}};
const page = {id: "txt2img_lora", style: {}, isConnected: true, querySelectorAll(selector) { if (selector === "div.card") { scans++; return cards; } return []; }};
const tabs = {querySelectorAll: () => [page]};
const parent = {appendChild(fragment) { cards = fragment.children; }};
const selectors = new Map([
    ["#txt2img_extra_tabs", tabs], ["#txt2img_lora_extra_search", search],
    ["#txt2img_lora_extra_sort_dir", {dataset: {sortdir: "Ascending"}}],
    ["#txt2img_lora_extra_refresh", {}], ["#txt2img_lora_controls", {}],
    ["#txt2img_lora_cards", parent], ["#forge_ui_preset input", preset],
]);
const app = {querySelector: selector => selectors.get(selector), querySelectorAll: selector => selector === "#txt2img_lora div.card" ? cards : []};
const context = vm.createContext({document, window: {addEventListener() {}}, onUiLoaded() {}, onUiUpdate() {},
    gradioApp: () => app, get_uiTabList: () => ({querySelector: () => controlsDiv}), opts: {lora_preset_filter: true},
    setTimeout(fn) { const id = ++timerId; timers.set(id, fn); return id; }, clearTimeout(id) { timers.delete(id); },
});
vm.runInContext(SOURCE, context);
context.toggleCss("test", "a{}", true); context.toggleCss("test", "b{}", true);
assert.equal(styles.get("test").textContent, "b{}");
context.toggleCss("test", "", false); assert.equal(styles.size, 0);
assert(context.setupExtraNetworksForTab("txt2img"));
const initialScans = scans;
for (const value of ["a", "al", "alp", "alpha"]) { search.value = value; handlers.input({isTrusted: true}); }
assert.equal(scans, initialScans); assert.equal(timers.size, 1);
function flush() { const work = [...timers.values()]; timers.clear(); work.forEach(fn => fn()); }
flush(); assert.equal(scans, initialScans + 1);
assert.equal(cards.filter(c => !c.classList.contains("hidden")).length, 2);
search.value = "beta"; handlers.input({isTrusted: true}); preset.value = "krea";
vm.runInContext('extraNetworksApplyFilter.txt2img_lora(true)', context);
assert.equal(timers.size, 0);
assert.equal(cards.find(c => c.text === "beta cat").classList.contains("hidden"), false);
// Read the current DOM text instead of returning stale cached metadata.
cards.find(c => c.text === "beta cat").text = "renamed";
search.value = "renamed"; handlers.input({isTrusted: true}); flush();
assert.equal(cards.find(c => c.text === "renamed").classList.contains("hidden"), false);
const beforeDetach = scans; handlers.input({isTrusted: true}); page.isConnected = false; flush();
assert.equal(scans, beforeDetach);
let descendantScans = 0;
const directory = {hidden: true, querySelectorAll() { descendantScans++; return []; }};
context.extraNetworksApplyLoraTreePresetFilter({querySelectorAll: selector => selector.includes("'dir'") ? [directory] : []}, "sd", false);
assert.equal(directory.hidden, false); assert.equal(descendantScans, 0);
'''
        harness = "const SOURCE = " + json.dumps(source) + ";\n" + harness
        result = subprocess.run([node], input=harness, capture_output=True, text=True, check=False, timeout=20)  # noqa: S603
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
