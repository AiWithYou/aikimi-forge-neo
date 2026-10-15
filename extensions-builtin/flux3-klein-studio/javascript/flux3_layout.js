(function () {
    "use strict";
    const states = new WeakMap();
    const NS = "http://www.w3.org/2000/svg";
    let scheduled = false;

    function parseRows(text) {
        try {
            const rows = JSON.parse(text || "[]");
            return Array.isArray(rows) && rows.every(row => row && typeof row === "object" && !Array.isArray(row)) ? rows : null;
        } catch (_) { return null; }
    }
    function rectangle(start, end) {
        const [x1, x2] = [start[0], end[0]].sort((a, b) => a - b);
        const [y1, y2] = [start[1], end[1]].sort((a, b) => a - b);
        const box = [y1, x1, y2, x2].map(value => Math.round(Math.max(0, Math.min(1, value)) * 10000) / 10000);
        return box[2] - box[0] >= 0.01 && box[3] - box[1] >= 0.01 ? box : null;
    }
    function appendRectangle(rows, bbox) {
        let number = 1;
        while (rows.some(row => row.id === `obj_${number}`)) number++;
        return rows.concat({id: `obj_${number}`, desc: `要素 ${number}`, bbox});
    }
    function clearRectangles(rows) { return rows.filter(row => !Array.isArray(row.bbox)); }
    function svgNode(kind, attributes = {}) {
        const node = document.createElementNS(NS, kind);
        Object.entries(attributes).forEach(([key, value]) => node.setAttribute(key, String(value)));
        return node;
    }
    function drawBox(svg, box, number) {
        const [top, left, bottom, right] = box.map(value => value * 1000);
        const group = svgNode("g", {"data-row": number});
        group.append(svgNode("rect", {x: left, y: top, width: right - left, height: bottom - top, "vector-effect": "non-scaling-stroke"}));
        const label = svgNode("text", {x: left + 12, y: top + 34});
        label.textContent = String(number + 1);
        group.append(label);
        svg.append(group);
    }
    function commit(state, rows, redraw = true) {
        state.rows = rows;
        state.lastValue = JSON.stringify(rows, null, 2);
        state.input.value = state.lastValue;
        updateInput(state.input);
        if (redraw) render(state);
    }
    function render(state) {
        state.svg.replaceChildren();
        state.list.replaceChildren();
        state.note.textContent = state.rows ? "" : "JSONを配列として正しく入力すると、描画を再開できます。";
        state.clear.disabled = !state.rows?.some(row => Array.isArray(row.bbox));
        if (!state.rows) return;
        state.rows.forEach((row, index) => {
            if (Array.isArray(row.bbox) && row.bbox.length === 4 && row.bbox.every(Number.isFinite)) drawBox(state.svg, row.bbox, index);
            const line = document.createElement("div");
            line.className = "flux3-layout-row";
            const label = document.createElement("label");
            label.textContent = `${index + 1} · ${row.id || "編集配置"}`;
            const input = document.createElement("input");
            input.type = "text";
            input.value = typeof row.desc === "string" ? row.desc : "";
            input.placeholder = "この要素の説明";
            input.setAttribute("aria-label", `説明 ${row.id || index + 1}`);
            input.dataset.index = String(index);
            input.addEventListener("input", () => {
                if (state.rows) commit(state, state.rows.map((item, i) => i === index ? {...item, desc: input.value} : item), false);
            });
            label.append(input);
            const remove = document.createElement("button");
            remove.type = "button";
            remove.textContent = "削除";
            remove.setAttribute("aria-label", `${row.id || index + 1} を削除`);
            remove.addEventListener("click", () => {
                if (state.rows) commit(state, state.rows.filter((_, i) => i !== index));
            });
            line.append(label, remove);
            state.list.append(line);
        });
        if (!state.rows.length) {
            const empty = document.createElement("span");
            empty.className = "flux3-layout-empty";
            empty.textContent = "枠を描くと、ここに要素の説明を入力できます。";
            state.list.append(empty);
        }
    }
    function pointer(state, event) {
        const rect = state.svg.getBoundingClientRect();
        return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
    }
    function connect(root) {
        const state = {root, svg: root.querySelector("svg"), list: root.querySelector(".flux3-layout-rows"),
            clear: root.querySelector(".flux3-layout-clear"), note: root.querySelector(".flux3-layout-note"), lastValue: null};
        state.clear.addEventListener("click", () => { if (state.rows) commit(state, clearRectangles(state.rows)); });
        state.svg.addEventListener("pointerdown", event => {
            if (event.button !== 0 || !state.rows) return;
            event.preventDefault();
            state.drag = {start: pointer(state, event), row: event.target.closest("[data-row]")?.dataset.row};
            state.svg.setPointerCapture(event.pointerId);
        });
        state.svg.addEventListener("pointermove", event => {
            if (!state.drag) return;
            state.preview?.remove();
            const box = rectangle(state.drag.start, pointer(state, event));
            if (box) {
                state.preview = svgNode("rect", {class: "flux3-layout-draft", x: box[1] * 1000, y: box[0] * 1000,
                    width: (box[3] - box[1]) * 1000, height: (box[2] - box[0]) * 1000, "vector-effect": "non-scaling-stroke"});
                state.svg.append(state.preview);
            }
        });
        state.svg.addEventListener("pointerup", event => {
            if (!state.drag) return;
            const box = rectangle(state.drag.start, pointer(state, event));
            const selected = state.drag.row;
            state.drag = null;
            state.preview?.remove();
            state.svg.releasePointerCapture(event.pointerId);
            if (box) {
                commit(state, appendRectangle(state.rows, box));
                const input = state.list.querySelector(`input[data-index="${state.rows.length - 1}"]`);
                input?.focus();
                input?.select();
            } else if (selected !== undefined) state.list.querySelector(`input[data-index="${selected}"]`)?.focus();
        });
        state.svg.addEventListener("pointercancel", () => { state.drag = null; state.preview?.remove(); });
        states.set(root, state);
        return state;
    }
    function refresh() {
        scheduled = false;
        const app = gradioApp();
        app.querySelectorAll("#flux3-layout-editor").forEach(root => {
            const state = states.get(root) || connect(root);
            const input = app.querySelector("#flux3-layout-json textarea");
            if (!input) return;
            if (state.input !== input) {
                state.input = input;
                state.lastValue = null;
                input.addEventListener("input", schedule);
            }
            if (state.lastValue !== input.value) {
                state.lastValue = input.value;
                state.rows = parseRows(input.value);
                render(state);
            }
            const size = app.querySelector('#flux3-settings input[role="listbox"], #flux3-settings input[role="combobox"]');
            if (size && state.size !== size) { state.size = size; size.addEventListener("input", schedule); }
            const dimensions = size?.value.match(/(\d+)\s*[×x]\s*(\d+)/);
            const reference = app.querySelector("#flux3-reference-images img");
            if (reference && state.reference !== reference) {
                state.reference = reference;
                reference.addEventListener("load", schedule, {once: true});
            }
            const ratio = dimensions ? Number(dimensions[1]) / Number(dimensions[2]) : (reference?.naturalWidth / reference?.naturalHeight || 1);
            if (state.ratio !== ratio) {
                state.ratio = ratio;
                const stage = root.querySelector(".flux3-layout-stage");
                stage.style.aspectRatio = String(ratio);
                stage.style.width = `${Math.min(560, 380 * ratio)}px`;
            }
        });
    }
    function schedule() { if (!scheduled) { scheduled = true; requestAnimationFrame(refresh); } }
    onUiLoaded(schedule);
    onUiUpdate(schedule);
    if (typeof module !== "undefined" && module.exports) module.exports = {parseRows, rectangle, appendRectangle, clearRectangles};
})();
