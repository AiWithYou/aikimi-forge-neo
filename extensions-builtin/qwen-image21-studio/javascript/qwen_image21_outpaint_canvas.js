(function () {
    "use strict";
    const states = new WeakMap();
    let drag = null;
    let pendingFocus = null;
    let observedView = null;
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const root = entry.target.closest('.qoc');
            if (root && drag?.root !== root) render(root);
        }
    });
    const clamp = (n, low, high) => Math.max(low, Math.min(high, n));
    const aligned = (size, before, after) => {
        const extra = ((-size - before - after) % 32 + 32) % 32;
        if (before && after) return [before + Math.floor(extra / 2), after + Math.ceil(extra / 2)];
        if (before) return [before + extra, after];
        if (after) return [before, after + extra];
        return [before, after];
    };

    function geometry(st) {
        const [l, r] = aligned(st.w, st.pads[0], st.pads[2]);
        const [t, b] = aligned(st.h, st.pads[1], st.pads[3]);
        return {l, t, r, b, w: st.w + l + r, h: st.h + t + b};
    }

    function render(root, fit = true) {
        const st = states.get(root);
        if (!st) return;
        const view = root.querySelector('.qoc-viewport');
        const frame = root.querySelector('.qoc-frame');
        const picture = root.querySelector('.qoc-picture');
        const g = geometry(st);
        if (fit || !st.view) {
            const s = Math.min((view.clientWidth - 56) / g.w, (view.clientHeight - 56) / g.h);
            st.view = {s: Math.max(0.001, s), ix: (view.clientWidth - st.w * s) / 2 + (g.l - g.r) * s / 2,
                iy: (view.clientHeight - st.h * s) / 2 + (g.t - g.b) * s / 2};
        }
        const {s, ix, iy} = st.view;
        Object.assign(frame.style, {left: `${ix - g.l * s}px`, top: `${iy - g.t * s}px`,
            width: `${g.w * s}px`, height: `${g.h * s}px`});
        Object.assign(picture.style, {left: `${g.l * s}px`, top: `${g.t * s}px`,
            width: `${st.w * s}px`, height: `${st.h * s}px`});
        const invalid = g.w > 4096 || g.h > 4096 || g.w * g.h > 2097152;
        root.classList.toggle('qoc-invalid', invalid);
        root.querySelector('.qoc-size').textContent = `${g.w} × ${g.h} px${invalid ? ' · 約2 MP以内に調整' : ''}`;
        const ratios = [...root.querySelectorAll('[data-ratio]')];
        const distance = b => Math.abs(g.w / g.h / Number(b.dataset.ratio) - 1);
        const nearest = ratios.reduce((a, b) => distance(a) < distance(b) ? a : b);
        ratios.forEach(b => {
            b.setAttribute('aria-pressed', String(b === nearest && distance(b) < 0.06));
        });
        root.querySelectorAll('.qoc-h').forEach(b => {
            const key = b.dataset.h;
            b.setAttribute('aria-valuetext', `${key.includes('w') ? g.l : key.includes('e') ? g.r : key === 'n' ? g.t : g.b} px`);
        });
    }

    function commit(root) {
        const st = states.get(root);
        if (!st) return;
        const focus = document.activeElement;
        pendingFocus = root.contains(focus) && focus.matches('.qoc-h, .qoc-picture') ?
            (focus.dataset.h ? `[data-h="${focus.dataset.h}"]` : '.qoc-picture') : null;
        st.pads = st.pads.map(v => clamp(Math.round(v), 0, 4096));
        render(root);
        const input = gradioApp().querySelector('#qwen21-outpaint-canvas-commit textarea, #qwen21-outpaint-canvas-commit input');
        if (!input) return;
        input.value = JSON.stringify({w: st.w, h: st.h, pads: st.pads, nonce: Date.now()});
        updateInput(input);
    }

    function setup() {
        gradioApp().querySelectorAll('#qwen21-outpaint-canvas .qoc').forEach(root => {
            if (drag?.root === root) return;
            const previous = states.get(root);
            if (previous?.serial === root.dataset.layout && root.querySelector('.qoc-frame')?.style.width &&
                root.querySelector('.qoc-size')?.textContent) return;
            try {
                const st = JSON.parse(root.dataset.layout);
                st.serial = root.dataset.layout;
                states.set(root, st);
                const view = root.querySelector('.qoc-viewport');
                if (observedView !== view) {
                    resizeObserver.disconnect();
                    observedView = view;
                    resizeObserver.observe(view);
                }
                render(root);
                if (pendingFocus) root.querySelector(pendingFocus)?.focus({preventScroll: true});
            } catch (e) { console.error('Outpaint canvas initialization failed', e); }
        });
    }

    document.addEventListener('pointerdown', e => {
        const root = e.target.closest?.('#qwen21-outpaint-canvas .qoc');
        const control = e.target.closest?.('.qoc-h, .qoc-picture');
        if (!root || !control || e.button !== 0 || !states.has(root)) return;
        const st = states.get(root);
        const mode = control.dataset.h || 'move', g = geometry(st);
        drag = {root, control, pointer: e.pointerId, mode, x: e.clientX, y: e.clientY,
            pads: mode === 'move' ? [g.l, g.t, g.r, g.b] : [...st.pads],
            originalPads: [...st.pads], view: {...st.view}};
        control.setPointerCapture(e.pointerId);
        root.classList.add('qoc-dragging');
        e.preventDefault();
    });
    document.addEventListener('pointermove', e => {
        if (!drag || e.pointerId !== drag.pointer) return;
        if (!drag.root.isConnected) { drag = null; return; }
        const st = states.get(drag.root);
        const dx = (e.clientX - drag.x) / drag.view.s, dy = (e.clientY - drag.y) / drag.view.s;
        const [l, t, r, b] = drag.pads;
        if (drag.mode === 'move') {
            const shiftX = clamp(Math.round(dx), -l, r), shiftY = clamp(Math.round(dy), -t, b);
            st.pads = [l + shiftX, t + shiftY, r - shiftX, b - shiftY];
            st.view.ix = drag.view.ix + shiftX * drag.view.s;
            st.view.iy = drag.view.iy + shiftY * drag.view.s;
        } else {
            st.pads = [l, t, r, b];
            if (drag.mode.includes('w')) st.pads[0] = clamp(Math.round(l - dx), 0, 4096);
            if (drag.mode.includes('e')) st.pads[2] = clamp(Math.round(r + dx), 0, 4096);
            if (drag.mode.includes('n')) st.pads[1] = clamp(Math.round(t - dy), 0, 4096);
            if (drag.mode.includes('s')) st.pads[3] = clamp(Math.round(b + dy), 0, 4096);
        }
        render(drag.root, false);
        e.preventDefault();
    });
    function end(e, cancel = false) {
        if (!drag || e.pointerId !== drag.pointer) return;
        const current = drag;
        drag = null;
        current.root.classList.remove('qoc-dragging');
        if (current.control.hasPointerCapture(e.pointerId)) current.control.releasePointerCapture(e.pointerId);
        if (!current.root.isConnected) return;
        if (cancel) states.get(current.root).pads = current.originalPads;
        commit(current.root);
    }
    document.addEventListener('pointerup', e => end(e));
    document.addEventListener('pointercancel', e => end(e, true));
    window.addEventListener('blur', () => { if (drag) end({pointerId: drag.pointer}, true); });
    document.addEventListener('focusin', e => {
        if (!e.target.closest?.('#qwen21-outpaint-canvas .qoc')) pendingFocus = null;
    });
    document.addEventListener('click', e => {
        const button = e.target.closest?.('#qwen21-outpaint-canvas .qoc button');
        if (!button || button.classList.contains('qoc-h') || button.classList.contains('qoc-picture')) return;
        const root = button.closest('.qoc'), st = states.get(root);
        if (!st) return;
        const g = geometry(st);
        let w = g.w, h = g.h;
        if (button.dataset.ratio) {
            const ratio = Number(button.dataset.ratio);
            w = Math.ceil(Math.max(st.w, st.h * ratio));
            h = Math.ceil(Math.max(st.h, st.w / ratio));
        } else if (button.dataset.action === 'grow') {
            w = Math.ceil(w * 1.15); h = Math.ceil(h * 1.15);
        } else if (button.dataset.action === 'reset') {
            st.pads = [0, 0, 0, 0]; commit(root); return;
        }
        const x = Math.max(0, w - st.w), y = Math.max(0, h - st.h);
        st.pads = [Math.floor(x / 2), Math.floor(y / 2), Math.ceil(x / 2), Math.ceil(y / 2)];
        commit(root);
    });
    document.addEventListener('keydown', e => {
        const button = e.target.closest?.('#qwen21-outpaint-canvas .qoc-h, #qwen21-outpaint-canvas .qoc-picture');
        if (!button || !['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Escape'].includes(e.key)) return;
        if (e.key === 'Escape') { if (drag) end({pointerId: drag.pointer}, true); return; }
        const root = button.closest('.qoc'), st = states.get(root), mode = button.dataset.h || 'move';
        const step = e.shiftKey ? 128 : 32;
        const dx = e.key === 'ArrowLeft' ? -step : e.key === 'ArrowRight' ? step : 0;
        const dy = e.key === 'ArrowUp' ? -step : e.key === 'ArrowDown' ? step : 0;
        const [l, t, r, b] = st.pads;
        if (mode === 'move') {
            const g = geometry(st);
            const x = clamp(dx, -g.l, g.r), y = clamp(dy, -g.t, g.b);
            st.pads = [g.l + x, g.t + y, g.r - x, g.b - y];
        } else {
            if (mode.includes('w')) st.pads[0] = clamp(l - dx, 0, 4096);
            if (mode.includes('e')) st.pads[2] = clamp(r + dx, 0, 4096);
            if (mode.includes('n')) st.pads[1] = clamp(t - dy, 0, 4096);
            if (mode.includes('s')) st.pads[3] = clamp(b + dy, 0, 4096);
        }
        e.preventDefault(); commit(root);
    });
    onUiLoaded(() => {
        setup();
        new MutationObserver(setup).observe(gradioApp(), {subtree: true, childList: true});
    });
    onAfterUiUpdate(setup);
})();
