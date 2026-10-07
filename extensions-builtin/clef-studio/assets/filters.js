// Gradio HTML component scope: each row keeps its own choices and values.
let model = null;
let pending = null;
const node = (tag, className, text) => {
    const result = document.createElement(tag);
    if (className) result.className = className;
    if (text !== undefined) result.textContent = text;
    return result;
};
const emit = () => {
    model.revision += 1;
    clearTimeout(pending);
    pending = setTimeout(() => {
        pending = null;
        trigger('input', {
            run_id: model.run_id, panel_id: model.panel_id, revision: model.revision,
            rows: structuredClone(model.rows), mode: model.mode,
        });
    }, 100);
};
const receive = () => {
    clearTimeout(pending);
    pending = null;
    model = props.value ? structuredClone(props.value) : null;
    const root = node('div', 'clef-filters');
    if (!model || !model.questions?.length) {
        root.append(node('p', 'clef-filter-empty', '判定結果を開くと、判断項目ごとに条件を設定できます。'));
        element.replaceChildren(root);
        return;
    }
    const modes = node('fieldset', 'clef-filter-modes');
    modes.append(node('legend', '', '条件の組み合わせ'));
    for (const mode of ['すべて', 'いずれか']) {
        const label = node('label');
        const radio = node('input');
        radio.type = 'radio';
        radio.name = `clef-filter-mode-${model.panel_id}`;
        radio.value = mode;
        radio.checked = model.mode === mode;
        radio.addEventListener('change', () => { model.mode = mode; emit(); });
        label.append(radio, document.createTextNode(mode));
        modes.append(label);
    }
    const clear = node('button', 'clef-filter-clear', '条件をすべて解除');
    clear.type = 'button';
    clear.addEventListener('click', () => {
        for (const row of model.rows) row.enabled = false;
        root.querySelectorAll('.clef-filter-enabled').forEach(input => { input.checked = false; });
        emit();
    });
    modes.append(clear);
    root.append(modes);
    for (const question of model.questions) {
        const row = model.rows.find(row => row.qid === question.qid);
        const section = node('div', 'clef-filter-row');
        section.dataset.qid = question.qid;
        const label = node('label', 'clef-filter-question');
        const enabled = node('input', 'clef-filter-enabled');
        enabled.type = 'checkbox';
        enabled.checked = row.enabled;
        enabled.setAttribute('aria-label', `条件 · ${question.qid}`);
        enabled.addEventListener('change', () => { row.enabled = enabled.checked; emit(); });
        label.append(enabled, node('span', '', question.label));
        section.append(label);
        const controls = node('div', 'clef-filter-controls');
        const target = node('select');
        target.setAttribute('aria-label', `対象 · ${question.qid}`);
        for (const [label, value] of question.targets) {
            const option = node('option', '', label);
            option.value = value;
            target.append(option);
        }
        target.value = row.target;
        const activate = () => { row.enabled = true; enabled.checked = true; };
        target.addEventListener('change', () => { row.target = target.value; activate(); emit(); });
        const slider = node('input');
        slider.type = 'range';
        slider.min = '0'; slider.max = '1'; slider.step = '0.01'; slider.value = row.minimum;
        slider.setAttribute('aria-label', `最低確率スライダー · ${question.qid}`);
        const number = node('input');
        number.type = 'number';
        number.min = '0'; number.max = '1'; number.step = '0.01'; number.value = row.minimum;
        number.setAttribute('aria-label', `最低確率 · ${question.qid}`);
        const setMinimum = (value, syncNumber = true) => {
            if (value === '') return false;
            const parsed = Number(value);
            if (!Number.isFinite(parsed)) return false;
            row.minimum = Math.min(1, Math.max(0, parsed));
            slider.value = row.minimum;
            if (syncNumber) number.value = row.minimum;
            activate();
            return true;
        };
        slider.addEventListener('input', () => { setMinimum(slider.value); emit(); });
        slider.addEventListener('change', emit);
        number.addEventListener('input', () => { if (setMinimum(number.value, false)) emit(); });
        number.addEventListener('change', () => { setMinimum(number.value); number.value = row.minimum; emit(); });
        controls.append(target, slider, number);
        section.append(controls);
        root.append(section);
    }
    root.append(node('p', 'clef-filter-note', '最低確率は0〜1で指定します。正答率ではありません。'));
    element.replaceChildren(root);
};
watch('value', receive);
receive();
