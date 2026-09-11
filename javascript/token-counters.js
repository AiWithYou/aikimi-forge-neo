const promptTokenCountUpdateFunctions = {};
const tokenCounterTextareas = new WeakMap();
let tokenCounterSetupTimer = null;
let tokenCounterSetupAttempts = 0;
const TOKEN_COUNTER_SETUP_MAX_ATTEMPTS = 40;
const TOKEN_COUNTER_CONFIGS = Object.freeze([
    Object.freeze(["txt2img_prompt", "txt2img_token_counter", "txt2img_token_button"]),
    Object.freeze(["txt2img_neg_prompt", "txt2img_negative_token_counter", "txt2img_negative_token_button"]),
    Object.freeze(["img2img_prompt", "img2img_token_counter", "img2img_token_button"]),
    Object.freeze(["img2img_neg_prompt", "img2img_negative_token_counter", "img2img_negative_token_button"]),
]);

function update_txt2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("txt2img_token_button");
    update_token_counter("txt2img_negative_token_button");
    return (args.length === 2) ? args[0] : args;
}

function update_img2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("img2img_token_button");
    update_token_counter("img2img_negative_token_button");
    return (args.length === 2) ? args[0] : args;
}

function update_token_counter(button_id) {
    promptTokenCountUpdateFunctions[button_id]?.();
}

function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

function recalculate_prompts_txt2img() {
    // Called from Gradio
    recalculatePromptTokens("txt2img_prompt");
    recalculatePromptTokens("txt2img_neg_prompt");
    return Array.from(arguments);
}

function recalculate_prompts_img2img() {
    // Called from Gradio
    recalculatePromptTokens("img2img_prompt");
    recalculatePromptTokens("img2img_neg_prompt");
    return Array.from(arguments);
}

function setupTokenCounting(id, id_counter, id_button) {
    let prompt = gradioApp().getElementById(id);
    let counter = gradioApp().getElementById(id_counter);
    let textarea = gradioApp().querySelector(`#${id} > label > textarea`);
    if (!prompt || !counter || !textarea || !prompt.parentElement) return false;

    if (counter.parentElement !== prompt.parentElement) {
        prompt.parentElement.insertBefore(counter, prompt);
        prompt.parentElement.style.position = "relative";
    }

    let func = tokenCounterTextareas.get(textarea);
    if (!func) {
        func = onEdit(id, textarea, 1000, function () {
            if (counter.classList.contains("token-counter-visible")) {
                gradioApp().getElementById(id_button)?.click();
            }
        });
        tokenCounterTextareas.set(textarea, func);
        textarea.dataset.forgeTokenCounterBound = "true";
    }
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
    toggleTokenCountingVisibility(id, id_counter, id_button);
    return true;
}

function toggleTokenCountingVisibility(id, id_counter, id_button) {
    let counter = gradioApp().getElementById(id_counter);
    if (!counter) return false;

    counter.style.display = opts.disable_token_counters ? "none" : "block";
    counter.classList.toggle("token-counter-visible", !opts.disable_token_counters);
    return true;
}

function runCodeForTokenCounters(fun) {
    return TOKEN_COUNTER_CONFIGS.map((config) => fun(...config)).every(Boolean);
}

function tokenCounterTargetsAreReady() {
    return TOKEN_COUNTER_CONFIGS.every(function ([id]) {
        const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
        return textarea?.dataset.forgeTokenCounterBound === "true";
    });
}

function tokenCounterMountIsPending() {
    return TOKEN_COUNTER_CONFIGS.some(function ([id, idCounter]) {
        const prompt = gradioApp().getElementById(id);
        const counter = gradioApp().getElementById(idCounter);
        const textarea = gradioApp().querySelector(`#${id} > label > textarea`);
        return Boolean(
            (prompt || counter || textarea) &&
            textarea?.dataset.forgeTokenCounterBound !== "true"
        );
    });
}

function requestTokenCounterSetup() {
    if (tokenCounterTargetsAreReady()) return;
    tokenCounterSetupAttempts = TOKEN_COUNTER_SETUP_MAX_ATTEMPTS;
    if (tokenCounterSetupTimer !== null) return;

    function retry() {
        tokenCounterSetupTimer = null;
        if (tokenCounterTargetsAreReady()) return;
        runCodeForTokenCounters(setupTokenCounting);
        if (tokenCounterTargetsAreReady()) return;
        tokenCounterSetupAttempts -= 1;
        if (tokenCounterSetupAttempts <= 0 || !tokenCounterMountIsPending()) return;
        tokenCounterSetupTimer = window.setTimeout(retry, 50);
    }
    tokenCounterSetupTimer = window.setTimeout(retry, 0);
}

function tokenCounterMutationMayMountTarget(mutationRecords) {
    if (tokenCounterTargetsAreReady()) return false;
    return Array.from(mutationRecords || []).some(function (record) {
        return Array.from(record.addedNodes || []).some(function (node) {
            if (node.nodeType !== Node.ELEMENT_NODE) return false;
            return TOKEN_COUNTER_CONFIGS.some(function ([id, idCounter, idButton]) {
                return [id, idCounter, idButton].includes(node.id) ||
                    Boolean(node.querySelector?.(`#${id}, #${idCounter}, #${idButton}`));
            });
        });
    });
}

onUiLoaded(requestTokenCounterSetup);
onUiTabChange(requestTokenCounterSetup);
onUiUpdate(function (mutationRecords) {
    if (tokenCounterMutationMayMountTarget(mutationRecords)) requestTokenCounterSetup();
});

onOptionsChanged(function () {
    runCodeForTokenCounters(toggleTokenCountingVisibility);
    requestTokenCounterSetup();
});
