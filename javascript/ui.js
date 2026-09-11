// various functions for interaction with ui.py not large enough to warrant putting them in separate files

function set_theme(theme) {
    const gradioURL = window.location.href;
    if (!gradioURL.includes("?__theme=")) {
        window.location.replace(gradioURL + "?__theme=" + theme);
    }
}

function all_gallery_buttons() {
    const allGalleryButtons = gradioApp().querySelectorAll(
        '[style="display: block;"].tabitem div[id$=_gallery].gradio-gallery .thumbnails > .thumbnail-item.thumbnail-small',
    );
    const visibleGalleryButtons = [];
    allGalleryButtons.forEach(function (elem) {
        if (elem.parentElement.offsetParent) {
            visibleGalleryButtons.push(elem);
        }
    });
    return visibleGalleryButtons;
}

function selected_gallery_button() {
    return all_gallery_buttons().find((elem) => elem.classList.contains("selected")) ?? null;
}

function selected_gallery_index() {
    return all_gallery_buttons().findIndex((elem) => elem.classList.contains("selected"));
}

function gallery_container_buttons(gallery_container) {
    return gradioApp().querySelectorAll(`#${gallery_container} .thumbnail-item.thumbnail-small`);
}

function selected_gallery_index_id(gallery_container) {
    return Array.from(gallery_container_buttons(gallery_container)).findIndex((elem) =>
        elem.classList.contains("selected"),
    );
}

function extract_image_from_gallery(gallery) {
    if (gallery.length === 0) {
        return [null];
    }

    let index = selected_gallery_index();

    if (index < 0 || index >= gallery.length) {
        // Use the first image in the gallery as the default
        index = 0;
    }

    return [[gallery[index]]];
}

window.args_to_array = Array.from; // Compatibility with e.g. extensions that may expect this to be around

function switch_to_txt2img() {
    get_uiTopTabButton("tab_txt2img")?.click();

    return Array.from(arguments);
}

function switch_to_img2img_tab(no) {
    get_uiTopTabButton("tab_img2img")?.click();
    get_uiTabButtons(gradioApp().getElementById("mode_img2img"))[no]?.click();
}

function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    get_uiTopTabButton("tab_extras")?.click();

    return Array.from(arguments);
}

function get_tab_index(tabId) {
    const buttons = get_uiTabButtons(gradioApp().getElementById(tabId));
    for (let i = 0; i < buttons.length; i++) {
        if (
            buttons[i].classList.contains("selected") ||
            buttons[i].getAttribute("aria-selected") === "true"
        ) {
            return i;
        }
    }
    return 0;
}

function create_tab_index_args(tabId, args) {
    const res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    const res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index("mode_img2img");
    return res;
}

function create_submit_args(args) {
    // Currently, txt2img and img2img also send the output args (gallery / player / generation_info / infotext / html_log) whenever you generate a new image.
    const res = Array.from(args);

    if (Array.isArray(res[res.length - 5])) return res.slice(0, res.length - 5);
    else if (Array.isArray(res[res.length - 4])) return res.slice(0, res.length - 4);
    else if (Array.isArray(res[res.length - 3])) return res.slice(0, res.length - 3);

    // NOTE: If gradio at some point stops sending outputs, this may break something
    return res;
}

function setSubmitButtonsVisibility(tabname, showInterrupt, showSkip, showInterrupting) {
    gradioApp().getElementById(tabname + "_interrupt").style.display = showInterrupt ? "block" : "none";
    gradioApp().getElementById(tabname + "_skip").style.display = showSkip ? "block" : "none";
    gradioApp().getElementById(tabname + "_interrupting").style.display = showInterrupting ? "block" : "none";
}

function showSubmitButtons(tabname, show) {
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

function showSubmitInterruptingPlaceholder(tabname) {
    setSubmitButtonsVisibility(tabname, false, true, true);
}

function showRestoreProgressButton(tabname, show) {
    const button = gradioApp().getElementById(tabname + "_restore_progress");
    if (!button) return;
    button.style.setProperty("display", show ? "flex" : "none", "important");
}

const submitTasksByTab = new Map();

function startSubmitTask(tabname, id) {
    let tasks = submitTasksByTab.get(tabname);
    if (!tasks) {
        tasks = new Set();
        submitTasksByTab.set(tabname, tasks);
    }
    tasks.add(id);
    showSubmitButtons(tabname, false);
}

function finishSubmitTask(tabname, id) {
    const tasks = submitTasksByTab.get(tabname);
    if (!tasks) return;
    tasks.delete(id);
    if (tasks.size > 0) return;
    submitTasksByTab.delete(tabname);
    showSubmitButtons(tabname, true);
}

function submit() {
    const id = randomId();
    startSubmitTask("txt2img", id);
    localSet("txt2img_task_id", id);

    requestProgress(
        id,
        gradioApp().getElementById("txt2img_gallery_container"),
        gradioApp().getElementById("txt2img_gallery"),
        function () {
            finishSubmitTask("txt2img", id);
            localRemove("txt2img_task_id");
            showRestoreProgressButton("txt2img", false);
        },
    );

    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function submit_txt2img_upscale() {
    const res = submit(...arguments);
    res[2] = selected_gallery_index();
    return res;
}

function submit_img2img() {
    const id = randomId();
    startSubmitTask("img2img", id);
    localSet("img2img_task_id", id);

    requestProgress(
        id,
        gradioApp().getElementById("img2img_gallery_container"),
        gradioApp().getElementById("img2img_gallery"),
        function () {
            finishSubmitTask("img2img", id);
            localRemove("img2img_task_id");
            showRestoreProgressButton("img2img", false);
        },
    );

    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function submitQueued(tabname, args) {
    const id = randomId();
    startSubmitTask(tabname, id);

    requestProgress(
        id,
        gradioApp().getElementById(tabname + "_gallery_container"),
        gradioApp().getElementById(tabname + "_gallery"),
        function () {
            finishSubmitTask(tabname, id);
        },
    );

    const res = create_submit_args(args);
    res[0] = id;
    return res;
}

function submit_txt2img_queue() {
    return submitQueued("txt2img", arguments);
}

function submit_img2img_queue() {
    return submitQueued("img2img", arguments);
}

function submit_extras() {
    showSubmitButtons("extras", false);

    const id = randomId();

    requestProgress(
        id,
        gradioApp().getElementById("extras_gallery_container"),
        gradioApp().getElementById("extras_gallery"),
        function () {
            showSubmitButtons("extras", true);
        },
    );

    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function restoreProgressTxt2img() {
    showRestoreProgressButton("txt2img", false);
    const id = localGet("txt2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder("txt2img");
        requestProgress(
            id,
            gradioApp().getElementById("txt2img_gallery_container"),
            gradioApp().getElementById("txt2img_gallery"),
            function () {
                showSubmitButtons("txt2img", true);
            },
            null,
            0,
        );
    }

    return id;
}

function restoreProgressImg2img() {
    showRestoreProgressButton("img2img", false);
    const id = localGet("img2img_task_id");

    if (id) {
        showSubmitInterruptingPlaceholder("img2img");
        requestProgress(
            id,
            gradioApp().getElementById("img2img_gallery_container"),
            gradioApp().getElementById("img2img_gallery"),
            function () {
                showSubmitButtons("img2img", true);
            },
            null,
            0,
        );
    }

    return id;
}

/**
 * Configure the width and height elements on `tabname` to accept
 * pasting of resolutions in the form of "width x height".
 */
const resolutionPasteBindings = new Map();

function setupResolutionPasting(tabname) {
    const width = gradioApp().querySelector(`#${tabname}_width input[type=number]`);
    const height = gradioApp().querySelector(`#${tabname}_height input[type=number]`);
    if (!width || !height) return false;

    const existing = resolutionPasteBindings.get(tabname);
    if (existing?.width === width && existing.height === height) return true;
    if (existing) {
        existing.width.removeEventListener("paste", existing.handler);
        existing.height.removeEventListener("paste", existing.handler);
    }

    const handler = function (event) {
        const pasteData = event.clipboardData?.getData("text/plain") || "";
        const parsed = pasteData.match(/^\s*(\d+)\D+(\d+)\s*$/);
        if (parsed) {
            width.value = parsed[1];
            height.value = parsed[2];
            updateInput(width);
            updateInput(height);
            event.preventDefault();
        }
    };
    width.addEventListener("paste", handler);
    height.addEventListener("paste", handler);
    width.dataset.forgeResolutionPasteBound = "true";
    height.dataset.forgeResolutionPasteBound = "true";
    resolutionPasteBindings.set(tabname, { width, height, handler });
    return true;
}

/**
 * Allow the user to click on the Style name in order to deselect it just like Gradio 3
 */
const styleDeselectionTargets = new WeakSet();

function restoreStyleDeselection(tabname) {
    const dropdown = document.getElementById(`${tabname}_styles`);
    if (!dropdown) return false;
    if (styleDeselectionTargets.has(dropdown)) return true;
    dropdown.addEventListener("click", (e) => {
        const remove = e.target.closest("div.token-remove");
        if (remove) return;
        const style = e.target.closest("div.token");
        if (style) {
            style.querySelector("div.token-remove").click();
            e.preventDefault();
            e.stopPropagation();
        }
    });
    styleDeselectionTargets.add(dropdown);
    dropdown.dataset.forgeStyleDeselectionBound = "true";
    return true;
}

onUiLoaded(load_webui_settings);

const generationTabSetupReady = { txt2img: false, img2img: false };
let generationTabSetupTimer = null;
let generationTabSetupAttempts = 0;
const GENERATION_TAB_SETUP_MAX_ATTEMPTS = 40;

function setupGenerationTabInteractions(tabname) {
    showRestoreProgressButton(tabname, localGet(`${tabname}_task_id`));
    const resolutionReady = setupResolutionPasting(tabname);
    const stylesReady = restoreStyleDeselection(tabname);
    generationTabSetupReady[tabname] = resolutionReady && stylesReady;
    return generationTabSetupReady[tabname];
}

function generationTabsAreReady() {
    return generationTabSetupReady.txt2img && generationTabSetupReady.img2img;
}

function generationTabMountIsPending() {
    return ["txt2img", "img2img"].some(function (tabname) {
        if (generationTabSetupReady[tabname]) return false;
        return Boolean(
            gradioApp().getElementById(`${tabname}_width`) ||
            gradioApp().getElementById(`${tabname}_height`) ||
            document.getElementById(`${tabname}_styles`) ||
            get_uiCurrentTab()?.getAttribute("aria-controls") === `tab_${tabname}`
        );
    });
}

function requestGenerationTabSetup() {
    if (generationTabsAreReady()) return;
    generationTabSetupAttempts = GENERATION_TAB_SETUP_MAX_ATTEMPTS;
    if (generationTabSetupTimer !== null) return;

    function retry() {
        generationTabSetupTimer = null;
        for (const tabname of ["txt2img", "img2img"]) {
            if (!generationTabSetupReady[tabname]) setupGenerationTabInteractions(tabname);
        }
        if (generationTabsAreReady()) return;
        generationTabSetupAttempts -= 1;
        if (generationTabSetupAttempts <= 0 || !generationTabMountIsPending()) return;
        generationTabSetupTimer = window.setTimeout(retry, 50);
    }
    generationTabSetupTimer = window.setTimeout(retry, 0);
}

function generationTabMutationMayMountTarget(mutationRecords) {
    if (generationTabsAreReady()) return false;
    return Array.from(mutationRecords || []).some(function (record) {
        return Array.from(record.addedNodes || []).some(function (node) {
            if (node.nodeType !== Node.ELEMENT_NODE) return false;
            return ["txt2img", "img2img"].some(function (tabname) {
                const ids = [`${tabname}_width`, `${tabname}_height`, `${tabname}_styles`];
                return ids.includes(node.id) ||
                    Boolean(node.querySelector?.(ids.map((id) => `#${id}`).join(", ")));
            });
        });
    });
}

onUiLoaded(requestGenerationTabSetup);
onUiTabChange(requestGenerationTabSetup);
onUiUpdate(function (mutationRecords) {
    if (generationTabMutationMayMountTarget(mutationRecords)) requestGenerationTabSetup();
});

function modelmerger() {
    const id = randomId();
    requestProgress(id, gradioApp().getElementById("modelmerger_results_panel"), null, function () { });

    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

function ask_for_style_name(_, prompt_text, negative_prompt_text) {
    const name = prompt("Style name:");
    return [name, prompt_text, negative_prompt_text];
}

function confirm_clear_prompt(prompt, negative_prompt) {
    if (confirm("Delete prompt?")) {
        prompt = "";
        negative_prompt = "";
    }

    return [prompt, negative_prompt];
}

function load_webui_settings(attempt = 0) {
    const retry = function () {
        if (attempt >= 200) {
            console.error("WebUI settings did not become available within 10 seconds.");
            return;
        }
        setTimeout(function () {
            load_webui_settings(attempt + 1);
        }, 50);
    };
    const json_elem = gradioApp().getElementById("settings_json");
    if (json_elem == null) {
        retry();
        return;
    }

    const textarea = json_elem.querySelector("textarea");
    if (textarea == null || !textarea.value) {
        retry();
        return;
    }
    const jsdata = textarea.value;
    try {
        opts = JSON.parse(jsdata);
    } catch (_error) {
        retry();
        return;
    }

    executeCallbacks(optionsAvailableCallbacks); // global optionsAvailableCallbacks
    executeCallbacks(optionsChangedCallbacks); // global optionsChangedCallbacks

    Object.defineProperty(textarea, "value", {
        set: function (newValue) {
            const valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
            const oldValue = valueProp.get.call(textarea);
            valueProp.set.call(textarea, newValue);

            if (oldValue != newValue) {
                opts = JSON.parse(textarea.value);
            }

            executeCallbacks(optionsChangedCallbacks);
        },
        get: function () {
            const valueProp = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value");
            return valueProp.get.call(textarea);
        },
    });

    json_elem.parentElement.style.display = "none";
}

onOptionsChanged(function () {
    const elem = gradioApp().getElementById("sd_checkpoint_hash");
    const sd_checkpoint_hash = opts.sd_checkpoint_hash || "";
    const shorthash = sd_checkpoint_hash.substring(0, 10);

    if (elem && elem.textContent != shorthash) {
        elem.textContent = shorthash;
        elem.title = sd_checkpoint_hash;
        elem.href = "https://civitai.com/search/models?query=" + sd_checkpoint_hash;
    }
});

let txt2img_textarea = undefined;
let img2img_textarea = undefined;

let isReloading = false;

function restart_reload() {
    document.body.style.backgroundColor = "var(--background-fill-primary)";
    document.body.innerHTML =
        '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    const requestPing = function () {
        requestGet(
            "./internal/ping",
            {},
            function (data) {
                location.reload();
            },
            function () {
                setTimeout(requestPing, 500);
            },
        );
    };

    setTimeout(requestPing, 2000);
    isReloading = true;
    return [];
}

window.addEventListener("beforeunload", (e) => {
    if (isReloading || !opts.confirm_leave) return;
    e.preventDefault();
    e.returnValue = "";
});

// Simulate an `input` DOM event for Gradio Textbox component. Needed after you edit its contents in javascript,
// otherwise your edits will only visible on web page and not sent to python.
function updateInput(target) {
    const e = new Event("input", { bubbles: true });
    Object.defineProperty(e, "target", { value: target });
    target.dispatchEvent(e);
}

function selectCheckpoint(name) {
    const input = gradioApp().getElementById("change_checkpoint_text").querySelector("textarea");
    input.value = name;
    updateInput(input);
    gradioApp().getElementById("change_checkpoint").click();
}

function currentImg2imgSourceResolution(w, h, r) {
    const img = gradioApp().querySelector('#mode_img2img > div[style="display: block;"] :is(img, canvas)');
    if (!img) return [0, 0, r];
    const width = img.naturalWidth || img.width;
    const height = img.naturalHeight || img.height;
    return [width, height, r];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    // At the time this is called from gradio, the image has no yet been replaced.
    // There may be a better solution, but this is simple and straightforward so I'm going with it.

    setTimeout(function () {
        gradioApp().getElementById("img2img_update_resize_to").click();
    }, 500);

    return [];
}

function setRandomSeed(elem_id) {
    const input = gradioApp().querySelector("#" + elem_id + " input");
    if (!input) return [];

    input.value = "-1";
    updateInput(input);
    return [];
}

function switchWidthHeight(tabname) {
    const width = gradioApp().querySelector("#" + tabname + "_width input[type=number]");
    const height = gradioApp().querySelector("#" + tabname + "_height input[type=number]");
    if (!width || !height) return [];

    const tmp = width.value;
    width.value = height.value;
    height.value = tmp;

    updateInput(width);
    updateInput(height);
    return [];
}

let onEditTimers = {};

// calls func after afterMs milliseconds has passed since the input elem has been edited by user
function onEdit(editId, elem, afterMs, func) {
    const edited = function () {
        const existingTimer = onEditTimers[editId];
        if (existingTimer) clearTimeout(existingTimer);
        onEditTimers[editId] = setTimeout(func, afterMs);
    };

    elem.addEventListener("input", edited);

    return edited;
}
