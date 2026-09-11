function gradioApp() {
    const elems = document.getElementsByTagName("gradio-app");
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function (id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the direct tab list rendered for a Gradio Tabs container.
 *
 * Gradio 6 wraps the tab list in `.tab-wrapper`; older Gradio releases put
 * `.tab-nav` directly below the Tabs container. Keeping this lookup scoped to
 * the direct tab list avoids matching Gradio's hidden mirror/scroll buttons.
 */
function get_uiTabList(tabs) {
    if (!tabs) return null;
    return tabs.querySelector(
        ':scope > .tab-wrapper > .tab-container[role="tablist"], :scope > .tab-nav[role="tablist"], :scope > .tab-nav',
    );
}

/**
 * Get only the native tab buttons owned by a Gradio Tabs container.
 */
function get_uiTabButtons(tabs) {
    const tabList = get_uiTabList(tabs);
    if (!tabList) return [];
    return Array.from(tabList.children).filter((element) => element.tagName === "BUTTON");
}

/**
 * Resolve a tab button by the panel it controls, with an index fallback for
 * legacy Gradio markup that did not expose `aria-controls`.
 */
function get_uiTabButton(tabs, panelId) {
    const buttons = get_uiTabButtons(tabs);
    const controlled = buttons.find((button) => button.getAttribute("aria-controls") === panelId);
    if (controlled) return controlled;

    const panels = Array.from(tabs?.children || []).filter(
        (element) => element.classList?.contains("tabitem") && element.id,
    );
    const panelIndex = panels.findIndex((panel) => panel.id === panelId);
    return panelIndex >= 0 ? buttons[panelIndex] || null : null;
}

function get_uiTopTabButton(panelId) {
    return get_uiTabButton(gradioApp().querySelector("#tabs"), panelId);
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 */
function get_uiCurrentTab() {
    return get_uiTabButtons(gradioApp().querySelector("#tabs")).find(
        (button) => button.classList.contains("selected") || button.getAttribute("aria-selected") === "true",
    ) || null;
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 */
function get_uiCurrentTabContent() {
    return Array.from(gradioApp().querySelectorAll('#tabs > .tabitem[id^="tab_"]')).find(
        (panel) => uiElementIsVisible(panel),
    ) || null;
}

const uiUpdateCallbacks = [];
const uiAfterUpdateCallbacks = [];
const uiLoadedCallbacks = [];
const uiTabChangeCallbacks = [];
const optionsChangedCallbacks = [];
const optionsAvailableCallbacks = [];
let uiCurrentTab = null;
let executedOnLoaded = false;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
function onUiLoaded(callback) {
    if (executedOnLoaded) {
        callback();
        return;
    }
    uiLoadedCallbacks.push(callback);
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
function onUiTabChange(callback) {
    uiTabChangeCallbacks.push(callback);
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 */
function onOptionsChanged(callback) {
    optionsChangedCallbacks.push(callback);
}

let opts = {};

/**
 * Register callback to be called when the options (in opts global variable) are available.
 * The callback receives no arguments.
 * If you register the callback after the options are available, it's just immediately called.
 */
function onOptionsAvailable(callback) {
    if (Object.keys(opts).length > 0) {
        callback();
        return;
    }

    optionsAvailableCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

let uiAfterUpdateTimeout = null;

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function is made.
 * TL;DR: The callbacks are executed only once even when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    clearTimeout(uiAfterUpdateTimeout);
    uiAfterUpdateTimeout = setTimeout(function () {
        executeCallbacks(uiAfterUpdateCallbacks);
    }, 250);
}

document.addEventListener("DOMContentLoaded", function () {
    const mutationObserver = new MutationObserver(function (m) {
        if (!executedOnLoaded && gradioApp().querySelector("#txt2img_prompt")) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks);
        }

        executeCallbacks(uiUpdateCallbacks, m);
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && newTab !== uiCurrentTab) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks);
        }
    });
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });
});

const pendingGenerationRestarts = new WeakMap();

function cancelGenerationRestart(interruptButton) {
    const observer = pendingGenerationRestarts.get(interruptButton);
    if (!observer) return;
    observer.disconnect();
    pendingGenerationRestarts.delete(interruptButton);
}

// A real click on Interrupt means stop, not restart. Programmatic clicks below
// must keep their observer. Capture also covers handlers that stop propagation.
document.addEventListener("click", function (event) {
    if (!event.isTrusted) return;
    const button = event.composedPath().find((node) => node.matches?.("button[id$='_interrupt']"));
    if (button) cancelGenerationRestart(button);
}, true);

// Keyboard Shortcuts:
// - Ctrl + Enter to start/restart a generation
// - Alt / Option + Enter to skip a generation
// - Esc to interrupt a generation

document.addEventListener("keydown", function (e) {
    if (e.defaultPrevented || e.repeat || e.isComposing || e.keyCode === 229) return;

    const isEnter = e.key === "Enter" || e.code === "Enter";
    const isCtrlKey = e.metaKey || e.ctrlKey;
    const isAltKey = e.altKey;
    const isEsc = e.key === "Escape";
    // Ordinary prompt typing should not traverse tabs or query layout.
    if (!isEsc && !(isEnter && (isCtrlKey || isAltKey))) return;

    const currentTab = get_uiCurrentTabContent();
    if (!currentTab) return;

    const generateButton = currentTab.querySelector("button[id$=_generate]");
    const interruptButton = currentTab.querySelector("button[id$=_interrupt]");
    const skipButton = currentTab.querySelector("button[id$=_skip]");

    if (isCtrlKey && isEnter && generateButton) {
        e.preventDefault();
        if (interruptButton?.style.display === "block") {
            if (opts.ctrl_enter_interrupt) {
                cancelGenerationRestart(interruptButton);
                interruptButton.click();
                return;
            }
            if (pendingGenerationRestarts.has(interruptButton)) return;
            const observer = new MutationObserver(function () {
                if (interruptButton.style.display !== "none") return;
                cancelGenerationRestart(interruptButton);
                if (generateButton.isConnected && uiElementIsVisible(currentTab)) generateButton.click();
            });
            pendingGenerationRestarts.set(interruptButton, observer);
            // Observe before clicking: the interrupt handler may update synchronously.
            observer.observe(interruptButton, { attributes: true, attributeFilter: ["style"] });
            interruptButton.click();
        } else {
            cancelGenerationRestart(interruptButton);
            generateButton.click();
        }
        return;
    }

    if (isAltKey && isEnter && skipButton) {
        skipButton.click();
        e.preventDefault();
    }

    if (isEsc) {
        const globalPopup = document.querySelector(".global-popup");
        const lightboxModal = document.querySelector("#lightboxModal");
        if (!globalPopup || globalPopup.style.display === "none") {
            if (document.activeElement === lightboxModal) return;
            cancelGenerationRestart(interruptButton);
            if (interruptButton?.style.display === "block") {
                interruptButton.click();
                e.preventDefault();
            }
        }
    }
});

/**
 * Check whether an UI element is not in another hidden element or tab content
 */
function uiElementIsVisible(el) {
    if (!el || !el.isConnected) return false;
    // ShadowRoot is not an Element; cross it through its host instead of
    // passing it (or a detached node's null parent) to getComputedStyle.
    for (let node = el; node; node = node.parentNode || node.host) {
        if (node.nodeType === Node.ELEMENT_NODE && getComputedStyle(node).display === "none") {
            return false;
        }
    }
    return true;
}

function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}
