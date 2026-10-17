(function () {
    "use strict";
    let lastMode = "txt2img";
    let pendingFeature = null;
    let scheduled = false;
    const observed = new WeakSet();
    const observer = new MutationObserver(schedule);
    const imageFeatures = new Set(["qwen_image21", "ming_image", "sensenova", "minimax_h3_image", "nanosaur2", "iris", "flux3", "moss_image"]);
    const forgeFeatures = new Set(["krea2", "anima38"]);

    function root() { return typeof gradioApp === "function" ? gradioApp() : document; }
    function modeButtons(surface) {
        const tabs = surface?.querySelector(".aikimi-image-modes");
        const list = typeof get_uiTabList === "function" ? get_uiTabList(tabs) : tabs?.querySelector('[role="tablist"]');
        return Array.from(list?.querySelectorAll('button[role="tab"]') || []);
    }
    function refresh() {
        scheduled = false;
        root().querySelectorAll(".aikimi-image-ui, .aikimi-image-panel-host").forEach(surface => {
            if (observed.has(surface)) return;
            observed.add(surface);
            observer.observe(surface, {subtree: true, childList: true, attributes: true, attributeFilter: ["class", "style"]});
        });
        root().querySelectorAll(".aikimi-image-panel-toggle").forEach(button => {
            const targetId = button.id.replace(/-toggle$/, "");
            const target = root().querySelector(`#${targetId}`);
            if (!target) return;
            const expanded = String(getComputedStyle(target).display !== "none");
            if (button.getAttribute("aria-controls") !== targetId) button.setAttribute("aria-controls", targetId);
            if (button.getAttribute("aria-expanded") !== expanded) button.setAttribute("aria-expanded", expanded);
        });
        if (!pendingFeature) return;
        const feature = pendingFeature;
        if (!imageFeatures.has(feature)) { pendingFeature = null; return; }
        const container = window.AikimiTabs?.getActiveContainer?.();
        const surface = container?.querySelector(".aikimi-image-ui");
        const buttons = modeButtons(surface);
        if (buttons.length !== 2) return;
        const target = lastMode === "img2img" && !buttons[1].disabled ? buttons[1] : buttons[0];
        pendingFeature = null;
        if (!buttons[1].disabled) lastMode = target === buttons[1] ? "img2img" : "txt2img";
        if (target.getAttribute("aria-selected") !== "true") target.click();
    }
    function schedule() {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(refresh);
    }
    document.addEventListener("click", event => {
        const button = event.target.closest?.('button[role="tab"]');
        if (!button || button.disabled) return;
        if (button.closest(".aikimi-image-modes")) lastMode = button.id.endsWith("-img2img-button") ? "img2img" : "txt2img";
        else if (event.isTrusted && ["tab_txt2img", "tab_img2img"].includes(button.getAttribute("aria-controls"))) lastMode = button.getAttribute("aria-controls").slice(4);
    });
    document.addEventListener("click", event => {
        if (!event.target.closest?.("#aikimi-feature-nav button")) return;
        const container = window.AikimiTabs?.getActiveContainer?.();
        const buttons = modeButtons(container?.querySelector(".aikimi-image-ui"));
        const selected = buttons.find(button => button.getAttribute("aria-selected") === "true");
        if (selected && !buttons[1].disabled) lastMode = selected.id.endsWith("-img2img-button") ? "img2img" : "txt2img";
        else if (forgeFeatures.has(window.AikimiTabs?.getActiveFeature?.()) && ["tab_txt2img", "tab_img2img"].includes(container?.id)) lastMode = container.id.slice(4);
    }, true);
    document.addEventListener("aikimi:feature-tab-change", event => {
        pendingFeature = event.detail?.feature;
        schedule();
    });
    onUiLoaded(schedule);
    onUiUpdate(schedule);
    onAfterUiUpdate(schedule);
})();
