(function () {
    "use strict";
    let scheduled = false;
    let reposition = false;
    let observed = null;
    let revealFocused = false;
    const observer = new ResizeObserver(schedule);

    function update() {
        scheduled = false;
        const workspace = gradioApp().querySelector("#qwen21-workspace");
        if (!workspace || !workspace.getClientRects().length) return;
        // Shared headers vary with window width; reserve their actual height.
        const top = Math.max(8, workspace.getBoundingClientRect().top);
        const height = `${Math.max(300, window.innerHeight - top - 12)}px`;
        if (workspace.style.getPropertyValue("--qwen-dock-height") !== height) {
            workspace.style.setProperty("--qwen-dock-height", height);
            reposition = true;
        }
        const controls = workspace.querySelector("#qwen21-controls");
        const actions = workspace.querySelector("#qwen21-run-dock");
        if (controls && actions) {
            const bounds = controls.getBoundingClientRect();
            const values = {
                "--qwen-dock-left": `${bounds.left}px`,
                "--qwen-dock-width": `${bounds.width}px`,
                "--qwen-actions-height": `${actions.getBoundingClientRect().height}px`,
            };
            Object.entries(values).forEach(([name, value]) => {
                if (workspace.style.getPropertyValue(name) !== value) {
                    workspace.style.setProperty(name, value);
                    reposition = true;
                }
            });
            if (revealFocused) {
                revealFocused = false;
                const input = document.activeElement;
                if (workspace.contains(input) && /^(INPUT|TEXTAREA)$/.test(input.tagName)) {
                    const rect = input.getBoundingClientRect();
                    const viewport = window.visualViewport;
                    const viewportBottom = viewport ? viewport.offsetTop + viewport.height : window.innerHeight;
                    const bottom = Math.min(viewportBottom, actions.getBoundingClientRect().top) - 12;
                    if (rect.bottom > bottom) window.scrollBy(0, rect.bottom - bottom);
                }
            }
        }
        if (reposition) {
            reposition = false;
            document.dispatchEvent(new CustomEvent("aikimi:layout-change"));
        }
        if (observed !== workspace.parentElement) {
            observer.disconnect();
            observed = workspace.parentElement;
            observer.observe(observed);
        }
    }
    function schedule(event) {
        if (event?.type === "resize" || event?.type === "aikimi:feature-tab-change") reposition = true;
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(update);
    }
    window.addEventListener("resize", schedule, {passive: true});
    window.addEventListener("scroll", schedule, {passive: true});
    document.addEventListener("focusin", event => {
        if (!event.target.closest?.("#qwen21-workspace")) return;
        revealFocused = true;
        schedule();
    });
    window.visualViewport?.addEventListener("resize", () => {
        revealFocused = true;
        schedule();
    }, {passive: true});
    onUiLoaded(schedule);
    onAfterUiUpdate(schedule);
    document.addEventListener("aikimi:feature-tab-change", schedule);
})();
