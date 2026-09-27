(function () {
    "use strict";
    let scheduled = false;
    let revealInput = false;
    let observed = null;
    const observer = new ResizeObserver(schedule);

    function update() {
        scheduled = false;
        const root = gradioApp().querySelector("#qwen21-outpaint");
        const dock = root?.querySelector("#qwen21-outpaint-run-dock");
        if (!root?.getClientRects().length || !dock) return;
        if (observed !== dock) {
            observer.disconnect();
            observed = dock;
            observer.observe(dock);
        }
        const height = `${dock.getBoundingClientRect().height}px`;
        if (root.style.getPropertyValue("--outpaint-dock-height") !== height) {
            root.style.setProperty("--outpaint-dock-height", height);
        }
        if (!revealInput) return;
        revealInput = false;
        const input = document.activeElement;
        if (!root.contains(input) || !/^(INPUT|TEXTAREA)$/.test(input.tagName)) return;
        const viewport = window.visualViewport;
        const bottom = Math.min(
            viewport ? viewport.offsetTop + viewport.height : window.innerHeight,
            window.innerWidth <= 850 ? dock.getBoundingClientRect().top : window.innerHeight
        ) - 12;
        const overlap = input.getBoundingClientRect().bottom - bottom;
        if (overlap > 0) window.scrollBy(0, overlap);
    }

    function schedule() {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(update);
    }
    document.addEventListener("focusin", event => {
        if (!event.target.closest?.("#qwen21-outpaint")) return;
        revealInput = true;
        schedule();
    });
    window.visualViewport?.addEventListener("resize", () => {
        revealInput = true;
        schedule();
    }, {passive: true});
    window.addEventListener("resize", schedule, {passive: true});
    onUiLoaded(schedule);
    onAfterUiUpdate(schedule);
    document.addEventListener("aikimi:feature-tab-change", schedule);
})();
