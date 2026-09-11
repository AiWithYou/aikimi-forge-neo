(function () {
    const patchedImages = new WeakSet();
    const targetIds = Object.freeze(["extras_image", "pnginfo_image"]);
    let enabled = false;
    let setupTimer = null;
    let setupAttempts = 0;
    const SETUP_MAX_ATTEMPTS = 40;

    /** @param {HTMLDivElement} gradioImage */
    function patchDragAndDrop(gradioImage) {
        if (!gradioImage) return false;
        if (patchedImages.has(gradioImage)) return true;
        gradioImage.addEventListener("dragover", (e) => {
            const dt = e.dataTransfer;
            const isDroppingImage = dt?.types?.includes("text/uri-list") || dt?.types?.includes("text/html");
            if (!isDroppingImage) return;

            const closeButton = gradioImage.querySelector('button[aria-label="Remove Image"]');
            if (closeButton) closeButton.click();
        });
        patchedImages.add(gradioImage);
        gradioImage.dataset.forgeImageDropBound = "true";
        return true;
    }

    function setup() {
        return targetIds.map((id) => patchDragAndDrop(gradioApp().getElementById(id))).every(Boolean);
    }

    function targetsAreReady() {
        return targetIds.every(function (id) {
            const target = gradioApp().getElementById(id);
            return target?.dataset.forgeImageDropBound === "true";
        });
    }

    function mountIsPending() {
        return targetIds.some(function (id) {
            const target = gradioApp().getElementById(id);
            return target && target.dataset.forgeImageDropBound !== "true";
        });
    }

    function requestSetup() {
        if (!enabled || targetsAreReady()) return;
        setupAttempts = SETUP_MAX_ATTEMPTS;
        if (setupTimer !== null) return;

        function retry() {
            setupTimer = null;
            if (targetsAreReady() || setup()) return;
            setupAttempts -= 1;
            if (setupAttempts <= 0 || !mountIsPending()) return;
            setupTimer = window.setTimeout(retry, 50);
        }
        setupTimer = window.setTimeout(retry, 0);
    }

    function mutationMayMountTarget(mutationRecords) {
        if (!enabled || targetsAreReady()) return false;
        return Array.from(mutationRecords || []).some(function (record) {
            return Array.from(record.addedNodes || []).some(function (node) {
                if (node.nodeType !== Node.ELEMENT_NODE) return false;
                return targetIds.includes(node.id) ||
                    Boolean(node.querySelector?.(targetIds.map((id) => `#${id}`).join(", ")));
            });
        });
    }

    onUiLoaded(requestSetup);
    onUiTabChange(requestSetup);
    onUiUpdate(function (mutationRecords) {
        if (mutationMayMountTarget(mutationRecords)) requestSetup();
    });
    onOptionsAvailable(function () {
        enabled = Boolean(opts.remove_image_on_hover);
        requestSetup();
    });
})();
