(function () {
    // Tag Autocomplete still targets the pre-Gradio-6 label > textarea shape.
    // Keep its own completion logic and settings; only teach it where Forge's
    // prompt textareas now live.
    onUiLoaded(() => {
        if (typeof core === "undefined" || !Array.isArray(core) || typeof getTextAreaIdentifier !== "function") {
            return;
        }
        if (!document.querySelector('script[src*="/tagAutocomplete.js?"]')) return;

        const promptIds = ["txt2img_prompt", "txt2img_neg_prompt", "img2img_prompt", "img2img_neg_prompt"];
        for (const id of promptIds) {
            const selector = `#${id} textarea`;
            if (!core.includes(selector)) core.push(selector);
        }

        const originalIdentifier = getTextAreaIdentifier;
        getTextAreaIdentifier = function (textArea) {
            const id = textArea.closest("#txt2img_prompt, #txt2img_neg_prompt, #img2img_prompt, #img2img_neg_prompt")?.id;
            if (id === "txt2img_prompt") return ".txt2img.p";
            if (id === "txt2img_neg_prompt") return ".txt2img.n";
            if (id === "img2img_prompt") return ".img2img.p";
            if (id === "img2img_neg_prompt") return ".img2img.n";
            return originalIdentifier(textArea);
        };

        // TAC measures the caret inside the textarea, but Gradio 6 positions
        // the popup inside a separate input container. Convert through viewport
        // coordinates so either prompt layout keeps the list beside the caret.
        if (typeof showResults === "function" && typeof getCaretCoordinates === "function") {
            const originalShowResults = showResults;
            let activeArea = null;
            let observedPopup = null;
            let resizeFrame = null;
            const popupResize = new ResizeObserver(() => {
                if (resizeFrame !== null) return;
                resizeFrame = requestAnimationFrame(() => {
                    resizeFrame = null;
                    if (activeArea) positionResults(activeArea);
                });
            });

            function positionResults(area) {
                const id = area.closest("#txt2img_prompt, #txt2img_neg_prompt, #img2img_prompt, #img2img_neg_prompt")?.id;
                if (!id || !area.isConnected) return;

                const popup = area.parentElement?.querySelector(".autocompleteParent");
                if (!popup || popup.style.display !== "flex") return;
                if (observedPopup !== popup) {
                    if (observedPopup) popupResize.unobserve(observedPopup);
                    popupResize.observe(popup);
                    observedPopup = popup;
                }

                const margin = 8;
                const areaRect = area.getBoundingClientRect();
                if (areaRect.bottom <= 0 || areaRect.top >= window.innerHeight || areaRect.right <= 0 || areaRect.left >= window.innerWidth) {
                    hideResults(area);
                    return;
                }
                const list = popup.querySelector(".autocompleteResults:not(.sideInfo)");
                const configuredHeight = Math.max(50, (Number(TAC_CFG?.maxResults) || 5) * 50);
                if (list) list.style.maxHeight = `${Math.min(configuredHeight, window.innerHeight - margin * 3)}px`;

                const sliding = TAC_CFG?.slidingPopup;
                const caret = sliding ? getCaretCoordinates(area, area.selectionEnd) : null;
                const lineHeight = parseFloat(getComputedStyle(area).lineHeight) || 20;
                const caretTop = caret ? areaRect.top + caret.top - area.scrollTop : areaRect.bottom;
                const targetX = caret ? areaRect.left + caret.left - area.scrollLeft : areaRect.left;
                const targetY = caretTop + (caret ? lineHeight : 0) + 4;
                const popupRect = popup.getBoundingClientRect();
                const x = Math.max(margin, Math.min(targetX, window.innerWidth - popupRect.width - margin));
                const spaceBelow = window.innerHeight - targetY - margin;
                const spaceAbove = caretTop - margin;
                const placeAbove = popupRect.height > spaceBelow && spaceAbove > spaceBelow;

                if (list) {
                    const available = placeAbove ? spaceAbove : spaceBelow;
                    list.style.maxHeight = `${Math.min(configuredHeight, Math.max(80, available - 12))}px`;
                }

                const height = popup.getBoundingClientRect().height;
                const targetTop = placeAbove ? caretTop - height - 4 : targetY;
                const y = Math.max(margin, Math.min(targetTop, window.innerHeight - height - margin));
                const offsetParent = popup.offsetParent || document.documentElement;
                const parentRect = offsetParent.getBoundingClientRect();
                popup.style.left = `${x - parentRect.left - offsetParent.clientLeft + offsetParent.scrollLeft}px`;
                popup.style.top = `${y - parentRect.top - offsetParent.clientTop + offsetParent.scrollTop}px`;
            }

            showResults = function (area) {
                originalShowResults(area);
                activeArea = area;
                positionResults(area);
            };

            const reposition = () => {
                if (activeArea) positionResults(activeArea);
            };
            window.addEventListener("resize", reposition);
            document.addEventListener("scroll", reposition, true);
        }

        // Gradio creates img2img's textarea only when its tab is opened. TAC's
        // one-time setup has already run by then, so attach its normal handler.
        onAfterUiUpdate(() => {
            if (typeof TAC_CFG === "undefined" || !TAC_CFG || typeof addAutocompleteToArea !== "function") return;
            for (const id of promptIds) {
                const area = gradioApp().querySelector(`#${id} textarea`);
                if (area && !area.classList.contains("autocomplete")) addAutocompleteToArea(area);
            }
        });
    });
})();
