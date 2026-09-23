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
