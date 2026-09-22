(function () {
    "use strict";
    let previous = null;
    let previousApi = null;
    function update() {
        const api = window.AikimiStatus;
        const element = gradioApp().querySelector("#qwen21-assistant-state [data-state]");
        if (!api || !element) return;
        const data = element.dataset;
        const signature = JSON.stringify([data.job, data.state, data.progress, data.message, data.model, data.result]);
        if (signature === previous && api === previousApi) return;
        previous = signature;
        previousApi = api;
        const state = {
            loading: "loading_model", loaded: "loading_model", rewriting: "loading_model",
            sampling: "generating", generating: "generating", decoding: "generating", saving: "generating",
            complete: "completed", failed: "error", error: "error",
        }[data.state];
        if (!state || ["idle", "cancelled"].includes(data.state)) {
            api.clear("qwen-image21");
            return;
        }
        const progress = Number(data.progress);
        api.publish("qwen-image21", {
            state, message: data.message, modelName: data.model,
            progress: Number.isFinite(progress) ? progress : null,
            errorDetails: state === "error" ? data.message : null,
            resultElementId: data.result || null,
        });
    }
    onAfterUiUpdate(update);
    onUiLoaded(update);
    document.addEventListener("aikimi:status-visibility-change", () => {
        previous = null;
        update();
    });
})();
