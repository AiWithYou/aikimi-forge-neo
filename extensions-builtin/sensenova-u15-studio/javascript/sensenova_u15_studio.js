(function () {
    const DRAFT_KEY = "forge-neo:sensenova-u15:prompt-draft:v1";
    let draftTimer = null;
    let chromeFrame = null;
    let lastPromptValue = null;

    function studioIsActive() {
        const studio = gradioApp().querySelector("#sensenova-u15-studio");
        return Boolean(studio && studio.offsetParent !== null);
    }

    function promptElement() {
        return gradioApp().querySelector("#sn-prompt textarea");
    }

    function buttonElement(id) {
        return gradioApp().querySelector(`#${id} button, button#${id}`);
    }

    function syncStudioChrome() {
        const app = gradioApp();
        app.classList.toggle("sensenova-studio-active", studioIsActive());
    }

    function syncControlSemantics() {
        const app = gradioApp();
        const prompt = promptElement();
        const generate = buttonElement("sn-generate");
        const cancel = buttonElement("sn-cancel");
        const mode = app.querySelector("#sn-mode fieldset, #sn-mode > div");
        if (prompt) {
            prompt.setAttribute("aria-label", "SenseNova プロンプト");
            prompt.setAttribute("aria-describedby", "sn-prompt-help");
        }
        if (generate) {
            generate.setAttribute("aria-keyshortcuts", "Control+Enter Meta+Enter");
        }
        if (cancel) cancel.setAttribute("aria-keyshortcuts", "Escape");
        if (mode) {
            mode.setAttribute("role", "radiogroup");
            mode.setAttribute("aria-label", "生成モード");
        }
    }

    function scheduleStudioChrome() {
        if (chromeFrame !== null) return;
        chromeFrame = window.requestAnimationFrame(function () {
            chromeFrame = null;
            syncStudioChrome();
        });
    }

    function syncPromptCount() {
        const prompt = promptElement();
        const counter = gradioApp().querySelector("#sn-prompt-count");
        if (!prompt || !counter) return;
        const length = Array.from(prompt.value).length;
        counter.textContent = `${length.toLocaleString()} / 20,000`;
        counter.dataset.overLimit = String(length > 20000);
        if (lastPromptValue !== null && lastPromptValue !== prompt.value) scheduleDraftSave();
        lastPromptValue = prompt.value;
    }

    function saveDraft() {
        const prompt = promptElement();
        const status = gradioApp().querySelector("#sn-draft-status");
        if (!prompt) return;
        try {
            if (prompt.value) {
                window.localStorage.setItem(DRAFT_KEY, prompt.value);
                if (status) status.textContent = "下書きをこの端末に保存済み";
            } else {
                window.localStorage.removeItem(DRAFT_KEY);
                if (status) status.textContent = "下書きをこの端末に自動保存";
            }
        } catch (_error) {
            if (status) status.textContent = "下書き保存は利用できません";
        }
    }

    function scheduleDraftSave() {
        if (draftTimer !== null) window.clearTimeout(draftTimer);
        draftTimer = window.setTimeout(function () {
            draftTimer = null;
            saveDraft();
        }, 300);
    }

    function syncDraftStatus() {
        const prompt = promptElement();
        const status = gradioApp().querySelector("#sn-draft-status");
        if (!prompt || !status || !prompt.value) return;
        if (status.textContent === "下書きをこの端末に自動保存") {
            status.textContent = "前回の下書きを復元しました";
        }
    }

    function syncBusyState() {
        const studio = gradioApp().querySelector("#sensenova-u15-studio");
        const progress = gradioApp().querySelector("#sn-progress [data-stage]");
        const generate = buttonElement("sn-generate");
        if (!generate || !progress) return;
        const stage = progress.dataset.stage || "idle";
        const busy = !["idle", "complete", "error", "cancelled", "cancel"].includes(stage);
        if (studio) studio.dataset.snBusy = String(busy);
        generate.setAttribute("aria-busy", String(busy));

        if (!window.AikimiStatus) return;
        if (["idle", "cancelled", "cancel"].includes(stage)) {
            window.AikimiStatus.clear("sensenova-u15");
            return;
        }
        const state = {
            prepare: "generating",
            queued: "queued",
            running: "generating",
            complete: "completed",
            error: "error",
        }[stage] || (busy ? "generating" : null);
        if (!state) return;
        const message = progress.querySelector("strong")?.textContent?.trim() || "";
        const exactError = gradioApp().querySelector("#sn-validation .sn-inline-error")?.textContent?.trim() || "";
        const progressText = progress.querySelector(".sn-progress-head span")?.textContent || "";
        const progressMatch = progressText.match(/(\d+)%/);
        window.AikimiStatus.publish("sensenova-u15", {
            state,
            progress: progressMatch ? Number(progressMatch[1]) / 100 : null,
            errorDetails: stage === "error" ? exactError || message : null,
        });
    }

    function setupStudio() {
        const app = gradioApp();
        app.addEventListener("input", function (event) {
            if (!event.target.closest?.("#sn-prompt")) return;
            syncPromptCount();
            scheduleDraftSave();
        });
        syncPromptCount();
        syncDraftStatus();
        syncBusyState();
        syncStudioChrome();
        syncControlSemantics();
        window.addEventListener("resize", scheduleStudioChrome, { passive: true });
    }

    onUiLoaded(setupStudio);
    onUiTabChange(function () {
        syncStudioChrome();
        syncControlSemantics();
    });
    onAfterUiUpdate(function () {
        syncPromptCount();
        syncDraftStatus();
        syncBusyState();
        syncStudioChrome();
        syncControlSemantics();
    });

    document.addEventListener("keydown", function (event) {
        if (!studioIsActive() || event.defaultPrevented || event.isComposing || event.keyCode === 229) return;
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            const generate = buttonElement("sn-generate");
            if (generate && !generate.disabled && generate.getAttribute("aria-busy") !== "true") {
                event.preventDefault();
                generate.click();
            }
            return;
        }
        if (event.key !== "Escape" || event.target.closest?.("[role='dialog'], [role='listbox']")) return;
        const cancel = buttonElement("sn-cancel");
        if (cancel && !cancel.disabled) {
            event.preventDefault();
            cancel.click();
        }
    });
})();
