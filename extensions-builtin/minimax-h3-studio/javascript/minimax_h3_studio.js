(function () {
    const H3_PRESET_KEYS = ["quick", "recommended", "final"];
    const H3_MODE_HELP = {
        text: [
            "TEXT · FL2VA",
            "言葉だけから映像と32kHzステレオ音声を同時生成します。最初の一作におすすめです。",
        ],
        keyframes: [
            "KEYFRAMES · FL2VA",
            "開始画像、終了画像、または両方を指定して、その間の動きと音を生成します。",
        ],
        references: [
            "REFERENCES · REF2VA",
            "人物・画風・動き・声を画像／動画／音声から参照します。タグ順序を確認して使います。",
        ],
    };
    const H3_ADVANCED_CONTROLS = new Set(["steps", "seed", "scheduler", "ref_image_size"]);
    const H3_PROMPT_DRAFT_KEY = "forge-neo:minimax-h3:prompt-draft:v1";
    let h3ChromeFrame = null;
    let h3PromptDraftTimer = null;
    let h3LastPromptValue = null;
    let h3LastProgressAnnouncement = null;
    let h3InitializationTrigger = null;

    function setH3Text(node, value) {
        if (node && node.textContent !== value) node.textContent = value;
    }

    function setH3Attribute(node, name, value) {
        const rendered = String(value);
        if (node && node.getAttribute(name) !== rendered) node.setAttribute(name, rendered);
    }

    function setH3Data(node, name, value) {
        const rendered = String(value);
        if (node && node.dataset[name] !== rendered) node.dataset[name] = rendered;
    }

    function saveH3PromptDraft() {
        const prompt = gradioApp().querySelector("#h3-prompt textarea");
        const status = gradioApp().querySelector("#h3-draft-status");
        if (!prompt) return;
        try {
            if (prompt.value) {
                window.localStorage.setItem(H3_PROMPT_DRAFT_KEY, prompt.value);
                setH3Text(status, "下書きをこの端末に自動保存済み");
            } else {
                window.localStorage.removeItem(H3_PROMPT_DRAFT_KEY);
                setH3Text(status, "下書きをこの端末に自動保存");
            }
        } catch (_error) {
            setH3Text(status, "下書き保存は利用できません");
        }
    }

    function scheduleH3PromptDraftSave() {
        if (h3PromptDraftTimer !== null) window.clearTimeout(h3PromptDraftTimer);
        h3PromptDraftTimer = window.setTimeout(function () {
            h3PromptDraftTimer = null;
            saveH3PromptDraft();
        }, 300);
    }

    function restoreH3PromptDraft() {
        const prompt = gradioApp().querySelector("#h3-prompt textarea");
        const status = gradioApp().querySelector("#h3-draft-status");
        if (!prompt || prompt.value) return;
        try {
            const draft = window.localStorage.getItem(H3_PROMPT_DRAFT_KEY);
            if (!draft) return;
            const setter = Object.getOwnPropertyDescriptor(
                window.HTMLTextAreaElement.prototype,
                "value"
            )?.set;
            if (!setter) return;
            setter.call(prompt, draft);
            prompt.dispatchEvent(new window.Event("input", { bubbles: true }));
            setH3Text(status, "前回の下書きを復元しました");
            syncH3PromptCount();
        } catch (_error) {
            setH3Text(status, "下書き保存は利用できません");
        }
    }

    function addH3DescribedBy(node, token) {
        if (!node) return;
        const tokens = new Set((node.getAttribute("aria-describedby") ?? "").split(/\s+/).filter(Boolean));
        tokens.add(token);
        setH3Attribute(node, "aria-describedby", Array.from(tokens).join(" "));
    }

    function removeH3DescribedBy(node, token) {
        if (!node) return;
        const tokens = (node.getAttribute("aria-describedby") ?? "")
            .split(/\s+/)
            .filter((candidate) => candidate && candidate !== token);
        if (tokens.length) {
            setH3Attribute(node, "aria-describedby", tokens.join(" "));
        } else if (node.hasAttribute("aria-describedby")) {
            node.removeAttribute("aria-describedby");
        }
    }

    function isH3StudioActive() {
        return get_uiCurrentTabContent()?.id === "tab_minimax_h3_studio";
    }

    function syncH3StudioChrome() {
        const app = gradioApp();
        const active = isH3StudioActive();
        app.classList.toggle("h3-studio-active", active);

        if (!active) {
            const actionBar = app.querySelector("#h3-mobile-action-bar");
            setH3Data(actionBar, "visible", false);
            setH3Attribute(actionBar, "aria-hidden", true);
            return;
        }

        const studio = app.querySelector("#h3-studio");
        const generate = app.querySelector("#h3-generate button, button#h3-generate");
        const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
        const busy = Boolean(generate?.disabled && generate.textContent.includes("生成"));
        setH3Data(studio, "h3Busy", busy);
        if (generate) {
            setH3Attribute(generate, "title", "映像＋音声を生成 (Ctrl+Enter)");
            setH3Attribute(generate, "aria-keyshortcuts", "Control+Enter Meta+Enter");
            setH3Attribute(generate, "aria-busy", busy);
        }
        if (cancel) {
            setH3Attribute(cancel, "title", "実行中の生成を停止 (Esc)");
            setH3Attribute(cancel, "aria-keyshortcuts", "Escape");
        }
        syncH3MobileActions(generate, cancel, busy);
    }

    function syncH3MobileActions(generate, cancel, busy) {
        const app = gradioApp();
        const actionBar = app.querySelector("#h3-mobile-action-bar");
        const generateProxy = app.querySelector("#h3-mobile-generate-proxy");
        const cancelProxy = app.querySelector("#h3-mobile-cancel-proxy");
        if (generateProxy && generate) {
            setH3Text(generateProxy, generate.textContent.trim());
            if (generateProxy.disabled !== generate.disabled) generateProxy.disabled = generate.disabled;
            setH3Attribute(generateProxy, "aria-busy", busy);
        }
        if (cancelProxy && cancel) {
            if (cancelProxy.disabled !== cancel.disabled) cancelProxy.disabled = cancel.disabled;
            if (cancelProxy.hidden !== cancel.disabled) cancelProxy.hidden = cancel.disabled;
        }
        if (actionBar && generate) {
            const bounds = generate.getBoundingClientRect();
            const focusedProxy = actionBar.contains(document.activeElement);
            const visible =
                isH3StudioActive() &&
                window.matchMedia("(max-width: 620px)").matches &&
                (focusedProxy || bounds.top > window.innerHeight - 16 || bounds.bottom < 0);
            setH3Data(actionBar, "visible", visible);
            setH3Attribute(actionBar, "aria-hidden", !visible);
            if (generateProxy) generateProxy.tabIndex = visible ? 0 : -1;
            generate.tabIndex = visible ? -1 : 0;
            if (cancelProxy) cancelProxy.tabIndex = visible && !cancelProxy.hidden ? 0 : -1;
            if (cancel) cancel.tabIndex = visible ? -1 : 0;
        }
    }

    function getH3ModeInputs() {
        return Array.from(gradioApp().querySelectorAll("#h3-mode input[type='radio']"));
    }

    function syncH3ModeKeyboard() {
        const inputs = getH3ModeInputs();
        if (!inputs.length) return;
        const selected = inputs.find((input) => input.checked) ?? inputs[0];
        for (const input of inputs) {
            if (input.name !== "h3-generation-mode") input.name = "h3-generation-mode";
            const tabIndex = input === selected ? 0 : -1;
            if (input.tabIndex !== tabIndex) input.tabIndex = tabIndex;
        }
    }

    function syncH3Mode() {
        const studio = gradioApp().querySelector("#h3-studio");
        const selected = studio?.querySelector("#h3-mode input:checked");
        if (studio && selected) {
            setH3Data(studio, "h3Mode", selected.value);
            const help = H3_MODE_HELP[selected.value] ?? H3_MODE_HELP.text;
            const helpRoot = studio.querySelector("#h3-mode-help .h3-mode-help");
            const helpTitle = helpRoot?.querySelector("span");
            const helpCopy = helpRoot?.querySelector("p");
            setH3Text(helpTitle, help[0]);
            setH3Text(helpCopy, help[1]);
        }
        syncH3ModeKeyboard();
    }

    function syncH3Aspect() {
        const studio = gradioApp().querySelector("#h3-studio");
        const input = studio?.querySelector("#h3-aspect input");
        const value = input?.value?.trim();
        if (studio && value) setH3Data(studio, "h3Aspect", value.replace(":", "-"));
    }

    function syncH3PromptCount() {
        const app = gradioApp();
        const prompt = app.querySelector("#h3-prompt textarea");
        const counter = app.querySelector("#h3-prompt-count");
        if (!prompt || !counter) return;
        const count = Array.from(prompt.value).length;
        setH3Text(counter, `${count.toLocaleString()} / 20,000`);
        setH3Data(counter, "tone", count > 20000 ? "error" : count >= 18000 ? "warn" : "ready");
        if (h3LastPromptValue !== prompt.value) {
            h3LastPromptValue = prompt.value;
            scheduleH3PromptDraftSave();
        }
    }

    function syncH3ProgressAnnouncement() {
        const app = gradioApp();
        const progress = app.querySelector("#h3-progress .h3-progress[data-stage]");
        const announcer = app.querySelector("#h3-progress-announcer .h3-sr-only");
        const stage = progress?.dataset.stage;
        if (!progress || !announcer || !stage) return;
        const message = progress.querySelector("strong")?.textContent?.trim() ?? "";
        const signature = `${stage}:${message}`;
        if (signature === h3LastProgressAnnouncement) return;
        h3LastProgressAnnouncement = signature;
        const label = progress.querySelector(".h3-progress-copy span")?.textContent?.trim() ?? stage;
        setH3Attribute(announcer, "role", stage === "error" ? "alert" : "status");
        setH3Attribute(announcer, "aria-live", stage === "error" ? "assertive" : "polite");
        setH3Text(announcer, `${label}: ${message}`);

        if (!window.AikimiStatus) return;
        if (["idle", "cancelled"].includes(stage)) {
            window.AikimiStatus.clear("minimax-h3");
            return;
        }
        const state = {
            validation: "warning",
            runtime: "updating",
            prepare: "generating",
            queued: "queued",
            running: "generating",
            reconnecting: "updating",
            complete: "completed",
            error: "error",
            active: "generating",
        }[stage];
        if (!state) return;
        const progressbar = progress.querySelector("[role='progressbar']");
        const progressAttribute = progressbar?.getAttribute("aria-valuenow");
        const progressNow = progressAttribute === null ? Number.NaN : Number(progressAttribute);
        window.AikimiStatus.publish("minimax-h3", {
            state,
            progress: Number.isFinite(progressNow) ? progressNow / 100 : null,
            errorDetails: ["validation", "error"].includes(stage) ? message : null,
        });
    }

    function requestH3Initialization() {
        if (!isH3StudioActive()) return;
        const trigger = gradioApp().querySelector(
            "#h3-initialize-trigger button, button#h3-initialize-trigger"
        );
        if (!trigger || trigger.disabled || trigger === h3InitializationTrigger) return;
        h3InitializationTrigger = trigger;
        trigger.click();
    }

    function labelH3RadioGroup(selector, label, descriptionId) {
        const root = gradioApp().querySelector(selector);
        const fieldset = root?.matches?.("fieldset") ? root : root?.querySelector("fieldset");
        if (!fieldset) return;
        setH3Attribute(fieldset, "aria-label", label);
        if (descriptionId) setH3Attribute(fieldset, "aria-describedby", descriptionId);
    }

    function syncH3ControlLabels() {
        labelH3RadioGroup("#h3-mode", "生成モード", "h3-mode-help");
        labelH3RadioGroup("#h3-runtime-profile", "起動プロファイル");
        labelH3RadioGroup("#h3-ref-image-size", "参照画像サイズ");
        setH3Attribute(gradioApp().querySelector("#h3-prompt textarea"), "aria-label", "H3 プロンプト");
        setH3Attribute(
            gradioApp().querySelector("#h3-history-selector input"),
            "aria-label",
            "履歴を選択"
        );
    }

    function syncH3PresetState() {
        const app = gradioApp();
        const state = app.querySelector("#h3-preset-state [data-h3-preset]");
        const selected = state?.dataset.h3Preset ?? "custom";
        for (const key of H3_PRESET_KEYS) {
            const button = app.querySelector(
                `#h3-preset-${key} button, button#h3-preset-${key}`
            );
            setH3Attribute(button, "aria-pressed", selected === key);
        }
    }

    function syncH3Validation() {
        const app = gradioApp();
        const studio = app.querySelector("#h3-studio");
        const validation = app.querySelector("#h3-input-validation [data-h3-invalid]");
        if (!studio || !validation) {
            if (studio) {
                const candidates = studio.querySelectorAll(
                    "[aria-describedby~='h3-input-validation-message']"
                );
                for (const candidate of candidates) {
                    candidate.removeAttribute("aria-invalid");
                    removeH3DescribedBy(candidate, "h3-input-validation-message");
                }
                if ("h3Validation" in studio.dataset) delete studio.dataset.h3Validation;
                const validationProgress = studio.querySelector(
                    "#h3-progress .h3-progress[data-stage='validation']"
                );
                if (validationProgress) {
                    setH3Data(validationProgress, "stage", "idle");
                    const message = validationProgress.querySelector("strong");
                    const stage = validationProgress.querySelector(".h3-progress-copy span");
                    setH3Text(message, "入力を修正しました。生成できます");
                    setH3Text(stage, "待機中");
                }
            }
            return;
        }

        const target = validation.dataset.h3Invalid ?? "settings";
        const controlName = validation.dataset.h3Control ?? target;
        const selectors = {
            prompt: "#h3-prompt textarea",
            first_frame: "#h3-first-frame button, #h3-first-frame input[type='file']",
            reference_images: "#h3-reference-images button, #h3-reference-images input[type='file']",
            reference_videos: "#h3-reference-videos button, #h3-reference-videos input[type='file']",
            reference_audios: "#h3-reference-audios button, #h3-reference-audios input[type='file']",
            aspect: "#h3-aspect input, #h3-aspect button",
            quality: "#h3-quality input, #h3-quality button",
            duration: "#h3-duration input",
            steps: "#h3-steps input",
            seed: "#h3-seed input",
            scheduler: "#h3-scheduler input, #h3-scheduler button",
            ref_image_size: "#h3-ref-image-size input:checked, #h3-ref-image-size input",
        };
        const controlCandidates = Array.from(
            studio.querySelectorAll(selectors[controlName] ?? selectors.aspect)
        );
        const control =
            controlCandidates.find((candidate) => candidate.offsetParent !== null && !candidate.disabled) ??
            controlCandidates[0];
        const candidates = studio.querySelectorAll(
            "[aria-describedby~='h3-input-validation-message']"
        );
        for (const candidate of candidates) {
            if (candidate === control) continue;
            candidate.removeAttribute("aria-invalid");
            removeH3DescribedBy(candidate, "h3-input-validation-message");
        }
        if (control) {
            setH3Attribute(control, "aria-invalid", true);
            addH3DescribedBy(control, "h3-input-validation-message");
        }

        const signature = `${target}:${controlName}:${validation.textContent.trim()}`;
        if (studio.dataset.h3Validation === signature) return;
        studio.dataset.h3Validation = signature;
        if (!isH3StudioActive()) return;
        const focusControl = function () {
            control?.focus({ preventScroll: true });
            control?.scrollIntoView({
                block: "center",
                behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
                    ? "auto"
                    : "smooth",
            });
        };
        const focusAfterRender = function () {
            window.setTimeout(function () {
                if (studio.dataset.h3Validation === signature) focusControl();
            }, 160);
        };
        if (H3_ADVANCED_CONTROLS.has(controlName)) {
            const advanced = studio.querySelector("#h3-advanced");
            const toggle = advanced?.querySelector(":scope > button, summary");
            const expanded =
                advanced?.matches?.("[open]") ||
                toggle?.getAttribute("aria-expanded") === "true" ||
                (toggle?.classList.contains("open") &&
                    Boolean(advanced?.querySelector("#h3-seed input")?.offsetParent));
            if (toggle && !expanded) toggle.click();
            requestAnimationFrame(focusAfterRender);
        } else {
            requestAnimationFrame(focusAfterRender);
        }
    }

    function syncH3DynamicState() {
        syncH3StudioChrome();
        syncH3ProgressAnnouncement();
        if (!isH3StudioActive()) return;
        syncH3Mode();
        syncH3Aspect();
        syncH3PresetState();
        syncH3PromptCount();
        syncH3ControlLabels();
        syncH3Validation();
        requestH3Initialization();
    }

    function scheduleH3StudioChrome() {
        if (h3ChromeFrame !== null) return;
        h3ChromeFrame = requestAnimationFrame(function () {
            h3ChromeFrame = null;
            syncH3StudioChrome();
        });
    }

    function setupH3Studio() {
        syncH3DynamicState();
        const app = gradioApp();
        if (app.dataset.h3ModeListener === "ready") return;
        app.dataset.h3ModeListener = "ready";
        app.addEventListener("change", function (event) {
            if (event.target.closest?.("#h3-mode")) syncH3Mode();
            if (event.target.closest?.("#h3-aspect")) syncH3Aspect();
        });
        app.addEventListener("input", function (event) {
            if (event.target.closest?.("#h3-prompt")) {
                syncH3PromptCount();
                const promptValidation = app.querySelector(
                    "#h3-input-validation [data-h3-invalid='prompt']"
                );
                if (promptValidation) {
                    promptValidation.remove();
                    syncH3Validation();
                }
            }
            if (event.target.closest?.("#h3-aspect")) syncH3Aspect();
        });
        app.addEventListener("click", function (event) {
            const generateProxy = event.target.closest?.("#h3-mobile-generate-proxy");
            const cancelProxy = event.target.closest?.("#h3-mobile-cancel-proxy");
            if (generateProxy) {
                const generate = app.querySelector("#h3-generate button, button#h3-generate");
                if (generate && !generate.disabled) generate.click();
            } else if (cancelProxy) {
                const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
                if (cancel && !cancel.disabled) cancel.click();
            }
        });
        app.addEventListener("keydown", function (event) {
            const input = event.target.closest?.("#h3-mode input[type='radio']");
            if (!input) return;
            const inputs = getH3ModeInputs();
            const index = inputs.indexOf(input);
            if (index < 0) return;
            let nextIndex = null;
            if (event.key === "ArrowRight" || event.key === "ArrowDown") {
                nextIndex = (index + 1) % inputs.length;
            } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
                nextIndex = (index - 1 + inputs.length) % inputs.length;
            } else if (event.key === "Home") {
                nextIndex = 0;
            } else if (event.key === "End") {
                nextIndex = inputs.length - 1;
            }
            if (nextIndex === null) return;
            event.preventDefault();
            inputs[nextIndex].click();
            inputs[nextIndex].focus();
        });
        window.addEventListener("scroll", scheduleH3StudioChrome, { passive: true });
        window.addEventListener("resize", scheduleH3StudioChrome, { passive: true });
        restoreH3PromptDraft();
    }

    onUiLoaded(setupH3Studio);
    onUiTabChange(function () {
        syncH3DynamicState();
    });
    onAfterUiUpdate(syncH3DynamicState);

    document.addEventListener("keydown", function (event) {
        if (!isH3StudioActive() || event.defaultPrevented || event.isComposing || event.keyCode === 229) return;

        const app = gradioApp();
        if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
            const generate = app.querySelector("#h3-generate button, button#h3-generate");
            if (generate && !generate.disabled && generate.getAttribute("aria-busy") !== "true") {
                event.preventDefault();
                generate.click();
            }
        } else if (
            event.key === "Escape" &&
            !event.target.closest?.("[role='dialog'], [role='listbox'], [aria-expanded='true']")
        ) {
            const cancel = app.querySelector("#h3-cancel button, button#h3-cancel");
            if (cancel && !cancel.disabled) {
                event.preventDefault();
                cancel.click();
            }
        }
    });
})();
