(function () {
    "use strict";

    const STATE_PRIORITY = {
        out_of_memory: 90,
        error: 80,
        warning: 70,
        loading_model: 60,
        updating: 50,
        generating: 50,
        queued: 40,
        completed: 30,
        idle: 10,
    };
    const STATUS_LABELS = {
        idle: "Idle",
        loading_model: "Loading model",
        generating: "Generating",
        completed: "Completed",
        queued: "Queued",
        warning: "Warning",
        error: "Error",
        out_of_memory: "Out of memory",
        updating: "Updating",
    };
    const VALID_STATES = new Set(Object.keys(STATE_PRIORITY));
    const VALID_SIZES = new Set(["small", "medium", "large"]);
    const SAFE_ASSET_NAME = /^[a-z0-9][a-z0-9._-]*$/i;
    const ERROR_SELECTORS = ["#html_log_txt2img .error", "#html_log_img2img .error", "#html_log_extras .error"];
    const ACTIVE_POLL_MS = 1500;
    const IDLE_POLL_MS = 5000;
    const MAX_BACKOFF_MS = 60000;
    const FETCH_TIMEOUT_MS = 8000;
    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    let panel = null;
    let details = null;
    let portrait = null;
    let message = null;
    let compactMetrics = null;
    let progressValue = null;
    let manifest = null;
    let manifestPromise = null;
    let manifestRetryTimer = null;
    let pollingTimer = null;
    let pollingController = null;
    let pollingFailures = 0;
    let pollingGeneration = 0;
    let optionsAvailable = false;
    let enabled = false;
    let animationEnabled = true;
    let dialogueEnabled = true;
    let stillMode = reducedMotion.matches;
    let selectedSize = "medium";
    let activeFeature = null;
    let activeContainer = null;
    let lastRenderedState = null;
    let completedUntil = 0;
    let currentIssue = null;
    let navigationIssue = null;
    let assetIssue = null;
    let portraitLoadIssue = null;
    let portraitRequestUrl = null;
    let snapshot = null;

    const tasks = new Map();
    const interruptedTasks = new Set();
    const completionTimers = new Map();
    const published = new Map();
    const publishedTimers = new Map();
    const observedErrors = new WeakSet();
    const preloadedImages = new Map();
    const failedAssetUrls = new Set();

    function appUrl(relativePath) {
        return new URL(relativePath, window.location.href).href;
    }

    function setText(element, value) {
        const next = value == null || value === "" ? "—" : String(value);
        if (element && element.textContent !== next) element.textContent = next;
    }

    function setAttribute(element, name, value) {
        const next = String(value);
        if (element && element.getAttribute(name) !== next) element.setAttribute(name, next);
    }

    function publicTechnicalDetail(value) {
        return String(value || "")
            .replace(/(?:[a-z]:[\\/]|\\\\)[^\s<>"|]+/gi, "<local-path>")
            .replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/gi, "$1<redacted>@")
            .replace(/([?&](?:token|secret|password|credential|auth|key|sig)[^=]*=)[^&#\s]+/gi, "$1<redacted>")
            .replace(/\b(token|secret|password|credential|authorization|cookie|auth)\s*[=:]\s*[^\s,;]+/gi, "$1=<redacted>")
            .slice(0, 500);
    }

    function formatBytes(value) {
        if (!Number.isFinite(value) || value < 0) return "—";
        return `${(value / 1024 ** 3).toFixed(1)} GB`;
    }

    function formatSeconds(value) {
        if (!Number.isFinite(value) || value < 0) return "—";
        if (value >= 3600) return `${Math.floor(value / 3600)}h ${Math.floor((value % 3600) / 60)}m`;
        if (value >= 60) return `${Math.floor(value / 60)}m ${Math.round(value % 60)}s`;
        return `${value.toFixed(value < 10 ? 2 : 1)} sec`;
    }

    function createMetricRow(label, field) {
        const row = document.createElement("div");
        const term = document.createElement("dt");
        const value = document.createElement("dd");
        term.textContent = label;
        value.dataset.field = field;
        row.append(term, value);
        return row;
    }

    function bindDetailsEvents() {
        if (!details || details.dataset.aikimiEventsBound === "true") return;
        details.dataset.aikimiEventsBound = "true";
        details.addEventListener("toggle", render);
    }

    function featureLabel(feature) {
        return {
            krea2: "Krea2",
            anima38: "Anima 3.8B",
            sensenova: "SenseNova",
            minimax_h3: "MiniMax H3",
        }[feature] || "Aikimi";
    }

    function statusIsActive() {
        return Boolean(
            enabled &&
                activeFeature &&
                activeContainer?.isConnected &&
                panel?.isConnected &&
                !panel.hidden,
        );
    }

    function createPanel(mountPoint) {
        if (!mountPoint?.isConnected) return null;
        if (panel?.isConnected) {
            if (panel.parentElement !== mountPoint) mountPoint.prepend(panel);
            return panel;
        }

        const existing = gradioApp().querySelector("#aikimi-status");
        if (existing) {
            panel = existing;
            details = panel.querySelector("details");
            portrait = panel.querySelector(".aikimi-status__portrait");
            message = panel.querySelector(".aikimi-status__message");
            compactMetrics = panel.querySelector(".aikimi-status__compact-metrics");
            progressValue = panel.querySelector(".aikimi-status__progress-value");
            bindDetailsEvents();
            bindPortraitEvents();
            if (panel.parentElement !== mountPoint) mountPoint.prepend(panel);
            return panel;
        }

        panel = document.createElement("aside");
        panel.id = "aikimi-status";
        panel.dataset.state = "idle";
        panel.dataset.size = "medium";
        panel.dataset.dialogue = "on";
        panel.dataset.motion = "animated";
        panel.setAttribute("aria-label", "Aikimi Status");
        panel.setAttribute("aria-hidden", "true");
        panel.hidden = true;

        details = document.createElement("details");
        details.className = "aikimi-status__disclosure";
        bindDetailsEvents();

        const summary = document.createElement("summary");
        summary.className = "aikimi-status__summary";
        summary.setAttribute("aria-label", "Open Aikimi Status details");

        const portraitWrap = document.createElement("span");
        portraitWrap.className = "aikimi-status__portrait-wrap";
        portrait = document.createElement("img");
        portrait.className = "aikimi-status__portrait";
        portrait.alt = "";
        portrait.decoding = "async";
        bindPortraitEvents();
        portraitWrap.appendChild(portrait);

        const summaryBody = document.createElement("span");
        summaryBody.className = "aikimi-status__summary-body";

        const eyebrow = document.createElement("span");
        eyebrow.className = "aikimi-status__eyebrow";
        eyebrow.dataset.field = "feature";
        eyebrow.textContent = "AIKIMI";

        message = document.createElement("span");
        message.className = "aikimi-status__message";
        message.setAttribute("role", "status");
        message.setAttribute("aria-live", "polite");
        message.textContent = "状態を確認中……";

        compactMetrics = document.createElement("span");
        compactMetrics.className = "aikimi-status__compact-metrics";
        compactMetrics.textContent = "Runtime — · Backend — · Queue —";

        const progressTrack = document.createElement("span");
        progressTrack.className = "aikimi-status__progress";
        progressTrack.setAttribute("aria-hidden", "true");
        progressValue = document.createElement("span");
        progressValue.className = "aikimi-status__progress-value";
        progressTrack.appendChild(progressValue);

        const disclosureHint = document.createElement("span");
        disclosureHint.className = "aikimi-status__disclosure-hint";
        disclosureHint.textContent = "Details";

        summaryBody.append(eyebrow, message, compactMetrics, progressTrack, disclosureHint);
        summary.append(portraitWrap, summaryBody);

        const detailPanel = document.createElement("section");
        detailPanel.className = "aikimi-status__details";
        detailPanel.setAttribute("aria-label", "Aikimi Status technical details");

        const detailHeading = document.createElement("div");
        detailHeading.className = "aikimi-status__details-heading";
        const detailTitle = document.createElement("strong");
        detailTitle.textContent = "Technical status";
        const detailCaption = document.createElement("span");
        detailCaption.textContent = "Existing logs remain authoritative";
        detailHeading.append(detailTitle, detailCaption);

        const metrics = document.createElement("dl");
        metrics.className = "aikimi-status__metrics";
        metrics.append(
            createMetricRow("Status", "status"),
            createMetricRow("Model", "model"),
            createMetricRow("Load time", "load-time"),
            createMetricRow("Progress", "progress"),
            createMetricRow("ETA", "eta"),
            createMetricRow("VRAM", "vram"),
            createMetricRow("Queue", "queue"),
            createMetricRow("Backend", "backend"),
            createMetricRow("Error details", "error"),
        );
        detailPanel.append(detailHeading, metrics);
        details.append(summary, detailPanel);
        panel.appendChild(details);
        mountPoint.prepend(panel);

        return panel;
    }

    function validManifest(value) {
        if (!value || typeof value !== "object" || !value.assets || !value.states) return false;
        if (!Number.isInteger(value.version) || value.version <= 0) return false;
        if (!VALID_STATES.has(value.default_state)) return false;
        if (!Array.isArray(value.preload) || !value.preload.every((state) => VALID_STATES.has(state))) return false;

        const assetKeys = new Set();
        for (const state of VALID_STATES) {
            const stateConfig = value.states[state];
            if (!stateConfig || typeof stateConfig.asset !== "string") return false;
            const asset = value.assets[stateConfig.asset];
            if (!asset || typeof asset !== "object") return false;
            if (typeof asset.animated !== "string" || typeof asset.still !== "string") return false;
            if (!SAFE_ASSET_NAME.test(asset.animated) || !SAFE_ASSET_NAME.test(asset.still)) return false;
            if (!Number.isInteger(asset.frames) || asset.frames < 2) return false;
            if (!Array.isArray(asset.durations_ms) || asset.durations_ms.length !== asset.frames) return false;
            if (!asset.durations_ms.every((duration) => Number.isInteger(duration) && duration >= 20)) return false;
            if (!new Set(["ping-pong", "once"]).has(asset.loop)) return false;
            assetKeys.add(stateConfig.asset);
        }
        return assetKeys.size === VALID_STATES.size;
    }

    async function loadManifest() {
        if (manifest) return manifest;
        if (manifestPromise) return manifestPromise;

        manifestPromise = fetch(appUrl("./aikimi-assets/manifest.json"), {
            cache: "no-store",
            headers: { Accept: "application/json" },
        })
            .then((response) => {
                if (!response.ok) throw new Error(`Aikimi manifest returned ${response.status}`);
                return response.json();
            })
            .then((value) => {
                if (!validManifest(value)) throw new Error("Aikimi manifest is invalid");
                manifest = value;
                assetIssue = null;
                preloadConfiguredAssets();
                return value;
            })
            .catch((error) => {
                assetIssue = error.message;
                manifestPromise = null;
                if (!manifestRetryTimer) {
                    manifestRetryTimer = window.setTimeout(function () {
                        manifestRetryTimer = null;
                        if (statusIsActive()) loadManifest().then(render);
                    }, 5000);
                }
                render();
                return null;
            });

        return manifestPromise;
    }

    function sourceState(sourceElementId) {
        return sourceElementId === "extensions_installed_html" ? "updating" : "generating";
    }

    function stateCandidate(state, extras = {}) {
        return {
            state: VALID_STATES.has(state) ? state : "idle",
            priority: STATE_PRIORITY[state] || 0,
            ...extras,
        };
    }

    function highestPublishedCandidate() {
        let selected = null;
        for (const value of published.values()) {
            if (!value || !VALID_STATES.has(value.state)) continue;
            const candidate = stateCandidate(value.state, value);
            if (!selected || candidate.priority > selected.priority) selected = candidate;
        }
        return selected;
    }

    function highestTaskCandidate() {
        let selected = null;
        for (const task of tasks.values()) {
            const response = task.response || {};
            let state = task.state;
            if (response.queued) state = "queued";
            else if (response.active) state = sourceState(task.sourceElementId);

            const candidate = stateCandidate(state, {
                progress: response.progress,
                eta: response.eta,
                text: response.textinfo,
            });
            if (!selected || candidate.priority > selected.priority) selected = candidate;
        }
        return selected;
    }

    function deriveCandidate() {
        const now = Date.now();
        if (currentIssue && currentIssue.expiresAt <= now) currentIssue = null;
        const candidates = [];

        if (currentIssue) candidates.push(stateCandidate(currentIssue.state, { errorDetails: currentIssue.details }));
        if (navigationIssue) {
            candidates.push(
                stateCandidate("warning", {
                    message: navigationIssue,
                    errorDetails: navigationIssue,
                }),
            );
        }

        const external = highestPublishedCandidate();
        if (external) candidates.push(external);

        const task = highestTaskCandidate();
        const taskWaitingForModel =
            task &&
            snapshot?.model?.reload_pending &&
            (!Number.isFinite(task.progress) || task.progress === 0);
        if (snapshot?.model?.loading || taskWaitingForModel) {
            candidates.push(
                stateCandidate("loading_model", {
                    progress: snapshot.generation?.progress,
                    eta: snapshot.generation?.eta,
                }),
            );
        }

        if (task) candidates.push(task);

        if (snapshot?.generation?.active) {
            candidates.push(
                stateCandidate("generating", {
                    progress: snapshot.generation.progress,
                    eta: snapshot.generation.eta,
                    text: snapshot.generation.text,
                }),
            );
        }

        if ((snapshot?.generation?.queue_size || 0) > 0) candidates.push(stateCandidate("queued"));
        if (completedUntil > now) candidates.push(stateCandidate("completed"));
        if (pollingFailures >= 3) {
            candidates.push(
                stateCandidate("warning", {
                    errorDetails: "Backend status is temporarily unavailable.",
                    priority: 20,
                }),
            );
        }
        if (assetIssue) candidates.push(stateCandidate("warning", { errorDetails: assetIssue, priority: 15 }));
        candidates.push(stateCandidate("idle"));

        return candidates.reduce((selected, candidate) => (candidate.priority > selected.priority ? candidate : selected));
    }

    function field(name) {
        return panel?.querySelector(`[data-field="${name}"]`);
    }

    function versionedAssetUrl(filename) {
        const url = new URL(`./aikimi-assets/${filename}`, window.location.href);
        url.searchParams.set("v", String(manifest.version));
        return url.href;
    }

    function resolvePortrait(state) {
        const config = manifest?.states?.[state] || manifest?.states?.[manifest?.default_state];
        const asset = config ? manifest?.assets?.[config.asset] : null;
        if (!asset) return null;

        const animatedUrl = versionedAssetUrl(asset.animated);
        const stillUrl = versionedAssetUrl(asset.still);
        if (stillMode) {
            return failedAssetUrls.has(stillUrl) ? null : { selected: stillUrl };
        }
        if (!failedAssetUrls.has(animatedUrl)) {
            return { selected: animatedUrl };
        }
        return failedAssetUrls.has(stillUrl) ? null : { selected: stillUrl };
    }

    function clearPreloadedImages() {
        for (const image of preloadedImages.values()) image.removeAttribute("src");
        preloadedImages.clear();
    }

    function preloadStateAsset(state) {
        const descriptor = resolvePortrait(state);
        const url = descriptor?.selected;
        if (!url || preloadedImages.has(url)) return;
        const image = new Image();
        image.decoding = "async";
        image.addEventListener("load", () => preloadedImages.set(url, image), { once: true });
        image.addEventListener(
            "error",
            () => {
                preloadedImages.delete(url);
                failedAssetUrls.add(url);
                preloadStateAsset(state);
                render();
            },
            { once: true },
        );
        preloadedImages.set(url, image);
        image.src = url;
    }

    function preloadConfiguredAssets() {
        if (!statusIsActive() || !manifest) return;
        for (const state of manifest.preload) preloadStateAsset(state);
    }

    function bindPortraitEvents() {
        if (!portrait || portrait.dataset.aikimiEventsBound === "true") return;
        portrait.dataset.aikimiEventsBound = "true";
        portrait.addEventListener("load", function () {
            if ((portrait.currentSrc || portrait.src) !== portraitRequestUrl) return;
            portraitLoadIssue = null;
            portrait.hidden = false;
        });
        portrait.addEventListener("error", function () {
            const failedUrl = portrait.currentSrc || portrait.src;
            if (!failedUrl || failedUrl !== portraitRequestUrl) return;
            failedAssetUrls.add(failedUrl);
            portraitLoadIssue = "Aikimi portrait asset could not be loaded.";
            portraitRequestUrl = null;
            render();
        });
    }

    function syncPortrait(descriptor) {
        const selectedUrl = descriptor?.selected || null;
        if (!selectedUrl) {
            portraitRequestUrl = null;
            portrait.removeAttribute("src");
            portrait.hidden = true;
            return;
        }
        if (portraitRequestUrl === selectedUrl) return;
        portraitRequestUrl = selectedUrl;
        portrait.hidden = false;
        portrait.src = selectedUrl;
    }

    function render() {
        if (!statusIsActive()) return;

        const candidate = deriveCandidate();
        const state = candidate.state;
        const stateConfig = manifest?.states?.[state] || manifest?.states?.[manifest?.default_state] || {};
        const backendGeneration = snapshot?.generation || {};
        const model = snapshot?.model || {};
        const memory = snapshot?.memory || {};
        const backend = snapshot?.backend || {};
        const progress = Number.isFinite(candidate.progress) ? candidate.progress : backendGeneration.progress;
        const eta = Number.isFinite(candidate.eta) ? candidate.eta : backendGeneration.eta;
        const snapshotQueueSize = Number.isFinite(backendGeneration.queue_size) ? backendGeneration.queue_size : 0;
        const queueSize = Math.max(snapshotQueueSize, state === "queued" ? 1 : 0);
        const progressPercent = Number.isFinite(progress) ? Math.round(Math.min(Math.max(progress, 0), 1) * 100) : null;
        const stateMessage = candidate.message || stateConfig.message || STATUS_LABELS[state] || state;
        const portraitDescriptor = resolvePortrait(state);
        const modelName = model.loaded_name || "Not loaded";
        const modelLabel =
            model.reload_pending && model.selected_name
                ? `${modelName} · selected ${model.selected_name}`
                : modelName;

        const isUrgent = state === "error" || state === "out_of_memory";
        setAttribute(message, "role", isUrgent ? "alert" : "status");
        setAttribute(message, "aria-live", isUrgent ? "assertive" : "polite");

        if (panel.dataset.state !== state) panel.dataset.state = state;
        setText(field("feature"), featureLabel(activeFeature));
        if (lastRenderedState !== state) {
            setText(message, stateMessage);
            lastRenderedState = state;
        } else if (message.textContent !== stateMessage) {
            setText(message, stateMessage);
        }

        syncPortrait(portraitDescriptor);

        const compact = [`Runtime ${STATUS_LABELS[state] || state}`];
        compact.push(`Backend ${backend.ready ? "Online" : "Unavailable"}`);
        compact.push(`Queue ${queueSize}`);
        if (progressPercent != null && ["loading_model", "generating", "updating"].includes(state)) {
            compact.push(`${progressPercent}%`);
        }
        setText(compactMetrics, compact.join(" · "));

        const progressWidth = progressPercent == null ? 0 : progressPercent;
        if (progressValue.style.width !== `${progressWidth}%`) progressValue.style.width = `${progressWidth}%`;
        setAttribute(
            details.querySelector("summary"),
            "aria-label",
            dialogueEnabled
                ? `${STATUS_LABELS[state] || state}: ${stateMessage}. ${details.open ? "Close" : "Open"} technical details.`
                : `${STATUS_LABELS[state] || state}. ${details.open ? "Close" : "Open"} technical details.`,
        );

        setText(field("status"), STATUS_LABELS[state] || state);
        setText(field("model"), modelLabel);
        setText(field("load-time"), formatSeconds(model.last_load_seconds));
        setText(field("progress"), progressPercent == null ? "—" : `${progressPercent}%${candidate.text ? ` · ${candidate.text}` : ""}`);
        setText(field("eta"), formatSeconds(eta));
        setText(
            field("vram"),
            memory.available
                ? `${formatBytes(memory.used)} / ${formatBytes(memory.total)} · allocated ${formatBytes(memory.allocated)}`
                : memory.error || "Unavailable",
        );
        setText(field("queue"), `${queueSize} waiting${candidate.state === "queued" && candidate.text ? ` · ${candidate.text}` : ""}`);
        setText(field("backend"), backend.ready ? `Online · uptime ${formatSeconds(backend.uptime_seconds)}` : "Unavailable");
        setText(field("error"), publicTechnicalDetail(candidate.errorDetails || portraitLoadIssue || "None"));

    }

    function requestFreshSnapshot() {
        if (pollingTimer) window.clearTimeout(pollingTimer);
        pollingTimer = null;
        pollingGeneration += 1;
        if (pollingController) pollingController.abort();
        pollingController = null;
        schedulePoll();
    }

    function handleTaskStart(event) {
        if (!enabled) return;
        const detail = event.detail || {};
        if (!detail.taskId) return;

        currentIssue = null;
        completedUntil = 0;
        for (const timer of completionTimers.values()) window.clearTimeout(timer);
        completionTimers.clear();
        tasks.set(detail.taskId, {
            state: sourceState(detail.sourceElementId),
            sourceElementId: detail.sourceElementId,
            response: null,
        });
        requestFreshSnapshot();
        render();
    }

    function handleTaskProgress(event) {
        if (!enabled) return;
        const detail = event.detail || {};
        const response = detail.response || {};
        if (!detail.taskId) return;

        if (response.completed) {
            if (interruptedTasks.has(detail.taskId)) {
                tasks.delete(detail.taskId);
                interruptedTasks.delete(detail.taskId);
            } else {
                tasks.set(detail.taskId, {
                    state: sourceState(detail.sourceElementId),
                    sourceElementId: detail.sourceElementId,
                    response: { active: true, progress: 1, textinfo: "Finishing" },
                });
                const existingTimer = completionTimers.get(detail.taskId);
                if (existingTimer) window.clearTimeout(existingTimer);
                completionTimers.set(
                    detail.taskId,
                    window.setTimeout(function () {
                        completionTimers.delete(detail.taskId);
                        tasks.delete(detail.taskId);
                        if (!currentIssue) completedUntil = Date.now() + 4500;
                        render();
                    }, 750),
                );
            }
        } else {
            tasks.set(detail.taskId, {
                state: response.queued ? "queued" : sourceState(detail.sourceElementId),
                sourceElementId: detail.sourceElementId,
                response,
            });
        }
        render();
    }

    function handleTaskError(event) {
        if (!enabled) return;
        const detail = event.detail || {};
        const completionTimer = completionTimers.get(detail.taskId);
        if (completionTimer) window.clearTimeout(completionTimer);
        completionTimers.delete(detail.taskId);
        if (detail.taskId) tasks.delete(detail.taskId);
        if (detail.taskId) interruptedTasks.delete(detail.taskId);
        currentIssue = {
            state: "warning",
            details: "Progress status could not be retrieved.",
            expiresAt: Date.now() + 15000,
        };
        render();
    }

    function handleTaskEnd(event) {
        const taskId = event.detail?.taskId;
        if (!taskId) return;
        if (completionTimers.has(taskId)) return;
        tasks.delete(taskId);
        interruptedTasks.delete(taskId);
        render();
    }

    function scanOutputErrors() {
        if (!statusIsActive()) return;
        for (const selector of ERROR_SELECTORS) {
            for (const node of gradioApp().querySelectorAll(selector)) {
                if (observedErrors.has(node)) continue;
                observedErrors.add(node);
                const text = node.textContent.trim();
                if (!text) continue;

                const normalized = text.toLowerCase();
                const outOfMemory = normalized === "oom" || normalized.includes("out of memory");
                currentIssue = {
                    state: outOfMemory ? "out_of_memory" : "error",
                    details: publicTechnicalDetail(text),
                    expiresAt: Number.POSITIVE_INFINITY,
                };
                completedUntil = 0;
            }
        }
        render();
    }

    async function poll() {
        pollingTimer = null;
        if (!statusIsActive() || document.hidden) return;

        const generation = ++pollingGeneration;
        const controller = new AbortController();
        pollingController = controller;
        let timedOut = false;
        const timeout = window.setTimeout(() => {
            timedOut = true;
            controller.abort();
        }, FETCH_TIMEOUT_MS);
        try {
            const response = await fetch(appUrl("./internal/aikimi-status"), {
                cache: "no-store",
                headers: { Accept: "application/json" },
                signal: controller.signal,
            });
            if (!response.ok) throw new Error(`Status returned ${response.status}`);

            snapshot = await response.json();
            pollingFailures = 0;

            render();
        } catch (error) {
            if (error.name !== "AbortError" || timedOut) {
                pollingFailures += 1;
                render();
            }
        } finally {
            window.clearTimeout(timeout);
            if (pollingController === controller) pollingController = null;
            if (generation === pollingGeneration && statusIsActive() && !document.hidden) {
                const active = tasks.size > 0 || snapshot?.generation?.active || snapshot?.model?.loading;
                const baseDelay = active ? ACTIVE_POLL_MS : IDLE_POLL_MS;
                const delay = pollingFailures
                    ? Math.min(MAX_BACKOFF_MS, baseDelay * 2 ** Math.min(pollingFailures, 6))
                    : baseDelay;
                schedulePoll(delay);
            }
        }
    }

    function schedulePoll(delay = 0) {
        if (!statusIsActive() || document.hidden || pollingTimer || pollingController) return;
        pollingTimer = window.setTimeout(poll, delay);
    }

    function stopPolling() {
        if (pollingTimer) window.clearTimeout(pollingTimer);
        pollingTimer = null;
        if (pollingController) pollingController.abort();
        pollingController = null;
        pollingGeneration += 1;
    }

    function selectedOption(value, allowed, fallback) {
        return allowed.has(value) ? value : fallback;
    }

    function syncPreferences() {
        const nextAnimationEnabled = opts.aikimi_assistant_animation_enabled !== false;
        const nextDialogueEnabled = opts.aikimi_assistant_dialogue_enabled !== false;
        const nextStillMode = reducedMotion.matches || !nextAnimationEnabled;
        const portraitModeChanged = nextStillMode !== stillMode;

        animationEnabled = nextAnimationEnabled;
        dialogueEnabled = nextDialogueEnabled;
        stillMode = nextStillMode;
        selectedSize = selectedOption(opts.aikimi_assistant_size, VALID_SIZES, "medium");

        panel.dataset.size = selectedSize;
        panel.dataset.dialogue = dialogueEnabled ? "on" : "off";
        panel.dataset.motion = !animationEnabled ? "disabled" : reducedMotion.matches ? "reduced" : "animated";
        message.hidden = !dialogueEnabled;

        if (portraitModeChanged) {
            portraitRequestUrl = null;
            clearPreloadedImages();
        }
    }

    function readActiveFeatureContext(event) {
        const tabs = window.AikimiTabs;
        const eventFeature = event?.detail?.feature || null;
        const eventContainer = event?.detail?.container || null;
        const feature = tabs?.getActiveFeature?.() ?? eventFeature;
        const container = tabs?.getActiveContainer?.() ?? eventContainer;
        return {
            feature: typeof feature === "string" && feature ? feature : null,
            container: container instanceof Element ? container : null,
            warning:
                typeof event?.detail?.warning === "string" && event.detail.warning
                    ? publicTechnicalDetail(event.detail.warning)
                    : null,
        };
    }

    function syncFeatureContext(event) {
        const next = readActiveFeatureContext(event);
        const changed = next.feature !== activeFeature || next.container !== activeContainer;
        const featureEvent = event?.type === "aikimi:feature-tab-change";
        const nextNavigationIssue = next.feature && featureEvent ? next.warning : navigationIssue;
        const issueChanged = nextNavigationIssue !== navigationIssue;
        const mountNeedsRepair = Boolean(
            next.container && (!panel?.isConnected || panel.parentElement !== next.container),
        );
        if (!changed && !issueChanged && !mountNeedsRepair) return;

        activeFeature = next.feature;
        activeContainer = next.container;
        navigationIssue = next.feature ? nextNavigationIssue : null;
        if (details && changed) details.open = false;
        lastRenderedState = null;
        syncVisibility();
    }

    function syncVisibility() {
        if (!optionsAvailable) return;
        enabled = opts.aikimi_assistant_enabled !== false;
        const shouldShow = Boolean(enabled && activeFeature && activeContainer?.isConnected);

        if (shouldShow && createPanel(activeContainer)) {
            syncPreferences();
            panel.dataset.feature = activeFeature;
            panel.hidden = false;
            setAttribute(panel, "aria-hidden", "false");

            loadManifest().then(function () {
                preloadConfiguredAssets();
                render();
            });
            schedulePoll();
        } else {
            if (panel) {
                panel.hidden = true;
                setAttribute(panel, "aria-hidden", "true");
            }
            stopPolling();
        }

        if (!enabled) {
            portraitRequestUrl = null;
            portrait?.removeAttribute("src");
            if (portrait) portrait.hidden = true;
            clearPreloadedImages();
            tasks.clear();
            published.clear();
            for (const timer of completionTimers.values()) window.clearTimeout(timer);
            completionTimers.clear();
            for (const timer of publishedTimers.values()) window.clearTimeout(timer);
            publishedTimers.clear();
        }
    }

    function handleDocumentClick(event) {
        if (!enabled) return;
        const button = event.target.closest?.("button");
        if (!button) return;

        if (button.id?.endsWith("_interrupt")) {
            completedUntil = 0;
            const sourceId = `${button.id.slice(0, -"_interrupt".length)}_gallery_container`;
            for (const [taskId, task] of tasks) {
                if (task.sourceElementId === sourceId) interruptedTasks.add(taskId);
            }
        }

        if (button.id === "settings_restart_gradio") {
            published.set("aikimi-ui-reload", { state: "updating", message: "UIを更新してる……" });
            render();
        }
    }

    function handleVisibilityChange() {
        if (document.hidden) stopPolling();
        else if (statusIsActive()) schedulePoll();
    }

    function handleReducedMotionChange() {
        if (!statusIsActive()) return;
        syncPreferences();
        preloadConfiguredAssets();
        render();
    }

    function handleDetailsEscape(event) {
        if (event.key !== "Escape" || !details?.open) return;
        if (event.target.closest?.("[role='dialog'], #lightboxModal")) return;

        const popup = document.querySelector(".global-popup");
        if (popup && getComputedStyle(popup).display !== "none") return;

        event.preventDefault();
        event.stopImmediatePropagation();
        details.open = false;
        details.querySelector("summary")?.focus({ preventScroll: true });
    }

    function initialize() {
        if (window.__aikimiStatusInitialized) return;
        window.__aikimiStatusInitialized = true;

        window.addEventListener("webui:task-start", handleTaskStart);
        window.addEventListener("webui:task-progress", handleTaskProgress);
        window.addEventListener("webui:task-error", handleTaskError);
        window.addEventListener("webui:task-end", handleTaskEnd);
        document.addEventListener("click", handleDocumentClick, true);
        document.addEventListener("keydown", handleDetailsEscape, true);
        document.addEventListener("visibilitychange", handleVisibilityChange);
        document.addEventListener("aikimi:feature-tab-change", syncFeatureContext);
        reducedMotion.addEventListener("change", handleReducedMotionChange);

        window.AikimiStatus = {
            publish(source, value) {
                if (!enabled || !source || !value || !VALID_STATES.has(value.state)) return;
                const key = String(source);
                const existingTimer = publishedTimers.get(key);
                if (existingTimer) window.clearTimeout(existingTimer);
                publishedTimers.delete(key);
                published.set(key, { ...value });
                if (value.state === "completed") {
                    publishedTimers.set(
                        key,
                        window.setTimeout(function () {
                            published.delete(key);
                            publishedTimers.delete(key);
                            render();
                        }, 4500),
                    );
                }
                render();
            },
            clear(source) {
                const key = String(source);
                const existingTimer = publishedTimers.get(key);
                if (existingTimer) window.clearTimeout(existingTimer);
                publishedTimers.delete(key);
                published.delete(key);
                render();
            },
        };

        syncFeatureContext();
    }

    function handleOptionsAvailable() {
        optionsAvailable = true;
        syncVisibility();
    }

    function handleUiUpdate() {
        syncFeatureContext();
        if (statusIsActive()) scanOutputErrors();
    }

    onUiLoaded(initialize);
    onOptionsAvailable(handleOptionsAvailable);
    onOptionsChanged(handleOptionsAvailable);
    onAfterUiUpdate(handleUiUpdate);
})();
