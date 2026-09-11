(function () {
    "use strict";

    const API_VERSION = 4;
    const FEATURE_EVENT = "aikimi:feature-tab-change";
    const FEATURE_NAV_ID = "aikimi-feature-nav";
    const SETUP_RETRY_LIMIT = 80;
    const SETUP_RETRY_DELAY_MS = 50;
    const REPAIR_LIMIT = 12;

    const FEATURES = Object.freeze({
        krea2: Object.freeze({
            buttonId: "aikimi-tab-krea2",
            containerId: "tab_txt2img",
            containerIds: Object.freeze(["tab_txt2img", "tab_img2img"]),
            label: "Krea2",
            kind: "alias",
            preset: "krea",
        }),
        anima38: Object.freeze({
            buttonId: "aikimi-tab-anima38",
            containerId: "tab_txt2img",
            containerIds: Object.freeze(["tab_txt2img", "tab_img2img"]),
            label: "Anima",
            kind: "alias",
            preset: "anima",
        }),
        sensenova: Object.freeze({
            buttonId: "aikimi-tab-sensenova",
            containerId: "tab_sensenova_u15_studio",
            label: "SenseNova",
            kind: "native",
        }),
        minimax_h3: Object.freeze({
            buttonId: "aikimi-tab-minimax-h3",
            containerId: "tab_minimax_h3_studio",
            label: "MiniMax H3",
            kind: "native",
        }),
    });
    const FEATURE_ORDER = Object.freeze([
        "krea2",
        "anima38",
        "sensenova",
        "minimax_h3",
    ]);

    if (window.AikimiTabs?.apiVersion === API_VERSION) {
        window.AikimiTabs.refresh();
        return;
    }

    let activeFeature = null;
    let activatingFeature = null;
    let activationSequence = 0;
    let activationQueue = Promise.resolve();
    let setupTimer = null;
    let reconcileTimer = null;
    let repairCount = 0;
    let dispatchedContainer = null;

    function appRoot() {
        return typeof gradioApp === "function" ? gradioApp() : document;
    }

    function topTabs() {
        return appRoot().querySelector("#tabs");
    }

    function tabListFor(tabs) {
        if (!tabs) return null;
        if (typeof get_uiTabList === "function") return get_uiTabList(tabs);
        return tabs.querySelector(
            ':scope > .tab-wrapper > .tab-container[role="tablist"], :scope > .tab-nav[role="tablist"], :scope > .tab-nav',
        );
    }

    function topTabNav() {
        return tabListFor(topTabs());
    }

    function nativeTabButtons() {
        const nav = topTabNav();
        if (!nav) return [];
        return Array.from(nav.children).filter(function (element) {
            return element.tagName === "BUTTON";
        });
    }

    function topTabItems() {
        const tabs = topTabs();
        if (!tabs) return [];
        return Array.from(tabs.children).filter(function (element) {
            return element.classList?.contains("tabitem") && element.id?.startsWith("tab_");
        });
    }

    function nativeButtonFor(containerId) {
        if (typeof get_uiTopTabButton === "function") {
            const helperButton = get_uiTopTabButton(containerId);
            if (helperButton?.parentElement === topTabNav()) return helperButton;
        }

        const buttons = nativeTabButtons();
        const controlled = buttons.find(function (button) {
            return button.getAttribute("aria-controls") === containerId;
        });
        if (controlled) return controlled;

        const identified = buttons.find(function (button) {
            return button.id === `${containerId}-button`;
        });
        if (identified) return identified;

        const panelIndex = topTabItems().findIndex(function (panel) {
            return panel.id === containerId;
        });
        return panelIndex >= 0 ? buttons[panelIndex] || null : null;
    }

    function featureContainerIds(config) {
        if (!config) return [];
        return config.containerIds || [config.containerId];
    }

    function preferredContainerId(feature) {
        const config = FEATURES[feature];
        if (!config) return null;
        const selected = selectedNativeButton();
        return featureContainerIds(config).find(function (containerId) {
            return Boolean(selected) && nativeButtonFor(containerId) === selected;
        }) || featureContainerIds(config).find(function (containerId) {
            return Boolean(nativeButtonFor(containerId));
        }) || null;
    }

    function featureContainer(feature) {
        const containerId = preferredContainerId(feature);
        return containerId ? appRoot().querySelector(`#${containerId}`) : null;
    }

    function featureNavigation() {
        return appRoot().querySelector(`#${FEATURE_NAV_ID}`);
    }

    function featureButton(feature) {
        const config = FEATURES[feature];
        return config ? featureNavigation()?.querySelector(`#${config.buttonId}`) : null;
    }

    function syncFeatureButtonTarget(feature, containerId = preferredContainerId(feature)) {
        const button = featureButton(feature);
        const fallback = FEATURES[feature]?.containerId;
        if (button && (containerId || fallback)) {
            button.setAttribute("aria-controls", containerId || fallback);
        }
    }

    function syncFeatureButtonTargets() {
        FEATURE_ORDER.forEach(function (feature) {
            syncFeatureButtonTarget(feature);
        });
    }

    function dispatchFeatureChange(feature, warning) {
        const container = feature ? featureContainer(feature) : null;
        dispatchedContainer = container;
        document.dispatchEvent(
            new CustomEvent(FEATURE_EVENT, {
                detail: Object.freeze({
                    feature,
                    container,
                    ready: !warning,
                    warning: warning || null,
                }),
            }),
        );
    }

    function syncFeatureButtonState(feature) {
        const row = featureNavigation();
        if (!row) return;
        row.setAttribute("aria-busy", String(Boolean(activatingFeature)));
        if (feature) {
            row.dataset.activeFeature = feature;
        } else {
            delete row.dataset.activeFeature;
        }
        FEATURE_ORDER.forEach(function (candidate) {
            const button = featureButton(candidate);
            if (!button) return;
            const isActive = candidate === feature;
            const busy = candidate === activatingFeature;
            const label = FEATURES[candidate].label + (busy ? " …" : "");
            if (button.textContent !== label) button.textContent = label;
            button.setAttribute("aria-busy", String(busy));
            button.classList.toggle("aikimi-feature-active", isActive);
            if (isActive && !busy) {
                button.setAttribute("aria-current", "page");
            } else {
                button.removeAttribute("aria-current");
            }
        });
    }

    function setActiveFeature(feature, warning) {
        activeFeature = feature && FEATURES[feature] ? feature : null;
        syncFeatureButtonState(activeFeature);
        dispatchFeatureChange(activeFeature, warning);
    }

    function createFeatureButton(feature) {
        const config = FEATURES[feature];
        const button = document.createElement("button");
        button.type = "button";
        button.id = config.buttonId;
        button.className = `aikimi-feature-nav__button aikimi-feature-nav__button--${feature}`;
        button.dataset.aikimiFeature = feature;
        button.setAttribute("aria-controls", config.containerId);
        button.setAttribute("aria-label", `${config.label}をForgeで開く`);
        button.textContent = config.label;
        return button;
    }

    function createFeatureNavigation() {
        const row = document.createElement("nav");
        row.id = FEATURE_NAV_ID;
        row.className = "aikimi-feature-nav";
        row.setAttribute("aria-label", "Aikimi機能");
        FEATURE_ORDER.forEach(function (feature) {
            row.append(createFeatureButton(feature));
        });
        row.addEventListener("click", handleFeatureNavigationClick);
        row.addEventListener("keydown", handleFeatureNavigationKeydown);
        return row;
    }

    function ensureFeatureNavigation() {
        const tabs = topTabs();
        const nativeNav = topTabNav();
        if (!tabs?.parentElement || !nativeNav) return false;

        let row = featureNavigation();
        if (!row) row = createFeatureNavigation();
        if (row.parentElement !== tabs.parentElement || row.nextElementSibling !== tabs) {
            tabs.parentElement.insertBefore(row, tabs);
        }

        let availableCount = 0;
        FEATURE_ORDER.forEach(function (feature) {
            const button = featureButton(feature);
            const config = FEATURES[feature];
            const containerId = preferredContainerId(feature);
            const available = Boolean(containerId);
            syncFeatureButtonTarget(feature, containerId || config.containerId);
            button.hidden = !available;
            button.disabled = !available;
            if (available) availableCount += 1;
        });
        row.hidden = availableCount === 0;
        syncFeatureButtonState(activatingFeature || activeFeature);
        return true;
    }

    function waitFor(predicate, attempts = 40, delayMs = SETUP_RETRY_DELAY_MS, sequence = null) {
        return new Promise(function (resolve) {
            let remaining = attempts;
            function check() {
                // A superseded click must not wait for or mutate controls on another tab.
                if (sequence !== null && sequence !== activationSequence) {
                    resolve(null);
                    return;
                }
                const value = predicate();
                if (value || remaining <= 0) {
                    resolve(value || null);
                    return;
                }
                remaining -= 1;
                window.setTimeout(check, delayMs);
            }
            check();
        });
    }

    function normalizeGradioOptionLabel(value) {
        if (typeof value !== "string") return null;
        const trimmed = value.trim();
        return trimmed.startsWith("✓") ? trimmed.slice(1).trim() : trimmed;
    }

    function exactVisibleOption(dropdown, input, label) {
        const controlledId = input.getAttribute("aria-controls");
        const controlledListbox = controlledId
            ? appRoot().querySelector(`#${CSS.escape(controlledId)}`)
            : null;
        const optionRoot = controlledListbox || dropdown;
        return Array.from(optionRoot.querySelectorAll("[role='option']")).find(function (option) {
            if (
                option.getAttribute("aria-disabled") === "true" ||
                option.offsetParent === null
            ) {
                return false;
            }
            const candidates = [
                option.getAttribute("aria-label"),
                option.getAttribute("data-value"),
                option.getAttribute("value"),
                option.textContent,
            ];
            return candidates.some(function (candidate) {
                return normalizeGradioOptionLabel(candidate) === label;
            });
        }) || null;
    }

    function activateDropdownOption(option) {
        const gradio6Option = option.matches("li[data-index]") &&
            Boolean(option.closest("ul[role='listbox']"));
        if (gradio6Option) {
            option.dispatchEvent(new MouseEvent("mousedown", {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
        } else {
            option.click();
        }
    }

    function openGradioDropdown(input) {
        input.focus();
        input.dispatchEvent(new KeyboardEvent("keydown", {
            key: "ArrowDown",
            code: "ArrowDown",
            bubbles: true,
            cancelable: true,
        }));
    }

    async function selectFeaturePreset(feature, sequence) {
        const config = FEATURES[feature];
        if (!config?.preset) return null;
        const dropdown = await waitFor(function () {
            return appRoot().querySelector("#forge_ui_preset");
        }, 40, SETUP_RETRY_DELAY_MS, sequence);
        if (sequence !== activationSequence) return null;
        if (!dropdown) return `${config.label}のUI Preset欄が見つかりません。UIを再読み込みしてください。`;

        const input = dropdown.querySelector("input");
        if (!input) return `${config.label}のUI Preset欄を操作できません。UIを再読み込みしてください。`;

        if (input.value !== config.preset) {
            openGradioDropdown(input);
            let option = await waitFor(function () {
                return exactVisibleOption(dropdown, input, config.preset);
            }, 20, SETUP_RETRY_DELAY_MS, sequence);
            if (sequence !== activationSequence) return null;
            if (!option) {
                openGradioDropdown(input);
                option = await waitFor(function () {
                    return exactVisibleOption(dropdown, input, config.preset);
                }, 40, SETUP_RETRY_DELAY_MS, sequence);
            }
            if (sequence !== activationSequence) return null;
            if (option) activateDropdownOption(option);
            await waitFor(function () {
                return input.value === config.preset ? input : null;
            }, 60, SETUP_RETRY_DELAY_MS, sequence);
            if (sequence !== activationSequence) return null;
        }

        if (input.value !== config.preset) {
            return `${config.label}のUI Presetを選択できませんでした。UI Preset欄から手動で選択してください。`;
        }
        return null;
    }

    function animaAccordionId(containerId) {
        return containerId === "tab_img2img"
            ? "aikimi-img2img-anima38"
            : "aikimi-txt2img-anima38";
    }

    async function expandAnimaAccordion(containerId, sequence) {
        const accordionId = animaAccordionId(containerId);
        const controls = await waitFor(function () {
            const accordion = appRoot().querySelector(`#${accordionId}`);
            const visibleCheckbox = appRoot().querySelector(
                `#${accordionId}-visible-checkbox`,
            );
            const hiddenCheckbox = appRoot().querySelector(
                `#${accordionId}-checkbox input[type='checkbox']`,
            );
            const label = accordion?.querySelector(".label-wrap");
            if (
                !accordion ||
                !label ||
                !visibleCheckbox ||
                !hiddenCheckbox ||
                !accordion.visibleCheckbox ||
                typeof accordion.onVisibleCheckboxChange !== "function" ||
                typeof inputAccordionChecked !== "function"
            ) {
                return null;
            }
            return { accordion, label, visibleCheckbox, hiddenCheckbox };
        }, 60, SETUP_RETRY_DELAY_MS, sequence);
        if (sequence !== activationSequence) return null;
        if (!controls) return "Anima 3.8Bの有効化欄が準備できません。UIを再読み込みしてください。";

        inputAccordionChecked(accordionId, true);
        const opened = await waitFor(function () {
            return controls.label.classList.contains("open") &&
                controls.visibleCheckbox.checked &&
                controls.hiddenCheckbox.checked
                ? controls.accordion
                : null;
        }, 20, SETUP_RETRY_DELAY_MS, sequence);
        return opened ? null : "Anima 3.8Bを有効化できませんでした。設定欄を手動で有効にしてください。";
    }

    async function collapseAnimaAccordion(containerId, sequence) {
        if (sequence !== activationSequence) return null;
        const accordionId = animaAccordionId(containerId);
        const accordion = appRoot().querySelector(`#${accordionId}`);
        const visibleCheckbox = appRoot().querySelector(`#${accordionId}-visible-checkbox`);
        const hiddenCheckbox = appRoot().querySelector(
            `#${accordionId}-checkbox input[type='checkbox']`,
        );
        if (!accordion || !visibleCheckbox || !hiddenCheckbox) return null;
        if (!visibleCheckbox.checked && !hiddenCheckbox.checked) return null;
        if (typeof inputAccordionChecked !== "function") {
            return "Anima 3.8Bの設定を無効化できませんでした。設定欄から手動で無効にしてください。";
        }

        inputAccordionChecked(accordionId, false);
        const closed = await waitFor(function () {
            return !accordion.querySelector(".label-wrap")?.classList.contains("open") &&
                !visibleCheckbox.checked &&
                !hiddenCheckbox.checked
                ? accordion
                : null;
        }, 20, SETUP_RETRY_DELAY_MS, sequence);
        return closed ? null : "Anima 3.8Bの設定を無効化できませんでした。設定欄から手動で無効にしてください。";
    }

    async function collapseAnimaAccordions(sequence) {
        const warnings = await Promise.all([
            collapseAnimaAccordion("tab_txt2img", sequence),
            collapseAnimaAccordion("tab_img2img", sequence),
        ]);
        return warnings.find(Boolean) || null;
    }

    function clickNativeTab(containerId) {
        const button = nativeButtonFor(containerId);
        if (!button) return false;
        button.click();
        return true;
    }

    async function activateFeature(feature, sequence) {
        const config = FEATURES[feature];
        if (!config || sequence !== activationSequence) return;

        activatingFeature = feature;
        syncFeatureButtonState(feature);
        const containerId = preferredContainerId(feature);
        if (!containerId || !clickNativeTab(containerId)) {
            activatingFeature = null;
            setActiveFeature(null, `${config.label}のForgeタブが見つかりません。UIを再読み込みしてください。`);
            return;
        }
        syncFeatureButtonTarget(feature, containerId);

        let warning = null;
        if (feature === "krea2") {
            warning = await collapseAnimaAccordions(sequence);
        }
        if (sequence !== activationSequence) return;
        const presetWarning = await selectFeaturePreset(feature, sequence);
        if (sequence !== activationSequence) return;
        warning = warning || presetWarning;
        if (feature === "anima38") {
            warning = warning || await expandAnimaAccordion(containerId, sequence);
        } else {
            const mounted = await waitFor(function () {
                return featureContainer(feature);
            }, 60, SETUP_RETRY_DELAY_MS, sequence);
            if (!mounted) {
                warning = `${config.label}を開けませんでした。UIを再読み込みしてください。`;
            }
        }

        if (sequence !== activationSequence) return;
        activatingFeature = null;
        setActiveFeature(feature, warning);
    }

    function queueFeatureActivation(feature) {
        if (activatingFeature === feature) return;
        const sequence = ++activationSequence;
        activatingFeature = feature;
        syncFeatureButtonState(feature);
        activationQueue = activationQueue
            .catch(function () {
                return null;
            })
            .then(async function () {
                if (sequence !== activationSequence) return;
                await activateFeature(feature, sequence);
            });
        void activationQueue.catch(function (error) {
            if (sequence !== activationSequence) return;
            activatingFeature = null;
            console.error(`Aikimi ${FEATURES[feature]?.label || feature} activation failed`, error);
            setActiveFeature(null, `${FEATURES[feature]?.label || feature}を開けませんでした。UIを再読み込みしてください。`);
        });
    }

    function selectedNativeButton() {
        return nativeTabButtons().find(function (button) {
            return button.classList.contains("selected") ||
                button.getAttribute("aria-selected") === "true";
        }) || null;
    }

    function nativeFeatureFor(button) {
        return FEATURE_ORDER.find(function (feature) {
            return FEATURES[feature].kind === "native" &&
                nativeButtonFor(FEATURES[feature].containerId) === button;
        }) || null;
    }

    function selectedButtonMatches(feature, selected) {
        const config = FEATURES[feature];
        return Boolean(config && featureContainerIds(config).some(function (containerId) {
            return nativeButtonFor(containerId) === selected;
        }));
    }

    function presetMatches(feature) {
        const preset = FEATURES[feature]?.preset;
        if (!preset) return true;
        return appRoot().querySelector("#forge_ui_preset input")?.value === preset;
    }

    function aliasStateMatches(feature) {
        if (feature === "krea2") {
            return presetMatches(feature) && selectedButtonMatches(feature, selectedNativeButton());
        }
        if (feature === "anima38") {
            const accordionId = animaAccordionId(preferredContainerId(feature));
            const accordion = appRoot().querySelector(`#${accordionId}`);
            const visibleCheckbox = appRoot().querySelector(`#${accordionId}-visible-checkbox`);
            const hiddenCheckbox = appRoot().querySelector(
                `#${accordionId}-checkbox input[type='checkbox']`,
            );
            return Boolean(
                presetMatches(feature) &&
                accordion?.querySelector(".label-wrap")?.classList.contains("open") &&
                visibleCheckbox?.checked &&
                hiddenCheckbox?.checked,
            );
        }
        return false;
    }

    function redispatchWhenContainerMounts(feature) {
        if (featureContainer(feature)) return;
        void waitFor(function () {
            return featureContainer(feature);
        }, 60).then(function (container) {
            if (container && activeFeature === feature) dispatchFeatureChange(feature, null);
        });
    }

    function syncFromSelectedNativeTab() {
        syncFeatureButtonTargets();
        const selected = selectedNativeButton();
        if (!selected) return;
        if (activatingFeature && selectedButtonMatches(activatingFeature, selected)) {
            syncFeatureButtonState(activatingFeature);
            return;
        }

        const nativeFeature = nativeFeatureFor(selected);
        if (nativeFeature) {
            if (activeFeature !== nativeFeature) setActiveFeature(nativeFeature, null);
            redispatchWhenContainerMounts(nativeFeature);
            return;
        }

        if (
            activeFeature &&
            FEATURES[activeFeature]?.kind === "alias" &&
            selectedButtonMatches(activeFeature, selected)
        ) {
            if (aliasStateMatches(activeFeature)) {
                syncFeatureButtonState(activeFeature);
                syncFeatureButtonTarget(activeFeature);
                if (featureContainer(activeFeature) !== dispatchedContainer) {
                    dispatchFeatureChange(activeFeature, null);
                }
            } else {
                setActiveFeature(null, null);
            }
            return;
        }
        if (activeFeature !== null) setActiveFeature(null, null);
    }

    function scheduleAliasReconciliation() {
        if (
            reconcileTimer !== null ||
            activatingFeature ||
            !activeFeature ||
            FEATURES[activeFeature]?.kind !== "alias"
        ) {
            return;
        }
        reconcileTimer = window.setTimeout(function () {
            reconcileTimer = null;
            syncFromSelectedNativeTab();
        }, 0);
    }

    function setup() {
        if (!ensureFeatureNavigation()) return false;
        syncFromSelectedNativeTab();
        return true;
    }

    function setupWithRetry(attempt = 0) {
        if (setup()) return;
        if (attempt >= SETUP_RETRY_LIMIT) return;
        window.setTimeout(function () {
            setupWithRetry(attempt + 1);
        }, SETUP_RETRY_DELAY_MS);
    }

    function scheduleSetup() {
        if (setupTimer !== null) return;
        setupTimer = window.setTimeout(function () {
            setupTimer = null;
            setup();
        }, 0);
    }

    function mutationContainsNode(nodes, predicate) {
        return Array.from(nodes || []).some(function (node) {
            return node.nodeType === Node.ELEMENT_NODE && predicate(node);
        });
    }

    function repairExternalNavigationOnUiUpdate(mutationRecords) {
        const tabs = topTabs();
        const row = featureNavigation();
        if (row && tabs && row.parentElement === tabs.parentElement && row.nextElementSibling === tabs) {
            return;
        }
        const relevant = Array.from(mutationRecords || []).some(function (record) {
            if (record.target === row || row?.contains(record.target)) return false;
            return mutationContainsNode(record.removedNodes, function (node) {
                return node.id === FEATURE_NAV_ID || node.id === "tabs" ||
                    Boolean(node.querySelector?.(`#${FEATURE_NAV_ID}, #tabs`));
            }) || mutationContainsNode(record.addedNodes, function (node) {
                return node.id === "tabs" || Boolean(node.querySelector?.("#tabs"));
            });
        });
        if (!relevant || repairCount >= REPAIR_LIMIT) return;
        repairCount += 1;
        scheduleSetup();
    }

    function handleUiUpdate(mutationRecords) {
        const tabs = topTabs();
        const nav = topTabNav();
        // Native studios may mount after the feature row. Ordinary progress and
        // prompt mutations do not require rebuilding availability.
        if (Array.from(mutationRecords || []).some(function (record) {
            return record.type === "childList" && (record.target === tabs || record.target === nav);
        })) {
            scheduleSetup();
        }
        repairExternalNavigationOnUiUpdate(mutationRecords);
        scheduleAliasReconciliation();
    }

    function topNavButtonFromTarget(target) {
        const button = target?.closest?.("button[role='tab']");
        return button?.parentElement === topTabNav() ? button : null;
    }

    function handleNativeTabClick(event) {
        const button = topNavButtonFromTarget(event.target);
        if (!button) return;
        if (!event.isTrusted && activatingFeature && selectedButtonMatches(activatingFeature, button)) return;
        if (
            !activatingFeature && activeFeature &&
            FEATURES[activeFeature]?.kind === "alias" &&
            selectedButtonMatches(activeFeature, button)
        ) {
            scheduleAliasReconciliation();
            return;
        }

        activationSequence += 1;
        activatingFeature = null;
        const nativeFeature = nativeFeatureFor(button);
        if (nativeFeature) {
            setActiveFeature(nativeFeature, null);
            redispatchWhenContainerMounts(nativeFeature);
        } else {
            setActiveFeature(null, null);
        }
    }

    function handleManualPresetChange(event) {
        if (!event.isTrusted || !activatingFeature || !event.target.closest?.("#forge_ui_preset")) return;
        activationSequence += 1;
        activatingFeature = null;
        setActiveFeature(null, null);
    }

    function handleFeatureNavigationClick(event) {
        const button = event.target.closest?.(".aikimi-feature-nav__button");
        if (!button || button.parentElement !== featureNavigation() || button.disabled) return;
        queueFeatureActivation(button.dataset.aikimiFeature);
    }

    function handleFeatureNavigationKeydown(event) {
        const button = event.target.closest?.(".aikimi-feature-nav__button");
        if (!button || button.parentElement !== featureNavigation() || event.defaultPrevented) return;
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            if (event.repeat) return;
            button.click();
            return;
        }
        if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;

        const buttons = Array.from(
            featureNavigation()?.querySelectorAll(":scope > .aikimi-feature-nav__button:not([hidden])") || [],
        );
        const index = buttons.indexOf(button);
        if (index < 0 || !buttons.length) return;
        let nextIndex = index;
        if (event.key === "ArrowLeft") nextIndex = (index - 1 + buttons.length) % buttons.length;
        if (event.key === "ArrowRight") nextIndex = (index + 1) % buttons.length;
        if (event.key === "Home") nextIndex = 0;
        if (event.key === "End") nextIndex = buttons.length - 1;
        event.preventDefault();
        buttons[nextIndex].focus();
    }

    window.AikimiTabs = Object.freeze({
        apiVersion: API_VERSION,
        getActiveFeature: function () {
            return activeFeature;
        },
        getActiveContainer: function () {
            return activeFeature ? featureContainer(activeFeature) : null;
        },
        refresh: function () {
            scheduleSetup();
        },
    });

    document.addEventListener("input", handleManualPresetChange);
    document.addEventListener("change", handleManualPresetChange);
    document.addEventListener("click", handleNativeTabClick);
    document.addEventListener("click", scheduleAliasReconciliation);
    document.addEventListener("input", scheduleAliasReconciliation);
    document.addEventListener("change", scheduleAliasReconciliation);
    onUiLoaded(function () {
        setupWithRetry();
    });
    onUiUpdate(handleUiUpdate);
    onUiTabChange(syncFromSelectedNativeTab);
})();
