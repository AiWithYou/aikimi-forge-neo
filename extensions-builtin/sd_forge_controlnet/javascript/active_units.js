(function () {
    const CONTROLNET_TABS = ["txt2img", "img2img"];
    const CONTROLNET_HOST_SELECTOR = "#txt2img_controlnet, #img2img_controlnet";
    const MAX_INITIALIZATION_ATTEMPTS = 100;
    const initializedByTab = new Map();
    let initializationTimer = null;
    let initializationAttempts = 0;

    class ControlNetAccordion {
        constructor(label) {
            const badge = document.createElement("span");
            badge.classList.add("cnet-badge");
            badge.style.visibility = "hidden";

            label.appendChild(badge);
            label.style.display = "flex";

            this.badge = badge;
            this.count = 0;
        }

        increase() {
            this.count += 1;
            this.badge.textContent = `${this.count}x Unit`;
            this.badge.style.visibility = "visible";
        }

        decrease() {
            this.count -= 1;
            this.badge.textContent = `${this.count}x Unit`;
            if (this.count === 0) this.badge.style.visibility = "hidden";
        }
    }

    class ControlNetUnitTab {
        constructor(cnet, elements, isImg2Img) {
            this.cnet = cnet;
            this.unitHeader = elements.unitHeader;
            this.enabledCheckbox = elements.enabledCheckbox;
            this.inputImage = elements.inputImage;
            this.controlTypeRadios = elements.controlTypeRadios;
            this.isImg2Img = isImg2Img;

            this.attachEnabledButtonListener();
            this.attachControlTypeRadioListener();
            this.attachImageUploadListener();
            this.attachA1111SendInfoObserver();
        }

        attachEnabledButtonListener() {
            this.enabledCheckbox.addEventListener("change", () => {
                this.updateActiveState();
            });
        }

        attachControlTypeRadioListener() {
            for (const radio of this.controlTypeRadios) {
                radio.addEventListener("change", () => {
                    this.updateActiveControlType();
                });
            }
        }

        attachImageUploadListener() {
            this.inputImage.addEventListener("change", (event) => {
                if (!event.target.files) return;
                if (!this.enabledCheckbox.checked) this.enabledCheckbox.click();
            });
        }

        attachA1111SendInfoObserver() {
            const pasteButtons = document.querySelectorAll("#paste");
            const pngButtons = document.querySelectorAll(this.isImg2Img ? "#img2img_tab, #inpaint_tab" : "#txt2img_tab");

            for (const button of [...pasteButtons, ...pngButtons]) {
                button.addEventListener("click", () => {
                    setTimeout(() => {
                        this.updateActiveState();
                    }, 2500);
                });
            }
        }

        updateActiveState() {
            if (this.enabledCheckbox.checked) {
                this.unitHeader.classList.add("cnet-unit-active");
                this.cnet.increase();
            }
            else {
                this.unitHeader.classList.remove("cnet-unit-active");
                this.cnet.decrease();
            }
        }

        updateActiveControlType() {
            const controlTypeSuffix = this.unitHeader.querySelector(".control-type-suffix");
            if (controlTypeSuffix) controlTypeSuffix.remove();

            const controlType = this.getActiveControlType();
            if (controlType === "All") return;

            const span = document.createElement("span");
            span.innerHTML = `[${controlType}]`;
            span.classList.add("control-type-suffix");
            this.unitHeader.appendChild(span);
        }

        getActiveControlType() {
            for (const radio of this.controlTypeRadios) if (radio.checked) return radio.value;
        }
    }

    function currentControlNet(tabName) {
        return document.getElementById(`${tabName}_controlnet`)?.querySelector("#controlnet") || null;
    }

    function refreshInitializedPanels() {
        for (const [tabName, ext] of initializedByTab) {
            if (!ext.isConnected || currentControlNet(tabName) !== ext) {
                initializedByTab.delete(tabName);
            }
        }
    }

    function unitElements(tab, index) {
        if (typeof get_uiTabButtons !== "function") return null;
        const unitHeader = get_uiTabButtons(tab.parentNode)[index];
        const enabledCheckbox = tab.querySelector(".cnet-unit-enabled input");
        const inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
        if (!unitHeader || !enabledCheckbox || !inputImage) return null;
        return {
            unitHeader,
            enabledCheckbox,
            inputImage,
            controlTypeRadios: tab.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]'),
        };
    }

    function initializeControlNet(tabName) {
        const ext = currentControlNet(tabName);
        if (!ext) return false;
        if (initializedByTab.get(tabName) === ext) return true;

        const label = ext.querySelector("button.label-wrap span");
        const tabs = Array.from(ext.querySelectorAll(".tabitem"));
        if (!label || tabs.length === 0) return false;

        const units = tabs.map(unitElements);
        if (units.some((elements) => elements === null)) return false;

        const cnet = new ControlNetAccordion(label);
        for (const elements of units) {
            new ControlNetUnitTab(cnet, elements, tabName === "img2img");
        }
        initializedByTab.set(tabName, ext);
        return true;
    }

    function allControlNetsInitialized() {
        refreshInitializedPanels();
        return CONTROLNET_TABS.every((tabName) => initializedByTab.has(tabName));
    }

    function initializeAvailableControlNets() {
        initializationTimer = null;
        initializationAttempts += 1;
        for (const tabName of CONTROLNET_TABS) {
            initializeControlNet(tabName);
        }
    }

    function scheduleInitialization(resetAttempts = false) {
        if (allControlNetsInitialized()) return;
        if (resetAttempts) initializationAttempts = 0;
        if (initializationTimer !== null || initializationAttempts >= MAX_INITIALIZATION_ATTEMPTS) return;
        initializationTimer = window.setTimeout(initializeAvailableControlNets, 0);
    }

    function mutationTouchesControlNet(records) {
        return Array.from(records || []).some((record) => {
            if (record.target?.closest?.(CONTROLNET_HOST_SELECTOR)) return true;
            return Array.from(record.addedNodes || []).some((node) => {
                if (node.nodeType !== Node.ELEMENT_NODE) return false;
                return Boolean(
                    node.matches?.(CONTROLNET_HOST_SELECTOR) ||
                    node.closest?.(CONTROLNET_HOST_SELECTOR) ||
                    node.querySelector?.(CONTROLNET_HOST_SELECTOR)
                );
            });
        });
    }

    onUiLoaded(() => {
        scheduleInitialization(true);
    });

    onUiUpdate((records) => {
        if (!allControlNetsInitialized() && mutationTouchesControlNet(records)) {
            scheduleInitialization();
        }
    });

    onUiTabChange(() => {
        if (!allControlNetsInitialized()) scheduleInitialization(true);
    });
})();
