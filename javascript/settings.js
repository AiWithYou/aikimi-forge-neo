let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_about: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp()
        .querySelectorAll("#settings > div")
        .forEach(function (elem) {
            if (settingsExcludeTabsFromShowAll[elem.id]) return;

            elem.style.display = "block";
        });
}

function settingsShowOneTab() {
    gradioApp().querySelector("#settings_show_one_page")?.click();
}

let settingsSetupInput = null;
let settingsSetupTimer = null;
let settingsSetupAttempts = 0;
const SETTINGS_SETUP_MAX_ATTEMPTS = 40;

function setupSettingsSearch() {
    let edit = gradioApp().querySelector("#settings_search");
    let editTextarea = gradioApp().querySelector("#settings_search > label > input");
    let buttonShowAllPages = gradioApp().getElementById("settings_show_all_pages");
    if (!edit || !editTextarea || !buttonShowAllPages) return false;

    if (editTextarea.dataset.forgeSettingsSearchBound !== "true") {
        onEdit("settingsSearch", editTextarea, 500, function () {
            let searchText = (editTextarea.value || "").trim().toLowerCase();

            gradioApp()
                .querySelectorAll("#settings > div[id^=settings_] div[id^=column_settings_] > *")
                .forEach(function (elem) {
                    let visible = elem.textContent.trim().toLowerCase().indexOf(searchText) != -1;
                    elem.style.display = visible ? "" : "none";
                });

            if (searchText != "") {
                settingsShowAllTabs();
            } else {
                settingsShowOneTab();
            }
        });
        editTextarea.dataset.forgeSettingsSearchBound = "true";
    }

    if (buttonShowAllPages.dataset.forgeSettingsShowAllBound !== "true") {
        buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
        buttonShowAllPages.dataset.forgeSettingsShowAllBound = "true";
    }
    settingsSetupInput = editTextarea;
    return true;
}

function settingsSearchIsReady() {
    return Boolean(
        settingsSetupInput?.isConnected &&
        settingsSetupInput === gradioApp().querySelector("#settings_search > label > input") &&
        settingsSetupInput.dataset.forgeSettingsSearchBound === "true"
    );
}

function settingsMountIsPending() {
    return Boolean(
        gradioApp().querySelector("#settings") ||
        get_uiCurrentTab()?.getAttribute("aria-controls") === "tab_settings"
    );
}

function requestSettingsSetup() {
    if (settingsSearchIsReady()) return;
    settingsSetupAttempts = SETTINGS_SETUP_MAX_ATTEMPTS;
    if (settingsSetupTimer !== null) return;

    function retry() {
        settingsSetupTimer = null;
        if (settingsSearchIsReady() || setupSettingsSearch()) return;
        settingsSetupAttempts -= 1;
        if (settingsSetupAttempts <= 0 || !settingsMountIsPending()) return;
        settingsSetupTimer = window.setTimeout(retry, 50);
    }
    settingsSetupTimer = window.setTimeout(retry, 0);
}

function settingsMutationMayMountTarget(mutationRecords) {
    if (settingsSearchIsReady()) return false;
    return Array.from(mutationRecords || []).some(function (record) {
        return Array.from(record.addedNodes || []).some(function (node) {
            if (node.nodeType !== Node.ELEMENT_NODE) return false;
            return ["settings", "settings_search", "settings_show_all_pages"].includes(node.id) ||
                Boolean(node.querySelector?.("#settings, #settings_search, #settings_show_all_pages"));
        });
    });
}

onUiLoaded(requestSettingsSetup);
onUiTabChange(requestSettingsSetup);
onUiUpdate(function (mutationRecords) {
    if (settingsMutationMayMountTarget(mutationRecords)) requestSettingsSetup();
});

onOptionsChanged(function () {
    if (gradioApp().querySelector("#settings .settings-category")) return;

    let sectionMap = {};
    gradioApp()
        .querySelectorAll("#settings > div > button")
        .forEach(function (x) {
            sectionMap[x.textContent.trim()] = x;
        });

    opts._categories.forEach(function (x) {
        let section = localization[x[0]] ?? x[0];
        let category = localization[x[1]] ?? x[1];

        let span = document.createElement("SPAN");
        span.textContent = category;
        span.className = "settings-category";

        let sectionElem = sectionMap[section];
        if (!sectionElem) return;

        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
});
