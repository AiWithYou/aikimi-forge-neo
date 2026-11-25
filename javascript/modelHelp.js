(function () {
    let link = null;

    function updateHelp() {
        const modules = document.getElementById("setting_sd_modules");
        const title = modules?.querySelector("span");
        if (title == null) return;

        if (link == null || !link.isConnected) {
            link = document.createElement("a");
            link.href = "https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/docs/local-models.md";
            link.title = "Where to get the modules?";
            link.textContent = "?";

            link.style.marginLeft = "0.5em";
            link.style.padding = "0em 0.5em";
            link.style.border = "var(--input-border-width) solid var(--input-border-color)";
            link.style.borderRadius = "var(--input-radius)";
            link.style.background = "var(--input-background-fill)";

            link.style.visibility = "hidden";
            title.appendChild(link);
        }

        const preset = document.getElementById("forge_ui_preset")?.querySelector("input");
        if (preset == null) return;
        if (preset.value === "xl") {
            link.style.visibility = "hidden";
            return;
        }

        const input = modules?.querySelector("div.wrap-inner");
        if (input == null) return;
        link.style.visibility = input.querySelector("div.token") ? "hidden" : "visible";
    }

    onUiLoaded(updateHelp);
    onAfterUiUpdate(updateHelp);
})();
