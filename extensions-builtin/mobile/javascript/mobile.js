(function () {
    function reportWindowSize() {
        const app = gradioApp();
        // Compactレイアウトは専用の配置を維持する。
        if (app.querySelector(".toprow-compact-tools")) return;

        for (const tab of ["txt2img", "img2img"]) {
            const results = app.getElementById(tab + "_results");
            // 未描画・非表示のタブは、描画／選択後のコールバックで配置する。
            if (!results || !results.offsetParent) continue;
            const currentlyMobile = results.offsetLeft === 0;
            const button = app.getElementById(tab + "_generate_box");
            const target = currentlyMobile ? results : app.getElementById(tab + "_actions_column");
            if (!button || !target) continue;

            // 不要なDOM更新を避けつつ、部品の再描画にも対応する。
            if (button.parentElement !== target || target.firstElementChild !== button) {
                target.insertBefore(button, target.firstElementChild);
            }
            results.classList.toggle("mobile", currentlyMobile);
        }
    }

    window.addEventListener("resize", reportWindowSize);

    onUiLoaded(reportWindowSize);
    onUiTabChange(reportWindowSize);
    onAfterUiUpdate(reportWindowSize);
})();
