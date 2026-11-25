function sRound(value, step) {
    return Number.isFinite(step) && step > 0 ? Math.round(value / step) * step : Number(value);
}

onOptionsAvailable(() => {
    // Delegate so inputs mounted on tab changes receive the same rounding.
    document.addEventListener("focusout", (event) => {
        const input = event.target;
        if (!input.matches?.("input:not([type=range])") ||
            !input.closest("#txt2img_width, #txt2img_height, #img2img_width, #img2img_height")) return;
        if (input.value.trim() === "" || !Number.isFinite(Number(input.value))) return;
        const step = Number(opts.res_step);
        const min = input.min === "" ? step : Number(input.min);
        const max = input.max === "" ? Infinity : Number(input.max);
        const value = Math.min(max, Math.max(min, sRound(Number(input.value), step)));
        if (Number.isFinite(value) && value !== Number(input.value)) {
            input.value = value;
            updateInput(input);
        }
    });
});
