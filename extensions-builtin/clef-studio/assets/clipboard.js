// Mount once on this accordion; forward image files to Gradio's normal uploader.
const scope = element.closest('#clef-individual-images');
if (!scope || scope.dataset.clefPasteBound) return;
scope.dataset.clefPasteBound = 'true';
scope.tabIndex = 0;
const hint = element.querySelector('.clef-paste-hint');
const notice = 'この欄で Ctrl+V（⌘V）を押すと、コピーした画像を追加できます。';
const editable = 'input:not([type="file"]), textarea, select, [contenteditable="true"], [contenteditable=""], [role="textbox"]';
const suffixes = { 'image/png': 'png', 'image/jpeg': 'jpg', 'image/webp': 'webp', 'image/bmp': 'bmp', 'image/x-ms-bmp': 'bmp' };

scope.addEventListener('pointerdown', event => {
    if (!event.target.closest('input, textarea, select, button, a, [contenteditable]')) {
        scope.focus({ preventScroll: true });
    }
});
scope.addEventListener('paste', event => {
    if (event.defaultPrevented || event.target.closest(editable)) return;
    const area = scope.querySelector('#clef-images');
    const input = scope.querySelector('#clef-images input[type="file"]');
    if (!area?.getClientRects().length || !input || input.disabled) return;

    const transfer = new DataTransfer();
    let rejected = '';
    for (const file of Array.from(event.clipboardData?.files || [])) {
        const suffix = suffixes[file.type];
        const supportedName = /\.(png|jpe?g|webp|bmp)$/i.test(file.name);
        if (!suffix && !supportedName) {
            if (file.type.startsWith('image/')) rejected = 'PNG/JPEG/WebP/BMPの画像を貼り付けてください。';
            continue;
        }
        if (file.size > 32 * 1024 ** 2) {
            rejected = '画像は1ファイル32MiBまでです。';
            continue;
        }
        transfer.items.add(supportedName ? file : new File([file], `clipboard.${suffix}`, { type: file.type }));
    }
    if (hint && (transfer.files.length || rejected)) hint.textContent = rejected || notice;
    if (!transfer.files.length) return;
    input.files = transfer.files;
    event.preventDefault();
    event.stopPropagation();
    input.dispatchEvent(new Event('change', { bubbles: true }));
    input.value = '';
});
