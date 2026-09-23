import { app } from "../../../scripts/app.js";
import { importEditable, tokenFromHash } from "./handoff-core.js";

// Capture before the normal workflow service rewrites the URL fragment.
const token = tokenFromHash(window.location.hash);
let started = false;
let running = false;
let dismissed = false;
let panel, status, openButton;

function download(value, filename) {
    const blob = new Blob([JSON.stringify(value, null, 2)], {type: "application/json"});
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url; anchor.download = filename; anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
}
function addButton(label, callback) {
    const button = document.createElement("button");
    button.type = "button"; button.textContent = label;
    button.style.cssText = "padding:6px 10px;margin:4px;border:1px solid currentColor;border-radius:5px;cursor:pointer";
    button.addEventListener("click", callback); panel.appendChild(button);
    return button;
}
function showPanel() {
    if (panel || !token) return;
    panel = document.createElement("section");
    panel.setAttribute("aria-label", "Aikimi H3 ワークフロー引き継ぎ");
    panel.style.cssText = "position:fixed;right:14px;bottom:14px;z-index:10000;max-width:min(560px,calc(100vw - 28px));box-sizing:border-box;padding:14px;border:1px solid #888;border-radius:10px;background:var(--comfy-menu-bg,#202020);color:var(--fg-color,#eee);box-shadow:0 4px 24px #0006;font:14px/1.5 sans-serif";
    status = document.createElement("p");
    status.setAttribute("role", "status"); status.setAttribute("aria-live", "polite");
    status.style.cssText = "white-space:pre-wrap;margin:0 0 8px";
    status.textContent = "H3のワークフローを準備しています。生成は自動実行しません。";
    panel.appendChild(status);
    openButton = addButton("このH3ワークフローを読み込む", () => void openWorkflow());
    addButton("閉じる", () => { started = true; dismissed = true; panel.remove(); });
    document.body.appendChild(panel);
}
async function readJson(url) {
    const response = await fetch(url, {credentials: "same-origin", cache: "no-store"});
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `HTTP ${response.status}: H3引き継ぎパックの導入・再起動を確認してください。`);
    return data;
}
async function openWorkflow() {
    if (!token || running || dismissed) return;
    started = true; running = true; showPanel(); openButton.disabled = true;
    let record;
    try {
        status.textContent = "保存したSeed・素材・ノード接続を確認しています…";
        record = await readJson(`/aikimi/h3/workflows/${token}`);
        const schemas = await readJson("/object_info");
        const workflow = await importEditable(app, record, schemas);
        const meta = record.metadata;
        status.textContent = `H3を編集可能なノードとして開きました。\nSeed ${meta.seed} · ${meta.width}×${meta.height} · ${meta.frames} frames\n生成は未実行です。変更後はComfyUIの実行ボタンを使ってください。`;
        addButton("読み込み時の編集用JSONを保存", () => download(workflow, `H3-${token.slice(0, 8)}.workflow.json`));
    } catch (error) {
        status.textContent = `読み込みを中止しました。\n${error.message}`;
        if (record) addButton("元の実行グラフJSONを保存", () => download(record.prompt, `H3-${token.slice(0, 8)}.api.json`));
        if (error.previousWorkflow) addButton("退避した編集画面JSONを保存", () => download(error.previousWorkflow, "previous-workflow.json"));
    } finally {
        running = false; openButton.disabled = false;
    }
}

app.registerExtension({
    name: "Aikimi.H3.EditableWorkflowHandoff",
    setup() {
        if (token) showPanel();
    },
    afterLoadGraph() {
        // Initial graph restoration finishes first. Do not race ComfyUI startup
        // or recursively import when our own native import fires this hook.
        if (token && !started) {
            started = true;
            setTimeout(() => void openWorkflow(), 0);
        }
    },
});
