/* No HTTP writes and no generation calls. Kept separate for Node contract tests. */
export function tokenFromHash(hash) {
    const match = /^#aikimi-h3=([0-9a-f]{32})$/.exec(hash || "");
    return match?.[1] ?? null;
}

const clone = (value) => JSON.parse(JSON.stringify(value));
const own = (obj, key) => Object.prototype.hasOwnProperty.call(obj, key);
function equal(a, b) {
    if (Object.is(a, b)) return true;
    if (!a || !b || typeof a !== "object" || typeof b !== "object") return false;
    if (Array.isArray(a) !== Array.isArray(b)) return false;
    const keys = Object.keys(a);
    return keys.length === Object.keys(b).length && keys.every((k) => own(b, k) && equal(a[k], b[k]));
}
function inputDefault(definition) {
    if (!Array.isArray(definition)) return {known: false};
    const options = definition[1] || {};
    if (own(options, "default")) return {known: true, value: options.default};
    const choices = Array.isArray(definition[0]) ? definition[0] : definition[0] === "COMBO" ? options.options : undefined;
    if (choices?.length) return {known: true, value: choices[0]};
    return {known: false};
}

export function roundTripDifferences(expected, actual, schemas = {}) {
    const issues = [];
    for (const id of Object.keys(expected)) {
        const original = expected[id], imported = actual?.[id];
        if (!imported || imported.class_type !== original.class_type) {
            issues.push(`${id}: ノードが欠落・変更されています`);
            continue;
        }
        for (const [name, value] of Object.entries(original.inputs)) {
            if (!own(imported.inputs || {}, name) || !equal(value, imported.inputs[name])) {
                issues.push(`${id}.${name}: 値または接続が変わりました`);
            }
        }
        // Native import can add widgets that the API prompt had left at defaults.
        // Only documented equal defaults are allowed; no guessing for dynamic inputs.
        const input = schemas[original.class_type]?.input || {};
        const definitions = {...input.required, ...input.optional};
        for (const [name, value] of Object.entries(imported.inputs || {})) {
            if (own(original.inputs, name)) continue;
            // New ComfyUI flattens SaveVideo's nested codec while retaining the
            // legacy optional codec field. Accept only an identical duplicate.
            if (original.class_type === "SaveVideo" && name === "format.codec" &&
                definitions.format?.[0] === "COMFY_DYNAMICCOMBO_V3" &&
                definitions.codec?.[0] === "COMFY_DYNAMICCOMBO_V3" &&
                own(original.inputs, "codec") && equal(original.inputs.codec, value)) continue;
            const fallback = inputDefault(definitions[name]);
            if (!fallback.known || !equal(fallback.value, value)) issues.push(`${id}.${name}: 未指定の入力が追加されました`);
        }
    }
    for (const id of Object.keys(actual || {})) {
        if (!own(expected, id)) issues.push(`${id}: 未指定のノードが追加されました`);
    }
    return issues;
}

export function validateAvailableNodes(prompt, schemas) {
    const missing = [...new Set(Object.values(prompt).map((n) => n.class_type))].filter((name) => !schemas[name]);
    if (missing.length) throw new Error(`必要なノードがありません: ${missing.join(", ")}`);
    const modelFields = {UNETLoader: "unet_name", CLIPLoader: "clip_name", VAELoader: "vae_name", ModelPatchLoader: "name"};
    for (const node of Object.values(prompt)) {
        const name = modelFields[node.class_type];
        if (!name || !own(node.inputs, name)) continue;
        const spec = schemas[node.class_type].input?.required?.[name];
        const choices = Array.isArray(spec?.[0]) ? spec[0] : spec?.[0] === "COMBO" ? spec[1]?.options : null;
        if (!choices || !choices.includes(node.inputs[name])) throw new Error(`モデルを確認してください: ${node.inputs[name]}`);
    }
}

export async function importEditable(app, record, schemas) {
    if (record?.format !== "aikimi-h3-handoff-v1" || !tokenFromHash(`#aikimi-h3=${record.token}`)) throw new Error("H3ワークフローの形式が不正です。");
    const prompt = record.prompt;
    if (!prompt || typeof prompt !== "object" || !Object.keys(prompt).length) throw new Error("実行グラフが空です。");
    if (!Number.isSafeInteger(record.metadata?.seed) || record.metadata.seed < 0) throw new Error("確定Seedを正しく扱えません。");
    if (typeof app.loadApiJson !== "function" || typeof app.graphToPrompt !== "function" || typeof app.loadGraphData !== "function") throw new Error("対応するComfyUIフロントエンドへ更新してください。");
    validateAvailableNodes(prompt, schemas);
    const graph = app.rootGraph || app.graph;
    if (!graph?.serialize) throw new Error("ComfyUIの編集画面がまだ準備できていません。");
    const previous = clone(graph.serialize());
    try {
        // This is ComfyUI's native editor import, NOT sending an API prompt to /prompt.
        await app.loadApiJson(clone(prompt), `Aikimi H3 ${record.token.slice(0, 8)}.json`);
        const importedGraph = app.rootGraph || app.graph;
        const nodes = importedGraph.nodes || importedGraph._nodes || [];
        for (const node of nodes) {
            if (node.has_errors) throw new Error(`読み込めないノードがあります: ${node.type || node.id}`);
            for (const widget of node.widgets || []) {
                if (/(^|_)control_(before|after)_generate$/.test(widget.name)) widget.value = "fixed";
            }
        }
        const roundTrip = await app.graphToPrompt();
        const differences = roundTripDifferences(prompt, roundTrip.output, schemas);
        if (differences.length) throw new Error(`同じ設定として開けません。\n${differences.slice(0, 12).join("\n")}`);
        if (!Array.isArray(roundTrip.workflow?.nodes)) throw new Error("編集可能なUIワークフローを取得できません。");
        const workflow = clone(roundTrip.workflow);
        workflow.extra = {...workflow.extra, aikimi_h3: {
            format: record.format, token: record.token, metadata: clone(record.metadata),
            prompt_sha256: record.prompt_sha256, submitted_prompt_sha256: record.submitted_prompt_sha256,
        }};
        return workflow;
    } catch (error) {
        // Preserve the user's previous canvas on loss, including missing dynamic inputs.
        try {
            await app.loadGraphData(previous, true, true);
        } catch (restoreError) {
            // Do not leave an incomplete imported graph accidentally executable.
            for (const node of (app.rootGraph || app.graph)?.nodes || (app.rootGraph || app.graph)?._nodes || []) node.mode = 2; // LiteGraph.NEVER
            const failure = new Error(`${error.message}\n前の画面の復元にも失敗しました。退避JSONを保存してください。`);
            failure.previousWorkflow = previous;
            failure.cause = restoreError;
            throw failure;
        }
        throw error;
    }
}
