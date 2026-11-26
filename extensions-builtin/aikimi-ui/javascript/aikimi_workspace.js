(function () {
    "use strict";

    const studios = {forge: "Forge", qwen_image21: "Qwen Image 2.1", minimax_h3_image: "H3 Image",
        minimax_h3: "MiniMax H3", flux3: "FLUX 3", moss_image: "MOSS Image", irodori: "Irodori TTS", yue2: "YuE2 Music"};
    const endpoint = (name) => new URL("./aikimi/api/v1/" + name, window.location.href);
    function element(tag, text) {
        const node = document.createElement(tag);
        if (text !== undefined) node.textContent = text;
        return node;
    }
    async function api(name, method = "GET", data) {
        const response = await fetch(endpoint(name), {method, credentials: "same-origin",
            headers: method === "GET" ? {} : {"Content-Type": "application/json"}, body: JSON.stringify(data)});
        if (!response.ok) throw new Error("操作に失敗しました (HTTP " + response.status + ")");
        return response.json();
    }
    function setInput(input, value) {
        if (!input) throw new Error("このスタジオの入力欄を開いてください。");
        input.value = value;
        input.dispatchEvent(new Event("input", {bubbles: true}));
    }
    function field(label, tag = "input") {
        const wrapper = element("label", label);
        const input = element(tag);
        wrapper.append(input);
        return [wrapper, input];
    }
    function select(label, choices) {
        const [wrapper, input] = field(label, "select");
        for (const [value, caption] of Object.entries(choices)) input.add(new Option(caption, value));
        return [wrapper, input];
    }
    function action(label, callback, feedback) {
        const button = element("button", label);
        button.type = "button";
        button.addEventListener("click", async () => {
            button.disabled = true;
            try { await callback(); }
            catch (error) { feedback.textContent = error.message; }
            finally { button.disabled = false; }
        });
        return button;
    }
    function visible(selector, parent = document) {
        return [...parent.querySelectorAll(selector)].find(node => node.getClientRects().length && !node.readOnly);
    }
    const promptFields = {
        forge: ["#txt2img_prompt textarea,#img2img_prompt textarea", "#txt2img_neg_prompt textarea,#img2img_neg_prompt textarea"],
        moss_image: ["#moss-image-prompt textarea", "#moss-image-negative-prompt textarea"],
        qwen_image21: ["#qwen21-prompt textarea,#qwen21-outpaint-prompt textarea"],
        minimax_h3_image: ["#ph3-prompt textarea,#official-h3-prompt textarea"],
        minimax_h3: ["#h3-prompt textarea"],
        flux3: ["#flux3-prompt textarea"],
        irodori: ["#irodori-text textarea"]
    };
    function insert(prompt, negative = "") {
        const parent = window.AikimiTabs?.getActiveContainer() || document;
        const [positiveSelector, negativeSelector] = promptFields[window.AikimiTabs?.getActiveFeature()] || promptFields.forge;
        const input = visible(positiveSelector, parent)
            || [...parent.querySelectorAll("label")].filter(label => /^(プロンプト|テキスト|歌詞)/.test(label.textContent.trim()))
                .map(label => label.querySelector("textarea")).find(node => node?.getClientRects().length && !node.readOnly);
        setInput(input, prompt);
        const neg = negativeSelector && visible(negativeSelector, parent);
        if (neg) setInput(neg, negative);
    }
    function createDrawer() {
        const drawer = element("dialog");
        drawer.id = "aikimi-prompt-library";
        drawer.setAttribute("aria-label", "プロンプト集");
        const message = element("output");
        const [pickerLabel, picker] = select("テンプレート", {});
        const [titleLabel, title] = field("タイトル");
        const [promptLabel, prompt] = field("プロンプト", "textarea");
        const [negLabel, negative] = field("ネガティブプロンプト", "textarea");
        const [tagsLabel, tags] = field("タグ（カンマ区切り）");
        const [studioLabel, studio] = select("スタジオ", {"": "共通", ...studios});
        const [jsonLabel, fragment] = field("JSON", "textarea");
        const [seedLabel, seed] = field("Seed");
        seed.type = "number";
        let entries = [];
        function loadEntry() {
            const entry = entries.find(item => item.id === picker.value) || {};
            title.value = entry.title || "";
            prompt.value = entry.prompt || "";
            negative.value = entry.negative_prompt || "";
            tags.value = (entry.tags || []).join(", ");
            studio.value = entry.studio || "";
            fragment.value = JSON.stringify(entry.json_fragment || {});
        }
        async function refresh(id = "") {
            entries = await api("prompt-library");
            picker.replaceChildren(new Option("新規", ""), ...entries.map(e => new Option(e.title, e.id)));
            picker.value = id;
            loadEntry();
        }
        picker.onchange = loadEntry;
        const buttons = element("div");
        buttons.className = "aikimi-workspace-actions";
        buttons.append(action("保存", async () => {
            const entry = await api("prompt-library", "POST", {id: picker.value || undefined, title: title.value,
                prompt: prompt.value, negative_prompt: negative.value, tags: tags.value.split(",").map(v => v.trim()).filter(Boolean),
                studio: studio.value, json_fragment: JSON.parse(fragment.value || "{}")});
            await refresh(entry.id);
            message.textContent = "保存しました。";
        }, message), action("削除", async () => {
            if (picker.value && window.confirm("このテンプレートを削除しますか？")) {
                await api("prompt-library/" + encodeURIComponent(picker.value), "DELETE");
                await refresh();
            }
        }, message), action("展開", async () => {
            if (seed.value && !Number.isSafeInteger(Number(seed.value))) throw new Error("Seed は正確に表現できる整数を入力してください。");
            const result = await api("prompt-library/expand", "POST", {text: prompt.value, seed: seed.value === "" ? null : Number(seed.value)});
            prompt.value = result.prompt;
            seed.value = result.seed;
        }, message), action("スタジオへ挿入", () => { insert(prompt.value, negative.value); drawer.close(); }, message),
        action("閉じる", () => drawer.close(), message));
        drawer.append(pickerLabel, titleLabel, promptLabel, negLabel, tagsLabel, studioLabel, jsonLabel, seedLabel, buttons, message);
        document.body.append(drawer);
        return async () => {
            drawer.showModal();
            message.textContent = "";
            try { await refresh(); } catch (error) { message.textContent = error.message; }
        };
    }
    let openDrawer;
    async function waitFor(selector) {
        for (let attempt = 0; attempt < 50; attempt++) {
            const input = document.querySelector(selector);
            if (input) return input;
            await new Promise(resolve => setTimeout(resolve, 60));
        }
        throw new Error("送信先を開けませんでした。");
    }
    async function canvasPng(file) {
        const bitmap = await createImageBitmap(file);
        try {
            if (bitmap.width * bitmap.height > 40000000) throw new Error("Canvas の画像サイズ上限を超えています。");
            const canvas = document.createElement("canvas");
            canvas.width = bitmap.width;
            canvas.height = bitmap.height;
            canvas.getContext("2d").drawImage(bitmap, 0, 0);
            return canvas.toDataURL("image/png");
        } finally { bitmap.close(); }
    }
    async function sendTo(item, destination, url) {
        let file;
        if (destination !== "txt2img" && item.media === "image") {
            const response = await fetch(url);
            if (!response.ok) throw new Error("画像を読み込めません。");
            const blob = await response.blob();
            file = new File([blob], item.id.split("/").pop(), {type: blob.type});
        }
        const background = file && destination === "img2img" ? await canvasPng(file) : null;
        const tab = get_uiTopTabButton("tab_" + destination);
        if (!tab) throw new Error("送信先を開けませんでした。");
        tab.click();
        // Gradio mounts the destination asynchronously. Select the image mode
        // only after its tab list exists, and avoid reselecting an active tab.
        await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
        if (destination !== "txt2img") {
            const buttons = Array.from(get_uiTabButtons(await waitFor("#mode_" + destination)));
            const selected = buttons.find(button => button.getAttribute("aria-selected") === "true");
            if (selected && selected !== buttons[0]) buttons[0].click();
        }
        if (destination !== "extras") {
            setInput(await waitFor("#" + destination + "_prompt textarea"), item.metadata.prompt || "");
            setInput(await waitFor("#" + destination + "_neg_prompt textarea"), item.metadata.negative_prompt || "");
        }
        if (background) {
            const container = await waitFor("#img2img_image .forge-container");
            const id = container.id.slice("container_".length);
            // LogicalImage stores the source even when ForgeCanvas mounts later.
            setInput(await waitFor("#" + CSS.escape(id) + ".logical_image_background textarea"), background);
        } else if (file) {
            const input = await waitFor("#extras_image input[type=file]");
            const transfer = new DataTransfer();
            transfer.items.add(file);
            input.files = transfer.files;
            input.dispatchEvent(new Event("change", {bubbles: true}));
        }
    }
    function mountGallery(root) {
        root.dataset.ready = "true";
        const [studioLabel, studio] = select("スタジオ", {"": "すべて", ...studios});
        const [mediaLabel, media] = select("種類", {"": "すべて", image: "画像", video: "動画", audio: "音声"});
        const [sortLabel, sort] = select("並び順", {newest: "新しい順", oldest: "古い順"});
        const [queryLabel, query] = field("キーワード");
        const [favLabel, favorite] = field("お気に入りのみ");
        favorite.type = "checkbox";
        const message = element("output");
        const list = element("div");
        list.className = "aikimi-gallery-results";
        const detail = element("div");
        detail.className = "aikimi-gallery-detail";
        let offset = 0, generation = 0;
        function show(item) {
            const name = item.id.split("/").pop();
            const url = endpoint("gallery/media?id=" + encodeURIComponent(item.id));
            const preview = element({image: "img", video: "video", audio: "audio"}[item.media]);
            preview.src = url;
            if (item.media === "image") preview.alt = name;
            else preview.controls = true;
            const meta = element("pre", JSON.stringify(item.metadata, null, 2));
            const buttons = element("div");
            buttons.className = "aikimi-workspace-actions";
            buttons.append(action(item.favorite ? "★ 解除" : "☆ お気に入り", async () => {
                await api("gallery/favorite", "POST", {id: item.id, enabled: !item.favorite});
                await refresh();
            }, message), action("隔離", async () => {
                if (window.confirm(name + " を隔離しますか？")) {
                    await api("gallery/quarantine", "POST", {id: item.id});
                    await refresh();
                }
            }, message));
            if (item.metadata.prompt) {
                buttons.append(action("プロンプトをコピー", () => navigator.clipboard.writeText(item.metadata.prompt), message),
                    action("プロンプト集へ保存", async () => {
                        await api("prompt-library", "POST", {title: name.slice(0, 160), prompt: item.metadata.prompt,
                            negative_prompt: item.metadata.negative_prompt || "", studio: item.studio});
                        message.textContent = "プロンプト集に保存しました。";
                    }, message));
            }
            if (item.metadata.prompt && item.studio !== "forge") {
                const target = {moss_image: "moss-image", flux3: "flux3", minimax_h3: "minimax-h3",
                    minimax_h3_image: "h3-image", qwen_image21: "qwen-image21", irodori: "irodori", yue2: "yue2"}[item.studio];
                buttons.append(action(studios[item.studio] + " へ送る", async () => {
                    (await waitFor("#aikimi-tab-" + target)).click();
                    for (let attempt = 0; attempt < 80; attempt++) {
                        if (window.AikimiTabs?.getActiveFeature() === item.studio) {
                            insert(item.metadata.prompt, item.metadata.negative_prompt);
                            return;
                        }
                        await new Promise(resolve => setTimeout(resolve, 50));
                    }
                    throw new Error("送信先を開けませんでした。");
                }, message));
            }
            for (const target of ["txt2img", "img2img", "extras"]) {
                if (target === "extras" ? item.media === "image" : item.metadata.prompt || (target === "img2img" && item.media === "image")) {
                    buttons.append(action(target + " へ送る", () => sendTo(item, target, url), message));
                }
            }
            detail.replaceChildren(element("h3", name), preview, buttons, meta);
        }
        async function refresh() {
            const version = ++generation;
            const params = new URLSearchParams({studio: studio.value, media: media.value, query: query.value,
                favorites: favorite.checked, order: sort.value, offset, limit: 24});
            const result = await api("gallery?" + params);
            if (version !== generation) return;
            if (offset && !result.items.length) { offset = Math.max(0, offset - 24); return refresh(); }
            list.replaceChildren();
            detail.replaceChildren();
            message.textContent = result.total + " 件" + (result.truncated ? "（走査上限に到達）" : "");
            for (const item of result.items) {
                const button = action(item.id.split("/").pop(), () => show(item), message);
                if (item.media === "image") {
                    const img = element("img");
                    img.alt = "";
                    img.loading = "lazy";
                    img.src = endpoint("gallery/media?id=" + encodeURIComponent(item.id));
                    button.prepend(img);
                }
                list.append(button);
            }
            prev.disabled = offset <= 0;
            next.disabled = offset + 24 >= result.total;
        }
        const prev = element("button", "前へ"), next = element("button", "次へ");
        const reload = () => refresh().catch(error => { message.textContent = error.message; });
        prev.onclick = () => { offset -= 24; reload(); };
        next.onclick = () => { offset += 24; reload(); };
        for (const input of [studio, media, sort, favorite, query]) input.onchange = () => { offset = 0; reload(); };
        const filters = element("div");
        filters.className = "aikimi-workspace-actions";
        filters.append(studioLabel, mediaLabel, sortLabel, queryLabel, favLabel, action("検索・更新", reload, message));
        root.append(filters, message, detail, list, prev, next);
        reload();
    }
    function mount() {
        const nav = document.getElementById("aikimi-feature-nav");
        if (nav && !document.getElementById("aikimi-library-open")) {
            openDrawer ||= createDrawer();
            const button = element("button", "プロンプト集");
            button.id = "aikimi-library-open";
            button.onclick = openDrawer;
            nav.before(button);
        }
        const gallery = document.getElementById("aikimi-gallery-workspace");
        if (gallery && !gallery.dataset.ready && window.AikimiTabs?.getActiveFeature() === "gallery") mountGallery(gallery);
    }
    document.addEventListener("keydown", event => {
        if (event.altKey && event.code === "KeyP" && openDrawer && !event.repeat) {
            event.preventDefault();
            openDrawer();
        }
    });
    onUiLoaded(mount);
    onUiUpdate(mount);
    document.addEventListener("aikimi:feature-tab-change", () => requestAnimationFrame(mount));
    window.AikimiWorkspace = {insert, setInput};
})();
