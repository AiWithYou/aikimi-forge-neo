"""A compact multi-LoRA selector using Forge's existing extra-network pipeline."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import gradio as gr

from modules import scripts, shared
from modules_forge import local_assets


def available():
    roots = [shared.cmd_opts.lora_dir, *shared.cmd_opts.lora_dirs]
    return local_assets.choices("forge_lora", roots, suffixes=(".safetensors",))


def apply_selections(processing, names, rows):
    import network
    import networks

    settings = local_assets.lora_settings(names, rows)
    prompts = processing.prompt if isinstance(processing.prompt, list) else [processing.prompt]
    active = []
    seen = set()
    for item in settings:
        if item["strength"] == 0:
            continue
        path = local_assets.local_path(item["name"], suffixes=(".safetensors",))
        if path in seen:
            raise ValueError("同じ実ファイルのLoRAを二重に選択できません。")
        seen.add(path)
        local_assets.read_header(path)
        alias = next(
            (name for name, entry in networks.available_networks.items() if Path(entry.filename).resolve() == path),
            None,
        )
        if alias is None:
            alias = "local_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        for tag in re.findall(r"<lora:([^:>]+):", "\n".join(prompts)):
            known = networks.available_network_aliases.get(tag)
            if known and Path(known.filename).resolve() == path:
                raise ValueError(f"プロンプトとLoRA選択欄に同じファイルがあります: {path.name}")
        entry = network.NetworkOnDisk(alias, str(path))
        networks.available_networks[alias] = entry
        networks.available_network_aliases[alias] = entry
        active.append((alias, path, item["strength"]))
    for alias, _path, strength in active:
        tag = f" <lora:{alias}:{strength:.12g}>"
        processing.prompt = (
            [value + tag for value in processing.prompt]
            if isinstance(processing.prompt, list)
            else processing.prompt + tag
        )
        processing.all_prompts = [value + tag for value in processing.all_prompts]
    local_assets.remember("forge_lora", [str(path) for _, path, _ in active])
    local_assets.save_selection("forge_lora", {"loras": list(settings)})
    if active:
        processing.extra_generation_params["Local LoRA stack"] = ", ".join(
            f"{path.name}:{strength:.12g}" for _, path, strength in active
        )


class Script(scripts.Script):
    def title(self):
        return "Local LoRA stack"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        saved = local_assets.selection("forge_lora").get("loras", [])
        with gr.Accordion("LoRAを組み合わせる", open=False) as section:
            names = gr.Dropdown(
                available(),
                value=[x["name"] for x in saved],
                multiselect=True,
                allow_custom_value=True,
                label="LoRA（複数選択・フルパスも入力可）",
            )
            names.do_not_save_to_config = True  # The asset library owns saved selection and strengths.
            rows = gr.Dataframe(
                value=local_assets.lora_rows([x["name"] for x in saved], [[x["name"], x["strength"]] for x in saved]),
                headers=["LoRA", "強度"],
                datatype=["str", "number"],
                static_columns=[0],
                type="array",
                column_count=(2, "fixed"),
                row_count=len(saved),
                max_chars=64,
                interactive=True,
                visible=False,
                label="強度 −2〜2 · 0で無効",
            )
            refresh = gr.Button("一覧を更新", size="sm")
            names.change(
                lambda ns, rs: gr.update(value=local_assets.lora_rows(ns, rs), visible=bool(ns)),
                inputs=[names, rows],
                outputs=rows,
                queue=False,
            )
            section.expand(
                lambda ns, rs: gr.update(value=local_assets.lora_rows(ns, rs), visible=bool(ns)),
                inputs=[names, rows],
                outputs=rows,
                queue=False,
            )
            refresh.click(
                lambda current: gr.update(choices=available(), value=current), inputs=names, outputs=names, queue=False
            )
        return [names, rows]

    def process(self, p, names, rows):
        apply_selections(p, names, rows)
