"""A compact multi-LoRA selector using Forge's existing extra-network pipeline."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

import gradio as gr

from modules import scripts, shared
from modules_forge import local_assets, model_catalog


def available(current=()):
    roots = [shared.cmd_opts.lora_dir, *shared.cmd_opts.lora_dirs]
    paths = model_catalog.discover_files(roots, suffixes=(".safetensors",))
    return model_catalog.choice_labels([*paths, *(current or ())], roots)


def _rows_update(names, rows):
    return gr.update(value=local_assets.lora_rows(names, rows), visible=True if names else "hidden")


def _save_strengths(names, rows):
    rows = local_assets.lora_rows(names, rows)
    local_assets.save_selection("forge_lora", {"loras": list(local_assets.lora_settings(names, rows))})


def _save_selection(names, rows):
    _save_strengths(names, rows)
    return _rows_update(names, rows)


def _load_selection():
    saved = local_assets.selection("forge_lora").get("loras", [])
    names = [item["name"] for item in saved]
    rows = [[item["name"], item["strength"]] for item in saved]
    return gr.update(choices=available(names), value=names), _rows_update(names, rows)


def _reset_names(_params):
    return gr.update(value=[])


def _reset_rows(_params):
    return gr.update(value=[], visible="hidden")


def apply_selections(processing, names, rows):
    import network
    import networks

    settings = local_assets.lora_settings(names, rows)
    prompts = processing.prompt if isinstance(processing.prompt, list) else [processing.prompt]
    prompts = [*prompts, *processing.all_prompts]
    prompt_tags = re.findall(r"<lora:\s*([^:>]+?)\s*:", "\n".join(prompts))
    anima_paths = set(getattr(processing, "_anima_standard_lora_paths", ()))
    active = []
    seen = set()
    for item in settings:
        if item["strength"] == 0:
            continue
        path = local_assets.local_path(item["name"], suffixes=(".safetensors",))
        if path in seen:
            raise ValueError("同じ実ファイルのLoRAを二重に選択できません。")
        seen.add(path)
        if path in anima_paths:
            raise ValueError(f"Anima欄とLoRA選択欄に同じファイルがあります: {path.name}")
        alias = next(
            (name for name, entry in networks.available_networks.items() if Path(entry.filename).resolve() == path),
            None,
        )
        if alias is None:
            alias = "local_" + hashlib.sha256(os.path.normcase(str(path)).encode()).hexdigest()[:16]
        for tag in prompt_tags:
            known = networks.available_networks.get(tag) or networks.available_network_aliases.get(tag)
            if known and Path(known.filename).resolve() == path:
                raise ValueError(f"プロンプトとLoRA選択欄に同じファイルがあります: {path.name}")
        local_assets.read_header(path)
        active.append((alias, path, item["strength"]))
    for alias, path, _strength in active:
        entry = network.NetworkOnDisk(alias, str(path))
        networks.available_networks[alias] = entry
        networks.available_network_aliases[alias] = entry
    for alias, _path, strength in active:
        tag = f" <lora:{alias}:{strength:.12g}>"
        processing.prompt = (
            [value + tag for value in processing.prompt]
            if isinstance(processing.prompt, list)
            else processing.prompt + tag
        )
        processing.all_prompts = [value + tag for value in processing.all_prompts]
    if active:
        processing._local_lora_paths = tuple(path for _, path, _ in active)
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
                available([x["name"] for x in saved]),
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
                visible=True if saved else "hidden",
                label="強度 −2〜2 · 0で無効",
            )
            rows.do_not_save_to_config = True
            refresh = gr.Button("一覧を更新", size="sm")
            names.change(
                _rows_update,
                inputs=[names, rows],
                outputs=rows,
                queue=False,
                show_progress="hidden",
            )
            section.expand(
                _rows_update,
                inputs=[names, rows],
                outputs=rows,
                queue=False,
                show_progress="hidden",
            )
            names.input(_save_selection, inputs=[names, rows], outputs=rows, queue=False, show_progress="hidden")
            rows.input(_save_strengths, inputs=[names, rows], queue=False, show_progress="hidden")
            refresh.click(
                lambda current: gr.update(choices=available(current), value=current),
                inputs=names,
                outputs=names,
                queue=False,
                show_progress="hidden",
            )
        gr.on(fn=_load_selection, outputs=[names, rows], queue=False, show_progress="hidden")
        self.infotext_fields = [
            (names, _reset_names),
            (rows, _reset_rows),
        ]
        self.paste_field_names = [_reset_names, _reset_rows]
        return [names, rows]

    def process(self, p, names, rows):
        apply_selections(p, names, rows)
