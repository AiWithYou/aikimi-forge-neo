"""Visible Studio selectors backed only by the local asset library."""

from modules_forge import local_assets, model_catalog


def inventory(family, roots, validator, *, current="", folders=False):
    sources = model_catalog.checkpoint_roots(family, roots)
    paths = model_catalog.discover_files(sources, suffixes=(".safetensors",), folders=folders, validator=validator)
    return [("標準モデル", ""), *model_catalog.choice_labels(paths, sources, current=current)]


def save(engine, checkpoint):
    local_assets.save_selection(engine, {"checkpoint": checkpoint or ""})


def create(engine, family, roots, validator, *, gradio, folders=False, details=""):
    gr = gradio
    value = local_assets.selection(engine)
    current = value.get("checkpoint", "")
    with gr.Row():
        checkpoint = model_catalog.configure_component(
            gr.Dropdown(
                choices=inventory(family, roots, validator, current=current, folders=folders),
                value=current,
                label="チェックポイント",
                scale=5,
                interactive=True,
            )
        )
        refresh = gr.Button("一覧を更新", scale=0, min_width=105)
    if details:
        with gr.Accordion("モデル詳細", open=False):
            gr.Markdown(details)
    private = {"api_visibility": "private", "show_progress": "hidden", "queue": False}

    def reload(selected):
        save(engine, selected)
        return gr.update(
            choices=inventory(family, roots, validator, current=selected or "", folders=folders),
            value=selected or "",
        )

    refresh.click(reload, inputs=checkpoint, outputs=checkpoint, **private)
    checkpoint.input(lambda selected: save(engine, selected), inputs=checkpoint, **private)

    def restore():
        latest = local_assets.selection(engine)
        selected = latest.get("checkpoint", "")
        return gr.update(choices=inventory(family, roots, validator, current=selected, folders=folders), value=selected)

    gr.on(fn=restore, outputs=checkpoint, **private)
    return checkpoint, refresh
