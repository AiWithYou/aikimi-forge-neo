"""Visible Studio selectors backed only by the local asset library."""

from modules_forge import local_assets, model_catalog


def inventory(family, roots, validator, *, current="", search_directory="", folders=False):
    sources = model_catalog.checkpoint_roots(family, roots, search_directory)
    paths = model_catalog.discover_files(sources, suffixes=(".safetensors",), folders=folders, validator=validator)
    return [("標準モデル", ""), *model_catalog.choice_labels(paths, sources, current=current)]


def save(engine, checkpoint, search_directory):
    value = local_assets.selection(engine)
    value.update(checkpoint=checkpoint or "", search_directory=search_directory or "")
    local_assets.save_selection(engine, value)


def create(engine, family, roots, validator, *, gradio, folders=False, details=""):
    gr = gradio
    value = local_assets.selection(engine)
    current = value.get("checkpoint", "")
    search_value = value.get("search_directory", "")
    with gr.Row():
        checkpoint = model_catalog.configure_component(
            gr.Dropdown(
                choices=inventory(
                    family, roots, validator, current=current, search_directory=search_value, folders=folders
                ),
                value=current,
                label="チェックポイント",
                scale=5,
                interactive=True,
            )
        )
        refresh = gr.Button("一覧を更新", scale=0, min_width=105)
    with gr.Accordion("モデルの探索先", open=False):
        search = model_catalog.configure_component(
            gr.Textbox(value=search_value, label="探索フォルダー", placeholder=f"models/{family}/checkpoints/")
        )
        if details:
            gr.Markdown(details)
    private = {"api_visibility": "private", "show_progress": "hidden", "queue": False}

    def reload(selected, directory):
        save(engine, selected, directory)
        return gr.update(
            choices=inventory(
                family, roots, validator, current=selected or "", search_directory=directory or "", folders=folders
            ),
            value=selected or "",
        )

    refresh.click(reload, inputs=[checkpoint, search], outputs=checkpoint, **private)
    for component in (checkpoint, search):
        component.input(
            lambda selected, directory: save(engine, selected, directory), inputs=[checkpoint, search], **private
        )

    def restore():
        latest = local_assets.selection(engine)
        selected = latest.get("checkpoint", "")
        directory = latest.get("search_directory", "")
        return (
            gr.update(
                choices=inventory(
                    family, roots, validator, current=selected, search_directory=directory, folders=folders
                ),
                value=selected,
            ),
            gr.update(value=directory),
        )

    gr.on(fn=restore, outputs=[checkpoint, search], **private)
    return checkpoint, search, refresh
