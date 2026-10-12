"""Shared image controls, arranged once inside their owning Gradio Blocks.

The model retains its components, events, job state and service. No components
are copied across Blocks and panel disclosure never changes a feature's value.
"""

from dataclasses import dataclass
from functools import partial

import gradio as gr

PRIVATE = {"api_visibility": "private", "show_progress": "hidden", "queue": False}


def badge(label, value):
    if isinstance(value, (list, tuple)):
        return f"{label} · {len(value)}" if value else label
    if isinstance(value, bool):
        return f"{label} · ON" if value else label
    return f"{label} · {value}" if value and value != "off" else label


@dataclass
class Panel:
    button: gr.Button
    group: gr.Group
    state: gr.State

    @staticmethod
    def toggle(opened):
        return not opened, gr.update(visible="hidden" if opened else True)

    def reveal(self):
        return True, gr.update(visible=True)


def descendants(block):
    yield block
    for child in getattr(block, "children", ()):
        yield from descendants(child)


def create_panel(label, components, key, tools, panel_slot, index, indicators=()):
    def indicator_value(values):
        return any(values) if len(values) > 1 else values[0]

    with tools:
        button = gr.Button(
            badge(label, indicator_value([item.value for item in indicators])) if indicators else label,
            size="sm",
            scale=0,
            min_width=90,
            elem_id=f"{key}-panel-{index}-toggle",
            elem_classes=["aikimi-image-panel-toggle"],
        )
    state = gr.State(False)
    state.do_not_save_to_config = True
    with panel_slot:
        with gr.Group(visible="hidden", elem_id=f"{key}-panel-{index}", elem_classes=["aikimi-image-panel"]) as group:
            for component in components:
                for child in descendants(component):
                    if isinstance(child, gr.Accordion) and "input-accordion" not in (child.elem_classes or []):
                        child.open = True
                        child.elem_classes = [*(child.elem_classes or []), "aikimi-image-panel-content"]
                        child._constructor_args[0].update(open=True, elem_classes=child.elem_classes)
                        child.do_not_save_to_config = True
                component.unrender().render()
    panel = Panel(button, group, state)
    button.click(panel.toggle, inputs=state, outputs=[state, group], **PRIVATE)
    for indicator in indicators:
        indicator.change(
            fn=lambda *values: gr.update(value=badge(label, indicator_value(values))),
            inputs=list(indicators),
            outputs=button,
            **PRIVATE,
        )
    return panel


def forge_toolbar(tab, key, scripts, *, before=None):
    """Disclose the existing optional script groups above Forge's prompts."""
    before = before or tab.children[0]
    panels = {}
    with gr.Column(elem_id=f"{key}-image-toolbar", elem_classes=["aikimi-image-panel-host"]) as toolbar:
        with gr.Row(elem_id=f"{key}-tools", elem_classes=["aikimi-image-tools"]) as tools:
            pass
        with gr.Column(elem_classes=["aikimi-image-panels"]) as slot:
            pass
    for script in scripts:
        group = getattr(script, "group", None)
        if group is None or not group.children or getattr(script, "section", None) is not None:
            continue
        label = script.title()
        if label == "Local LoRA stack":
            label = "LoRA"
        children = list(descendants(group))
        indicators = [item for item in children if isinstance(item, gr.Dropdown) and item.multiselect]
        if not indicators:
            indicators = [
                item
                for item in children
                if isinstance(item, gr.Checkbox)
                and (hasattr(item, "accordion_id") or str(item.label).lower() in {"enable", "enabled"})
            ]
        panels[label] = create_panel(
            label, [group], key, tools, slot, len(panels), indicators[:1] if label == "LoRA" else indicators
        )
    tab.children.remove(toolbar)
    tab.children.insert(tab.children.index(before), toolbar)
    return panels


class ImageWorkspace:
    def __init__(
        self,
        tab,
        key,
        *,
        editable=False,
        models=(),
        prompts=(),
        actions=(),
        settings=(),
        references=(),
        outputs=(),
        panels=None,
        badges=None,
        native_mode=None,
        mode_values=("text", "edit"),
    ):
        original = tuple(tab.children)
        self.editable = editable
        self.native_mode = native_mode
        self.mode_values = mode_values
        self.panels = {}
        for component in actions:
            if isinstance(component, gr.Button) and component.value in {"画像を生成", "生成・編集", "デザインを生成"}:
                component.value = "生成"
            elif isinstance(component, gr.Button) and component.value in {"停止", "キャンセル"}:
                component.value = "中断"
        self.mode = gr.State("txt2img")
        self.mode.do_not_save_to_config = True
        with gr.Group(elem_id=f"{key}-image-ui", elem_classes=["aikimi-image-ui"]) as self.root:
            with gr.Row(elem_id=f"{key}-model", elem_classes=["aikimi-image-model"]) as self.models:
                self.place(models)
            with gr.Tabs(selected="txt2img", elem_id=f"{key}-modes", elem_classes=["aikimi-image-modes"]) as self.tabs:
                with gr.Tab("txt2img", id="txt2img", elem_id=f"{key}-txt2img", render_children=True) as text_tab:
                    pass
                with gr.Tab(
                    "img2img", id="img2img", elem_id=f"{key}-img2img", interactive=editable, render_children=True
                ) as image_tab:
                    pass
            with gr.Row(elem_id=f"{key}-tools", elem_classes=["aikimi-image-tools"]) as tools:
                pass
            with gr.Row(elem_id=f"{key}-toprow", elem_classes=["aikimi-image-toprow"]):
                with gr.Column(
                    scale=6, min_width=280, elem_id=f"{key}-prompts", elem_classes=["aikimi-image-prompts"]
                ) as self.prompts:
                    self.place(prompts)
                with gr.Column(scale=1, min_width=160, elem_id=f"{key}-actions", elem_classes=["aikimi-image-actions"]):
                    self.place(actions)
            with gr.Column(elem_classes=["aikimi-image-panels"]) as panel_slot:
                pass
            with gr.Row(elem_id=f"{key}-workspace", elem_classes=["aikimi-image-workspace"]):
                with gr.Column(
                    scale=1, min_width=280, elem_id=f"{key}-settings", elem_classes=["aikimi-image-settings"]
                ) as self.settings:
                    with gr.Group(
                        visible="hidden", elem_id=f"{key}-references", elem_classes=["aikimi-image-references"]
                    ) as self.references:
                        self.place(references)
                    with gr.Group(elem_classes=["aikimi-image-parameters"]):
                        self.place(settings)
                with gr.Column(
                    scale=1, min_width=280, elem_id=f"{key}-output", elem_classes=["aikimi-image-output"]
                ) as self.output:
                    self.place(outputs)
        for label, components in (panels or {}).items():
            self.add_panel(label, components, key, tools, panel_slot, (badges or {}).get(label))
        # Everything else remains mounted, including preparation controls,
        # hidden callback outputs and state. Existing events retain their IDs.
        remaining = [block for block in original if getattr(block, "parent", None) is tab]
        if remaining:
            self.add_panel("モデル詳細", remaining, key, tools, panel_slot)
        targets = [self.mode, self.tabs, self.references]
        if native_mode is not None:
            native_mode.visible = "hidden"
            targets.append(native_mode)
            native_mode.change(self.sync_native_mode, inputs=[native_mode, self.mode], outputs=targets[:3], **PRIVATE)
        inputs = [self.mode] + ([native_mode] if native_mode is not None else [])
        text_tab.select(partial(self.tab_selected, "txt2img"), inputs=inputs, outputs=targets, **PRIVATE)
        if editable:
            image_tab.select(partial(self.tab_selected, "img2img"), inputs=inputs, outputs=targets, **PRIVATE)
        self.tabs.do_not_save_to_config = True
        for panel in self.panels.values():
            panel.state.do_not_save_to_config = True

    @staticmethod
    def place(components):
        for component in components:
            component.unrender().render()

    def add_panel(self, label, components, key, tools, panel_slot, indicator=None):
        self.panels[label] = create_panel(
            label,
            components,
            key,
            tools,
            panel_slot,
            len(self.panels),
            [indicator] if indicator is not None else [],
        )

    def select_mode(self, mode):
        if mode not in {"txt2img", "img2img"} or (mode == "img2img" and not self.editable):
            raise ValueError("このモデルはimg2imgに対応していません。")
        result = [mode, gr.update(selected=mode), gr.update(visible=True if mode == "img2img" else "hidden")]
        if self.native_mode is not None:
            result.append(self.mode_values[mode == "img2img"])
        return tuple(result)

    def tab_selected(self, mode, current_mode, native_value=None):
        result = list(self.select_mode(mode))
        result[1] = gr.skip()
        if mode == current_mode:
            result[:3] = [gr.skip()] * 3
        if self.native_mode is not None and native_value == self.mode_values[mode == "img2img"]:
            result[3] = gr.skip()
        return tuple(result)

    def sync_native_mode(self, value, current_mode):
        mode = "img2img" if value == self.mode_values[1] else "txt2img"
        if mode == current_mode:
            return (gr.skip(),) * 3
        return self.select_mode(mode)[:3]
