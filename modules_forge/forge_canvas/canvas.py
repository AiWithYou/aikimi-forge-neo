"""
Forge Canvas
Copyright (C) 2024 lllyasviel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.
"""

import base64
import json
import os
import sys
import uuid
from functools import wraps
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import gradio.component_meta
import numpy as np
from PIL import Image

from modules.gradio_compat import keep_hidden_component_mounted
from modules.gradio_file_url import gradio_file_url

create_or_modify_pyi_org = gradio.component_meta.create_or_modify_pyi


def create_or_modify_pyi_org_patched(component_class, class_name, events):
    try:
        if component_class.__name__ == "LogicalImage":
            return
        return create_or_modify_pyi_org(component_class, class_name, events)
    except Exception:
        return


gradio.component_meta.create_or_modify_pyi = create_or_modify_pyi_org_patched


DEBUG_MODE = False
canvas_js_root_path = os.path.dirname(__file__)


def web_js(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    url = gradio_file_url(full_path, cache_key=os.path.getmtime(full_path))
    return f'<script src="{url}"></script>\n'


def web_css(file_name):
    full_path = os.path.join(canvas_js_root_path, file_name)
    url = gradio_file_url(full_path, cache_key=os.path.getmtime(full_path))
    return f'<link rel="stylesheet" href="{url}">\n'


canvas_html = open(os.path.join(canvas_js_root_path, "canvas.html"), encoding="utf-8").read()
canvas_head = "".join((web_css("canvas.css"), web_js("canvas.js")))


def image_to_base64(image_array, numpy=True):
    image = Image.fromarray(image_array) if numpy else image_array
    image = image.convert("RGBA")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{image_base64}"


def base64_to_image(base64_str, numpy=True):
    if base64_str.startswith("data:image/png;base64,"):
        base64_str = base64_str.replace("data:image/png;base64,", "")
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    image = image.convert("RGBA")
    image_array = np.array(image) if numpy else image
    return image_array


class LogicalImage(gr.Textbox):
    @wraps(gr.Textbox.__init__)
    def __init__(self, *args, numpy=True, file_transport=False, **kwargs):
        self.numpy = numpy
        self.file_transport = file_transport
        self.infotext = dict()
        self._file_paths: set[Path] = set()

        # Textbox calls this component's postprocess for the initial value too.
        super().__init__(*args, **kwargs)

    def preprocess(self, payload):
        if not isinstance(payload, str):
            return None

        if self.file_transport:
            if not payload:
                return None
            if len(payload) > 8192 or not payload.startswith("forge-file:"):
                raise ValueError("Invalid canvas image reference")
            path = Path(json.loads(payload[len("forge-file:") :])["path"]).resolve()
            if path not in self._file_paths or not path.is_file():
                raise ValueError("Canvas image must be an uploaded cache file")
            if path.stat().st_size > 64 * 1024 * 1024:
                raise ValueError("Canvas image exceeds 64 MB")
            with Image.open(path) as source:
                if source.width * source.height > 40_000_000 or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("Invalid canvas image size or frame count")
                image = source.convert("RGBA")
            image.info = self.infotext
            return np.array(image) if self.numpy else image

        if not payload.startswith("data:image/png;base64,"):
            return None

        image = base64_to_image(payload, numpy=self.numpy)
        if hasattr(image, "info"):
            image.info = self.infotext

        return image

    def postprocess(self, value):
        if value is None:
            return None

        if hasattr(value, "info"):
            self.infotext = value.info

        if self.file_transport:
            image = Image.fromarray(value) if self.numpy else value
            path = gr.processing_utils.save_pil_to_cache(image.convert("RGBA"), self.GRADIO_CACHE, format="png")
            # Forge redirects Gradio's image cache to its configured temp dir.
            # Accept only files emitted by this component, not arbitrary paths
            # under either application's shared cache directory.
            self._file_paths.add(Path(path).resolve())
            return "forge-file:" + json.dumps({"path": path, "url": gradio_file_url(path)}, ensure_ascii=False)

        return image_to_base64(value, numpy=self.numpy)

    def get_block_name(self):
        return "textbox"


class ForgeCanvas:
    def __init__(
        self,
        no_upload=False,
        no_scribbles=False,
        contrast_scribbles=False,
        height=None,
        scribble_color="#000000",
        scribble_color_fixed=False,
        scribble_width=25,
        scribble_width_fixed=False,
        scribble_alpha=100,
        scribble_alpha_fixed=False,
        scribble_softness=0,
        scribble_softness_fixed=False,
        visible=True,
        numpy=False,
        initial_image=None,
        elem_id=None,
        elem_classes=None,
        file_background=False,
    ):
        # Resolve settings when the canvas is built, not while importing its
        # reusable component. Standalone extension tests need no model runtime.
        opts = getattr(sys.modules.get("modules.shared"), "opts", None)
        if opts is None:
            opts = SimpleNamespace(
                forge_canvas_plain=False,
                forge_canvas_plain_color="#808080",
                forge_canvas_toolbar_always=False,
                forge_canvas_height=512,
                forge_canvas_consistent_brush=False,
            )
        self.uuid = "uuid_" + uuid.uuid4().hex

        canvas_html_uuid = canvas_html.replace("forge_mixin", self.uuid)

        if opts.forge_canvas_plain:
            canvas_html_uuid = canvas_html_uuid.replace(
                'class="forge-image-container"',
                f'class="forge-image-container plain" style="background-color: {opts.forge_canvas_plain_color}"',
            ).replace('stroke="white"', "stroke=#444")
        if opts.forge_canvas_toolbar_always:
            canvas_html_uuid = canvas_html_uuid.replace('class="forge-toolbar"', 'class="forge-toolbar-static"')

        canvas_arguments = [
            self.uuid,
            no_upload,
            no_scribbles,
            contrast_scribbles,
            height or opts.forge_canvas_height,
            scribble_color,
            scribble_color_fixed,
            scribble_width,
            scribble_width_fixed,
            opts.forge_canvas_consistent_brush,
            scribble_alpha,
            scribble_alpha_fixed,
            scribble_softness,
            scribble_softness_fixed,
        ]
        # A Blocks.load handler can run before a lazy tab mounts this element.
        # Initialize with the HTML's own lifecycle, after its assets are ready.
        self.block = gr.HTML(
            canvas_html_uuid,
            visible=visible,
            elem_id=elem_id,
            elem_classes=elem_classes,
            head=canvas_head,
            js_on_load=f"new ForgeCanvas(...{json.dumps(canvas_arguments)});",
        )
        self.foreground = LogicalImage(
            visible=keep_hidden_component_mounted(DEBUG_MODE),
            label="foreground",
            numpy=numpy,
            elem_id=self.uuid,
            elem_classes=["logical_image_foreground"] + ([] if DEBUG_MODE else ["forge-logical-image"]),
        )
        self.background = LogicalImage(
            visible=keep_hidden_component_mounted(DEBUG_MODE),
            file_transport=file_background,
            label="background",
            numpy=numpy,
            value=initial_image,
            elem_id=self.uuid,
            elem_classes=["logical_image_background"] + ([] if DEBUG_MODE else ["forge-logical-image"]),
        )
