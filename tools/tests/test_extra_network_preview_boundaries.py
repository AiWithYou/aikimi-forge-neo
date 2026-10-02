"""Exercise real preview callbacks, file responses, and literal metadata names."""

import ast
import datetime
import html
import json
import os
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin

ROOT = Path(__file__).resolve().parents[2]


def load_boundaries():
    namespace = {
        "datetime": datetime,
        "html": html,
        "json": json,
        "os": os,
        "Path": Path,
        "HTTPException": HTTPException,
        "Image": Image,
        "PngImagePlugin": PngImagePlugin,
        "opts": SimpleNamespace(enable_pnginfo=True, jpeg_quality=95),
        "image_from_url_text": lambda value: value,
        "read_info_from_image": lambda image: (image.info.get("parameters"), image.info.copy()),
        "allowed_dirs": set(),
        "allowed_preview_extensions": lambda: {"png"},
    }
    for filename, names in (
        ("modules/images.py", {"save_image_with_geninfo"}),
        ("modules/ui_extra_networks.py", {"path_is_parent", "fetch_file", "setup_ui"}),
        ("modules/sysinfo.py", {"pretty_bytes"}),
    ):
        path = ROOT / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    namespace["sysinfo"] = SimpleNamespace(pretty_bytes=namespace["pretty_bytes"])
    namespace["errors"] = SimpleNamespace(display=lambda error, *_: (_ for _ in ()).throw(error))
    namespace["ui_extra_networks"] = SimpleNamespace(path_is_parent=namespace["path_is_parent"])
    namespace["images"] = SimpleNamespace(
        read_info_from_image=namespace["read_info_from_image"],
        save_image_with_geninfo=namespace["save_image_with_geninfo"],
    )
    namespace["infotext_utils"] = SimpleNamespace(image_from_url_text=namespace["image_from_url_text"])
    path = ROOT / "modules/ui_extra_networks_user_metadata.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "UserMetadataEditor")
    names = {
        "__init__",
        "get_user_metadata",
        "get_card_html",
        "relative_path",
        "get_metadata_table",
        "put_values_into_components",
        "save_preview",
        "write_user_metadata",
    }
    owner.body = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


class TextCells(HTMLParser):
    def __init__(self):
        super().__init__()
        self.cells = []
        self.active = False

    def handle_starttag(self, tag, attrs):
        if tag == "td":
            self.active = True
            self.cells.append("")

    def handle_endtag(self, tag):
        if tag == "td":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.cells[-1] += data


class ExtraNetworkPreviewBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.namespace = load_boundaries()
        self.folder = tempfile.TemporaryDirectory(prefix="forge-preview-bounds-")
        self.addCleanup(self.folder.cleanup)
        self.base = Path(self.folder.name)
        self.models = self.base / "models"
        self.models.mkdir()
        self.outside = self.base / "models-extra"
        self.outside.mkdir()

    def external_link(self):
        target = self.outside / "protected.png"
        Image.new("RGB", (2, 2), "blue").save(target)
        link = self.models / "escaped.png"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error}")
        return link, target

    def test_model_file_name_is_literal_in_metadata_table(self):
        for name in ("日本語 &copy; &notin;.safetensors", "normal.safetensors"):
            with self.subTest(name=name):
                path = self.models / name
                path.write_bytes(b"model")
                page = SimpleNamespace(
                    extra_networks_tabname="lora",
                    items={name: {"filename": str(path), "description": "original description"}},
                    allowed_directories_for_previews=lambda: [str(self.models)],
                    find_preview=lambda _: None,
                )
                editor = self.namespace["UserMetadataEditor"](None, "txt2img", page)
                values = editor.put_values_into_components(name)
                parser = TextCells()
                parser.feed(values[2])
                self.assertEqual(parser.cells[0], name)
                self.assertEqual(values[0], html.escape(name))
                self.assertEqual(values[1], "original description")

    def test_path_containment_handles_siblings_symlinks_and_windows_case(self):
        contains = self.namespace["path_is_parent"]
        link, target = self.external_link()
        self.assertTrue(contains(str(self.models), str(self.models / "new.png")))
        for path in (target, link):
            with self.subTest(path=path):
                self.assertFalse(contains(str(self.models), str(path)))
        if os.name == "nt":
            self.assertTrue(contains(str(self.models).upper(), str(self.models / "new.png")))
            self.assertFalse(contains(str(self.models), "Z:\\unrelated\\new.png"))

    def test_thumbnail_serves_real_png_and_rejects_external_symlink(self):
        link, target = self.external_link()
        valid = self.models / "日本語.png"
        valid.write_bytes(target.read_bytes())
        self.namespace["allowed_dirs"].add(str(self.models))
        app = FastAPI()
        app.add_api_route("/thumb", self.namespace["fetch_file"], methods=["GET"])
        response = TestClient(app).get("/thumb", params={"filename": str(valid)})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, valid.read_bytes())
        self.assertEqual(response.headers["content-type"], "image/png")
        with self.assertRaises(ValueError):
            TestClient(app).get("/thumb", params={"filename": str(link)})

    def test_preview_save_callback_keeps_sibling_and_symlink_targets(self):
        link, target = self.external_link()
        original = target.read_bytes()
        callbacks = []
        page = SimpleNamespace(
            allowed_directories_for_previews=lambda: [str(self.models)], create_html=lambda _: "cards"
        )
        ui = SimpleNamespace(
            stored_extra_pages=[page],
            pages=[],
            user_metadata_editors=[],
            tabname="txt2img",
            preview_target_filename=object(),
            button_save_preview=SimpleNamespace(click=lambda **kwargs: callbacks.append(kwargs["fn"])),
        )
        self.namespace["setup_ui"](ui, object())
        image = Image.new("RGB", (3, 3), "red")
        image.info["parameters"] = "literal generation info"
        for path in (target, link):
            with self.subTest(path=path), self.assertRaises(AssertionError):
                callbacks[0](0, [image], str(path))
        self.assertEqual(target.read_bytes(), original)
        valid = self.models / "new.png"
        self.assertEqual(callbacks[0](0, [image], str(valid)), ["cards"])
        with Image.open(valid) as saved:
            self.assertEqual(saved.size, image.size)
            self.assertEqual(saved.info["parameters"], image.info["parameters"])

    def test_preview_symlink_cannot_disguise_model_weights_as_png(self):
        target = self.models / "model.safetensors"
        target.write_bytes(b"model weights must remain unchanged")
        link = self.models / "model.png"
        try:
            link.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error}")
        self.namespace["allowed_dirs"].add(str(self.models))
        with self.subTest(action="read"), self.assertRaises(ValueError):
            self.namespace["fetch_file"](str(link))
        callbacks = []
        page = SimpleNamespace(
            allowed_directories_for_previews=lambda: [str(self.models)], create_html=lambda _: "cards"
        )
        ui = SimpleNamespace(
            stored_extra_pages=[page],
            pages=[],
            user_metadata_editors=[],
            tabname="txt2img",
            preview_target_filename=object(),
            button_save_preview=SimpleNamespace(click=lambda **kwargs: callbacks.append(kwargs["fn"])),
        )
        self.namespace["setup_ui"](ui, object())
        with self.subTest(action="write"), self.assertRaises(KeyError):
            callbacks[0](0, [Image.new("RGB", (2, 2), "red")], str(link))
        self.assertEqual(target.read_bytes(), b"model weights must remain unchanged")

    def test_modern_preview_editor_rejects_external_link_and_keeps_valid_save(self):
        link, target = self.external_link()
        original = target.read_bytes()
        updates = []
        item = {"filename": str(self.models / "model.safetensors"), "local_preview": str(link)}
        page = SimpleNamespace(
            extra_networks_tabname="lora",
            items={"model": item},
            allowed_directories_for_previews=lambda: [str(self.models)],
            find_preview=lambda _: None,
            lister=SimpleNamespace(update_file_entry=updates.append),
        )
        editor = self.namespace["UserMetadataEditor"](None, "txt2img", page)
        image = Image.new("RGB", (3, 3), "red")
        image.info["parameters"] = "日本語 generation info"
        with self.assertRaises(AssertionError):
            editor.save_preview(0, [image], "model")
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(updates, [])
        valid = self.models / "new.png"
        item["local_preview"] = str(valid)
        self.assertEqual(editor.save_preview(0, [image], "model")[1], "")
        with Image.open(valid) as saved:
            self.assertEqual(saved.size, image.size)
            self.assertEqual(saved.info["parameters"], image.info["parameters"])
        self.assertEqual(updates, [str(valid)])

    def test_metadata_sidecar_cannot_write_through_external_symlink(self):
        target = self.outside / "protected.json"
        original = b'{"original": true}'
        target.write_bytes(original)
        sidecar = self.models / "model.json"
        try:
            sidecar.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error}")
        updates = []
        item = {"filename": str(self.models / "model.safetensors")}
        page = SimpleNamespace(
            extra_networks_tabname="lora",
            items={"model": item},
            lister=SimpleNamespace(update_file_entry=updates.append),
        )
        editor = self.namespace["UserMetadataEditor"](None, "txt2img", page)
        with self.assertRaises(AssertionError):
            editor.write_user_metadata("model", {"notes": "日本語"})
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(updates, [])
        sidecar.unlink()
        editor.write_user_metadata("model", {"notes": "日本語"})
        self.assertEqual(json.loads(sidecar.read_text(encoding="utf-8")), {"notes": "日本語"})
        self.assertEqual(updates, [str(sidecar)])
        item["filename"] = str(self.outside / "explicit-external-model.safetensors")
        editor.write_user_metadata("model", {"notes": "external model sidecar"})
        external_sidecar = self.outside / "explicit-external-model.json"
        self.assertEqual(json.loads(external_sidecar.read_text(encoding="utf-8")), {"notes": "external model sidecar"})

    def test_registered_editor_links_cannot_overwrite_model_weights(self):
        target = self.models / "model.safetensors"
        original = b"model weights must remain unchanged"
        target.write_bytes(original)
        preview = self.models / "model.png"
        sidecar = self.models / "model.json"
        try:
            preview.symlink_to(target)
            sidecar.symlink_to(target)
        except OSError as error:
            self.skipTest(f"symlink creation is unavailable: {error}")
        updates = []
        item = {"filename": str(target), "local_preview": str(preview)}
        page = SimpleNamespace(
            extra_networks_tabname="lora",
            items={"model": item},
            allowed_directories_for_previews=lambda: [str(self.models)],
            lister=SimpleNamespace(update_file_entry=updates.append),
        )
        editor = self.namespace["UserMetadataEditor"](None, "txt2img", page)
        with self.subTest(action="preview"), self.assertRaises(KeyError):
            editor.save_preview(0, [Image.new("RGB", (2, 2), "red")], "model")
        with self.subTest(action="metadata"), self.assertRaises(AssertionError):
            editor.write_user_metadata("model", {"notes": "日本語"})
        self.assertEqual(target.read_bytes(), original)
        self.assertEqual(updates, [])
