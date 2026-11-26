import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, PngImagePlugin


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_gallery_actions_and_metadata(self):
        from modules_forge.aikimi_gallery import Gallery

        file = self.root / "outputs/moss-image/sample.png"
        file.parent.mkdir(parents=True)
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("parameters", "sky\nNegative prompt: noise\nSteps: 20, Seed: 7, Model: blue")
        Image.new("RGB", (1, 1)).save(file, pnginfo=metadata)
        gallery = Gallery(self.root)
        (item,) = gallery.list_items(query="sky", media="image", studio="moss_image")["items"]
        self.assertEqual(item["metadata"]["seed"], "7")
        self.assertNotIn(str(self.root), json.dumps(item))
        gallery.favorite(item["id"], True)
        self.assertEqual(gallery.list_items(favorites=True)["total"], 1)
        gallery.quarantine(item["id"])
        self.assertFalse(file.exists())
        self.assertEqual(gallery.list_items()["total"], 0)

    def test_gallery_excludes_directory_links_and_metadata_paths(self):
        import os

        from modules_forge.aikimi_gallery import Gallery

        outputs = self.root / "outputs"
        outputs.mkdir()
        other = self.root / "other"
        other.mkdir()
        (other / "private.wav").touch()
        (outputs / "sound.wav").touch()
        (outputs / "result.json").write_text(json.dumps({"model": "/opt/private/model", "seed": 19}))
        try:
            (outputs / "redirect").symlink_to(other, target_is_directory=True)
        except OSError:
            if os.name != "nt":
                raise
            import _winapi

            _winapi.CreateJunction(str(other), str(outputs / "redirect"))
        gallery = Gallery(self.root)
        self.assertEqual(gallery.list_items()["total"], 1)
        with self.assertRaises(ValueError):
            gallery.quarantine("outputs/redirect/private.wav")
        self.assertTrue((other / "private.wav").exists())
        self.assertNotIn("/opt/private", json.dumps(gallery.list_items()))

    def test_gallery_reads_bounded_compressed_international_text_after_image_data(self):
        from modules_forge.aikimi_gallery import Gallery

        path = self.root / "outputs/art.png"
        path.parent.mkdir()
        import struct
        import zlib

        payload = json.dumps({"prompt": "雲", "seed": 15}, ensure_ascii=False).encode()
        body = b"iTXtmoss_image" + bytes((0, 1, 0, 0, 0)) + zlib.compress(payload)
        chunk = struct.pack(">I", len(body) - 4) + body + struct.pack(">I", zlib.crc32(body))
        Image.new("RGB", (2, 2)).save(path)
        png = path.read_bytes()
        path.write_bytes(png[:-12] + chunk + png[-12:])
        self.assertEqual(Gallery(self.root).list_items()["items"][0]["metadata"]["prompt"], "雲")

    def test_comfy_sidecar_and_media_pagination(self):
        from modules_forge.aikimi_gallery import Gallery

        directory = self.root / "output"
        directory.mkdir()
        (directory / "clip.mp4").touch()
        (directory / "clip.json").write_text('{"request":{"seed":42}}')
        graph = {"1": {"inputs": {"text": "sky"}}, "2": {"inputs": {"positive": ["1", 0], "seed": 43}}}
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("prompt", json.dumps(graph))
        Image.new("RGB", (1, 1)).save(directory / "sample.png", pnginfo=metadata)
        gallery = Gallery(self.root)
        self.assertEqual(gallery.list_items(query="sky")["items"][0]["metadata"]["seed"], "43")
        self.assertEqual(gallery.list_items(query="42", media="video")["total"], 1)
        self.assertEqual(len(gallery.list_items(offset=1, limit=1)["items"]), 1)

    def test_h3_parameters_and_yue_result_sidecars(self):
        from modules_forge.aikimi_gallery import Gallery

        gallery = Gallery(self.root)
        for studio, directory, media, sidecar in (
            ("minimax_h3_image", "minimax_h3/images", "result.png", "parameters.json"),
            ("yue2", "yue2", "song.wav", "studio-result.json"),
        ):
            folder = self.root / "outputs" / directory / "job"
            folder.mkdir(parents=True)
            (folder / media).touch()
            (folder / sidecar).write_text(json.dumps({"prompt": "studio sample"}))
            self.assertEqual(gallery.list_items(query="studio sample", studio=studio)["total"], 1)

    def test_authenticated_workspace_routes(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from modules.aikimi_workspace_api import register_workspace_routes

        app = FastAPI()
        app.auth = True
        app.tokens = {"test-session": "operator"}
        app.cookie_id = "workspace"
        from unittest.mock import patch

        with patch("modules.paths_internal.data_path", str(self.root)):
            register_workspace_routes(app)
        with TestClient(app) as client:
            base = "/aikimi/api/v1/"
            for route in ("gallery", "prompt-library", "models/summary", "status"):
                self.assertEqual(client.get(base + route).status_code, 401)
            self.assertEqual(client.post(base + "gpu/release", json={}).status_code, 401)
            client.cookies.set("access-token-workspace", "test-session")
            self.assertEqual(client.get(base + "gallery").json()["total"], 0)
            self.assertEqual(client.post(base + "gpu/release").status_code, 415)
            entry = client.post(base + "prompt-library", json={"prompt": "<tag>", "title": "test"}).json()
            self.assertEqual(entry["prompt"], "<tag>")
            self.assertEqual(client.post(base + "prompt-library", json={"title": "x", "id": "../x"}).status_code, 400)
            self.assertEqual(
                client.request(
                    "DELETE", base + "prompt-library/" + entry["id"], headers={"Content-Type": "application/json"}
                ).status_code,
                200,
            )

    def test_gallery_refuses_paths(self):
        from modules_forge.aikimi_gallery import Gallery

        for key in ("outputs/../secrets/x.png", "outputs/C:/x.png", "/foo.png", "tmp/file.png", "outputs/x.txt"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                Gallery(self.root).media_path(key)

    def test_library_round_trip(self):
        from modules_forge.prompt_library import PromptLibrary

        library = PromptLibrary(self.root)
        entry = library.save({"title": "sky", "prompt": "{day|night}"})
        self.assertIn(entry, PromptLibrary(self.root).entries())
        library.save({**entry, "prompt": "sun"})
        self.assertEqual(next(x for x in library.entries() if x["id"] == entry["id"])["prompt"], "sun")
        library.delete(entry["id"])
        with self.assertRaises(ValueError):
            library.delete("../secrets")

    def test_nested_choices_and_limits(self):
        from modules_forge.prompt_library import expand_wildcards

        text = r"{red|{blue|pink}} \{keep\}"
        result, seed = expand_wildcards(text)
        self.assertEqual((result, seed), expand_wildcards(text, seed))
        self.assertIn(result.split()[0], ["red", "blue", "pink"])
        self.assertTrue(result.endswith("{keep}"))
        with self.assertRaises(ValueError):
            expand_wildcards("{" * 20000)

    def test_summary_and_capabilities(self):
        from modules.aikimi_diagnostics import DiagnosticPaths
        from modules_forge.model_summary import model_summary

        models = self.root / "models"
        (models / "Lora").mkdir(parents=True)
        (models / "Lora/test.safetensors").write_bytes(b"123")
        diffusion = models / "MiniMax-H3/diffusion_models"
        diffusion.mkdir(parents=True)
        (diffusion / "p-h3-image_w4a8.safetensors").write_bytes(b"12")
        (diffusion / "minimax_h3_ref2va.safetensors").write_bytes(b"1")
        report = model_summary(DiagnosticPaths(self.root, models, self.root / "outputs"))
        self.assertEqual(report["loras"], 1)
        self.assertEqual(report["total_bytes"], 6)
        self.assertEqual(report["features"]["minimax_h3_image"]["checkpoints"], 1)
        self.assertEqual(report["features"]["minimax_h3"]["checkpoints"], 1)
        self.assertEqual(len(report["features"]), 13)

    def test_routes(self):
        from fastapi import FastAPI

        from modules.aikimi_workspace_api import register_workspace_routes

        app = FastAPI()
        register_workspace_routes(app)
        routes = {(path, method.upper()) for path, methods in app.openapi()["paths"].items() for method in methods}
        for suffix, method in [
            ("gallery", "GET"),
            ("prompt-library", "GET"),
            ("prompt-library", "POST"),
            ("prompt-library/{entry_id}", "DELETE"),
            ("models/summary", "GET"),
            ("gpu/release", "POST"),
        ]:
            self.assertIn(("/aikimi/api/v1/" + suffix, method), routes)

    def test_resident_owner_visible_across_threads(self):
        from concurrent.futures import ThreadPoolExecutor

        from modules_forge import gpu_residency

        gpu_residency.begin("moss_image")
        try:
            with ThreadPoolExecutor() as pool:
                self.assertEqual(pool.submit(gpu_residency.residency_snapshot).result()["current_engine"], "moss_image")
        finally:
            gpu_residency.finish()
