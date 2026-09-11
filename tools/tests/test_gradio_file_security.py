import ast
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urljoin

from modules.aikimi_security.paths import (
    UnsafeAllowedPathError,
    build_gradio_allowed_paths,
    build_gradio_blocked_paths,
)
from modules.gradio_file_url import gradio_file_url

ROOT = Path(__file__).resolve().parents[2]


def _attribute_name(node):
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


class GradioFileSecurityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.script_root = self.base / "repo"
        self.data_root = self.base / "data"
        self.canvas_root = self.script_root / "modules_forge" / "forge_canvas"
        self.javascript_root = self.script_root / "javascript"
        self.extension_root = self.script_root / "extensions-builtin" / "active #日本%ui"
        self.extra_extension_root = self.data_root / "extensions" / "active-extra-ui"
        self.inactive_extension_root = self.data_root / "extensions" / "inactive-ui"
        self.webui_assets_root = self.script_root / "modules" / "web"
        for path in (
            self.script_root,
            self.data_root,
            self.script_root / "output",
            self.data_root / "outputs",
            self.data_root / "tmp",
            self.canvas_root,
            self.javascript_root,
            self.extension_root / "javascript",
            self.extension_root / "scripts",
            self.extra_extension_root / "javascript",
            self.inactive_extension_root / "javascript",
            self.script_root / "html",
            self.webui_assets_root / "css",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("canvas.js", "canvas.css", "canvas.py"):
            (self.canvas_root / name).write_text(name, encoding="utf-8")

        self.root_script = self.script_root / "script.js"
        self.root_style = self.script_root / "style.css"
        self.root_javascript = self.javascript_root / "core.js"
        self.root_module = self.javascript_root / "module.mjs"
        self.extension_javascript = self.extension_root / "javascript" / "extension.js"
        self.extension_module = self.extension_root / "javascript" / "extension.mjs"
        self.extension_style = self.extension_root / "style.css"
        self.extension_python = self.extension_root / "scripts" / "extension.py"
        self.extra_extension_javascript = self.extra_extension_root / "javascript" / "extra.js"
        self.extra_extension_style = self.extra_extension_root / "style.css"
        self.inactive_extension_javascript = self.inactive_extension_root / "javascript" / "inactive.js"
        self.notification_audio = self.script_root / "notification.mp3"
        self.card_placeholder = self.script_root / "html" / "card-no-preview.jpg"
        self.private_html = self.script_root / "html" / "private.html"
        self.font_stylesheet = self.webui_assets_root / "css" / "sourcesanspro.css"
        for path in (
            self.root_script,
            self.root_style,
            self.root_javascript,
            self.root_module,
            self.extension_javascript,
            self.extension_module,
            self.extension_style,
            self.extension_python,
            self.extra_extension_javascript,
            self.extra_extension_style,
            self.inactive_extension_javascript,
            self.notification_audio,
            self.card_placeholder,
            self.private_html,
            self.font_stylesheet,
        ):
            path.write_text(path.name, encoding="utf-8")

        self.javascript_paths = (
            self.root_javascript,
            self.root_module,
            self.extension_javascript,
            self.extension_module,
            self.extra_extension_javascript,
        )
        self.stylesheet_paths = (
            self.root_style,
            self.extension_style,
            self.extra_extension_style,
        )

    def build_allowed_paths(self):
        return build_gradio_allowed_paths(
            self.script_root,
            self.data_root,
            canvas_root=self.canvas_root,
            javascript_paths=self.javascript_paths,
            stylesheet_paths=self.stylesheet_paths,
            notification_audio=self.notification_audio,
        )

    def test_allowed_paths_contain_managed_directories_and_exact_ui_assets(self):
        result = {Path(path) for path in self.build_allowed_paths()}

        self.assertIn((self.script_root / "output").resolve(), result)
        self.assertIn((self.data_root / "outputs").resolve(), result)
        self.assertIn((self.data_root / "tmp").resolve(), result)
        self.assertIn((self.canvas_root / "canvas.js").resolve(), result)
        self.assertIn((self.canvas_root / "canvas.css").resolve(), result)
        for path in (
            self.root_script,
            self.root_style,
            *self.javascript_paths,
            self.extension_style,
            self.extra_extension_style,
            self.notification_audio,
            self.card_placeholder,
        ):
            self.assertIn(path.resolve(), result)
        self.assertNotIn(self.script_root.resolve(), result)
        self.assertNotIn(self.data_root.resolve(), result)
        self.assertNotIn(self.javascript_root.resolve(), result)
        self.assertNotIn(self.extension_root.resolve(), result)
        self.assertNotIn((self.canvas_root / "canvas.py").resolve(), result)
        self.assertNotIn(self.extension_python.resolve(), result)
        self.assertNotIn(self.inactive_extension_javascript.resolve(), result)
        self.assertNotIn(self.private_html.resolve(), result)

    def test_fresh_managed_directories_are_created_before_first_output(self):
        script_root = self.base / "fresh-repository"
        data_root = self.base / "fresh-data"
        script_root.mkdir()
        data_root.mkdir()

        allowed = build_gradio_allowed_paths(script_root, data_root)
        first_output = data_root / "output" / "first-generation.png"
        first_output.write_bytes(b"first")

        from gradio.routes import file_fetch

        policy = SimpleNamespace(allowed_paths=allowed, blocked_paths=[])
        upload_dir = data_root / "upload"
        upload_dir.mkdir()
        response = file_fetch(str(first_output), SimpleNamespace(headers={}), policy, upload_dir)

        self.assertEqual(response.status_code, 200)
        for name in ("output", "outputs", "tmp"):
            managed = (data_root / name).resolve()
            self.assertTrue(managed.is_dir())
            self.assertIn(str(managed), allowed)
            self.assertFalse((script_root / name).exists())

    def test_user_stylesheet_is_an_exact_allowed_asset(self):
        user_stylesheet = self.data_root / "user.css"
        user_stylesheet.write_text("body { color: canvastext; }", encoding="utf-8")

        allowed = self.build_allowed_paths()

        from gradio.routes import file_fetch

        policy = SimpleNamespace(allowed_paths=allowed, blocked_paths=[])
        upload_dir = self.data_root / "upload-user-css"
        upload_dir.mkdir()
        response = file_fetch(str(user_stylesheet), SimpleNamespace(headers={}), policy, upload_dir)
        self.assertEqual(response.status_code, 200)
        self.assertIn(str(user_stylesheet.resolve()), allowed)

    def test_gradio_file_url_encodes_windows_safe_url_delimiters_and_unicode(self):
        url = gradio_file_url(self.extension_javascript, cache_key="revision #1")

        self.assertNotIn("#", url)
        self.assertNotIn(" ", url)
        self.assertNotIn("日本", url)
        self.assertIn("active%20%23%E6%97%A5%E6%9C%AC%25ui", url)
        self.assertTrue(url.endswith("?revision%20%231"), url)

    def test_static_asset_inputs_reject_non_ui_files_and_symlink_escapes(self):
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                javascript_paths=[self.extension_python],
            )

        outside = self.base / "secret.js"
        outside.write_text("secret", encoding="utf-8")
        escaped = self.javascript_root / "escaped.js"
        try:
            escaped.symlink_to(outside)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                javascript_paths=[escaped],
            )

    def test_arbitrary_requested_path_and_symlink_escape_are_rejected(self):
        outside = self.base / "private"
        outside.mkdir()
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                requested_paths=[outside],
            )

        link = self.data_root / "outputs" / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                requested_paths=[link],
            )

    def test_managed_root_and_canvas_asset_symlink_escapes_are_rejected(self):
        outside = self.base / "outside"
        outside.mkdir()
        output_link = self.script_root / "outputs"
        try:
            output_link.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"symlink creation is unavailable: {exc}")

        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                canvas_root=self.canvas_root,
            )

        output_link.unlink()
        secret = outside / "secret.txt"
        secret.write_text("secret", encoding="utf-8")
        canvas_asset = self.canvas_root / "canvas.js"
        canvas_asset.unlink()
        canvas_asset.symlink_to(secret)
        with self.assertRaises(UnsafeAllowedPathError):
            build_gradio_allowed_paths(
                self.script_root,
                self.data_root,
                canvas_root=self.canvas_root,
            )

    def test_blocked_paths_cover_state_credentials_logs_models_and_dynamic_secrets(self):
        dynamic = (
            self.data_root / ".env.local",
            self.data_root / "sysinfo-2026.json",
            self.data_root / "server.key",
            self.script_root / "client.p12",
        )
        for path in dynamic:
            path.write_text("secret", encoding="utf-8")

        result = {
            Path(path)
            for path in build_gradio_blocked_paths(
                self.script_root,
                self.data_root,
            )
        }

        for relative in (
            ".git",
            "config.json",
            "ui-config.json",
            "forge_neo_model_paths.yaml",
            "models",
            "repositories",
            "venv",
            "logs",
            "secrets",
            "api-auth.txt",
            "gradio-auth.txt",
            "params.txt",
            "webui-user.bat",
        ):
            with self.subTest(relative=relative):
                self.assertIn((self.data_root / relative).resolve(), result)
        for path in dynamic:
            self.assertIn(path.resolve(), result)
        self.assertNotIn((self.script_root / "extensions").resolve(), result)
        self.assertNotIn((self.script_root / "extensions-builtin").resolve(), result)

    def test_gradio_async_file_validator_is_not_monkeypatched(self):
        source_path = ROOT / "modules" / "ui_tempdir.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        assignments = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                assignments.extend(_attribute_name(target) for target in targets)

        self.assertNotIn(
            "gradio.processing_utils.async_move_files_to_cache",
            assignments,
            "A copied validator must not replace Gradio's maintained async file checks.",
        )
        self.assertIn("gradio.processing_utils.save_pil_to_cache", assignments)

    def test_gradio_allowed_path_cli_default_is_empty(self):
        tree = ast.parse((ROOT / "modules" / "cmd_args.py").read_text(encoding="utf-8"))
        matches = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            if node.args[0].value != "--gradio-allowed-path":
                continue
            defaults = [keyword.value for keyword in node.keywords if keyword.arg == "default"]
            matches.extend(defaults)

        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], ast.List)
        self.assertEqual(matches[0].elts, [])

    def test_gradio_file_policy_blocks_state_but_serves_managed_output(self):
        from gradio.routes import file_fetch
        from starlette.exceptions import HTTPException

        config = self.data_root / "config.json"
        git_config = self.data_root / ".git" / "config"
        model = self.data_root / "models" / "private.safetensors"
        output = self.data_root / "outputs" / "result.txt"
        config.write_text("{}", encoding="utf-8")
        git_config.parent.mkdir(parents=True)
        git_config.write_text("private", encoding="utf-8")
        model.parent.mkdir(parents=True)
        model.write_text("model", encoding="utf-8")
        output.write_text("result", encoding="utf-8")

        allowed = self.build_allowed_paths()
        blocked = build_gradio_blocked_paths(self.script_root, self.data_root)
        policy = SimpleNamespace(
            allowed_paths=allowed,
            blocked_paths=blocked,
        )
        request = SimpleNamespace(headers={})
        upload_dir = self.base / "uploads"
        upload_dir.mkdir()

        for path in (
            self.script_root,
            config,
            git_config,
            model,
            self.extension_python,
            self.inactive_extension_javascript,
            self.private_html,
            self.canvas_root / "canvas.py",
        ):
            with self.subTest(path=path), self.assertRaises(HTTPException) as raised:
                file_fetch(str(path), request, policy, upload_dir)
            self.assertEqual(raised.exception.status_code, 403)

        for path in (
            output,
            self.canvas_root / "canvas.js",
            self.root_script,
            self.root_style,
            *self.javascript_paths,
            self.extension_style,
            self.extra_extension_style,
            self.notification_audio,
            self.card_placeholder,
        ):
            with self.subTest(path=path):
                response = file_fetch(str(path), request, policy, upload_dir)
                self.assertEqual(response.status_code, 200)

    def test_gradio_6_html_exact_static_assets_are_fetchable(self):
        import gradio as gr
        from fastapi import FastAPI
        from fastapi.staticfiles import StaticFiles
        from fastapi.testclient import TestClient

        html_assets = (
            self.root_script,
            self.root_style,
            self.root_javascript,
            self.root_module,
            self.extension_javascript,
            self.extension_module,
            self.extension_style,
            self.extra_extension_javascript,
            self.extra_extension_style,
            self.canvas_root / "canvas.js",
            self.canvas_root / "canvas.css",
        )
        tags = []
        for asset in html_assets:
            url = gradio_file_url(asset, cache_key="test")
            if asset.suffix == ".css":
                tags.append(f'<link rel="stylesheet" href="{url}">')
            else:
                tags.append(f'<script src="{url}"></script>')

        with gr.Blocks() as demo:
            gr.Markdown("static asset regression")
        host = FastAPI()
        host.mount(
            "/aikimi/webui-assets",
            StaticFiles(directory=self.webui_assets_root),
            name="webui-assets",
        )
        app = gr.mount_gradio_app(
            host,
            demo,
            path="/aikimi",
            allowed_paths=self.build_allowed_paths(),
            blocked_paths=build_gradio_blocked_paths(self.script_root, self.data_root),
            head="".join(tags),
        )

        with TestClient(app) as client:
            document = client.get("/aikimi/")
            self.assertEqual(document.status_code, 200)
            config_match = re.search(
                r"window\.gradio_config = (\{.*?\});</script>",
                document.text,
                flags=re.DOTALL,
            )
            self.assertIsNotNone(config_match, document.text[-2000:])
            config = json.loads(config_match.group(1))
            self.assertEqual(config["head"], "".join(tags))
            urls = re.findall(r'(?:src|href)="(gradio_api/file=[^"]+)"', config["head"])
            self.assertEqual(len(urls), len(html_assets), document.text[-2000:])
            for url in urls:
                with self.subTest(url=url):
                    response = client.get(urljoin(str(document.url), url))
                    self.assertEqual(response.status_code, 200, response.text)

            font_css = client.get(urljoin(str(document.url), "webui-assets/css/sourcesanspro.css"))
            self.assertEqual(font_css.status_code, 200, font_css.text)
            card = client.get(
                urljoin(
                    str(document.url),
                    f"gradio_api/file={self.card_placeholder.as_posix()}",
                )
            )
            self.assertEqual(card.status_code, 200, card.text)
            private_html = client.get(
                urljoin(
                    str(document.url),
                    f"gradio_api/file={self.private_html.as_posix()}",
                )
            )
            self.assertEqual(private_html.status_code, 403)

    def test_production_url_builders_and_webui_use_gradio_6_static_routes(self):
        ui_extensions = (ROOT / "modules" / "ui_gradio_extensions.py").read_text(encoding="utf-8")
        canvas = (ROOT / "modules_forge" / "forge_canvas" / "canvas.py").read_text(encoding="utf-8")
        webui = (ROOT / "webui.py").read_text(encoding="utf-8")

        self.assertIn("from modules.gradio_file_url import gradio_file_url", ui_extensions)
        self.assertIn("gradio_file_url(util.truncate_path(fn)", ui_extensions)
        self.assertIn("if os.path.isfile(user_css):", ui_extensions)
        self.assertIn("from modules.gradio_file_url import gradio_file_url", canvas)
        self.assertNotIn('src="file=', canvas)
        self.assertNotIn('href="file=', canvas)
        self.assertIn('scripts.list_scripts("javascript", extension)', webui)
        self.assertIn('scripts.list_files_with_name("style.css")', webui)
        root_style = (ROOT / "style.css").read_text(encoding="utf-8")
        self.assertIn('@import url("../webui-assets/css/sourcesanspro.css")', root_style)


if __name__ == "__main__":
    unittest.main()
