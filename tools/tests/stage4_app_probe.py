"""Run isolated contracts against a normally initialized CPU Forge application."""

import base64
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        from modules import paths_internal

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.enterContext(patch.object(paths_internal, "data_path", str(self.root)))

    def test_workspace_and_styles_auth(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from modules import shared
        from modules.api.api import Api
        from tools.tests.test_stage2_integration import load

        builder = load("extensions-builtin/sd_forge_json_builder/scripts/builder.py")
        with (
            patch.object(shared.cmd_opts, "api_auth", "reader:test,読者:合言葉"),
            patch.object(shared.cmd_opts, "gradio_auth", "operator:test"),
        ):
            for api_only in (False, True):
                with patch.object(shared.cmd_opts, "nowebui", api_only):
                    app = FastAPI()
                    if not api_only:
                        app.auth = {"operator": "test"}
                        app.cookie_id = "fixture"
                        app.tokens = {"session": "operator"}
                    api = Api(app, None, workspace_only=True)
                    api.add_api_route("/basic-only", lambda: {}, methods=["GET"])
                    builder.json_builder_api(None, app)
                    with TestClient(app) as client:
                        with self.subTest(api_only=api_only, unauthenticated_post=True):
                            self.assertEqual(client.post("/aikimi/api/v1/prompt-library").status_code, 401)
                        for path in ("/aikimi/api/v1/gallery", "/internal/json_builder_prompt_styles"):
                            with self.subTest(path=path, api_only=api_only):
                                self.assertEqual(client.get(path).status_code, 401)
                                for pair in ("reader:test", "読者:合言葉"):
                                    basic = base64.b64encode(pair.encode()).decode()
                                    self.assertEqual(
                                        client.get(path, headers={"Authorization": "Basic " + basic}).status_code, 200
                                    )
                                client.cookies.set("access-token-fixture", "session")
                                self.assertEqual(client.get(path).status_code, 401 if api_only else 200)
                                client.cookies.clear()
                        basic = base64.b64encode("読者:合言葉".encode()).decode()
                        self.assertEqual(
                            client.get("/basic-only", headers={"Authorization": "Basic " + basic}).status_code, 200
                        )
                        self.assertEqual(client.get("/basic-only").status_code, 401)

    def test_password_replacement(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from modules import shared
        from modules.api.api import Api

        for inline, file_content in (("user:old,user:new", ""), ("", "user:old\nuser:new"), ("user:old", "user:new")):
            path = self.root / "auth.txt"
            path.write_text(file_content)
            with (
                patch.object(shared.cmd_opts, "api_auth", inline),
                patch.object(shared.cmd_opts, "api_auth_path", str(path) if file_content else None),
            ):
                app = FastAPI()
                api = Api(app, None, workspace_only=True)
                api.add_api_route("/basic-only", lambda: {}, methods=["GET"])
                with TestClient(app) as client:
                    for route in ("/basic-only", "/aikimi/api/v1/gallery"):
                        for password, status in (("new", 200), ("old", 401)):
                            with self.subTest(route=route, inline=bool(inline), file=bool(file_content), status=status):
                                self.assertEqual(client.get(route, auth=("user", password)).status_code, status)

    def test_workspace_body_limit(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from modules.api.api import Api

        app = FastAPI()
        Api(app, None, workspace_only=True)
        data = b'{"title":"' + b"x" * 1048576 + b'"}'
        with TestClient(app) as client:
            for streamed in (False, True):
                with (
                    self.subTest(streamed=streamed),
                    patch("json.loads", side_effect=AssertionError("JSON parsed")) as parse,
                    patch("modules_forge.prompt_library.PromptLibrary.save") as save,
                ):
                    body = iter([data[:50], data[50:]]) if streamed else data
                    response = client.post(
                        "/aikimi/api/v1/prompt-library", content=body, headers={"Content-Type": "application/json"}
                    )
                    self.assertEqual(response.status_code, 413)
                    parse.assert_not_called()
                    save.assert_not_called()

    def test_extensions_chromium(self):
        import gradio as gr

        from modules import ui_extensions as ext
        from modules.gradio_frontend_compat import install_gradio_tabs_compatibility_route
        from tools.tests.chromium_helpers import cdp_page, find_chromium, reserve_local_port

        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chromium unavailable")
        states = {"Snapshot": {"filepath": "fixture"}}
        self.enterContext(patch.object(ext.config_states, "list_config_states"))
        self.enterContext(patch.object(ext.config_states, "all_config_states", states))
        self.enterContext(patch.object(ext, "update_config_states_table", lambda name: name))
        restore = self.enterContext(patch.object(ext.config_states, "restore_webui_config"))
        save = self.enterContext(patch.object(ext.shared.opts, "save"))
        restart = self.enterContext(patch.object(ext.shared.state, "request_restart"))
        url = "https://example.invalid/fixture.git"
        self.enterContext(
            patch.object(
                ext, "available_extensions", {"extensions": [{"url": url, "commit_time": "2026-01-01T00:00:00Z"}]}
            )
        )

        def refresh(index, *filters):
            html, _ = ext.refresh_available_extensions_from_data(*filters)
            return index, html, [], "", ""

        self.enterContext(patch.object(ext, "refresh_available_extensions", refresh))
        called = threading.Event()

        def record_install(directory, selected_url):
            called.set()
            return "", "installed"

        install = self.enterContext(patch.object(ext, "install_extension_from_url", side_effect=record_install))
        blocks = ext.create_ui()
        self.addCleanup(blocks.close)

        with blocks:
            ready = gr.Textbox(value="", elem_id="extensions-ready")
            blocks.load(lambda: "ready", outputs=ready)
        for component in blocks.blocks.values():
            if component.elem_id == "extension_backup_saved_configs":
                component.value = "Snapshot"
            if component.elem_id == "extension_backup_restore_type":
                component.value = "both"
        app, address, _ = blocks.launch(server_port=reserve_local_port(), prevent_thread_lock=True, quiet=True)
        install_gradio_tabs_compatibility_route(app)
        with cdp_page(chromium, address) as page:
            self.assertTrue(page.wait_for("#extensions-ready textarea", "element.value==='ready'"))
            page.evaluate(
                Path("javascript/extensions.js").read_text("utf8")
                + """
                window.config_state_confirm_restore = config_state_confirm_restore;
                window.install_extension_from_index = install_extension_from_index;
                window.gradioApp = () => document;
                window.updateInput = el => el.dispatchEvent(new Event('input', {bubbles:true}));
                window.restart_reload = () => {};
                window.tabClick = name => [...document.querySelectorAll('button')].find(e=>e.getClientRects().length && e.textContent.trim()===name).click();
            """
            )
            with self.subTest(action="restore"):
                page.evaluate(
                    "document.querySelectorAll('#tabs_extensions .tab-container[role=tablist] > button')[3].click()"
                )
                self.assertTrue(
                    page.wait_for("#extension_backup_restore", "true"),
                    (page.exceptions, page.evaluate("document.body.innerText.slice(-2500)")),
                )
                page.evaluate("window.confirm=()=>false;document.querySelector('#extension_backup_restore').click()")
                self.assertTrue(page.wait_for("body", "element.textContent.includes('Cancelled')", 8000))
                self.assertEqual([f.call_count for f in (restore, save, restart)], [0, 0, 0])
                page.evaluate("window.confirm=()=>true;document.querySelector('#extension_backup_restore').click()")
                self.assertTrue(page.wait_for("body", "!element.textContent.includes('Cancelled')"))
                self.assertEqual([f.call_count for f in (restore, save, restart)], [1, 1, 1])
            with self.subTest(action="install"):
                page.evaluate(
                    "document.querySelectorAll('#tabs_extensions .tab-container[role=tablist] > button')[1].click()"
                )
                self.assertTrue(
                    page.wait_for(
                        "body",
                        "[...element.querySelectorAll('button')].some(e=>e.textContent.trim()==='Load from Index')",
                    )
                )
                page.evaluate("tabClick('Load from Index')")
                self.assertTrue(page.wait_for("#available_extensions button", "true"))
                page.evaluate("document.querySelector('#available_extensions button').click()")
                self.assertTrue(called.wait(15), (page.exceptions, page.evaluate("document.body.innerText.slice(-1400)")))
                install.assert_called_once_with(None, url)

    def test_gallery_library_and_moss_chromium(self):
        import gradio as gr

        from modules.api.api import Api
        from tools.tests.chromium_helpers import cdp_page, find_chromium, reserve_local_port
        from tools.tests.test_moss_image_ui import UITests
        from tools.tests.test_stage2_integration import load

        chromium = find_chromium()
        if not chromium:
            self.skipTest("Chromium unavailable")
        folder = self.root / "outputs/moss-image/job"
        folder.mkdir(parents=True)
        (folder / "output.png").touch()
        (folder / "output.json").write_text(json.dumps({"prompt": "sunrise", "negative_prompt": "rain"}))
        for name in ("p_h3_images", "official_h3_images"):
            images = self.root / "outputs/minimax_h3" / name
            images.mkdir(parents=True)
            (images / "image.png").touch()
            (images / "parameters.json").write_text(json.dumps({"prompt": name}))
        h3 = load("extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py").on_ui_tabs()[0][0]
        self.addCleanup(h3.close)
        blocks = UITests().load_screen().on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        with blocks:
            gr.HTML(
                '<nav id="aikimi-feature-nav"><button id="aikimi-tab-moss-image">MOSS</button><button id="aikimi-tab-h3-image">H3 Image</button></nav><div id="aikimi-gallery-workspace"></div>'
            )
            h3.render()
        app, address, _ = blocks.launch(server_port=reserve_local_port(), prevent_thread_lock=True, quiet=True)
        Api(app, None, workspace_only=True)
        with cdp_page(chromium, address) as page:
            self.assertTrue(page.wait_for("#aikimi-feature-nav", "true"))
            page.evaluate(Path("script.js").read_text("utf8") + "\nwindow.get_uiTabButton = get_uiTabButton;")
            page.evaluate(
                """
                window.feature='gallery'; window.AikimiTabs={getActiveFeature:()=>feature, getActiveContainer:()=>document};
                document.querySelector('#aikimi-tab-moss-image').onclick=()=>{window.feature='moss_image'};
                document.querySelector('#aikimi-tab-h3-image').onclick=()=>{window.feature='minimax_h3_image'};
                window.onUiLoaded=fn=>fn(); window.onUiUpdate=()=>{};
            """
                + Path("extensions-builtin/aikimi-ui/javascript/aikimi_workspace.js").read_text("utf8")
            )
            self.assertTrue(page.wait_for(".aikimi-gallery-results", "element.children.length===3"))
            page.evaluate(
                "[...document.querySelectorAll('.aikimi-gallery-results button')].find(b=>b.textContent==='output.png').click()"
            )
            page.evaluate("""window.clickText = name => [...document.querySelectorAll('button')].find(e=>e.textContent===name).click();
                clickText('プロンプト集へ保存')""")
            self.assertTrue(
                page.wait_for("#aikimi-gallery-workspace output", "element.textContent.includes('保存しました')")
            )
            page.evaluate("clickText('MOSS Image へ送る')")
            self.assertTrue(page.wait_for("#moss-image-prompt textarea", "element.value==='sunrise'"))
            self.assertTrue(page.wait_for("#moss-image-negative-prompt textarea", "element.value==='rain'"))
            page.evaluate("document.querySelector('#aikimi-library-open').click()")
            self.assertTrue(
                page.wait_for("#aikimi-prompt-library select", "[...element.options].some(o=>o.text==='output.png')")
            )
            page.evaluate("""const d=document.querySelector('#aikimi-prompt-library');const s=d.querySelector('select');
                s.value=[...s.options].find(o=>o.text==='output.png').value;s.onchange();
                d.querySelectorAll('textarea')[1].value='';clickText('スタジオへ挿入')""")
            self.assertTrue(page.wait_for("#moss-image-negative-prompt textarea", "element.value===''"))
            page.evaluate("""const filter = document.querySelector('#aikimi-gallery-workspace select');
                filter.value='minimax_h3_image'; filter.onchange()""")
            self.assertTrue(page.wait_for(".aikimi-gallery-results", "element.children.length===2"))
            self.assertTrue(page.wait_for("#ph3-prompt textarea", "element.getClientRects().length"))
            page.evaluate("""window.h3Writes = []; document.addEventListener('input', e => {
                if (e.target.closest('#ph3-prompt, #official-h3-prompt')) h3Writes.push(e.target.closest('[id$=prompt]').id);
            });""")
            for name, field in (("official_h3_images", "official-h3-prompt"), ("p_h3_images", "ph3-prompt")):
                page.evaluate(f"""{{const query=document.querySelector('#aikimi-gallery-workspace input');
                    query.value='{name}'; query.onchange()}}""")
                self.assertTrue(page.wait_for(".aikimi-gallery-results", "element.children.length===1"))
                self.assertTrue(page.wait_for(".aikimi-gallery-detail", "!element.children.length"))
                page.evaluate("h3Writes.length=0; clickText('image.png'); clickText('H3 Image へ送る')")
                self.assertTrue(page.wait_for("#" + field + " textarea", f"element.getClientRects().length && element.value==='{name}'"))
                self.assertEqual(page.evaluate("h3Writes"), [field])
            page.evaluate("get_uiTabButton(document.querySelector('#h3-image-tabs'), 'official_h3_image').click()")
            self.assertTrue(page.wait_for("#official-h3-prompt textarea", "element.value==='official_h3_images'"))



if __name__ == "__main__":
    from modules_forge.initialization import initialize_forge

    with tempfile.TemporaryDirectory() as root:
        selected_tests = sys.argv[1:]
        sys.argv = [sys.argv[0], "--cpu", "--data-dir", root]
        initialize_forge()
        from modules import initialize

        initialize.imports()
        unittest.main(argv=[sys.argv[0], *selected_tests], verbosity=2)
