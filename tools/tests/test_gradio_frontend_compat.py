from __future__ import annotations

import hashlib
import json
import shutil
import socket
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urljoin
from urllib.request import urlopen

import gradio
import httpx
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.testclient import TestClient

from modules import gradio_frontend_compat
from modules.aikimi_security.gradio_file_guard import install_gradio_file_url_guard


class GradioFrontendCompatibilityTests(unittest.TestCase):
    def test_dataframe_column_count_preserves_compiled_header_semantics(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled dataframe headers")
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_text(encoding="utf-8")
        asset = module.build_patched_dataframe_asset()
        script = (
            "import assert from 'node:assert/strict';\n"
            + "const originalSource = "
            + json.dumps(source)
            + ";\nconst servedSource = "
            + json.dumps(asset.content.decode("utf-8"))
            + ";\n"
            + r"""
            function between(source, start, end) {
                const offset = source.indexOf(start);
                assert.notEqual(offset, -1, start);
                assert.equal(source.indexOf(start, offset + 1), -1, start);
                const finish = source.indexOf(end, offset + start.length);
                assert.notEqual(finish, -1, end);
                return source.slice(offset + start.length, finish);
            }
            function columnGetter(source) {
                const matches = source.match(/get col_count\(\)\{return r\.props\.col_count[^}]*\}/g) ?? [];
                assert.equal(matches.length, 1);
                return matches[0];
            }
            const headerArrow = between(originalSource, 'M=R(', '),I=R(()=>r.props.value?.data??[])');
            const headerDerived = '()=>{let f=o()??[];' + between(
                originalSource, 'B=R(()=>{let f=o()??[];', '),ie=R(()=>l(B).map('
            );
            assert.equal(headerArrow, '()=>r.props.value?.headers??[]');
            assert.equal(headerDerived, '()=>{let f=o()??[];' + between(
                servedSource, 'B=R(()=>{let f=o()??[];', '),ie=R(()=>l(B).map('
            ));
            assert.equal(originalSource.split('get headers(){return l(M)}').length, 2);
            // Evaluate the actual compiled expressions; this models lazy reads,
            // not Svelte's scheduler or a browser mount lifecycle.
            const R = fn => fn, l = signal => signal();
            const make = source => new Function('r', 'R', 'l',
                `const M=R(${headerArrow});return({get headers(){return l(M)},${columnGetter(source)}});`
            );
            const original = make(originalSource), guarded = make(servedSource);
            const derive = new Function('o', 't', `return (${headerDerived})();`);
            const evaluate = node => derive(() => node.headers, node);
            const headers = Object.freeze(['LoRA', '強度']);
            const data = Object.freeze([Object.freeze(['chosen', 0.5])]);
            let cases = 0;
            function check(run) { run(); cases++; }
            for (const state of ['undefined', 'null', 'absent']) {
                check(() => {
                    const value = { headers, data };
                    const props = { value };
                    if (state !== 'absent') props.col_count = state === 'null' ? null : undefined;
                    const before = JSON.stringify(props);
                    const hadCount = Object.hasOwn(props, 'col_count');
                    assert.throws(() => evaluate(original({ props }, R, l)),
                        error => error instanceof TypeError && /reading '1'/.test(error.message));
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), ['LoRA', '強度']);
                    assert.deepEqual(node.col_count, [2, 'fixed']);
                    assert.equal(props.value, value);
                    assert.equal(props.value.headers, headers);
                    assert.equal(props.value.data, data);
                    assert.equal(Object.hasOwn(props, 'col_count'), hadCount);
                    assert.equal(JSON.stringify(props), before);
                });
            }
            for (const tuple of [[2, 'fixed'], [4, 'fixed'], [4, 'dynamic'], [0, 'fixed']]) {
                check(() => {
                    const col_count = Object.freeze(tuple);
                    const value = { headers, data };
                    const props = { value, col_count };
                    const before = JSON.stringify(props);
                    const node = guarded({ props }, R, l);
                    assert.equal(node.col_count, col_count);
                    assert.deepEqual(evaluate(node), evaluate(original({ props }, R, l)));
                    assert.equal(props.value, value);
                    assert.equal(props.value.headers, headers);
                    assert.equal(props.value.data, data);
                    assert.equal(props.col_count, col_count);
                    assert.equal(JSON.stringify(props), before);
                });
            }
            for (const absent of [undefined, null]) {
                for (const kind of ['fixed', 'dynamic']) {
                    check(() => {
                        const props = { value: { headers, data }, col_count: absent };
                        const node = guarded({ props }, R, l);
                        assert.deepEqual(evaluate(node), ['LoRA', '強度']);
                        const restored = Object.freeze([4, kind]);
                        props.col_count = restored;
                        assert.equal(node.col_count, restored);
                        assert.deepEqual(evaluate(node), kind === 'fixed'
                            ? ['LoRA', '強度', '2', '3'] : ['LoRA', '強度']);
                        assert.equal(props.value.headers, headers);
                        assert.equal(props.value.data, data);
                    });
                }
            }
            for (const initial of [[], undefined, null]) {
                check(() => {
                    const value = { headers: initial, data };
                    const props = { value };
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), []);
                    for (const next of [headers, Object.freeze(['LoRA', '強度', 'Note']), [], null]) {
                        value.headers = next;
                        const expected = next ?? [];
                        const before = JSON.stringify(props);
                        assert.deepEqual(node.col_count, [expected.length, 'fixed']);
                        assert.deepEqual(evaluate(node), expected);
                        assert.equal(value.headers, next);
                        assert.equal(value.data, data);
                        assert.equal(props.value, value);
                        assert.equal(Object.hasOwn(props, 'col_count'), false);
                        assert.equal(JSON.stringify(props), before);
                    }
                    value.headers = headers;
                    for (const kind of ['fixed', 'dynamic']) {
                        const restored = Object.freeze([3, kind]);
                        props.col_count = restored;
                        assert.equal(node.col_count, restored);
                        assert.deepEqual(evaluate(node), kind === 'fixed'
                            ? ['LoRA', '強度', '2'] : ['LoRA', '強度']);
                    }
                });
            }
            for (const kind of ['fixed', 'dynamic']) {
                check(() => {
                    const empty = Object.freeze([]), col_count = Object.freeze([3, kind]);
                    const props = { value: { headers: empty, data }, col_count };
                    const node = guarded({ props }, R, l);
                    assert.deepEqual(evaluate(node), ['0', '1', '2']);
                    assert.deepEqual(evaluate(node), evaluate(original({ props }, R, l)));
                    assert.equal(node.col_count, col_count);
                    assert.equal(props.value.headers, empty);
                    assert.equal(props.value.data, data);
                });
            }
            for (const value of [undefined, null]) {
                check(() => {
                    const props = { value };
                    assert.deepEqual(evaluate(guarded({ props }, R, l)), []);
                    assert.equal(props.value, value);
                    assert.equal(Object.hasOwn(props, 'col_count'), false);
                });
            }
            console.log(`compiled dataframe col_count: ${cases} scenarios passed`);
            """
        )
        result = subprocess.run(  # noqa: S603 - execute expressions extracted from the audited frontend asset
            [node, "--input-type=module", "-"], input=script, text=True, capture_output=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("compiled dataframe col_count: 18 scenarios passed", result.stdout)

    def test_dataframe_column_count_getter_missing_or_duplicated_fails_closed(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        source = path.read_bytes()
        getter = b"get col_count(){return r.props.col_count}"
        self.assertEqual(source.count(getter), 1)
        for name, modified in (
            ("missing", source.replace(getter, b"get col_count(){return undefined}", 1)),
            ("duplicated", source + getter),
        ):
            with (
                self.subTest(snippet=name),
                patch.object(module, "DATAFRAME_ORIGINAL_SHA256", hashlib.sha256(modified).hexdigest()),
                patch.object(Path, "read_bytes", return_value=modified),
            ):
                with self.assertRaisesRegex(
                    module.GradioFrontendCompatibilityError, "column-count getter snippet was not found exactly once"
                ):
                    module.build_patched_dataframe_asset()

    def test_tab_sync_preserves_registry_when_update_omits_initial_tabs(self):
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is required to execute the compiled tab synchronization")
        module = gradio_frontend_compat
        asset = module.build_patched_tabs_asset()
        self.assertIn(module._STATIC_INITIAL_TAB_SYNC, asset.content)
        self.assertIn(module._STATIC_TAB_REGISTRATION, asset.content)
        script = (
            """
            import assert from 'node:assert/strict';
            const current = { id: 'active', label: 'Renamed', visible: false, interactive: false };
            const l = { value: [current] }, he = { value: [current] }, I = { value: [] };
            const le = { value: false }, we = { value: false }, ae = new Set([0]);
            const P = { value: 'active' }, X = { value: 0 };
            const e = state => state.value, d = (state, value) => state.value = value;
            const N = d, v = () => e(P);
            const Ce = () => Promise.resolve();
        """
            + module._STATIC_INITIAL_TAB_SYNC.decode("utf-8")
            + "\nconst registry = ({"
            + module._STATIC_TAB_REGISTRATION.decode("utf-8")
            + "});\n"
            + """
            Se(undefined);
            Se(null);
            await Promise.resolve();
            assert.deepEqual(e(l), [current]);
            assert.deepEqual(e(he), [current]);
            const added = { id: 'new', label: 'New', visible: true, interactive: true };
            Se([{ id: 'active', label: 'Old', visible: true, interactive: true }, added]);
            await Promise.resolve();
            assert.deepEqual(e(l), [current, added]);
            assert.deepEqual(e(he), [current, added]);
            registry.register_tab(current, 0);
            assert.equal(e(P), 'new');
            assert.equal(e(X), 1);
            registry.register_tab({ ...added, visible: 'hidden' }, 1);
            assert.equal(e(P), false);
            assert.equal(e(X), -1);
            registry.register_tab({ ...current, visible: true, interactive: false }, 0);
            assert.equal(e(P), false);
            registry.register_tab(added, 1);
            assert.equal(e(P), 'new');
            Se([]);
            await Promise.resolve();
            assert.deepEqual(e(l), []);
            assert.deepEqual(e(he), []);
        """
        )
        result = subprocess.run(  # noqa: S603 - execute the audited frontend snippet
            [node, "--input-type=module", "-"], input=script, text=True, capture_output=True, timeout=10
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_apptree_visibility_patch_is_exact_served_and_keeps_the_wheel(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.APPTREE_ASSET_NAME
        before = path.read_bytes()
        asset = module.build_patched_apptree_asset()
        self.assertEqual(before, path.read_bytes())
        self.assertNotIn(module._ORIGINAL_APPTREE_VISIBILITY_SYNC, asset.content)
        self.assertIn(module._REACTIVE_APPTREE_VISIBILITY_SYNC, asset.content)
        app = module.create_gradio_compatibility_app(module.build_patched_tabs_asset())
        response = TestClient(app).get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        for attribute, value in (
            ("APPTREE_ORIGINAL_SHA256", "0" * 64),
            ("_ORIGINAL_APPTREE_VISIBILITY_SYNC", b"missing audited snippet"),
        ):
            with self.subTest(attribute=attribute), patch.object(module, attribute, value):
                with self.assertRaises(module.GradioFrontendCompatibilityError):
                    module.build_patched_apptree_asset()
        with patch.object(gradio, "__version__", "6.29.0"):
            with self.assertRaises(module.GradioFrontendCompatibilityError):
                module.build_patched_apptree_asset()

    def test_dataframe_route_is_verified_and_does_not_modify_the_wheel(self):
        module = gradio_frontend_compat
        path = Path(gradio.__file__).resolve().parent / "templates/frontend/assets" / module.DATAFRAME_ASSET_NAME
        before = path.read_bytes()
        asset = module.build_patched_dataframe_asset()
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(hashlib.sha256(before).hexdigest(), asset.original_sha256)
        self.assertEqual(hashlib.sha256(asset.content).hexdigest(), asset.patched_sha256)
        self.assertEqual(before.count(b"get col_count(){return r.props.col_count}"), 1)
        self.assertEqual(asset.content.count(b"get col_count(){return r.props.col_count}"), 0)
        self.assertEqual(asset.content.count(b'get col_count(){return r.props.col_count??[l(M).length,"fixed"]}'), 1)
        self.assertNotIn(module._ORIGINAL_TABLE_RESIZE, asset.content)
        self.assertEqual(asset.content.count(module._REACTIVE_TABLE_RESIZE), 1)
        app = module.create_gradio_compatibility_app(module.build_patched_tabs_asset())
        client = TestClient(app)
        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertNotEqual(response.content, before)
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')
        cached = client.get(f"/assets/{asset.filename}", headers={"If-None-Match": f'"{asset.patched_sha256}"'})
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(cached.content, b"")
        self.assertEqual(cached.headers["etag"], f'"{asset.patched_sha256}"')
        with patch.object(module, "DATAFRAME_ORIGINAL_SHA256", "0" * 64):
            with self.assertRaises(module.GradioFrontendCompatibilityError):
                module.build_patched_dataframe_asset()

    def test_webui_validates_patch_before_starting_gradio_listener(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertLess(
            source.index("patched_tabs = build_patched_tabs_asset()"),
            source.index("app, local_url, share_url = shared.demo.launch("),
        )

    def test_exact_pinned_asset_is_patched_without_modifying_site_packages(self):
        asset_path = (
            Path(gradio.__file__).resolve().parent
            / "templates"
            / "frontend"
            / "assets"
            / gradio_frontend_compat.TABS_ASSET_NAME
        )
        before = asset_path.read_bytes()

        patched = gradio_frontend_compat.build_patched_tabs_asset()

        self.assertEqual(hashlib.sha256(before).hexdigest(), patched.original_sha256)
        self.assertEqual(before, asset_path.read_bytes())
        self.assertNotEqual(before, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_OVERFLOW_FUNCTION, patched.content)
        self.assertIn(gradio_frontend_compat._NO_OVERFLOW_FUNCTION, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_INITIAL_TAB_SYNC, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_INITIAL_TAB_SYNC, patched.content)
        self.assertNotIn(gradio_frontend_compat._ORIGINAL_TAB_REGISTRATION, patched.content)
        self.assertIn(gradio_frontend_compat._STATIC_TAB_REGISTRATION, patched.content)
        self.assertEqual(hashlib.sha256(patched.content).hexdigest(), patched.patched_sha256)
        self.assertEqual(patched.patched_sha256, gradio_frontend_compat.PATCHED_ASSET_SHA256)

    def test_preconfigured_route_is_available_during_gradio_auto_launch(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        prepared_app = gradio_frontend_compat.create_gradio_compatibility_app(asset)
        self.assertIs(prepared_app.router.default_response_class, ORJSONResponse)
        install_gradio_file_url_guard(prepared_app)
        fetched_during_open: list[bytes] = []
        external_status_during_open: list[tuple[int, str | None]] = []

        def fetch_from_browser_open(local_url: str) -> bool:
            with urlopen(urljoin(local_url, f"assets/{asset.filename}"), timeout=5) as response:  # noqa: S310
                fetched_during_open.append(response.read())
            external = httpx.get(
                local_url.rstrip("/") + "/gradio_api/file=https://redirect.invalid/pre-listener",
                follow_redirects=False,
                timeout=5,
            )
            external_status_during_open.append((external.status_code, external.headers.get("location")))
            return True

        with gradio.Blocks(analytics_enabled=False) as demo:
            gradio.Markdown("pre-listener compatibility route")

        try:
            with patch("webbrowser.open", side_effect=fetch_from_browser_open):
                app, _, _ = demo.launch(
                    server_name="127.0.0.1",
                    server_port=port,
                    prevent_thread_lock=True,
                    quiet=True,
                    inbrowser=True,
                    _app=prepared_app,
                )
        finally:
            demo.close()

        self.assertIs(app, prepared_app)
        self.assertEqual(fetched_during_open, [asset.content])
        self.assertEqual(external_status_during_open, [(403, None)])

    def test_webui_opens_browser_only_after_routes_and_middleware_are_ready(self):
        root = Path(__file__).resolve().parents[2]
        source = (root / "webui.py").read_text(encoding="utf-8")

        self.assertIn("inbrowser=False", source)
        self.assertIn("debug=False", source)
        self.assertIn("show_error=cmd_opts.gradio_debug", source)
        self.assertIn("ssr_mode=False", source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = "0"', source)
        self.assertIn('os.environ["GRADIO_DEBUG"] = previous_gradio_debug', source)
        self.assertIn("tunnel_baseline = gradio_runtime.tunnel_snapshot()", source)
        self.assertNotIn("shared.demo.close()", source)
        self.assertGreaterEqual(source.count("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)"), 5)
        launch_block = source.split("app, local_url, share_url = shared.demo.launch(", 1)[1].split(
            'startup_timer.record("gradio launch")',
            1,
        )[0]
        self.assertIn("except Exception:", launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", launch_block)
        self.assertLess(
            source.index("install_gradio_file_url_guard(prepared_app)"), source.index("shared.demo.launch(")
        )
        self.assertLess(source.index("ui.setup_ui_api(app)"), source.index("webbrowser.open(browser_url)"))
        self.assertLess(
            source.index('with startup_timer.subcategory("app_started_callback")'),
            source.index("webbrowser.open(browser_url)"),
        )
        post_launch_block = source.split('startup_timer.record("gradio launch")', 1)[1].split(
            "server_command = shared.state.wait_for_server_command",
            1,
        )[0]
        self.assertIn("initialize_util.setup_middleware(app)", post_launch_block)
        self.assertIn("ui.setup_ui_api(app)", post_launch_block)
        self.assertIn("webbrowser.open(browser_url)", post_launch_block)
        self.assertIn("except Exception:", post_launch_block)
        self.assertIn("gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)", post_launch_block)

    def test_version_mismatch_fails_closed(self):
        with patch.object(gradio, "__version__", "6.18.0"):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "not supported",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_hash_mismatch_fails_closed(self):
        with patch.object(gradio_frontend_compat, "ORIGINAL_ASSET_SHA256", "0" * 64):
            with self.assertRaisesRegex(
                gradio_frontend_compat.GradioFrontendCompatibilityError,
                "does not match",
            ):
                gradio_frontend_compat.build_patched_tabs_asset()

    def test_tampered_prebuilt_asset_fails_closed(self):
        asset = gradio_frontend_compat.build_patched_tabs_asset()
        tampered = gradio_frontend_compat.PatchedFrontendAsset(
            filename=asset.filename,
            content=asset.content + b" ",
            original_sha256=asset.original_sha256,
            patched_sha256=asset.patched_sha256,
        )

        with self.assertRaisesRegex(
            gradio_frontend_compat.GradioFrontendCompatibilityError,
            "integrity validation",
        ):
            gradio_frontend_compat.install_gradio_tabs_compatibility_route(
                FastAPI(),
                tampered,
            )

    def test_exact_route_precedes_generic_assets_and_supports_etag(self):
        app = FastAPI()

        @app.get("/assets/{path:path}")
        async def generic_asset(path: str):
            return {"generic": path}

        asset = gradio_frontend_compat.build_patched_tabs_asset()
        installed = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app, asset)
        self.assertIs(installed, asset)
        client = TestClient(app)

        response = client.get(f"/assets/{asset.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, asset.content)
        self.assertEqual(response.headers["content-length"], str(len(asset.content)))
        self.assertEqual(response.headers["etag"], f'"{asset.patched_sha256}"')
        self.assertEqual(response.headers["x-aikimi-gradio-compat"], "6.17.3")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")

        head = client.head(f"/assets/{asset.filename}")
        self.assertEqual(head.status_code, 200)
        self.assertEqual(head.content, b"")
        self.assertEqual(head.headers["content-length"], str(len(asset.content)))
        self.assertEqual(head.headers["etag"], f'"{asset.patched_sha256}"')

        cached = client.get(
            f"/assets/{asset.filename}",
            headers={"If-None-Match": f'"{asset.patched_sha256}"'},
        )
        self.assertEqual(cached.status_code, 304)
        self.assertEqual(client.get("/assets/other.js").json(), {"generic": "other.js"})

    def test_route_is_idempotent_and_works_with_asgi_root_path(self):
        app = FastAPI(root_path="/aikimi")
        app.router.routes.insert(0, object())
        first = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)
        second = gradio_frontend_compat.install_gradio_tabs_compatibility_route(app)

        self.assertIs(first, second)
        self.assertEqual(
            sum(getattr(route, "name", None) == gradio_frontend_compat.PATCH_ROUTE_NAME for route in app.router.routes),
            1,
        )
        response = TestClient(app, root_path="/aikimi").get(f"/assets/{first.filename}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, first.content)


if __name__ == "__main__":
    unittest.main()
