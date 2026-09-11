if __name__ == "__main__":
    raise SystemError("Call launch.py instead")


import os
import time
import webbrowser
from threading import Thread

from fastapi import Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from modules import gradio_runtime, initialize, initialize_util, timer
from modules.aikimi_security.auth import install_remote_auth_middleware
from modules.aikimi_security.gradio_file_guard import install_gradio_file_url_guard
from modules.aikimi_security.paths import build_gradio_allowed_paths, build_gradio_blocked_paths
from modules.aikimi_security.redaction import safe_error_message
from modules_forge.initialization import initialize_forge

startup_timer = timer.startup_timer
startup_timer.record("launcher")

initialize.shush()

with startup_timer.subcategory("forge init"):
    initialize_forge()

initialize.imports()

initialize.check_versions()

initialize.initialize()


def _handle_exception(request: Request, e: Exception):
    error_information = vars(e)
    content = {
        "error": type(e).__name__,
        "detail": safe_error_message(error_information.get("detail", "")),
        "message": safe_error_message(e),
    }
    return JSONResponse(status_code=int(error_information.get("status_code", 500)), content=jsonable_encoder(content))


def create_api(app):
    from modules.api.api import Api
    from modules.call_queue import queue_lock

    api = Api(app, queue_lock)
    return api


def api_only_worker():
    from fastapi import FastAPI

    from modules.shared_cmd_options import cmd_opts

    app = FastAPI(exception_handlers={Exception: _handle_exception})
    initialize_util.setup_middleware(app)
    install_remote_auth_middleware(app, cmd_opts)
    api = create_api(app)

    from modules import script_callbacks

    script_callbacks.before_ui_callback()
    script_callbacks.app_started_callback(None, app)

    print(f"Startup time: {startup_timer.summary()}.")
    api.launch(server_name=initialize_util.gradio_server_name(), port=cmd_opts.port if cmd_opts.port else 7861, root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "")


def webui_worker():
    from modules.shared_cmd_options import cmd_opts

    launch_api = cmd_opts.api

    from modules import (
        progress,
        script_callbacks,
        scripts,
        shared,
        ui,
        ui_extra_networks,
        ui_tempdir,
    )

    while 1:
        if shared.opts.clean_temp_dir_at_start:
            ui_tempdir.cleanup_tmpdr()
            startup_timer.record("cleanup temp dir")

        script_callbacks.before_ui_callback()
        startup_timer.record("scripts before_ui_callback")

        shared.demo = ui.create_ui()
        startup_timer.record("create ui")

        if not cmd_opts.no_gradio_queue:
            shared.demo.queue(default_concurrency_limit=32)

        gradio_auth_creds = list(initialize_util.get_gradio_auth_creds()) or None

        auto_launch_browser = False
        if os.getenv("SD_WEBUI_RESTARTING") != "1":
            if shared.opts.auto_launch_browser == "Remote" or cmd_opts.autolaunch:
                auto_launch_browser = True
            elif shared.opts.auto_launch_browser == "Local":
                auto_launch_browser = not cmd_opts.webui_is_non_local

        from modules_forge.forge_canvas.canvas import canvas_head, canvas_js_root_path

        javascript_paths = [
            script.path
            for extension in (".js", ".mjs")
            for script in scripts.list_scripts("javascript", extension)
        ]
        stylesheet_paths = scripts.list_files_with_name("style.css")
        notification_audio = (
            os.path.join(initialize_util.script_path, "notification.mp3")
            if shared.opts.notification_audio
            else None
        )

        allowed_paths = build_gradio_allowed_paths(
            initialize_util.script_path,
            initialize_util.data_path,
            canvas_root=canvas_js_root_path,
            javascript_paths=javascript_paths,
            stylesheet_paths=stylesheet_paths,
            notification_audio=notification_audio,
            requested_paths=cmd_opts.gradio_allowed_path,
        )
        blocked_paths = build_gradio_blocked_paths(
            initialize_util.script_path, initialize_util.data_path
        )

        from modules.gradio_frontend_compat import (
            GradioFrontendCompatibilityError,
            build_patched_tabs_asset,
            create_gradio_compatibility_app,
        )

        # Validate the exact third-party asset before Gradio starts listening.
        # A version/hash mismatch must fail closed instead of exposing the known
        # Tabs mount storm through the generic assets route.
        patched_tabs = build_patched_tabs_asset()
        app_kwargs = {
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "exception_handlers": {Exception: _handle_exception},
        }
        prepared_app = create_gradio_compatibility_app(
            patched_tabs,
            app_kwargs=app_kwargs,
            debug=cmd_opts.gradio_debug,
        )
        install_gradio_file_url_guard(prepared_app)
        install_remote_auth_middleware(prepared_app, cmd_opts)
        tunnel_baseline = gradio_runtime.tunnel_snapshot()
        aikimi_favicon = os.path.join(os.path.dirname(__file__), "assets", "aikimi", "favicon.png")

        previous_gradio_debug = os.environ.get("GRADIO_DEBUG")
        os.environ["GRADIO_DEBUG"] = "0"
        try:
            try:
                app, local_url, share_url = shared.demo.launch(
                    share=cmd_opts.share,
                    server_name=initialize_util.gradio_server_name(),
                    server_port=cmd_opts.port,
                    ssl_keyfile=cmd_opts.tls_keyfile,
                    ssl_certfile=cmd_opts.tls_certfile,
                    ssl_verify=cmd_opts.disable_tls_verify,
                    debug=False,
                    auth=gradio_auth_creds,
                    inbrowser=False,
                    show_error=cmd_opts.gradio_debug,
                    prevent_thread_lock=True,
                    favicon_path=aikimi_favicon if os.path.isfile(aikimi_favicon) else None,
                    allowed_paths=allowed_paths,
                    blocked_paths=blocked_paths,
                    app_kwargs=app_kwargs,
                    root_path=f"/{cmd_opts.subpath}" if cmd_opts.subpath else "",
                    # Gradio 6.17.3 does not stop its Node SSR proxy from Blocks.close().
                    # Forge owns the Python listener lifecycle, so keep one process tree.
                    ssr_mode=False,
                    theme=shared.gradio_theme,
                    head=canvas_head,
                    _app=prepared_app,
                )
            except Exception:
                gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)
                raise
        finally:
            if previous_gradio_debug is None:
                os.environ.pop("GRADIO_DEBUG", None)
            else:
                os.environ["GRADIO_DEBUG"] = previous_gradio_debug

        try:
            startup_timer.record("gradio launch")
            if app is not prepared_app:
                raise GradioFrontendCompatibilityError(
                    "Gradio did not preserve the preconfigured compatibility app; the server was stopped."
                )
            print(
                "Aikimi Gradio compatibility: serving audited tabs patch "
                f"{patched_tabs.patched_sha256[:12]}."
            )

            # gradio uses a very open CORS policy via app.user_middleware, which makes it possible for
            # an attacker to trick the user into opening a malicious HTML page, which makes a request to the
            # running web ui and do whatever the attacker wants, including installing an extension and
            # running its code. We disable this here. Suggested by RyotaK.
            app.user_middleware = [
                x for x in app.user_middleware if "cors" not in x.cls.__name__.casefold()
            ]

            initialize_util.setup_middleware(app)
            install_remote_auth_middleware(app, cmd_opts)

            progress.setup_progress_api(app)
            ui.setup_ui_api(app)

            if launch_api:
                create_api(app)

            ui_extra_networks.add_pages_to_demo(app)

            startup_timer.record("add APIs")

            with startup_timer.subcategory("app_started_callback"):
                script_callbacks.app_started_callback(shared.demo, app)

            if auto_launch_browser:
                browser_url = share_url if cmd_opts.share and share_url else local_url
                webbrowser.open(browser_url)

            timer.startup_record = startup_timer.dump()
            print(f"Startup time: {startup_timer.summary()}.")
        except Exception:
            gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)
            raise

        try:
            while True:
                server_command = shared.state.wait_for_server_command(timeout=5)
                if server_command:
                    if server_command in ("stop", "restart"):
                        break
                    else:
                        print(f"Unknown server command: {server_command}")
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, stopping...")
            server_command = "stop"
        except Exception:
            gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)
            raise

        if server_command == "stop":
            print("Stopping server...")
            # If we catch a keyboard interrupt, we want to stop the server and exit.
            gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)
            break

        # disable auto launch webui in browser for subsequent UI Reload
        os.environ.setdefault("SD_WEBUI_RESTARTING", "1")

        print("Restarting UI...")
        gradio_runtime.close_gradio_runtime(shared.demo, tunnel_baseline)
        time.sleep(0.5)
        startup_timer.reset()
        script_callbacks.app_reload_callback()
        startup_timer.record("app reload callback")
        script_callbacks.script_unloaded_callback()
        startup_timer.record("scripts unloaded callback")
        initialize.initialize_rest(reload_script_modules=True)


def api_only():
    Thread(target=api_only_worker, daemon=True).start()


def webui():
    Thread(target=webui_worker, daemon=True).start()
