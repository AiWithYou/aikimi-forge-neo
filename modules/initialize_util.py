import json
import os
import re
import signal
import sys

import starlette

from modules.timer import startup_timer
from modules.paths_internal import data_path, script_path


def gradio_server_name():
    from modules.shared_cmd_options import cmd_opts
    from modules.aikimi_security.remote_access import server_bind_name

    return server_bind_name(cmd_opts)


def fix_torch_version():
    import torch

    # truncate version number of nightly/local build of PyTorch
    if ".dev" in torch.__version__ or "+git" in torch.__version__:
        torch.__long_version__ = torch.__version__
        torch.__version__ = re.search(r"[\d.]+[\d]", torch.__version__).group(0)


def fix_asyncio_event_loop_policy():
    """
    The default `asyncio` event loop policy only automatically creates
    event loops in the main threads. Other threads must create event
    loops explicitly or `asyncio.get_event_loop` and `.IOLoop.current`
    will fail. Installing this policy allows event loops to be created
    automatically on any thread, matching the behavior of Tornado prior to 5.0
    """

    import asyncio

    if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        # "Any thread" and "selector" should be orthogonal, but there's not a clean
        # interface for composing policies so pick the right base.
        _BasePolicy = asyncio.WindowsSelectorEventLoopPolicy  # type: ignore
    else:
        _BasePolicy = asyncio.DefaultEventLoopPolicy

    class AnyThreadEventLoopPolicy(_BasePolicy):  # type: ignore
        """Event loop policy that allows loop creation on any thread.
        Usage::

            asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())
        """

        def get_event_loop(self) -> asyncio.AbstractEventLoop:
            try:
                return super().get_event_loop()
            except (RuntimeError, AssertionError):
                # This was an AssertionError in python 3.4.2 (which ships with debian jessie)
                # and changed to a RuntimeError in 3.4.3.
                # "There is no current event loop in thread %r"
                loop = self.new_event_loop()
                self.set_event_loop(loop)
                return loop

    asyncio.set_event_loop_policy(AnyThreadEventLoopPolicy())


def restore_config_state_file():
    from modules import config_states, shared

    config_state_file = shared.opts.restore_config_state_file
    if config_state_file == "":
        return

    shared.opts.restore_config_state_file = ""
    shared.opts.save(shared.config_filename)

    if os.path.isfile(config_state_file):
        print(f"*** About to restore extension state from file: {config_state_file}")
        with open(config_state_file, "r", encoding="utf-8") as f:
            config_state = json.load(f)
            config_states.restore_extension_config(config_state)
        startup_timer.record("restore extension config")
    elif config_state_file:
        print(f"!!! Config state backup not found: {config_state_file}")


def validate_tls_options():
    from modules.shared_cmd_options import cmd_opts

    if not (cmd_opts.tls_keyfile and cmd_opts.tls_certfile):
        return

    try:
        if not os.path.exists(cmd_opts.tls_keyfile):
            print("Invalid path to TLS keyfile given")
        if not os.path.exists(cmd_opts.tls_certfile):
            print(f"Invalid path to TLS certfile: '{cmd_opts.tls_certfile}'")
    except TypeError:
        cmd_opts.tls_keyfile = cmd_opts.tls_certfile = None
        print("TLS setup invalid, running webui without TLS")
    else:
        print("Running with TLS")
    startup_timer.record("TLS")


def get_gradio_auth_creds():
    """
    Convert the gradio_auth and gradio_auth_path commandline arguments into
    an iterable of (username, password) tuples.
    """
    from modules.shared_cmd_options import cmd_opts
    from modules.aikimi_security.auth import credentials_from_options

    yield from credentials_from_options(cmd_opts, "gradio")


def dumpstacks():
    import threading
    import traceback

    id2name = {th.ident: th.name for th in threading.enumerate()}
    code = []
    for threadId, stack in sys._current_frames().items():
        code.append(f"\n# Thread: {id2name.get(threadId, '')}({threadId})")
        for filename, lineno, name, line in traceback.extract_stack(stack):
            code.append(f"""File: "{filename}", line {lineno}, in {name}""")
            if line:
                code.append("  " + line.strip())

    print("\n".join(code))


def configure_sigint_handler():
    # make the program just exit at Ctrl + C without waiting for anything

    from modules import shared

    def sigint_handler(sig, frame):
        print(f"Interrupted with signal {sig} in {frame}")

        if shared.opts.dump_stacks_on_signal:
            dumpstacks()

        os._exit(0)

    if not os.environ.get("COVERAGE_RUN"):
        # Don't install the immediate-quit handler when running under coverage,
        # as then the coverage report won't be generated.
        signal.signal(signal.SIGINT, sigint_handler)


def reserve_memory():
    from backend.memory_management import set_reserved_memory
    from modules.shared import opts

    set_reserved_memory(opts.setting_allocated_vram)


def clear_references():
    from backend.args import dynamic_args

    dynamic_args.ref_latents.clear()


def migrate_renamed_options() -> bool:
    from modules import shared

    old_key = "klein_no_reference"
    new_key = "klein_do_reference"
    if old_key not in shared.opts.data:
        return False

    old_value = shared.opts.data[old_key]
    new_key_existed = new_key in shared.opts.data
    new_value = shared.opts.data.get(new_key)
    if new_key not in shared.opts.data:
        shared.opts.data[new_key] = not bool(shared.opts.data[old_key])
    del shared.opts.data[old_key]
    if not shared.cmd_opts.freeze_settings:
        try:
            shared.opts.save(shared.config_filename)
        except Exception:
            shared.opts.data[old_key] = old_value
            if new_key_existed:
                shared.opts.data[new_key] = new_value
            else:
                shared.opts.data.pop(new_key, None)
            import logging

            logging.getLogger("startup").exception(
                "Could not save the Klein reference option migration; continuing with the original setting"
            )
            return False
    return True


def configure_opts_onchange():
    from modules import shared, ui_tempdir

    migrate_renamed_options()
    shared.opts.onchange("temp_dir", ui_tempdir.on_tmpdir_changed)
    shared.opts.onchange("gradio_theme", shared.reload_gradio_theme)
    shared.opts.onchange("setting_allocated_vram", reserve_memory)
    shared.opts.onchange("klein_do_reference", clear_references)
    shared.opts.onchange("anima_do_reference", clear_references)
    shared.opts.onchange("krea2_do_reference", clear_references)
    startup_timer.record("opts onchange")


def setup_middleware(app):
    from starlette.middleware.gzip import GZipMiddleware

    from modules.aikimi_security.gradio_file_guard import install_gradio_file_url_guard

    app.user_middleware.insert(0, starlette.middleware.Middleware(GZipMiddleware, minimum_size=1000))
    configure_cors_middleware(app)
    install_gradio_file_url_guard(app)
    app.middleware_stack = app.build_middleware_stack()  # rebuild middleware stack on-the-fly


def configure_cors_middleware(app):
    from starlette.middleware.cors import CORSMiddleware

    from modules.shared_cmd_options import cmd_opts

    cors_options = {
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "allow_credentials": True,
    }
    if cmd_opts.cors_allow_origins:
        cors_options["allow_origins"] = cmd_opts.cors_allow_origins.split(",")
    if cmd_opts.cors_allow_origins_regex:
        cors_options["allow_origin_regex"] = cmd_opts.cors_allow_origins_regex

    app.user_middleware.insert(0, starlette.middleware.Middleware(CORSMiddleware, **cors_options))
