import os

import launch
from modules import cmd_args, script_loading
from modules.paths_internal import data_path, extensions_builtin_dir, extensions_dir, models_path, script_path  # noqa: F401

parser = cmd_args.parser

script_loading.preload_extensions(extensions_dir, parser, extension_list=launch.list_extensions(launch.args.ui_settings_file))
script_loading.preload_extensions(extensions_builtin_dir, parser)

if os.environ.get("IGNORE_CMD_ARGS_ERRORS", None) is None:
    cmd_opts = parser.parse_args()
else:
    cmd_opts, _ = parser.parse_known_args()

from modules.aikimi_security.auth import AuthenticationConfigError, validate_auth_configuration
from modules.aikimi_security.remote_access import RemoteAccessError, exposure_reasons, validate_remote_access

try:
    remote_reasons = validate_remote_access(cmd_opts)
    validate_auth_configuration(cmd_opts)
except (AuthenticationConfigError, RemoteAccessError) as exc:
    parser.error(str(exc))

cmd_opts.webui_is_non_local = bool(exposure_reasons(cmd_opts))
cmd_opts.disable_extension_access = cmd_opts.webui_is_non_local and not cmd_opts.enable_insecure_extension_access

if remote_reasons:
    print(
        "[Aikimi Studio Neo] WARNING: authenticated remote mode is enabled "
        f"({', '.join(remote_reasons)}). Treat every enabled extension as remotely reachable."
    )
