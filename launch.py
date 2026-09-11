from modules import launch_utils

args = launch_utils.args
python = launch_utils.python
git = launch_utils.git
index_url = launch_utils.index_url
dir_repos = launch_utils.dir_repos

if args.uv or args.uv_symlink or args.uv_local_cache:
    from modules_forge.uv_hook import patch

    patch(not args.uv, args.uv_local_cache)

git_tag = launch_utils.git_tag

run = launch_utils.run
is_installed = launch_utils.is_installed
repo_dir = launch_utils.repo_dir

run_pip = launch_utils.run_pip
check_run_python = launch_utils.check_run_python
git_clone = launch_utils.git_clone
git_pull_recursive = launch_utils.git_pull_recursive
list_extensions = launch_utils.list_extensions
run_extension_installer = launch_utils.run_extension_installer
prepare_environment = launch_utils.prepare_environment
start = launch_utils.start


def main():
    import os

    from modules.aikimi_security.auth import AuthenticationConfigError, validate_auth_configuration
    from modules.aikimi_security.paths import UnsafeAllowedPathError, build_gradio_allowed_paths
    from modules.aikimi_security.remote_access import RemoteAccessError, validate_remote_access
    from modules.paths_internal import data_path, script_path

    try:
        validate_remote_access(args)
        validate_auth_configuration(args)
        build_gradio_allowed_paths(
            script_path,
            data_path,
            requested_paths=getattr(args, "gradio_allowed_path", ()),
        )
    except (AuthenticationConfigError, RemoteAccessError, UnsafeAllowedPathError) as exc:
        raise SystemExit(f"Aikimi Studio Neo launch policy error: {exc}") from exc

    # Framework environment variables must not silently widen the reviewed CLI
    # policy. Gradio temporary files stay under the managed data tmp directory.
    for variable in (
        "GRADIO_ALLOWED_PATHS",
        "GRADIO_BLOCKED_PATHS",
        "GRADIO_SERVER_NAME",
        "GRADIO_SHARE",
    ):
        os.environ.pop(variable, None)
    managed_gradio_temp = os.path.join(data_path, "tmp", "gradio")
    try:
        os.makedirs(managed_gradio_temp, exist_ok=True)
    except OSError as exc:
        raise SystemExit("Aikimi Studio Neo could not prepare its managed temporary directory.") from exc
    os.environ["GRADIO_TEMP_DIR"] = managed_gradio_temp

    if args.dump_sysinfo:
        filename = launch_utils.dump_sysinfo()

        print(f"Sysinfo saved as {filename}. Exiting...")

        exit(0)

    launch_utils.verify_version()

    launch_utils.startup_timer.record("initial startup")

    with launch_utils.startup_timer.subcategory("prepare environment"):
        if not args.skip_prepare_environment:
            prepare_environment()

    if args.forge_ref_a1111_home:
        launch_utils.configure_a1111_reference(args.forge_ref_a1111_home)
    if args.forge_ref_comfy_home:
        launch_utils.configure_comfy_reference(args.forge_ref_comfy_home)
    if args.forge_ref_comfy_yaml:
        launch_utils.configure_comfy_yaml(args.forge_ref_comfy_yaml)

    start()


if __name__ == "__main__":
    main()
