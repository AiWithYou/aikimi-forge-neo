import shlex
import subprocess
from copy import copy
from functools import wraps

PIP_MODULE_NAMES = frozenset({"pip", "pip.__main__"})
BAD_PIP_FLAGS = ("--prefer-binary", "--ignore-installed", "-I")


def _pre_check():
    try:
        subprocess.run(["uv", "--help"], capture_output=True)
    except FileNotFoundError:
        print("\n[Error] uv is not installed...")
    except Exception:
        print("\n[Error] Failed to access uv...")
    else:
        return

    input("Press Enter to Continue...")
    raise SystemExit


def _set_cache():
    import os

    webui = os.path.dirname(os.path.dirname(__file__))
    cache = os.path.normpath(os.path.join(webui, ".uv-cache"))

    if not os.path.exists(cache):
        print("[uv] Creating .uv-cache folder...")
        os.makedirs(cache)

    os.environ.setdefault("UV_CACHE_DIR", cache)


def _has_python_option(arguments: list[str]) -> bool:
    return any(argument == "--python" or argument.startswith("--python=") for argument in arguments)


def _interpreter_from_command(command: list[str]) -> str | None:
    """Return the interpreter from a `python -m pip ...` invocation, if present."""
    if len(command) < 3 or command[1] != "-m" or command[2] not in PIP_MODULE_NAMES:
        return None
    interpreter = command[0].strip()
    return interpreter or None


def _pip_index(command: list[str]) -> int | None:
    for index, argument in enumerate(command):
        if argument in PIP_MODULE_NAMES:
            return index
    return None


def rewrite_pip_to_uv(command: list[str], *, symlink: bool = False) -> list[str] | None:
    """Rewrite a pip invocation to uv.

    Returns None when the command is not a pip invocation. When the original
    command used `python -m pip ...`, the interpreter is preserved with
    `--python` so uv does not fall back to PATH discovery (which can pick up
    broken or unrelated Python shims).
    """
    pip_index = _pip_index(command)
    if pip_index is None:
        return None

    interpreter = _interpreter_from_command(command)
    pip_args = [arg for arg in command[pip_index + 1 :] if arg not in BAD_PIP_FLAGS]

    modified: list[str] = ["uv", "pip"]
    if pip_args and not pip_args[0].startswith("-"):
        modified.append(pip_args[0])
        pip_args = pip_args[1:]

    if interpreter and not _has_python_option(pip_args):
        modified.extend(["--python", interpreter])

    modified.extend(pip_args)
    if symlink:
        modified.extend(["--link-mode", "symlink"])
    return modified


def patch(symlink: bool, local: bool):
    if hasattr(subprocess, "__original_run"):
        return

    _pre_check()

    if local:
        _set_cache()

    subprocess.__original_run = subprocess.run

    @wraps(subprocess.__original_run)
    def patched_run(*args, **kwargs):
        _original_args = copy(args)
        _original_kwargs = copy(kwargs)

        if args:
            command, *_args = args
        else:
            command, _args = kwargs.pop("args", ""), ()

        if isinstance(command, str):
            command = shlex.split(command)
        else:
            command = [arg.strip() for arg in command]

        assert isinstance(command, list)

        modified_command = rewrite_pip_to_uv(command, symlink=symlink)
        if modified_command is None:
            return subprocess.__original_run(*_original_args, **_original_kwargs)

        command = [*modified_command, *_args]
        if kwargs.get("shell", False):
            command = shlex.join(command).replace("'", '"')

        return subprocess.__original_run(command, **kwargs)

    subprocess.run = patched_run
