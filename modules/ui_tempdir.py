import os
import tempfile
import time
from collections import namedtuple
from contextlib import suppress
from pathlib import Path

import gradio as gr
import gradio.processing_utils
import gradio.utils
from PIL import Image, PngImagePlugin

Savedfile = namedtuple("Savedfile", ["name"])
MANAGED_TEMP_PREFIX = "aikimi-gradio-"
shared = None


def _shared_module():
    global shared
    if shared is None:
        from modules import shared as shared_module

        shared = shared_module
    return shared


def register_tmp_file(gradio_app: gr.Blocks, filename: os.PathLike):
    filename = gradio.utils.abspath(filename)
    # Preserve the shared registry and avoid copying every previously saved path.
    gradio_app.temp_file_sets[0].add(filename)


def check_tmp_file(gradio_app: gr.Blocks, filename: os.PathLike) -> bool:
    filename = gradio.utils.abspath(filename)
    return any(filename in fileset for fileset in gradio_app.temp_file_sets)


def save_pil_to_file(
    pil_image: Image.Image,
    cache_dir: os.PathLike = None,
    name: str = "image",
    format: str = "png",
) -> str:
    shared_module = _shared_module()
    already_saved_as = getattr(pil_image, "already_saved_as", None)
    if already_saved_as and os.path.isfile(already_saved_as):
        register_tmp_file(shared_module.demo, already_saved_as)
        # Gradio 6 は実在するパスを開いて内容ハッシュを計算する。
        # URL用のクエリをファイル名へ付加するとWindowsで読込に失敗する。
        return os.fspath(already_saved_as)

    directory = shared_module.opts.temp_dir or cache_dir
    os.makedirs(directory, exist_ok=True)

    normalized_format = str(format or "png").lower()
    save_format = "JPEG" if normalized_format in {"jpg", "jpeg"} else normalized_format.upper()
    save_kwargs = {"format": save_format}
    if normalized_format == "png":
        metadata = PngImagePlugin.PngInfo()
        use_metadata = False
        for key, value in pil_image.info.items():
            if isinstance(key, str) and isinstance(value, str):
                metadata.add_text(key, value)
                use_metadata = True
        if use_metadata:
            save_kwargs["pnginfo"] = metadata

    suffix = f".{normalized_format.replace('jpeg', 'jpg')}"
    file_obj = tempfile.NamedTemporaryFile(
        delete=False,
        prefix=MANAGED_TEMP_PREFIX,
        suffix=suffix,
        dir=directory,
    )
    file_obj.close()
    try:
        pil_image.save(file_obj.name, **save_kwargs)
    except BaseException:
        # Remove only this failed output; cleanup must not mask the save error.
        with suppress(OSError):
            os.unlink(file_obj.name)
        raise
    return file_obj.name


def install_ui_tempdir_override():
    """Preserve PNG metadata without replacing Gradio's path validator."""

    gradio.processing_utils.save_pil_to_cache = save_pil_to_file
    # Keep Gradio's maintained async file validator. Replacing it with a copied
    # implementation can silently bypass upstream upload/path hardening.


def on_tmpdir_changed():
    shared_module = _shared_module()
    if shared_module.opts.temp_dir == "" or shared_module.demo is None:
        return

    os.makedirs(shared_module.opts.temp_dir, exist_ok=True)

    register_tmp_file(shared_module.demo, os.path.join(shared_module.opts.temp_dir, "x"))


def cleanup_tmpdr():
    temp_dir = _shared_module().opts.temp_dir
    if temp_dir == "" or not os.path.isdir(temp_dir):
        return

    managed_root = Path(temp_dir).resolve(strict=False)
    for root, _, files in os.walk(managed_root, topdown=False, followlinks=False):
        for name in files:
            if not name.startswith(MANAGED_TEMP_PREFIX):
                continue

            filename = (Path(root) / name).resolve(strict=False)
            try:
                filename.relative_to(managed_root)
            except ValueError:
                continue
            for attempt in range(3):
                try:
                    filename.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 2:
                        break
                    time.sleep(0.05 * (attempt + 1))


def is_gradio_temp_path(path: str) -> bool:
    """Check if the path is a temp dir used by gradio"""
    path = Path(path)
    shared_module = _shared_module()
    if shared_module.opts.temp_dir and path.is_relative_to(shared_module.opts.temp_dir):
        return True
    if gradio_temp_dir := os.environ.get("GRADIO_TEMP_DIR"):
        if path.is_relative_to(gradio_temp_dir):
            return True
    if path.is_relative_to(Path(tempfile.gettempdir()) / "gradio"):
        return True
    return False
