"""Build relative, encoded URLs for files served by Gradio 6."""

from __future__ import annotations

import os
from urllib.parse import quote

from gradio.route_utils import API_PREFIX


def gradio_file_url(path: str | os.PathLike[str], *, cache_key: object | None = None) -> str:
    """Return a mount-relative Gradio file URL without leaking URL delimiters from paths."""

    normalized = os.fspath(path).replace("\\", "/")
    encoded_path = quote(normalized, safe="/:")
    url = f"{API_PREFIX.lstrip('/')}/file={encoded_path}"
    return f"{url}?{quote(str(cache_key), safe='')}" if cache_key is not None else url
