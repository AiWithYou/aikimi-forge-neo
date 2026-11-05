"""Give pinned uv wheels version-specific, validated bootstrap paths."""

import re
from dataclasses import replace
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from tools.aikimi_setup import SetupError


def versioned_uv_spec(spec):
    parsed = urlsplit(spec.url)
    filename = PurePosixPath(parsed.path).name
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.pythonhosted.org"
        or parsed.query
        or parsed.fragment
        or not re.fullmatch(r"uv-\d+\.\d+\.\d+-py3-none-win_amd64\.whl", filename)
        or spec.relative_path not in {"bootstrap/uv.whl", f"bootstrap/{filename}"}
    ):
        raise SetupError("固定版uvの取得先または保存先が不正です。")
    return replace(spec, relative_path=f"bootstrap/{filename}")
