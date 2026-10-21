"""Repackage a verified upstream wheel with Pillow 13-compatible metadata."""

import base64
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
UPSTREAM = json.loads((ROOT / "upstream.json").read_text(encoding="utf-8"))
DIST_INFO = f"gradio-{UPSTREAM['version']}.dist-info"
WHEEL_NAME = f"gradio-{UPSTREAM['version']}-py3-none-any.whl"
OLD_DIST_INFO = f"gradio-{UPSTREAM['upstream_version']}.dist-info"


def patched_metadata():
    source = (ROOT / "UPSTREAM-METADATA").read_bytes()
    if hashlib.sha256(source).hexdigest() != UPSTREAM["metadata_sha256"]:
        raise ValueError("Upstream Gradio metadata identity mismatch")
    old_version = f"Version: {UPSTREAM['upstream_version']}\n".encode()
    old_pillow = b"Requires-Dist: pillow<13.0,>=8.0\n"
    if source.count(old_version) != 1 or source.count(old_pillow) != 1:
        raise ValueError("Unexpected upstream Gradio requirements")
    return source.replace(old_version, f"Version: {UPSTREAM['version']}\n".encode()).replace(
        old_pillow, b"Requires-Dist: pillow<14.0,>=8.0\n"
    )


def get_requires_for_build_wheel(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    target = Path(metadata_directory) / DIST_INFO
    (target / "licenses").mkdir(parents=True, exist_ok=True)
    (target / "METADATA").write_bytes(patched_metadata())
    (target / "WHEEL").write_bytes((ROOT / "UPSTREAM-WHEEL").read_bytes())
    (target / "licenses/LICENSE").write_bytes((ROOT / "LICENSE").read_bytes())
    return DIST_INFO


def download_upstream():
    # The immutable PyPI artifact and expected hash are committed together.
    with urllib.request.urlopen(UPSTREAM["url"], timeout=60) as response:  # noqa: S310 -- Fixed HTTPS PyPI artifact.
        return response.read(UPSTREAM["size"] + 1)


def build_from_upstream(source, wheel_directory):
    if len(source) != UPSTREAM["size"] or hashlib.sha256(source).hexdigest() != UPSTREAM["sha256"]:
        raise ValueError("Official Gradio wheel size or SHA-256 mismatch")
    metadata = patched_metadata()
    with zipfile.ZipFile(io.BytesIO(source)) as upstream:
        if upstream.read(f"{OLD_DIST_INFO}/METADATA") != (ROOT / "UPSTREAM-METADATA").read_bytes():
            raise ValueError("Official Gradio wheel metadata mismatch")
        if upstream.read(f"{OLD_DIST_INFO}/WHEEL") != (ROOT / "UPSTREAM-WHEEL").read_bytes():
            raise ValueError("Official Gradio wheel tags mismatch")
        entries = []
        for info in upstream.infolist():
            name = info.filename
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe upstream wheel member")
            if info.is_dir() or name == f"{OLD_DIST_INFO}/RECORD":
                continue
            target = name.replace(f"{OLD_DIST_INFO}/", f"{DIST_INFO}/", 1)
            content = metadata if name == f"{OLD_DIST_INFO}/METADATA" else upstream.read(info)
            entries.append((target, content))

    output = Path(wheel_directory)
    output.mkdir(parents=True, exist_ok=True)
    records = io.StringIO(newline="")
    writer = csv.writer(records, lineterminator="\n")
    with zipfile.ZipFile(output / WHEEL_NAME, "w", compression=zipfile.ZIP_DEFLATED) as wheel:
        for name, content in entries:
            wheel.writestr(name, content)
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode("ascii")
            writer.writerow((name, f"sha256={digest}", len(content)))
        record_name = f"{DIST_INFO}/RECORD"
        writer.writerow((record_name, "", ""))
        wheel.writestr(record_name, records.getvalue().encode("utf-8"))
    return WHEEL_NAME


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return build_from_upstream(download_upstream(), wheel_directory)
