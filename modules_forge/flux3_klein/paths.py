"""Normalize real aliases without following redirecting managed-path entries."""

import stat
from pathlib import Path


def canonical_path(value):
    path = Path(value).absolute()
    # Inspect before realpath expands Windows 8.3 names or follows links.
    # Name-surrogate reparse tags include junctions and Windows symlinks,
    # while nonredirecting tags (for example cloud metadata) stay allowed.
    for entry in (path, *path.parents):
        try:
            attributes = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(attributes.st_mode) or getattr(attributes, "st_reparse_tag", 0) & 0x20000000:
            raise ValueError("linked paths are not allowed for FLUX 3 managed files")
    return path.resolve(strict=False)
