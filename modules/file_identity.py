"""Cache stamps for files and directory callers, including Windows ChangeTime."""

import os
import stat


def cache_file_identity(filename):
    info = os.stat(filename)
    size, modified, changed, inode = info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino
    if os.name == "nt" and stat.S_ISREG(info.st_mode):
        from modules_forge.local_assets import file_version

        try:
            size, modified, changed, inode = file_version(filename)
        except OSError:
            changed = None
    # The seventh field invalidates old six-field entries. None forbids caching.
    return (
        os.path.normcase(os.path.abspath(filename)),
        modified,
        size,
        info.st_ctime_ns,
        inode,
        info.st_dev,
        changed,
    )
