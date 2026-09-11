import hashlib
import os.path
import threading
from contextlib import contextmanager

import modules.cache
from modules import shared

dump_cache = modules.cache.dump_cache
cache = modules.cache.cache

_hash_locks_guard = threading.Lock()
_hash_locks = {}


@contextmanager
def _hash_request_lock(filename, title, use_addnet_hash):
    """同一要求の読み取りを直列化し、待機者がいなくなったロックは解放する。"""
    key = (os.path.normcase(os.path.abspath(filename)), title, bool(use_addnet_hash))
    with _hash_locks_guard:
        entry = _hash_locks.get(key)
        if entry is None:
            entry = [threading.Lock(), 0]
            _hash_locks[key] = entry
        entry[1] += 1
    try:
        with entry[0]:
            yield
    finally:
        with _hash_locks_guard:
            entry[1] -= 1
            if entry[1] == 0:
                del _hash_locks[key]


def calculate_sha256_real(filename: os.PathLike):
    with open(filename, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


calculate_sha256 = calculate_sha256_real


def _hash_file_identity(filename):
    stat = os.stat(filename)
    return (
        os.path.normcase(os.path.abspath(filename)),
        stat.st_mtime_ns,
        stat.st_size,
        stat.st_ctime_ns,
        stat.st_ino,
        stat.st_dev,
    )


def sha256_from_cache(filename: os.PathLike, title: str, use_addnet_hash=False):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")
    try:
        file_identity = _hash_file_identity(filename)
    except FileNotFoundError:
        return None

    # One database read; legacy entries are refreshed rather than trusted.
    entry = hashes.get(title)
    if not entry or entry.get("file_identity") != file_identity:
        return None

    return entry.get("sha256")


def sha256(filename: os.PathLike, title: str, use_addnet_hash=False):
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None or shared.cmd_opts.no_hashing:
        return sha256_value

    with _hash_request_lock(filename, title, use_addnet_hash):
        return _sha256_locked(filename, title, use_addnet_hash)


def _sha256_locked(filename, title, use_addnet_hash):
    hashes = cache("hashes-addnet") if use_addnet_hash else cache("hashes")

    # 待機中に完了した計算を再利用する。ファイルの差し替えも再確認する。
    sha256_value = sha256_from_cache(filename, title, use_addnet_hash)
    if sha256_value is not None:
        return sha256_value

    if shared.cmd_opts.no_hashing:
        return None

    file_identity = _hash_file_identity(filename)
    print(f"Calculating sha256 for {filename}: ", end="", flush=True)
    if use_addnet_hash:
        with open(filename, "rb") as file:
            sha256_value = addnet_hash_safetensors(file)
    else:
        sha256_value = calculate_sha256_real(filename)
    if _hash_file_identity(filename) != file_identity:
        raise RuntimeError(f"File changed while calculating sha256: {filename}")
    print(sha256_value)

    hashes[title] = {
        "mtime": file_identity[1] / 1_000_000_000,
        "file_identity": file_identity,
        "sha256": sha256_value,
    }

    dump_cache()

    return sha256_value


def addnet_hash_safetensors(b):
    """kohya-ss hash for safetensors from https://github.com/kohya-ss/sd-scripts/blob/main/library/train_util.py"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()
