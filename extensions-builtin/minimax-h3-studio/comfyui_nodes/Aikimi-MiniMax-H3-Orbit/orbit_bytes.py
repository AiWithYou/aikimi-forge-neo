"""Read the pinned Orbit file once and validate the exact bytes to deserialize."""

import hashlib
import json
import os
from pathlib import Path

MODEL_NAME = "minimax_h3_flf2v_lora_v1.safetensors"
MODEL_BYTES = 155_111_424
MODEL_SHA256 = "14f13e3effaf3e729fdc0c97680344aa63f473be0c55963f963d718b3db2a4d4"


def _handle_version(stream):
    stat = os.fstat(stream.fileno())
    changed = stat.st_ctime_ns
    if os.name == "nt":
        import ctypes
        import msvcrt

        class FileBasicInfo(ctypes.Structure):
            _fields_ = [(name, ctypes.c_longlong) for name in ("created", "accessed", "written", "changed")] + [
                ("attributes", ctypes.c_ulong)
            ]

        info = FileBasicInfo()
        success = ctypes.windll.kernel32.GetFileInformationByHandleEx(
            ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno())), 0, ctypes.byref(info), ctypes.sizeof(info)
        )
        changed = info.changed if success else None
    return [stat.st_size, stat.st_mtime_ns, changed, stat.st_ino]


def verify_version(path, expected):
    with Path(path).open("rb") as stream:
        if _handle_version(stream) != expected:
            raise ValueError("Orbit LoRAが受付後に変更されました。")
        if expected[2] is None:
            digest = hashlib.sha256()
            received = 0
            while block := stream.read(min(1024 * 1024, MODEL_BYTES + 1 - received)):
                received += len(block)
                digest.update(block)
                if received > MODEL_BYTES:
                    break
            if received != MODEL_BYTES or digest.hexdigest() != MODEL_SHA256 or _handle_version(stream) != expected:
                raise ValueError("Orbit LoRAが受付後に変更されました。")


def read_verified(canonical_path, roots, size, sha256, version):
    path = Path(canonical_path)
    if path.is_symlink() or not path.is_file() or str(path.resolve()) != canonical_path or path.name != MODEL_NAME:
        raise ValueError("Orbit LoRAの実パスが不正です。")
    if not any(path.is_relative_to(Path(root).resolve()) for root in roots):
        raise ValueError("Orbit LoRAの実パスは登録済みlorasフォルダー内に指定してください。")
    if type(size) is not int or size != MODEL_BYTES or sha256 != MODEL_SHA256:
        raise ValueError("Orbit LoRAのサイズまたはSHA-256が固定版と異なります。")
    expected = json.loads(version)
    if not isinstance(expected, list) or len(expected) != 4 or expected[0] != size:
        raise ValueError("Orbit LoRAの変更情報が不正です。")
    with path.open("rb") as stream:
        before = _handle_version(stream)
        if before != expected:
            raise ValueError("Orbit LoRAが受付後に変更されました。")
        data = stream.read(MODEL_BYTES + 1)
        if _handle_version(stream) != before:
            raise ValueError("Orbit LoRAが読み込み中に変更されました。")
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Orbit LoRAのサイズまたはSHA-256が固定版と異なります。")
    verify_version(path, expected)
    return data, {"path": canonical_path, "size": len(data), "sha256": sha256, "version": before}
