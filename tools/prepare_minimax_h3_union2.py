"""Install only a structurally verified native Union 2.0 patch.

--list queries two upstream-linked model repositories, freezes each revision,
and probes only small safetensors headers. --download is the explicit large
transfer. --source imports an already downloaded compatible native checkpoint.
No dependency/model download is triggered by importing the module or Forge UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from modules_forge.minimax_h3_runtime import configured_model_root, managed_runtime_root, setup_lock
from modules_forge.minimax_h3_union2_vae import (
    MAX_HEADER_BYTES,
    UNION2_MODEL,
    inspect_union2,
    read_header,
    sha256_file,
    validate_union2_header,
)

REPOSITORIES = ("Comfy-Org/MiniMax-H3", "Kijai/MiniMax-H3-experimental")
LICENSE = "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
SHA = re.compile(r"^[a-f0-9]{64}$")
REV = re.compile(r"^[a-f0-9]{40}$")


def open_url(url, *, headers=None):
    # Only HTTPS is used. urllib sends no browser cookies or HF token.
    if not url.startswith("https://huggingface.co/"):
        raise ValueError("許可されたHugging Face URLではありません。")
    request = urllib.request.Request(url, headers={"User-Agent": "Aikimi-H3-Union2-Installer/1", **(headers or {})})
    return urllib.request.urlopen(request, timeout=90)


def get_json(url):
    with open_url(url) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("モデルAPI応答が大きすぎます。")
    return json.loads(raw)


def remote_header(url):
    # Read from byte zero so a server ignoring Range is also safe: stream.close
    # stops the large body after the bounded header has been read.
    with open_url(url, headers={"Range": f"bytes=0-{MAX_HEADER_BYTES + 7}"}) as response:
        if response.status not in (200, 206):
            raise ValueError("モデルの先頭データを取得できません。")
        if response.status == 206 and not response.headers.get("Content-Range", "").startswith("bytes 0-"):
            raise ValueError("モデルサーバーが誤った範囲を返しました。")
        header, _ = read_header(response)
    return header


def candidates(*, precision="int8", get=get_json, probe=remote_header):
    found, diagnostics = [], []
    for repository in REPOSITORIES:
        try:
            info = get(f"https://huggingface.co/api/models/{repository}?blobs=true")
            if not isinstance(info, dict) or not isinstance(info.get("siblings"), list):
                raise ValueError("モデルAPI応答の形式が不正です。")
            revision = info.get("sha", "")
            if not isinstance(revision, str) or not REV.fullmatch(revision):
                raise ValueError("固定revisionを取得できません。")
            for file in info.get("siblings", []):
                if not isinstance(file, dict) or not isinstance(file.get("rfilename"), str):
                    continue
                name = file["rfilename"]
                lower = name.lower()
                if (
                    not name.startswith("model_patches/")
                    or not name.endswith(".safetensors")
                    or "union" not in lower
                    or not re.search(r"(?:v2|2[._-]0|union[_-]?2)", lower)
                ):
                    continue
                if ".." in Path(name).parts or "\\" in name:
                    continue
                lfs = file.get("lfs") or {}
                if not isinstance(lfs, dict):
                    lfs = {}
                digest, size = lfs.get("sha256", ""), lfs.get("size", file.get("size"))
                if (
                    not isinstance(digest, str)
                    or not SHA.fullmatch(digest)
                    or type(size) is not int
                    or not 0 < size <= 32 * 1024**3
                ):
                    diagnostics.append(f"{repository}/{name}: LFSサイズ/SHA-256を確認できません。")
                    continue
                url = f"https://huggingface.co/{repository}/resolve/{revision}/{urllib.parse.quote(name, safe='/')}"
                try:
                    actual_precision = validate_union2_header(probe(url))
                except (OSError, ValueError) as exc:
                    diagnostics.append(f"{repository}/{name}: {exc}")
                    continue
                if actual_precision == precision:
                    found.append(
                        {
                            "repository": repository,
                            "revision": revision,
                            "path": name,
                            "size": size,
                            "sha256": digest,
                            "precision": precision,
                            "url": url,
                        }
                    )
        except (OSError, ValueError) as exc:
            diagnostics.append(f"{repository}: {exc}")
    return found, diagnostics


def install_verified(source: Path, models: Path, *, provenance=None):
    source = source.resolve(strict=True)
    description = inspect_union2(source)
    digest = sha256_file(source)
    provenance = dict(provenance or {"source": "local-file", "original_filename": source.name})
    if "sha256" in provenance and digest != provenance["sha256"]:
        raise ValueError("モデルのSHA-256が配布元と一致しません。")
    target_dir = models / "model_patches"
    target_dir.mkdir(parents=True, exist_ok=True)
    if target_dir.is_symlink() or target_dir.resolve() != target_dir.absolute():
        raise ValueError("model_patchesへのリンクは使用しません。共有先はモデル保存先に設定してください。")
    target = target_dir / UNION2_MODEL
    if target.exists() or target.is_symlink():
        if target.is_symlink() or sha256_file(target) != digest:
            raise ValueError("別のUnion 2.0が既にあります。既存ファイルは上書きしません。")
        return target
    if shutil.disk_usage(target_dir).free < description["size"] + 1024**3:
        raise ValueError("モデル保存先の空き容量が不足しています（モデルサイズ＋1 GiBが必要）。")
    fd, temp = tempfile.mkstemp(prefix=".union2-", dir=target_dir)
    stage = Path(temp)
    receipt = target.with_suffix(".provenance.json")
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as incoming:
            shutil.copyfileobj(incoming, output, 8 * 1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if sha256_file(stage) != digest:
            raise ValueError("保存したモデルのハッシュが一致しません。")
        record = {
            **provenance,
            **description,
            "schema_version": 1,
            "sha256": digest,
            "installed_name": UNION2_MODEL,
            "license_url": LICENSE,
        }
        # Same-directory hard link is atomic and refuses an existing target.
        # The staging name is removed in finally; no partially written model
        # becomes visible under the filename used by the loader.
        os.link(stage, target)
        try:
            with receipt.open("x", encoding="utf-8") as output:
                json.dump(record, output, ensure_ascii=False, indent=2)
                output.write("\n")
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        return target
    finally:
        stage.unlink(missing_ok=True)


def download_verified(candidate, destination, *, opener=open_url):
    digest = hashlib.sha256()
    received = 0
    created = False
    try:
        with opener(candidate["url"]) as response, Path(destination).open("xb") as stream:
            created = True
            while block := response.read(8 * 1024 * 1024):
                received += len(block)
                if received > candidate["size"]:
                    raise ValueError("配布情報より大きいモデルデータを受信しました。")
                stream.write(block)
                digest.update(block)
            stream.flush()
            os.fsync(stream.fileno())
        if received != candidate["size"] or digest.hexdigest() != candidate["sha256"]:
            raise ValueError("モデルのサイズまたはSHA-256が一致しません。")
        inspect_union2(Path(destination))
    except BaseException:
        if created:
            Path(destination).unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--source", type=Path)
    parser.add_argument("--precision", choices=["int8", "bf16"], default="int8")
    parser.add_argument("--repository", choices=REPOSITORIES)
    parser.add_argument("--filename", help="Select an exact path returned by --list")
    args = parser.parse_args()
    models = configured_model_root(ROOT)
    if args.source:
        with setup_lock(managed_runtime_root(ROOT)):
            print(install_verified(args.source, models))
        return
    choices, diagnostics = candidates(precision=args.precision)
    if args.repository:
        choices = [c for c in choices if c["repository"] == args.repository]
    if args.filename:
        choices = [c for c in choices if c["path"] == args.filename]
    print(json.dumps({"compatible_candidates": choices, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))
    if args.list:
        return
    if not choices:
        raise ValueError(
            "互換性を確認できる配布ファイルがありません。未確認の重みはダウンロードしません。--sourceで変換済みファイルを指定することもできます。"
        )
    if len(choices) != 1:
        raise ValueError("候補が複数あります。--repository と --filename で一つを明示してください。")
    candidate = choices[0]
    print(f"License: {LICENSE}\nDownloading {candidate['size'] / 1024**3:.2f} GiB at {candidate['revision']}")
    models.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(models).free < candidate["size"] * 2 + 1024**3:
        raise ValueError("作業コピーを含む空き容量が不足しています。モデルサイズの2倍＋1 GiBが必要です。")
    with (
        setup_lock(managed_runtime_root(ROOT)),
        tempfile.TemporaryDirectory(prefix=".union2-download-", dir=models) as folder,
    ):
        source = Path(folder) / "model.safetensors"
        download_verified(candidate, source)
        print(install_verified(source, models, provenance=candidate))
    print("導入完了。H3を再起動し、Union 2.0を選択してください。推論は実行していません。")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"Union 2.0: {error}", file=sys.stderr)
        raise SystemExit(1) from error
