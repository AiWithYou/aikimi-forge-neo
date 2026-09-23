"""H3 Studio / ComfyUI handoff integration. Imports the GPU bridge lazily."""

from __future__ import annotations

import hashlib
import html
import logging
import os
import shutil
import tempfile
from pathlib import Path

from modules_forge.minimax_h3_handoff_store import (
    PACK,
    HandoffError,
    create_snapshot,
    handoff_url,
    read_receipt,
    read_snapshot,
    write_receipt,
)

_LOG = logging.getLogger(__name__)
_BUNDLE = Path(__file__).resolve().parents[1] / "extensions-builtin" / "minimax-h3-studio" / "comfyui_nodes" / PACK
# Filled at packaging time. These include store.py copied from the pure module.
PACK_HASHES = {
    "__init__.py": "55f48de8b29646e20a9531dd660c3e688118c72c7ea4ed95ea82c954fc85b8b4",
    "store.py": "7889860183fa3402a3ffced2881c63a88d1c97a8958df91b48658ff06018ef5b",
    "web/handoff-core.js": "8bc0ec305fe1581a619f4e19c1d7585d7445b2d1cc0bf9ac9d4592493ef142f2",
    "web/handoff.js": "c307aacb94bfebd549039e444f60da36a9f2a4218be42d3027ee79e57cdc49c0",
    "web/package.json": "609158e6c5fbc237939fa3ddf7faab80ab690bdc0c8d584414a885130103c4e8",
}  # BUILD_HASHES


def install_bundle(runtime_root: Path) -> Path:
    """Install only this audited local UI pack; do not enable arbitrary nodes."""
    root = Path(runtime_root).resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise HandoffError("引き継ぎ先はComfyUIルートを指定してください。")
    custom = root / "custom_nodes"
    if custom.is_symlink() or getattr(custom, "is_junction", lambda: False)() or custom.resolve() != custom:
        raise HandoffError("リンクされたcustom_nodesには導入しません。")
    custom.mkdir(exist_ok=True)
    source_files = {
        name: (_BUNDLE / name if name != "store.py" else Path(__file__).with_name("minimax_h3_handoff_store.py"))
        for name in PACK_HASHES
    }
    if not source_files:
        raise HandoffError("引き継ぎパックの検証情報がありません。")
    for name, source in source_files.items():
        if (
            source.is_symlink()
            or source.resolve() != source.absolute()
            or not source.is_file()
            or hashlib.sha256(source.read_bytes()).hexdigest() != PACK_HASHES[name]
        ):
            raise HandoffError(f"引き継ぎパックの同梱ファイルが固定版と異なります: {name}")
    target = custom / PACK
    if target.is_symlink() or target.resolve() != target:
        raise HandoffError("引き継ぎパックの導入先がリンクです。")
    if target.exists():
        for name, checksum in PACK_HASHES.items():
            candidate = target / name
            if (
                candidate.is_symlink()
                or candidate.resolve() != candidate.absolute()
                or not candidate.is_file()
                or hashlib.sha256(candidate.read_bytes()).hexdigest() != checksum
            ):
                raise HandoffError(
                    "導入済みの引き継ぎパックが異なります。実行環境を停止し、既存パックを退避してから再起動してください。"
                )
        unexpected = {
            p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file() and "__pycache__" not in p.parts
        } - set(PACK_HASHES)
        if unexpected:
            raise HandoffError("導入済みの引き継ぎパックに未知のファイルがあります。上書きしません。")
        return target
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-handoff-", dir=custom))
    try:
        for name, source in source_files.items():
            destination = stage / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target


def _metadata(request, seed, readiness, profile, *, source, generation_id=None):
    return {
        "source": source,
        "generation_id": generation_id,
        "mode": request.mode,
        "seed": seed,
        "width": request.dimensions[0],
        "height": request.dimensions[1],
        "frames": request.frame_count,
        "fps": 24,
        "steps": request.steps,
        "scheduler": request.scheduler,
        "acceleration": request.acceleration.to_dict(),
        "control": request.control.to_dict(),
        "runtime_profile": profile,
        "runtime_args": list(readiness.runtime_args),
        "comfy_version": readiness.comfy_version,
        "core_revision": readiness.core_revision,
        "package_versions": dict(readiness.package_versions),
        "reproducibility": "graph-and-prepared-inputs; not a bit-identical GPU output guarantee",
    }


def capture_generation(request, workflow, prepared, seed, readiness, runtime_root, runtime_profile, submission_id):
    """Optional capture failure must not discard a successful video generation."""
    try:
        record = create_snapshot(
            Path(runtime_root) / "input",
            workflow,
            prepared,
            _metadata(request, seed, readiness, runtime_profile, source="generation", generation_id=submission_id),
        )
        return record, ""
    except Exception as exc:
        _LOG.warning("H3 workflow snapshot unavailable: %s", exc)
        return None, "再編集用ワークフローの保存に失敗しました。ログと空き容量を確認してください。"


def finish_generation(video, record, warning="", *, source=None):
    if record is not None:
        try:
            write_receipt(Path(video), record)
            if source is not None and Path(source).resolve() != Path(video).resolve():
                write_receipt(Path(source), record)
        except Exception as exc:
            _LOG.warning("H3 workflow receipt unavailable: %s", exc)
            warning = "動画は保存できましたが、履歴とワークフローの関連付けに失敗しました。"
    return warning


def _validate_control_handoff(bridge, readiness, control, acceleration, server_url):
    from modules_forge.minimax_h3_union2_vae import check_runtime

    check_runtime(readiness, decode_mode=acceleration.decode_mode, union2=control.is_union2)
    client = bridge.ComfyH3Client(server_url)
    try:
        bridge.fun_control.validate_nodes(client.object_info(bridge.fun_control.NODES), control)
    finally:
        client.close()
    bridge.fun_control.validate_model(readiness.runtime_root, control)


def export_current(request, runtime_root, server_url, log_directory, runtime_profile):
    from modules_forge import minimax_h3_bridge as bridge

    bridge.validate_request(request)
    root = bridge.resolve_runtime_root(runtime_root)
    # URL validation precedes any runtime startup or file IO.
    handoff_url(server_url, "0" * 32)
    readiness = bridge.ensure_ready(
        root, server_url, log_directory, runtime_profile=runtime_profile, acceleration=request.acceleration
    )
    if request.control.enabled:
        _validate_control_handoff(bridge, readiness, request.control, request.acceleration, server_url)
    prepared = bridge.prepare_media(request, root)
    try:
        seed = request.resolved_seed  # Resolve once, exactly as generation does.
        prompt = bridge.build_workflow(request, prepared, seed=seed)
        record = create_snapshot(
            root / "input",
            prompt,
            prepared,
            _metadata(request, seed, readiness, runtime_profile, source="current-settings"),
        )
    finally:
        bridge.cleanup_prepared_media(prepared, root)
    return record, handoff_url(server_url, record["token"])


def export_history(selected, items, runtime_root, server_url, log_directory):
    from modules_forge import minimax_h3_bridge as bridge
    from modules_forge.minimax_h3_acceleration import H3Acceleration

    item = next((item for item in items if item.public_id == selected), None)
    if item is None:
        raise HandoffError("履歴を選択してください。削除された履歴は開けません。")
    # Never accept an arbitrary browser-supplied file path.
    receipt = read_receipt(item.path)
    root = bridge.resolve_runtime_root(runtime_root)
    record = read_snapshot(root / "input", receipt["token"])
    if any(
        receipt.get(key) != record.get(key) for key in ("snapshot_sha256", "prompt_sha256", "submitted_prompt_sha256")
    ):
        raise HandoffError("履歴と保存ワークフローが一致しません。")
    metadata = record["metadata"]
    handoff_url(server_url, record["token"])
    from modules_forge.minimax_h3_fun_control import H3FunControl

    acceleration = H3Acceleration.from_dict(metadata["acceleration"])
    control = H3FunControl.from_dict(metadata["control"])
    readiness = bridge.ensure_ready(
        root, server_url, log_directory, runtime_profile=metadata["runtime_profile"], acceleration=acceleration
    )
    if control.enabled:
        _validate_control_handoff(bridge, readiness, control, acceleration, server_url)
    return record, handoff_url(server_url, record["token"])


def result_html(record, url):
    meta = record["metadata"]
    return (
        '<div role="status" aria-live="polite">'
        f"<strong>Seed {html.escape(str(meta['seed']))} のワークフローを準備しました。</strong> "
        '<a href="' + html.escape(url, quote=True) + '" target="_blank" rel="noopener noreferrer">'
        "ComfyUIで編集画面を開く</a><br>"
        "<small>自動生成はしません。ポップアップが開かない場合は上のリンクを押してください。"
        "同じPCのブラウザー専用です。入力素材の保存でディスク容量を使用します。</small></div>"
    )
