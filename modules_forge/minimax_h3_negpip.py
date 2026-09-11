"""Opt-in integration policy for hako-mikan's pinned MiniMax H3 NegPiP node.

No ComfyUI or GPU imports, network access, or installation at import time.
The attention implementation remains upstream's, inside the isolated runtime.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

NEGPIP_NODE = "ApplyMiniMaxH3NegPiP"
NEGPIP_PACK = "Aikimi-MiniMax-H3-NegPiP"
NEGPIP_COMMIT = "f725718b4c597fab92cdb1fe981dfe3544a75665"
BUNDLE_BLOBS = {
    "__init__.py": "ddbbf398ad63d403ee8710c7626eab239bbbb445",
    "minimax_h3_negpip.py": "f6a7d758952ecf365fc413157ac30be4b397648d",
    "LICENSE": "0ad25db4bd1d86c452db3f9602ccdbe172438f52",
}
BUNDLE_ROOT = (
    Path(__file__).resolve().parents[1] / "extensions-builtin" / "minimax-h3-studio" / "comfyui_nodes" / NEGPIP_PACK
)


@dataclass(frozen=True)
class H3NegPiP:
    enabled: bool = False
    value_strength: float = 1.0
    apply_positive_weights: bool = True
    block_start: int = 0
    block_end: int = 999
    block_stride: int = 1
    protect_text_rows: bool = False
    measure_attention_mass: bool = False

    def validate(self) -> None:
        for name in ("enabled", "apply_positive_weights", "protect_text_rows", "measure_attention_mass"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"NegPiP {name}はオン／オフで指定してください。")
        value = self.value_strength
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= 8
        ):
            raise ValueError("NegPiP value_strengthは0〜8の有限値で指定してください。")
        for name, low, high in (("block_start", 0, 999), ("block_end", 0, 999), ("block_stride", 1, 16)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value != int(value)
                or not low <= value <= high
            ):
                raise ValueError(f"NegPiP {name}は{low}〜{high}の整数で指定してください。")
        if self.block_start > self.block_end:
            raise ValueError("NegPiPの開始ブロックは終了ブロック以下にしてください。")

    @classmethod
    def from_dict(cls, value: Any) -> H3NegPiP:
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - {f.name for f in fields(cls)}:
            raise ValueError("NegPiP設定の形式が不正です。")
        result = cls(**value)
        result.validate()
        return result

    @classmethod
    def from_values(cls, values: Sequence) -> H3NegPiP:
        if not values:
            return cls()
        if len(values) != len(fields(cls)):
            raise ValueError("NegPiPの項目数が一致しません。UIを再読み込みしてください。")
        result = cls(*values)
        result.validate()
        return result

    def values(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def node_inputs(self) -> dict[str, Any]:
        self.validate()
        result = self.to_dict()
        result.pop("enabled")
        for name in ("block_start", "block_end", "block_stride"):
            result[name] = int(result[name])
        return result

    def apply_workflow(
        self, graph: dict, *, guider: str = "9", condition: str = "5", scheduler: str = "8", node_id: str = "17"
    ) -> None:
        """Connect BOTH cloned outputs, preserving upstream attention/model patches."""
        self.validate()
        if not self.enabled:
            return
        if node_id in graph:
            raise ValueError(f"NegPiP用のnode ID {node_id}が重複しています。")
        model = list(graph[guider]["inputs"]["model"])
        clip = list(graph[condition]["inputs"]["clip"])
        graph[node_id] = {"class_type": NEGPIP_NODE, "inputs": {"model": model, "clip": clip, **self.node_inputs()}}
        graph[guider]["inputs"]["model"] = [node_id, 0]
        graph[scheduler]["inputs"]["model"] = [node_id, 0]
        graph[condition]["inputs"]["clip"] = [node_id, 1]

    def validate_nodes(self, nodes: Mapping) -> None:
        if not self.enabled:
            return
        schema = nodes.get(NEGPIP_NODE)
        if not isinstance(schema, Mapping):
            raise ValueError("NegPiPノードがありません。「選択設定で再起動」で同梱版を導入・読み込みしてください。")
        spec = schema.get("input", {})
        if not isinstance(spec, Mapping):
            raise ValueError("NegPiPノードの入力仕様が不正です。")
        required, optional = spec.get("required", {}), spec.get("optional", {})
        if not isinstance(required, Mapping) or not isinstance(optional, Mapping):
            raise ValueError("NegPiPノードの入力仕様が不正です。")
        inputs = {**required, **optional}
        expected = {
            "model": "MODEL",
            "clip": "CLIP",
            "value_strength": "FLOAT",
            "apply_positive_weights": "BOOLEAN",
            "block_start": "INT",
            "block_end": "INT",
            "block_stride": "INT",
            "protect_text_rows": "BOOLEAN",
            "measure_attention_mass": "BOOLEAN",
        }
        for name, kind in expected.items():
            value = inputs.get(name)
            if not isinstance(value, (list, tuple)) or not value or value[0] != kind:
                raise ValueError(f"NegPiPの{name}の仕様が未対応です。同梱版のnodeを使用してください。")
        if set(required) - expected.keys() or tuple(schema.get("output", ())) != ("MODEL", "CLIP"):
            raise ValueError("NegPiPの入出力仕様が未対応です。同梱版のnodeを使用してください。")


def custom_node_whitelist(arguments: Sequence[str]) -> tuple[str, ...] | None:
    """Parse the single nargs='+' option without weakening other launch guards."""
    option = "--whitelist-custom-nodes"
    matches = [i for i, arg in enumerate(arguments) if arg == option or arg.startswith(option + "=")]
    if not matches:
        return ()
    if len(matches) != 1:
        return None
    index = matches[0]
    value = arguments[index]
    result = [value.split("=", 1)[1]] if "=" in value else []
    for arg in arguments[index + 1 :]:
        if arg.startswith("--"):
            break
        result.append(arg)
    if not result or any(not value for value in result) or len(result) != len(set(result)):
        return None
    return tuple(result)


def _verify_bundle(directory: Path) -> None:
    for name, expected in BUNDLE_BLOBS.items():
        path = directory / name
        if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
            raise ValueError(f"NegPiPの同梱ファイルが不足、またはリンクです: {path}")
        content = path.read_bytes()
        actual = hashlib.sha1(
            b"blob " + str(len(content)).encode("ascii") + b"\0" + content, usedforsecurity=False
        ).hexdigest()
        if actual != expected:
            raise ValueError(f"NegPiPファイルの内容が固定版と異なります。上書きしません: {path}")


def install_bundled_negpip(runtime_root: Path, *, bundle_root: Path | None = None) -> Path:
    """Copy audited local files atomically; never download or overwrite user edits.

    Called only immediately before starting a stopped, selected ComfyUI runtime.
    The bridge owns the lifecycle lock and verifies the local runtime identity.
    """
    source = (bundle_root or BUNDLE_ROOT).absolute()
    _verify_bundle(source)
    root = runtime_root.resolve(strict=True)
    if not (root / "main.py").is_file() or not (root / "models").is_dir():
        raise ValueError("NegPiPの導入先はComfyUIのルートフォルダーを指定してください。")
    custom = root / "custom_nodes"
    if custom.is_symlink() or custom.resolve() != custom:
        raise ValueError("NegPiPはリンクされたcustom_nodesへ導入しません。")
    custom.mkdir(exist_ok=True)
    target = custom / NEGPIP_PACK
    if target.is_symlink() or target.resolve() != target:
        raise ValueError("NegPiPの導入先がリンクになっています。上書きしません。")
    if target.exists():
        _verify_bundle(target)
        unexpected = {p.name for p in target.iterdir()} - set(BUNDLE_BLOBS) - {"__pycache__"}
        if unexpected:
            raise ValueError(
                "管理対象NegPiPフォルダーに未知のファイルがあります。上書きしません: " + ", ".join(sorted(unexpected))
            )
        return target
    stage = Path(tempfile.mkdtemp(prefix=".aikimi-negpip-", dir=custom))
    try:
        for name in BUNDLE_BLOBS:
            shutil.copyfile(source / name, stage / name)
        _verify_bundle(stage)
        # rename, not replace: an existing installation must never be overwritten.
        os.rename(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return target
