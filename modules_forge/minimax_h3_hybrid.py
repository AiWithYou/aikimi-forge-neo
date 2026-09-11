"""長尺H3の区間設定と標準KSampler用グラフ。GPUには依存しない。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PACK = "ComfyUI-HybridWindows"
REVISION = "10a0895f24ab72307fd7f87b6f7dfd92a4a2e706"
# 全尺VAEの処理中は状態APIの応答も遅れるため、通常生成と待機時間を分ける。
POLL_TIMEOUT_SECONDS = 120.0
NODES = {"H3HybridWindows", "KSamplerAdvanced", "EmptyMiniMaxH3LatentAV", "MiniMaxH3SigmaShift"}
MANIFEST = Path(__file__).resolve().parents[1] / "tools/minimax_h3_hybrid_manifest.json"


def manifest() -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if data["repository"] != "Jalen-Brunson/ComfyUI-HybridWindows" or data["revision"] != REVISION:
        raise ValueError("HybridWindowsの固定版が一致しません。")
    expected = {
        "__init__.py",
        "nodes.py",
        "windows.py",
        "sampler.py",
        "joint.py",
        "control.py",
        "color.py",
        "dependencies.py",
    }
    if {entry["path"] for entry in data["files"]} != expected or len(data["files"]) != len(expected):
        raise ValueError("HybridWindowsのファイル一覧が不正です。")
    return data


def blob_matches(content: bytes, entry: dict) -> bool:
    return (
        len(content) == entry["bytes"]
        and hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content, usedforsecurity=False).hexdigest()
        == entry["blob"]
    )


def verify_directory(directory: Path) -> None:
    for entry in manifest()["files"]:
        path = directory / entry["path"]
        if not path.is_file() or path.is_symlink() or not blob_matches(path.read_bytes(), entry):
            raise ValueError(
                f"HybridWindowsの固定版ファイルを確認できません: {entry['path']}。tools/install_minimax_h3_hybrid.pyで導入してください。"
            )


def validate_nodes(nodes: dict) -> None:
    expected = {
        "H3HybridWindows": {"model": "MODEL", "window_frames": "INT", "overlap_frames": "INT", "total_steps": "INT"},
        "EmptyMiniMaxH3LatentAV": {"width": "INT", "height": "INT", "length": "INT"},
        "MiniMaxH3SigmaShift": {"model": "MODEL", "shift_video": "FLOAT", "shift_audio": "FLOAT"},
        "KSamplerAdvanced": {
            "model": "MODEL",
            "latent_image": "LATENT",
            "steps": "INT",
            "start_at_step": "INT",
            "end_at_step": "INT",
        },
    }
    for name, required in expected.items():
        spec = nodes.get(name, {}).get("input", {})
        inputs = {**spec.get("required", {}), **spec.get("optional", {})}
        if any(not inputs.get(key) or inputs[key][0] != kind for key, kind in required.items()):
            raise ValueError(
                f"長尺生成に必要な{name}の入力仕様が一致しません。ComfyUIとHybridWindowsの導入を確認してください。"
            )


@dataclass(frozen=True)
class H3Hybrid:
    enabled: bool = False
    windows: int = 3
    overlap: int = 39
    switch_step: int = 16
    prompts: str = ""

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("長尺生成の有効・無効を選択してください。")
        for name, low, high in (("windows", 2, 10), ("overlap", 5, 345), ("switch_step", 1, 100)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not low <= value <= high
                or int(value) != value
            ):
                raise ValueError(f"長尺設定 {name} は {low}〜{high} の整数で指定してください。")
        if not isinstance(self.prompts, str) or len(self.prompts) > 20_000:
            raise ValueError("区間プロンプトは20,000文字以内で入力してください。")
        if self.overlap % 17 != 5:
            raise ValueError("重なりは5、22、39…フレームで指定してください。")
        if self.enabled and self.prompts.strip() and len(self.prompts.strip().splitlines()) != self.windows:
            raise ValueError("区間数と同じ行数で、1行に1区間の指示を入力してください。")
        if (
            self.enabled
            and self.prompts.strip()
            and any(not line.strip() for line in self.prompts.strip().splitlines())
        ):
            raise ValueError("区間プロンプトの空行を埋めてください。")

    def frame_count(self, window_frames: int) -> int:
        self.validate()
        if self.overlap >= window_frames:
            raise ValueError("重なりは1区間の長さより短くしてください。")
        total = window_frames + (int(self.windows) - 1) * (window_frames - int(self.overlap))
        if total > 3600:
            raise ValueError("全体の長さが150秒を超えます。区間数または区間の長さを減らしてください。")
        return total

    def values(self) -> tuple:
        return self.enabled, self.windows, self.overlap, self.switch_step, self.prompts

    @classmethod
    def from_dict(cls, value: Any) -> H3Hybrid:
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - set(asdict(cls())):
            raise ValueError("長尺設定の形式が不正です。")
        result = cls(**value)
        result.validate()
        return result


def apply_workflow(workflow: dict, option: H3Hybrid, window_frames: int, steps: int, scheduler: str, seed: int) -> None:
    """各区間の条件を分離し、全尺のAV latentを2段階のEulerで生成する。"""
    total = option.frame_count(window_frames)
    template = workflow["5"]
    common = template["inputs"]["prompt"]
    prompts = option.prompts.strip().splitlines() if option.prompts.strip() else [""] * int(option.windows)
    model = workflow["9"]["inputs"]["model"]
    workflow["h3_shift"] = {
        "class_type": "MiniMaxH3SigmaShift",
        "inputs": {"model": model, "shift_video": 12.0, "shift_audio": 3.0},
    }
    inputs = {
        "model": ["h3_shift", 0],
        "window_frames": window_frames,
        "overlap_frames": int(option.overlap),
        "total_steps": steps,
    }
    for index, prompt in enumerate(prompts):
        node_id = "5" if index == 0 else f"h3_cond_{index}"
        conditioning = {
            **template["inputs"],
            "length": window_frames,
            "prompt": common + ("\n" + prompt.strip() if prompt.strip() else ""),
        }
        if index:
            conditioning.pop("first_frame", None)
        # 終了画像は各区間の共通ガイド。最終区間だけで急に構図を変えない。
        workflow[node_id] = {"class_type": template["class_type"], "inputs": conditioning}
        inputs[f"prompts.positive_{index}"] = [node_id, 0]
    workflow["h3_windows"] = {"class_type": "H3HybridWindows", "inputs": inputs}
    workflow["h3_latent"] = {
        "class_type": "EmptyMiniMaxH3LatentAV",
        "inputs": {
            "width": template["inputs"]["width"],
            "height": template["inputs"]["height"],
            "length": total,
        },
    }
    sampling = {
        "noise_seed": seed,
        "steps": steps,
        "cfg": 1.0,
        "sampler_name": "euler",
        "scheduler": scheduler,
        "positive": ["h3_windows", 2],
        "negative": ["h3_windows", 2],
    }
    workflow["h3_warmup"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **sampling,
            "model": ["h3_windows", 0],
            "latent_image": ["h3_latent", 0],
            "add_noise": "enable",
            "start_at_step": 0,
            "end_at_step": int(option.switch_step),
            "return_with_leftover_noise": "enable",
        },
    }
    workflow["10"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **sampling,
            "model": ["h3_windows", 1],
            "latent_image": ["h3_warmup", 0],
            "add_noise": "disable",
            "start_at_step": int(option.switch_step),
            "end_at_step": steps,
            "return_with_leftover_noise": "disable",
        },
    }
    for node_id in ("6", "7", "8", "9"):
        del workflow[node_id]
