"""MiniMax H3 Fun ControlNetの設定、入力動画の整形、標準ノードへの接続。"""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from modules_forge.minimax_h3_runtime import model_root

MODEL = "minimax_h3_fun_controlnet_union_pruned_int8_convrot.safetensors"
MODEL_BYTES = 2296635360
MODEL_SHA256 = "9c645c0a308c8af361efd43b409710f6f8fec0db297c29503e141a84991fed0c"
MODEL_REVISION = "f4cac997f880e93cf6940af61ee8d58ef31ff7f3"
NODES = ("ModelPatchLoader", "MiniMaxH3FunControlNetApply", "LoadVideo", "GetVideoComponents")


@dataclass(frozen=True)
class H3FunControl:
    mode: str = "off"
    strength: float = 1.0

    @property
    def enabled(self):
        return self.mode != "off"

    def validate(self):
        if not isinstance(self.mode, str) or self.mode not in {"off", "canny", "preprocessed"}:
            raise ValueError("Fun ControlNetの入力方式が不正です。")
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)) or not math.isfinite(self.strength) or not 0.01 <= self.strength <= 2:
            raise ValueError("Fun ControlNetの強さは0.01〜2で指定してください。")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if value is None:
            return cls()
        if not isinstance(value, dict) or set(value) - {"mode", "strength"}:
            raise ValueError("Fun ControlNet設定の形式が不正です。")
        result = cls(**value)
        result.validate()
        return result


def validate_model(runtime_root):
    path = model_root(Path(runtime_root)) / "model_patches" / MODEL
    if not path.is_file() or path.stat().st_size != MODEL_BYTES:
        raise ValueError(f"Fun ControlNet INT8モデルがありません。models/model_patches/{MODEL}を配置してください。")


def validate_nodes(nodes):
    for name in NODES:
        if name not in nodes:
            raise ValueError("Fun ControlNetの標準ノードがありません。対応版ComfyUIへ更新してください。")
    required = nodes["MiniMaxH3FunControlNetApply"].get("input", {}).get("required", {})
    expected = {"model": "MODEL", "model_patch": "MODEL_PATCH", "vae": "VAE", "strength": "FLOAT", "start_percent": "FLOAT", "end_percent": "FLOAT"}
    for key, kind in expected.items():
        if not isinstance(required.get(key), (list, tuple)) or required[key][0] != kind:
            raise ValueError(f"Fun ControlNetの{key}の仕様が未対応です。")
    optional = nodes["MiniMaxH3FunControlNetApply"].get("input", {}).get("optional", {})
    if not optional.get("control_video") or optional["control_video"][0] != "IMAGE":
        raise ValueError("Fun ControlNetの制御動画入力が未対応です。")
    choices = nodes["ModelPatchLoader"].get("input", {}).get("required", {}).get("name", [[]])[0]
    if MODEL not in choices:
        raise ValueError("接続先ComfyUIにFun ControlNet INT8モデルがありません。")


def prepare_video(source, target, width, height, frame_count, mode):
    """先頭から24fpsで取り出し、中央を切り抜く。短い動画は最終フレームで補う。"""
    import av
    import cv2
    import numpy as np
    from PIL import Image, ImageOps

    with av.open(str(source)) as container:
        if not container.streams.video:
            raise ValueError("Fun ControlNetには映像のある動画を指定してください。")
        stream = container.streams.video[0]
        fps = float(stream.average_rate or 0)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("制御動画のフレームレートを取得できません。")
        frames = iter(enumerate(container.decode(video=0)))
        first = next(frames, None)
        if first is None:
            raise ValueError("制御動画からフレームを読み込めません。")
        current_index, current = first
        origin = current.time or 0.0
        upcoming = next(frames, None)
        converted_index = -1
        rgb = None
        with av.open(str(target), mode="w") as output:
            encoded = output.add_stream("libx264rgb", rate=24)
            encoded.width, encoded.height = width, height
            encoded.pix_fmt = "rgb24"
            encoded.options = {"crf": "0", "preset": "veryfast"}
            for index in range(frame_count):
                timestamp = index / 24
                while upcoming is not None:
                    position, next_frame = upcoming
                    next_time = next_frame.time - origin if next_frame.time is not None else position / fps
                    if next_time > timestamp + 1e-8:
                        break
                    current_index, current = upcoming
                    upcoming = next(frames, None)
                if converted_index != current_index:
                    resized = ImageOps.fit(current.to_image(), (width, height), method=Image.Resampling.BILINEAR)
                    rgb = np.asarray(resized.convert("RGB"))
                    if mode == "canny":
                        edges = cv2.Canny(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 100, 200)
                        rgb = np.repeat(edges[..., None], 3, axis=-1)
                    converted_index = current_index
                frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                frame.pts = index
                for packet in encoded.encode(frame):
                    output.mux(packet)
            for packet in encoded.encode():
                output.mux(packet)


def apply_workflow(graph, control, video_name):
    if not control.enabled:
        return
    if not video_name:
        raise ValueError("Fun ControlNetの制御動画が準備されていません。")
    if any(str(number) in graph for number in range(100, 104)):
        raise ValueError("Fun ControlNetのノードIDが重複しています。")
    graph["100"] = {"class_type": "ModelPatchLoader", "inputs": {"name": MODEL}}
    graph["101"] = {"class_type": "LoadVideo", "inputs": {"file": video_name}}
    graph["102"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["101", 0]}}
    graph["103"] = {
        "class_type": "MiniMaxH3FunControlNetApply",
        "inputs": {
            "model": list(graph["9"]["inputs"]["model"]), "model_patch": ["100", 0], "vae": ["3", 0],
            "control_video": ["102", 0], "strength": control.strength, "start_percent": 0.0, "end_percent": 1.0,
        },
    }
    graph["9"]["inputs"]["model"] = ["103", 0]
    graph["8"]["inputs"]["model"] = ["103", 0]
