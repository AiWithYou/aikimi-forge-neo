"""BiRefNetによる背景除去。モデルの取得と読み込みは初回実行時だけ行う。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops


@dataclass(frozen=True)
class BackgroundModel:
    label: str
    repository: str
    revision: str
    resolution: int


MODELS = {
    "birefnet": BackgroundModel(
        "BiRefNet / 標準", "ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", 1024
    ),
    "birefnet-hr": BackgroundModel(
        "BiRefNet HR / 高解像度", "ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1", 2048
    ),
    "birefnet-hr-matting": BackgroundModel(
        "BiRefNet HR Matting / 髪・半透明",
        "ZhengPeng7/BiRefNet_HR-matting",
        "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f",
        2048,
    ),
}
DEFAULT_MODEL = "birefnet"
MODEL_FILES = ("config.json", "BiRefNet_config.py", "birefnet.py", "model.safetensors")


def download_model(model_id, model_root):
    """固定リビジョンの必要ファイルだけ取得し、以後は通信せず再利用する。"""
    if model_id not in MODELS:
        raise ValueError("背景除去モデルを選択してください。")
    spec = MODELS[model_id]
    directory = Path(model_root) / model_id / spec.revision
    if not all((directory / name).is_file() for name in MODEL_FILES):
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=spec.repository,
            revision=spec.revision,
            local_dir=str(directory),
            allow_patterns=list(MODEL_FILES),
            token=False,
        )
    return directory


def apply_alpha(image, mask):
    """元の透明部分を保持し、二値化せず推定アルファを重ねる。"""
    rgba = image.convert("RGBA")
    alpha = ImageChops.multiply(rgba.getchannel("A"), mask.convert("L"))
    rgba.putalpha(alpha)
    return rgba, alpha


class BackgroundRemover:
    def __init__(self):
        self.model = None
        self.model_id = None

    def _load(self, model_id, model_root):
        if self.model is not None and self.model_id == model_id:
            return self.model
        self.model = None
        self.model_id = None
        directory = download_model(model_id, model_root)

        from accelerate import init_empty_weights
        from safetensors.torch import load_file
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        # 公式の固定リビジョンの実装を利用。重みはsafetensorsだけ読み込む。
        model_class = get_class_from_dynamic_module("birefnet.BiRefNet", str(directory), local_files_only=True)
        with init_empty_weights():
            model = model_class()
        model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True, assign=True)
        model.eval().requires_grad_(False)
        self.model = model
        self.model_id = model_id
        return model

    def remove(self, image, model_id, model_root, device):
        import torch
        from torchvision.transforms import InterpolationMode, Normalize, Resize, ToTensor

        if model_id not in MODELS:
            raise ValueError("背景除去モデルを選択してください。")
        spec = MODELS[model_id]
        model = self._load(model_id, model_root)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        tensor = ToTensor()(Resize((spec.resolution, spec.resolution), InterpolationMode.BICUBIC)(image.convert("RGB")))
        tensor = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor).unsqueeze(0)
        try:
            model.to(device=device, dtype=dtype)
            with torch.inference_mode():
                prediction = model(tensor.to(device=device, dtype=dtype))[-1].sigmoid().float().cpu()
            values = prediction[0, 0].numpy()
            if not np.isfinite(values).all():
                raise RuntimeError("背景除去の推定結果が不正です。")
            mask = Image.fromarray((values.clip(0, 1) * 255).round().astype(np.uint8))
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
            return apply_alpha(image, mask)
        finally:
            # 成功・失敗の両方でGPUを返す。CPU上には選択中の1モデルだけ保持する。
            model.to(device="cpu", dtype=torch.float32)
