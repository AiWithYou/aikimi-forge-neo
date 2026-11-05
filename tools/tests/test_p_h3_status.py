"""H3 Image availability reflects its separate image workflows."""

from unittest.mock import patch

import pytest

from modules import aikimi_capabilities as capabilities
from modules.aikimi_diagnostics import DiagnosticPaths
from modules_forge import p_h3_image as ph3


@pytest.mark.parametrize("ph3_present,legacy_present", [(True, False), (False, True), (True, True), (False, False)])
def test_image_status_does_not_require_video_models_for_p_h3(tmp_path, ph3_present, legacy_present):
    for relative in ("modules_forge/minimax_h3_bridge.py", "modules_forge/p_h3_image.py",
                     "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_studio.py",
                     "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    runtime = tmp_path / "runtime"
    if ph3_present:
        for relative in (f"diffusion_models/{ph3.MODEL}", f"text_encoders/{ph3.TEXT_ENCODER}", f"vae/{ph3.VAE}"):
            file = runtime / "models" / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.touch()
    paths = DiagnosticPaths(root=tmp_path, models_root=tmp_path / "models", output_root=tmp_path / "output")
    with (patch("modules_forge.minimax_h3_runtime.installed_runtime_root", return_value=runtime),
          patch("modules_forge.minimax_h3_runtime.candidate_runtime_root", return_value=runtime),
          patch("modules_forge.minimax_h3_bridge.model_file_status", return_value={
              "FL2VA": legacy_present, "Ref2VA": False, "Qwen3-VL 32B": legacy_present,
              "Video VAE": legacy_present, "Audio VAE": legacy_present,
          }),
          patch("modules_forge.local_assets.selection", return_value={})):
        check = capabilities.feature_check("minimax_h3_image", paths)
    assert check.available is (ph3_present or legacy_present)
    assert "P-H3" in check.summary
    assert "標準H3" in check.summary
    if ph3_present:
        assert "補助" in check.summary


def test_selected_local_checkpoint_is_counted_without_standard_p_h3_model(tmp_path):
    for relative in ("modules_forge/minimax_h3_bridge.py", "modules_forge/p_h3_image.py",
                     "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_studio.py",
                     "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    runtime = tmp_path / "runtime"
    for relative in (f"text_encoders/{ph3.TEXT_ENCODER}", f"vae/{ph3.VAE}"):
        file = runtime / "models" / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    custom = tmp_path / "custom.safetensors"
    custom.touch()
    paths = DiagnosticPaths(root=tmp_path, models_root=tmp_path / "models", output_root=tmp_path / "output")
    with (patch("modules_forge.minimax_h3_runtime.installed_runtime_root", return_value=runtime),
          patch("modules_forge.minimax_h3_runtime.candidate_runtime_root", return_value=runtime),
          patch("modules_forge.minimax_h3_bridge.model_file_status", return_value={}),
          patch("modules_forge.local_assets.selection", side_effect=lambda engine: {"checkpoint": str(custom)} if engine == "p-h3-image" else {})):
        check = capabilities.feature_check("minimax_h3_image", paths)
    assert check.available
    assert str(custom) not in check.summary


def test_old_runtime_does_not_claim_p_h3_is_available(tmp_path):
    for relative in ("modules_forge/minimax_h3_bridge.py", "modules_forge/p_h3_image.py",
                     "extensions-builtin/minimax-h3-studio/scripts/minimax_h3_image_studio.py"):
        file = tmp_path / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    old = tmp_path / "old"
    for relative in (f"diffusion_models/{ph3.MODEL}", f"text_encoders/{ph3.TEXT_ENCODER}", f"vae/{ph3.VAE}"):
        file = old / "models" / relative
        file.parent.mkdir(parents=True, exist_ok=True)
        file.touch()
    paths = DiagnosticPaths(root=tmp_path, models_root=tmp_path / "models", output_root=tmp_path / "output")
    with (patch("modules_forge.minimax_h3_runtime.installed_runtime_root", return_value=old),
          patch("modules_forge.minimax_h3_runtime.candidate_runtime_root", return_value=tmp_path / "new"),
          patch("modules_forge.minimax_h3_bridge.model_file_status", return_value={}),
          patch("modules_forge.local_assets.selection", return_value={})):
        check = capabilities.feature_check("minimax_h3_image", paths)
    assert check.available is False
