from dataclasses import replace

import pytest

from modules_forge import p_h3_image as image
from tools import setup_minimax_h3_image as setup


def test_official_sampling_contract():
    request = image.PH3Request("A red fox", official=True, use_upsampler=False, width=4096, height=4096)
    graph = image.build_workflow(request, {}, 123)
    condition = graph["condition"]
    assert condition["class_type"] == "MiniMaxH3ImageConditioning"
    assert condition["inputs"]["width"] == condition["inputs"]["height"] == 4096
    sample = graph["sample"]["inputs"]
    assert (sample["seed"], sample["steps"], sample["cfg"], sample["denoise"]) == (123, 6, 1.0, 1.0)
    assert (sample["sampler_name"], sample["scheduler"]) == ("euler", "simple")
    assert sample["model"] == ["model", 0]
    assert "shift" not in graph and "upsampler" not in graph


def test_official_edit_reference_order_and_limits():
    images = [f"{index}.png" for index in range(16)]
    request = image.PH3Request("Use <Picture 16>", mode="edit", target_image=images[0],
                               reference_images=tuple(images[1:]), official=True, use_upsampler=False)
    graph = image.build_workflow(request, {"images": images}, 0)
    for index, path in enumerate(images, 1):
        assert graph[f"image_{index}"]["inputs"]["image"] == path
        assert graph["condition"]["inputs"][f"images.image_{index}"] == [f"image_{index}", 0]
    for invalid in (replace(request, reference_images=tuple(images)), replace(request, width=2049),
                    replace(request, use_upsampler=True), replace(request, width=8192, height=8192)):
        with pytest.raises(image.PH3Error):
            invalid.validate()


def test_official_manifest_matches_workflow():
    request = image.PH3Request("A red fox", official=True, use_upsampler=False)
    graph = image.build_workflow(request, {}, 123)
    filenames = {graph["model"]["inputs"]["unet_name"], graph["clip"]["inputs"]["clip_name"],
                 graph["vae"]["inputs"]["vae_name"]}
    profile = setup.model_profile()
    assert {artifact.relative_path.rsplit("/", 1)[-1] for artifact in profile.artifacts} == filenames
