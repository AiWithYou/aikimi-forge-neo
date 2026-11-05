"""P-H3 image output validation and atomic publication."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from modules_forge import p_h3_image as ph3


def test_output_is_limited_to_the_expected_runtime_output_directory(tmp_path):
    root = tmp_path / "runtime"
    path = root / "output/image/result.png"
    path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 32), "red").save(path)
    request = ph3.PH3Request("test", width=32, height=32, use_upsampler=False)
    item = {"filename": "result.png", "subfolder": "image", "type": "output"}
    history = {"job": {"outputs": {"save": {"images": [item]}}}}
    assert ph3.extract_output(history, "job", root, request) == path.resolve()
    for changes in ({"filename": "../result.png"}, {"subfolder": "../outside"}, {"type": "temp"}):
        history["job"]["outputs"]["save"]["images"] = [{**item, **changes}]
        with pytest.raises(ph3.PH3Error):
            ph3.extract_output(history, "job", root, request)


def test_save_validates_pixels_before_publishing_and_keeps_prompt_metadata(tmp_path):
    source = tmp_path / "source.png"
    Image.new("RGB", (32, 32), "red").save(source)
    ready = SimpleNamespace(core_revision="abc", comfy_version="test", package_versions={}, runtime_profile="fast")
    request = ph3.PH3Request("original", width=32, height=32, use_upsampler=True)
    result = ph3.save_result(source, request, 42, "job", ready, tmp_path / "results", "expanded", None)
    assert Path(result["path"]).is_file()
    assert result["metadata"]["prompt"] == "original"
    assert result["metadata"]["upsampled_prompt"] == "expanded"
    assert result["metadata"]["width"] == 32
    assert not list((tmp_path / "results").glob(".p-h3-*"))
    bad = ph3.PH3Request("original", width=64, height=32, use_upsampler=False)
    with pytest.raises(ph3.PH3Error):
        ph3.save_result(source, bad, 42, "job", ready, tmp_path / "results", None, None)
    assert len(list((tmp_path / "results").glob("P_H3_Image_*"))) == 1
    assert not list((tmp_path / "results").glob(".p-h3-*"))
