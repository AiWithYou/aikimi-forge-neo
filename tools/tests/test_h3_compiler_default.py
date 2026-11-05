"""The shared runtime must not activate the unpatched H3 compiler."""

from pathlib import Path

import pytest

from modules_forge import minimax_h3_bridge as bridge
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_negpip import H3NegPiP


def test_compiler_is_off_for_default_controlnet_and_negpip():
    options = (H3Acceleration(), H3Acceleration(negpip=H3NegPiP(enabled=True)))
    for option in options:
        assert option.compiler_mode == "off"
        assert "--disable-comfy-compiler" in bridge._runtime_command(Path("python"), 8189, acceleration=option)
    assert H3Acceleration.from_values(H3Acceleration().values()[:8]).compiler_mode == "off"


def test_explicit_compiler_on_is_rejected_before_start():
    with pytest.raises(ValueError):
        H3Acceleration(compiler_mode="on").validate()
    with pytest.raises(ValueError):
        bridge._runtime_command(Path("python"), 8189, acceleration=H3Acceleration(compiler_mode="on"))
