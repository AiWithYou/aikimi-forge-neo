from collections.abc import Sequence
from typing import Any


def overlay_script_args(
    script_args: list[Any],
    args_from: int,
    args_to: int,
    provided: Sequence[Any] | None,
) -> None:
    """Overlay provided values without changing the global script-argument layout."""
    if args_from < 0 or args_to < args_from or args_to > len(script_args):
        raise ValueError("script argument range is outside the initialized layout")
    if not provided:
        return

    expected = args_to - args_from
    for index, value in enumerate(provided[:expected]):
        script_args[args_from + index] = value
