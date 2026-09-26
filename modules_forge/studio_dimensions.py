"""Validation shared by explicit Studio canvas controls; never rounds user input."""

import math


def custom_dimensions(width, height, *, minimum=256, maximum=4096, grid=32, max_pixels=None):
    result = []
    for label, value in (("幅", width), ("高さ", height)):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{label}を入力してください。") from None
        if isinstance(value, bool) or not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"{label}は整数で入力してください。")
        number = int(number)
        if not minimum <= number <= maximum or number % grid:
            raise ValueError(f"{label}は{minimum}〜{maximum} px、{grid}の倍数で指定してください。")
        result.append(number)
    if max_pixels is not None and result[0] * result[1] > max_pixels:
        raise ValueError(f"合計{max_pixels / 1_000_000:.2f} MP以内になるよう、幅か高さを小さくしてください。")
    return tuple(result)
