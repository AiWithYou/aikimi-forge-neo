"""見本プレビュー上の対角2点を、原寸の矩形へ変換する。"""

import gradio as gr
from PIL import ImageDraw, ImageOps


def make_reference_preview(image):
    if image is None:
        return None, None, None, None, "画像を読み込んでください", ""
    source = ImageOps.exif_transpose(image)
    size = source.size
    source.thumbnail((768, 768))
    preview = source.convert("RGB")
    return preview, preview.copy(), size, None, f"原寸 {size[0]} × {size[1]} px。対角の2点を順に選択してください。", ""


def select_reference(preview, original_size, corner, upscale_enabled, evt: gr.SelectData):
    if upscale_enabled:
        gr.Warning("画像上の見本選択ではUpscaleをオフにしてください。拡大と併用する場合は処理後の座標を入力します。")
        return gr.skip(), None, gr.skip(), "Upscaleをオフにすると画像上で選択できます。"
    if preview is None or original_size is None:
        return gr.skip(), None, "", "画像を読み込んでください。"
    point = tuple(int(v) for v in evt.index)
    if len(point) != 2 or not 0 <= point[0] < preview.width or not 0 <= point[1] < preview.height:
        raise gr.Error("選択座標が画像の範囲外です")
    selected = preview.copy()
    draw = ImageDraw.Draw(selected)
    if corner is None:
        x, y = point
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ff7800")
        return selected, point, "", "1点目を選びました。対角の点を選んでください。"
    left, right = sorted((corner[0], point[0]))
    top, bottom = sorted((corner[1], point[1]))
    # 画素中心ではなく選択画素の外縁を原寸へ写す。
    x = int(left * original_size[0] / preview.width)
    y = int(top * original_size[1] / preview.height)
    end_x = min(original_size[0], int((right + 1) * original_size[0] / preview.width))
    end_y = min(original_size[1], int((bottom + 1) * original_size[1] / preview.height))
    if min(end_x - x, end_y - y) < 32:
        return preview, None, "", "範囲が小さすぎます。原寸で縦横32画素以上になるように選び直してください。"
    draw.rectangle((left, top, right, bottom), outline="#ff7800", width=3)
    roi = f"{x}, {y}, {end_x - x}, {end_y - y}"
    return (
        selected,
        None,
        roi,
        f"原寸 {original_size[0]} × {original_size[1]} px ／ 選択: {roi}。次のクリックで選び直せます。",
    )
