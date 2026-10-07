"""Open the same Clef view locally without loading Forge's diffusion UI."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from modules_forge.clef.core import OUTPUTS  # noqa: E402
from modules_forge.clef.ui import build_ui  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()
    with gr.Blocks(title="Clef · 画像・文章・JSONの判定") as app:
        build_ui()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
        css=(ROOT / "extensions-builtin/clef-studio/style.css").read_text(),
        allowed_paths=[str(OUTPUTS.resolve())],
        show_error=True,
    )
