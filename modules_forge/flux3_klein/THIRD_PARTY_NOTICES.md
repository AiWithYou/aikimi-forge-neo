# FLUX 3 reference preprocessing

`conditioning.py` adapts bounding-box planning and reconciliation, image preparation,
and prompt-upscaler request formatting from Black Forest Labs' FLUX 3 Image Klein
reference inference code:

- Source: https://github.com/black-forest-labs/flux-3-image-klein
- Revision: `45e57c9fb06f9751cdced29a65c5d2171003be05`
- License: Apache License 2.0; the full text is in `LICENSE.bfl-reference`.

The image-model weights have their own FLUX Kommunity License. They are not
distributed with this adapter. The separate ComfyUI runtime retains its own license.
