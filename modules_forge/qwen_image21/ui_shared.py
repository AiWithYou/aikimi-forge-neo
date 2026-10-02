"""Build both Qwen interfaces together; Gradio keeps their input state per client."""

_outpaint_builder = None


def register_outpaint(builder):
    global _outpaint_builder
    _outpaint_builder = builder


def build_outpaint(controls):
    return _outpaint_builder(controls) if _outpaint_builder is not None else []
