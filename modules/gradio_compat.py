"""Small compatibility contracts for the audited Gradio runtime."""

SUPPORTED_GRADIO_VERSION = "6.17.3"


def keep_hidden_component_mounted(visible):
    """Explicitly opt into the Gradio 6.6-6.17 visibility workaround."""

    return "hidden" if visible is False else visible


def normalize_single_selection(value) -> str:
    """Reject list-shaped values sent to a single-select Gradio callback."""

    return value if isinstance(value, str) else ""


def normalize_unit_interval(start: float, end: float) -> tuple[float, float]:
    """Clamp two values to [0, 1] and return them in ascending order."""

    start = max(0.0, min(float(start), 1.0))
    end = max(0.0, min(float(end), 1.0))
    return min(start, end), max(start, end)
