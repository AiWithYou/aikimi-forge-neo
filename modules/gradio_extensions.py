import inspect
import warnings
from functools import wraps

import gradio as gr
import gradio.blocks
import gradio.component_meta
import gradio.events

from modules import patches, scripts, ui_tempdir


class GradioDeprecationWarning(DeprecationWarning):
    pass


def add_classes_to_gradio_component(comp: gr.components.Component):
    """
    this adds gradio-* to the component for css styling (ie gradio-button to gr.Button), as well as some others
    """

    comp.elem_classes = [f"gradio-{comp.get_block_name()}", *(getattr(comp, "elem_classes", None) or [])]

    if getattr(comp, "multiselect", False):
        comp.elem_classes.append("multiselect")


def IOComponent_init(self, *args, **kwargs):
    self.webui_tooltip = kwargs.pop("tooltip", None)

    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_IOComponent_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Block_get_config(self, *args, **kwargs):
    config = original_Block_get_config(self, *args, **kwargs)

    webui_tooltip = getattr(self, "webui_tooltip", None)
    if webui_tooltip:
        config["webui_tooltip"] = webui_tooltip

    config.pop("example_inputs", None)

    return config


def BlockContext_init(self, *args, **kwargs):
    if scripts.scripts_current is not None:
        scripts.scripts_current.before_component(self, **kwargs)

    scripts.script_callbacks.before_component_callback(self, **kwargs)

    res = original_BlockContext_init(self, *args, **kwargs)

    add_classes_to_gradio_component(self)

    scripts.script_callbacks.after_component_callback(self, **kwargs)

    if scripts.scripts_current is not None:
        scripts.scripts_current.after_component(self, **kwargs)

    return res


def Blocks_get_config_file(self, *args, **kwargs):
    config = original_Blocks_get_config_file(self, *args, **kwargs)

    for comp_config in config["components"]:
        if "example_inputs" in comp_config:
            comp_config["example_inputs"] = {"serialized": []}

    return config


def BlocksConfig_set_event_trigger(self, *args, **kwargs):
    bound = blocks_config_set_event_trigger_signature.bind_partial(self, *args, **kwargs)
    if "api_name" not in bound.arguments and "api_visibility" not in bound.arguments:
        kwargs["api_visibility"] = "private"
    return original_BlocksConfig_set_event_trigger(self, *args, **kwargs)


def Blocks_init(self, *args, **kwargs):
    result = original_Blocks_init(self, *args, **kwargs)
    self.load = EventWrapper(self.load)
    return result


original_IOComponent_init = patches.patch(
    __name__, obj=gr.components.Component, field="__init__", replacement=IOComponent_init
)
original_Block_get_config = patches.patch(
    __name__, obj=gradio.blocks.Block, field="get_config", replacement=Block_get_config
)
original_BlockContext_init = patches.patch(
    __name__, obj=gradio.blocks.BlockContext, field="__init__", replacement=BlockContext_init
)
original_Blocks_get_config_file = patches.patch(
    __name__, obj=gradio.blocks.Blocks, field="get_config_file", replacement=Blocks_get_config_file
)
original_BlocksConfig_set_event_trigger = patches.patch(
    __name__, obj=gradio.blocks.BlocksConfig, field="set_event_trigger", replacement=BlocksConfig_set_event_trigger
)
blocks_config_set_event_trigger_signature = inspect.signature(original_BlocksConfig_set_event_trigger)
original_Blocks_init = patches.patch(__name__, obj=gradio.blocks.Blocks, field="__init__", replacement=Blocks_init)


ui_tempdir.install_ui_tempdir_override()


def gradio_component_meta_create_or_modify_pyi(component_class, class_name, events):
    # Runtime type-stub generation writes into the checkout and is not needed
    # for packaged components. Gradio 6 otherwise creates modules/*.pyi while
    # importing WebUI subclasses.
    return None


# this prevents creation of .pyi files in webui dir
patches.patch(__file__, gradio.component_meta, "create_or_modify_pyi", gradio_component_meta_create_or_modify_pyi)

# this function is broken and does not seem to do anything useful
gradio.component_meta.updateable = lambda x: x


def _prepare_event_kwargs(kwargs):
    if "_js" in kwargs:
        kwargs["js"] = kwargs.pop("_js")
    if "api_name" not in kwargs and "api_visibility" not in kwargs:
        kwargs["api_visibility"] = "private"
    return kwargs


class EventWrapper:
    def __init__(self, replaced_event):
        self.replaced_event = replaced_event
        self.has_trigger = getattr(replaced_event, "has_trigger", None)
        self.event_name = getattr(replaced_event, "event_name", None)
        self.callback = getattr(replaced_event, "callback", None)
        self.real_self = getattr(replaced_event, "__self__", None)

    def __call__(self, *args, **kwargs):
        return self.replaced_event(*args, **_prepare_event_kwargs(kwargs))

    @property
    def __self__(self):
        return self.real_self


original_gradio_on = gr.on


@wraps(original_gradio_on)
def gradio_on(*args, **kwargs):
    return original_gradio_on(*args, **_prepare_event_kwargs(kwargs))


gr.on = gradio_on
gradio.events.on = gradio_on


def repair(grclass):
    if not getattr(grclass, "EVENTS", None):
        return

    @wraps(grclass.__init__)
    def __repaired_init__(self, *args, tooltip=None, source=None, original=grclass.__init__, **kwargs):
        if source:
            kwargs["sources"] = [source]

        allowed_kwargs = inspect.signature(original).parameters
        fixed_kwargs = {}
        for k, v in kwargs.items():
            if k in allowed_kwargs:
                fixed_kwargs[k] = v
            else:
                warnings.warn(
                    f"unexpected argument for {grclass.__name__}: {k}", GradioDeprecationWarning, stacklevel=2
                )

        original(self, *args, **fixed_kwargs)

        self.webui_tooltip = tooltip

        for event in self.EVENTS:
            replaced_event = getattr(self, str(event))
            fun = EventWrapper(replaced_event)
            setattr(self, str(event), fun)

    grclass.__init__ = __repaired_init__
    grclass.update = gr.update


for component in set(gr.components.__all__ + gr.layouts.__all__):
    repair(getattr(gr, component, None))


class Dependency(gradio.events.Dependency):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for event_name in ("then", "success", "failure"):
            original_event = getattr(self, event_name)

            @wraps(original_event)
            def chained_event(*xargs, _event=original_event, **xkwargs):
                return _event(*xargs, **_prepare_event_kwargs(xkwargs))

            setattr(self, event_name, chained_event)


gradio.events.Dependency = Dependency
gr.Box = gr.Group
