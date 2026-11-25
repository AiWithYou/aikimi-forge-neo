# https://github.com/ruwwww/ComfyUI-Spectrum-sdxl/blob/main/src/spectrum_node.py

import math

from .forecaster import ChebyshevForecaster, Spectrum


class SpectrumNode:
    @staticmethod
    def patch(model, steps, w, m, lam, window_size, flex_window, warmup_steps, stop_caching_step):
        step = 0
        window = float(window_size)
        cached = 0
        previous_sigma = None
        signature = None
        forecast = None
        split_pass = False
        new_model = model.clone()
        previous_wrapper = getattr(new_model, "model_options", {}).get("model_function_wrapper")

        def wrapper(model_function, args):
            nonlocal step, window, cached, previous_sigma, signature, forecast, split_pass
            x, timestep, conditions = args["input"], args["timestep"], args["c"]
            sigma = float(timestep.flatten()[0])
            options = conditions.get("transformer_options", {})
            branches = tuple(args["cond_or_uncond"])
            current = (x.shape, x.device, x.dtype, branches)

            if previous_sigma is None or sigma > previous_sigma or current != signature:
                forecast = None
                step, cached, window = 0, 0, float(window_size)
                split_pass = False
            # A second evaluation at one sigma may carry another AND condition.
            # Its history cannot be identified by the CFG branch alone.
            if sigma == previous_sigma or options.get("split_conditions") or len(set(branches)) != len(branches):
                split_pass = True
            signature, previous_sigma = current, sigma
            if split_pass or previous_wrapper is not None:
                forecast = None
                if previous_wrapper is not None:
                    return previous_wrapper(model_function, args)
                return model_function(x, timestep, **conditions)

            if forecast is None:
                forecast = Spectrum(ChebyshevForecaster(M=m, K=100, lam=lam, t_max=float(steps)), w)
            eligible = warmup_steps <= step < int(steps * stop_caching_step) and forecast.ready()
            if eligible and cached + 1 < max(1, math.floor(window)):
                result = forecast.predict(step).to(x)
                cached += 1
            else:
                result = model_function(x, timestep, **conditions)
                forecast.update(step, result)
                cached = 0
            if step >= warmup_steps:
                window += flex_window
            step += 1
            return result

        new_model.set_model_unet_function_wrapper(wrapper)
        return new_model
