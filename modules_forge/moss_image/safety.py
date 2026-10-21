"""Conservative worker-local numerical policy; importing this module needs no Torch."""

from contextlib import contextmanager

from .core import MossImageError


class PolicyError(MossImageError):
    public_error_code = "unsupported_attention"


class InferencePolicy:
    ROWS_PER_CHUNK = 1024

    @classmethod
    def install(cls, torch, ops):
        existing = getattr(ops, "_moss_inference_policy", None)
        if existing is not None:
            existing.validate()
            return existing
        return cls(torch, ops)

    def __init__(self, torch, ops):
        from torch.nn.attention import SDPBackend, sdpa_kernel
        from torch.overrides import TorchFunctionMode

        self.torch, self.ops = torch, ops
        self.backends = [SDPBackend.EFFICIENT_ATTENTION]
        self.priority = ops.SDPA_BACKEND_PRIORITY
        self.priority[:] = self.backends
        self.original_linear = ops.linear_input_act
        self.linear_wrapper = self._linear
        ops.linear_input_act = self.linear_wrapper
        self.calls = self.situ_calls = 0
        self.dtypes = set()
        for name, enabled in (("flash", False), ("cudnn", False), ("math", False), ("mem_efficient", True)):
            getattr(torch.backends.cuda, f"enable_{name}_sdp")(enabled)
        sdpa = torch.nn.functional.scaled_dot_product_attention
        owner = self

        class Guard(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):
                kwargs = kwargs or {}
                if func is sdpa:
                    owner.validate()
                    args, kwargs = owner._expand_gqa(args, kwargs)
                    owner.calls += 1
                    query = args[0] if args else kwargs["query"]
                    owner.dtypes.add(str(query.dtype).removeprefix("torch."))
                    try:
                        with sdpa_kernel(owner.backends, set_priority=True):
                            return func(*args, **kwargs)
                    except RuntimeError as exc:
                        if "No available kernel" in str(exc) or "No viable backend" in str(exc):
                            raise PolicyError("この設定では指定のattention計算方式を使用できません。") from exc
                        raise
                return func(*args, **kwargs)

        self.guard = Guard
        ops._moss_inference_policy = self

    @staticmethod
    def _expand_gqa(args, kwargs):
        if not kwargs.get("enable_gqa", False):
            return args, kwargs
        values = [args[i] if i < len(args) else kwargs[name] for i, name in enumerate(("query", "key", "value"))]
        query, key, value = values
        if min(t.ndim for t in values) < 3:
            raise PolicyError("GQAの入力次元が一致しません。")
        heads, shared = query.shape[-3], key.shape[-3]
        if shared <= 0 or heads <= 0 or heads % shared or value.shape[-3] != shared:
            raise PolicyError("GQAのヘッド数が一致しません。")
        args, kwargs = list(args), dict(kwargs)
        if heads != shared:
            for index, name, tensor in ((1, "key", key), (2, "value", value)):
                expanded = tensor.repeat_interleave(heads // shared, dim=-3)
                if index < len(args):
                    args[index] = expanded
                else:
                    kwargs[name] = expanded
        kwargs["enable_gqa"] = False
        return tuple(args), kwargs

    def validate(self):
        flags = self.torch.backends.cuda
        if (
            self.ops.SDPA_BACKEND_PRIORITY is not self.priority
            or self.priority != self.backends
            or self.ops.linear_input_act is not self.linear_wrapper
            or flags.cudnn_sdp_enabled()
            or flags.math_sdp_enabled()
            or flags.flash_sdp_enabled()
            or not flags.mem_efficient_sdp_enabled()
        ):
            raise PolicyError("推論の計算方式が変更されたため処理を停止しました。")

    @contextmanager
    def enforce(self):
        self.validate()
        self.calls = self.situ_calls = 0
        self.dtypes = set()
        with self.guard():
            yield

    def _linear(self, linear, x, input_act=None, act_weight=None, act_eps=0.0, residual=None, residual_scale=None):
        if input_act != "situ":
            return self.original_linear(linear, x, input_act, act_weight, act_eps, residual, residual_scale)
        if x.shape[-1] % 2 or act_weight is not None or act_eps != 0:
            raise PolicyError("SiTU活性化の入力条件が一致しません。")
        flat = x.reshape(-1, x.shape[-1])
        output = x.new_empty((flat.shape[0], flat.shape[1] // 2))
        for start in range(0, flat.shape[0], self.ROWS_PER_CHUNK):
            hidden, gate = flat[start : start + self.ROWS_PER_CHUNK].float().chunk(2, dim=-1)
            activated = (4 * self.torch.tanh(gate / 4) * self.torch.sigmoid(gate)) * (25 * self.torch.tanh(hidden / 25))
            output[start : start + self.ROWS_PER_CHUNK] = activated.to(x.dtype)
        self.situ_calls += 1
        return self.original_linear(linear, output.reshape(*x.shape[:-1], -1), None, None, 0, residual, residual_scale)

    def snapshot(self):
        return {
            "sdpa_backends": ["efficient"],
            "situ_activation": "eager_fp32",
            "situ_rows_per_chunk": self.ROWS_PER_CHUNK,
            "sdpa_calls_checked": self.calls,
            "situ_eager_calls": self.situ_calls,
            "attention_dtypes": sorted(self.dtypes),
            "kernel_identity_observed": False,
        }
