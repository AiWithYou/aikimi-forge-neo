"""Sampling contracts shared by the UI request and the isolated worker."""

OFFICIAL_TURBO_PRECISIONS = {
    "turbo_official_int8": "int8",
    "turbo_official_w4a8": "w4a8",
    "turbo_official_bf16": "bf16",
}
PRECISIONS = frozenset({"int8", "bf16", "w4a8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"}) | frozenset(
    OFFICIAL_TURBO_PRECISIONS
)


def fixed_steps(precision, fun_acc=False):
    if precision in OFFICIAL_TURBO_PRECISIONS:
        return 8
    if precision in {"turbo_bf16", "turbo_q4_k_m"} or fun_acc:
        return 4
    return None


def quantization_precision(precision):
    return OFFICIAL_TURBO_PRECISIONS.get(precision, precision)


def validate_sampling(values):
    precision = values.get("precision", "int8")
    if not isinstance(precision, str) or precision not in PRECISIONS:
        raise ValueError("モデル・精度の指定が不正です。")
    turbo = fixed_steps(precision) is not None
    fun_acc = values.get("fun_acc", False)
    if not isinstance(fun_acc, bool):
        raise ValueError("Fun Accの指定が不正です。")
    if turbo and fun_acc:
        raise ValueError("TurboとFun Accは別の高速化方式です。同時には使用できません。")
    required_steps = fixed_steps(precision, fun_acc)
    if required_steps is not None and values.get("steps", 40) != required_steps:
        raise ValueError(f"Turbo／Fun Accは{required_steps} stepsで生成してください。")
    if fun_acc and values.get("operation", "generate") != "generate":
        raise ValueError("Fun Accは画像生成で使用してください。")
    control = values.get("control_kind", "off") != "off" or bool(values.get("control_image"))
    if values.get("sparse_mode", "off") != "off" and (fun_acc or control):
        raise ValueError("Sparse AttentionはKVキャッシュを使います。Fun Acc・ControlNetとの併用はできません。")
    if values.get("local_model") and turbo:
        raise ValueError("Turboを使う場合は標準Turboモデルを選択してください。外部モデルと同時には指定できません。")
