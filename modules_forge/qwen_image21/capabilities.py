"""Sampling contracts shared by the UI request and the isolated worker."""

PRECISIONS = frozenset({"int8", "bf16", "w4a8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"})


def validate_sampling(values):
    precision = values.get("precision", "int8")
    if not isinstance(precision, str) or precision not in PRECISIONS:
        raise ValueError("モデル・精度の指定が不正です。")
    turbo = precision.startswith("turbo_")
    fun_acc = values.get("fun_acc", False)
    if not isinstance(fun_acc, bool):
        raise ValueError("Fun Accの指定が不正です。")
    if turbo and fun_acc:
        raise ValueError("TurboとFun Accは別の4-step方式です。同時には使用できません。")
    if (turbo or fun_acc) and values.get("steps", 40) != 4:
        raise ValueError("Turbo／Fun Accは4 stepsで生成してください。")
    if fun_acc and values.get("operation", "generate") != "generate":
        raise ValueError("Fun Accは画像生成で使用してください。")
    control = values.get("control_kind", "off") != "off" or bool(values.get("control_image"))
    if values.get("sparse_mode", "off") != "off" and (fun_acc or control):
        raise ValueError("Sparse AttentionはKVキャッシュを使います。Fun Acc・ControlNetとの併用はできません。")
    if values.get("local_model") and turbo:
        raise ValueError("Turboを使う場合は標準Turboモデルを選択してください。外部モデルと同時には指定できません。")
