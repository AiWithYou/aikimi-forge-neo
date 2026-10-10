"""Sampling contracts shared by the UI request and the isolated worker."""

OFFICIAL_TURBO_PRECISIONS = {
    "turbo_official_int8": "int8",
    "turbo_official_w4a8": "w4a8",
    "turbo_official_bf16": "bf16",
}
PRECISIONS = frozenset({"int8", "bf16", "w4a8", "base_q4_k_m", "turbo_bf16", "turbo_q4_k_m"}) | frozenset(
    OFFICIAL_TURBO_PRECISIONS
)


def recommended_steps(precision, fun_acc=False):
    if fun_acc:
        return 4
    if precision in OFFICIAL_TURBO_PRECISIONS:
        return 8
    if precision in {"turbo_bf16", "turbo_q4_k_m"}:
        return 4
    return None


def sampling_recommendations(values):
    notices = []
    precision = values.get("precision", "int8")
    fun_acc = values.get("fun_acc", False)
    recommended = recommended_steps(precision, fun_acc)
    if recommended is not None and values.get("steps", 40) != recommended:
        notices.append(f"このTurboは{recommended} stepsを推奨します。指定したStepsで実行します。")
    control = values.get("control_kind", "off") != "off"
    if fun_acc and recommended_steps(precision) is not None:
        notices.append("TurboとFun Accの併用は画質実験です。Fun Accの4 steps samplerで実行します。")
    elif control and recommended_steps(precision) is not None:
        notices.append("TurboとControlNetの併用は画質実験です。指定した設定で実行します。")
    if values.get("sparse_mode", "off") != "off" and (fun_acc or control):
        notices.append("ControlNet・Fun Accとの併用ではSparseが働かず、通常Attentionで実行します。")
    if control and not 0 <= values.get("control_strength", 1.0) <= 2:
        notices.append("ControlNetの強度の目安は0〜2です。指定した強度で実行します。")
    return notices


def quantization_precision(precision):
    return OFFICIAL_TURBO_PRECISIONS.get(precision, precision)


def validate_sampling(values):
    precision = values.get("precision", "int8")
    if not isinstance(precision, str) or precision not in PRECISIONS:
        raise ValueError("モデル・精度の指定が不正です。")
    turbo = recommended_steps(precision) is not None
    fun_acc = values.get("fun_acc", False)
    if not isinstance(fun_acc, bool):
        raise ValueError("Fun Accの指定が不正です。")
    if fun_acc and values.get("steps", 40) != 4:
        raise ValueError("現在のFun Acc samplerは4 steps専用です。4 stepsで生成してください。")
    if fun_acc and values.get("operation", "generate") != "generate":
        raise ValueError("Fun Accは画像生成で使用してください。")
    if values.get("local_model") and turbo:
        raise ValueError("Turboを使う場合は標準Turboモデルを選択してください。外部モデルと同時には指定できません。")
