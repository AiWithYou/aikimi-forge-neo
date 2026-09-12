"""未修正の依存APIを、このアプリのモデル読込経路から遮断する。"""


def _reject_accelerate_checkpoint(*args, **kwargs):
    raise RuntimeError("AccelerateのcheckpointロードAPIは未修正の脆弱性があるため、このアプリでは使用できません。")


def restrict_accelerate_checkpoint_loading() -> None:
    """初期化時に公開別名も閉じる。init_empty_weightsなどはそのまま使える。"""
    import accelerate
    import accelerate.big_modeling
    import accelerate.utils
    import accelerate.utils.modeling

    for module, names in (
        (accelerate, ("load_checkpoint_in_model", "load_checkpoint_and_dispatch")),
        (accelerate.big_modeling, ("load_checkpoint_in_model", "load_checkpoint_and_dispatch")),
        (accelerate.utils, ("load_checkpoint_in_model",)),
        (accelerate.utils.modeling, ("load_checkpoint_in_model",)),
    ):
        for name in names:
            if not callable(getattr(module, name, None)):
                raise RuntimeError("AccelerateのAPI構成が変わっています。安全性の再確認が必要です。")
            setattr(module, name, _reject_accelerate_checkpoint)
