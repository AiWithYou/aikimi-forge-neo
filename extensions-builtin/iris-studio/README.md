# Iris Studio

Iris-3Bの画像生成、相対深度推定、復元・4倍拡大を通常版／INT8で使用します。

`.\aikimi-iris-setup.bat`で画像生成用INT8を準備し、通常の`aikimi-launch.bat`で起動して上部の**Iris**を開きます。画面の**モデルを準備**でも選択中の処理に必要なモデルを取得できます。

通常版の例: `.\aikimi-iris-setup.bat --precision normal --task generate`。深度は`--task depth`、復元は`--task upscale`、全処理は`--task all`です。INT8は[Hugging Face](https://huggingface.co/Aikimi/iris-3b-int8)の固定commitから取得し、利用時に変換しません。

画像生成の既定は1024×1024・100 Steps・CFG 3。公式デモに合わせた7種類の生成サイズから選び、Seedを固定して同じ条件を再使用できます。深度はカラーPNGと元の浮動小数点NPY、復元はPNGを保存し、入力画像と比較できます。画像・条件・実測値は`outputs/iris/`へ保存します。

[検証・仕様](../../docs/iris-studio.md) · [公式モデル](https://huggingface.co/speridlabs/iris-3b)
