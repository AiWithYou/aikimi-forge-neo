# Jev / Sparse Attention

H3・Anima・Qwen Image 2.1の任意の比較機能です。通常生成はOFFが既定です。

- [使い方・キーの登録](../../docs/jev-sparse.md)
- [GPU実測と検証範囲](../../docs/jev-sparse-validation.md)
- [Qwen 2.1の適用範囲](../../docs/qwen21-sparse.md)

Animaの自己Attentionにはcomfy-kitchen 0.2.33のCUDAカーネル、H3にはComfyUIのnative SLA、Qwenには専用workerのblock-gather + SDPAを使います。Jevは層別の保持率を選ぶ制御部分です。

速度優先版のJevは既定で1回だけ問い合わせ、画像モデルでは25/50/75/100%、H3では1/3/5/10%から選びます。構図や細部の違いは許容する設計です。有効な回答を信頼度だけで100%へ置き換えず、回答・信頼度・適用値を記録します。APIエラー時の退避、キャンセル、呼び出し上限は維持します。

各モデルにAPIキーの入力・保存欄があります。キーはユーザー領域に保存し、WindowsではDPAPIで暗号化します。Git・生成履歴・比較ログには保存しません。通常・固定・数値ルールはAPIを呼びません。

H3はFused Turbo・4 steps・res_multistep限定です。既存の標準H3環境に `aikimi-jev-setup.bat --h3` でノードを追加できます。通常のH3モデルを4 stepsに自動変更する機能ではありません。

オフラインテスト:

```powershell
venv\Scripts\python.exe -m pytest tests/jev_sparse -q
```

H3ノードはsepiablue-aiのnative_sla実装を参考にしたGPL-3.0-onlyコードです。`LICENSE.h3` と各ソースの帰属表記を維持しています。
