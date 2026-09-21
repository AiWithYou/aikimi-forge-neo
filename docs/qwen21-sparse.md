# Qwen Image 2.1 Sparse / Jev

[設定・APIキーの登録・比較手順](jev-sparse.md) / [実測結果](jev-sparse-validation.md)

通常起動したQwen Image 2.1タブの「生成設定 → Sparse Attention」でOFF・Dense・固定・数値ルール・Jevを切り替えます。APIキーは「実行環境 → Jev APIキー設定」で登録します。設定変更は次の生成から反映し、Jev専用ランチャーは必須ではありません。

Jevは速度優先で25/50/75/100%を選び、既定で1回だけ問い合わせます。固定モードの保持率はスライダーで変更できます。CLIの `aikimi-qwen21-sparse-launch.ps1` も起動時の初期値を指定する用途で使えます。

実装対象はDiffusers `6256aa7666cedd47443adc8f82da9a10e110b09c` のQwenImage21AttnProcessorです。初回のprefix計算とKVキャッシュは公式Processorに任せ、以後の生成画像側だけをblock-gather + SDPAで処理します。テキスト・参照画像の有効なKVとpaddingを保ちます。旧Qwenのdual-stream処理やH3のnative SLAとは別の実装です。

不要なtarget接続の計算は省きますが、ブロック選択やK/Vの取り出しにも時間がかかります。速度は実測で判断してください。INT8/BF16/W4A8と既存のGPU配置・CPU退避を使用でき、追加のモデル変換は行いません。

`request.json` に設定、`metadata.json` / `result.json` に実行回数と時間、ジョブ内 `jev-sparse` にJSONLを保存します。`attention_calls.sparse_target` が0ならSparseは実行されていません。Jevの原回答・信頼度・適用率も区別して記録します。

独自Processor、分散実行、未確認のworker改変は拒否します。終了・キャンセル・失敗時はProcessorとhookを復元します。API失敗・不正回答は残りをDenseに戻します。キー・画像・プロンプト・重みを判断用データへ混ぜません。

一次資料: [DiffusersのQwen 2.1実装](https://github.com/huggingface/diffusers/blob/6256aa7666cedd47443adc8f82da9a10e110b09c/src/diffusers/models/transformers/transformer_qwenimage21.py)
