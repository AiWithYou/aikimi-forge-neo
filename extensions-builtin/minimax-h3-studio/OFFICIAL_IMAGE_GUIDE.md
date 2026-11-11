# MiniMax-H3-Image

Forge の **H3 Image → MiniMax-H3-Image** で、MiniMax の静止画専用モデルによる生成と編集を行えます。最初に **環境とモデルを準備** を実行してください。Windows の共通 ComfyUI 0.45.0 環境を使用します。

- 既定は 2048×2048 です。4096×4096 を指定すると、そのサイズの潜在画像から直接生成します。高解像度や参照画像の追加に必要なメモリは実行環境によって変わります。
- 入力画像を追加すると画像編集になります。先頭が編集対象の `<Picture 1>` です。追加の参照画像は画面の並び順に `<Picture 2>` から最大 `<Picture 16>` まで使えます。
- サンプリングは6ステップ、Euler、simple、CFG 1です。プロンプトは入力した内容を直接渡します。
- 標準構成は Comfy-Org の INT8 画像本体、NVFP4 テキストエンコーダー、専用 FP16 画像VAEです。既存の同一エンコーダーは共用できます。初回に全ファイルを取得すると約37.8 GB、エンコーダーがあれば約22.1 GBです。
- 結果は `outputs/minimax_h3/official_h3_images/` に PNG と生成条件の JSON として保存します。

モデルと生成物には地域制限があり、EU・英国・韓国・米国は許諾地域から除外されています。利用や公開の前に[ライセンス原文](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)を確認してください。

[MiniMax のモデルカード](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) · [ComfyUI の公式ガイド](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)
