# Jev / Sparse 0.1.0 実装時検証

2026-09-21。基準リポジトリ: AiWithYou/aikimi-studio-neo@298baed5a82b698790374ee680ae85cd605d5812。

Linux / Python 3.13.5 / PyTorch 2.10.0+cpu / Gradio 6.5.1 / pytest 9.0.2。

実行コマンド:

```text
python -m compileall -q modules_forge/jev_sparse extensions-builtin/jev-sparse-experiments/scripts tools/setup_jev_sparse.py tools/report_jev_sparse.py
python -m pytest tests/jev_sparse -q --disable-warnings
79 passed
```

確認対象: 設定値とJev回答の厳密検査、呼び出し上限と周期、API失敗後の回路遮断、SDK timeout/cancel時の子プロセス終了、無効/固定モードの非通信、機密環境変数の子プロセス範囲、Animaの自己Attention限定パッチ・従来wrapper連鎖・終了時の解除、CPUテンソルの形状/出力投影、H3四段sampling接続、8/20 steps・別sampler・Control重複の拒否、原子的ノード導入とユーザー編集保護、Gradio実コンポーネントの構築、比較ログ集計。

Forge・ComfyUI・Jev APIの境界はモックです。CUDA用sol_attnカーネル自体の数値/速度検証ではありません。元リポジトリ全体の回帰テスト、Windows実機、実ブラウザー操作、実モデルのGPU生成、有料Jev API通信は未実行です。依存パッケージのネットワーク導入もdry-run以外は未実行です。

実画像・動画を生成した結果や高速化率は含みません。H3はsamplingのみ、Animaは測定用同期を含むモデル評価合計時間を記録し、通常の全生成時間と混同しません。
