# 変更履歴

## v1.0.0 — 2026-09-12

Aikimi Studio Neoの初回バージョンです。Forge Neoの画像生成を基盤に、次の機能をまとめています。

- Krea2の導入支援、Anima 3.8B v1.1拡張とINT8変換、SenseNova U1.5 Studio、MiniMax H3 Studio。
- モデルを選んで本体・専用環境を準備する`aikimi-setup.bat`と、通常起動用の`aikimi-launch.bat`。
- HyperWeave、Grain Cleaner、Color Flatten、CD Tunerなどの画像仕上げ機能。
- 追加機能へのナビゲーション、実行状態の表示、設定保存の保護、GPUの使用権管理とジョブ復旧。

今回、READMEをモデル対応、画像生成と全体の操作、拡張・仕上げの順に整理し、初回セットアップを案内しました。画面下部とDiagnosticsには`v1.0.0`を表示します。

Forge Neoの最新更新を確認し、未対応GPUでのPyTorch起動エラーの案内を取り込みました。反映済みの修正と見送った更新は[確認記録](docs/upstream-sync.md)を参照してください。

実験機能やモデルごとの導入条件は[README](README.md)と各機能のガイドに記載しています。コードはAGPL-3.0、モデルとブランド素材はそれぞれの配布条件に従います。
