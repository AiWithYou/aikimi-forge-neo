# 実機検証のチェックリスト

以下をWindowsのローカルCodex等へ渡し、実機で確認してください。ネットワーク・大容量取得・GPU使用が発生します。モデル利用条件は利用者本人が確認し、未承諾のまま `--accept-model-terms` を付けないでください。

```text
対象: AiWithYou/aikimi-studio-neo の neo
追加機能: YuE2 Music 0.1.0

1. git status と現在のブランチ・コミットを記録する。
   未コミット変更を削除しない。reset/clean/force pushを使わない。
   AGENTS.md等があれば先に読み、既存のForge/SenseNova/H3を置き換えない。

2. Python 3.12、NVIDIA GPU、VRAM、ドライバー、空きディスクを実測する。
   使用GPU・Torch/CUDA版・OS版・対象コミットを検証記録に残す。
   他の生成中ジョブを停止したり、勝手にドライバーを更新したりしない。

3. python -m pytest tests/yue2 -q を実行する。
   WindowsではPOSIXの模擬CLIテストはskipされる。そのskipをWindows合格と数えない。

4. 利用条件とモデル取得の了承後、専用setup.pyをPython 3.12で実行する。
   pip check、モデル/VAEリビジョン、package lockを記録する。
   Forge本体の環境・既存モデルを変更していないことを差分確認する。

5. 実際のForgeから起動する。既存タブの起動回帰を確認する。
   YuE2 Musicで曲調、短い自作英語歌詞、Seed 42、候補1、Steps 32、
   full、FP8 off、CPU退避 on、音声トークン上限9000を指定し、実生成する。
   音声を試聴し、48kHz/stereo、非空・非ゼロ、NaNなし、歌詞・終端を確認する。
   VRAMピーク・経過秒は実測値として記録し、想像で埋めない。

6. 楽譜のみを生成→ABCを編集欄へ移す→別曲調で新しい録音を作る。
   元の候補が変わらないこと、トークン上限警告、project.jsonの復元を確認する。
   候補2つ、A/B再生、64bit Seed、ABC import、JSON importを画面操作で確認する。

7. 読み込み/AR/NAR/復号中で停止を試す。
   Windows Job Objectが正常に割り当てられ、workerとその子プロセスが消えた後だけ
   GPUが次ジョブに返ることを確認する。強制終了で残るPID・GPU使用があれば修正する。
   Forgeを終了した場合も孤児workerが残らないことを確認する。
   別ブラウザーから他セッションのジョブを停止できないことも確認する。

8. Forge画像→YuE2→H3→YuE2の切替で、排他・VRAM残留・保存に問題がないか確認する。
   残留VRAMに対処するなら既存の安全なruntime解放APIを使う。
   関係ないComfyUI・Pythonプロセスを全停止するような実装はしない。

9. 任意: audio.cppの実バイナリ＋対応DLLを登録し、Q8で歌とABC条件付けを生成する。
   binary hashとモデルリビジョンを記録する。Q4/BF16は導入済みの場合のみ比較する。
   v0.8.0でLoRA・生成ABC出力が使えると決めつけない。
   16GB、FP8、日本語は個別の実験として記録し、24GB/BF16の結果と混同しない。

10. 実画面をデスクトップと狭幅で確認する。余計な横スクロール、ボタンの誤反応、
    遅延、Consoleエラー、誤った完了表示、読めない色、過密な余白を確認する。
    不具合は再現テストを追加して修正する。GPU未実行や聞いていない音質を合格にしない。
    差分と実行記録を残し、検証した環境・設定と未確認の範囲を区別する。
```
