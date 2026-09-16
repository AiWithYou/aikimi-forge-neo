# YuE2 Music for Aikimi Studio Neo · 0.1.0

Forgeの既存タブを置き換えず、**YuE2 Music**タブを追加します。モデルの重みは同梱しません。

## できること

- 歌詞・曲調から作曲。公式Python版は48 kHzステレオのFLAC・WAVを保存します。
- メロディ＋コード／メロディのみ／楽譜なしを選択。公式版では楽譜だけを先に生成できます。
- ABC楽譜の読み込み、編集、生成結果から編集欄への転送。元の録音は保持して別テイクを生成します。
- 1〜8候補の順次生成、A/B試聴、曲名・確定Seed付き履歴、入力復元、project.jsonの入出力。
- 公式版のCPU退避と実験的FP8 AR。別エンジンのaudio.cppでQ8_0／Q4_0／BF16 GGUFを選択できます。
- 専用プロセスの停止、アプリ内GPU排他、途中失敗時の完了済み候補の保持、トークン上限警告。

実機で確認した環境と結果は[Windows検証記録](../../docs/yue2-windows-validation.md)にまとめています。16GB VRAM、FP8、audio.cpp、日本語歌唱の品質は個別の確認が必要です。

## 導入

前提はPython 3.12・Git・NVIDIAドライバーです。Forge本体のPython 3.13とは別の環境を作ります。
公式の推奨出発点はLinux・BF16対応NVIDIA GPU・24GB VRAMです。Windowsではこの統合と実機の組み合わせを検証してください。

### 公式Python版

リポジトリ直下の `aikimi-yue2-setup.bat` を実行し、`1` を選択してください。利用条件とモデル取得を確認した後、導入が始まります。Python 3.12が見つからない場合は先にインストールしてください。

コマンドで実行する場合:

```powershell
py -3.12 modules_forge/yue2_studio/setup.py --engine official
```

インストーラーは専用venvにPyTorch 2.10.0（CUDA 12.8）と固定版YuE2を導入し、`pip check`を実行します。Hugging FaceのモデルとVAEをダウンロードし、それぞれのリビジョンと導入日時を `runtime.json` に保存します。取得後の生成はローカルファイルだけを使います。

普段使っている方法でForgeを起動し、**YuE2 Music → 実行環境・利用条件 → 導入状態を確認**へ進みます。導入状態の確認はGPU生成テストではありません。

### audio.cpp / GGUF版（任意）

[音声生成に対応したaudio.cppリリース](https://github.com/0xShug0/audio.cpp/releases/tag/v0.8.0)から、環境に合うCUDA版の `audiocpp_cli.exe` と必要なDLLを同じフォルダーに配置してください。このインストーラーは第三者の実行ファイルを自動取得・更新しません。

```powershell
py -3.12 modules_forge/yue2_studio/setup.py --engine cpp --binary "D:\audio.cpp\audiocpp_cli.exe" --gguf q8_0
```

必要なGGUF・VAE・sidecarだけを取得します。別形式を追加する場合は `--gguf q4_0` または `--gguf bf16` で再実行してください。既存モデルを使う場合:

```powershell
py -3.12 modules_forge/yue2_studio/setup.py --engine cpp --binary "D:\audio.cpp\audiocpp_cli.exe" --models "D:\Models\YuE2-GGUF" --gguf q8_0
```

生成設定で **audio.cpp · GGUF** と導入済み形式を選びます。CLIはv0.8.0のYuE2インターフェイスを参照していますが、実バイナリでのGPU完走は未検証です。コード・重みの組み合わせが異なる場合は[実機確認](HARDWARE_TEST.md)を実施してください。

## 作曲・編集

曲調は言語、ジャンル、楽器、声質、テンポを含めて指定します。歌詞には `[Verse]`、`[Chorus]` などの区間タグを使用できます。インストを試す場合は `[instrumental]` を指定します。専用インストLoRAは組み込んでいないため、声が入らないことは保証しません。

最初は「公式Python」「メロディ＋コード」「候補1」「Steps 32」「FP8オフ」で確認してください。Seed `-1` は実行時に確定し、履歴とproject.jsonに残ります。複数候補は確定Seedから1ずつ増やします。

アレンジは「この候補の楽譜 → この楽譜を編集に使う」でABCを入力欄へ移し、楽譜や曲調を修正して再生成します。**元音声の一部分だけを置き換える音声インペイントではありません。** 元の波形・歌声の同一性は保証しません。メロディのみの条件で伴奏を自由にする場合は、コード記号を除いたABCを使用してください。一般的なABC記法すべての互換性は保証せず、YuE2のVocal/Ins形式を基準にします。

入力復元は保存時の入力とSeedを復元します。保存された生成済み楽譜を使う操作とは別です。生成済み楽譜のABCを再入力すると再トークン化され、公式`SymbolicPlan.load()`による厳密なトークン再利用ではありません。別GPU・別依存版・別エンジン間の完全再現は保証しません。

音声トークン上限は秒数指定ではありません。公式版で上限に達した場合は履歴に警告します。audio.cpp v0.8.0では確実な上限到達情報が得られないため、「未判定」として曲の終端確認を求めます。

## 保存・停止

```text
outputs/yue2/<実行ID>/
  project.json             バッチに投入した入力
  status.json / worker.log 状態・実行ログ
  take-1/                  完了した候補（候補2以降も同様）
    audio.flac / audio.wav エンジンに応じた音声
    score.abc              楽譜がある場合
    project.json           この候補の入力と確定Seed
    studio-result.json     エンジン、上限到達、導入情報
    ...                    公式版はトークン・潜在表現・生成記録も保存
```

作成途中の候補は `.take-N.partial` に残し、完了した候補だけ履歴へ出します。再生成で既存の候補を上書きしません。失敗した実行はエラー欄のログパスから確認できます。履歴は直近60候補を表示します。

停止は現在のブラウザーセッションから投入したYuE2ジョブだけを対象にします。協調停止が進まなければ専用プロセスツリーを終了します。WindowsはJob Object、POSIXは専用プロセスグループと親パイプ監視を使います。Windowsではvenvランチャーの起動順に依存しないよう、実行Python自身がJobへの登録を確認してからモデルを読み込みます。GPUの所有権はworkerの終了確認後に返します。

ブラウザーを閉じても実行管理は継続し、完了結果を履歴で確認できます。**ページの再読み込み後に実行中ジョブへ再接続する機能は未実装**です。その場合、実行中の停止が必要ならForgeアプリ自体を終了してください。プロセス終了後も完了済み候補は残ります。PC電源断に対する完全な耐久性や途中生成の再開は保証しません。

アプリ内の既存GPU排他を利用しますが、別プロセスのComfyUIや他アプリのGPU使用を自動制御するものではありません。VRAM不足の場合は他のGPUアプリ・常駐モデルを解放し、CPU退避を有効にしてください。FP8 ARはCompute Capability 8.9以上に限定し、RTX 3090では拒否します。NARはBF16に戻るため、FP8を選んでも全工程の省メモリ・高速化は保証しません。

## 他の実装との関係

| 候補 | この統合で採用した範囲 | 採用しなかった範囲・理由 |
|---|---|---|
| 公式YuE2 | 段階別API、ABC条件付け、生成記録、CPU退避、FP8 AR | vLLM/TritonはWindowsや既存環境との依存リスクを避けて導入しない |
| audio.cpp | 別プロセスのGGUF実行、量子化切替、ABC入力 | v0.8.0後に追加されたAR LoRAや生成ABC書き出しを、安定版で動く機能として扱わない |
| ComfyUI-FL-YuE2 | ABCファイルを経由した連携先として案内 | ピアノロールや学習機能は同梱しない。H3用ComfyUIを更新して互換性を崩さない |
| SheetSage2 | 外部で採譜したYuE2用ABCの入力 | 音源アップロードからの自動採譜は未実装。別Python・PyTorch・FFmpeg環境が必要 |

音源からの自動採譜、MIDI入出力、GUIピアノロール、LoRA学習・適用、音声変換、任意プラグインの自動実行は、このバージョンには含まれません。

## 検証

```powershell
python -m pytest tests/yue2 -q
```

2026-09-16、Linux・Python 3.13.5・Gradio 6.5.1で**45件のCPUテストに合格**しました。入力境界、Seed精度、保存、所有者を限定した停止、実worker＋模擬CLIのプロセス連携、後続候補失敗時の保持、親パイプ切断時の終了、公式段階APIの模擬契約、Gradio画面構築・コールバック引数を確認しています。

模擬CLIの音声はテスト用の無音WAVであり、YuE2の実生成成功を示しません。公式段階APIテストもモデルを模擬しています。追加のWindows・ブラウザー・GPU検証結果は[Windows検証記録](../../docs/yue2-windows-validation.md)、再確認の項目は[HARDWARE_TEST.md](HARDWARE_TEST.md)を参照してください。YuE2のテストはGitHubのCPU／Windowsワークフローでも実行します。

## 利用条件・出典

モデルのCC BY-NC 4.0表示と、個人による出力収益化を認める公式組織の回答を分けて案内しています。企業サービスや重みの再配布まで一律に無制限と解釈せず、実際の用途について権利者の条件を確認してください。追加モデルには独自の条件があります。

詳細な出典とライセンスは[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)を参照してください。Aikimi側の追加コードにはリポジトリのLICENSEが適用されます。
