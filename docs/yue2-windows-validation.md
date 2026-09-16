# YuE2 MusicのWindows検証

検証日: 2026年9月16日〜17日。対象はAikimi Studio Neoの`neo`ブランチです。

## 環境と導入

- Windows 11、NVIDIA GeForce RTX 3090（24 GiB）、システムRAM 64 GiB。
- Forge本体: Python 3.13.14、PyTorch 2.11.0+cu130、Gradio 6.17.3。
- YuE2専用環境: Python 3.12.13、PyTorch 2.10.0+cu128、yue2-infer 0.1.6。
- `pip check`に合格。生成と同じ環境変数で`YuE2ForCausalLM`をインポートでき、CUDAを認識。
- ソース: `4d53bd5fc7e96a53cb907d3eb407a65df67a8b79`。
- YuE2-3B: `14fc6c6f146441b1dd6363fcb2e01e82a6914cb7`。
- YuE2-Vae: `9a94e1d0ea9f8087e98f77fa88df4a4068104d2a`。

導入先は`extensions-builtin/yue2-studio/runtime/`。モデルと専用環境はGitの対象外です。Forge本体とSenseNova／H3の依存定義は変更していません。

## 実機で見つけて修正した不具合

- 未操作のABC欄がGradioから`None`として渡され、通常の作曲が入力エラーになる問題を修正。未入力の曲調には、曲調を入力するよう表示します。
- Windowsのvenvランチャーから実Pythonが先に起動すると、後からランチャーをJob Objectへ割り当てても実Pythonが保護されない競合を再現。実Pythonが名前付きJobへ自分を登録してからモデルをロードするよう修正しました。
- Windowsで別スレッドがCRTの標準入力を読み続けると、NumPyのDLL初期化が待ち続ける現象を実ジョブで確認。Windowsの親終了検知はJob Objectのkill-on-closeを使い、POSIXのみパイプ監視を使います。
- 環境変数の制限から`USERNAME`が抜けており、PyTorchのキャッシュ初期化が`getpass.getuser()`から`pwd`を読み込もうとして失敗する問題を修正。セットアップ時にも実モデルクラスを同じ環境でインポートして確認します。

## 自動テストと画面

- `python -m pytest tests/yue2 -q`: **44件合格、5件skip**。skipはPOSIX用模擬CLIのテストです。
- Windowsでは、起動直後／起動を先行させた場合の両方でNumPy初期化、対象の親子プロセス停止、Jobハンドルを閉じたときの終了、無関係なプロセスの維持を確認。
- 既存のCPU回帰スイート: **1,381件実行、失敗なし**（43 skip、1 expected failure）。GPU所有権とCI経路の追加確認も8件合格。
- 通常のForgeからYuE2タブを表示し、曲調・歌詞・設定・入力チェックを操作。デスクトップと狭幅で目視確認し、狭幅でページ全体の`scrollWidth == clientWidth`を確認。
- YuE2のテストをGitHubのCPU／Windowsワークフローに追加。

## 確認範囲

音質、日本語歌唱、16GB GPU、FP8、audio.cpp／GGUFの実生成、Forge画像・SenseNova・H3との連続したGPU実生成切替は、この記録の合格範囲には含めません。模擬テストの無音WAVも実生成の証拠には数えません。
