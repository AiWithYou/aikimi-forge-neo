# Jev / Sparse Attention 実験 0.1.0

対象は H3 の4ステップ動画生成と Anima の画像自己Attentionです。Qwen Image 2.1、SDXL、Krea2、SenseNovaはこの実装の対象ではありません。初期値は無効です。既存ソース・重みを上書きせず、通常の生成経路は従来の関数へ委譲します。

Neo基準: `298baed5a82b698790374ee680ae85cd605d5812`。この追加実装はGPU速度・画質を検証するためのものです。高速化率や画質同等性を保証しません。H3の公開実験を参考にしていますが、判断指示やログ、固定比較はNeo用に変更しており、41.7%短縮の厳密な再現実装ではありません。

## 導入

ブランチ `experiment/jev-sparse-h3-anima` を取得してNeoを終了します。未保存の作業を上書きする `reset --hard` は不要です。

```powershell
git fetch origin
git switch --track origin/experiment/jev-sparse-h3-anima
```

GitHubから取得したZIPを使う場合はリポジトリ全体です。会話添付の追加ファイルZIPを使う場合は、展開した `modules_forge`・`extensions-builtin`・`tools` と起動ファイルをNeoのルートへ配置してください。同名の追加ファイルが既にある場合は変更を確認してから更新してください。

### H3の実験環境

通常用ComfyUIを勝手に更新しません。既存のH3モデルフォルダーを共有した、別のComfyUI/Python環境を作ります。Python 3.12とGitが必要です。

```powershell
# 通信をしない固定比較だけならSDKは不要
py -3.12 tools/setup_jev_sparse.py --create-h3-runtime --models "D:\Models\MiniMax-H3"

# Jevも使う場合だけ追加
py -3.12 tools/setup_jev_sparse.py --sdk
```

モデルを標準の `models/MiniMax-H3` に保存している場合は `--models` を省略できます。`aikimi-jev-setup.bat` はSDKと実験用ComfyUIの両方を準備します。モデル重みとAPIキーはダウンロード・生成しません。Fused Turbo、Qwen encoder、video/audio VAEは既存のH3機能と同じ場所に用意してください。

実験用ComfyUIは `repositories/minimax-h3-jev/ComfyUI`、専用Pythonはその親の `.venv` です。ComfyUIは `7a0b5eede3f9721c8faab290689893f36edc6d66` を固定し、Sparse内部APIのファイル内容も検査します。GPU依存パッケージすべてのハッシュ固定ではありません。実際の依存版は `repositories/minimax-h3-jev/installed-packages.txt` に記録されます。

セットアップ失敗後は不完全な既存環境を無断で消去・再利用せず停止します。ログと既存ディレクトリを確認してください。コード更新後、実験環境のノードだけを更新するには、Neo/ComfyUIを終了して次を使います。

```powershell
py -3.12 tools/setup_jev_sparse.py --h3 --comfy-root "D:\aikimi-studio-neo\repositories\minimax-h3-jev\ComfyUI"
```

ノード内のユーザー編集、未知のファイル、リンクを検出した場合は上書きしません。

### Animaのスパースカーネル

Forgeが実際に使うPython環境で、Comfy-Kitchenの `sol_attn` がGPUに対応している必要があります。例えば標準venvでは次で確認できます。

```powershell
.\venv\Scripts\python.exe -c "import torch, comfy_kitchen as ck; print(torch.cuda.is_available()); print(ck.sol_attn_is_available(torch.device('cuda:0')))"
```

接続仕様はComfy-Kitchen 0.2.34を使う固定ComfyUIのコードに基づきます。すでに入っているForge環境のパッケージは自動で更新・降格しません。関数がない・GPU非対応の場合はSparseを開始せずエラーにします。Dense比較と通常生成はこのカーネルを要求しません。

## 使用方法

### API通信なしで比較する

通常の `aikimi-launch.bat` で起動します。

H3 Studioの「実行環境とモデル」に、これまで非表示だった実行環境パスが表示されます。実験用ComfyUIの絶対パスを指定します。通常用の値は勝手に変更しません。別環境が同じポートを使用している場合は停止し、Neoを再起動してから切り替えます。

「高速化」で `Turbo + 4 Steps` を適用し、CLIPキャッシュ・NegPiP・長尺・Fun ControlNetをOFFにします。Attentionに追加された次のモードから選び、「選択設定で再起動」を実行します。

| モード | 処理 |
|---|---|
| H3比較 Dense | 4ステップの通常Attentionを計測。 |
| H3固定SLA 5% | 全4ステップで保持率5%。SDK/API呼び出しゼロ。 |
| H3固定SLA 10% | 全4ステップで保持率10%。SDK/API呼び出しゼロ。 |
| H3 Jev制御SLA | 初回50層、以後49層を一括判断。最大4回のAPI呼び出し。 |

H3 Image、8/20ステップ、別samplerは拒否します。全50層は実行します。音声・条件保護、短い系列の扱いはnative SLAに従います。`min_tokens=12288`、追加token補助0、開始から適用します。既存UIの固定SLA用の保持率/開始位置は使いません。

Animaは通常のtxt2img/img2img画面にある「Anima Self-Attention・実験」で選びます。OFF、Dense、固定SLA、数値ルール、Jev制御を用意しています。固定の初期保持率は75%、数値ルール/Jevは50・75・100%から選びます。100%は従来のDense処理です。

Animaの対象は生成Transformerの `self_attn.compute_attention` のみです。Cross-Attention、テキストエンコーダー、Q/K/V投影、正規化、RoPE、重みは変えません。参照latentを連結する編集は初版では拒否します。通常のimg2imgと参照latent条件は別です。対応head dimensionは128です。

最初のDense評価回数、変更間隔、問い合わせ上限を指定できます。ここでの「モデル評価回数」はsamplerのStepsとは異なり、CFG分岐や追加評価で増えることがあります。前ジョブの判断や観測を持ち越さず、キャンセル・例外時にも有効化コンテキストを解除します。

### Jevを使う

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\aikimi-jev-launch.ps1
```

外部送信に `y` で同意し、APIキーを非表示で入力します。ファイル・生成履歴・コマンド引数へキーを書きません。同意しなければこの起動セッションのJev送信を無効にして通常起動します。

送信先は `https://api.typesafe.ai` に固定。SDKは `typesafe-sdk==0.7.0`、モデルは `jev-1.13.0`。H3は初回にプロンプト、その後に層別の集約残差統計を送ります。Animaは集約した出力ノルムと変化量のみで、プロンプト本文は送りません。生画像・生音声・重みは送りません。通常のNeo生成履歴は従来どおりプロンプト等を保存します。

Jevは重要度の正解や画質を直接測定していません。代理統計に基づき候補保持率を選びます。不正JSON・範囲外回答・タイムアウト・API失敗は再試行せず回路を遮断します。Animaは以後Dense、H3は初回失敗時10%、後段失敗時5%へ退避します。低confidenceも同じ保持率へ補正します。H3のAPI timeoutは20秒、Animaは初期3秒、SDK起動に最大2秒の余白を加えています。

## 検証ログ

`outputs/jev-sparse/*.jsonl` に各runの選択保持率、処理回数、API回数/待ち時間、終了状態を保存します。ログには生のプロンプトではなくSHA-256を保存します。H3側の外部起動では既定先がComfyUIの `output/aikimi-sparse` です。

```powershell
py -3.12 tools/report_jev_sparse.py outputs/jev-sparse
```

H3の時間はモデル読込・VAE・保存を除いたsamplingのみです。Animaの `model_seconds` はモデル評価を合計した時間で、測定用のCUDA同期、統計処理、API待ちを含みます。AnimaのJSONL全体のelapsedは後処理等も含み得るため、レポートはmodel_secondsを使います。元投稿の生成全体時間とは直接比較できません。

H3の `native_sla_producer` はスパースproducer実行、`native_producer_not_used` はproducerを使わなかったという記録です。後者を実効Denseや実効スパース率と断定しません。保持率は正確な計算量削減率ではありません。Animaの `sparse` がゼロならSparse高速化を測った結果ではありません。

同一のモデルファイル・入力素材・プロンプト・Seed・解像度・長さ・Steps・samplerで、Dense/固定/Jevを複数回実行します。初回読込と温まった連続実行を分けます。動画では顔・手・動き・音声同期、画像では顔・細線・文字・構図を比較してください。固定Sparseに対するJevの差が、追加API待ちを超えるかが検証対象です。レポートは条件不一致の結果から自動で高速化率を算出しません。

Forgeが拡張の事前処理エラーを表示して通常生成を続行する場合があります。Animaの生成条件 `Anima Sparse status` が `active_experiment` であることに加え、対応JSONLの実行回数を確認してください。`not_active` やログなしの生成を実験成功に数えないでください。

## テストと確認範囲

```powershell
python -m pytest tests/jev_sparse -q
```

追加実装のCPUテスト、PyTorch CPUテンソルの形状/出力処理、Gradioコンポーネントの構築を確認しました。Forge/ComfyUI/API境界はモックです。リポジトリ全体の回帰テスト、Windows実機、ブラウザー操作、実際のCUDAカーネル、実モデル生成、Jevの有料API通信はこの実装時には実行していません。数値結果は検証記録を参照してください。

## 出典・ライセンス

H3ノードは [sepiablue-ai/native_sla.py](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA/blob/fa29225664909ea3dca69d0cdc35752bc38cd7fa/native_sla.py) を参考にした改変版でGPL-3.0-onlyです。`LICENSE.h3`を同梱し、ノード導入先にもコピーします。

Sparse実行APIは [ComfyUI nodes_sparse_attention.py](https://github.com/Comfy-Org/ComfyUI/blob/7a0b5eede3f9721c8faab290689893f36edc6d66/comfy_extras/nodes_sparse_attention.py)、Jev APIの利用形は同実験の [native_sla_worker.py](https://github.com/sepiablue-ai/ComfyUI-MiniMax-H3-W4A4-VSA/blob/fa29225664909ea3dca69d0cdc35752bc38cd7fa/native_sla_worker.py) に基づきます。Neo本体・モデル・SDK・ComfyUIはそれぞれのライセンスとサービス条件が別途適用されます。
