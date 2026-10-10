# 公式Qwen Image 2.1 Turbo

[Qwen公式のTurbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)は、画像生成と参照編集に8 stepsを推奨するQwen Image 2.1モデルです。Viggle Turboの推奨4 stepsとは重みとサンプリング条件が異なります。通常版の初期設定は変えず、公式Turboを追加して選択できます。

## 導入

Neoを終了して、ルートで実行します。

```powershell
.\aikimi-qwen-image21-setup.bat --official-turbo-only
```

公式Turbo本体約14.23 GBとスケジューラを取得し、固定revision・サイズ・SHA-256を検証します。通常版と共通のテキストエンコーダー・VAE・processorは導入済みなら再利用し、不足する場合は約18.9 GBを追加します。実行環境とINT8／W4A8の変換版の保存容量も必要です。生成時にはダウンロードしません。

導入内容の表示は`--official-turbo-only --dry-run`、取得後の再検証は`--official-turbo-only --verify`を使います。通常版のフル本体を追加する必要はありません。公式TurboのGGUFはこの導入には含めません。

### 保存済みINT8／W4A8を取得する

変換済みコンポーネントを[AikimiのINT8](https://huggingface.co/Aikimi/Forge-Neo-Image-2.1-Turbo-INT8)と[W4A8](https://huggingface.co/Aikimi/Forge-Neo-Image-2.1-Turbo-W4A8)から取得できます。Qwen公式の量子化配布物ではなく、Neo用の生成Transformerと共通テキストエンコーダーです。VAE・processor・元モデルと専用環境は上のセットアップで準備してください。元のBF16モデルを置き換える操作ではありません。

Neoを終了してルートで実行します。W4A8はrepo名・精度・取得先のINT8部分をそれぞれW4A8・`turbo_official_w4a8`・`official-turbo-w4a8`に変えます。

```powershell
& .\models\Qwen-Image-2.1\worker-env\Scripts\hf.exe download Aikimi/Forge-Neo-Image-2.1-Turbo-INT8 --local-dir .\work\hf-models\official-turbo-int8
& .\models\Qwen-Image-2.1\worker-env\Scripts\python.exe -X utf8 tools\qwen21_hub_release.py install --precision turbo_official_int8 --release-dir .\work\hf-models\official-turbo-int8
```

配布ファイルのサイズ・SHA-256、元モデルのrevision、量子化条件、専用環境の版を照合してから保存先へ取り込みます。条件が異なる環境には取り込めません。専用環境と元モデルを確認するか、その環境で変換してください。INT8はbitsandbytesの保存形式、W4A8はNeoの専用ローダーで読む圧縮形式です。取得後は同じ精度を選んで生成してください。

## 使い方

1. Neoを起動し、**Qwen Image 2.1**のモデル欄で**公式Turbo · W4A8／INT8／BF16 · 推奨8 steps**を選びます。24GB GPUではまずINT8・CPU退避で試してください。
2. プロンプトを入力して生成します。参照画像を追加すると編集になります。Stepsの初期値は推奨の8で、任意の正整数へ変更できます。8 stepsでは公式のsigma列・CFG 1・スケジューラを使用します。推奨と異なる設定は生成開始時にポップアップで案内し、生成を続けます。
3. INT8／W4A8は初回に変換して保存し、同じ条件なら次回から再利用します。**変換モデルを保存**で画像生成前に準備することもできます。通常版の本体とは保存先を区別し、共通のテキストエンコーダーは再利用します。

「公式Turbo」は重みの配布元を示します。INT8／W4A8はStudio側で量子化する方式で、Qwen公式の量子化配布物ではありません。

BF16は本体だけで約14.23 GBです。テキストエンコーダーなども必要なため、モデル全体のGPU常駐にはさらに容量が必要です。GPU常駐を選んで容量が足りない場合は、読み込み前にCPU退避または量子化モデルを案内します。

**画像を広げる**にも公式Turboを引き継ぎ、Outpaint側のStepsを使います。画風LoRA・Outpaint LoRA・Fun ControlNetは既存のQwen経路を使いますが、通常版と同じ画質や編集特性は保証しません。Fun Accも試せますが、配布元の推奨組み合わせではなく、Fun Accの専用PDDスケジューラと4 stepsを優先します。SparseとFun ControlNetの併用条件は[Qwenガイド](../extensions-builtin/qwen-image21-studio/README.md)と同じです。

## 固定モデルとサンプリング

- 配布元: `Qwen/Qwen-Image-2.1-Turbo`
- revision: `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`
- 8 stepsのsigma列: `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`
- スケジューラ: `FlowMatchEulerDiscreteScheduler`、dynamic shiftingなし、`shift=1.0`、`shift_terminal=null`

8 stepsでは固定したDiffusersへ公式sigma列を明示して渡します。それ以外では同じ公式Turboスケジューラ設定で、指定した回数に応じたFlowMatchのsigma列を使います。結果の生成情報には配布元・revision・指定Steps・sigma列・スケジューラを残します。推奨以外の回数で画質が改善するとは限りません。

生成中の時刻列も検査し、選択したスケジュールと一致しない場合は画像を保存せず停止します。生成情報には実際の時刻列と回数も残します。

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/d65dbc9a7e8f6b5479e33dee6030eaab2a906509/LICENSE)が適用されます。モデルの利用は研究・評価目的に限られ、商用利用には別途ライセンスが必要です。
