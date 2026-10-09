# 公式Qwen Image 2.1 Turbo

[Qwen公式のTurbo](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo)は、画像生成と参照編集を8 stepsで行うQwen Image 2.1モデルです。Viggle Turboの4 stepsとは重みとサンプリング条件が異なります。通常版の初期設定は変えず、公式Turboを追加して選択できます。

## 導入

Neoを終了して、ルートで実行します。

```powershell
.\aikimi-qwen-image21-setup.bat --official-turbo-only
```

公式Turbo本体約14.23 GBとスケジューラを取得し、固定revision・サイズ・SHA-256を検証します。通常版と共通のテキストエンコーダー・VAE・processorは導入済みなら再利用し、不足する場合は約18.9 GBを追加します。実行環境とINT8／W4A8の変換版の保存容量も必要です。生成時にはダウンロードしません。

導入内容の表示は`--official-turbo-only --dry-run`、取得後の再検証は`--official-turbo-only --verify`を使います。通常版のフル本体を追加する必要はありません。公式TurboのGGUFはこの導入には含めません。

## 使い方

1. Neoを起動し、**Qwen Image 2.1**のモデル欄で**公式Turbo · W4A8／INT8／BF16 · 8 steps**を選びます。24GB GPUではまずINT8・CPU退避で試してください。
2. プロンプトを入力して生成します。参照画像を追加すると編集になります。Stepsは8に固定し、公式のsigma列・CFG 1・スケジューラを使用します。
3. INT8／W4A8は初回に変換して保存し、同じ条件なら次回から再利用します。**変換モデルを保存**で画像生成前に準備することもできます。通常版の本体とは保存先を区別し、共通のテキストエンコーダーは再利用します。

「公式Turbo」は重みの配布元を示します。INT8／W4A8はStudio側で量子化する方式で、Qwen公式の量子化配布物ではありません。

BF16は本体だけで約14.23 GBです。テキストエンコーダーなども必要なため、モデル全体のGPU常駐にはさらに容量が必要です。GPU常駐を選んで容量が足りない場合は、読み込み前にCPU退避または量子化モデルを案内します。

**画像を広げる**にも公式Turboと8 stepsを引き継ぎます。画風LoRA・Outpaint LoRA・Fun ControlNetは既存のQwen経路を使いますが、通常版と同じ画質や編集特性は保証しません。Fun Accは別の高速化方式のため併用できません。SparseとFun ControlNetの併用条件は[Qwenガイド](../extensions-builtin/qwen-image21-studio/README.md)と同じです。

## 固定モデルとサンプリング

- 配布元: `Qwen/Qwen-Image-2.1-Turbo`
- revision: `d65dbc9a7e8f6b5479e33dee6030eaab2a906509`
- 8 stepsのsigma列: `1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568`
- スケジューラ: `FlowMatchEulerDiscreteScheduler`、dynamic shiftingなし、`shift=1.0`、`shift_terminal=null`

既存の固定Diffusersへ公式sigma列を明示して渡します。Stepsだけを変更して通常版のスケジュールを使う方式には切り替えません。結果の生成情報には配布元・revision・Steps・sigma列・スケジューラを残します。

生成中の時刻列も検査し、公式の8時刻と一致しない場合は画像を保存せず停止します。生成情報には実際の時刻列と回数も残します。

[Qwen Research License](https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/blob/d65dbc9a7e8f6b5479e33dee6030eaab2a906509/LICENSE)が適用されます。モデルの利用は研究・評価目的に限られ、商用利用には別途ライセンスが必要です。
