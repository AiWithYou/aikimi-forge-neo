# Krea2 PhaseWeave 4K

## 概要

`Krea2 PhaseWeave 4K` は、位置を半格子ずらした二つのタイル配置を別々に完成させ、局所的に候補A、候補B、入力維持を選ぶ4K向けの実験mergeです。

- Product / UI name: `Krea2 PhaseWeave 4K`
- Profile display name: `Krea2 PhaseWeave 4K (Experimental)`
- Profile key: `phaseweave_4k`
- Merge mode: `phase_weave`
- PNG metadata key: `krea2_phaseweave`
- Selection mode: `ternary_input_fallback`

二候補を先に平均しません。候補差分の高周波と低周波を分け、入力との輪郭・明暗・色の一致を含めて評価します。勝敗が確かでない場所では自動的にAへ倒さず、周囲からの選択伝播、入力補間への復帰、位置合わせ後の弱い融合の順で解決します。

## 処理の全体像

```text
入力画像を目標寸法へ補間
→ 配置Aの全タイルを処理し、候補Aを独立に完成
→ 半格子ずらした配置Bの全タイルを処理し、候補Bを独立に完成
→ 各候補の差分を高周波と低周波へ分離
→ 高周波量、配置内安定度、入力忠実度を局所評価
→ A / B / 未確定の三値判定
→ 高信頼領域からラベルを伝播し、小さな島・穴を整理
→ 両候補が不適切なら入力補間へ戻す
→ 十分近い二候補だけ局所位置合わせ後に弱く融合
→ 整理済み境界だけを5pxで接続
→ PNG metadataとrun manifestを保存
```

## 二つの配置

各配置は画像の一部だけを担当するのではなく、それぞれ画像全体の4K候補を作ります。

タイルの大きさや配置数は画像とprofileによって決まります。画像外は端画素で補い、画像と交差する部分だけを合成します。

## 配置ごとの独立再構成

入力補間画像に対する配置 `g`、領域 `t` の差分を `Δgt`、窓重みを `wgt` とします。配置ごとに差分和 `Dg`、重み和 `Wg`、二乗量 `Eg` を別々のdisk memmapへ蓄積します。

```text
Dg = Σt wgt Δgt
Wg = Σt wgt
Eg = Σt wgt mean(Δgt²)
Rg = Dg / Wg
```

`Eg` と `Rg` から、同じ配置内で重なる領域がどれだけ揃っているかを表す安定度 `Cg` を求めます。この段階ではAとBを混ぜません。

## 低周波を入力へ固定する

候補差分 `Rg` を標準偏差12px相当の平滑化で低周波 `RgL` と高周波 `RgH` に分けます。

```text
RgL = G12(Rg)
RgH = Rg - RgL
R̂g = RgH + ηY RgL,Y + ηC RgL,C
```

初期値は次のとおりです。

| 項目 | 値 |
|---|---:|
| 低周波分離sigma | 12px |
| 輝度低周波係数 `ηY` | 0.32 |
| 色差低周波係数 `ηC` | 0.18 |
| 明部しきい値 | 入力輝度0.85 |
| 明部の輝度低周波倍率 | 0.50 |

髪の細線、本棚の傷、刺繍などの細部は残しつつ、衣服の大きな明暗模様、白髪の発光感、肌色の変化を抑える狙いです。

## 入力忠実度を含む品質評価

差分のRMSが大きいだけでは高品質とみなしません。高周波差分の量 `Hg`、配置内安定度 `Cg`、入力忠実度 `Fg` を組み合わせます。

```text
Qg = Hg (0.25 + 0.75 Cg) Fg
Fg = exp(-1.25 dedge - 1.00 dlow - 0.75 dchroma)
```

- `dedge`: 入力と候補の輪郭方向の不一致、および入力にない輪郭の増加
- `dlow`: 低周波輝度の変化
- `dchroma`: 低周波色差の変化

忠実度は入力輝度を案内画像とする半径8のガイド付きフィルターで整理します。品質得点は同じ案内画像で半径16に整理します。平坦部では選択をまとまりやすくし、入力の強い輪郭ではA/B境界が輪郭を越えて広がりにくくします。

## A、B、入力維持の三値選択

```text
S = (QB - QA) / (QA + QB + ε)

B      : S >  0.03 かつ FB >= 0.42
A      : S < -0.03 かつ FA >= 0.42
未確定 : それ以外
```

未確定領域は次の順で処理します。

1. 周囲の高信頼A/Bから、半径24・最低信頼度0.15で選択を伝播する。
2. 3000px未満のA/B選択島を周囲の多数ラベルへ統合する。
3. 両候補の忠実度が0.42未満なら入力補間へ戻す。
4. A/Bの差分RMSが3.0以下かつ局所構造類似度が0.96以上なら、最大±1pxで局所位置合わせして弱く融合する。
5. 512px未満の弱い入力島は周囲のA/Bへ統合する。ただし明確な低忠実度拒否は残す。
6. 整理後の確定境界だけを半径5pxで接続する。

`selected_phase` 診断画像では、A=橙、B=青、入力維持=灰、近接融合=紫です。

## 補助候補

選択した差分を `R*`、位置合わせしたもう一方を `R-align`、二候補の近さを `a` とします。局所構造類似度が0.90以上で輪郭方向も揃う場合だけ補助します。

```text
Rsup = R* + 0.10 a² (R-align - R*)
Rout = (0.90 + 0.10 a) Rsup
```

補助上限は10%で、`a²` により候補がかなり近い場合だけ効きます。代表候補の倍率は0.90未満へ下げません。

## Exact Steps

Forgeの通常img2imgでは、`img2img_fix_steps` がOFFの場合に指定stepsとdenoiseから実効step数が減ります。次の内部処理だけ、Exact Stepsをrequest-localに有効化します。

- VRAM-Canvas GUIの全タイル
- VRAM-Canvas CLI/APIの全タイル
- Krea2 Local Supersample Detail GUIの全候補
- 同機能のCLI/API経路

GUIは共有context managerでprocessing objectとoverride辞書を退避し、外側の `finally` で復元します。CLI/APIは各requestの `override_settings` に `img2img_fix_steps=true` と `override_settings_restore_afterwards=true` を設定します。通常設定を処理時間全体にわたって無条件に変更しません。

PNGとmanifestには次を保存します。

```json
{
  "exact_img2img_steps": true,
  "exact_img2img_steps_scope": "internal_tiles_only"
}
```


## processing状態の復元

GUI経路では処理開始時にprocessing objectの全属性と、list・dict・setの内容をsnapshotします。内部で変更する保存、顔補正、tiling、mask、batch、override関連の状態は、成功・例外・中断・skip・stop・OOMを通る外側の `finally` で復元します。完成画像へ必要なmetadataだけを明示的に残します。

## GUIとCLIのprofile共有

GUIのprofile選択、quick button、CLIの `--krea2-profile phaseweave_4k` は、`modules_forge/krea2_highres.py` の同じprofile dictionaryを使います。

Forge GUI:

```text
img2img
→ Script: VRAM-Canvas 4K/8K Highres
→ Krea2 PhaseWeave 4K
→ Generate
```

CLI:

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py `
  --input '<input.png>' `
  --api 'http://127.0.0.1:7861' `
  --krea2-profile phaseweave_4k `
  --append-krea2-detail-prompt `
  --long-edge 4096
```

A単独、B単独、選択図も同じrunから保存する場合:

```powershell
.\venv\Scripts\python.exe .\tools\vram_canvas_highres.py `
  --input '<input.png>' `
  --api 'http://127.0.0.1:7861' `
  --krea2-profile phaseweave_4k `
  --save-phase-candidates `
  --long-edge 4096
```

`--save-phase-candidates` は `phase_weave` 専用です。別mergeとの組み合わせはモデル処理前に失敗します。

## 一時diskの事前検査

各stageの全画素に対して、必要なfloat32 accumulatorを列挙した保守的な見積りを使います。

| merge | novel off | novel on |
|---|---:|---:|
| consensus | 32 byte/pixel | 48 byte/pixel |
| phase_weave、2配置 | 52 byte/pixel | 84 byte/pixel |

`--save-phase-candidates` 使用時は、A、B、選択図の作業領域とPNG保存余裕として18 byte/pixelを加えます。全stageの合計と出力余裕を開始前に空き容量と比較します。不足時はモデル処理前に明示的に停止し、暗黙の画質低下や別方式への切り替えは行いません。

## PNG metadataとmanifest

最終PNGの `krea2_phaseweave` はformat version 4です。主要項目は次のとおりです。

```json
{
  "format_version": 4,
  "product_name": "Krea2 PhaseWeave 4K",
  "profile_key": "phaseweave_4k",
  "merge_mode": "phase_weave",
  "selection_mode": "ternary_input_fallback",
  "selection_margin": 0.03,
  "fidelity_reject_threshold": 0.42,
  "fidelity_guided_radius": 8,
  "island_min_area": 3000,
  "input_island_min_area": 512,
  "feather_radius": 5,
  "low_frequency_sigma": 12.0,
  "low_frequency_luma_gain": 0.32,
  "low_frequency_chroma_gain": 0.18,
  "detail_floor": 0.90,
  "support_mix": 0.10,
  "support_confidence_power": 2.0,
  "support_alignment_radius": 1,
  "exact_img2img_steps": true,
  "exact_img2img_steps_scope": "internal_tiles_only"
}
```

manifest format version 5には全タイルの座標・配置・steps・skip、格子情報、三値選択率、忠実度、入力維持率、近接融合率、境界率、補助率、低周波係数を保存します。候補保存時はA、B、選択図のパスも記録します。

## 制約

- 現在のPhaseWeaveは2配置専用です。
- 4K全体を一度に生成する方式ではなく、複数回の局所img2imgを行うため時間がかかります。
- 入力補間は正解画像ではなく、細部の正しさを自動判定できません。
- 生成モデルが作る偽の文字や材質を完全に防ぐものではありません。入力維持と忠実度は拒否機構であり、意味理解による真偽判定ではありません。
