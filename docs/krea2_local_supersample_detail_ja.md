# Krea2 Local Supersample Detail 使用ガイド

## 1. 位置づけ

`Krea2 Local Supersample Detail` は、目視承認済みの4K画像へ局所描き込みを追加するための実験的なForge img2img Scriptです。既存の `VRAM-Canvas 4K/8K Highres`、既存profile、CLI、PNG metadata、8K経路の既定動作は変更しません。

本機能には2経路があります。通常modeは約512×512pxの固定payloadを1536×1536または2048×2048へ一時的に拡大し、round-trip基準との差から抽出した安全側の局所detail残差だけを元画像へ加えます。`Focused ROI Rewrite` は顔などを実際に描き直す経路です。tight target ROI全体と周囲contextを1枚の正方形として1536へ拡大し、Krea2で1回のcoherent regenerationを行い、同じ縮小経路で戻したフル差分をtarget内だけへフェザー合成します。

実画像に対する品質はprompt、checkpoint、入力detail、denoise、対象領域によって変わります。2026-07-14にRTX 3090で4096×1756画像のfocused実生成を行い、5.12倍のmodel入力、target内16,646画素変更、target外0画素を確認しました。これは単一事例であり、「必ず高画質になる」「既存方式より優れる」とは主張しません。100%表示での目視確認を省略しないでください。既存の `docs/vram_canvas_b5_ja.md` とPDFに記録された実測値、図表、SHA-256は別経路の結果です。

## 2. 標準ワークフロー

1. Krea2でnative画像を生成します。
2. `VRAM-Canvas 4K/8K Highres` で4Kを作り、全体構図と主要部を目視承認します。
3. 承認済み4Kを通常のimg2imgへ入れ直します。
4. `Script` で `Krea2 Local Supersample Detail` を選択します。
5. 保守的なdetail追加は `Safe 1536` または `Ultra Detail 1536` を実行します。
6. 顔全体の再描画は `Focused ROI Rewrite` と `Focused Face Rewrite 1536` を選び、顔または頭部を囲むtight targetを指定します。
7. 100% cropで顔、目、髪、手、衣装、輪郭、色、tile境界を確認します。
8. 2048は通常の `ROI Boxes` で追加描写が必要な領域だけに使います。
9. 必要であれば、局所仕上げ済み4Kを既存の8K経路へ渡します。

Batch CountとBatch Sizeは1にしてください。通常のimg2imgだけに対応し、inpaint maskには対応しません。

## 3. Profile

Profile値は `modules_forge/krea2_local_supersample.py` の辞書を正本とし、GUIへ重複記述していません。Dropdownを変更すると各algorithm sliderへ即時反映され、`Apply Profile` でも同じ値を再適用できます。

| Profile | Payload | Core | Overlap | Process Edge | Steps | Denoise | Candidates | Luma/Chroma cap | Context | Feather |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Safe 1536 | 512 | 384 | 64 | 1536 | 4 | 0.10 | 1 | 8 / 2 | - | - |
| Ultra Detail 1536 | 512 | 384 | 64 | 1536 | 5 | 0.15 | 2 | 12 / 3 | - | - |
| ROI Ultra 2048 | 512 | 384 | 64 | 2048 | 5 | 0.14 | 2 | 12 / 3 | - | - |
| Focused Face Rewrite 1536 | target依存 | target | 0 | 1536 | 6 | 0.38 | 2 | full rewrite | 2.0× | 20 px |

`Focused Face Rewrite 1536` は `Focused ROI Rewrite` と1個以上のtargetが必須です。Focused経路では固定Payload/Core/Overlap値を使わず、target長辺×Context Scaleから正方形payloadを計画します。`ROI Ultra 2048` は `ROI Boxes` modeと1個以上のROIが必須です。`Full Image Grid` でProcess Edge 2048を選ぶ場合は、`Allow expensive 2048 full-grid` を明示的にONにしない限り、model処理前に停止します。2048でVRAM適合を保証するものではありません。

## 4. GUI項目

- `Mode`
  - `Full Image Grid`: 入力全体をcore gridで覆います。
  - `ROI Boxes`: 指定ROIと交差するtileだけを処理し、書き戻しをROI内へ限定します。
  - `Focused ROI Rewrite`: targetごとに1枚の周辺contextを作り、targetを複数tileへ分割せず再生成します。
- `ROI Boxes / Focus Targets`
  - 元画像pixel座標の `left,top,right,bottom` です。
  - 複数boxは `1200,400,1800,1000;2100,600,2500,1100` のようにセミコロンで区切ります。
  - boxは元画像内に完全に収まり、`left < right`、`top < bottom` である必要があります。
  - Focusedでは1顔につき1個のtight boxを指定し、box同士を重ねられません。
- `Focused Context Scale`
  - target長辺へ掛ける周辺context倍率です。既定2.0です。120×150px targetなら300×300px contextになります。
- `Focused Rewrite Feather`
  - target境界から内側へ書き直しを立ち上げるsource pixel幅です。既定20pxです。target外へは広がりません。
- `Crop Payload`
  - modelへ送る前の元画像側context寸法です。既定512です。
- `Core Size`
  - payload中央で元canvasへ書き戻す範囲です。既定384です。
- `Core Overlap`
  - 隣接coreの重なりです。既定64です。
- `Process Edge`
  - Krea2へ送る正方形の一辺です。1536が標準、2048は明示的な高負荷経路です。
- `Steps` / `Denoising Strength`
  - tile内のKrea2 img2img設定です。
- `Candidate Count`
  - 1または2です。2の場合も候補画像や残差を平均しません。
- `Luma Residual Cap` / `Chroma Residual Cap`
  - linear-light residualの輝度成分と色成分を独立制限します。
- `Low-frequency Reject Radius`
  - 大域的な明暗、色、形状driftを除く外側スケールです。既定12です。
- `Strong Edge Protection`
  - 元画像の強輪郭で残差を連続的に減衰します。二値maskで全面ゼロにはしません。
- `Append mode-specific Krea2 guidance`
  - 元promptを完全なprefixとして保持し、通常modeでは局所detail指示、Focusedでは同一人物の顔を高解像度で再描画する指示を1回だけ追加します。
- `Save QA crops`
  - ONの場合だけ代表tileの診断画像を保存します。
- `Allow expensive 2048 full-grid`
  - 2048の全画面gridを明示許可します。既定OFFです。
- `Maximum Tile Count`
  - 計画tile数の上限です。既定256です。候補数はこのtile数へ別途乗算されます。

## 5. Round-trip residual

元画像からedgeまたはreflect padで取り出した固定payloadを (B)、sRGB LanczosによるProcess Edgeへの拡大を (U)、Krea2 img2imgを (K)、linear-light area縮小を (D) とします。

```text
U(B) = sRGB Lanczosで1536または2048へ拡大したprocess input
R    = K(U(B))
C0   = D(U(B))
C1   = D(R)
delta_raw = C1 - C0
```

`C0` と `C1` は必ず同じ `linear_area_downsample()` を通ります。高解像度sRGBをnormalized linear RGBへdecodeし、OpenCV `INTER_AREA` で元payload寸法へ縮小します。内部の差分、帯域処理、gate、cap、元画像への加算はnormalized linear RGBのまま行い、QA画像と最終PNGを作る段階でsRGBへencodeします。C0だけを丸める、C1だけを別のresizerへ通す、途中だけsRGB差分へ戻す、という経路はありません。

候補 (R) が (U(B)) と同一なら、同じ演算入力から得るC0とC1も同一であり、`C1 - C0` はゼロです。元payload (B) は差分の右辺へ入りません。両modeとも、次の非対称経路は禁止されています。

```text
C1 - B をdetail残差として使う
C1をround-trip補償なしで完成tileとして貼る
Rを別resizerで縮小して貼る
高解像度候補そのものをcanvasへ貼る
```

通常modeは `C1-C0` から低周波を除いたbounded detailだけを使います。Focused modeは顔形状を含む書き直しを目的とするため、選択候補のフル `C1-C0` を使いますが、同じ `D` を通すround-trip補償、target内限定、内向きfeatherは維持します。

## 6. Residual guard

### 6.1 Low-frequency rejection

`delta_raw` に対し、inner sigma 0.65と指定outer radiusのDifference of Gaussiansを適用します。既定ではおおむね1～12px帯域の局所差を残し、一様な色shiftや大きな明暗・形状変化を除きます。prompt側でもrandom grainやfake noiseを禁止します。

### 6.2 Structure gate

`C1-C0` の低周波linear luminance差が大きい場所では、detail residualを指数的に減衰します。候補の形状、明暗、色面がround-trip基準から大きく動いた領域ほど寄与が小さくなります。

### 6.3 Strong-edge guard

元payload (B) のlinear luminanceからSobel勾配を計算し、強輪郭ほどresidualを滑らかに減衰します。最小gateを残すため、元の細線detailまで全面的にゼロにはしません。目的は顔輪郭、物体外形、衣装境界などでdouble contourやhaloが生じる可能性を下げることです。

### 6.4 Luma/chroma分離

linear RGB residualのluminanceをRec.709係数で求めます。

```text
Y = 0.2126 R + 0.7152 G + 0.0722 B
chroma = RGB - Y
```

GUIのcapは「8-bit相当のnormalized linear-light単位」です。たとえばLuma cap 8は内部で `8 / 255`、Chroma cap 2は `2 / 255` です。sRGB code値の±8/±2という意味ではありません。lumaとchromaを別々に制限し、色残差を輝度残差より厳しくします。

### 6.5 Clippingとfinite validation

元payloadへresidualを加える前のRGB channel clipping率を記録します。実際のresidualはRGB vector全体をpixel単位で縮小してsource channelのheadroom内へ収めるため、luma/chroma比と既存capを壊すchannel別clipを行いません。NaN、Inf、不正shape、負weight、非finite設定は受け入れず、処理を明示的に失敗させます。

## 7. Candidate Count 2

候補AとBは、global seed、tileのcore座標、candidate indexから決定論的に別seedを作ります。同じ入力、設定、global seedでは同じseed列になります。

各候補について、次を測ります。

- mean/p95 low-frequency drift
- detail energy増加
- mean/p95 bounded residual
- RGB clipping fraction
- payload境界付近のresidual

品質gateに通った候補のうち、低周波drift、clipping、境界residualが小さく、適切なdetail増加がある一方を代表候補にします。もう一方はsupportです。片方だけが品質gateを通る場合も、通った側だけが代表候補になり、他方のresidualはsupport判定以外へ加算しません。両方が不合格ならtileは厳密なzero residualとなり、理由をmetadataへ残します。

agreement maskは、代表候補とsupport候補の次の関係から作ります。

- linear luminance residualの符号一致
- RGB residual vectorの方向一致
- Gaussian近傍の局所相関
- 同位置でのresidual magnitude support

最終tile residualは次だけです。

```text
tile residual = representative residual * agreement mask
```

`(A+B)/2`、候補画像のaverage、residualのaverage、first-moment平均は行いません。逆符号detail、位置の合わないdetail、無相関noiseは抑制されます。

`Focused ROI Rewrite` は顔を確実に書き直す目的なので、agreement maskを意図的に使いません。detail gate合格候補が1つ以上あれば、その集合から `quality_score` が最小の1候補を選びます。全候補が不合格でも、全候補中の最小scoreを選んで `C1-C0` の低周波を含むフル差分を採用し、その棄却理由を監査用の `quality_gate_override_reason` として残します。候補同士の平均は行いません。これは「高精細と判定された残差だけ」ではなく、Krea2が縮小後に書き直した見た目を反映する専用経路です。

## 8. Tile、pad、weight、ROI

512 payload / 384 coreでは、通常coreの周囲へ64pxずつcontextがあります。画像端でも仮想payload boxを先に決め、画像外へ出た分だけ上下左右をedge padします。たとえば左上coreが `(0,0,384,384)` なら仮想payloadは `(-64,-64,448,448)`、payload内coreは常に `(64,64,448,448)` です。padによってcoreが移動することはありません。

core起点は `core - overlap` strideで計画し、右端と下端を覆う強制終端tileを必ず追加します。重複部分はsmoothstep weightです。強制終端ではraw weight和が1を超えることがあるため、x/yの全canvas normalizerで各tile weightを割り、全pixelの寄与和を1へ正規化します。

品質不合格tileもzero residualとしてweightへ参加します。不合格tileを分母から外して隣接tileの寄与を増幅することはありません。

通常のROI modeでもcontextはROI外を含む元画像から読めますが、weightへROI union maskを掛けるため書き戻しはROI内だけです。重複ROIはunionとして扱い、二重加算しません。ROI外のpixelは元uint8値をそのままコピーし、linear RGB round-tripへ通しません。

Focused modeでは各target自体がcoreです。target長辺を $L$、Context Scaleを $k$ とすると正方形payload辺を $\lceil kL\rceil$ とし、target中心へ配置します。画像端から出たcontextだけedge padします。1 targetを1 payloadとして処理するため、1つの顔が独立seedの複数tileへ分割されません。書き戻しweightはtarget境界で0、指定feather幅の内側で1になるsmoothstepです。分母は1に固定し、featherが正規化で相殺されないようにします。target外は入力uint8値を直接コピーします。

## 9. Forge統合と失敗時動作

長時間処理の前に次を検査します。

- Batch Count 1、Batch Size 1
- 通常img2img入力が1枚あること
- inpaint maskがないこと
- PNG metadataが有効なこと
- Krea2 engineとKrea2 model config
- Qwen3-VL text encoder実体
- Qwen Image VAE実体
- Krea2 checkpoint / Qwen moduleの設定名
- ROI、2048明示許可、tile上限
- 一時disk空き容量

内部candidate処理では、次を一時設定します。

- 1 tile、1 candidateずつ処理
- Batch Count / Batch Size 1
- 内部sample/grid保存OFF
- `Restore faces` OFF
- `Tiling` OFF
- mask/refiner OFF
- profileのProcess Edge、Steps、Denoise
- 座標由来candidate seed

`p` の全属性と、list/dict/setの元内容を処理前にsnapshotします。成功、通常例外、中断、skip、stop、OOMのすべてで、新規属性を除去し、prompt、negative prompt、seed、subseed、steps、width/height、denoise、init image、mask、save flag、Restore faces、Tiling、override settingsなどを元の値・元の有無へ戻します。

各candidate後にsampling bufferを解放し、`devices.torch_gc()` を呼びます。`state.interrupted`、`state.skipped`、`state.stopping_generation` はcandidate前後とfinalize前に確認します。中断時に直前candidateや途中canvasを完成画像として返しません。

OOMを握り潰したり、2048から1536へ自動的に画質を変更したりしません。2048で割り当てに失敗した場合は、Process Edge 1536へ変更し、2048を小さなROIへ限定する具体的なエラーを返します。1536自体で割り当てに失敗した場合は、他のGPU workloadを解放するかROIを減らす必要があります。

## 10. CPU accumulatorと一時disk

4K全体のaccumulatorをGPUへ置きません。処理中は一時directoryに次のdisk memmapを作ります。

```text
residual_sum: height × width × 3 × float32
weight_sum:   height × width × float32
```

一時disk見積もりは正確なmemmap payloadである `width × height × 16 bytes` です。空き容量が不足する場合は最初のKrea2 candidateより前に停止します。Windowsで一時directoryを確実に削除できるよう、正常、例外、中断、OOMのすべてでmemmapをflushし、mapping handleをcloseしてからdirectoryを抜けます。

最終合成は256行ずつCPUで行い、元画像と同寸法のuint8 RGBを作ります。最終出力は1枚だけです。

## 11. Prompt

`Append Krea2 local-detail guidance` がONでも、元promptを削除、並べ替え、要約、trimしません。元文字列をbyte-for-byteのprefixとして保持し、その後ろへ局所crop用の一般指示を1回だけ追加します。suffixが既にある場合は再追加しません。

追加指示は次を明示します。

- 入力が全体画像ではなく拡大局所cropであること
- identity、顔、表情、解剖、物体数、輪郭、構図、crop境界を維持すること
- 元画像に既に示唆される髪線、虹彩、睫毛、縫い目、刺繍、レース、材質detailだけを加えること
- 人物、手足、指、目、物体、文字、logo、輪郭、反復模様を追加しないこと
- grain、fake noise、halo、double contour、tile seamを加えないこと

重要な禁止条件はpositive側へ入るため、CFG 1.0でnegative promptが効かない構成でも指示自体は渡ります。通常log、state表示、JSON manifest、QA manifestへprompt全文やbase64画像を複製しません。最終PNGの標準 `parameters` chunkにpromptが入るのはForgeの通常契約です。

## 12. PNG metadata

最終PNGは入力画像にある文字列metadata chunkを引き継ぎ、`parameters` を最終寸法・global seedへ直したうえで、`krea2_local_supersample` JSONを追加します。

主なfield:

- `format_version`
- `input_size`, `output_size`
- `profile`, `mode`, `global_seed`
- `payload`, `core`, `overlap`, `process_edge`
- `steps`, `denoise`, `candidate_count`
- `luma_cap`, `chroma_cap`, `low_frequency_reject_radius`
- `focused_rewrite`, `focused_context_scale`, `focused_rewrite_feather`, `focused_region_count`
- `tile_count`, `processed_tile_count`, `rejected_noop_tile_count`
- `agreement_coverage`
- `mean_low_frequency_drift`, `p95_low_frequency_drift`
- `mean_residual`, `p95_residual`
- `clipping_fraction`, `candidate_clipping_fraction`
- `input_sha256`, `output_sha256`
- per-region/tile `core`, `payload_box`, `payload_side`, `effective_zoom`, `candidate_seed`, `selected_candidate`, metrics、agreement、rejection reason

PNG自身のfile hashを同じPNG内へ埋めると自己参照になるため、SHA-256の対象は「decoded RGBの寸法文字列と連続pixel bytes」です。`sha256_scope` fieldにもこの定義を保存します。これはfile container全体のSHA-256ではありません。

aggregateのmean/p95は、全処理candidateがmetadataへ残すtile-level summaryを集約した値です。架空のbenchmarkや品質比較値ではありません。

## 13. QA crop

`Save QA crops` がONの場合だけ、次へtimestamp別directoryを作ります。

```text
<Forge sample output>/krea2_local_supersample_qa/YYYYMMDD_HHMMSS_microseconds/
```

保存内容:

- `source_payload.png`
- `process_input.png`
- `high_resolution_candidate.png`
- `downsampled_candidate_c1.png`
- `roundtrip_baseline_c0.png`
- `residual_visualization.png`
- `before_payload.png`
- `after_payload.png`
- `qa_manifest.json`

通常は最初の代表tileまたはFocused targetを保存します。全tileがno-opの場合は最初の診断候補を保存し、`selected_candidate: null` とrejection reasonをmanifestへ記録します。Focusedでは選択候補、`payload_box`、`effective_zoom`、元の `quality_gate_override_reason` を記録します。QA画像をForge galleryへ追加せず、既存ファイルを上書きしません。QA manifestにprompt全文は入れません。

## 14. CPU自動テスト

実modelや外部downloadを使わないテスト:

```powershell
.\venv\Scripts\python.exe -m unittest -v `
  tools.tests.test_krea2_local_supersample `
  tools.tests.test_krea2_local_supersample_gui
```

compileとpure import:

```powershell
.\venv\Scripts\python.exe -m py_compile `
  .\modules_forge\krea2_local_supersample.py `
  .\scripts\krea2_local_supersample_detail.py

.\venv\Scripts\python.exe -c "import modules_forge.krea2_local_supersample"
```

既存関連回帰:

```powershell
.\venv\Scripts\python.exe -m unittest -v `
  tools.tests.test_vram_canvas `
  tools.tests.test_vram_canvas_gui `
  tools.tests.test_krea2_tiled_refine `
  tools.tests.test_krea2_subject_refine `
  tools.tests.test_krea2_quality
```

## 15. Windows / Forge手動確認

1. Forgeを再起動し、Krea2 checkpoint、Qwen Image VAE、Qwen3-VLを選択します。
2. `img2img` へ承認済み4Kと元promptを入れ、Batch Count / Batch Sizeを1にします。
3. 保守的な全体detailは `Krea2 Local Supersample Detail` → `Safe 1536` を選びます。
4. `Restore faces` と `Tiling` を任意の元状態にし、Generate後もその状態へ戻ることを確認します。
5. 入力と出力の寸法が完全に同じで、galleryへ最終画像が1枚だけ返ることを確認します。
6. 100%で顔、目、髪、手、衣装、強輪郭、色面、tile境界を比較します。
7. `ROI Boxes` で小さな領域を指定し、ROI外をpixel比較して不変であることを確認します。
8. `Ultra Detail 1536` のCandidate Count 2を実行し、PNG metadataに別seedと代表候補、agreementが記録されることを確認します。
9. 必要なROIだけで `ROI Ultra 2048` を試します。OOM時に1536へ自動再試行されず、具体的な案内が出ることを確認します。
10. `Save QA crops` をONにし、同じ512px範囲のbefore/after、C0/C1、residualを比較します。
11. 顔を書き直す場合は `Focused ROI Rewrite` / `Focused Face Rewrite 1536` を選び、1顔を囲むtight targetを指定します。job表示とPNG metadataで実効拡大率が1倍を超えることを確認します。
12. 実行中断、skip、意図的なvalidation error後にもprompt、seed、mask、Restore faces、Tiling、保存設定が元へ戻ることを確認します。
13. 最終PNGの `parameters` と `krea2_local_supersample` chunkを確認します。

## 16. 指定プロンプト実機ケーススタディ（2026-07-13）

指定された次のbase promptを、綴り・大小文字・末尾commaを含めてそのまま使用しました。seedは `3883506083`、入力は既存4K preflightを通過した `2896x4096` RGB PNGです。

```text
light blue hair,long_wavy_hair,devil’s_horn,purple horn,purple_eyes,green_slime,jig eyes,smile,jitome,Expressionless,
```

RTX 3090 24 GB上の実測結果:

| 条件 | 処理tile / 候補 | 採用tile | 変更画素 | 時間 | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 顔ROI / Safe 1536 | 9 / 9 | 0 | 0% | 310.1 s（初回model load込み） | 21,479 MiB |
| 顔ROI / Ultra Detail 1536 | 9 / 18 | 0 | 0% | 197.4 s | 21,617 MiB |
| 目ROI / ROI Ultra 2048 | 1 / 2 | 0 | 0% | 49.4 s | 22,600 MiB |
| 全画面 / Safe 1536 | 117 / 117 | 12 | 3.7205% | 1,387.2 s | 22,478 MiB |

顔・目の3条件は候補の局所detail energyが増えず、品質gateにより全件不採用になりました。出力は入力と変更0画素で、decoded RGB pixel SHA-256も `241cba297e05…` のままです。2048は単一ROIでOOMなく完走しましたが、ピーク22,600 MiBで有用な残差は得られませんでした。

全画面Safe 1536は117 tile中12 tileだけを採用し、変更441,330画素（3.7205%）、全RGB channelの平均絶対code差0.02018、p95=0、p99=1、最大8でした。採用位置は上端・左端・下部衣装の一部に偏り、顔中心は不採用です。候補全体のdetail-energy増分は平均−0.1733 codeで、多くは既存4Kより平滑化方向でした。合成metadataのclipping fractionは0です。

この単一事例が示すのは一般的な画質向上ではなく、改善根拠がない候補をbit-identical no-opへ戻せることです。本画像では全画面よりSafe 1536 ROIを先に試し、採用tileと100% cropを確認する運用が妥当です。

- [JIS B5判2ページ・採否マップと原寸crop入り実測短報](krea2_local_supersample_b5_ja.pdf)
- [短報テキスト版](krea2_local_supersample_b5_ja.md)
- 再現実行器: `tools/run_krea2_local_supersample_experiment.py`
- B5 PDF生成器: `tools/build_krea2_local_supersample_paper.py`

再現実行器は画像と無関係なpromptを誤って送らないよう `--prompt` を必須にしています。上記ケースを再現するときも、別画像を評価するときも、画像内容と一致するbase promptを `--prompt '<exact prompt>'` で明示してください。標準出力にはprompt全文を出さず、SHA-256だけを表示します。

## 17. Focused ROI Rewrite実機ケース（2026-07-14）

4096×1756の雪景色の駅構内画像を入力とし、画面右寄りの人物の顔をtarget `(2608,635,2728,785)` としました。targetは120×150px、Context Scale 2.0から得た正方形contextは `(2518,560,2818,860)` の300×300pxです。これを1536×1536へ拡大したため実効拡大率は **5.12倍** です。顔は1枚のprocess input中央に収まり、複数tileや複数seedへ分割されていません。

設定は `Focused Face Rewrite 1536`、6 steps、denoise 0.38、2候補、source側20px inward featherです。checkpointは `turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors`、Qwen Image VAE、Qwen3-VL 4B bf16、GPUはRTX 3090 24GBです。

| 測定項目 | 結果 |
|---|---:|
| 入力 / 出力 | 4096×1756 / 4096×1756 |
| target / context | 120×150 / 300×300 px |
| Krea2 process input | 1536×1536 px |
| 実効拡大率 | 5.12× |
| target内変更 | 16,646画素 / 92.4778% |
| target外変更 | 0画素 |
| target内RGB絶対差 | mean 9.007 / p95 26 / max 77 code |
| 処理時間 | 49.1 s |
| peak VRAM | 20,957 MiB（GPU全体、1秒標本） |

選択された高解像度候補では、元4Kで潰れていた両目、虹彩、まつ毛、鼻口、顎線が再構成され、縮小後の4Kへ反映されました。これは自動的な知覚品質保証ではなく、ground truthや盲検評価を持たない単一画像の観察です。一方、処理が実際に1536入力を通り、書き直しがtarget内へ反映され、target外がbit-exactであることは画像差分とmetadataで検証済みです。

- 実行結果: `output/krea2_smart4k_snow_station/focused_face_rewrite_1536/local_supersample_20260714_122916_778319/`
- 再現実行器: `tools/run_krea2_local_supersample_experiment.py`
- 実測短報: `docs/krea2_focused_roi_rewrite_b5_ja.pdf`

## 18. 制約と残存リスク

- Krea2が局所cropを全体画像として誤解する可能性はpromptと低denoiseで抑えますが、完全には排除できません。
- 通常modeのCandidate agreementは誤った同一detailが2候補で一致する場合まで真偽判定できません。
- Focused modeは品質gateを採否に使わないため、同一性、表情、顔形状、明度が意図以上に変わる可能性があります。
- Focusedのquality scoreは候補の知覚的な顔品質を直接測るものではありません。`high_resolution_candidate.png` と4K書き戻しの目視比較が必要です。
- 強輪郭guardは二重線の可能性を下げますが、顔や物体輪郭の目視確認は必要です。
- 2048は処理時間とVRAM負荷が大きく、適合を保証しません。
- luma/chroma capはlinear-light単位であり、見た目のsRGB差は元の明るさによって変わります。
- Focusedはdetail増加、低周波drift、agreementの棄却を採否に使わないため、通常modeより顔形状・明度・柔らかさが変わります。意図的な書き直し用途に限定してください。
- RGB pixel SHA-256はPNG containerやmetadataのfile hashではありません。
- Focusedの実画像比較は画像1枚・target 1件・seed系列1件であり、品質改善や設定値の一般性を断定できません。
