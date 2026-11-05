# Krea2 Local Supersample Detail 使用ガイド

## 1. 位置づけ

`Krea2 Local Supersample Detail` は、目視承認済みの4K画像へ局所描き込みを追加するための実験的なForge img2img Scriptです。既存の `VRAM-Canvas 4K/8K Highres`、既存profile、CLI、PNG metadata、8K経路の既定動作は変更しません。

本機能には2経路があります。通常modeは約512×512pxの固定payloadを1536×1536または2048×2048へ一時的に拡大し、round-trip基準との差から抽出した安全側の局所detail残差だけを元画像へ加えます。`Focused ROI Rewrite` は顔などを実際に描き直す経路です。tight target ROI全体と周囲contextを1枚の正方形として1536へ拡大し、Krea2で1回のcoherent regenerationを行い、同じ縮小経路で戻したフル差分をtarget内だけへフェザー合成します。

実画像に対する品質はprompt、checkpoint、入力detail、denoise、対象領域によって変わります。Focused modeでは顔や表情が意図以上に変化する場合があります。100%表示で目視確認してください。

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

## 14. 制約と残存リスク

- Krea2が局所cropを全体画像として誤解する可能性はpromptと低denoiseで抑えますが、完全には排除できません。
- 通常modeのCandidate agreementは誤った同一detailが2候補で一致する場合まで真偽判定できません。
- Focused modeは品質gateを採否に使わないため、同一性、表情、顔形状、明度が意図以上に変わる可能性があります。
- Focusedのquality scoreは候補の知覚的な顔品質を直接測るものではありません。`high_resolution_candidate.png` と4K書き戻しの目視比較が必要です。
- 強輪郭guardは二重線の可能性を下げますが、顔や物体輪郭の目視確認は必要です。
- 2048は処理時間とVRAM負荷が大きく、適合を保証しません。
- luma/chroma capはlinear-light単位であり、見た目のsRGB差は元の明るさによって変わります。
- Focusedはdetail増加、低周波drift、agreementの棄却を採否に使わないため、通常modeより顔形状・明度・柔らかさが変わります。意図的な書き直し用途に限定してください。
- RGB pixel SHA-256はPNG containerやmetadataのfile hashではありません。
