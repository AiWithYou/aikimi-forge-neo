# Krea2 A-Series × VRAM-Canvasによる1:√2縦長イラストの8K生成

*A reproducible single-image case study of progressive high-resolution diffusion refinement*

**技術ケーススタディ / 2026-07-12 / AiWithYou / Forge neo-2.26**

## 概要

ユーザー指定の語彙からKrea2 A-Seriesのネイティブ候補を4枚生成し、定性的に選定した1枚をVRAM-Canvasで1024×1448から2896×4096、さらに5792×8192へ漸進的にrefineした。最終画像は47,448,064画素で、縦横比は1:1.4143646、√2に対する相対誤差は0.0107%である。4K工程は62/62、8K工程は140/140の局所API呼出しが成功し、OOM、NaN、HTTP failureは観測しなかった。間欠的な`nvidia-smi`確認で観測した最大使用量は21,656 MiBであった。

最終段には、意図的な髪・石材・スライムの微細構造を守るためdespeckleを無効にしたSmart Finishを適用した。内部chroma-mura heuristicのp95は8Kで5.25から3.67へ、`Δchroma > 5`の面積率は5.64%から1.68%へ低下した。全体像と6個の原寸相当cropを目視し、明瞭なtile seam、人物や角の増殖、二重輪郭、平坦部の反復微細模様は確認されなかった。

本稿は単一画像・単一GPU・評価者1名の非blindケーススタディである。一般的な知覚品質の優位性、他方式に対する性能優位、文字・顔・手の正しさ、集団レベルの再現性は主張しない。

**Keywords:** high-resolution diffusion, tiled refinement, VRAM budget, Krea2, frequency residual, consensus gate, base-detail gate

## 1. 問題設定

高解像度の納品canvasをそのまま拡散modelへ投入すると、空間activationは概ね画素数に比例して増え、24 GiB GPUでは8K級処理が困難になる。一方、tileを単純に生成して貼り合わせると、局所構図のずれ、境界、重複領域の競合、平坦部への偽微細模様が生じやすい。

本例では次の要求を同時に満たすことを目標とした。

1. ユーザー指定語彙を保った単一成人キャラクターを生成する。
2. 縦横比を1:1.414、すなわちほぼ1:√2に保つ。
3. 4Kを成功条件として確認した後、8Kへ進む。
4. 既存の顔、角、髪、衣装、スライム、構図を維持し、局所的な高周波detailだけを追加する。
5. RTX 3090 24 GiBのVRAM範囲で処理する。
6. 生成、選定、refinement、最終仕上げ、目視QAまでを記録し、再現可能なケーススタディとする。

## 2. ネイティブ画像生成

### 2.1 Prompt設計

元の指定語彙は次の通りである。

```text
light blue hair,long_wavy_hair,devil’s_horn,purple horn,purple_eyes,
green_slime,jig eyes,smile,jitome,Expressionless
```

`smile`と`Expressionless`の衝突を、「無表情を基調にしたごく小さな閉口微笑」として具体化した。`jitome`と`jig eyes`は半眼の紫眼としてまとめ、人物は単独の成人であることをpositive prompt内に明記した。

実際のネイティブpromptは次の通りである。

```text
light blue hair, long_wavy_hair, devil’s_horn, purple horn, purple_eyes,
green_slime, jig eyes, smile, jitome, Expressionless, single adult anime woman,
two coherent purple devil horns, very long wavy light-blue hair,
half-lidded purple eyes, a restrained tiny closed-mouth smile with an otherwise
calm expressionless face, translucent green slime curling around her lower body
and pooling near her feet, elegant dark fantasy outfit, full body visible,
vertical portrait, detailed anime illustration, cinematic dark fantasy stone
chamber, subtle mist, purple and green rim light, intricate hair strands,
glossy translucent slime, detailed fabric, coherent anatomy,
clean natural linework, no text, no logo, no watermark
```

### 2.2 生成設定

| 項目 | 値 |
|---|---:|
| Checkpoint | `turbo_gpt0630_krea2_final_forge_bnb_nf4` |
| Model hash | `47a2b78020` |
| Module 1 / 2 | `qwen_image_vae` / `qwen3vl_4b_bf16` |
| Native size | 1024×1448 |
| Aspect ratio | 1:1.4140625 |
| Steps | 4 |
| Sampler | DPM++ 2M SDE |
| Scheduler | Simple |
| CFG / distilled CFG | 1.0 / 1.15 |
| Seeds | 20260712, 20260713, 20260714, 20260715 |

候補1はmodel loadを含み444.918秒、候補2から4はそれぞれ20.935秒、21.096秒、21.290秒で生成された。

### 2.3 候補選定

4候補を同一表示条件で比較し、Candidate 3、seed `20260714`を選定した。選定は非blindの定性的レビューであり、顔・紫角・半眼の紫眼・長い水色の髪・透明な緑色スライム・全身構図の均衡を判断基準とした。

選定画像のSHA-256は次の通りである。

```text
D7BF08C1A59A8ADCB58F3B213897A59784FA52217413644AE72781BBAB183664
```

### 2.4 Negative promptの制約

Forge logは`Negative Prompts are Ignored when CFG = 1.0`と記録した。このため、本例でnegative promptを有効な制約として評価してはならない。重要な禁止条件はpositive promptにも`no text`、`no watermark`、`no tile seams`などとして重ねた。

## 3. VRAM-Canvasによる高解像度化

### 3.1 漸進的解像度計画

採用したpipelineは次の通りである。

```text
1024×1448 native
  → 1728×2432 refinement
  → 2896×4096 4K delivery
  → 5792×8192 8K delivery
  → Smart Finish
```

4K工程は2段、8K工程は4K raw出力からの2倍1段である。4Kと8Kの比率は同じ1:1.4143646であり、√2に対する相対誤差は約0.0107%である。最終画素数はnativeの正確に32倍である。

### 3.2 高解像度refinement prompt

```text
light blue hair, long_wavy_hair, devil’s_horn, purple horn, purple_eyes,
green_slime, jig eyes, smile, jitome, Expressionless. Preserve the selected
source image exactly: the same single adult anime woman, identity, face,
half-lidded purple eyes, restrained tiny closed-mouth smile, two purple devil
horns, very long wavy light-blue hair, elegant dark fantasy outfit, full-body
pose, proportions, camera, framing, stone chamber, lighting, mist, and
translucent green slime geometry. Add only coherent high-frequency detail in
hair strands, horn surface, eyes, fabric, stone, and glossy slime. Do not add or
remove objects, people, limbs, horns, text, logos, signatures, watermarks,
grain, speckles, halos, or tile seams.
```

### 3.3 Tile geometryとVRAM計画

| Parameter | 値 |
|---|---:|
| Explicit VRAM budget | 24.0 GiB |
| VRAM use fraction | 0.85 |
| Model reserve | 5.5 GiB |
| Activation estimate | 4096 bytes/pixel |
| Diffusion tile | 1280px |
| Halo | 160px |
| Core | 960px |
| Core overlap | 80px |
| Grid phases | 2 |
| Low-pass radius | 12px |
| Per-channel detail delta | ±32 |
| Structure sigma | 18 |
| Base-detail sigma | 6 |
| Consensus sigma | 8 |

納品canvas全体ではなく、halo付き1280pxの局所payloadを1枚ずつForge img2imgへ送る。GPU側のcanvas依存空間項は目標の$W\times H$ではなくtileの$T\times T$で上限化される。manifestに記録された全画面/tile面積比は4Kで7.24倍、8Kで28.96倍である。これは総peak VRAMの保証ではなく、空間activation部分の理論比である。

### 3.4 Structure gate

段階基準像を$B$、局所生成候補を$R_i$、低域通過を$L$、高域を$H=I-L$、輝度を$Y$とする。低周波構造の差からstructure gateを得る。

$$
a_i=\exp\left(-\frac{|Y(LR_i)-Y(LB_i)|}{\tau_s}\right)
$$

$a_i$が小さい候補は、基準像の大域構造と一致しないため抑制される。

### 3.5 Base-detail gate

平坦部に局所生成器が偽の粒状textureを作る問題を抑えるため、基準像自身の高周波量を測る。

$$
e_B=\sqrt{\operatorname{mean}_c(H(B)_c^2)},\qquad
b_i=\frac{e_B}{e_B+\tau_b}
$$

最終的な候補残差を次のように制限する。

$$
\Delta_i=\operatorname{clip}\left(
a_i b_i [H(R_i)-H(B_i)],-d,d
\right)
$$

本例の$\tau_b$は6、$d$は32である。$e_B\approx0$の平坦部では$b_i\approx0$となり、基準像に存在しない高周波の注入を抑える。

### 3.6 Consensus gate

重複tile候補の一次・二次モーメントから加重平均$\mu$と不一致$V$を得る。$E_2$を加重二乗平均、$\kappa$をnoise floor、$\lambda$をstrengthとすると、

$$
g_{con}=\exp\left(-\frac{\lambda V}{E_2+\kappa^2}\right),\qquad
O=\operatorname{clip}(B+g_{con}\mu,0,255)
$$

全候補が一致すると$V=0$、$g_{con}=1$であり、合意detailは保持される。異符号や位置ずれで不一致が増えると残差が減衰する。ただし全候補が同じ誤りに合意した場合は検出できない。

## 4. 測定結果

### 4.1 Stage別統計

| Stage | Size | Tile success | Denoise | Mean structure gate | Mean base-detail gate | Mean consensus gate | Mean absolute delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 4K-1 | 1728×2432 | 18/18 | 0.12 | 0.779 | 0.370 | 0.925 | 1.663 |
| 4K-2 | 2896×4096 | 44/44 | 0.08 | 0.786 | 0.285 | 0.967 | 0.842 |
| 8K | 5792×8192 | 140/140 | 0.08 | 0.839 | 0.196 | 0.990 | 0.394 |

後段ほどbase-detail gate平均と平均絶対残差が小さくなった。これは高解像度段でdetail注入量が抑制されたという実装統計であり、知覚品質の向上を直接示すものではない。

### 4.2 残差clip

| Stage | Clipped fraction |
|---|---:|
| 4K-1 | 0.09494% |
| 4K-2 | 0.00561% |
| 8K | 0.00023% |

8Kで±32の上限へ到達した画素割合は約0.00023%であった。

### 4.3 実行時間とresource観測

| 項目 | 結果 | 注意 |
|---|---:|---|
| 4K wall time | 572.971秒 | run開始とraw最終PNGのファイル時刻差 |
| 8K wall time | 1545.639秒 | run開始とraw最終PNGのファイル時刻差 |
| 観測GPU memory | 最大21,656 MiB | 間欠的`nvidia-smi` sampling。真の瞬間peakではない |
| OOM / NaN / HTTP failure pattern | 0 | server log scan |
| 4K API calls | 62/62成功 | 18 + 44 |
| 8K API calls | 140/140成功 | 2 phases |

速度はcheckpoint、attention backend、host負荷、storageによって変わる。ここでのwall timeにSmart Finishは含まない。

## 5. Smart Finish

最終PNGを上書きせず別名で処理した。細かな髪、石材、霧、スライムの意図的detailを守るため、isolated-speckle repairは無効にし、adaptive chroma-mura correctionだけを適用した。

| Metric | 4K before | 4K after | 8K before | 8K after |
|---|---:|---:|---:|---:|
| Chroma p95 | 5.21 | 3.70 | 5.25 | 3.67 |
| Area `Δchroma > 5` | 5.54% | 1.76% | 5.64% | 1.68% |
| Mean chroma shift | - | 0.823 | - | 0.852 |
| Max chroma shift | - | 6.00 | - | 6.00 |
| Despeckle | off | off | off | off |

このchroma指標はLab a/bの局所平滑参照との差に基づく内部heuristicであり、標準化された知覚尺度ではない。補正後も内部判定は`CHECK: light chroma mura detected`であり、完全除去を意味しない。

## 6. 視覚QA

8K全体像に加え、正規化座標から作成した次の6領域を原寸相当で目視した。

| 領域 | 主な確認対象 | 本例の観察 |
|---|---|---|
| Face / horns | 追加角、二重輪郭、眼の不一致 | 明瞭な該当artifactなし |
| Torso / sleeves | 人物重複、輪郭ghost、局所継ぎ目 | 明瞭な該当artifactなし |
| Green slime | 透明面の破断、反復模様、色境界 | 連続性を維持 |
| Feet / floor | 接地破綻、tile seam | 明瞭なseamなし |
| Background left | 平坦部の偽微細模様、縦境界 | 反復hallucinationなし |
| Background right | 平坦部の偽微細模様、縦境界 | 反復hallucinationなし |

「見つからなかった」は「存在しない」の証明ではない。評価者1名、単一画像、非blindである。手は衣装とスライムに隠れる構図のため、この例から手指の正しさを検証したとは扱わない。

## 7. 再現性

### 7.1 出力

| 成果物 | 値 |
|---|---|
| 4K canonical PNG | 2896×4096 RGB |
| 8K canonical PNG | 5792×8192 RGB |
| 8K file SHA-256 | `5AD221B73DB66CFC8593D80E1CBF2EDA8DF0BABB6B6999616CC4A21651421402` |
| 8K pixel SHA-256 | `1341E1275DAF5873DB12F2F8B8FCE08182501EF7D891098FA7B5D1EEAC462933` |
| PNG metadata | `parameters`, `vram_canvas`, `krea2_smart_finish` |

Canonical PNGはSmart Finish出力と画素dataが同一で、`--prompt`の記録不整合だけを別名出力で修正した。元run manifestとSmart Finish出力は保存しており、上書きしていない。

### 7.2 実装上見つかった再現性bug

従来のCLIは`--prompt`で有効promptを上書きしても、最終PNGへ元画像のpromptを残し、`run_manifest.json`にも有効promptを保存していなかった。画質には影響しないが、実験再現性を損なう。

本ケーススタディと同時に次を修正した。

1. manifestへ`prompt`と`negative_prompt`を保存する。
2. 最終PNGの`parameters`へ実際に使用したpromptを保存する。
3. 最終seedとsizeも実行値へ更新する。
4. dry-runとmock Forge APIを用いたintegration testでprompt、negative prompt、sizeを再読込確認する。

## 8. 限界と今後の評価

1. 単一prompt、単一選定画像、単一checkpointの結果である。
2. 候補選定とartifact判定は非blindで、評価者間一致を測っていない。
3. Krea2のCFG 1.0ではnegative promptが無視された。
4. GPU memoryは間欠samplingで、instrumented peakではない。
5. Wall timeは環境依存で、他方式との同条件比較ではない。
6. Consensus gateは、全候補が同じ誤りに合意した場合を検出できない。
7. 局所法は、基準像にない大域構図、長距離意味関係、文字や手指の正しさを保証しない。
8. Chroma-mura metricは内部heuristicで、標準化された知覚尺度ではない。

今後は複数prompt、seed、画像種、modelを用い、blind pairwise preference、identity/structure distance、tile-seam detector、連続GPU telemetry、base-detail gateのablation、同一計算量条件での対照方式を事前登録して評価する必要がある。

## 9. 結論

Krea2 A-Seriesで生成した1024×1448の縦長イラストを、VRAM-Canvasのstructure gate、base-detail gate、consensus gateによって段階的に5792×8192へ拡張し、Smart Finishと原寸相当QAまで完了した。全202局所API呼出しは成功し、8K最終PNGは有効prompt、VRAM-Canvas manifest、Smart Finish reportを埋め込み保持する。

この結果は、RTX 3090 24 GiB上で47.45 MPの納品canvasを処理できた再現可能な単一例である。一般的な知覚品質の優位性については、今後の対照実験に結論を留保する。

## 参考文献

1. R. Rombach et al., [High-Resolution Image Synthesis with Latent Diffusion Models](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html), CVPR, 2022.
2. O. Bar-Tal et al., [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://proceedings.mlr.press/v202/bar-tal23a.html), ICML, 2023.
3. R. Du et al., [DemoFusion: Democratising High-Resolution Image Generation With No $$$](https://openaccess.thecvf.com/content/CVPR2024/html/Du_DemoFusion_Democratising_High-Resolution_Image_Generation_With_No__CVPR_2024_paper.html), CVPR, 2024.
4. Z. Lin et al., [AccDiffusion: An Accurate Method for Higher-Resolution Image Generation](https://arxiv.org/abs/2407.10738), 2024.
5. T. Vontobel et al., [HiWave: Training-Free High-Resolution Image Generation via Wavelet-Based Diffusion Sampling](https://arxiv.org/abs/2506.20452), 2025.
