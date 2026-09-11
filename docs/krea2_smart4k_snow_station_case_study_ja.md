# 位相合意付き帯域制限タイル拡散による任意画像の4K化

## Krea2 / Forge Neo / RTX 3090 による単一画像実測短報

**著者:** AiWithYou<br>
**実験日:** 2026-07-14<br>
**実装:** `VRAM-Canvas 4K/8K Highres` + `Krea2 Local Supersample Detail`

## 要旨

Krea2 Turbo 系モデルの推奨生成域を超える画像を、全キャンバスの構図と小さな人物の同一性を保ちながら4K化するため、段階拡大、halo付き重複タイル img2img、周波数分離残差、半strideずらしの2位相合意、disk-backed合成、局所超標本化を組み合わせた。添付された1915×821のアニメ調駅舎画像を、アスペクト比を保つ長辺4Kの4096×1756へ変換した。処理は2800×1200と4096×1756の2段、合計44タイルで行い、44/44成功、skip 0、拡散工程387.911秒、GPU全体の観測peak VRAM 22,789 MiBで完走した。後段のcoherent-detail処理は平坦領域を0画素変更したまま、方向整合領域の重み付きdetail energyを1.263811倍にした。

人物の顔を512px payloadから1536pxへ拡大生成して縮小差分だけを戻す追加実験では、4タイル×2候補を73.655秒で処理したが、全4タイルが `detail_energy_did_not_increase` と判定された。最終画素は入力4Kと完全一致した。これは全候補を無条件に貼り付ける方式と異なり、改善根拠のない局所生成を no-op へ戻す fail-closed 特性を示す。本稿の数値は単一画像・単一seedの結果であり、普遍的な画質改善率ではない。

**Keywords:** Krea2, image-to-image, 4K, tiled diffusion, progressive upscaling, frequency residual, consensus gate, local supersampling, fail-closed

## 1. 背景と問題設定

Krea2公式実装は Raw を最大約1K、Turboを約1K〜2Kの生成域として案内している [1]。Krea2 Technical Reportも、学習解像度を256、512、1024pxへ段階的に上げたこと、native 2K/4Kを将来能力として挙げている [2]。したがって本実装は、4096幅をKrea2へ一括投入する native 4K生成ではない。

重複領域を持つ複数の拡散経路を重み付き統合する考え方は MultiDiffusion [3] および Mixture of Diffusers [4] に先行例がある。DemoFusion [5] は、低解像度の全体整合性を保ちながら局所詳細を追加するためのprogressive upscalingとresidual guidanceを報告している。一方、単純な重複平均だけでは次の問題が残る。

1. タイルごとに人物の顔、手、文字、小物を微妙に描き替える。
2. 平坦面へgrainや反復模様を発生させる。
3. 低周波の明度・色・輪郭移動まで合成して構図を変える。
4. 一度だけ出た局所hallucinationを「detail」と誤認する。
5. 大きく生成して縮小した候補が元画像よりぼけても貼り付けてしまう。

本研究の目的は、生成候補の全面採用ではなく、入力画像が既に持つ構図と画風を基準とし、複数の独立観測が支持した小振幅・高周波の差分だけを採用することである。ここでいう「4K」は入力アスペクト比を維持した長辺4096pxを意味し、16:9 UHD 3840×2160へのcropや引き延ばしは行わない。

## 2. 提案手法

### 2.1 段階拡大とVRAM固定タイル

入力を \(I_0\)、目標を \((W,H)\) とする。各辺の拡大率が2を超えないように段階列 \(S_1,\ldots,S_n\) を作り、各段の基準画像をLanczosで拡大する。

\[
B_s = \mathcal{R}_{S_s}(I_{s-1}),\qquad
\max\left(\frac{W_s}{W_{s-1}},\frac{H_s}{H_{s-1}}\right)\le 2.
\]

各段はpayload edge \(T\)、halo \(h\)、core edge \(c=T-2h\)、core overlap \(o\) のタイルへ分ける。GPUへ載る空間活性量は目標 \(W\times H\) ではなく、おおむね \(T\times T\) で上限が決まる。今回の最終段では \(T=1280\)、\(h=160\)、\(c=960\)、\(o=80\) とした。

### 2.2 任意画像向けKrea2 guidance

Forge GUIでは、ユーザーのpromptをprefixとして変更せず、次の意味を持つ一般化suffixを1回だけ追加する。

- 人物の同一性、顔比率、年齢、表情、視線、手指数、ポーズを保持する。
- カメラ、framing、被写界深度、照明、物体数、文字、scene geometryを保持する。
- 髪、虹彩、布、木、石、植生、透明物、液体、線画など、入力に実在する材質だけを精密化する。
- アニメ、flat-color、graphic designへ写真風の毛穴やgrainを強制しない。
- 人物、手足、眼、物体、文字、輪郭、反復模様、tile seamを追加しない。

旧suffixに含まれていた特定題材の `horn` / `slime` 語彙は、任意画像へ不要な内容を誘導し得るため除去した。

### 2.3 既存detail残差

タイル \(i\) のKrea2出力を \(K_i\)、同じ入力contextを \(B_i\)、半径 \(r\) の低域演算を \(L_r\) とする。まず低周波を除いた差を得る。

\[
\Delta_i^{E}
= \left[(K_i-L_r(K_i))-(B_i-L_r(B_i))\right]
\odot G_{\mathrm{structure}}
\odot G_{\mathrm{base}}.
\]

\(G_{\mathrm{structure}}\) は大きなluma driftを抑え、\(G_{\mathrm{base}}\) は元画像の局所detailと整合する差分を優先する。RGB各channelの差分は最大±32 codeへ制限する。

### 2.4 novel-detail残差

元画像にまだ存在しない微細描写を全面的に許すとnoise化しやすい。このため2〜8px相当の輝度帯だけを候補とし、色差と低周波構造を直接描き替えない。

\[
\Delta_i^{N}
= \operatorname{clip}_{[-8,8]}
\left(
Y\left[(L_2(K_i)-L_8(K_i))-(L_2(B_i)-L_8(B_i))\right]
\odot G_{\mathrm{structure}}
\odot G_{\mathrm{novelty}}
\right).
\]

輝度差を3channelへ同量で戻すため、novel branchは直接の色変化を作らない。

### 2.5 重なり、2位相、合意gate

core境界にはsmoothstep重み \(w_i(x)\) を与える。第1位相の通常gridに加え、第2位相を半strideずらして別seedで処理する。画素ごとの一次moment、二次momentをdisk-backed float32配列へ蓄積する。

\[
\mu=\frac{\sum_i w_i\Delta_i}{\sum_i w_i},\qquad
v=\frac{\sum_i w_i\|\Delta_i\|_2^2/3}{\sum_i w_i}
-\|\mu\|_2^2/3.
\]

\[
g=\exp\left(-\frac{4v}{e+\sigma^2}\right),\qquad
I_s=B_s+g\mu.
\]

ここで \(e\) は重み付きdetail energy、\(\sigma=8\) codeである。複数位相で方向が一致する差分は通り、一度だけ現れた強い差分は分散 \(v\) により減衰する。novel branchには独立coverageも要求する。

### 2.6 Smart Finish

最終キャンバス上で、既存輝度高域へtexture energy、structure-tensor coherence、強輪郭・暗部・白飛びguardを掛ける。今回のcolor strengthは0であり、chroma補正は行っていない。候補は次を満たすときだけ採用する。

- 方向整合領域のdetail energyが増える。
- 平坦領域の変更画素が0。
- channel clippingが0.05%以下。
- 不採用時はbit-identical no-opである。

### 2.7 局所超標本化

顔などのROIでは、約512×512のpayload \(B\) を1536または2048へLanczos拡大してKrea2へ渡す。ただし高解像度候補を直接貼らない。モデルを通さない往復基準 \(C_0\) と、生成候補の同一縮小経路 \(C_1\) を比較する。

\[
C_0=D(U(B)),\qquad C_1=D(K(U(B))),\qquad
\Delta_{local}=\operatorname{BandPass}(C_1-C_0).
\]

linear-light area縮小、低周波除去、luma/chroma cap、強輪郭保護、detail増加、drift、clip、境界残差を検査する。2候補設定では画像や残差を平均せず、一方を代表候補、他方をagreement判定にだけ用いる。不合格タイルは \(\Delta_{local}=0\) とする。

## 3. 実験

### 3.1 環境

| 項目 | 値 |
|---|---|
| OS / shell | Windows 11 / PowerShell 7.6 |
| GPU | NVIDIA GeForce RTX 3090 24 GiB |
| Forge | Forge Neo, branch `neo` |
| Checkpoint | `turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors` |
| Checkpoint SHA-256 | `47a2b7802017a39621bfe48fc779ce82ac76ac6e3d7206eecb5bbcbbb3af6f27` |
| Additional modules | `qwen_image_vae.safetensors`, `qwen3vl_4b_bf16.safetensors` |
| Sampler / schedule | DPM++ 2M SDE / Simple |
| CFG | 1.0 |
| Global seed | 3883506083 |

GPU数値は `nvidia-smi` によるGPU全体の観測値であり、プロセス専有値ではない。

### 3.2 入力と4K設定

| 項目 | 値 |
|---|---:|
| 入力 | 1915×821 RGB PNG |
| 入力file SHA-256 | `B8563596CFEDE2055EF9BBA0D3258CDDA9EC3EBCB8A3C17DFD5E4637A4B441E9` |
| 目標 | 4096×1756, 7,192,576 pixels |
| 段階 | 2800×1200 → 4096×1756 |
| Tile / halo / core / overlap | 1280 / 160 / 960 / 80 px |
| Grid phases | 2 |
| Steps | detail-adaptive 3〜4 |
| Denoise | 0.16 → 0.13 |
| Detail / novel gain | 1.25 / 1.0 |
| Detail / novel cap | ±32 / ±8 code |
| Smart Finish | ON, chroma 0, detail strength 0.75 |

### 3.3 4K結果

| 指標 | 結果 |
|---|---:|
| 計画 / 成功 / skip | 44 / 44 / 0 tiles |
| 第1段 / 第2段 | 16 / 28 tiles |
| 拡散工程 | 387.911 s |
| Smart Finish | 4.994 s |
| GPU telemetry samples | 350 |
| peak VRAM | 22,789 MiB |
| peak GPU utilization | 100% |
| peak temperature | 84°C |
| 第2段 consensus gate平均 | 0.866941 |
| 第2段 disagreement平均 | 1.553024 code |
| 第2段 novel consensus gate平均 | 0.605080 |
| Smart Finish変更 | 1,345,641 px, 18.708749% |
| 平坦領域の変更 | 0 px |
| channel clipping | 0.0024748% |
| 重み付きdetail energy | 2.335850 → 2.952073, 1.263811× |
| 最終file SHA-256 | `45CF42C6CEC5447D6ABEAD221956649429F0541E13D4927FCE7CA10730391BD3` |

全体像と固定7地点の1024×1024原寸cropを目視した。人物の顔、髪流れ、新聞、衣装、脚、車、アーチ窓、木枠、床反射について、明瞭な人物増殖、手足増加、二重輪郭、周期模様、tile seamを認めなかった。ただしこれは単一観察者による1事例の定性評価である。

### 3.4 顔ROI超標本化

4K上の顔周辺 `2580,560,2870,930` をROIとし、Ultra Detail 1536を実行した。

| 指標 | 結果 |
|---|---:|
| Payload / core / overlap | 512 / 384 / 64 px |
| Process edge | 1536 px |
| Candidate count | 2 |
| Tile count | 4 |
| Steps / denoise | 5 / 0.15 |
| Duration | 73.655 s |
| peak VRAM / temperature | 22,025 MiB / 85°C |
| 採用 / no-op | 0 / 4 tiles |
| 変更画素 | 0 px |
| 入出力pixel SHA-256 | `78d02e584a4a8bf447a541ae5e3c34f7e0c5b0098fc26b7d0d40741db9634eb3` |

全候補の棄却理由は `detail_energy_did_not_increase` だった。代表QA cropでも、1536候補を512へ戻した画像は基準より軟化しており、残差を採用しない判断と整合した。ROI外だけでなく画像全体がbit-identicalである。

ROI Ultra 2048も試行したが、最初の1候補が806秒時点でも完了せず、VRAM約23.1 GiB、GPU使用率約100%を維持したため明示的に中断した。2048を自動で1536へ落とすfallbackは行っていない。本環境では1536が実用既定、2048は極小ROIで時間を許容できる場合のみの実験設定である。

## 4. 考察

本手法の重要な結果は、すべての生成を採用したことではない。全体4Kパスでは2位相が支持した帯域制限残差とcoherent detailが採用された一方、顔の追加超標本化は全候補が棄却された。したがって、「顔だから必ず再生成する」という固定規則ではなく、画像ごとに追加情報の根拠を検査できた。

一方、生成モデルが加える微細描写は、未知の真値を復元したものとは限らない。入力とpromptに整合する推定描写であり、科学・医療・証拠画像の復元へ用いるべきではない。PSNRやSSIMを計算できる4K ground truthも存在しないため、本稿は超解像ベンチマークを主張しない。

2位相とoverlapは計算量を増やす。今回も最高解像度段が28/44タイルを占め、総時間の中心となった。これはDemoFusionが報告するprogressive / patch-wise inferenceの計算負荷 [5] と同じ方向の制約である。ただし空間活性量をtile edgeで制限できるため、7.19MPの全キャンバスを拡散モデルへ一括投入せずRTX 3090で完走できた。

## 5. Forge GUIでの利用

1. Forgeを再起動し、Krea2 checkpoint、Qwen Image VAE、Qwen3-VLを選択する。
2. `img2img`へ入力画像と、その画像を正確に説明するpromptを入れる。
3. `Script` から `VRAM-Canvas 4K/8K Highres` を選ぶ。
4. `4K Smart - long edge 4096 + profile` を押す。
5. `Quality Profile` が `Krea2 Dense Detail 4K`、`Grid Phases` が2であることを確認する。
6. `Generate` を押し、4K全体像と100% cropを確認する。
7. 顔などに追加処理が必要な場合だけ、完成4Kをimg2imgへ入れ直し、`Krea2 Local Supersample Detail` を選ぶ。
8. まず `Ultra Detail 1536` と `ROI Boxes` を使う。座標は4K画像のpixel座標で指定する。
9. 2048は自動fallbackされない高負荷設定であり、必要な小領域だけに限定する。

## 6. 再現性

主要成果物:

- 4K: `output/krea2_smart4k_snow_station/smart8k_20260714_013427_407859/smart4k_preflight.png`
- 4K manifest: 同directoryの `smart8k_manifest.json`
- VRAM-Canvas manifest: `vram_4k/vram_canvas_20260714_013428_861365/run_manifest.json`
- 顔ROI manifest: `output/krea2_smart4k_snow_station/face_refine_1536/local_supersample_20260714_015753_389398/experiment_manifest.json`
- 顔ROI QA: `output/img2img-images/krea2_local_supersample_qa/20260714_015905_497128/`

4K PNGは `parameters`、`vram_canvas`、`krea2_smart_finish` text chunkを保持する。局所処理PNGはさらに `krea2_local_supersample` chunkを持つ。

## 7. 結論

Krea2のnative域を尊重しつつ、段階拡大、halo付きタイル、周波数分離、2位相合意、bounded novel detail、Smart Finish、fail-closed局所超標本化を組み合わせることで、1915×821画像を4096×1756へRTX 3090上で変換した。全体4Kパスは44/44タイル成功、顔ROIは改善根拠がない候補を4/4 no-opへ戻した。画質向上を無条件に保証する方式ではなく、生成の自由度を局所・帯域・振幅・合意で制限し、証拠が弱い場合に元画像を保つ方式である。

## 参考文献

1. Krea AI, [Krea 2 official inference code - Usage](https://github.com/krea-ai/krea-2#usage), 2026.
2. S. Lee et al., [Krea 2 Technical Report](https://www.krea.ai/blog/krea-2-technical-report), 2026.
3. O. Bar-Tal, L. Yariv, Y. Lipman, T. Dekel, [MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation](https://proceedings.mlr.press/v202/bar-tal23a.html), ICML 2023.
4. A. Barbero Jiménez, [Mixture of Diffusers for scene composition and high resolution image generation](https://arxiv.org/abs/2302.02412), arXiv:2302.02412, 2023.
5. R. Du et al., [DemoFusion: Democratising High-Resolution Image Generation With No $$$](https://openaccess.thecvf.com/content/CVPR2024/html/Du_DemoFusion_Democratising_High-Resolution_Image_Generation_With_No_CVPR_2024_paper.html), CVPR 2024.
