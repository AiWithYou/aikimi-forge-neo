# 文脈拡大・単一領域再生成による4K顔局所再描画

**Context-Magnified Single-ROI Regeneration for Focused Face Rewriting in 4K Images**

AiWithYou — Technical Short Paper / 2026-07-14

## 要旨

大画像を固定タイルへ分割するimg2imgでは、小さな顔がタイル境界に掛かり、独立seedで部分ごとに処理されるため、顔全体の整合した再描画にならない場合がある。本研究は、書き込み対象と生成用文脈を分離する `Focused ROI Rewrite` をForge Neoへ実装した。元画像上のtight targetを唯一の書き込み領域とし、その長辺の2倍を一辺とする正方形contextを切り出す。context全体を1536×1536へLanczos拡大し、Krea2 img2imgで顔全体を1回に再生成する。無処理拡大像と生成像を同じlinear-light area縮小へ通し、その差分をtarget内だけへ20pxのinward smoothstepで合成する。target外は元のuint8画素を直接コピーする。

RTX 3090上の4096×1756画像で、120×150pxの顔target、300×300pxのcontext、1536px process input、実効5.12倍、6 steps、denoise 0.38、2候補を評価した。detail gateを通った候補2を採用し、target内16,646画素（92.4778%）が変化、target外は0画素、target内RGB絶対差は平均9.007、95 percentile 26、最大77 codeであった。処理時間は49.1秒、GPU全体の標本ピークは20,957MiBである。高解像度候補と4K書き戻しでは、元4Kで潰れていた両目、虹彩、まつ毛、鼻口、顎線の再構成を確認した。本単一事例は知覚品質の一般的改善を証明しないが、「実際に拡大再生成すること」と「指定領域外を一画素も変更しないこと」を同時に実証した。

**Keywords:** Krea2, focused regeneration, region of interest, context crop, round-trip compensation, 4K, RTX 3090

## 1. はじめに

拡散モデルで4K画像を局所仕上げするとき、単純な固定タイルはVRAMを抑えられる一方、意味単位とタイル境界が一致しない。とくに小さな顔が複数タイルへ分断されると、左右の目、輪郭、髪が別seedで生成され、各タイルを1536へ拡大していても「顔全体を拡大して再生成した」ことにはならない。また、denoiseが低すぎる保守的残差方式は、元画像保護には有効でも顔の再構成をほぼ生じない。

本手法の設計目標は次の4点である。

1. 1つの顔を1つのmodel inputで生成し、タイル分断しない。
2. 小さなtargetへ周辺の髪・頭・照明をcontextとして与える。
3. Krea2の縮小後の再描画を確実に4Kへ反映する。
4. target外はbit-exactに元画像を維持する。

## 2. 手法

### 2.1 Targetとcontextの分離

入力画像を $X$、書き込みtargetを $T=(l,t,r,b)$、target長辺を $L=\max(r-l,b-t)$、Context Scaleを $k$ とする。生成用正方形context $B$ の一辺 $s_B$ は次式で決める。

$$
s_B=\lceil kL\rceil
$$

$B$ は $T$ の中心へ配置し、画像端を越える部分だけedge padする。$T$ は書き込み範囲、$B$ は読取りと生成の文脈であり、両者を同一視しない。本実装はtargetごとに1個の $B$ を作るため、同じ顔を複数の独立タイルへ分割しない。重複targetは二重加算を避けるため開始前に拒否する。

### 2.2 拡大再生成と往復補償

context $B$ のsRGB Lanczos拡大を $U(B)$、Krea2 img2imgを $K$、linear-light area縮小を $D$ とする。無処理往復像 $C_0$ と候補 $i$ の往復像 $C_{1,i}$ は

$$
C_0=D(U(B)),\qquad C_{1,i}=D(K(U(B);z_i)),\qquad \Delta_i=C_{1,i}-C_0
$$

である。$C_0$ と $C_1$ は同じ $D$ を使うため、拡大縮小だけで生じるround-trip差を候補差分へ混入させない。高解像度候補を直接貼る、候補だけ別resizerで縮小する、$C_1-B$ を差分にする処理は行わない。

通常のdetail modeは $\Delta_i$ の低周波を除去し、luma/chromaを制限する。Focused modeは顔形状を含む再描画が目的なので、選択候補のフル $\Delta_i$ を用いる。ただし往復補償とtarget外不変は維持する。

### 2.3 候補選択

2候補はglobal seed、target座標、candidate indexから決定論的な別seedを作る。各候補についてdetail-energy、低周波drift、clipping、境界差、quality scoreを計測する。合格候補が1枚以上あれば、その集合からquality score最小を選ぶ。全候補が不合格の場合だけ、全候補中score最小を選び、元の棄却理由をmetadataへ残す。候補画像や差分の平均は行わない。

### 2.4 限定書き戻し

target境界からの内側距離を $d_T(p)$、feather幅を $f$ とし、$M_T(p)=\operatorname{smoothstep}(0,f,d_T(p))$ とする。出力は

$$
Y(p)=
\begin{cases}
X(p)+M_T(p)\Delta_{i^*}(p), & p\in T\\
X(p), & p\notin T
\end{cases}
$$

である。Focused経路では正規化分母を1に固定し、target featherがweight正規化で相殺されないようにする。$p\notin T$ はlinear-light往復へ通さず、入力uint8値を直接コピーする。

## 3. 実装

Forge img2imgの `Krea2 Local Supersample Detail` へ `Focused ROI Rewrite` modeと `Focused Face Rewrite 1536` profileを追加した。profileはProcess Edge 1536、6 steps、denoise 0.38、2候補、Context Scale 2.0、Rewrite Feather 20pxである。GUIはtarget座標、context倍率、featherを入力できる。model処理前にKrea2 engine、Qwen Image VAE、Qwen3-VL、target、非重複、実拡大率、tile上限、一時diskを検査する。

処理中は `Restore faces`、wrap-around `Tiling`、mask、内部画像保存を無効にし、成功・例外・中断・OOMの全経路でprocessing stateを復元する。最終PNGへtarget、context box、payload辺、実効拡大率、candidate seed、選択候補、quality統計、RGB pixel SHA-256を記録する。診断optionではprocess input、高解像度候補、$C_0$、$C_1$、before/after payloadをtimestamp別directoryへ保存する。

## 4. 実験

### 4.1 条件

入力は添付画像をKrea2 Smart 4Kで4096×1756へ仕上げたRGB PNGである。base promptは駅構内、雪、白髪の人物、新聞を記述した元manifestの文字列をそのまま使用し、SHA-256は `F3D288BFA6AB…` である。global seedは2846268111、候補seedは233307026と2360494037。checkpointは `turbo_gpt0630_krea2_final_forge_bnb_nf4.safetensors`、VAEはQwen Image VAE、text encoderはQwen3-VL 4B bf16、GPUはRTX 3090 24GBである。

| 項目 | 値 |
|---|---:|
| 入力 / 出力 | 4096×1756 / 4096×1756 |
| target $T$ | (2608,635,2728,785), 120×150px |
| context $B$ | (2518,560,2818,860), 300×300px |
| process input | 1536×1536px |
| 実効拡大率 | 1536 / 300 = 5.12× |
| steps / denoise / candidates | 6 / 0.38 / 2 |
| feather | source側20px inward |

### 4.2 評価

入力と出力のdecoded RGBを比較し、target内外の変更画素数、RGB code絶対差、寸法、hashを計測した。Forge APIの処理時間を記録し、`nvidia-smi` を1秒間隔で取得した。GPU telemetryはプロセス専有値でも瞬間ピーク保証でもなく、実行中のGPU全体標本である。視覚確認は高解像度候補、同一targetの4倍表示、4K全体で行った。

## 5. 結果

| 指標 | 結果 |
|---|---:|
| 選択候補 | 2（detail gate合格） |
| target内変更 | 16,646 / 18,000画素（92.4778%） |
| target外変更 | 0画素 |
| target内RGB絶対差 | mean 9.007 / p95 26 / max 77 code |
| 処理時間 | 49.1秒 |
| peak VRAM | 20,957MiB |
| peak GPU utilization / temperature | 100% / 84°C |
| 入力pixel SHA-256 | `78d02e584a4a…` |
| 出力pixel SHA-256 | `618cfba205a0…` |

候補2はdetail-energy増分が正で品質gateを通り、Focused経路の選択対象になった。高解像度候補では左右の目、虹彩、上まつ毛、鼻口、顎線、前髪の境界が再構成された。4Kへの縮小・feather後も元のぼやけた顔とは異なる線が残り、単なるLanczos拡大ではない。出力全体は元画像と同じ4096×1756であり、target外変更0画素を機械確認した。

## 6. 考察

旧固定payload方式の問題はProcess Edgeの数値ではなく、意味単位と生成単位の不一致だった。顔の各部分を別々に1536へ拡大しても、モデルは顔全体を同時に観測できない。Focused経路はtargetとcontextを分け、300px contextを1枚として1536へ5.12倍拡大することで、顔全体を1回のsamplingへ収めた。

target内の92.48%が変化し、target外が0であることは、再生成が実際に反映された一方、変更範囲が座標契約どおり限定されたことを示す。inward featherにより矩形境界で差分を0へ収束させるが、targetが小さすぎると有効な全強度領域も狭くなる。顔だけでなく前髪や顎まで含むtight boxと、長辺の約2倍のcontextが実用上の出発点になる。

本手法は元画像を復元する超解像ではなく、Krea2による条件付き再描画である。ground truthの未知細部を取り戻したとは言えず、identity、表情、目形状が変わる可能性がある。quality gateは候補選択の補助であり、知覚的顔品質を直接測定しない。高解像度候補と4K書き戻しの目視確認は必要である。

## 7. 限界

本評価は画像1枚、target 1個、prompt 1件、seed系列1件である。複数画風・顔サイズ・写真・複数人物、他手法との盲検比較、ground truth、複数評価者、identity embedding距離を含まない。denoise 0.38、Context Scale 2.0、feather 20pxの一般最適性は未検証である。候補生成は確率的であり、別seedで形状変化や平滑化が生じる可能性がある。

## 8. 結論

顔targetと生成contextを分離し、顔全体を1枚の1536入力へ5.12倍拡大して再生成し、round-trip補償後の差分をtarget内だけへ戻すFocused ROI Rewriteを実装した。実機例ではdetail gate合格候補を採用し、target内16,646画素を変更しながらtarget外を0画素変更に保った。これにより、固定タイルを単に拡大しただけの処理から、意味単位を保った局所再生成へ移行できた。実運用では正確なbase prompt、1顔1target、100%比較、QA保存を必須とする。

## 参考文献

1. S. Lee et al., “Krea 2 Technical Report,” Krea, 2026. <https://www.krea.ai/blog/krea-2-technical-report>
2. R. Rombach et al., “High-Resolution Image Synthesis with Latent Diffusion Models,” CVPR, 2022. <https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html>
3. O. Bar-Tal et al., “MultiDiffusion: Fusing Diffusion Paths for Controlled Image Generation,” ICML, 2023. <https://proceedings.mlr.press/v202/bar-tal23a.html>

## 再現用成果物

- 実行器: `tools/run_krea2_local_supersample_experiment.py`
- manifest: `output/krea2_smart4k_snow_station/focused_face_rewrite_1536/local_supersample_20260714_122916_778319/experiment_manifest.json`
- 4K出力: 同directoryの `krea2_local_supersample.png`
- 差分検証: 同directoryの `focused_roi_validation.json`
- 比較画像: 同directoryの `focused_face_comparison.png`
- 入力file SHA-256: `45CF42C6CEC5447D6ABEAD221956649429F0541E13D4927FCE7CA10730391BD3`
- 出力file SHA-256: `47EE4FAA7E42A277FC6581BE91F6B9413CEBDDBE3ABE42FCF52574EA2428F0D7`
