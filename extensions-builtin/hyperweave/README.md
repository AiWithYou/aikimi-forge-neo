# HyperWeave 4K/8K for Forge

HyperWeave は、通常の補間型超解像ではなく、現在 Forge に読み込まれている生成モデルで入力画像を段階的に再作画する img2img Script です。入力を構図・人物配置・ポーズ・表情・顔向き・髪型シルエット・衣装・物体・代表色・画風の設計図として扱い、その制約を満たす候補から、髪の内部線、目やまぶた、布目、縫い目、紙・木・金属・透明体の反射などの中周波／高周波ディテールを採用します。

これは元画像に存在した情報の「真の復元」ではありません。特に小さい顔の瞳、まつ毛、口、材質の微細構造は生成モデルによる推定です。

## Forge への統合

本機能は `extensions-builtin/hyperweave` の独立 Extension です。Forge 本体、通常 img2img、txt2img、Extras、既存 upscaler、VRAM Canvas、DetailWeave、ControlNet のコードパスへモード分岐を追加していません。

Forge の img2img を開き、ページ下部の `Script` から `HyperWeave 4K/8K` を選択します。画面上部で納品サイズ、再作画Preset、内容Profileを選ぶと、Exact Steps、lock、tile、accumulatorをライブプランで確認できます。Custom長辺またはCustom幅・高さは、そのtarget modeを選んだときだけ表示されます。8K、Max Overdraw、未入力のCustomサイズは注意表示へ切り替わります。低レベルのpass、mask、quality constraint、prompt suffix、debug値は折りたたみ内にあります。

`HyperWeaveを有効化` をオフにした場合は、選択 Script 内から通常の `processing.process_images()` を呼び、WebUI と REST API の双方で通常 img2img と同じ経路へ戻ります。

内部タイル生成は次を引き継ぎます。

- 現在読み込まれている checkpoint と VAE
- img2img の prompt / negative prompt
- sampler / scheduler / CFG / distilled CFG / shift
- prompt 内の LoRA と embedding
- Forge の dtype、autocast、offload、VAE tile 設定
- Forge の AlwaysVisible Script callback 経路

HyperWeave 自身は選択式 Script なので、内部 `processing.process_images()` から自身の `run()` が再帰することはありません。ControlNet などの既存 AlwaysVisible Script を無効化も shallow-copy もしません。既に有効な unit は Forge の正式 callback 経路で処理されます。HyperWeave 側から新しい ControlNet/IP-Adapter unit を推測して注入することはありません。

## 処理フロー

各入力画像は次の順で処理されます。

1. Source Analyzer
   - linear RGB、alpha、輝度、複数スケール edge、structure tensor、局所 coherence、flatness、texture、manual mask、顔 ROI を元入力座標で解析します。
2. Stage Planner
   - 1 stage の倍率を 2.0 以下に制限します。
   - 例: 1024→2048→4096、1024→2048→4096→8192。
   - diffusion canvas は VAE alignment に切り上げ、返却時は指定寸法へ正確に戻します。
3. Global Anchor
   - 全構図、照明、色、画風を揃える低 strength の基準候補を作ります。
4. Global Overdraw
   - candidate index ごとに画像全体を完成させ、hard constraints 通過候補だけを ranking します。
5. Frequency-aware composition
   - candidate−anchor を `HIGH_0 / HIGH_1 / MID_HIGH / MID / MID_LOW / LOW` に分解します。
   - linear RGB の輝度と色差を分け、structure protection、輪郭方向、new-edge、tile variance、round-trip、manual mask を帯域別に掛けます。
   - 単純な Laplacian 最大候補や候補画像全面貼り付けは行いません。
6. Material redraw
   - edge、texture、flatness、manual boost、全人物の Face Core union から Detail Potential Map を作り、衣装・本・紙・木・金属・透明体などの候補残差を限定採用します。
   - Face Core 内の Material 残差は採用しません。
7. Micro-detail
   - 最終 stage だけ、`HIGH_0 / HIGH_1 / 一部 MID_HIGH` を低 strength で採用します。
   - 全人物の Face Core を write mask から除外します。
8. Hair redraw
   - 人物別の head 近似領域から全人物の Face Core を保護し、元の方向場に沿う細線を優先します。
   - 直交線、crossing、シルエット差を減点または reject します。
9. Face redraw
   - Material / Micro / Hair 後の current canvas を context とし、最後の意味的再作画 pass として実行します。
   - 人物ごとの広い context crop は重なっても統合せず、Face Core component を評価 mask、所有権付きの拡張領域を書込 mask として使います。
   - 顔候補も mask-aware round-trip、色、対称 edge、new edge、境界で hard reject し、周波数合成します。
10. Seam analysis
    - 計画 tile boundary の residual 勾配を同数の非境界線と比較し、閾値超過箇所だけを局所整合します。
11. Low-frequency back projection
    - 前段へ縮小した低周波誤差だけを戻します。髪の細線、布目、まつ毛などの高周波は補正対象にしません。
12. exact resize、最終出力の人物別 face metrics、metadata、保存、一時領域 cleanup

## タイルと Coordinate Noise

既定 geometry は次です。

| 項目 | 既定値 |
|---|---:|
| Model input | 1280×1280 |
| Writable core | 960×960 |
| Context | 各辺 160 px |
| Stride | 768 px |
| Nominal core overlap | 192 px |

最終 row / column は画像端へ揃え、全画素を被覆します。画像外は reflect padding、極小画像で reflect が成立しない場合だけ edge padding です。外周へ接する blend window は外端を 0 にしません。

各 stage / pass / candidate / ROI について、BLAKE2b で固定 namespace seed を作り、PCG64 で CPU 側の全体 latent noise canvas を一枚生成します。タイルは絶対 latent 座標の crop を使うため、重複領域の初期 latent noise は bit-exact に一致します。Python の `hash()` には依存しません。`seed=-1` はジョブ開始時に一度だけ concrete seed へ解決され、PNG metadata に記録されます。

Forge の通常の静止画 latent `(B,C,H,W)` と、Qwen Image / Krea2 が使う singleton temporal latent `(B,C,1,H,W)` の双方へ明示的に適応します。時間軸が 1 より大きい video latent は、静止画用ノイズを暗黙 broadcast せず明確に失敗させます。先に実行された AlwaysVisible callback が初期ノイズへ加えた差分は保持し、ランダム基底だけを座標ノイズへ置換します。

現在の保証範囲は「初期 latent noise」です。SDE/Brownian/ancestral sampler が sampling step 内で追加生成する secondary noise は Forge sampler 内部の契約であり、HyperWeave は sampler を別実装へ置換しません。この制限は metadata の `coordinate_noise.scope` に記録されます。

## Candidate hard constraints と ranking

候補は先に以下を検査します。

- NaN / Inf / clipping
- source または前段解像度への round-trip SSIM
- low-frequency error
- color drift
- edge displacement
- large continuous new edges
- duplicate-edge proxy
- tile boundary residual

通過候補だけを、Anchor の同帯域 RMS に対して正規化した MID/MID_HIGH/HIGH detail、勾配が有効な画素だけの方向一致、line continuity、material richness、style consistency、noise penalty、duplicate edge、境界、structure error で ranking します。raw frequency energy と正規化後の detail score は metadata で区別します。

全画面候補と ROI 候補は hard constraints を通るだけでは採用されません。同じ reference、mask、境界条件で評価した current Anchor より `Candidate score margin over Anchor`（既定 0.02）以上高く、正規化 detail が正であることが必要です。すべて reject の場合は Anchor へ戻り、「最も悪くない候補」を無条件採用しません。

round-trip edge は candidate→reference と reference→candidate の対称 displacement で、reference gradient 分布から共通閾値を作ります。edge precision / recall / F1 も記録するため、元の線を消した候補と余計な線を足した候補を別々に罰します。spectral flatness はゼロ残差を 0 とし、固定 epsilon によって「変更なし」を白色雑音扱いしません。

scalar round-trip confidence は全面へ一様適用せず、reference へ縮小した局所低周波誤差から作る local round-trip map を実際の周波数合成へ渡します。危険領域では MID / MID_LOW を強く抑え、HIGH も無制限には採用しません。debug の round-trip map は実際に合成へ使った map と同一です。

### Spatial Residual Rescue

`Spatial rescue after all whole-canvas candidates reject` は既定で有効です。全キャンバス候補が1枚でも通常のhard constraintを通過した場合は従来どおり全体winnerを使い、全候補がrejectされた場合だけ局所救済を試します。既存の全体閾値を緩める機能ではありません。

- 既定480pxの粗いdecision cellを、周辺contextと元解像度の対応cropを含めて既存のSSIM、低周波、色、edge displacement、duplicate/new edge、clipping、tile seamで再評価します。
- 各cellは候補1枚またはAnchorを離散選択します。複数候補の残差を平均せず、Anchorより既定0.05以上scoreが高い候補だけを採用します。
- 既定2cell未満の孤立した採用領域を除去し、label切替率が既定0.45を超える断片的な結果は全体を不採用にします。
- 候補labelが切り替わる境界は、既定48pxのsmoothstep collarで一度Anchorへ戻します。candidate同士を直接cross-fadeしないため、異なる線の位相を平均したghostを避けます。
- 局所合成後に既存の全キャンバスhard constraintをもう一度通します。失敗、NaN/Inf、過剰clipping、境界jump、局所採用なし、過剰な断片化はすべてAnchorへfail closedします。
- 候補は逐次処理し、現在のwinner画像、confidence、粗いlabel/score gridだけを保持します。候補数に比例して4K/8K RGB画像をRAMへ積みません。

設定値、cell label、採用率、除去cell数、fragmentation、境界jump、各候補の局所通過数、最終全体validationはPNG内manifestの各stage `selection_reports` に保存します。これは未知の真の細部を復元したという意味ではなく、Krea2が生成した局所残差のうち既存の決定論的gateを通った部分だけを採用した記録です。

## 顔検出と Manual ROI

実行時に detector、segmentation、IP-Adapter、InsightFace 等のモデルをダウンロードしません。

利用可能な provider は次です。

- `Manual ROI`: 常時利用可能。`Manual Face Core Mask` の白い連結成分を元入力座標の顔本体として扱います。
- `OpenCV Haar (photo only)`: OpenCV 同梱 XML またはユーザー指定ローカル XML が存在し、Content profile が明確に Photo の場合だけ利用します。
- `Auto (local only)`: Manual ROI を最優先し、明確な Photo の場合だけ Haar fallback を検討します。不確かな画像は Illustration / Anime 寄りへ倒します。

Haar はアニメ顔へ適しません。Illustration / Anime では、ユーザーが Haar を明示選択しても結果を無条件採用せず、Manual ROI を案内します。

`Manual Face Core Mask` には入力と同寸でなくてもよい RGBA 画像を指定できます。顔そのものを塗り、頭全体や広い背景は塗りません。`Manual mask channel` で Luminance または Alpha を選択します。context は自動拡張されます。複数人物は人物ごとに離れた連結成分を作ります。

component mask は bbox へ縮約せず、Face 候補の evaluation mask と write mask の基礎に使います。Haar のように mask を持たない検出結果には soft ellipse の face core を作ります。人物別 context crop は重なってよい一方、region id、元顔サイズ、candidate count、processing size は統合しません。近接する人物は正規化した顔中心距離で画素所有権を分け、ある人物の Face pass が別人物の Face Core を変更しないようにします。

Identity Reference は UI へ入力できますが、現在のローカル環境に正式な生成条件 provider がない場合は無言で無視せず `参照条件providerなし` をログと metadata に残します。外部モデルを自動取得しません。

## Manual protection / boost mask

- `Structure Protection Mask`
  - 白ほど元の輪郭・中低周波を強く保持します。
  - 自動 structure protection と `max()` で統合します。
- `Overdraw Boost Mask`
  - 白ほど採用可能な残差 gain を増やします。
  - `base_gain × (1 + boost_strength × mask)` です。

どちらも RGBA、異寸法、Alpha/Luminance 選択に対応します。標準 img2img inpaint mask (`p.image_mask`) とは混ぜません。

## Preset

### Structure Safe

- Anchor 0.12 / Global 0.24 / Face 0.22 / Hair 0.30 / Material 0.30 / Micro 0.14
- Structural Lock 0.90 / Low Frequency Lock 1.00 / Overdraw 0.75
- Global 1 / Face 4 / Hair 3 / Material 1 candidates
- Flat Region Detail 0.15

入力構造優先。初回 4K、人物が多い画像、比較基準に向きます。

### Overdraw（既定）

- Anchor 0.15 / Global 0.34 / Face 0.30 / Hair 0.40 / Material 0.42 / Micro 0.20
- Structural Lock 0.78 / Low Frequency Lock 0.96 / Overdraw 1.00
- Global 2 / Face 6 / Hair 4 / Material 2 candidates
- Flat Region Detail 0.35

構造と書き込み量の標準バランスです。

### Max Overdraw

- Anchor 0.18 / Global 0.42 / Face 0.36 / Hair 0.48 / Material 0.50 / Micro 0.24
- Structural Lock 0.68 / Low Frequency Lock 0.92 / Overdraw 1.30
- Global 2 / Face 8 / Hair 6 / Material 2 candidates
- Flat Region Detail 0.55

処理時間と創作量が最も大きい設定です。入力にない材質表現も増えます。

Preset を変更すると関連 slider を更新します。`Custom` は現在の slider 値を使います。

## 4K 推奨設定

最初は次を推奨します。

- Target: `4K long edge`
- Preset: `Structure Safe`
- Content: 明示できるなら `Illustration / Anime`、`Photo`、`3D / Render`
- Exact Steps: 6
- Tile: 1280 / 960 / 160 / 768
- Accumulator: Auto
- Global candidates: 1
- Face ROI: Manual mask が用意できる場合だけ有効
- Hair / Material / Micro: 最初の構造確認では必要に応じて無効
- Debug: 最初は無効

構図、顔、髪型、tile boundary を確認後、`Overdraw` と候補数を上げて比較します。

## 8K 推奨設定

8K は、合格した 4K から x2 する方が処理時間と候補確認を管理しやすくなります。

- Target: `8K long edge` または 4K 入力の `x2`
- Preset: `Structure Safe` から開始
- Accumulator: `Disk-backed memmap`
- Save all candidates: オフ
- Save debug images/maps: 通常はオフ
- ROI stages: Final stage only または Last two stages
- Exact Steps: 4～6 で比較
- 一時ディスクの十分な空きを確保

元の約2K画像から8Kへ直接進める場合は複数 stage となり、最終 stage の全体 pass だけでも多数の 1280 tile を処理します。

## RTX 3090 24GB 向け

- tile input 1280 を基準にします。
- OOM 時は一度だけ 1024 / 768 / 128 / 640 へ下げて、同じ concrete seed でジョブ全体を再試行できます。変更値は metadata とログへ残ります。
- GPU には現在 tile、latent、モデル、小評価 tensor だけを置きます。
- 4K/8K canvas、candidate、confidence、accumulator は CPU または memmap です。
- `torch_gc()` は tile ごとではなく pass 境界と OOM 回復時に実行します。
- 8K、全 debug、全 candidate 保存の同時使用は避けてください。

`Accumulator mode=Auto` は最大辺8192または設定 RAM 上限を超える見込みで memmap を選びます。開始前に RAM と disk を見積もり、不足時は生成前に失敗します。

RTX 3090 / Krea2 int8 / Qwen3VL fp8 の実測スモークでは、362×512→724×1024、1280 tile、Anchor 2 tile + Global 2 tile + Face ROI 1回、Exact Steps 6で最大約23047 MiBを使用しました。モデル常駐後の処理時間は約57秒でした。cold start の checkpoint / text encoder 転送時間は別途必要で、ディスク、offload、常駐状態によって大きく変わります。この値は4K/8K所要時間の推定値ではありません。

旧1.1系の同じ構成のフル4K実測では、1664×2353→2897×4096（内部2904×4096）、1 stage、24 tiles/pass、Anchor + Global + 当時の統合Face ROIで49内部呼び出し、Structure Safe / Exact Steps 1で完走しました。HyperWeave処理時間は約662秒、cold API test全体は約700秒、PyTorch peak allocatedは約19983 MiB、peak reservedは約20714 MiB、CPU working RAM見積もりは約771 MiBでした。1.2.0 は人物別 ROI と pass 順が異なるため、この呼出数を現行性能値として扱わないでください。

## RGBA

RGB と RGBA を受け付けます。RGBA は次のように扱います。

- RGB と alpha を分離
- linear RGB で premultiply して resize
- model へ渡す透明領域の背景は周辺色推定（または White / Black）
- 最終 alpha は入力由来を高品質拡大
- model に alpha を生成・変更させない
- back projection は alpha を変更しない
- hidden RGB を透明境界へにじませない

完全不透明な RGBA 画像では RGBA 経路を通りますが、半透明境界の受入確認は合成 alpha unit test も参照してください。

## Debug 出力

`Save debug images` を有効にすると、ジョブ専用 temp directory に一旦 staging し、正常終了時だけ指定出力先へコピーします。候補、map、ROI、metrics の追加保存は個別 checkbox で制御します。

主なファイル:

- `_hw_stage01_base.png`
- `_hw_stage01_anchor.png`
- `_hw_stage01_global_candidate00.png`
- `_hw_stage01_global_selected.png`
- `_hw_stage01_structure_protect.png`
- `_hw_stage01_orientation_confidence.png`
- `_hw_stage01_new_edge_confidence.png`
- `_hw_stage01_tile_confidence.png`
- `_hw_stage01_roundtrip_confidence.png`
- `_hw_stage01_material_roundtrip_confidence.png`
- `_hw_stage01_micro_roundtrip_confidence.png`
- `_hw_stage01_frequency_high.png`
- `_hw_stage01_frequency_mid.png`
- `_hw_stage01_frequency_midlow.png`
- `_hw_stage01_seam_map.png`
- `_hw_stage01_detail_potential.png`
- `_hw_stage01_composed.png`
- `_hw_stage01_before_backprojection.png`
- `_hw_stage01_final.png`
- `_hw_face_000_anchor.png`
- `_hw_face_000_candidate00.png`
- `_hw_face_000_selected.png`
- `_hw_face_000_mask.png`
- `_hw_face_000_roundtrip_confidence.png`
- `_hw_face_000_score.json`
- `_hw_metrics.json`
- `_hw_candidates.csv`
- `_hw_settings.json`

通常終了、例外、interrupt のいずれでも memmap を明示 close した後に temp directory を削除します。debug 無効時に候補画像を恒久保存しません。

## PNG metadata

標準 `parameters` infotext に次を含めます。

- HyperWeave version / preset / target / stage plan / concrete seed
- model / VAE / sampler / scheduler / CFG / Exact Steps
- strengths / candidate counts / tile settings
- detector / structure conditioner / frequency gains
- back projection / debug / processing time / peak PyTorch VRAM / RAM・disk estimate / memmap

さらに `hyperweave` PNG text chunk に再現用 JSON manifest を保存します。内部 tile の size、seed、suffix ではなく、最終 target、元 prompt、concrete seed で infotext を作り直します。

ProofWeave 1.2.0 の manifest には、`scoring_version`、`candidate_score_margin`、local round-trip gate、semantic face ownership、face protection、symmetric edge metrics の有効状態、意味的 pass 順、人物別 `final_face_metrics` を追加します。各 stage の最終 face metrics は Material / Micro / Hair / Face / seam / back projection / exact resize 後の実際に返す 8-bit 出力 crop から計算し、round-trip SSIM、対称 edge displacement、edge precision / recall / F1、低周波 MSE、color drift を記録します。旧 candidate 選択時の score と既存 key は残します。

## 比較ツール

`tools/compare_hyperweave.py` は同じ入力に対する Lanczos と任意候補を比較します。

```powershell
.\venv\Scripts\python.exe .\tools\compare_hyperweave.py `
  --source H:\dl\image-cropped.png `
  --candidate "HyperWeave=H:\path\result.png" `
  --crop "left_face=280,500,760,980" `
  --crop "right_face=900,560,1420,1080" `
  --output H:\path\comparison
```

出力:

- contact sheet
- 同一座標 crop strip
- absolute difference
- HIGH / MID / MID_LOW frequency map
- structure map
- source round-trip confidence map
- HyperWeave の最終 stage tile 境界に沿った seam map
- round-trip SSIM / PSNR
- low-frequency error / color drift / symmetric edge displacement
- edge precision / recall / F1
- MID / MID_HIGH / HIGH energy
- coherent-line / face structure / hair-flow / noise penalty / seam ratio
- manifest が持つ processing time / peak VRAM / RAM・disk estimate / memmap

HyperWeave 1.2.0 manifest に最終 face metrics があれば、その最終出力 SSIM を `face_structure_score` の主値にし、選択時の Face candidate SSIM は `selected_face_candidate_roundtrip_ssim` として残します。manifest に最終値がない旧出力だけ candidate 選択時の値へ fallback します。`face` または `head` を名前に含む crop が与えられた場合、manifest 最終値がない方式について全方式を同じ領域で縮小比較します。HyperWeave manifest を持たない任意候補では、処理時間・VRAM・RAM・disk・tile seam ratio は推測せず `null` にします。

単純な「高周波量が多いほど高品質」という判定にはしません。

## テスト

CPU unit / stub integration:

```powershell
.\venv\Scripts\python.exe -m compileall extensions-builtin\hyperweave
.\venv\Scripts\ruff.exe check extensions-builtin\hyperweave `
  tools\compare_hyperweave.py tools\tests\test_hyperweave_*.py
.\venv\Scripts\python.exe -m unittest discover -s tools\tests `
  -p "test_hyperweave_*.py" -v
```

通常のテストは GPU、checkpoint、ネットワーク、外部 detector download を要求しません。StubGenerator は coherent detail、位置ずれ、random noise、髪方向と直交する線を再現します。4D/5D latent、RAM/memmap一致、interrupt cleanup、RGBA、候補reject、seam、back projection、比較artifactも含みます。

実モデル test は明示 opt-in の場合だけ実行します。

```powershell
$env:HYPERWEAVE_RUN_GPU_TESTS='1'
$env:HYPERWEAVE_LIVE_API='http://127.0.0.1:7861'
$env:HYPERWEAVE_TEST_IMAGE='H:\dl\image-cropped.png'
$env:HYPERWEAVE_LIVE_STEPS='6'
.\venv\Scripts\python.exe -m unittest -v `
  tools.tests.test_hyperweave_live_api
```

live test は、無効時の通常 img2img 委譲と、有効時の Anchor + Global + Manual Face Core、正確な出力寸法、PNG text chunk、保存後のmetadata再読込、一時領域cleanupを確認します。

元画像を縮小せず4K long edgeで実行する場合:

```powershell
$env:HYPERWEAVE_RUN_GPU_TESTS='1'
$env:HYPERWEAVE_TEST_MAX_EDGE='0'
$env:HYPERWEAVE_LIVE_TARGET='4K long edge'
$env:HYPERWEAVE_TEST_FACE_ROIS='both'
$env:HYPERWEAVE_LIVE_STEPS='1'
.\venv\Scripts\python.exe -m unittest -v `
  tools.tests.test_hyperweave_live_api.HyperWeaveLiveApiTests.test_supplied_image_anchor_global_and_manual_face
```

`HYPERWEAVE_TEST_MAX_EDGE=0` は入力を原寸で使います。`HYPERWEAVE_TEST_FACE_ROIS=both` は同梱テスト画像の左右キャラクターへ人物別の Manual Face Core を作ります。拡張 context が重なっても人物別 ROI のまま処理します。

## 既知の制限

- 元画像にない細部は生成モデルによる推定です。
- 極小顔の真の瞳、まつ毛、口を復元するものではありません。
- 同一人物性は入力情報、manual ROI、利用モデル、利用可能な reference provider に依存します。
- Anime face detector がない環境では Manual ROI が必要です。
- OpenCV Haar は Photo fallback 限定です。
- Identity Reference は正式なローカル provider がなければ生成条件へ適用されません。
- 現在の Coordinate Noise 保証は初期 latent noise です。SDE/Brownian sampler 内の secondary noise は sampler 実装に従います。
- Qwen Image形式の5D latentは静止画の時間長1だけに対応し、video latentは対象外です。
- Max Overdraw と多数候補は処理時間が長くなります。
- 8K と全 debug は大量の一時 disk と CPU RAM を使います。
- 強い Overdraw は入力にない材質表現を追加します。
- hard constraints を通過しても美的正しさを完全保証しません。
- HyperWeave は未知の真の高解像度情報の復元を保証しません。最終 face metrics も入力整合性の評価であり、真の正解画像との比較ではありません。
- Spatial Residual Rescueの局所gateを通過しても、複数領域を跨ぐ髪流れ、布の位相、照明、物体解釈の意味的一貫性は完全保証しません。孤立領域除去、境界Anchor collar、最終全体gateで抑制しますが、実画像A/B確認が必要です。
- CFG 1.0 のモデルでは negative prompt が実質使われない場合があります。重要な構造禁止は positive suffix にも含めています。

## 将来候補（1.2.0 では未実装）

- Global Brownian Atlas / Space-Time Coordinate Noise
- Operator Jackknife
- Causal Evidence Probe
- Conformal Risk Control

これらの sampler 変更、統計的 risk control、外部評価モデル、再学習、新規 checkpoint、自動 download は ProofWeave 1.2.0 に placeholder も含めて実装していません。

## トラブルシューティング

### No face ROI detected

Illustration / Anime では正常な fail-safe です。`Manual Face Core Mask` を指定します。

### target must be larger than input

HyperWeave はアップスケール専用です。入力より大きい target を指定します。

### tile geometry validation error

`tile_input = core + 2 × context`、`stride <= core`、各値が latent alignment 8 の倍数である必要があります。

### all candidates rejected

Anchor へ安全に戻っています。candidate score JSON の round-trip、color drift、edge displacement、seam、duplicate edge を確認します。必要なら Overdraw strength を下げるか tolerance を少し緩めます。

### OOM

自動再試行を有効にすると一度だけ 1024 tile へ下げます。それでも失敗する場合は、他の GPU 処理を終え、tile を明示的に小さくし、候補数を減らします。

### temp cleanup failure on Windows

HyperWeave は memmap を flush/close してから directory を削除します。残る場合は antivirus や画像 viewer が debug/temp file を開いていないか確認してください。OS終了、強制kill、電源断では Python の cleanup が走らないため、古い `hyperweave_*` directory が残ることがあります。内容と実行中ジョブがないことを確認してから手動で扱ってください。元画像は変更しません。
