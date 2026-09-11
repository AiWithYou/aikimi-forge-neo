# MiniMax H3：CLIP条件のキャッシュ

H3 Studioの「高速化 → 5. CLIP条件キャッシュ」で、同じプロンプト・参照画像から作った条件を再利用できます。SeedやStepsを変えて試すときに、Qwen3-VLの読み込みと再計算を省くための機能です。初期設定はオフです。

## 導入

Forgeの仮想環境から、H3で使用するComfyUIのフォルダーを指定してください。

```powershell
.\venv\Scripts\python.exe -X utf8 tools\install_minimax_h3_clipcache.py --runtime-root "H:\path\to\ComfyUI"
```

導入するのは[Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/tree/80ef7eb3b01565ff4190b519ce0db2f42b5e14e2)の固定版です。各ファイルをマニフェストのGit blobハッシュと照合し、MIT Licenseも配置します。既存ファイルが固定版と違う場合は上書きせず停止するため、独自に編集した内容を先に確認してください。

画面でキャッシュを選び、「実行環境とモデル → 選択設定で再起動」を押してください。管理起動では、この拡張だけを追加の許可リストへ入れます。使用するFast VAEなどの既存の選択も反映します。

## 選択と動作

| 選択 | 動作 |
| --- | --- |
| オフ | 従来のCLIPLoaderとH3 conditioningノードで処理。キャッシュを使わない。 |
| 自動 | 保存済みの同じ条件を読み込む。未保存の条件はエンコードして新しく保存。 |
| 再計算して更新 | 毎回エンコードを実行し、その条件のキャッシュを更新。再利用に戻すときは自動を選ぶ。 |

テキスト生成・開始／終了キーフレーム・参照素材の各モードで使えます。参照の上限は既存の画像9枚・動画3本・音声3本で、素材の順序と動画に付属する音声を引き継ぐ構成です。設定JSONと履歴には選択したキャッシュモードを保存します。

NegPiPと同時に選ぶと、同梱の専用ノードを使用します。初回は内部で読み込んだエンコーダーにNegPiPを適用し、条件テンソルと単語の重み・適用時間・追加トークン情報を保存します。再利用時も生成モデル側のNegPiP処理は有効です。切り替えでプロンプトやサンプラーを自動変更することはありません。

専用ノードは管理起動時に選択先のComfyUIへ配置します。CLIPCachedとNegPiPの配布コードは変更しません。外部でComfyUIを起動している場合は一度停止し、Forgeの「選択設定で再起動」を使用してください。

## 保存データと確認方法

保存先は、選択したComfyUIの次のフォルダーです。

```text
custom_nodes/ComfyUI-MiniMaxH3-CLIPCached/cache/
```

NegPiP併用時は、次の専用フォルダーに分けて保存します。通常のCache Managerには表示されません。

```text
custom_nodes/Aikimi-H3-NegPiP-Cache/cache/
```

NegPiP用の保存対象は、条件テンソルと重み・適用時間などです。サムネイルや通常版の閲覧用メタデータは作成しません。プロンプト・エンコーダー入力・NegPiPの全設定に加え、NegPiPと専用ノードのコード、エンコーダー実装の識別子を照合して再利用を判定します。NegPiP設定の変更は再計算の対象です。SeedやStepsだけの変更では、条件が同じなら再利用できます。

条件テンソルのほか、プロンプトや参照由来のメタデータ・サムネイルも保存されます。自動整理はないため、条件を多く試すほどディスク使用量が増えます。管理機能は配布元の[Cache Managerガイド](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/blob/80ef7eb3b01565ff4190b519ce0db2f42b5e14e2/docs/CACHE_MANAGER.md)を参照してください。

同じキャッシュフォルダーは、1つのComfyUIプロセスで使用してください。エンコーダー実装や依存パッケージを更新した場合は、古いキャッシュを整理して再計算する必要があります。配布元の識別方法は[技術資料](https://github.com/Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached/blob/80ef7eb3b01565ff4190b519ce0db2f42b5e14e2/docs/TECHNICAL_DETAILS.md)に記載されています。

H3のVAE処理、参照の前処理、サンプリング、動画保存は通常どおり実行します。条件を再利用しても、これらの時間は必要です。初回保存・再利用・通常処理を比較する際は、プロンプト、素材、Seed、解像度、長さ、Stepsを揃えてください。

設定JSONの`acceleration.clip_cache`は要求した動作です。実際のHIT・MISSはComfyUIログで確認してください。再利用の実測では新しいComfyUIプロセスを使用し、メモリ上の実行結果が残っているだけの状態と区別します。

## RTX 3090での実測（2026-09-08）

開始画像1枚、384×512、124フレーム、20 Steps、固定Seedで、各回ComfyUIを起動し直して比較しました。

| 処理 | 生成所要時間 |
| --- | ---: |
| 通常処理 | 1,126.387秒 |
| 再計算して保存 | 1,065.418秒 |
| 自動・キャッシュ再利用 | 750.318秒 |

この試行では通常処理から約33%短縮しました。再利用ログには`CACHE HIT`が記録され、Qwen3-VLのロード要求はありません。MP4をデコードして比較した全124フレームのRGB値と音声データは、3本とも同一でした。

各条件1回の測定で、OSのファイルキャッシュなどによるモデル読み込み時間の差も含みます。すべての条件で同じ短縮率や出力の同一性を保証する測定ではありません。参照素材モードは接続テストで確認しており、今回のGPU比較は開始キーフレーム方式です。比較記録は`outputs/minimax-clipcache-audit/benchmark.json`に保存しています。

### NegPiP併用の比較

同じ映像条件に`(camera shake:-0.7@v0-2)`を加え、ComfyUI 0.31.0・Comfy Kitchen 0.2.30で比較しました。各回H3プロセスを起動し直しています。

| 処理 | 生成所要時間 |
| --- | ---: |
| NegPiP単独 | 1,125.416秒 |
| NegPiP＋キャッシュ初回保存 | 1,080.486秒 |
| NegPiP＋キャッシュ再利用 | 725.291秒 |

3本とも全124フレームのRGB値と音声データが同一でした。再利用ログでは`CACHE HIT`とNegPiPの全50ブロックへの適用を確認し、Qwenのロード要求はありません。保存データにも負の重み2トークン、映像の0〜2秒という適用範囲、追加トークン数が残っています。記録は`outputs/minimax-negpip-cache-audit/benchmark.json`と`metadata-check.json`です。

Fun ControlNetの導入に伴うComfyUI 0.34.0への更新後は、エンコーダーの識別子が変わるため、以前のキャッシュは初回に再計算します。[Fun ControlNetの説明](minimax-h3-fun-control.md)も参照してください。
