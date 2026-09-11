# セキュリティポリシー

## 対応版

Aikimi Studio Neoは、default branchである`neo`の最新commitだけをセキュリティ修正の対象とします。リポジトリに残るupstream由来のtagは、Aikimi Studio Neoの保守版を示すものではありません。

## 脆弱性の報告

認証情報、個人パス、未公開モデル、生成物、sysinfo、再現用データを公開Issueや公開コメントへ投稿しないでください。GitHub Private Vulnerability Reportingが有効な場合は、次の非公開フォームを使います。

<https://github.com/AiWithYou/aikimi-studio-neo/security/advisories/new>

フォームを開けない場合は、秘密を公開せず、リポジトリ所有者と別途合意した非公開経路を使ってください。このリポジトリには、監査時点で公開済みの専用セキュリティメールアドレスがありません。リリース担当者は、第三者へ配布する前にGitHub Private Vulnerability Reportingを有効にします。

報告には、可能な範囲で次を含めてください。

- Aikimi Studio Neoのcommit SHA
- WindowsとPythonの版
- 問題が発生する最小手順
- 想定した安全境界と実際の挙動
- 秘密値を削除したログまたはリクエスト例
- 公開前に連絡してほしい期限や条件

## sysinfoとログ

Aikimi Studio Neoは、認証情報、URL userinfo、tokenに見える値、機密性が高いパスを共通処理でマスクします。ただし、自動マスクだけで安全を保証できません。sysinfoやログを共有する前に、次を目視で確認してください。

- `Authorization`と`Cookie`
- API、Gradio、ngrok、Hugging Faceの認証情報
- URLのquery stringとuserinfo
- `COMMANDLINE_ARGS`、`INDEX_URL`、`TORCH_INDEX_URL`
- ユーザー名を含む絶対パス
- prompt、入力ファイル名、生成物の個人情報

## 外部公開

通常起動は`127.0.0.1`だけを使います。LAN、share、ngrok、loopback以外のserver nameは既定で無効です。外部公開には`--aikimi-remote`と明示的な公開指定が必要で、WebUIとAPIを有効にする場合は両方の認証も必要です。

外部公開は、TLS終端、firewall、利用者管理、更新手順まで用意できる環境だけで使ってください。Aikimi Studio NeoのBasic認証だけを、インターネットへ直接公開するための完全な境界とは扱わないでください。詳しい条件は[docs/security-model.md](docs/security-model.md)を参照してください。

## 対象外のデータ

モデルweight、dataset、checkpoint、生成画像、動画、音声、ローカルの認証ファイルは、このGitリポジトリへ報告用として追加しないでください。必要な再現情報は、権利と機密性を確認した最小fixtureへ置き換えてください。
