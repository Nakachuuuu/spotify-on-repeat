# Spotify トップ曲を Discord ウィジェットに表示する手順書

全体の流れ:
Discord側の準備 → ウィジェット作成 → Spotify側の準備 → Identity発行 → スクリプトでデータ更新 → プロフィールに表示

必要なもの: Discordアカウント、Spotifyアカウント(Developer ModeにはPremiumが必要)、Python 3.10以降

---

## フェーズ1: Discord アプリケーションの作成

1. https://discord.com/developers/applications を開き「New Application」で新規作成。
   名前はウィジェットの左上に表示されます(例: `Spotify Stats`)。
2. General Information で App Icon を設定(Spotifyロゴ風は避け、自作アイコン推奨)。
3. サイドバー **OAuth2** → 「Add Redirect」で `https://discord.com` を追加して保存。
4. サイドバー **Games → Social SDK** のフォームを記入して送信
   (会社情報は適当でOK。*付きの必須項目だけ埋めて「I consent」にチェック)。
5. ブラウザの開発者ツール(F12)のConsoleタブで、記事に載っている
   ApexExperimentStore のオーバーライドスニペットを実行し、
   `2026-03-widget-config-editor` を有効化。
   実行後、ページを**リロードせず**戻る→アプリページを開き直すと
   Games セクションに **Widget** ページが出現します。

> スニペットの実行は自己責任です。内容が理解できるもの(記事掲載のもの)だけを使い、
> 出所不明のコードは絶対に貼らないでください。

## フェーズ2: ウィジェットのデザイン作成

Widget ページ → Create Widget でエディタを開き、以下を作成します。
**Widget Top / Widget Bottom / Add Widget Preview の3つは必須**です。

このプロジェクトでは、後述のスクリプトが送るデータのキー名と
エディタで設定する Data Field 名を一致させる必要があります。
以下のキー名をそのまま使ってください:

| 用途 | Value Type | Data Field(キー名) |
|---|---|---|
| ヘッダー画像 | User Data (Image) | `top_image` |
| ヘッダーのタイトル | User Data (Text) | `top_title` |
| ヘッダーのサブタイトル | User Data (Text) | `top_subtitle1` |
| 1〜6位の曲名 | User Data (Text) | `value1`〜`value6` |
| 1〜6位のアーティスト | User Data (Text) | `label1`〜`label6` |
| 1〜6位のアルバムアート | User Data (Image) | `icon1`〜`icon6` |

レイアウト例:
- **Widget Top**: タイトルを `top_title`、サブタイトルを `top_subtitle1`、画像を `top_image`
- **Widget Bottom**: 6統計グリッドを選び、各順位の曲名を `value1`〜`value6`、
  アーティストを `label1`〜`label6`、画像を `icon1`〜`icon6` に設定

各 User Data フィールドには **fallback**(データ未送信時の表示)を
Custom String で設定しておくと安全です(例: `Loading...`)。

設定が済んだら:
1. Sample Data タブでデモデータを入れて見た目を確認
2. 「Generate Json」で JSON を保存(後で使います)
3. 「Save Changes」→「**Publish**」

## フェーズ3: OAuth 認可(自分のアカウントに許可を出す)

1. Developer Portal の OAuth2 ページ → OAuth2 URL Generator で
   スコープ **`openid`** と **`sdk.social_layer`** にチェック。
2. Redirect URI に `https://discord.com` を選択して URL をコピー。
3. URL 内の `response_type=code` を **`response_type=token`** に書き換えてブラウザで開き、認可。
4. リダイレクト後の URL に含まれる `access_token=...` は今回の構成では保存不要ですが、
   エラー(invalid scopes)が出る場合は Social SDK フォームが未提出です。
5. Discord の設定 → 認証済みアプリ に自分のアプリが表示されていればOK。

## フェーズ4: Bot トークンの取得

Developer Portal → **Bot** ページ → Reset Token でトークンをコピー。

**このトークンは絶対に誰にも共有しないこと。** Botアカウントの全権限を奪われます。
GitHub等に上げる場合も必ず環境変数や .gitignore 済みファイルで管理してください。

## フェーズ5: Spotify アプリの登録

1. https://developer.spotify.com/dashboard でアプリを作成。
2. Redirect URI に `http://127.0.0.1:8888/callback` を追加して保存。
3. Client ID と Client Secret を控える。

## フェーズ6: スクリプトの設定と実行

```bash
pip install -r requirements.txt
```

`config.json` を作成(config.example.json をコピーして値を埋める):

```json
{
  "spotify_client_id": "...",
  "spotify_client_secret": "...",
  "spotify_refresh_token": "",
  "discord_app_id": "アプリのApplication ID",
  "discord_user_id": "自分のユーザーID(Discordで自分を右クリック→IDをコピー)",
  "discord_bot_token": "...",
  "time_range": "short_term",
  "track_count": 6,
  "widget_title": "Spotify On Repeat",
  "widget_subtitle": "直近4週間で最も聴いた曲 Top6",
  "top_image_url": ""
}
```

`top_image_url` に公開画像URLを指定すると、1位のアルバムアートより優先して `top_image` に送信します。
空文字の場合は従来どおり1位のアルバムアートを使います。
GIFのアニメーション可否はDiscordクライアント側の描画仕様に依存します。

1. **Spotifyのリフレッシュトークン取得(初回のみ)**
   ```bash
   python get_spotify_token.py
   ```
   ブラウザが開くので Spotify にログインして許可。
   表示されたリフレッシュトークンを config.json に貼り付け。

2. **ウィジェットのデータ更新**
   ```bash
   python refresh_widget.py
   ```
   初回の成功で Identity の発行も兼ねます(記事のPowerShell/curl手順の代わり)。
   `OK: widget update accepted` と出ればDiscordに更新が受理されています。

3. **定期実行(任意)**
   - Windows: タスクスケジューラで1日1回 `refresh_widget.py` を実行
   - Mac/Linux: cron 例 → `0 9 * * * cd /path/to/project && python3 refresh_widget.py`

## フェーズ7: GPT ImageでジャケットをLive2D風GIFにする

GitHub Actionsでは、Spotifyの実際の1位ジャケットを入力画像として使用します。
`gpt-image-2`には、人物のまばたき・髪・服・腕などが少し動いた
「別の1キーフレーム」だけを作らせます。その差分からOpenCVで局所的な
動きベクトルを推定し、元ジャケットの画素を変形して往復ループGIFにします。
パン・ズームで画像全体を動かす方式ではありません。

これはPSDレイヤーと手作業のリグを使う本物のLive2Dではなく、1枚絵からの
自動推定です。ジャケットの構図によって動きの精度は変わりますが、
生成キーフレームを直接表示せず元画像の画素を動かすため、文字・背景・絵柄の
変化を抑えられます。
全体の平行移動・拡大縮小・回転や、大半の描き直しを検出した場合は採用せず、
静止ジャケットへフォールバックします。

自動更新は次の順序で行われます。

1. Spotifyから現在の1位と正方形ジャケットを取得
2. 同じ曲・ジャケットのGIFがActionsキャッシュにあれば再利用
3. 未生成ならGPT Imageで小さな人物動作のキーフレームを1枚生成
4. 光学フローから元ジャケットを変形し、2 MiB以下のループGIFへ変換
5. Actions実行ごとの固有ファイル名でGitHub Pagesへデプロイ
6. 公開GIFがHTTP 200・`image/gif`になるまで確認してからDiscordを更新

初回だけ、GitHubリポジトリで次を設定してください。

1. **Settings → Secrets and variables → Actions** のRepository secretsに以下を登録
   - `SPOTIFY_CLIENT_ID`
   - `SPOTIFY_CLIENT_SECRET`
   - `SPOTIFY_REFRESH_TOKEN`
   - `DISCORD_APP_ID`
   - `DISCORD_USER_ID`
   - `DISCORD_BOT_TOKEN`
   - `OPENAI_API_KEY`
2. **Settings → Pages → Build and deployment → Source** を **GitHub Actions** に設定
3. **Actions → Refresh Spotify widget → Run workflow** から手動実行
生成関連ファイルを `main` にpushした場合も、自動で同じワークフローが実行されます。

正常時は `build-animation` → `deploy-animation` → `refresh-widget` の順に成功します。
キーフレーム生成、動き抽出、またはPages公開に失敗した場合でも
`refresh-widget`は実行され、`top_image`には1位の静止ジャケットが使われます。

モデルと品質は `.github/workflows/refresh.yml` の
`OPENAI_IMAGE_MODEL` / `OPENAI_IMAGE_QUALITY` で変更できます。
既定品質は`medium`です。曲・ジャケット・モデル・品質・生成バージョンが
同じ間はキャッシュを使うため、通常は毎日OpenAI APIを呼びません。
GIFと曲名・アーティストはGitHub Pages上で公開されます。

## フェーズ8: プロフィールに表示

この操作はまだ通常UIにないため、Discord Previews サーバーの該当スレッドにある
スニペットをブラウザ/クライアントの開発者ツールで実行して、
ウィジェットをプロフィールに追加します。
また、アカウントの実験フラグ `2026-03-application-widget-v2-renderer` を
**Variant 1** に設定する必要があります(不明点は同サーバーの general で質問可)。

2026年6月4日以降の制限により、**このウィジェットを追加できるのは
アプリの所有者(=あなた)だけ**です。表示自体は他の人からも見えます。

## トラブルシューティング

- 「Your game stats are still syncing. Keep playing!」のまま
  → フェーズ3のOAuth認可が済んでいないか、Identity発行(初回PATCH)が失敗しています。
- PATCH が 401 → Botトークンが間違っているか失効(Resetし直して更新)。
- PATCH が 403/404 → Application ID / User ID の取り違えが多いです。
- Spotify が 401 → リフレッシュトークンを取り直してください。
- ウィジェットに反映されない → エディタの Data Field 名とスクリプトの
  キー名(`top_image`、`value1` 等)が完全一致しているか、Publish 済みかを確認。
