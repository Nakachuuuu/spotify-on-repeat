#!/usr/bin/env python3
"""
Spotify のトップ曲を取得して Discord ウィジェットの identity を更新する。

使い方:
    python refresh_widget.py

初回実行時に identity の発行も兼ねる。成功すると "OK: widget updated" と出る。
config.json の time_range で集計期間を切り替え可能:
    short_term  = 約4週間
    medium_term = 約6ヶ月
    long_term   = 約1年
"""

import base64
import json
import os
import sys
from pathlib import Path

import requests

# Windows のコンソール(cp932 等)でも日本語や「—」を出力できるようにする。
# 表示だけの対策で、Discord へ送るデータは常に UTF-8 のまま。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG_PATH = Path(__file__).with_name("config.json")

SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_TOP_URL = "https://api.spotify.com/v1/me/top/tracks"
# identityId は 0 固定で問題ない(記事の推奨どおり)
DISCORD_IDENTITY_URL = (
    "https://discord.com/api/v9/applications/{app_id}"
    "/users/{user_id}/identities/0/profile"
)
# User-Agent はブラウザ以外から必須。任意の値でよい。
USER_AGENT = "SpotifyWidget (https://github.com/yourname/spotify-widget, 1.0.0)"

# ヘッダーのサブタイトル既定値(config に widget_subtitle が無いとき使用)。
PERIOD_LABELS = {
    "short_term": "Last 4 weeks",
    "medium_term": "Last 6 months",
    "long_term": "All time",
}


# 設定キー -> 対応する環境変数名。GitHub Actions 等では config.json を置かず、
# これらの環境変数(Secrets)で値を渡す。
ENV_KEYS = {
    "spotify_client_id": "SPOTIFY_CLIENT_ID",
    "spotify_client_secret": "SPOTIFY_CLIENT_SECRET",
    "spotify_refresh_token": "SPOTIFY_REFRESH_TOKEN",
    "discord_app_id": "DISCORD_APP_ID",
    "discord_user_id": "DISCORD_USER_ID",
    "discord_bot_token": "DISCORD_BOT_TOKEN",
    "time_range": "TIME_RANGE",
    "track_count": "TRACK_COUNT",
    "widget_title": "WIDGET_TITLE",
    "widget_subtitle": "WIDGET_SUBTITLE",
}


def load_config():
    # ローカルでは config.json を読む。CI では config.json が無くても、
    # 環境変数だけで動く。環境変数がセットされていればファイルより優先。
    cfg = {}
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for key, env in ENV_KEYS.items():
        val = os.environ.get(env)
        if val is not None and val != "":
            cfg[key] = val

    required = [
        "spotify_client_id",
        "spotify_client_secret",
        "spotify_refresh_token",
        "discord_app_id",
        "discord_user_id",
        "discord_bot_token",
    ]
    missing = [k for k in required if not str(cfg.get(k, "")).strip()]
    if missing:
        sys.exit(
            "設定が不足しています: " + ", ".join(missing)
            + "\n  config.json か環境変数("
            + ", ".join(ENV_KEYS[k] for k in missing)
            + ")で指定してください。"
        )
    cfg.setdefault("time_range", "short_term")

    # 日本語表記オーバーライド表。config.json に無ければ overrides.json
    # (コミット対象・非機密)を読む。環境変数 NAME_OVERRIDES(JSON)が最優先。
    if "name_overrides" not in cfg:
        ov_path = CONFIG_PATH.with_name("overrides.json")
        if ov_path.exists():
            data = json.loads(ov_path.read_text(encoding="utf-8"))
            cfg["name_overrides"] = {
                k: v for k, v in data.items() if not k.startswith("_")
            }
    env_ov = os.environ.get("NAME_OVERRIDES")
    if env_ov:
        try:
            cfg["name_overrides"] = json.loads(env_ov)
        except Exception:
            pass

    return cfg


def get_spotify_access_token(cfg):
    """リフレッシュトークンから短命のアクセストークンを取得する。"""
    basic = base64.b64encode(
        f"{cfg['spotify_client_id']}:{cfg['spotify_client_secret']}".encode()
    ).decode()
    resp = requests.post(
        SPOTIFY_TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "refresh_token",
            "refresh_token": cfg["spotify_refresh_token"],
        },
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(
            f"Spotify トークン更新に失敗 ({resp.status_code}): {resp.text}\n"
            "リフレッシュトークンが失効している場合は get_spotify_token.py を再実行してください。"
        )
    return resp.json()["access_token"]


def _pick_square(images):
    """アルバム画像から正方形(1:1)のURLを返す。無ければ None。"""
    for im in images:
        w, h = im.get("width"), im.get("height")
        if w and h and w == h:
            return im["url"]
    return None


def _ov(s, overrides):
    """name_overrides に完全一致すれば置換。英語登録のみの曲/アーティストを
    日本語表記にするための対応表(例: "The Ocean Waves" -> "海がきこえる")。"""
    return overrides.get(s, s) if overrides else s


def get_top_tracks(access_token, time_range, limit=6, overrides=None):
    overrides = overrides or {}
    # 同じ曲が別リリースで重複することがあるため、多めに取得してから
    # 重複除去して上位 limit 件に絞る(Spotify の上限は50件)。
    fetch = min(50, max(limit * 4, limit + 10))
    resp = requests.get(
        SPOTIFY_TOP_URL,
        headers={"Authorization": f"Bearer {access_token}"},
        params={"time_range": time_range, "limit": fetch},
        timeout=30,
    )
    if resp.status_code != 200:
        sys.exit(f"トップ曲の取得に失敗 ({resp.status_code}): {resp.text}")

    items = resp.json().get("items", [])
    unique = []
    seen = {}  # (曲名, アーティスト) -> unique 内のインデックス
    for it in items:
        # 曲名・各アーティスト名に日本語オーバーライドを適用。
        artists = ", ".join(
            _ov(a["name"], overrides) for a in it.get("artists", [])
        )
        name = _ov(it.get("name", ""), overrides)
        images = it.get("album", {}).get("images", [])
        largest = images[0]["url"] if images else None
        square = _pick_square(images)
        key = (name, artists)
        if key in seen:
            # 既出の曲。保持側のジャケットが正方形でなく、
            # この重複側に正方形があれば拝借する(例: 16:9のMVアート→通常ジャケット)。
            kept = unique[seen[key]]
            if not kept["square"] and square:
                kept["square"] = square
            continue
        seen[key] = len(unique)
        unique.append(
            {"name": name, "artist": artists,
             "largest": largest, "square": square}
        )

    tracks = []
    for t in unique[:limit]:
        # 正方形ジャケットを優先。無ければ最大画像。
        tracks.append(
            {"name": t["name"], "artist": t["artist"],
             "art": t["square"] or t["largest"]}
        )
    return tracks


def build_payload(cfg, tracks):
    """
    Discord identity 用のペイロードを組み立てる。
    dynamic の name(キー名)は、ウィジェットエディタで設定した
    Data Field 名と完全一致させること。
    type: 1 = テキスト, 3 = 画像。

    ウィジェットのフィールド構成:
      top_image                … ヘッダー画像(=1位のアルバムアート)
      top_title                … ヘッダーのタイトル(config: widget_title)
      top_subtitle1            … ヘッダーのサブタイトル(config: widget_subtitle)
      value1〜value6           … 各順位の曲名
      label1〜label6           … 各順位のアーティスト
      icon1〜icon6             … 各順位のアルバムアート
    """
    dynamic = []

    def add_text(name, value):
        dynamic.append({"type": 1, "name": name, "value": value})

    def add_image(name, url):
        dynamic.append({"type": 3, "name": name, "value": {"url": url}})

    # 何位まで送るか。config.json の track_count で調整可(既定6)。
    # ウィジェットエディタ側に対応する Data Field がある分だけ表示される。
    count = int(cfg.get("track_count", 6))

    # --- ヘッダー ---
    title = str(cfg.get("widget_title") or "Top Tracks")
    subtitle = str(
        cfg.get("widget_subtitle")
        or PERIOD_LABELS.get(cfg.get("time_range", "short_term"), "")
    )
    add_text("top_title", title)
    add_text("top_subtitle1", subtitle)
    if tracks and tracks[0]["art"]:
        add_image("top_image", tracks[0]["art"])

    # --- 各順位: value=曲名, label=アーティスト, icon=アルバムアート ---
    # 曲が足りない場合はダッシュで埋める。
    for i in range(count):
        rank = i + 1
        t = tracks[i] if i < len(tracks) else None
        # ランキングなので曲名の前に "1. " のような順位接頭辞を付ける。
        add_text(f"value{rank}", f"{rank}. {t['name']}" if t else "—")
        add_text(f"label{rank}", t["artist"] if t else "—")
        if t and t["art"]:
            add_image(f"icon{rank}", t["art"])

    return {
        "username": "Top Tracks",
        "data": {"dynamic": dynamic},
    }


def patch_identity(cfg, payload):
    url = DISCORD_IDENTITY_URL.format(
        app_id=cfg["discord_app_id"], user_id=cfg["discord_user_id"]
    )
    resp = requests.patch(
        url,
        headers={
            "Authorization": f"Bot {cfg['discord_bot_token']}",
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
        },
        data=json.dumps(payload),
        timeout=30,
    )
    if resp.status_code >= 400:
        hint = ""
        err_code = None
        try:
            err_code = resp.json().get("code")
        except Exception:
            pass
        if resp.status_code == 401:
            hint = "\n→ Botトークンが間違っているか失効しています。"
        elif err_code == 50025:
            hint = (
                "\n→ Invalid OAuth2 access token (50025)。主な原因:"
                "\n  1) discord_app_id / discord_bot_token / OAuth認可 が"
                "\n     すべて『同じ1つのアプリ』か? 別アプリの App ID・Secret・Bot を"
                "\n     混在させると必ずこのエラーになります(今回の原因)。"
                "\n  2) 対象アカウント本人がフェーズ3の認可"
                "\n     (scope: sdk.social_layer_presence)を実施済みか。"
            )
        elif resp.status_code in (403, 404):
            hint = "\n→ Application ID / User ID を確認してください。"
        sys.exit(f"Discord 更新に失敗 ({resp.status_code}): {resp.text}{hint}")
    return resp


def main():
    cfg = load_config()
    token = get_spotify_access_token(cfg)
    tracks = get_top_tracks(
        token,
        cfg["time_range"],
        int(cfg.get("track_count", 6)),
        overrides=cfg.get("name_overrides"),
    )

    if not tracks:
        print("警告: トップ曲が空でした。再生履歴が少ない可能性があります。")
    else:
        print(f"取得したトップ曲 ({cfg['time_range']}):")
        for i, t in enumerate(tracks, 1):
            print(f"  {i}. {t['name']} — {t['artist']}")

    payload = build_payload(cfg, tracks)
    patch_identity(cfg, payload)
    print("OK: widget updated")


if __name__ == "__main__":
    main()
