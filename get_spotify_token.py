#!/usr/bin/env python3
"""
Spotify のリフレッシュトークンを取得する初回セットアップ用スクリプト。

使い方:
    python get_spotify_token.py

事前に config.json の spotify_client_id / spotify_client_secret を
埋めておき、Spotify Dashboard の Redirect URI に
http://127.0.0.1:8888/callback を登録しておくこと。

ブラウザが開くので Spotify にログインして許可すると、
コンソールにリフレッシュトークンが表示される。
それを config.json の spotify_refresh_token に貼り付ける。
"""

import base64
import http.server
import json
import secrets
import sys
import threading
import urllib.parse
import webbrowser
from pathlib import Path

import requests

# Windows のコンソール(cp932 等)でも日本語を出力できるようにする。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

CONFIG_PATH = Path(__file__).with_name("config.json")
REDIRECT_URI = "http://127.0.0.1:8888/callback"
SCOPE = "user-top-read"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

# 認可コードを受け取るためのフラグ付き共有領域
_result = {"code": None, "state": None, "error": None}
_done = threading.Event()


class CallbackHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            return
        params = urllib.parse.parse_qs(parsed.query)
        _result["code"] = params.get("code", [None])[0]
        _result["state"] = params.get("state", [None])[0]
        _result["error"] = params.get("error", [None])[0]
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        msg = "認可が完了しました。このタブを閉じてターミナルに戻ってください。"
        if _result["error"]:
            msg = f"エラー: {_result['error']}。ターミナルを確認してください。"
        self.wfile.write(
            f"<html><body style='font-family:sans-serif'><h2>{msg}</h2>"
            "</body></html>".encode("utf-8")
        )
        _done.set()

    def log_message(self, *args):
        pass  # サーバーのアクセスログを抑制


def load_config():
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"config.json が見つかりません: {CONFIG_PATH}\n"
            "config.example.json をコピーして値を埋めてください。"
        )
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def main():
    cfg = load_config()
    client_id = cfg.get("spotify_client_id", "").strip()
    client_secret = cfg.get("spotify_client_secret", "").strip()
    if not client_id or not client_secret:
        raise SystemExit(
            "config.json の spotify_client_id / spotify_client_secret を先に埋めてください。"
        )

    state = secrets.token_urlsafe(16)
    auth_query = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPE,
            "state": state,
            "show_dialog": "true",
        }
    )
    auth_link = f"{AUTH_URL}?{auth_query}"

    # ローカルサーバーを起動してリダイレクトを待ち受ける
    server = http.server.HTTPServer(("127.0.0.1", 8888), CallbackHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print("ブラウザで Spotify の認可画面を開きます...")
    print("開かない場合は以下のURLを手動で開いてください:\n")
    print(auth_link, "\n")
    webbrowser.open(auth_link)

    # 認可完了を待機(最大5分)
    if not _done.wait(timeout=300):
        server.shutdown()
        raise SystemExit("タイムアウトしました。もう一度実行してください。")
    server.shutdown()

    if _result["error"]:
        raise SystemExit(f"認可が拒否されました: {_result['error']}")
    if _result["state"] != state:
        raise SystemExit("state が一致しません。攻撃の可能性があるため中断します。")
    code = _result["code"]
    if not code:
        raise SystemExit("認可コードを取得できませんでした。")

    # 認可コード → トークン交換
    basic = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    resp = requests.post(
        TOKEN_URL,
        headers={
            "Authorization": f"Basic {basic}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
        },
        timeout=30,
    )
    if resp.status_code != 200:
        raise SystemExit(f"トークン取得に失敗: {resp.status_code} {resp.text}")

    data = resp.json()
    refresh_token = data.get("refresh_token")
    if not refresh_token:
        raise SystemExit(f"refresh_token が返りませんでした: {data}")

    print("\n" + "=" * 60)
    print("リフレッシュトークンを取得しました。")
    print("以下を config.json の spotify_refresh_token に貼り付けてください:\n")
    print(refresh_token)
    print("=" * 60)

    # 自動で config.json に書き戻す(任意)
    try:
        cfg["spotify_refresh_token"] = refresh_token
        CONFIG_PATH.write_text(
            json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print("\nconfig.json に自動保存しました。")
    except Exception as e:
        print(f"\n自動保存に失敗しました(手動で貼り付けてください): {e}")


if __name__ == "__main__":
    main()
