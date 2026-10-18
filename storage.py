"""
storage.py
Simple JSON-backed persistent storage for the bot's state: watchlist,
panel coin list, authorized chat ids, invite token, price alerts.

Survives restarts/redeploys by living on a Railway persistent volume —
set DATA_DIR to the volume's mount path (defaults to /app/data).
Without a volume attached, this still works but resets on every deploy.
"""

import json
import os
import secrets
import threading

DATA_DIR = os.getenv("DATA_DIR", "/app/data")
DATA_FILE = os.path.join(DATA_DIR, "state.json")

_lock = threading.Lock()

_DEFAULT_STATE = {
    "watchlist": {},            # str(chat_id) -> {"symbol","timeframe","min_confidence"}
    "panel_symbols": None,      # None = use PANEL_SYMBOLS env default; else list[str]
    "authorized_chat_ids": [],  # list[int] — chat ids allowed to use the bot
    "invite_token": None,       # str — required as /start payload to get authorized
    "price_alerts": {},         # alert_id(str) -> {"chat_id","symbol","op","price"}
    "strategy": "guarded",       # conservative trend filter is the default
    "strategy_migrated_v2": True, # one-time migration away from the weak default strategy
    "lighter_last_signals": {},    # symbol -> last alerted direction for Lighter scan deduplication
}


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load() -> dict:
    _ensure_dir()
    if not os.path.exists(DATA_FILE):
        state = dict(_DEFAULT_STATE)
        state["invite_token"] = os.getenv("BOT_INVITE_TOKEN") or secrets.token_urlsafe(12)
        save(state)
        return state

    with _lock:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)

    changed = False
    # Single-strategy migration: remove old persisted choices safely.
    if state.get("strategy") != "guarded":
        state["strategy"] = "guarded"
        changed = True
    if not state.get("strategy_migrated_v2"):
        state["strategy_migrated_v2"] = True
        changed = True
    for k, v in _DEFAULT_STATE.items():
        if k not in state:
            state[k] = v
            changed = True
    if not state.get("invite_token"):
        state["invite_token"] = os.getenv("BOT_INVITE_TOKEN") or secrets.token_urlsafe(12)
        changed = True
    if changed:
        save(state)
    return state


def save(state: dict):
    _ensure_dir()
    with _lock:
        tmp_path = DATA_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, DATA_FILE)
