"""Lighter perpetual scanner shared by the Telegram bot and GitHub Actions.

Read-only public market data. No wallet keys and no order endpoints.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://mainnet.zklighter.elliot.ai/api/v1"
UA = "crypto-signal-bot-paper/1.0"

def get_json(path, params=None, attempts=5):
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    last_error = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in (429, 500, 502, 503, 504) or attempt == attempts - 1:
                raise
            try:
                delay = float(exc.headers.get("Retry-After", 2 ** (attempt + 1)))
            except (TypeError, ValueError):
                delay = 2 ** (attempt + 1)
            time.sleep(min(20, max(1, delay)))
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(min(12, 2 ** (attempt + 1)))
    raise RuntimeError(f"Lighter API failed: {type(last_error).__name__}")

def field(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and obj.get(name) is not None:
            return obj[name]
    return default

def list_markets():
    payload = get_json("/orderBooks", {"filter": "perp"})
    rows = field(payload, "order_books", "orderBooks", "markets", default=[])
    if isinstance(rows, dict):
        rows = rows.get("order_books", [])
    out = []
    excluded = {"US500", "US100", "US30", "USDCAD", "EURUSD", "GBPUSD",
                "USDJPY", "AUDUSD", "XAU", "XAG", "XCU"}
    for row in rows or []:
        market_id = field(row, "market_id", "marketId")
        symbol = field(row, "symbol", "market_symbol", "marketSymbol")
        status = str(field(row, "status", default="active")).lower()
        kind = str(field(row, "market_type", "marketType", default="perp")).lower()
        if market_id is None or not symbol or "inactive" in status or "closed" in status or "spot" in kind:
            continue
        base = str(symbol).upper().replace("-PERP", "").replace("/USDC", "").replace("/USDT", "")
        if base in excluded or (base.startswith("USD") and len(base) == 6):
            continue
        out.append({"id": int(market_id), "symbol": str(symbol)})
    priority = {"BTC": 0, "ETH": 1, "SOL": 2, "BNB": 3, "XRP": 4,
                "DOGE": 5, "ADA": 6, "AVAX": 7, "LINK": 8, "LTC": 9}
    def sort_key(item):
        base = item["symbol"].upper().replace("-PERP", "").replace("/USDC", "").replace("/USDT", "")
        return (priority.get(base, 100), item["symbol"])
    return sorted(out, key=sort_key)

def fetch_candles(market_id, resolution, count=260):
    now = int(time.time())
    step = 300 if resolution == "5m" else 900
    payload = get_json("/candles", {
        "market_id": market_id, "resolution": resolution,
        "start_timestamp": now - (count + 5) * step,
        "end_timestamp": now, "count_back": min(500, count + 5),
    })
    rows = field(payload, "candles", "c", default=[])
    out = {}
    for row in rows or []:
        try:
            ts = int(field(row, "timestamp", "time", "t"))
            if ts > 10_000_000_000:
                ts //= 1000
            if ts + step > now:
                continue
            out[ts] = {
                "timestamp": ts * 1000,
                "open": float(field(row, "open", "o")),
                "high": float(field(row, "high", "h")),
                "low": float(field(row, "low", "l")),
                "close": float(field(row, "close", "c")),
                "volume": float(field(row, "volume", "v", default=0)),
            }
        except (TypeError, ValueError):
            continue
    return [out[t] for t in sorted(out)][-count:]

def scan_lighter_markets(max_markets=15):
    import pandas as pd
    from analysis import analyze
    markets = list_markets()
    if not markets:
        raise RuntimeError("Lighter API returned no active perpetual markets.")
    chosen = markets[:max(1, min(int(max_markets), 40))]
    results, errors = [], []
    for market in chosen:
        try:
            c5 = fetch_candles(market["id"], "5m")
            c15 = fetch_candles(market["id"], "15m")
            if len(c5) < 220 or len(c15) < 220:
                continue
            s5 = analyze(pd.DataFrame(c5), strategy="guarded")
            s15 = analyze(pd.DataFrame(c15), strategy="guarded")
            if s5.direction in ("LONG", "SHORT") and s5.direction == s15.direction:
                results.append({
                    "symbol": market["symbol"], "direction": s5.direction,
                    "price": s5.entry, "confidence_5m": s5.confidence,
                    "confidence_15m": s15.confidence, "stop": s5.stop_loss,
                    "target": s5.take_profit, "reasons": s5.reasons[:3],
                })
        except Exception as exc:
            errors.append({"symbol": market["symbol"], "error": type(exc).__name__})
        time.sleep(0.1)
    return {"market_count": len(markets), "scanned": len(chosen),
            "signals": results, "errors": errors,
            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}

def telegram_send(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_OWNER_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN and TELEGRAM_OWNER_ID in GitHub Actions secrets.")
    body = urllib.parse.urlencode({"chat_id": chat_id, "text": text, "disable_web_page_preview": "true"}).encode()
    request = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage", data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": UA}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Telegram send failed with HTTP {exc.code}") from None
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected sendMessage: " + str(result.get("description", "unknown error")))

def format_report(report):
    lines = [
        "Lighter PAPER scan — реальные ордера отключены",
        report["time_utc"],
        f"Рынков найдено: {report['market_count']} | проверено: {report['scanned']} | ошибок: {len(report['errors'])}",
        "Фильтр-тренд: EMA20/50/200, ADX, RSI, MACD, объём; только закрытые свечи.",
        "",
    ]
    if report["signals"]:
        for s in report["signals"][:15]:
            lines.append(f"{s['direction']} {s['symbol']} | {s['price']:.8g} | SL {s['stop']:.8g} | TP {s['target']:.8g} | подтверждение 5m/15m")
    else:
        lines.append("Подтверждённых сигналов 5m + 15m нет.")
    lines += ["", "Симуляция: баланс 5 USDT; риск 0.5% (0.025 USDT); плечо максимум 3x.",
              "Фильтр не является доказательством прибыльности. Перед применением нужна проверка бэктестом."]
    return "\n".join(lines)

def main():
    try:
        count = int(os.getenv("LIGHTER_SCAN_MARKETS", "15"))
    except ValueError:
        count = 15
    report = scan_lighter_markets(count)
    message = format_report(report)
    print(message)
    telegram_send(message)

if __name__ == "__main__":
    main()
