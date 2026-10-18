"""Historical paper backtest for Lighter perpetual markets.
Uses public candles only; no private keys and no exchange orders.
"""
import argparse
import csv
import json
import math
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://mainnet.zklighter.elliot.ai/api/v1"
UA = "crypto-signal-bot-paper-backtest/1.0"
MIN_REQUEST_INTERVAL = 1.0
_last_request_at = 0.0


def get_json(path, params=None, max_attempts=7):
    """Rate-limited GET with exponential backoff for HTTP 429 and transient 5xx."""
    global _last_request_at
    url = BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    for attempt in range(max_attempts):
        wait = MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                _last_request_at = time.monotonic()
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            _last_request_at = time.monotonic()
            if exc.code != 429 and exc.code not in (500, 502, 503, 504):
                raise
            if attempt + 1 >= max_attempts:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60.0, 2.0 ** (attempt + 1))
            except (TypeError, ValueError):
                delay = min(60.0, 2.0 ** (attempt + 1))
            print(f"  API HTTP {exc.code}; retry {attempt + 1}/{max_attempts - 1} after {delay:.1f}s")
            time.sleep(max(1.0, delay))
        except (TimeoutError, urllib.error.URLError):
            _last_request_at = time.monotonic()
            if attempt + 1 >= max_attempts:
                raise
            delay = min(30.0, 2.0 ** (attempt + 1))
            print(f"  API connection issue; retry {attempt + 1}/{max_attempts - 1} after {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("API request failed after retries")


def field(obj, *names, default=None):
    for name in names:
        if isinstance(obj, dict) and obj.get(name) is not None:
            return obj[name]
    return default


def get_markets():
    data = get_json("/orderBooks", {"filter": "perp"})
    rows = field(data, "order_books", "orderBooks", "markets", default=[])
    if isinstance(rows, dict):
        rows = rows.get("order_books", [])
    out = []
    for row in rows or []:
        mid = field(row, "market_id", "marketId")
        sym = field(row, "symbol", "market_symbol", "marketSymbol")
        status = str(field(row, "status", default="active")).lower()
        typ = str(field(row, "market_type", "marketType", default="perp")).lower()
        if mid is not None and sym and "inactive" not in status and "closed" not in status and "spot" not in typ:
            out.append({"id": int(mid), "symbol": str(sym)})
    return out


def fetch_candles(market_id, resolution, start_ts, end_ts):
    # API supports up to 500 candles per call; request non-overlapping chunks.
    step = 300 if resolution == "5m" else 900
    out = {}
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(end_ts, cursor + step * 499)
        params = {"market_id": market_id, "resolution": resolution,
                  "start_timestamp": cursor, "end_timestamp": chunk_end, "count_back": 500}
        data = get_json("/candles", params)
        rows = field(data, "candles", "c", default=[])
        for row in rows or []:
            try:
                ts = int(field(row, "timestamp", "time", "t"))
                # Normalize milliseconds if API returns millisecond timestamps.
                if ts > 10_000_000_000:
                    ts //= 1000
                candle = {"t": ts, "o": float(field(row, "open", "o")),
                          "h": float(field(row, "high", "h")), "l": float(field(row, "low", "l")),
                          "c": float(field(row, "close", "c")), "v": float(field(row, "volume", "v", default=0))}
                step = 300 if resolution == "5m" else 900
                if start_ts <= ts and ts + step <= end_ts:
                    out[ts] = candle
            except (TypeError, ValueError):
                continue
        cursor = chunk_end + step
        # get_json enforces a global request interval and retries rate limits.
    return sorted(out.values(), key=lambda x: x["t"])


def ema(values, period):
    out = [None] * len(values)
    if len(values) < period:
        return out
    value = sum(values[:period]) / period
    out[period - 1] = value
    alpha = 2 / (period + 1)
    for i in range(period, len(values)):
        value = values[i] * alpha + value * (1 - alpha)
        out[i] = value
    return out


def rsi_series(values, period=14):
    out = [None] * len(values)
    if len(values) <= period:
        return out
    changes = [values[i] - values[i - 1] for i in range(1, len(values))]
    gain = sum(max(x, 0) for x in changes[:period]) / period
    loss = sum(max(-x, 0) for x in changes[:period]) / period
    def calc(g, l):
        if l == 0:
            return 100.0
        return 100 - 100 / (1 + g / l)
    out[period] = calc(gain, loss)
    for i in range(period + 1, len(values)):
        change = changes[i - 1]
        gain = (gain * (period - 1) + max(change, 0)) / period
        loss = (loss * (period - 1) + max(-change, 0)) / period
        out[i] = calc(gain, loss)
    return out


def signal_series(candles):
    """Guarded trend filter: EMA20/50/200, RSI, MACD, ADX, ATR and volume."""
    closes = [x["c"] for x in candles]
    highs = [x["h"] for x in candles]
    lows = [x["l"] for x in candles]
    volumes = [x.get("v", 0.0) for x in candles]
    e12, e20, e26, e50, e200 = (ema(closes, n) for n in (12, 20, 26, 50, 200))
    rsis = rsi_series(closes)
    macd_line = [None if e12[i] is None or e26[i] is None else e12[i] - e26[i] for i in range(len(closes))]
    macd_signal = [None] * len(closes)
    valid_macd = [x for x in macd_line if x is not None]
    signal_values = ema(valid_macd, 9)
    offset = len(macd_line) - len(valid_macd)
    for j, value in enumerate(signal_values):
        if value is not None:
            macd_signal[offset + j] = value
    atr_values = [None] * len(candles)
    dx_values = [None] * len(candles)
    plus_di = [None] * len(candles)
    minus_di = [None] * len(candles)
    for i in range(14, len(candles)):
        trs, pds, mds = [], [], []
        for k in range(i - 13, i + 1):
            trk = max(highs[k] - lows[k], abs(highs[k] - closes[k-1]), abs(lows[k] - closes[k-1]))
            uk, dk = highs[k] - highs[k-1], lows[k-1] - lows[k]
            trs.append(trk)
            pds.append(uk if uk > dk and uk > 0 else 0.0)
            mds.append(dk if dk > uk and dk > 0 else 0.0)
        atr_values[i] = sum(trs) / len(trs)
        total_tr = sum(trs)
        pdi = 100 * sum(pds) / total_tr if total_tr else 0.0
        mdi = 100 * sum(mds) / total_tr if total_tr else 0.0
        plus_di[i], minus_di[i] = pdi, mdi
        dx_values[i] = 100 * abs(pdi - mdi) / (pdi + mdi) if pdi + mdi else 0.0
    out = {}
    for i, c in enumerate(candles):
        if (e20[i] is None or e50[i] is None or e200[i] is None or rsis[i] is None
                or atr_values[i] is None or plus_di[i] is None or macd_signal[i] is None):
            continue
        dx_window = [x for x in dx_values[max(0, i-13):i+1] if x is not None]
        adx = sum(dx_window) / len(dx_window) if dx_window else 0.0
        vols = volumes[max(0, i-19):i+1]
        avg_vol = sum(vols) / len(vols) if vols else 0.0
        volume_ok = avg_vol > 0 and volumes[i] >= avg_vol * 0.8
        hist = (macd_line[i] - macd_signal[i]) if macd_line[i] is not None and macd_signal[i] is not None else 0
        long_ok = (closes[i] > e200[i] and e20[i] > e50[i] > e200[i]
                   and 52 <= rsis[i] <= 68 and hist > 0 and plus_di[i] > minus_di[i]
                   and adx >= 18 and volume_ok)
        short_ok = (closes[i] < e200[i] and e20[i] < e50[i] < e200[i]
                    and 32 <= rsis[i] <= 48 and hist < 0 and minus_di[i] > plus_di[i]
                    and adx >= 18 and volume_ok)
        side = "LONG" if long_ok else ("SHORT" if short_ok else "NEUTRAL")
        out[c["t"]] = (side, closes[i], rsis[i], atr_values[i] * 1.5)
    return out

def backtest(symbol, candles5, candles15, start_balance=5.0, risk_fraction=0.005,
             leverage=3.0, fee_rate=0.0005, slippage_rate=0.0005, min_notional=1.0, rr=2.0):
    s5, s15 = signal_series(candles5), signal_series(candles15)
    if len(candles5) < 100 or len(candles15) < 60:
        return [], {"symbol": symbol, "status": "insufficient_data", "bars_5m": len(candles5), "bars_15m": len(candles15)}
    times15 = sorted(s15)
    trades, balance, peak, max_dd = [], start_balance, start_balance, 0.0
    pos = None
    j15 = -1
    # Trade on next 5m candle open to avoid look-ahead bias.
    for i in range(1, len(candles5)):
        prev, bar = candles5[i - 1], candles5[i]
        t = prev["t"]
        while j15 + 1 < len(times15) and times15[j15 + 1] <= t:
            j15 += 1
        trend = s15[times15[j15]][0] if j15 >= 0 else "NEUTRAL"
        sig_data = s5.get(t, ("NEUTRAL", 0, 0, 0))
        sig = sig_data[0]
        closed_this_bar = False
        if pos:
            exit_price, reason = None, None
            if pos["side"] == "LONG":
                if bar["l"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["h"] >= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            else:
                if bar["h"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["l"] <= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            if exit_price is not None:
                exit_price *= (1 - slippage_rate if pos["side"] == "LONG" else 1 + slippage_rate)
                gross = pos["qty"] * (exit_price - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
                exit_fee = abs(pos["qty"] * exit_price) * fee_rate
                pnl = gross - exit_fee - pos["entry_fee"]
                balance += pnl
                peak = max(peak, balance)
                max_dd = max(max_dd, (peak - balance) / peak if peak > 0 else 1.0)
                trades.append({"symbol": symbol, "side": pos["side"], "entry_time": pos["entry_time"],
                               "exit_time": bar["t"], "entry": round(pos["entry"], 10),
                               "exit": round(exit_price, 10), "qty": round(pos["qty"], 10),
                               "reason": reason, "pnl": round(pnl, 8), "balance": round(balance, 8)})
                pos = None
                closed_this_bar = True
        if not pos and not closed_this_bar and balance > 0 and sig in ("LONG", "SHORT") and sig == trend:
            side = sig
            entry = bar["o"] * (1 + slippage_rate if side == "LONG" else 1 - slippage_rate)
            risk_cash = balance * risk_fraction
            stop_distance = max(float(sig_data[3]), entry * 0.002)
            # Budget for the stop move plus estimated entry/exit fees and slippage.
            risk_per_unit = stop_distance + entry * 2 * (fee_rate + slippage_rate)
            qty_by_risk = risk_cash / risk_per_unit
            qty_by_leverage = balance * leverage / entry
            qty = min(qty_by_risk, qty_by_leverage)
            # Conservative generic fee model; no exchange-specific min order/lot metadata available here.
            if qty <= 0 or qty * entry < min_notional:
                continue
            stop = entry - stop_distance if side == "LONG" else entry + stop_distance
            target = entry + stop_distance * rr if side == "LONG" else entry - stop_distance * rr
            entry_fee = qty * entry * fee_rate
            pos = {"side": side, "entry": entry, "qty": qty, "stop": stop, "target": target,
                   "entry_fee": entry_fee, "entry_time": bar["t"]}
    if pos:
        bar = candles5[-1]
        exit_price = bar["c"] * (1 - slippage_rate if pos["side"] == "LONG" else 1 + slippage_rate)
        gross = pos["qty"] * (exit_price - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
        pnl = gross - abs(pos["qty"] * exit_price) * fee_rate - pos["entry_fee"]
        balance += pnl
        peak = max(peak, balance)
        max_dd = max(max_dd, (peak - balance) / peak if peak > 0 else 1.0)
        trades.append({"symbol": symbol, "side": pos["side"], "entry_time": pos["entry_time"],
                       "exit_time": bar["t"], "entry": round(pos["entry"], 10), "exit": round(exit_price, 10),
                       "qty": round(pos["qty"], 10), "reason": "END", "pnl": round(pnl, 8), "balance": round(balance, 8)})
    wins = sum(1 for x in trades if x["pnl"] > 0)
    summary = {"symbol": symbol, "status": "ok", "bars_5m": len(candles5), "bars_15m": len(candles15),
               "trades": len(trades), "wins": wins, "win_rate_pct": round(100*wins/len(trades), 2) if trades else 0,
               "start_balance": start_balance, "end_balance": round(balance, 8),
               "model": "guarded trend filter; each market simulated independently",
               "risk_fraction_pct": risk_fraction * 100, "leverage_cap": leverage, "min_notional": min_notional,
               "net_pnl": round(balance-start_balance, 8), "return_pct": round(100*(balance/start_balance-1), 3),
               "max_drawdown_pct": round(max_dd*100, 3), "fees": "estimated taker 0.05% per side",
               "slippage": "estimated 0.05% per side", "funding": "NOT included; historical funding feed not implemented"}
    return trades, summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--max-markets", type=int, default=10, help="Safety cap; set 0 to attempt all markets")
    p.add_argument("--fee", type=float, default=0.0005, help="Fee per side as decimal")
    p.add_argument("--slippage", type=float, default=0.0005, help="Slippage per side as decimal")
    p.add_argument("--min-notional", type=float, default=1.0, help="Skip trades below this estimated notional")
    p.add_argument("--out", default="backtest-results")
    args = p.parse_args()
    if args.days < 30 or args.days > 90:
        p.error("--days must be between 30 and 90")
    os.makedirs(args.out, exist_ok=True)
    mkts = get_markets()
    if args.max_markets > 0:
        mkts = mkts[:args.max_markets]
    end = int(time.time())
    start = end - args.days * 86400
    summaries, all_trades, failures = [], [], []
    print(f"Perpetual markets selected: {len(mkts)}; history: {args.days} days; paper only.")
    for n, market in enumerate(mkts, 1):
        print(f"[{n}/{len(mkts)}] {market['symbol']} candles download")
        try:
            c5 = fetch_candles(market["id"], "5m", start, end)
            c15 = fetch_candles(market["id"], "15m", start, end)
            expected5, expected15 = args.days * 288, args.days * 96
            coverage5, coverage15 = len(c5) / expected5, len(c15) / expected15
            # Exclude partial listing/API histories instead of mislabeling them as a full-period test.
            if coverage5 < 0.90 or coverage15 < 0.90:
                summaries.append({"symbol": market["symbol"], "status": "incomplete_data",
                                  "bars_5m": len(c5), "bars_15m": len(c15),
                                  "expected_5m": expected5, "expected_15m": expected15,
                                  "coverage_5m_pct": round(coverage5 * 100, 2),
                                  "coverage_15m_pct": round(coverage15 * 100, 2),
                                  "start_balance": 5.0, "end_balance": 5.0,
                                  "note": "Excluded: less than 90% of requested history"})
                print(f"  INCOMPLETE: coverage={coverage5:.1%}/{coverage15:.1%}; excluded")
                continue
            trades, summary = backtest(market["symbol"], c5, c15, fee_rate=args.fee,
                                       slippage_rate=args.slippage, min_notional=args.min_notional)
            summaries.append(summary)
            all_trades.extend(trades)
            print(f"  {summary['status']}: bars={len(c5)}/{len(c15)}, trades={summary.get('trades', 0)}")
        except Exception as exc:
            failures.append({"symbol": market["symbol"], "error": type(exc).__name__ + ": " + str(exc)[:180]})
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:180]}")
        time.sleep(0.15)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        fields = sorted({k for row in summaries for k in row})
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(summaries)
    with open(os.path.join(args.out, "trades.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["symbol","side","entry_time","exit_time","entry","exit","qty","reason","pnl","balance"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_trades)
    with open(os.path.join(args.out, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    ok = [x for x in summaries if x.get("status") == "ok"]
    incomplete = [x for x in summaries if x.get("status") == "incomplete_data"]
    print(f"Finished: {len(ok)} full-coverage markets, {len(incomplete)} incomplete markets excluded, {len(failures)} API failures, {len(all_trades)} trades.")
    print("Funding is not included. Results are research estimates; each market has its own $5 balance, not a shared portfolio.")
    if not ok:
        raise SystemExit("No markets produced a valid backtest. Inspect failures.json.")


if __name__ == "__main__":
    main()
