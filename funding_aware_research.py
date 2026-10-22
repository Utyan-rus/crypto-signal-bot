"""Funding-aware hourly paper research for Lighter perpetuals.
Uses public candles and historical /fundings data only. Never places orders.
"""
import argparse
import csv
import json
import os
import time
from backtest import fetch_candles, get_json, get_markets
from hourly_candidate_research import FEE, SLIPPAGE, START_BALANCE, build_model, size_position, max_dd

def fetch_funding_rates(market_id, start_ts, end_ts):
    """Fetch hourly history in <=700-hour chunks; return timestamp -> signed rate."""
    out = {}
    cursor = start_ts
    while cursor < end_ts:
        chunk_end = min(end_ts, cursor + 700 * 3600)
        data = get_json("/fundings", {"market_id": market_id, "resolution": "1h",
            "start_timestamp": cursor, "end_timestamp": chunk_end, "count_back": 0})
        rows = data.get("fundings", []) if isinstance(data, dict) else []
        for row in rows or []:
            try:
                ts = int(row.get("timestamp", row.get("t")))
                if ts > 10_000_000_000:
                    ts //= 1000
                if row.get("rate") is None:
                    continue
                rate = float(row["rate"])
                direction = str(row.get("direction", "")).lower()
                if direction == "short":
                    rate = -abs(rate)
                elif direction == "long":
                    rate = abs(rate)
                if start_ts <= ts < end_ts:
                    out[ts] = rate
            except (TypeError, ValueError):
                continue
        cursor = chunk_end
    return out

def simulate_with_funding(symbol, candles, strategy, test_start, test_end, funding):
    signals, exit_levels = build_model(candles, strategy)
    balance, equity, pos, trades = START_BALANCE, [START_BALANCE], None, []
    funding_events = 0
    for i in range(1, len(candles)):
        prev, bar = candles[i - 1], candles[i]
        if bar["t"] >= test_end:
            break
        if pos and bar["t"] in funding and pos["entry_time"] < bar["t"]:
            rate = funding[bar["t"]]
            side_sign = 1 if pos["side"] == "LONG" else -1
            pos["funding_cost"] += pos["qty"] * bar["c"] * rate * side_sign
            funding_events += 1
        if pos:
            exit_price, reason = None, None
            if pos["side"] == "LONG" and bar["l"] <= pos["stop"]:
                exit_price, reason = pos["stop"], "SL"
            elif pos["side"] == "SHORT" and bar["h"] >= pos["stop"]:
                exit_price, reason = pos["stop"], "SL"
            else:
                lo, hi = exit_levels.get(prev["t"], (None, None))
                if pos["side"] == "LONG" and lo is not None and prev["c"] < lo:
                    exit_price, reason = bar["o"], "CHANNEL_EXIT"
                elif pos["side"] == "SHORT" and hi is not None and prev["c"] > hi:
                    exit_price, reason = bar["o"], "CHANNEL_EXIT"
            if exit_price is not None:
                exit_price *= (1-SLIPPAGE if pos["side"] == "LONG" else 1+SLIPPAGE)
                gross = pos["qty"] * (exit_price-pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
                pnl = gross - abs(pos["qty"]*exit_price)*FEE - pos["entry_fee"] - pos["funding_cost"]
                balance += pnl
                trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],
                    "entry_time":pos["entry_time"],"exit_time":bar["t"],"pnl":pnl,
                    "funding_cost":pos["funding_cost"],"reason":reason})
                pos = None
        current_sig = signals.get(prev["t"])
        active = bool(current_sig)
        previous_sig = signals.get(candles[i-2]["t"]) if i >= 2 else None
        fresh = active and (not previous_sig or previous_sig[0] != current_sig[0])
        if pos is None and test_start <= bar["t"] < test_end and fresh and balance > 0:
            side, stop_dist = current_sig
            entry = bar["o"] * (1+SLIPPAGE if side == "LONG" else 1-SLIPPAGE)
            qty = size_position(balance, entry, stop_dist)
            if qty > 0:
                stop = entry-stop_dist if side == "LONG" else entry+stop_dist
                pos = {"side":side,"entry":entry,"qty":qty,"stop":stop,
                    "entry_fee":qty*entry*FEE,"entry_time":bar["t"],"funding_cost":0.0}
                hit = bar["l"] <= stop if side == "LONG" else bar["h"] >= stop
                if hit:
                    exit_price = stop * (1-SLIPPAGE if side == "LONG" else 1+SLIPPAGE)
                    pnl = qty*(exit_price-entry)*(1 if side=="LONG" else -1) - abs(qty*exit_price)*FEE - pos["entry_fee"]
                    trades.append({"symbol":symbol,"strategy":strategy,"side":side,
                        "entry_time":bar["t"],"exit_time":bar["t"],"pnl":pnl,
                        "funding_cost":0.0,"reason":"SAME_BAR_SL"})
                    balance += pnl
                    pos = None
        mtm = balance
        if pos:
            mark = bar["c"]
            gross = pos["qty"]*(mark-pos["entry"])*(1 if pos["side"]=="LONG" else -1)
            mtm += gross - abs(pos["qty"]*mark)*(FEE+SLIPPAGE) - pos["entry_fee"] - pos["funding_cost"]
        equity.append(max(0.0, mtm))
    if pos:
        bar = next((b for b in reversed(candles) if b["t"] < test_end), candles[-1])
        exit_price = bar["c"] * (1-SLIPPAGE if pos["side"]=="LONG" else 1+SLIPPAGE)
        pnl = pos["qty"]*(exit_price-pos["entry"])*(1 if pos["side"]=="LONG" else -1) - abs(pos["qty"]*exit_price)*FEE - pos["entry_fee"] - pos["funding_cost"]
        balance += pnl
        trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],
            "entry_time":pos["entry_time"],"exit_time":bar["t"],"pnl":pnl,
            "funding_cost":pos["funding_cost"],"reason":"END"})
        equity.append(balance)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    return {"symbol":symbol,"strategy":strategy,"trades":len(trades),
        "return_pct":round(100*(balance/START_BALANCE-1),3),"end_balance":round(balance,6),
        "max_drawdown_mtm_pct":round(max_dd(equity),3),
        "profit_factor":round(gp/gl,3) if gl else (999.0 if gp else 0.0),
        "funding_events_charged":funding_events,
        "funding_net_cost_usd":round(sum(t.get("funding_cost",0) for t in trades),8),
        "funding_history_rows":len(funding),"fees_per_side_pct":FEE*100,
        "slippage_per_side_pct":SLIPPAGE*100,"min_notional_assumption_usd":1.0,
        "status":"tested" if len(trades)>=5 else "low_sample"}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=180)
    p.add_argument("--max-markets", type=int, default=5)
    p.add_argument("--out", default="funding-aware-results")
    args = p.parse_args()
    if args.days < 120:
        p.error("Use at least 120 days for warm-up and OOS windows.")
    os.makedirs(args.out, exist_ok=True)
    end = int(time.time()); start = end - args.days*86400
    markets = get_markets()
    if args.max_markets > 0: markets = markets[:args.max_markets]
    windows = (("oos_older_30d", end-60*86400, end-30*86400),
               ("oos_recent_30d", end-30*86400, end))
    strategies = ("donchian_20_10","ema_20_50_cross","ema_trend_pullback")
    rows, errors = [], []
    for market in markets:
        symbol, mid = market["symbol"], market["id"]
        print(f"[{symbol}] fetching candles and historical funding")
        try:
            candles = fetch_candles(mid, "1h", start, end)
            funding = fetch_funding_rates(mid, start, end)
            if len(candles) < args.days*24*0.8:
                rows.append({"symbol":symbol,"strategy":"ALL","status":"incomplete_candles","bars_1h":len(candles)})
                continue
            if not funding:
                rows.append({"symbol":symbol,"strategy":"ALL","status":"missing_funding_history","bars_1h":len(candles)})
                continue
            for strategy in strategies:
                for window, lo, hi in windows:
                    row = simulate_with_funding(symbol,candles,strategy,lo,hi,funding)
                    row.update({"window":window,"bars_1h":len(candles),
                        "funding_coverage_pct":round(100*len([t for t in funding if start <= t < end])/(args.days*24),2),
                        "funding_source":"Lighter /api/v1/fundings resolution=1h",
                        "funding_sign_convention":"positive rate: long pays, short receives"})
                    rows.append(row)
        except Exception as exc:
            errors.append({"symbol":symbol,"error":type(exc).__name__+": "+str(exc)[:200]})
            print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:200]}")
        time.sleep(0.15)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out,"funding_aware_summary.csv"),"w",newline="",encoding="utf-8") as f:
        w = csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with open(os.path.join(args.out,"errors.json"),"w",encoding="utf-8") as f:
        json.dump(errors,f,indent=2)
    print(f"Finished: {len(rows)} result rows, {len(errors)} API failures. Paper research only.")
    if errors: print(json.dumps(errors,indent=2))

if __name__ == "__main__":
    main()
