"""Hourly futures strategy research on public Lighter candles. Paper-only; never places orders.

Compares three lower-turnover, established strategy families on 180 days of 1h candles.
Indicators are computed on the full history; entries are restricted to the final 60 days.
Uses mark-to-market equity, fees and slippage on both sides, conservative stop-first handling.
"""
import argparse
import csv
import json
import os
import time

from backtest import fetch_candles, get_markets

FEE = 0.0005
SLIPPAGE = 0.0005
START_BALANCE = 5.0
RISK_FRACTION = 0.005
MAX_LEVERAGE = 3.0
MIN_NOTIONAL = 1.0


def ema(values, period):
    out = [None] * len(values)
    if not values:
        return out
    alpha = 2.0 / (period + 1)
    value = values[0]
    for i, x in enumerate(values):
        value = x * alpha + value * (1-alpha)
        if i >= period - 1:
            out[i] = value
    return out


def atr(candles, period=14):
    out = [None] * len(candles)
    tr = []
    for i, b in enumerate(candles):
        prev = candles[i-1]["c"] if i else b["c"]
        tr.append(max(b["h"]-b["l"], abs(b["h"]-prev), abs(b["l"]-prev)))
        if i >= period:
            out[i] = sum(tr[i-period+1:i+1]) / period
    return out


def rolling_high(values, period, end_exclusive=True):
    out = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = max(values[i-period:i] if end_exclusive else values[i-period+1:i+1])
    return out


def rolling_low(values, period, end_exclusive=True):
    out = [None] * len(values)
    for i in range(period, len(values)):
        out[i] = min(values[i-period:i] if end_exclusive else values[i-period+1:i+1])
    return out


def build_model(candles, name):
    close = [b["c"] for b in candles]
    high = [b["h"] for b in candles]
    low = [b["l"] for b in candles]
    a = atr(candles)
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    don_high = rolling_high(high, 20)
    don_low = rolling_low(low, 20)
    exit_low = rolling_low(low, 10)
    exit_high = rolling_high(high, 10)
    signals = {}
    exits = {}
    for i, b in enumerate(candles):
        if a[i] is None or a[i] <= 0 or e200[i] is None:
            continue
        side = None
        stop_dist = 2.5 * a[i]
        if name == "donchian_20_10":
            if don_high[i] is not None and close[i] > don_high[i] and close[i] > e200[i]:
                side = "LONG"
            elif don_low[i] is not None and close[i] < don_low[i] and close[i] < e200[i]:
                side = "SHORT"
            exits[b["t"]] = (exit_low[i], exit_high[i])
        elif name == "ema_trend_pullback":
            # Trend regime + a close back across EMA20; avoids repeated entries in same run.
            prev_i = i - 1
            if prev_i >= 0 and e20[prev_i] is not None:
                if close[i] > e200[i] and e20[i] > e50[i] and close[prev_i] <= e20[prev_i] < close[i]:
                    side = "LONG"
                elif close[i] < e200[i] and e20[i] < e50[i] and close[prev_i] >= e20[prev_i] > close[i]:
                    side = "SHORT"
            exits[b["t"]] = (e20[i], e20[i])
        elif name == "ema_20_50_cross":
            prev_i = i - 1
            if prev_i >= 0 and e20[prev_i] is not None and e50[prev_i] is not None:
                if e20[prev_i] <= e50[prev_i] and e20[i] > e50[i] and close[i] > e200[i]:
                    side = "LONG"
                elif e20[prev_i] >= e50[prev_i] and e20[i] < e50[i] and close[i] < e200[i]:
                    side = "SHORT"
            exits[b["t"]] = (e20[i], e20[i])
        if side:
            signals[b["t"]] = (side, max(stop_dist, close[i] * 0.003))
    return signals, exits


def max_dd(equity):
    peak = equity[0] if equity else START_BALANCE
    worst = 0.0
    for v in equity:
        peak = max(peak, v)
        if peak > 0:
            worst = max(worst, (peak-v)/peak)
    return worst * 100


def simulate(symbol, candles, strategy, test_start):
    signals, exit_levels = build_model(candles, strategy)
    balance = START_BALANCE
    equity = [balance]
    peak = balance
    pos = None
    trades = []
    last_signal_side = None
    last_signal_active = False
    for i in range(1, len(candles)):
        prev, bar = candles[i-1], candles[i]
        current_sig = signals.get(prev["t"])
        active = bool(current_sig)
        fresh = active and (not last_signal_active or current_sig[0] != last_signal_side)
        last_signal_active = active
        last_signal_side = current_sig[0] if active else None
        if pos:
            exit_price = None
            reason = None
            if pos["side"] == "LONG" and bar["l"] <= pos["stop"]:
                exit_price, reason = pos["stop"], "SL"
            elif pos["side"] == "SHORT" and bar["h"] >= pos["stop"]:
                exit_price, reason = pos["stop"], "SL"
            else:
                levels = exit_levels.get(prev["t"])
                if levels:
                    lo, hi = levels
                    if pos["side"] == "LONG" and lo is not None and prev["c"] < lo:
                        exit_price, reason = bar["o"], "CHANNEL_EXIT"
                    elif pos["side"] == "SHORT" and hi is not None and prev["c"] > hi:
                        exit_price, reason = bar["o"], "CHANNEL_EXIT"
            if exit_price is not None:
                exit_price *= (1-SLIPPAGE if pos["side"] == "LONG" else 1+SLIPPAGE)
                gross = pos["qty"] * (exit_price-pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
                pnl = gross - abs(pos["qty"]*exit_price)*FEE - pos["entry_fee"]
                balance += pnl
                trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],
                               "entry_time":pos["entry_time"],"exit_time":bar["t"],
                               "pnl":pnl,"reason":reason})
                pos = None
        if pos is None and bar["t"] >= test_start and fresh and balance > 0:
            sig = current_sig
            side, stop_dist = sig
            entry = bar["o"] * (1+SLIPPAGE if side == "LONG" else 1-SLIPPAGE)
            qty = balance * RISK_FRACTION / stop_dist
            qty = max(qty, MIN_NOTIONAL / entry)
            qty = min(qty, balance * MAX_LEVERAGE / entry)
            if qty > 0 and qty*entry >= MIN_NOTIONAL:
                stop = entry-stop_dist if side == "LONG" else entry+stop_dist
                pos = {"side":side,"entry":entry,"qty":qty,"stop":stop,
                       "entry_fee":qty*entry*FEE,"entry_time":bar["t"]}
                # Conservative same-bar stop handling.
                hit = bar["l"] <= stop if side == "LONG" else bar["h"] >= stop
                if hit:
                    exit_price = stop * (1-SLIPPAGE if side == "LONG" else 1+SLIPPAGE)
                    pnl = qty*(exit_price-entry)*(1 if side=="LONG" else -1) - abs(qty*exit_price)*FEE - pos["entry_fee"]
                    balance += pnl
                    trades.append({"symbol":symbol,"strategy":strategy,"side":side,
                                   "entry_time":bar["t"],"exit_time":bar["t"],"pnl":pnl,"reason":"SAME_BAR_SL"})
                    pos = None
        # Mark open position to market; include estimated exit cost conservatively.
        mtm = balance
        if pos:
            mark = bar["c"]
            gross = pos["qty"]*(mark-pos["entry"])*(1 if pos["side"]=="LONG" else -1)
            mtm += gross - abs(pos["qty"]*mark)*(FEE+SLIPPAGE) - pos["entry_fee"]
        equity.append(max(0.0, mtm))
    if pos:
        bar = candles[-1]
        exit_price = bar["c"] * (1-SLIPPAGE if pos["side"]=="LONG" else 1+SLIPPAGE)
        pnl = pos["qty"]*(exit_price-pos["entry"])*(1 if pos["side"]=="LONG" else -1) - abs(pos["qty"]*exit_price)*FEE - pos["entry_fee"]
        balance += pnl
        trades.append({"symbol":symbol,"strategy":strategy,"side":pos["side"],
                       "entry_time":pos["entry_time"],"exit_time":bar["t"],"pnl":pnl,"reason":"END"})
        equity.append(balance)
    wins = sum(t["pnl"] > 0 for t in trades)
    gp = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gl = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    return {"symbol":symbol,"strategy":strategy,"trades":len(trades),"wins":wins,
            "win_rate_pct":round(100*wins/len(trades),2) if trades else 0,
            "start_balance":START_BALANCE,"end_balance":round(balance,6),
            "return_pct":round(100*(balance/START_BALANCE-1),3),
            "max_drawdown_mtm_pct":round(max_dd(equity),3),
            "profit_factor":round(gp/gl,3) if gl else (999.0 if gp else 0.0),
            "long_trades":sum(t["side"]=="LONG" for t in trades),
            "short_trades":sum(t["side"]=="SHORT" for t in trades),
            "fees_per_side_pct":FEE*100,"slippage_per_side_pct":SLIPPAGE*100,
            "status":"tested" if len(trades)>=5 else "low_sample"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=180)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="hourly-candidate-results")
    args = parser.parse_args()
    if args.days < 120:
        parser.error("Use at least 120 days to retain warmup and a 60-day OOS window.")
    os.makedirs(args.out, exist_ok=True)
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    end = int(time.time())
    start = end - args.days*86400
    test_start = end - 60*86400
    strategies = ("donchian_20_10","ema_trend_pullback","ema_20_50_cross")
    results, failures = [], []
    print(f"Hourly research: {len(strategies)} strategies x {len(markets)} markets; {args.days}d context, final 60d OOS.")
    for n, market in enumerate(markets, 1):
        symbol = market["symbol"]
        print(f"[{n}/{len(markets)}] {symbol}")
        try:
            candles = fetch_candles(market["id"], "1h", start, end)
            if len(candles) < args.days*24*0.8:
                results.append({"symbol":symbol,"strategy":"ALL","status":"incomplete_history","bars":len(candles)})
                print(f"  insufficient hourly history: {len(candles)} bars")
                continue
            for strategy in strategies:
                row = simulate(symbol, candles, strategy, test_start)
                row["bars_1h"] = len(candles)
                row["test_window_days"] = 60
                row["cost_assumptions"] = "fee 0.05% + slippage 0.05% each side; no historical funding feed"
                results.append(row)
                print(f"  {strategy}: OOS {row['return_pct']:+.2f}%, DD {row['max_drawdown_mtm_pct']:.2f}%, PF {row['profit_factor']}, trades {row['trades']}")
        except Exception as exc:
            failures.append({"symbol":symbol,"error":type(exc).__name__+": "+str(exc)[:200]})
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}")
        time.sleep(0.2)
    if results:
        with open(os.path.join(args.out,"hourly_summary.csv"),"w",newline="",encoding="utf-8") as f:
            fields = sorted({k for row in results for k in row})
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(results)
    with open(os.path.join(args.out,"failures.json"),"w",encoding="utf-8") as f:
        json.dump(failures,f,ensure_ascii=False,indent=2)
    tested = [r for r in results if r.get("trades",0) >= 5 and r.get("return_pct") is not None]
    print("SUMMARY by strategy (equal-weight average per market):")
    for name in strategies:
        rows = [r for r in tested if r["strategy"] == name]
        if rows:
            print(f"  {name}: mean {sum(r['return_pct'] for r in rows)/len(rows):+.2f}%, "
                  f"positive markets {sum(r['return_pct']>0 for r in rows)}/{len(rows)}, "
                  f"mean MTM DD {sum(r['max_drawdown_mtm_pct'] for r in rows)/len(rows):.2f}%, "
                  f"mean PF {sum(r['profit_factor'] for r in rows)/len(rows):.2f}")
    print(f"Rows={len(results)}, API failures={len(failures)}. Paper-only; no orders placed.")
if __name__ == "__main__":
    main()
