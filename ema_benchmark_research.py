"""EMA 20/50 funding-aware OOS study with buy-and-hold benchmark. Paper only."""
import argparse
import csv
import os
import statistics
import time

from backtest import fetch_candles, get_markets
from funding_aware_research import fetch_funding_rates, simulate_with_funding
from hourly_candidate_research import FEE, SLIPPAGE


def buy_hold(symbol, candles, start, end, funding, balance):
    bars = [b for b in candles if start <= b["t"] < end]
    if len(bars) < 2:
        return {"symbol": symbol, "trades": 0, "status": "low_sample"}
    entry = bars[0]["o"] * (1 + SLIPPAGE)
    exit_price = bars[-1]["c"] * (1 - SLIPPAGE)
    qty = balance / (entry * (1 + FEE))
    gross = qty * (exit_price - entry)
    fees = qty * entry * FEE + qty * exit_price * FEE
    funding_cost = 0.0
    charged = 0
    for bar in bars[1:]:
        if bar["t"] in funding:
            funding_cost += qty * bar["c"] * funding[bar["t"]]
            charged += 1
    end_balance = balance + gross - fees - funding_cost
    return {
        "symbol": symbol, "trades": 1, "start_balance": balance,
        "end_balance": round(end_balance, 6),
        "return_pct": round(100 * (end_balance / balance - 1), 3),
        "funding_events_charged": charged,
        "funding_net_cost_usd": round(funding_cost, 8),
        "status": "tested",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--max-markets", type=int, default=5)
    p.add_argument("--out", default="ema-benchmark-results")
    p.add_argument("--balances", default="5,50,100")
    args = p.parse_args()
    if args.days < 365:
        p.error("Use at least 365 days for warm-up and eight independent 30-day OOS windows.")
    balances = [float(x.strip()) for x in args.balances.split(",") if x.strip()]
    if not balances or any(x <= 0 for x in balances):
        p.error("--balances must contain positive numbers")
    os.makedirs(args.out, exist_ok=True)
    end = int(time.time())
    start = end - args.days * 86400
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    rows, errors = [], []
    windows = [(f"oos_{i+1:02d}", end - (8-i)*30*86400, end - (7-i)*30*86400) for i in range(8)]
    for market in markets:
        symbol, mid = market["symbol"], market["id"]
        try:
            candles = fetch_candles(mid, "1h", start, end)
            funding = fetch_funding_rates(mid, start, end)
            if len(candles) < args.days * 24 * 0.8:
                rows.append({"symbol": symbol, "strategy": "ALL", "status": "incomplete_candles", "bars_1h": len(candles)})
                continue
            if not funding:
                rows.append({"symbol": symbol, "strategy": "ALL", "status": "missing_funding_history", "bars_1h": len(candles)})
                continue
            for window, lo, hi in windows:
                for balance in balances:
                    row = simulate_with_funding(symbol, candles, "ema_20_50_cross", lo, hi, funding, start_balance=balance)
                    row.update({"window": window, "benchmark": False, "bars_1h": len(candles),
                                "funding_coverage_pct": round(100 * len([t for t in funding if start <= t < end]) / (args.days*24), 2)})
                    rows.append(row)
                    bench = buy_hold(symbol, candles, lo, hi, funding, balance)
                    bench.update({"strategy": "buy_hold_long_1x", "window": window, "benchmark": True,
                                  "funding_coverage_pct": round(100 * len([t for t in funding if lo <= t < hi]) / max(1, ((hi-lo)//3600)), 2)})
                    rows.append(bench)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:200]})
        time.sleep(0.15)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "ema_benchmark_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out, "errors.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "error"])
        writer.writeheader()
        writer.writerows(errors)
    print(f"Rows={len(rows)}, API failures={len(errors)}. Paper-only.")
    for strategy in ("ema_20_50_cross", "buy_hold_long_1x"):
        for balance in balances:
            subset = [r for r in rows if r.get("strategy") == strategy and r.get("start_balance") == balance and r.get("status") == "tested" and r.get("trades", 0) >= (5 if strategy == "ema_20_50_cross" else 1)]
            returns = [r["return_pct"] for r in subset]
            if returns:
                print(f"{strategy} balance={balance:g}: n={len(returns)}, mean={statistics.mean(returns):+.3f}% median={statistics.median(returns):+.3f}% positive={sum(x > 0 for x in returns)}/{len(returns)}")
            else:
                print(f"{strategy} balance={balance:g}: no adequate samples")


if __name__ == "__main__":
    main()
