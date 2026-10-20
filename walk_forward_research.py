"""Fixed-parameter walk-forward audit of hourly Lighter strategy candidates.

Paper-only: public candles, no order placement. Uses causal indicators from full
chronological history and four non-overlapping 30-day OOS windows. Funding is
explicitly excluded because verified historical funding series are not integrated.
"""
import argparse
import csv
import json
import os
import statistics
import time

import hourly_candidate_research as hourly
from backtest import fetch_candles, get_markets

STRATEGIES = ("donchian_20_10", "ema_20_50_cross", "ema_trend_pullback")
WINDOW_DAYS = 30
OOS_WINDOWS = 4
CONTEXT_DAYS = 180
COSTS = (
    ("zero_fee_base_slippage", 0.0, 0.0005),
    ("assumed_taker_base_slippage", 0.0005, 0.0005),
    ("assumed_taker_double_slippage", 0.0005, 0.0010),
    ("double_fee_double_slippage", 0.0010, 0.0010),
)


def make_windows(end_ts, days=WINDOW_DAYS, count=OOS_WINDOWS):
    """Return chronological, non-overlapping OOS windows ending at end_ts."""
    seconds = days * 86400
    first = end_ts - count * seconds
    return [
        (f"oos_{i + 1}", first + i * seconds, first + (i + 1) * seconds)
        for i in range(count)
    ]


def summarize(rows):
    out = []
    keys = sorted({(r["strategy"], r["cost_scenario"]) for r in rows})
    for strategy, cost in keys:
        subset = [r for r in rows if r["strategy"] == strategy
                  and r["cost_scenario"] == cost and r.get("status") == "tested"]
        if not subset:
            continue
        returns = [float(r["return_pct"]) for r in subset]
        out.append({
            "strategy": strategy,
            "cost_scenario": cost,
            "observations": len(subset),
            "markets": len({r["symbol"] for r in subset}),
            "windows": len({r["window"] for r in subset}),
            "mean_return_pct": round(statistics.mean(returns), 3),
            "median_return_pct": round(statistics.median(returns), 3),
            "positive_market_windows": sum(x > 0 for x in returns),
            "positive_fraction_pct": round(100 * sum(x > 0 for x in returns) / len(returns), 1),
            "mean_mtm_drawdown_pct": round(statistics.mean(float(r["max_drawdown_mtm_pct"]) for r in subset), 3),
            "total_trades": sum(int(r["trades"]) for r in subset),
            "mean_profit_factor": round(statistics.mean(float(r["profit_factor"]) for r in subset), 3),
        })
    return out


def run_research(max_markets=5, days=CONTEXT_DAYS, end_ts=None, output_dir="walk-forward-results"):
    if days < CONTEXT_DAYS:
        raise ValueError(f"Need at least {CONTEXT_DAYS} days for EMA warmup and OOS windows")
    os.makedirs(output_dir, exist_ok=True)
    end_ts = int(end_ts or time.time())
    start_ts = end_ts - days * 86400
    windows = make_windows(end_ts)
    markets = get_markets()
    if max_markets > 0:
        markets = markets[:max_markets]
    results, failures = [], []
    original_fee, original_slippage = hourly.FEE, hourly.SLIPPAGE
    try:
        for n, market in enumerate(markets, 1):
            symbol = market["symbol"]
            print(f"[{n}/{len(markets)}] {symbol}")
            try:
                candles = fetch_candles(market["id"], "1h", start_ts, end_ts)
                if len(candles) < days * 24 * 0.8:
                    results.append({"symbol": symbol, "strategy": "ALL",
                                    "status": "incomplete_history", "bars_1h": len(candles)})
                    continue
                # Build causal indicators once; each trade is still gated to a distinct OOS window.
                for strategy in STRATEGIES:
                    for window_name, window_start, window_end in windows:
                        for cost_name, fee, slip in COSTS:
                            hourly.FEE, hourly.SLIPPAGE = fee, slip
                            row = hourly.simulate(symbol, candles, strategy, window_start, window_end)
                            row.update({
                                "window": window_name,
                                "window_start_utc": window_start,
                                "window_end_utc": window_end,
                                "window_days": WINDOW_DAYS,
                                "context_days": days,
                                "cost_scenario": cost_name,
                                "fees_per_side_pct": fee * 100,
                                "slippage_per_side_pct": slip * 100,
                                "funding": "excluded: verified historical funding not integrated",
                                "fixed_parameters": True,
                                "status": row["status"],
                            })
                            results.append(row)
                            print(f"  {strategy}/{window_name}/{cost_name}: "
                                  f"{row['return_pct']:+.2f}%, DD {row['max_drawdown_mtm_pct']:.2f}%, "
                                  f"PF {row['profit_factor']}, n={row['trades']}")
                time.sleep(0.2)
            except Exception as exc:
                failures.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:200]})
                print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:200]}")
    finally:
        hourly.FEE, hourly.SLIPPAGE = original_fee, original_slippage

    if results:
        with open(os.path.join(output_dir, "walk_forward_detail.csv"), "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for row in results for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    summary = summarize(results)
    with open(os.path.join(output_dir, "walk_forward_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["strategy", "cost_scenario", "observations", "markets", "windows",
                  "mean_return_pct", "median_return_pct", "positive_market_windows",
                  "positive_fraction_pct", "mean_mtm_drawdown_pct", "total_trades", "mean_profit_factor"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    print("WALK-FORWARD SUMMARY (equal-weight independent market-window tests):")
    for row in sorted(summary, key=lambda r: (r["cost_scenario"], -r["mean_return_pct"])):
        print(f"  {row['strategy']}/{row['cost_scenario']}: mean {row['mean_return_pct']:+.2f}%, "
              f"positive {row['positive_market_windows']}/{row['observations']}, "
              f"DD {row['mean_mtm_drawdown_pct']:.2f}%, PF {row['mean_profit_factor']:.2f}, "
              f"trades {row['total_trades']}")
    print(f"Rows={len(results)}, API failures={len(failures)}. Funding excluded; paper-only.")
    return results, summary, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=CONTEXT_DAYS)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="walk-forward-results")
    args = parser.parse_args()
    run_research(args.max_markets, args.days, output_dir=args.out)


if __name__ == "__main__":
    main()
