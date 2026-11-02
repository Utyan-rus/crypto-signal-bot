"""Fixed-parameter walk-forward audit of hourly Lighter strategy candidates.

Paper-only: public candles, no order placement. Uses causal indicators from full
chronological history and eight non-overlapping 30-day OOS windows. Funding is
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
OOS_WINDOWS = 8
CONTEXT_DAYS = 365
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
    """Summarize every market-window, retaining no-trade cash rows in the denominator."""
    out = []
    keys = sorted({(r["strategy"], r["cost_scenario"]) for r in rows})
    for strategy, cost in keys:
        subset = [r for r in rows if r["strategy"] == strategy
                  and r["cost_scenario"] == cost]
        if not subset:
            continue
        returns = [float(r["return_pct"]) for r in subset]
        tested = [r for r in subset if r.get("status") == "tested"
                  and int(r.get("trades", 0) or 0) >= 5]
        tested_returns = [float(r["return_pct"]) for r in tested]
        pf = [float(r["profit_factor"]) for r in tested
              if r.get("profit_factor") not in ("", None)]
        drawdowns = [float(r.get("max_drawdown_mtm_pct", 0) or 0) for r in subset]
        trade_counts = [int(r.get("trades", 0) or 0) for r in subset]
        out.append({
            "strategy": strategy,
            "cost_scenario": cost,
            "observations": len(subset),
            "tested_market_windows": len(tested),
            "low_sample_market_windows": len(subset) - len(tested),
            "active_market_windows": sum(n > 0 for n in trade_counts),
            "no_trade_market_windows": sum(n == 0 for n in trade_counts),
            "mean_return_pct": round(statistics.mean(returns), 3),
            "median_return_pct": round(statistics.median(returns), 3),
            "tested_only_mean_return_pct": round(statistics.mean(tested_returns), 3) if tested_returns else "",
            "positive_market_windows": sum(x > 0 for x in returns),
            "positive_fraction_pct": round(100 * sum(x > 0 for x in returns) / len(returns), 1),
            "cash_baseline_return_pct": 0.0,
            "mean_excess_vs_cash_pp": round(statistics.mean(returns), 3),
            "mean_mtm_drawdown_pct": round(statistics.mean(drawdowns), 3),
            "total_trades": sum(trade_counts),
            "pf_eligible_windows": len(pf),
            "mean_profit_factor": round(statistics.mean(pf), 3) if pf else "",
            "min_notional_rejections": sum(int(r.get("min_notional_rejections", 0) or 0) for r in subset),
        })
    return out


RESEARCH_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")


def select_research_markets(markets, max_markets=5):
    """Select a fixed, ordered crypto-major cohort instead of API response order."""
    if max_markets < 1 or max_markets > len(RESEARCH_SYMBOLS):
        raise ValueError(f"max_markets must be between 1 and {len(RESEARCH_SYMBOLS)}")
    by_symbol = {str(market.get("symbol", "")).upper(): market for market in markets}
    selected_symbols = RESEARCH_SYMBOLS[:max_markets]
    missing = [symbol for symbol in selected_symbols if symbol not in by_symbol]
    if missing:
        raise RuntimeError("predeclared markets unavailable: " + ",".join(missing))
    return [by_symbol[symbol] for symbol in selected_symbols]

def run_research(max_markets=5, days=CONTEXT_DAYS, end_ts=None, output_dir="walk-forward-results"):
    if days < CONTEXT_DAYS:
        raise ValueError(f"Need at least {CONTEXT_DAYS} days for EMA warmup and OOS windows")
    os.makedirs(output_dir, exist_ok=True)
    end_ts = int(end_ts or time.time())
    start_ts = end_ts - days * 86400
    windows = make_windows(end_ts)
    markets = select_research_markets(get_markets(), max_markets)
    metadata = {
        "experiment": "fixed_parameter_hourly_walk_forward_min_notional_corrected",
        "selection_rule": "predeclared ordered crypto-major cohort; no OOS-based market selection",
        "requested_symbols": list(RESEARCH_SYMBOLS[:max_markets]),
        "markets_tested": [market["symbol"] for market in markets],
        "days_of_history_requested": days,
        "warmup_days_available": days - OOS_WINDOWS * WINDOW_DAYS,
        "oos_window_count": len(windows),
        "oos_window_days": WINDOW_DAYS,
        "shared_end_timestamp": end_ts,
        "history_start_timestamp": start_ts,
        "start_balance": hourly.START_BALANCE,
        "minimum_quote_notional_usd": hourly.MIN_NOTIONAL,
        "risk_fraction": hourly.RISK_FRACTION,
        "max_leverage": hourly.MAX_LEVERAGE,
        "cost_scenarios": [
            {"name": name, "fee_per_side": fee, "slippage_per_side": slip}
            for name, fee, slip in COSTS
        ],
        "funding": "excluded: this $5 feasibility audit is not the funding-aware performance benchmark",
        "cash_baseline_return_pct": 0.0,
        "buy_hold_baseline": "not_executable: $5 1x quote notional is below the venue $10 minimum",
        "paper_only": True,
    }
    with open(os.path.join(output_dir, "walk_forward_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
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
                                "cash_return_pct": 0.0,
                                "excess_vs_cash_pp": row["return_pct"],
                                "buy_hold_status": "not_executable_minimum_quote_notional",
                                "buy_hold_min_quote_notional_usd": hourly.MIN_NOTIONAL,
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
        fields = ["strategy", "cost_scenario", "observations", "tested_market_windows",
                  "low_sample_market_windows", "active_market_windows", "no_trade_market_windows",
                  "mean_return_pct", "median_return_pct", "tested_only_mean_return_pct",
                  "positive_market_windows", "positive_fraction_pct", "cash_baseline_return_pct",
                  "mean_excess_vs_cash_pp", "mean_mtm_drawdown_pct", "total_trades",
                  "pf_eligible_windows", "mean_profit_factor", "min_notional_rejections"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    print("WALK-FORWARD SUMMARY (equal-weight independent market-window tests):")
    for row in sorted(summary, key=lambda r: (r["cost_scenario"], -r["mean_return_pct"])):
        print(f"  {row['strategy']}/{row['cost_scenario']}: mean {row['mean_return_pct']:+.2f}%, "
              f"positive {row['positive_market_windows']}/{row['observations']}, "
              f"active {row['active_market_windows']}, no-trade {row['no_trade_market_windows']}, "
              f"min-notional rejects {row['min_notional_rejections']}, "
              f"DD {row['mean_mtm_drawdown_pct']:.2f}%, PF {row['mean_profit_factor']}, "
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
