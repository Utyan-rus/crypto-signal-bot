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
from two_way_bootstrap_audit import leave_one_market_out, metrics as paired_metrics, two_way_bootstrap

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
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_BLOCK_LENGTH = 2
BOOTSTRAP_SEED = 20261103


def make_windows(end_ts, days=WINDOW_DAYS, count=OOS_WINDOWS):
    """Return chronological, non-overlapping OOS windows ending at end_ts."""
    seconds = days * 86400
    first = end_ts - count * seconds
    return [
        (f"oos_{i + 1}", first + i * seconds, first + (i + 1) * seconds)
        for i in range(count)
    ]


def buy_hold_baseline(candles, window_start, window_end, start_balance,
                      fee_rate, slippage_rate, min_notional):
    """Costed 1x long buy-and-hold; never bypass the minimum quote notional."""
    if start_balance < min_notional or start_balance <= 0:
        return {"executable": False, "status": "not_executable_below_min_notional",
                "return_pct": None}
    bars = [bar for bar in candles if window_start <= bar["t"] < window_end]
    if len(bars) < 2:
        return {"executable": False, "status": "insufficient_window_data",
                "return_pct": None}
    entry = bars[0]["o"] * (1 + slippage_rate)
    if entry <= 0:
        return {"executable": False, "status": "invalid_entry_price", "return_pct": None}
    qty = start_balance / entry
    if qty * entry < min_notional:
        return {"executable": False, "status": "not_executable_below_min_notional",
                "return_pct": None}
    exit_price = bars[-1]["c"] * (1 - slippage_rate)
    gross = qty * (exit_price - entry)
    fees = abs(qty * entry) * fee_rate + abs(qty * exit_price) * fee_rate
    pnl = gross - fees
    return {"executable": True, "status": "executable_1x_long_no_funding",
            "return_pct": round(100 * pnl / start_balance, 6)}


def summarize(rows):
    """Summarize all market-windows, retaining cash rows and paired buy-and-hold excess."""
    out = []
    keys = sorted({(r["strategy"], r["cost_scenario"]) for r in rows
                   if r.get("cost_scenario") is not None})
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
        benchmark = [r for r in subset if r.get("buy_hold_executable")
                     and r.get("buy_hold_return_pct") not in ("", None)]
        benchmark_returns = [float(r["buy_hold_return_pct"]) for r in benchmark]
        paired = [r for r in subset if r.get("paired_evaluable")
                  and r.get("excess_vs_buy_hold_pp") not in ("", None)]
        excess = [float(r["excess_vs_buy_hold_pp"]) for r in paired]
        benchmark_wins = [bool(r.get("strategy_beats_benchmark_observed")) for r in paired]
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
            "buy_hold_executable_market_windows": len(benchmark),
            "buy_hold_mean_return_pct": round(statistics.mean(benchmark_returns), 3) if benchmark_returns else "",
            "mean_excess_vs_buy_hold_pp": round(statistics.mean(excess), 3) if excess else "",
            "strategy_win_fraction_vs_buy_hold_pct": round(
                100 * sum(benchmark_wins) / len(benchmark_wins), 1
            ) if benchmark_wins else "",
            "mean_mtm_drawdown_pct": round(statistics.mean(drawdowns), 3),
            "total_trades": sum(trade_counts),
            "pf_eligible_windows": len(pf),
            "mean_profit_factor": round(statistics.mean(pf), 3) if pf else "",
            "min_notional_rejections": sum(int(r.get("min_notional_rejections", 0) or 0) for r in subset),
            "historical_funding_coverage_pct": 0.0,
        })
    return out


def bootstrap_summaries(rows, windows, symbols, replicates=BOOTSTRAP_REPLICATES,
                        block_length=BOOTSTRAP_BLOCK_LENGTH, seed=BOOTSTRAP_SEED):
    """Two-way market resampling plus contiguous OOS-window blocks for each fixed strategy/cost."""
    output = []
    keys = sorted({(r["strategy"], r["cost_scenario"]) for r in rows
                   if r.get("strategy") in STRATEGIES and r.get("cost_scenario") is not None})
    for index, (strategy, cost) in enumerate(keys):
        group = [r for r in rows if r.get("strategy") == strategy
                 and r.get("cost_scenario") == cost]
        point = paired_metrics(group)
        run_seed = seed + index
        try:
            intervals = two_way_bootstrap(
                group, windows, symbols, replicates=replicates,
                block_length=block_length, seed=run_seed
            )
            status, error = "ok", ""
        except ValueError as exc:
            intervals, status, error = {}, "incomplete_grid", str(exc)
        output.append({
            "strategy": strategy,
            "cost_scenario": cost,
            "status": status,
            "error": error,
            "markets": len(symbols),
            "oos_windows": len(windows),
            "market_window_rows": point["rows"],
            "eligible_strategy_market_windows": point["eligible_strategy_rows"],
            "paired_evaluable_market_windows": point["paired_rows"],
            "total_trades": point["total_trades"],
            "mean_return_pct": round(point["mean_return_pct"], 4) if point["mean_return_pct"] is not None else "",
            "median_return_pct": round(point["median_return_pct"], 4) if point["median_return_pct"] is not None else "",
            "positive_fraction_pct": round(point["positive_fraction_pct"], 2) if point["positive_fraction_pct"] is not None else "",
            "mean_mtm_drawdown_pct": round(point["mean_mtm_drawdown_pct"], 4) if point["mean_mtm_drawdown_pct"] is not None else "",
            "mean_profit_factor": round(point["mean_profit_factor"], 4) if point["mean_profit_factor"] is not None else "",
            "benchmark_mean_return_pct": round(point["benchmark_mean_return_pct"], 4) if point["benchmark_mean_return_pct"] is not None else "",
            "mean_excess_vs_benchmark_pp": round(point["mean_excess_vs_benchmark_pp"], 4) if point["mean_excess_vs_benchmark_pp"] is not None else "",
            "median_excess_vs_benchmark_pp": round(point["median_excess_vs_benchmark_pp"], 4) if point["median_excess_vs_benchmark_pp"] is not None else "",
            "strategy_win_fraction_vs_benchmark_pct": round(point["benchmark_win_fraction_pct"], 2) if point["benchmark_win_fraction_pct"] is not None else "",
            "bootstrap_replicates": replicates,
            "bootstrap_block_length_windows": block_length,
            "bootstrap_seed": run_seed,
            "bootstrap_method": "two_way_market_resampling_plus_contiguous_window_blocks",
            **intervals,
        })
    return output


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

def run_research(max_markets=5, days=CONTEXT_DAYS, end_ts=None,
                 output_dir="walk-forward-results", start_balance=5.0):
    if days < CONTEXT_DAYS:
        raise ValueError(f"Need at least {CONTEXT_DAYS} days for EMA warmup and OOS windows")
    if start_balance <= 0:
        raise ValueError("start_balance must be positive")
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
        "start_balance": start_balance,
        "balance_role": "diagnostic strategy-capital scenario; the user's $5 feasibility gate is audited separately",
        "minimum_quote_notional_usd": hourly.MIN_NOTIONAL,
        "risk_fraction": hourly.RISK_FRACTION,
        "max_leverage": hourly.MAX_LEVERAGE,
        "cost_scenarios": [
            {"name": name, "fee_per_side": fee, "slippage_per_side": slip}
            for name, fee, slip in COSTS
        ],
        "funding": "excluded: verified historical funding series is not integrated",
        "historical_funding_coverage_pct": 0.0,
        "cash_baseline_return_pct": 0.0,
        "buy_hold_baseline": (
            "not_executable_below_min_notional"
            if start_balance < hourly.MIN_NOTIONAL
            else "1x long buy-and-hold; same per-side fees/slippage; funding excluded"
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_length_windows": BOOTSTRAP_BLOCK_LENGTH,
            "seed": BOOTSTRAP_SEED,
            "method": "two_way_market_resampling_plus_contiguous_window_blocks",
        },
        "paper_only": True,
    }
    with open(os.path.join(output_dir, "walk_forward_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    results, failures = [], []
    original_fee, original_slippage, original_balance = hourly.FEE, hourly.SLIPPAGE, hourly.START_BALANCE
    try:
        hourly.START_BALANCE = start_balance
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
                            benchmark = buy_hold_baseline(
                                candles, window_start, window_end, start_balance,
                                fee, slip, hourly.MIN_NOTIONAL
                            )
                            excess = (
                                row["return_pct"] - benchmark["return_pct"]
                                if benchmark["executable"] else None
                            )
                            row.update({
                                "window": window_name,
                                "window_start_utc": window_start,
                                "window_end_utc": window_end,
                                "window_days": WINDOW_DAYS,
                                "context_days": days,
                                "cost_scenario": cost_name,
                                "direction_mode": "BOTH",
                                "fees_per_side_pct": fee * 100,
                                "slippage_per_side_pct": slip * 100,
                                "funding": "excluded: verified historical funding not integrated",
                                "historical_funding_coverage_pct": 0.0,
                                "cash_return_pct": 0.0,
                                "cash_baseline_return_pct": 0.0,
                                "excess_vs_cash_pp": row["return_pct"],
                                "buy_hold_executable": benchmark["executable"],
                                "buy_hold_status": benchmark["status"],
                                "buy_hold_return_pct": benchmark["return_pct"],
                                "buy_hold_min_quote_notional_usd": hourly.MIN_NOTIONAL,
                                "buy_hold_method": "1x long; fees/slippage on entry and exit; funding excluded",
                                "excess_vs_buy_hold_pp": excess,
                                "paired_evaluable": benchmark["executable"],
                                "strategy_return_pct": row["return_pct"],
                                "strategy_mtm_drawdown_pct": row["max_drawdown_mtm_pct"],
                                "strategy_status": row["status"],
                                "strategy_trades": row["trades"],
                                "strategy_profit_factor": row["profit_factor"],
                                "benchmark_executable": benchmark["executable"],
                                "benchmark_return_pct": benchmark["return_pct"],
                                "observed_excess_return_vs_benchmark_pp": excess,
                                "strategy_beats_benchmark_observed": bool(excess is not None and excess > 0),
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
        hourly.FEE, hourly.SLIPPAGE, hourly.START_BALANCE = original_fee, original_slippage, original_balance

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
                  "mean_excess_vs_cash_pp", "buy_hold_executable_market_windows",
                  "buy_hold_mean_return_pct", "mean_excess_vs_buy_hold_pp",
                  "strategy_win_fraction_vs_buy_hold_pct", "mean_mtm_drawdown_pct", "total_trades",
                  "pf_eligible_windows", "mean_profit_factor", "min_notional_rejections",
                  "historical_funding_coverage_pct"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)

    window_names = [window[0] for window in windows]
    symbols = [market["symbol"] for market in markets]
    bootstrap_summary = bootstrap_summaries(results, window_names, symbols)
    with open(os.path.join(output_dir, "walk_forward_bootstrap_summary.csv"),
              "w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in bootstrap_summary for key in row})
        writer = csv.DictWriter(f, fieldnames=fields or ["strategy", "cost_scenario", "status"])
        writer.writeheader()
        writer.writerows(bootstrap_summary)

    lomo_rows = []
    for strategy in STRATEGIES:
        strategy_rows = [row for row in results if row.get("strategy") == strategy]
        if not strategy_rows:
            continue
        strategy_lomo = leave_one_market_out(
            strategy_rows, [start_balance], [cost[0] for cost in COSTS],
            ["BOTH"], window_names, symbols
        )
        for item in strategy_lomo:
            expected = (len(symbols) if item["omitted_symbol"] == "NONE"
                        else len(symbols) - 1) * len(window_names)
            item["strategy"] = strategy
            item["grid_complete"] = item["market_window_rows"] == expected
            lomo_rows.append(item)
    with open(os.path.join(output_dir, "walk_forward_leave_one_market_out.csv"),
              "w", newline="", encoding="utf-8") as f:
        fields = sorted({key for row in lomo_rows for key in row})
        writer = csv.DictWriter(f, fieldnames=fields or ["strategy", "cost_scenario", "omitted_symbol"])
        writer.writeheader()
        writer.writerows(lomo_rows)
    with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    print("WALK-FORWARD SUMMARY (equal-weight independent market-window tests):")
    for row in sorted(summary, key=lambda r: (r["cost_scenario"], -r["mean_return_pct"])):
        print(f"  {row['strategy']}/{row['cost_scenario']}: mean {row['mean_return_pct']:+.2f}%, "
              f"positive {row['positive_market_windows']}/{row['observations']}, "
              f"active {row['active_market_windows']}, no-trade {row['no_trade_market_windows']}, "
              f"min-notional rejects {row['min_notional_rejections']}, "
              f"buy-hold {row['buy_hold_mean_return_pct']}, excess {row['mean_excess_vs_buy_hold_pp']}, "
              f"DD {row['mean_mtm_drawdown_pct']:.2f}%, PF {row['mean_profit_factor']}, "
              f"trades {row['total_trades']}")
    print("TWO-WAY BOOTSTRAP (10,000 replicates; 2-window contiguous blocks):")
    for row in bootstrap_summary:
        if row["status"] == "ok":
            print(f"  {row['strategy']}/{row['cost_scenario']}: "
                  f"mean CI95=[{row.get('two_way_mean_return_pct_ci95_low')},"
                  f"{row.get('two_way_mean_return_pct_ci95_high')}], "
                  f"excess-vs-buy-hold CI95=[{row.get('two_way_mean_excess_vs_benchmark_pp_ci95_low')},"
                  f"{row.get('two_way_mean_excess_vs_benchmark_pp_ci95_high')}]")
        else:
            print(f"  {row['strategy']}/{row['cost_scenario']}: CI not computed ({row['error']})")
    incomplete = sum(row.get("status") == "incomplete_history" for row in results)
    print(f"Leave-one-market-out rows={len(lomo_rows)}; incomplete markets={incomplete}; "
          f"funding coverage=0.00%; Rows={len(results)}, API failures={len(failures)}. Paper-only.")
    return results, summary, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=CONTEXT_DAYS)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="walk-forward-results")
    parser.add_argument("--start-balance", type=float, default=5.0)
    args = parser.parse_args()
    run_research(args.max_markets, args.days, output_dir=args.out,
                 start_balance=args.start_balance)


if __name__ == "__main__":
    main()
