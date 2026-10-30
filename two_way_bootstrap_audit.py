"""Two-way market/time block bootstrap and leave-one-market-out audit for paired OOS results.

This is a diagnostic of uncertainty across the predeclared market cohort and contiguous
OOS windows. It does not tune strategy parameters and is paper-only.
"""
import argparse
import csv
import random
import statistics
from collections import defaultdict

from paired_bootstrap_audit import (
    as_bool,
    as_float,
    as_int,
    moving_block_sample,
    quantile,
    read_csv,
    validate_rows,
    write_csv,
)


def metrics(rows):
    returns = [as_float(row.get("strategy_return_pct"), 0.0) for row in rows]
    drawdowns = [as_float(row.get("strategy_mtm_drawdown_pct"), 0.0) for row in rows]
    active_pf = [
        as_float(row.get("strategy_profit_factor"))
        for row in rows
        if row.get("strategy_status") == "tested" and as_int(row.get("strategy_trades")) >= 5
    ]
    active_pf = [value for value in active_pf if value is not None]
    benchmark_rows = [
        row for row in rows
        if as_bool(row.get("benchmark_executable"))
        and as_float(row.get("benchmark_return_pct")) is not None
    ]
    paired_rows = [
        row for row in rows
        if as_bool(row.get("paired_evaluable"))
        and as_bool(row.get("benchmark_executable"))
        and as_float(row.get("observed_excess_return_vs_benchmark_pp")) is not None
    ]
    benchmark_returns = [as_float(row.get("benchmark_return_pct")) for row in benchmark_rows]
    excess = [as_float(row.get("observed_excess_return_vs_benchmark_pp")) for row in paired_rows]
    wins = [as_bool(row.get("strategy_beats_benchmark_observed")) for row in paired_rows]
    return {
        "rows": len(rows),
        "mean_return_pct": statistics.mean(returns) if returns else None,
        "median_return_pct": statistics.median(returns) if returns else None,
        "positive_fraction_pct": 100 * sum(value > 0 for value in returns) / len(returns) if returns else None,
        "mean_mtm_drawdown_pct": statistics.mean(drawdowns) if drawdowns else None,
        "mean_profit_factor": statistics.mean(active_pf) if active_pf else None,
        "total_trades": sum(as_int(row.get("strategy_trades")) for row in rows),
        "eligible_strategy_rows": sum(
            row.get("strategy_status") == "tested" and as_int(row.get("strategy_trades")) >= 5
            for row in rows
        ),
        "benchmark_rows": len(benchmark_rows),
        "benchmark_mean_return_pct": statistics.mean(benchmark_returns) if benchmark_returns else None,
        "paired_rows": len(paired_rows),
        "mean_excess_vs_benchmark_pp": statistics.mean(excess) if excess else None,
        "median_excess_vs_benchmark_pp": statistics.median(excess) if excess else None,
        "benchmark_win_fraction_pct": 100 * sum(wins) / len(wins) if wins else None,
    }


def two_way_bootstrap(group_rows, windows, symbols, replicates=10000, block_length=2, seed=20261030):
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    if not symbols or not windows:
        raise ValueError("market cohort and OOS windows must be non-empty")
    if block_length < 1 or block_length > len(windows):
        raise ValueError("block length must be between 1 and the number of windows")

    grid = {(str(row["symbol"]).upper(), row["window"]): row for row in group_rows}
    expected = {(symbol, window) for symbol in symbols for window in windows}
    if set(grid) != expected:
        raise ValueError("two-way bootstrap requires a complete market/window grid")

    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(replicates):
        sampled_symbols = [rng.choice(symbols) for _ in symbols]
        sampled_windows = moving_block_sample(windows, block_length, rng)
        sampled_rows = [grid[(symbol, window)] for symbol in sampled_symbols for window in sampled_windows]
        stat = metrics(sampled_rows)
        for key in (
            "mean_return_pct",
            "median_return_pct",
            "positive_fraction_pct",
            "mean_mtm_drawdown_pct",
            "mean_profit_factor",
            "benchmark_mean_return_pct",
            "mean_excess_vs_benchmark_pp",
            "median_excess_vs_benchmark_pp",
            "benchmark_win_fraction_pct",
        ):
            if stat[key] is not None:
                samples[key].append(stat[key])

    intervals = {}
    for key, values in samples.items():
        intervals[f"two_way_{key}_ci95_low"] = quantile(values, 0.025)
        intervals[f"two_way_{key}_ci95_high"] = quantile(values, 0.975)
    return intervals


def leave_one_market_out(rows, balances, costs, directions, windows, symbols):
    output = []
    for balance in balances:
        for cost in costs:
            for direction in directions:
                group = [
                    row for row in rows
                    if float(row["start_balance"]) == balance
                    and row["cost_scenario"] == cost
                    and str(row.get("direction_mode", "BOTH")).upper() == direction
                ]
                for omitted in ["NONE"] + symbols:
                    subset = group if omitted == "NONE" else [
                        row for row in group if str(row["symbol"]).upper() != omitted
                    ]
                    stat = metrics(subset)
                    output.append({
                        "start_balance": balance,
                        "cost_scenario": cost,
                        "direction_mode": direction,
                        "omitted_symbol": omitted,
                        "markets_included": len(symbols) if omitted == "NONE" else len(symbols) - 1,
                        "oos_windows": len(windows),
                        "market_window_rows": stat["rows"],
                        "eligible_strategy_market_windows": stat["eligible_strategy_rows"],
                        "paired_evaluable_market_windows": stat["paired_rows"],
                        "total_trades": stat["total_trades"],
                        "strategy_mean_return_pct": round(stat["mean_return_pct"], 4) if stat["mean_return_pct"] is not None else "",
                        "strategy_median_return_pct": round(stat["median_return_pct"], 4) if stat["median_return_pct"] is not None else "",
                        "strategy_positive_fraction_pct": round(stat["positive_fraction_pct"], 2) if stat["positive_fraction_pct"] is not None else "",
                        "mean_strategy_mtm_drawdown_pct": round(stat["mean_mtm_drawdown_pct"], 4) if stat["mean_mtm_drawdown_pct"] is not None else "",
                        "mean_profit_factor_active_5plus_trades": round(stat["mean_profit_factor"], 4) if stat["mean_profit_factor"] is not None else "",
                        "benchmark_mean_return_pct": round(stat["benchmark_mean_return_pct"], 4) if stat["benchmark_mean_return_pct"] is not None else "",
                        "mean_excess_vs_benchmark_pp": round(stat["mean_excess_vs_benchmark_pp"], 4) if stat["mean_excess_vs_benchmark_pp"] is not None else "",
                        "median_excess_vs_benchmark_pp": round(stat["median_excess_vs_benchmark_pp"], 4) if stat["median_excess_vs_benchmark_pp"] is not None else "",
                        "strategy_win_fraction_vs_benchmark_pct": round(stat["benchmark_win_fraction_pct"], 2) if stat["benchmark_win_fraction_pct"] is not None else "",
                    })
    return output


def audit(rows, metadata, replicates=10000, block_length=2, seed=20261030):
    windows, _, balances, costs, symbols, directions = validate_rows(rows, metadata)
    summaries = []
    index = 0
    for balance in balances:
        for cost in costs:
            for direction in directions:
                group = [
                    row for row in rows
                    if float(row["start_balance"]) == balance
                    and row["cost_scenario"] == cost
                    and str(row.get("direction_mode", "BOTH")).upper() == direction
                ]
                point = metrics(group)
                intervals = two_way_bootstrap(
                    group, windows, symbols, replicates, block_length, seed + index
                )
                summaries.append({
                    "start_balance": balance,
                    "cost_scenario": cost,
                    "direction_mode": direction,
                    "markets": len(symbols),
                    "oos_windows": len(windows),
                    "market_window_rows": point["rows"],
                    "eligible_strategy_market_windows": point["eligible_strategy_rows"],
                    "paired_evaluable_market_windows": point["paired_rows"],
                    "total_trades": point["total_trades"],
                    "strategy_mean_return_pct": round(point["mean_return_pct"], 4),
                    "strategy_median_return_pct": round(point["median_return_pct"], 4),
                    "strategy_positive_fraction_pct": round(point["positive_fraction_pct"], 2),
                    "mean_strategy_mtm_drawdown_pct": round(point["mean_mtm_drawdown_pct"], 4),
                    "mean_profit_factor_active_5plus_trades": round(point["mean_profit_factor"], 4) if point["mean_profit_factor"] is not None else "",
                    "benchmark_mean_return_pct": round(point["benchmark_mean_return_pct"], 4) if point["benchmark_mean_return_pct"] is not None else "",
                    "mean_excess_vs_benchmark_pp": round(point["mean_excess_vs_benchmark_pp"], 4) if point["mean_excess_vs_benchmark_pp"] is not None else "",
                    "median_excess_vs_benchmark_pp": round(point["median_excess_vs_benchmark_pp"], 4) if point["median_excess_vs_benchmark_pp"] is not None else "",
                    "strategy_win_fraction_vs_benchmark_pct": round(point["benchmark_win_fraction_pct"], 2) if point["benchmark_win_fraction_pct"] is not None else "",
                    "cash_baseline_return_pct": 0.0,
                    "bootstrap_replicates": replicates,
                    "bootstrap_block_length_windows": block_length,
                    "bootstrap_seed": seed + index,
                    "bootstrap_method": "two_way_market_resampling_plus_contiguous_window_blocks",
                    **intervals,
                })
                index += 1
    lomo = leave_one_market_out(rows, balances, costs, directions, windows, symbols)
    return summaries, lomo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="donchian-benchmark-results/donchian_benchmark_pairs.csv")
    parser.add_argument("--metadata", default="donchian-benchmark-results/run_metadata.json")
    parser.add_argument("--out", default="donchian-benchmark-results/paired_two_way_bootstrap_summary.csv")
    parser.add_argument("--market-out", default="donchian-benchmark-results/paired_leave_one_market_out.csv")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()

    rows = read_csv(args.input)
    with open(args.metadata, encoding="utf-8") as handle:
        import json
        metadata = json.load(handle)
    summary, lomo = audit(rows, metadata, args.replicates, args.block_length, args.seed)
    write_csv(args.out, summary)
    write_csv(args.market_out, lomo)
    print(f"Two-way bootstrap complete: rows={len(rows)} groups={len(summary)} "
          f"leave_one_market_out_rows={len(lomo)} replicates={args.replicates}; historical diagnostic only.")
    for row in summary:
        if row["start_balance"] in (50.0, 100.0) and row["direction_mode"] == "BOTH":
            print(
                f"balance={row['start_balance']:g} cost={row['cost_scenario']} BOTH: "
                f"mean={row['strategy_mean_return_pct']:+.3f}% "
                f"CI95=[{row.get('two_way_mean_return_pct_ci95_low', '')},"
                f"{row.get('two_way_mean_return_pct_ci95_high', '')}] "
                f"excess={row.get('mean_excess_vs_benchmark_pp', '')}pp "
                f"CI95=[{row.get('two_way_mean_excess_vs_benchmark_pp_ci95_low', '')},"
                f"{row.get('two_way_mean_excess_vs_benchmark_pp_ci95_high', '')}]"
            )


if __name__ == "__main__":
    main()
