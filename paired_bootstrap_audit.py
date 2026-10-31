"""Moving-block bootstrap audit for paired Donchian OOS results. Diagnostic only; paper-only."""
import argparse
import csv
import json
import os
import random
import statistics
from collections import defaultdict

TRUE_VALUES = {"1", "true", "yes", "y"}


def as_bool(value):
    return str(value).strip().lower() in TRUE_VALUES


def as_float(value, default=None):
    if value is None or str(value).strip() == "":
        return default
    return float(value)


def as_int(value, default=0):
    if value is None or str(value).strip() == "":
        return default
    return int(float(value))


def mean_or_blank(values):
    return round(statistics.mean(values), 4) if values else ""


def median_or_blank(values):
    return round(statistics.median(values), 4) if values else ""


def quantile(values, probability):
    if not values:
        return ""
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 4)


def moving_block_sample(window_names, block_length, rng):
    """Sample contiguous, non-wrapping blocks until the original window count is reached."""
    count = len(window_names)
    if count < 1:
        raise ValueError("at least one OOS window is required")
    if block_length < 1 or block_length > count:
        raise ValueError("block length must be between 1 and the number of windows")
    sampled = []
    while len(sampled) < count:
        start = rng.randrange(0, count - block_length + 1)
        sampled.extend(window_names[start:start + block_length])
    return sampled[:count]


def validate_rows(rows, metadata):
    if not rows:
        raise ValueError("paired CSV has no rows")
    expected_symbols = set(metadata.get("markets_tested", []))
    actual_symbols = {str(row.get("symbol", "")).upper() for row in rows}
    if not expected_symbols or actual_symbols != expected_symbols:
        raise ValueError(
            f"symbol cohort mismatch: metadata={sorted(expected_symbols)} csv={sorted(actual_symbols)}"
        )
    expected_end = int(metadata["shared_end_timestamp"])
    if {as_int(row.get("shared_end_timestamp")) for row in rows} != {expected_end}:
        raise ValueError("rows do not share the metadata end timestamp")

    window_bounds = {}
    for row in rows:
        name = row["window"]
        bounds = (as_int(row["window_start_utc"]), as_int(row["window_end_utc"]))
        if name in window_bounds and window_bounds[name] != bounds:
            raise ValueError(f"inconsistent bounds for {name}")
        window_bounds[name] = bounds
    windows = sorted(window_bounds, key=lambda name: window_bounds[name][0])
    if len(windows) != int(metadata["oos_window_count"]):
        raise ValueError("OOS window count does not match metadata")
    width = int(metadata["oos_window_days"]) * 86400
    for index, name in enumerate(windows):
        start, end = window_bounds[name]
        if end - start != width:
            raise ValueError(f"unexpected OOS window width for {name}")
        if index and window_bounds[windows[index - 1]][1] != start:
            raise ValueError("OOS windows are not contiguous")
    if window_bounds[windows[-1]][1] != expected_end:
        raise ValueError("last OOS window does not end at shared timestamp")

    expected_costs = {item["name"] for item in metadata.get("cost_scenarios", [])}
    actual_costs = {row["cost_scenario"] for row in rows}
    if expected_costs and actual_costs != expected_costs:
        raise ValueError("cost scenarios do not match run metadata")
    expected_directions = {str(value).upper() for value in metadata.get("direction_modes", ["BOTH"])}
    actual_directions = {str(row.get("direction_mode", "BOTH")).upper() for row in rows}
    if not expected_directions or actual_directions != expected_directions:
        raise ValueError("direction modes do not match run metadata")

    balances = sorted({float(row["start_balance"]) for row in rows})
    costs = sorted(actual_costs)
    symbols = sorted(expected_symbols)
    directions = sorted(expected_directions)
    seen = set()
    for row in rows:
        key = (str(row["symbol"]), str(row["window"]), float(row["start_balance"]),
               str(row["cost_scenario"]), str(row.get("direction_mode", "BOTH")).upper())
        if key in seen:
            raise ValueError(f"duplicate result row: {key}")
        seen.add(key)
    expected_count = len(symbols) * len(windows) * len(balances) * len(costs) * len(directions)
    if len(rows) != expected_count:
        raise ValueError(f"incomplete result grid: got {len(rows)} rows, expected {expected_count}")
    expected_keys = {
        (symbol, window, balance, cost, direction)
        for symbol in symbols for window in windows for balance in balances
        for cost in costs for direction in directions
    }
    if seen != expected_keys:
        raise ValueError("result grid is missing one or more market/window/balance/cost/direction combinations")
    return windows, window_bounds, balances, costs, symbols, directions


def summarize_group(group_rows, windows, window_bounds, balance, cost, direction_mode,
                    replicates, block_length, seed):
    by_window = defaultdict(list)
    for row in group_rows:
        by_window[row["window"]].append(row)
    if any(not by_window[name] for name in windows):
        raise ValueError("missing rows in a balance/cost OOS window")

    detail = []
    for name in windows:
        current = by_window[name]
        returns = [as_float(row.get("strategy_return_pct"), 0.0) for row in current]
        benchmark_rows = [row for row in current if as_bool(row.get("benchmark_executable"))]
        benchmark_returns = [as_float(row.get("benchmark_return_pct")) for row in benchmark_rows]
        excess = [as_float(row.get("observed_excess_return_vs_benchmark_pp")) for row in benchmark_rows]
        excess = [value for value in excess if value is not None]
        wins = [as_bool(row.get("strategy_beats_benchmark_observed")) for row in benchmark_rows]
        active = [row for row in current if as_int(row.get("strategy_trades")) > 0]
        tested_active = [
            row for row in active
            if row.get("strategy_status") == "tested" and as_int(row.get("strategy_trades")) >= 5
        ]
        dds = [as_float(row.get("strategy_mtm_drawdown_pct")) for row in current]
        dds = [value for value in dds if value is not None]
        pfs = [as_float(row.get("strategy_profit_factor")) for row in tested_active]
        pfs = [value for value in pfs if value is not None]
        start, end = window_bounds[name]
        detail.append({
            "window": name, "window_start_utc": start, "window_end_utc": end,
            "start_balance": balance, "cost_scenario": cost, "direction_mode": direction_mode,
            "markets": len(current),
            "strategy_mean_return_pct": statistics.mean(returns),
            "strategy_median_return_pct": statistics.median(returns),
            "strategy_positive_markets": sum(value > 0 for value in returns),
            "strategy_positive_fraction_pct": 100 * sum(value > 0 for value in returns) / len(returns),
            "strategy_mean_mtm_drawdown_pct": statistics.mean(dds) if dds else None,
            "mean_profit_factor_active_5plus_trades": statistics.mean(pfs) if pfs else None,
            "active_market_windows": len(active),
            "eligible_strategy_market_windows": len(tested_active),
            "total_trades": sum(as_int(row.get("strategy_trades")) for row in current),
            "benchmark_executable_markets": len(benchmark_rows),
            "benchmark_mean_return_pct": statistics.mean(benchmark_returns) if benchmark_returns else None,
            "benchmark_median_return_pct": statistics.median(benchmark_returns) if benchmark_returns else None,
            "excess_vs_benchmark_mean_pp": statistics.mean(excess) if excess else None,
            "excess_vs_benchmark_median_pp": statistics.median(excess) if excess else None,
            "win_fraction_vs_benchmark_pct": 100 * sum(wins) / len(wins) if wins else None,
        })

    # The bootstrap resamples contiguous OOS windows as clusters, preserving all markets
    # within each sampled window. Estimates are conditional on this fixed market cohort.
    rng = random.Random(seed)
    boot_mean_return, boot_median_return, boot_positive_fraction = [], [], []
    boot_benchmark_mean, boot_excess_mean, boot_win_fraction = [], [], []
    detail_by_name = {row["window"]: row for row in detail}
    for _ in range(replicates):
        sampled_names = moving_block_sample(windows, block_length, rng)
        sampled_rows = [row for name in sampled_names for row in by_window[name]]
        returns = [as_float(row.get("strategy_return_pct"), 0.0) for row in sampled_rows]
        boot_mean_return.append(statistics.mean(
            detail_by_name[name]["strategy_mean_return_pct"] for name in sampled_names
        ))
        boot_median_return.append(statistics.median(returns))
        boot_positive_fraction.append(statistics.mean(
            detail_by_name[name]["strategy_positive_fraction_pct"] for name in sampled_names
        ))

        bench_means = [detail_by_name[name]["benchmark_mean_return_pct"] for name in sampled_names
                       if detail_by_name[name]["benchmark_mean_return_pct"] is not None]
        excess_means = [detail_by_name[name]["excess_vs_benchmark_mean_pp"] for name in sampled_names
                        if detail_by_name[name]["excess_vs_benchmark_mean_pp"] is not None]
        win_fractions = [detail_by_name[name]["win_fraction_vs_benchmark_pct"] for name in sampled_names
                         if detail_by_name[name]["win_fraction_vs_benchmark_pct"] is not None]
        if bench_means:
            boot_benchmark_mean.append(statistics.mean(bench_means))
        if excess_means:
            boot_excess_mean.append(statistics.mean(excess_means))
        if win_fractions:
            boot_win_fraction.append(statistics.mean(win_fractions))

    all_returns = [as_float(row.get("strategy_return_pct"), 0.0) for row in group_rows]
    all_dds = [as_float(row.get("strategy_mtm_drawdown_pct")) for row in group_rows]
    all_dds = [value for value in all_dds if value is not None]
    all_pfs = [
        as_float(row.get("strategy_profit_factor")) for row in group_rows
        if row.get("strategy_status") == "tested" and as_int(row.get("strategy_trades")) >= 5
    ]
    all_pfs = [value for value in all_pfs if value is not None]
    benchmark_rows = [row for row in group_rows if as_bool(row.get("benchmark_executable"))]
    benchmark_returns = [as_float(row.get("benchmark_return_pct")) for row in benchmark_rows]
    benchmark_returns = [value for value in benchmark_returns if value is not None]
    excess = [as_float(row.get("observed_excess_return_vs_benchmark_pp")) for row in benchmark_rows]
    excess = [value for value in excess if value is not None]
    wins = sum(as_bool(row.get("strategy_beats_benchmark_observed")) for row in benchmark_rows)
    trades = sum(as_int(row.get("strategy_trades")) for row in group_rows)
    eligible_strategy_rows = sum(
        row.get("strategy_status") == "tested" and as_int(row.get("strategy_trades")) >= 5
        for row in group_rows
    )
    window_means = [row["strategy_mean_return_pct"] for row in detail]
    window_excess_means = [row["excess_vs_benchmark_mean_pp"] for row in detail
                           if row["excess_vs_benchmark_mean_pp"] is not None]
    window_benchmark_means = [row["benchmark_mean_return_pct"] for row in detail
                              if row["benchmark_mean_return_pct"] is not None]
    no_executable = eligible_strategy_rows == 0

    summary = {
        "start_balance": balance, "cost_scenario": cost, "direction_mode": direction_mode,
        "markets": len({row["symbol"] for row in group_rows}), "oos_windows": len(windows),
        "market_window_rows": len(group_rows), "eligible_strategy_market_windows": eligible_strategy_rows,
        "active_market_windows": sum(as_int(row.get("strategy_trades")) > 0 for row in group_rows),
        "no_trade_market_windows": sum(as_int(row.get("strategy_trades")) == 0 for row in group_rows),
        "total_trades": trades,
        "strategy_mean_return_pct": round(statistics.mean(all_returns), 4),
        "strategy_median_return_pct": round(statistics.median(all_returns), 4),
        "strategy_positive_market_windows": sum(value > 0 for value in all_returns),
        "strategy_positive_fraction_pct": round(100 * sum(value > 0 for value in all_returns) / len(all_returns), 2),
        "positive_mean_windows": sum(value > 0 for value in window_means),
        "mean_strategy_mtm_drawdown_pct": round(statistics.mean(all_dds), 4) if all_dds else "",
        "median_strategy_mtm_drawdown_pct": round(statistics.median(all_dds), 4) if all_dds else "",
        "mean_profit_factor_active_5plus_trades": round(statistics.mean(all_pfs), 4) if all_pfs else "",
        "median_profit_factor_active_5plus_trades": round(statistics.median(all_pfs), 4) if all_pfs else "",
        "benchmark_executable_market_windows": len(benchmark_rows),
        "benchmark_mean_return_pct_equal_window": round(statistics.mean(window_benchmark_means), 4) if window_benchmark_means else "",
        "benchmark_median_return_pct_pooled": round(statistics.median(benchmark_returns), 4) if benchmark_returns else "",
        "excess_vs_benchmark_mean_pp_equal_window": round(statistics.mean(window_excess_means), 4) if window_excess_means else "",
        "excess_vs_benchmark_median_pp_pooled": round(statistics.median(excess), 4) if excess else "",
        "strategy_wins_vs_benchmark": wins,
        "strategy_win_fraction_vs_benchmark_pct": round(100 * wins / len(benchmark_rows), 2) if benchmark_rows else "",
        "bootstrap_replicates": replicates, "bootstrap_block_length_windows": block_length,
        "bootstrap_seed": seed,
        "mean_return_ci95_low": quantile(boot_mean_return, 0.025),
        "mean_return_ci95_high": quantile(boot_mean_return, 0.975),
        "median_return_ci95_low": quantile(boot_median_return, 0.025),
        "median_return_ci95_high": quantile(boot_median_return, 0.975),
        "positive_fraction_ci95_low": quantile(boot_positive_fraction, 0.025),
        "positive_fraction_ci95_high": quantile(boot_positive_fraction, 0.975),
        "benchmark_mean_ci95_low": quantile(boot_benchmark_mean, 0.025),
        "benchmark_mean_ci95_high": quantile(boot_benchmark_mean, 0.975),
        "excess_vs_benchmark_mean_ci95_low": quantile(boot_excess_mean, 0.025),
        "excess_vs_benchmark_mean_ci95_high": quantile(boot_excess_mean, 0.975),
        "benchmark_win_fraction_ci95_low": quantile(boot_win_fraction, 0.025),
        "benchmark_win_fraction_ci95_high": quantile(boot_win_fraction, 0.975),
        "evaluation_status": "no_executable_samples" if no_executable else "historical_diagnostic_only",
    }
    return summary, detail


def audit_rows(rows, metadata, replicates=10000, block_length=2, seed=20261029):
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    windows, bounds, balances, costs, symbols, directions = validate_rows(rows, metadata)
    summaries, details = [], []
    index = 0
    for balance in balances:
        for cost in costs:
            for direction_mode in directions:
                subset = [
                    row for row in rows
                    if float(row["start_balance"]) == balance and row["cost_scenario"] == cost
                    and str(row.get("direction_mode", "BOTH")).upper() == direction_mode
                ]
                summary, detail = summarize_group(
                    subset, windows, bounds, balance, cost, direction_mode,
                    replicates, block_length, seed + index)
                summaries.append(summary)
                details.extend(detail)
                index += 1
    return summaries, details


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not rows:
        raise ValueError("refusing to write an empty audit")
    # Some diagnostic groups (for example, no-trade balances) have no bootstrap
    # intervals, while later groups do. Build a stable union of keys across all rows.
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="donchian-benchmark-results/donchian_benchmark_pairs.csv")
    parser.add_argument("--metadata", default="donchian-benchmark-results/run_metadata.json")
    parser.add_argument("--out", default="donchian-benchmark-results/paired_bootstrap_summary.csv")
    parser.add_argument("--window-out", default="donchian-benchmark-results/paired_bootstrap_window_detail.csv")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261029)
    args = parser.parse_args()
    rows = read_csv(args.input)
    with open(args.metadata, encoding="utf-8") as handle:
        metadata = json.load(handle)
    summary, detail = audit_rows(rows, metadata, args.replicates, args.block_length, args.seed)
    write_csv(args.out, summary)
    write_csv(args.window_out, detail)
    print(f"Bootstrap diagnostic complete: rows={len(rows)} groups={len(summary)} "
          f"replicates={args.replicates} block_length={args.block_length}; historical only.")
    for row in summary:
        print(f"balance={row['start_balance']:g} cost={row['cost_scenario']} direction={row['direction_mode']}: "
              f"mean={row['strategy_mean_return_pct']:+.3f}% "
              f"CI95=[{row['mean_return_ci95_low']},{row['mean_return_ci95_high']}] "
              f"excess_vs_buy_hold={row['excess_vs_benchmark_mean_pp_equal_window']}pp "
              f"CI95=[{row['excess_vs_benchmark_mean_ci95_low']},{row['excess_vs_benchmark_mean_ci95_high']}] "
              f"status={row['evaluation_status']}")


if __name__ == "__main__":
    main()
