"""Paired two-way market/time bootstrap for the predeclared ADX(14)>=20 Donchian experiment."""
import argparse
import random
import statistics

from paired_bootstrap_audit import as_bool, as_float, as_int, moving_block_sample, quantile, read_csv, write_csv


BASELINE = "donchian_20_10"
CANDIDATE = "donchian_20_10_adx14_ge20"


def validate_rows(rows, metadata):
    if not rows:
        raise ValueError("ADX-filter CSV has no rows")
    symbols = list(metadata.get("markets_tested", []))
    strategies = list(metadata.get("strategy_names", []))
    if not symbols or set(strategies) != {BASELINE, CANDIDATE}:
        raise ValueError("metadata must declare the exact two-strategy cohort")
    expected_end = int(metadata["shared_end_timestamp"])
    expected_symbols = set(symbols)
    actual_symbols = {str(row.get("symbol", "")).upper() for row in rows}
    if actual_symbols != expected_symbols:
        raise ValueError(f"symbol cohort mismatch: metadata={sorted(expected_symbols)} csv={sorted(actual_symbols)}")

    bounds = {}
    for row in rows:
        name = str(row["window"])
        current = (as_int(row.get("window_start_utc")), as_int(row.get("window_end_utc")))
        if name in bounds and bounds[name] != current:
            raise ValueError(f"inconsistent bounds for {name}")
        bounds[name] = current
        if as_int(row.get("shared_end_timestamp")) != expected_end:
            raise ValueError("rows do not share the metadata end timestamp")
    windows = sorted(bounds, key=lambda name: bounds[name][0])
    if len(windows) != int(metadata["oos_window_count"]):
        raise ValueError("OOS window count does not match metadata")
    width = int(metadata["oos_window_days"]) * 86400
    for index, name in enumerate(windows):
        start, end = bounds[name]
        if end - start != width:
            raise ValueError(f"unexpected OOS window width for {name}")
        if index and bounds[windows[index - 1]][1] != start:
            raise ValueError("OOS windows are not contiguous")
    if bounds[windows[-1]][1] != expected_end:
        raise ValueError("last OOS window does not end at shared timestamp")

    balances = sorted(float(value) for value in metadata.get("balances", []))
    costs = sorted(item["name"] for item in metadata.get("cost_scenarios", []))
    actual_strategies = {str(row.get("strategy", "")) for row in rows}
    if actual_strategies != set(strategies):
        raise ValueError("strategy variants do not match metadata")
    if sorted({float(row["start_balance"]) for row in rows}) != balances:
        raise ValueError("balance grid does not match metadata")
    if sorted({str(row["cost_scenario"]) for row in rows}) != costs:
        raise ValueError("cost scenarios do not match metadata")

    seen = set()
    benchmark_by_pair = {}
    for row in rows:
        key = (
            str(row["symbol"]).upper(), str(row["window"]), float(row["start_balance"]),
            str(row["cost_scenario"]), str(row["strategy"]),
        )
        if key in seen:
            raise ValueError(f"duplicate result row: {key}")
        seen.add(key)
        pair_key = key[:-1]
        benchmark = (
            str(row.get("buy_hold_return_pct", "")),
            str(row.get("buy_hold_min_quote_check_pass", "")).lower(),
            str(row.get("buy_hold_status", "")),
        )
        if pair_key in benchmark_by_pair and benchmark_by_pair[pair_key] != benchmark:
            raise ValueError(f"buy-and-hold benchmark differs between paired variants: {pair_key}")
        benchmark_by_pair[pair_key] = benchmark

    expected = len(symbols) * len(windows) * len(balances) * len(costs) * len(strategies)
    if len(rows) != expected:
        raise ValueError(f"incomplete result grid: got {len(rows)} rows, expected {expected}")
    return symbols, windows, bounds, balances, costs


def _stats(rows):
    returns = [as_float(row.get("return_pct"), 0.0) for row in rows]
    drawdowns = [as_float(row.get("max_drawdown_mtm_pct"), 0.0) for row in rows]
    eligible = [
        row for row in rows
        if row.get("strategy_status") == "tested" and as_int(row.get("trades")) >= 5
    ]
    pfs = [as_float(row.get("profit_factor")) for row in eligible]
    pfs = [value for value in pfs if value is not None]
    executable_benchmarks = [
        row for row in rows
        if as_bool(row.get("buy_hold_executable"))
        and as_float(row.get("buy_hold_return_pct")) is not None
    ]
    excess = [
        as_float(row.get("excess_vs_buy_hold_pp")) for row in executable_benchmarks
    ]
    excess = [value for value in excess if value is not None]
    return {
        "rows": len(rows),
        "mean_return_pct": statistics.mean(returns) if returns else 0.0,
        "median_return_pct": statistics.median(returns) if returns else 0.0,
        "positive_market_windows": sum(value > 0 for value in returns),
        "positive_fraction_pct": 100 * sum(value > 0 for value in returns) / len(returns) if returns else 0.0,
        "mean_mtm_drawdown_pct": statistics.mean(drawdowns) if drawdowns else 0.0,
        "median_mtm_drawdown_pct": statistics.median(drawdowns) if drawdowns else 0.0,
        "mean_profit_factor_active_5plus": statistics.mean(pfs) if pfs else None,
        "median_profit_factor_active_5plus": statistics.median(pfs) if pfs else None,
        "total_trades": sum(as_int(row.get("trades")) for row in rows),
        "active_market_windows": sum(as_int(row.get("trades")) > 0 for row in rows),
        "eligible_market_windows": len(eligible),
        "no_trade_market_windows": sum(as_int(row.get("trades")) == 0 for row in rows),
        "funding_coverage_pct": statistics.mean(
            as_float(row.get("funding_coverage_pct"), 0.0) for row in rows
        ) if rows else 0.0,
        "funding_events_charged": sum(as_int(row.get("funding_events_charged")) for row in rows),
        "funding_net_cost_usd": sum(as_float(row.get("funding_net_cost_usd"), 0.0) for row in rows),
        "min_notional_rejections": sum(as_int(row.get("min_notional_rejections")) for row in rows),
        "buy_hold_executable_market_windows": len(executable_benchmarks),
        "buy_hold_mean_return_pct": (
            statistics.mean(as_float(row.get("buy_hold_return_pct"), 0.0) for row in executable_benchmarks)
            if executable_benchmarks else None
        ),
        "mean_excess_vs_buy_hold_pp": statistics.mean(excess) if excess else None,
    }


def audit(rows, metadata, replicates=10000, block_length=2, seed=20261031):
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    symbols, windows, bounds, balances, costs = validate_rows(rows, metadata)
    index = {
        (str(row["symbol"]).upper(), str(row["window"]), float(row["start_balance"]),
         str(row["cost_scenario"]), str(row["strategy"])): row
        for row in rows
    }
    summaries, leave_one_out, window_details = [], [], []
    group_index = 0
    for balance in balances:
        for cost in costs:
            base_rows = [index[(symbol, window, balance, cost, BASELINE)]
                         for symbol in symbols for window in windows]
            candidate_rows = [index[(symbol, window, balance, cost, CANDIDATE)]
                              for symbol in symbols for window in windows]
            base_stats = _stats(base_rows)
            candidate_stats = _stats(candidate_rows)
            deltas = [
                as_float(candidate.get("return_pct"), 0.0) - as_float(base.get("return_pct"), 0.0)
                for base, candidate in zip(base_rows, candidate_rows)
            ]
            executable_pairs = [
                (base, candidate) for base, candidate in zip(base_rows, candidate_rows)
                if as_bool(candidate.get("buy_hold_executable"))
                and as_float(candidate.get("buy_hold_return_pct")) is not None
            ]
            candidate_excess = [
                as_float(candidate.get("return_pct"), 0.0) - as_float(candidate.get("buy_hold_return_pct"), 0.0)
                for _, candidate in executable_pairs
            ]

            rng = random.Random(seed + group_index)
            boot_candidate_mean, boot_baseline_mean, boot_delta_mean = [], [], []
            boot_candidate_median, boot_positive_fraction, boot_buy_hold_excess = [], [], []
            for _ in range(replicates):
                sampled_symbols = [symbols[rng.randrange(len(symbols))] for _ in symbols]
                sampled_windows = moving_block_sample(windows, block_length, rng)
                sampled_base = [
                    index[(symbol, window, balance, cost, BASELINE)]
                    for symbol in sampled_symbols for window in sampled_windows
                ]
                sampled_candidate = [
                    index[(symbol, window, balance, cost, CANDIDATE)]
                    for symbol in sampled_symbols for window in sampled_windows
                ]
                base_returns = [as_float(row.get("return_pct"), 0.0) for row in sampled_base]
                candidate_returns = [as_float(row.get("return_pct"), 0.0) for row in sampled_candidate]
                boot_baseline_mean.append(statistics.mean(base_returns))
                boot_candidate_mean.append(statistics.mean(candidate_returns))
                boot_candidate_median.append(statistics.median(candidate_returns))
                boot_delta_mean.append(statistics.mean(
                    candidate_value - base_value
                    for candidate_value, base_value in zip(candidate_returns, base_returns)
                ))
                boot_positive_fraction.append(
                    100 * sum(value > 0 for value in candidate_returns) / len(candidate_returns)
                )
                excess_values = [
                    as_float(candidate.get("return_pct"), 0.0)
                    - as_float(candidate.get("buy_hold_return_pct"), 0.0)
                    for candidate in sampled_candidate
                    if as_bool(candidate.get("buy_hold_executable"))
                    and as_float(candidate.get("buy_hold_return_pct")) is not None
                ]
                if excess_values:
                    boot_buy_hold_excess.append(statistics.mean(excess_values))

            summary = {
                "start_balance": balance,
                "cost_scenario": cost,
                "markets": len(symbols),
                "oos_windows": len(windows),
                "market_window_rows_per_strategy": len(symbols) * len(windows),
                "candidate_strategy": CANDIDATE,
                "baseline_strategy": BASELINE,
                "candidate_mean_return_pct": round(candidate_stats["mean_return_pct"], 4),
                "candidate_median_return_pct": round(candidate_stats["median_return_pct"], 4),
                "candidate_positive_market_windows": candidate_stats["positive_market_windows"],
                "candidate_positive_fraction_pct": round(candidate_stats["positive_fraction_pct"], 2),
                "candidate_mean_mtm_drawdown_pct": round(candidate_stats["mean_mtm_drawdown_pct"], 4),
                "candidate_median_mtm_drawdown_pct": round(candidate_stats["median_mtm_drawdown_pct"], 4),
                "candidate_mean_profit_factor_active_5plus": (
                    round(candidate_stats["mean_profit_factor_active_5plus"], 4)
                    if candidate_stats["mean_profit_factor_active_5plus"] is not None else ""
                ),
                "candidate_total_trades": candidate_stats["total_trades"],
                "candidate_active_market_windows": candidate_stats["active_market_windows"],
                "candidate_eligible_market_windows": candidate_stats["eligible_market_windows"],
                "candidate_no_trade_market_windows": candidate_stats["no_trade_market_windows"],
                "baseline_mean_return_pct": round(base_stats["mean_return_pct"], 4),
                "baseline_median_return_pct": round(base_stats["median_return_pct"], 4),
                "baseline_positive_fraction_pct": round(base_stats["positive_fraction_pct"], 2),
                "baseline_mean_mtm_drawdown_pct": round(base_stats["mean_mtm_drawdown_pct"], 4),
                "baseline_mean_profit_factor_active_5plus": (
                    round(base_stats["mean_profit_factor_active_5plus"], 4)
                    if base_stats["mean_profit_factor_active_5plus"] is not None else ""
                ),
                "baseline_total_trades": base_stats["total_trades"],
                "candidate_minus_baseline_mean_pp": round(statistics.mean(deltas), 4),
                "candidate_minus_baseline_median_pp": round(statistics.median(deltas), 4),
                "candidate_beats_baseline_market_windows": sum(value > 0 for value in deltas),
                "candidate_beats_baseline_fraction_pct": round(100 * sum(value > 0 for value in deltas) / len(deltas), 2),
                "cash_baseline_return_pct": 0.0,
                "candidate_mean_excess_vs_cash_pp": round(candidate_stats["mean_return_pct"], 4),
                "buy_hold_executable_market_windows": candidate_stats["buy_hold_executable_market_windows"],
                "buy_hold_mean_return_pct": (
                    round(candidate_stats["buy_hold_mean_return_pct"], 4)
                    if candidate_stats["buy_hold_mean_return_pct"] is not None else ""
                ),
                "candidate_mean_excess_vs_buy_hold_pp": (
                    round(candidate_stats["mean_excess_vs_buy_hold_pp"], 4)
                    if candidate_stats["mean_excess_vs_buy_hold_pp"] is not None else ""
                ),
                "candidate_funding_coverage_pct": round(candidate_stats["funding_coverage_pct"], 3),
                "candidate_funding_events_charged": candidate_stats["funding_events_charged"],
                "candidate_funding_net_cost_usd": round(candidate_stats["funding_net_cost_usd"], 6),
                "candidate_min_notional_rejections": candidate_stats["min_notional_rejections"],
                "bootstrap_replicates": replicates,
                "bootstrap_block_length_windows": block_length,
                "bootstrap_seed": seed + group_index,
                "bootstrap_method": "two_way_market_resampling_plus_contiguous_window_blocks",
                "candidate_mean_return_ci95_low": quantile(boot_candidate_mean, 0.025),
                "candidate_mean_return_ci95_high": quantile(boot_candidate_mean, 0.975),
                "baseline_mean_return_ci95_low": quantile(boot_baseline_mean, 0.025),
                "baseline_mean_return_ci95_high": quantile(boot_baseline_mean, 0.975),
                "candidate_minus_baseline_mean_ci95_low": quantile(boot_delta_mean, 0.025),
                "candidate_minus_baseline_mean_ci95_high": quantile(boot_delta_mean, 0.975),
                "candidate_median_return_ci95_low": quantile(boot_candidate_median, 0.025),
                "candidate_median_return_ci95_high": quantile(boot_candidate_median, 0.975),
                "candidate_positive_fraction_ci95_low": quantile(boot_positive_fraction, 0.025),
                "candidate_positive_fraction_ci95_high": quantile(boot_positive_fraction, 0.975),
                "candidate_excess_vs_buy_hold_ci95_low": quantile(boot_buy_hold_excess, 0.025),
                "candidate_excess_vs_buy_hold_ci95_high": quantile(boot_buy_hold_excess, 0.975),
                "evaluation_status": (
                    "no_executable_samples" if candidate_stats["eligible_market_windows"] == 0
                    else "historical_diagnostic_only"
                ),
            }
            summaries.append(summary)

            for omitted in ["NONE"] + symbols:
                included = [symbol for symbol in symbols if symbol != omitted] if omitted != "NONE" else symbols
                omitted_base = [
                    index[(symbol, window, balance, cost, BASELINE)]
                    for symbol in included for window in windows
                ]
                omitted_candidate = [
                    index[(symbol, window, balance, cost, CANDIDATE)]
                    for symbol in included for window in windows
                ]
                omitted_deltas = [
                    as_float(candidate.get("return_pct"), 0.0) - as_float(base.get("return_pct"), 0.0)
                    for base, candidate in zip(omitted_base, omitted_candidate)
                ]
                omitted_stats = _stats(omitted_candidate)
                leave_one_out.append({
                    "start_balance": balance,
                    "cost_scenario": cost,
                    "omitted_symbol": omitted,
                    "markets_included": len(included),
                    "oos_windows": len(windows),
                    "candidate_mean_return_pct": round(omitted_stats["mean_return_pct"], 4),
                    "candidate_median_return_pct": round(omitted_stats["median_return_pct"], 4),
                    "candidate_positive_fraction_pct": round(omitted_stats["positive_fraction_pct"], 2),
                    "candidate_mean_mtm_drawdown_pct": round(omitted_stats["mean_mtm_drawdown_pct"], 4),
                    "candidate_mean_profit_factor_active_5plus": (
                        round(omitted_stats["mean_profit_factor_active_5plus"], 4)
                        if omitted_stats["mean_profit_factor_active_5plus"] is not None else ""
                    ),
                    "candidate_total_trades": omitted_stats["total_trades"],
                    "baseline_mean_return_pct": round(_stats(omitted_base)["mean_return_pct"], 4),
                    "candidate_minus_baseline_mean_pp": round(statistics.mean(omitted_deltas), 4),
                    "candidate_eligible_market_windows": omitted_stats["eligible_market_windows"],
                    "candidate_funding_coverage_pct": round(omitted_stats["funding_coverage_pct"], 3),
                })

            for window in windows:
                window_base = [index[(symbol, window, balance, cost, BASELINE)] for symbol in symbols]
                window_candidate = [index[(symbol, window, balance, cost, CANDIDATE)] for symbol in symbols]
                window_deltas = [
                    as_float(candidate.get("return_pct"), 0.0) - as_float(base.get("return_pct"), 0.0)
                    for base, candidate in zip(window_base, window_candidate)
                ]
                window_stats = _stats(window_candidate)
                window_details.append({
                    "start_balance": balance,
                    "cost_scenario": cost,
                    "window": window,
                    "window_start_utc": bounds[window][0],
                    "window_end_utc": bounds[window][1],
                    "markets": len(symbols),
                    "candidate_mean_return_pct": round(window_stats["mean_return_pct"], 4),
                    "candidate_median_return_pct": round(window_stats["median_return_pct"], 4),
                    "candidate_positive_fraction_pct": round(window_stats["positive_fraction_pct"], 2),
                    "candidate_mean_mtm_drawdown_pct": round(window_stats["mean_mtm_drawdown_pct"], 4),
                    "candidate_total_trades": window_stats["total_trades"],
                    "baseline_mean_return_pct": round(_stats(window_base)["mean_return_pct"], 4),
                    "candidate_minus_baseline_mean_pp": round(statistics.mean(window_deltas), 4),
                    "candidate_funding_coverage_pct": round(window_stats["funding_coverage_pct"], 3),
                    "buy_hold_mean_return_pct": (
                        round(window_stats["buy_hold_mean_return_pct"], 4)
                        if window_stats["buy_hold_mean_return_pct"] is not None else ""
                    ),
                })
            group_index += 1
    return summaries, leave_one_out, window_details


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="adx-filtered-donchian-results/adx_filter_pairs.csv")
    parser.add_argument("--metadata", default="adx-filtered-donchian-results/run_metadata.json")
    parser.add_argument("--out", default="adx-filtered-donchian-results/adx_filter_bootstrap_summary.csv")
    parser.add_argument("--market-out", default="adx-filtered-donchian-results/adx_filter_leave_one_market_out.csv")
    parser.add_argument("--window-out", default="adx-filtered-donchian-results/adx_filter_window_detail.csv")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261031)
    args = parser.parse_args()
    rows = read_csv(args.input)
    import json
    with open(args.metadata, encoding="utf-8") as handle:
        metadata = json.load(handle)
    summary, lomo, windows = audit(rows, metadata, args.replicates, args.block_length, args.seed)
    write_csv(args.out, summary)
    write_csv(args.market_out, lomo)
    write_csv(args.window_out, windows)
    print(
        f"Paired ADX-filter bootstrap complete: rows={len(rows)} groups={len(summary)} "
        f"leave_one_market_out_rows={len(lomo)} replicates={args.replicates}; historical diagnostic only."
    )
    for row in summary:
        if row["start_balance"] in (50.0, 100.0) and row["cost_scenario"] == "base_taker":
            print(
                f"balance={row['start_balance']:g} {row['cost_scenario']}: "
                f"candidate={row['candidate_mean_return_pct']:+.3f}% "
                f"CI95=[{row['candidate_mean_return_ci95_low']},{row['candidate_mean_return_ci95_high']}] "
                f"delta_vs_Donchian={row['candidate_minus_baseline_mean_pp']:+.3f}pp "
                f"CI95=[{row['candidate_minus_baseline_mean_ci95_low']},{row['candidate_minus_baseline_mean_ci95_high']}]"
            )


if __name__ == "__main__":
    main()
