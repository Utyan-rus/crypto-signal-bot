"""Fixed-parameter walk-forward audit of hourly Lighter strategy candidates.

Paper-only: public candles and settled funding, no order placement. Uses causal
indicators and eight non-overlapping 30-day OOS windows. Funding coverage is gated
before any performance metrics or bootstrap intervals are computed.
"""
import argparse
import csv
import json
import os
import statistics
import time

import hourly_candidate_research as hourly
import funding_aware_research as funding_research
from backtest import fetch_candles, get_markets
from two_way_bootstrap_audit import leave_one_market_out, metrics as paired_metrics, two_way_bootstrap

STRATEGIES = ("donchian_20_10", "ema_20_50_cross", "ema_trend_pullback",
              "mean_reversion_bollinger", "adaptive_regime_switch",
              "donchian_long_regime", "donchian_long_regime_vol")
DIRECTION_MODES = ("BOTH", "LONG", "SHORT")
WINDOW_DAYS = 30
OOS_WINDOWS = 8
CONTEXT_DAYS = 365
MIN_WARMUP_DAYS = 125
COSTS = (
    ("zero_fee_base_slippage", 0.0, 0.0005),
    ("assumed_taker_base_slippage", 0.0005, 0.0005),
    ("assumed_taker_double_slippage", 0.0005, 0.0010),
    ("double_fee_double_slippage", 0.0010, 0.0010),
)
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_BLOCK_LENGTH = 2
BOOTSTRAP_SEED = 20261103
MIN_FUNDING_COVERAGE_PCT = 99.0


def floor_to_hour(timestamp):
    """Use the last fully closed hourly boundary; never include the live partial candle."""
    return (int(timestamp) // 3600) * 3600


def make_windows(end_ts, days=WINDOW_DAYS, count=OOS_WINDOWS):
    """Return chronological, non-overlapping OOS windows ending at end_ts."""
    if count < 1:
        raise ValueError("count must be >= 1")
    seconds = days * 86400
    first = end_ts - count * seconds
    return [
        (f"oos_{i + 1}", first + i * seconds, first + (i + 1) * seconds)
        for i in range(count)
    ]


def validate_window_geometry(total_days, oos_windows):
    """Require the fixed 125-day pre-OOS warmup plus every requested OOS window."""
    if oos_windows < 1:
        raise ValueError("oos_windows must be >= 1")
    warmup_days = int(total_days) - int(oos_windows) * WINDOW_DAYS
    if warmup_days < MIN_WARMUP_DAYS:
        required = MIN_WARMUP_DAYS + int(oos_windows) * WINDOW_DAYS
        raise ValueError(
            f"Need at least {MIN_WARMUP_DAYS} warmup days plus "
            f"{oos_windows} OOS windows of {WINDOW_DAYS} days ({required} total days)"
        )
    return warmup_days


def buy_hold_baseline(candles, window_start, window_end, start_balance,
                      fee_rate, slippage_rate, min_notional, funding=None,
                      direction="LONG"):
    """Costed 1x directional perp hold baseline with signed historical funding."""
    direction = str(direction).upper()
    if direction not in ("LONG", "SHORT"):
        raise ValueError("direction must be LONG or SHORT")
    direction_sign = 1.0 if direction == "LONG" else -1.0
    funding = funding or {}
    if start_balance < min_notional or start_balance <= 0:
        return {"executable": False, "status": "not_executable_below_min_notional",
                "return_pct": None, "funding_net_cost_usd": 0.0,
                "funding_events_charged": 0}
    bars = [bar for bar in candles if window_start <= bar["t"] < window_end]
    if len(bars) < 2:
        return {"executable": False, "status": "insufficient_window_data",
                "return_pct": None, "funding_net_cost_usd": 0.0,
                "funding_events_charged": 0}
    entry_time = int(bars[0]["t"])
    entry_slippage = 1 + slippage_rate if direction == "LONG" else 1 - slippage_rate
    exit_slippage = 1 - slippage_rate if direction == "LONG" else 1 + slippage_rate
    entry = bars[0]["o"] * entry_slippage
    if entry <= 0:
        return {"executable": False, "status": "invalid_entry_price",
                "return_pct": None, "funding_net_cost_usd": 0.0,
                "funding_events_charged": 0}
    qty = start_balance / entry
    if qty * entry < min_notional:
        return {"executable": False, "status": "not_executable_below_min_notional",
                "return_pct": None, "funding_net_cost_usd": 0.0,
                "funding_events_charged": 0}
    close_by_time = {int(bar["t"]): float(bar["c"]) for bar in bars}
    funding_cost = 0.0
    funding_events = 0
    for timestamp, rate in sorted(funding.items()):
        timestamp = int(timestamp)
        if entry_time < timestamp < window_end:
            mark = close_by_time.get(timestamp)
            if mark is None:
                raise ValueError(f"funding timestamp {timestamp} has no matching hourly candle")
            # Positive signed rate means longs pay and shorts receive.
            funding_cost += direction_sign * qty * mark * float(rate)
            funding_events += 1
    exit_price = bars[-1]["c"] * exit_slippage
    gross = direction_sign * qty * (exit_price - entry)
    fees = abs(qty * entry) * fee_rate + abs(qty * exit_price) * fee_rate
    pnl = gross - fees - funding_cost
    return {"executable": True,
            "status": f"executable_1x_{direction.lower()}_with_funding",
            "return_pct": round(100 * pnl / start_balance, 6),
            "funding_net_cost_usd": round(funding_cost, 8),
            "funding_events_charged": funding_events}


def with_long_buy_hold_benchmark(rows):
    """Copy rows while mapping the 1x long hold into generic benchmark fields."""
    mapped = []
    for row in rows:
        item = dict(row)
        item["benchmark_executable"] = row.get(
            "buy_hold_executable", row.get("benchmark_executable", False)
        )
        item["benchmark_return_pct"] = row.get(
            "buy_hold_return_pct", row.get("benchmark_return_pct")
        )
        item["observed_excess_return_vs_benchmark_pp"] = row.get(
            "excess_vs_buy_hold_pp", row.get("observed_excess_return_vs_benchmark_pp")
        )
        item["strategy_beats_benchmark_observed"] = row.get(
            "strategy_beats_buy_hold_observed",
            row.get("strategy_beats_benchmark_observed", False),
        )
        item["paired_evaluable"] = row.get(
            "long_buy_hold_paired_evaluable", row.get("paired_evaluable", False)
        )
        mapped.append(item)
    return mapped



def funding_coverage(funding, window_start, window_end):
    """Measure exact hourly timestamp coverage; missing rows are never treated as zero."""
    first_expected = ((int(window_start) + 3599) // 3600) * 3600
    expected = set(range(first_expected, int(window_end), 3600))
    observed = {
        int(timestamp) for timestamp in funding
        if int(window_start) <= int(timestamp) < int(window_end)
    }
    covered = observed & expected
    missing = expected - covered
    unexpected = observed - expected
    expected_count = len(expected)
    return {
        "funding_rows": len(covered),
        "expected_funding_hours": expected_count,
        "funding_coverage_pct": round(100 * len(covered) / expected_count, 3) if expected_count else 0.0,
        "missing_funding_hours": len(missing),
        "unexpected_funding_rows": len(unexpected),
    }


def funding_coverage_is_acceptable(coverage):
    """Fail closed on material missing or misaligned hourly funding rows."""
    return (
        coverage.get("funding_coverage_pct", 0) >= MIN_FUNDING_COVERAGE_PCT
        and coverage.get("unexpected_funding_rows", 0) == 0
    )


def capital_feasibility_audit(balance, risk_fraction, minimum_stop_fraction,
                              max_leverage, min_notional):
    """Upper-bound the risk-sized notional under the simulator's minimum stop distance."""
    if (balance <= 0 or risk_fraction <= 0 or minimum_stop_fraction <= 0
            or max_leverage <= 0 or min_notional <= 0):
        raise ValueError("capital feasibility inputs must be positive")
    risk_notional = balance * risk_fraction / minimum_stop_fraction
    leverage_notional = balance * max_leverage
    maximum_allowed = min(risk_notional, leverage_notional)
    minimum_balance = max(
        min_notional * minimum_stop_fraction / risk_fraction,
        min_notional / max_leverage,
    )
    return {
        "balance_usd": round(balance, 6),
        "maximum_risk_sized_notional_usd": round(risk_notional, 6),
        "maximum_leverage_notional_usd": round(leverage_notional, 6),
        "maximum_allowed_notional_usd": round(maximum_allowed, 6),
        "minimum_quote_notional_assumption_usd": round(min_notional, 6),
        "minimum_balance_usd_at_minimum_stop": round(minimum_balance, 6),
        "minimum_notional_reachable_at_minimum_stop": maximum_allowed >= min_notional,
        "upper_bound_only": True,
    }


def simulate_funding_audited(symbol, candles, strategy, test_start, test_end,
                             funding, start_balance, fee_rate, slippage_rate,
                             direction_mode="BOTH"):
    """Run the funding-aware simulator and count every signal rejected by risk/min-notional sizing."""
    rejected_sizing_attempts = 0
    original_size_position = funding_research.size_position

    def tracked_size_position(balance, entry, stop_dist):
        nonlocal rejected_sizing_attempts
        quantity = original_size_position(balance, entry, stop_dist)
        if quantity <= 0 and balance > 0 and entry > 0 and stop_dist > 0:
            rejected_sizing_attempts += 1
        return quantity

    funding_research.size_position = tracked_size_position
    try:
        row = funding_research.simulate_with_funding(
            symbol, candles, strategy, test_start, test_end, funding,
            start_balance=start_balance, fee_rate=fee_rate,
            slippage_rate=slippage_rate, direction_mode=direction_mode
        )
    finally:
        funding_research.size_position = original_size_position
    row["min_notional_rejections"] = (
        int(row.get("min_notional_rejections", 0) or 0) + rejected_sizing_attempts
    )
    return row


def summarize(rows):
    """Summarize fixed strategy/cost/direction groups against cash and both hold baselines."""
    out = []
    keys = sorted({
        (r["strategy"], r["cost_scenario"], str(r.get("direction_mode", "BOTH")).upper())
        for r in rows if r.get("cost_scenario") is not None
    })
    for strategy, cost, direction in keys:
        subset = [r for r in rows if r["strategy"] == strategy
                  and r["cost_scenario"] == cost
                  and str(r.get("direction_mode", "BOTH")).upper() == direction]
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

        long_benchmark = [
            r for r in subset
            if r.get("buy_hold_executable", r.get("benchmark_executable"))
            and r.get("buy_hold_return_pct", r.get("benchmark_return_pct")) not in ("", None)
        ]
        long_benchmark_returns = [
            float(r.get("buy_hold_return_pct", r.get("benchmark_return_pct")))
            for r in long_benchmark
        ]
        long_paired = [
            r for r in subset
            if r.get("long_buy_hold_paired_evaluable", r.get("paired_evaluable"))
            and r.get("excess_vs_buy_hold_pp",
                      r.get("observed_excess_return_vs_benchmark_pp")) not in ("", None)
        ]
        long_excess = [
            float(r.get("excess_vs_buy_hold_pp",
                        r.get("observed_excess_return_vs_benchmark_pp")))
            for r in long_paired
        ]
        long_wins = [
            bool(r.get("strategy_beats_buy_hold_observed",
                       r.get("strategy_beats_benchmark_observed", False)))
            for r in long_paired
        ]

        directional_benchmark = [
            r for r in subset
            if r.get("benchmark_executable", r.get("buy_hold_executable"))
            and r.get("benchmark_return_pct", r.get("buy_hold_return_pct")) not in ("", None)
        ]
        directional_benchmark_returns = [
            float(r.get("benchmark_return_pct", r.get("buy_hold_return_pct")))
            for r in directional_benchmark
        ]
        directional_paired = [
            r for r in subset
            if r.get("paired_evaluable", r.get("long_buy_hold_paired_evaluable"))
            and r.get("observed_excess_return_vs_benchmark_pp",
                      r.get("excess_vs_buy_hold_pp")) not in ("", None)
        ]
        directional_excess = [
            float(r.get("observed_excess_return_vs_benchmark_pp",
                        r.get("excess_vs_buy_hold_pp")))
            for r in directional_paired
        ]
        directional_wins = [
            bool(r.get("strategy_beats_benchmark_observed",
                       r.get("strategy_beats_buy_hold_observed", False)))
            for r in directional_paired
        ]

        funding_coverages = [float(r.get("funding_coverage_pct", 0) or 0) for r in subset]
        strategy_funding_costs = [float(r.get("funding_net_cost_usd", 0) or 0) for r in subset]
        long_funding_costs = [float(r.get("buy_hold_funding_net_cost_usd", 0) or 0) for r in subset]
        directional_funding_costs = [
            float(r.get("benchmark_funding_net_cost_usd",
                        r.get("buy_hold_funding_net_cost_usd", 0)) or 0)
            for r in subset
        ]
        long_funding_events = [int(r.get("buy_hold_funding_events_charged", 0) or 0) for r in subset]
        directional_funding_events = [
            int(r.get("benchmark_funding_events_charged",
                      r.get("buy_hold_funding_events_charged", 0)) or 0)
            for r in subset
        ]
        strategy_funding_events = [int(r.get("funding_events_charged", 0) or 0) for r in subset]
        out.append({
            "strategy": strategy,
            "cost_scenario": cost,
            "direction_mode": direction,
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
            "buy_hold_executable_market_windows": len(long_benchmark),
            "buy_hold_mean_return_pct": round(statistics.mean(long_benchmark_returns), 3) if long_benchmark_returns else "",
            "mean_excess_vs_buy_hold_pp": round(statistics.mean(long_excess), 3) if long_excess else "",
            "strategy_win_fraction_vs_buy_hold_pct": round(
                100 * sum(long_wins) / len(long_wins), 1
            ) if long_wins else "",
            "directional_benchmark_executable_market_windows": len(directional_benchmark),
            "directional_benchmark_mean_return_pct": (
                round(statistics.mean(directional_benchmark_returns), 3)
                if directional_benchmark_returns else ""
            ),
            "mean_excess_vs_directional_benchmark_pp": (
                round(statistics.mean(directional_excess), 3) if directional_excess else ""
            ),
            "strategy_win_fraction_vs_directional_benchmark_pct": round(
                100 * sum(directional_wins) / len(directional_wins), 1
            ) if directional_wins else "",
            "mean_mtm_drawdown_pct": round(statistics.mean(drawdowns), 3),
            "total_trades": sum(trade_counts),
            "pf_eligible_windows": len(pf),
            "mean_profit_factor": round(statistics.mean(pf), 3) if pf else "",
            "min_notional_rejections": sum(int(r.get("min_notional_rejections", 0) or 0) for r in subset),
            "historical_funding_coverage_pct": round(statistics.mean(funding_coverages), 3) if funding_coverages else 0.0,
            "minimum_funding_coverage_pct": round(min(funding_coverages), 3) if funding_coverages else 0.0,
            "total_strategy_funding_net_cost_usd": round(sum(strategy_funding_costs), 8),
            "total_buy_hold_funding_net_cost_usd": round(sum(long_funding_costs), 8),
            "total_directional_benchmark_funding_net_cost_usd": round(sum(directional_funding_costs), 8),
            "total_buy_hold_funding_events": sum(long_funding_events),
            "total_directional_benchmark_funding_events": sum(directional_funding_events),
            "total_strategy_funding_events": sum(strategy_funding_events),
        })
    return out



def bootstrap_summaries(rows, windows, symbols, replicates=BOOTSTRAP_REPLICATES,
                        block_length=BOOTSTRAP_BLOCK_LENGTH, seed=BOOTSTRAP_SEED):
    """Two-way market/time bootstrap for directional and 1x-long benchmark comparisons."""
    output = []
    keys = sorted({
        (r["strategy"], r["cost_scenario"], str(r.get("direction_mode", "BOTH")).upper())
        for r in rows
        if r.get("strategy") in STRATEGIES and r.get("cost_scenario") is not None
    })
    for index, (strategy, cost, direction) in enumerate(keys):
        group = [r for r in rows if r.get("strategy") == strategy
                 and r.get("cost_scenario") == cost
                 and str(r.get("direction_mode", "BOTH")).upper() == direction]
        long_group = with_long_buy_hold_benchmark(group)
        point = paired_metrics(group)
        long_point = paired_metrics(long_group)
        run_seed = seed + index
        try:
            intervals = two_way_bootstrap(
                group, windows, symbols, replicates=replicates,
                block_length=block_length, seed=run_seed
            )
            status, error = "ok", ""
        except ValueError as exc:
            intervals, status, error = {}, "incomplete_grid", str(exc)
        try:
            long_intervals = two_way_bootstrap(
                long_group, windows, symbols, replicates=replicates,
                block_length=block_length, seed=run_seed
            )
            long_status, long_error = "ok", ""
        except ValueError as exc:
            long_intervals, long_status, long_error = {}, "incomplete_grid", str(exc)

        long_interval_aliases = {}
        for source_key, target_key in (
            ("two_way_benchmark_mean_return_pct_ci95_low",
             "two_way_long_buy_hold_mean_return_pct_ci95_low"),
            ("two_way_benchmark_mean_return_pct_ci95_high",
             "two_way_long_buy_hold_mean_return_pct_ci95_high"),
            ("two_way_mean_excess_vs_benchmark_pp_ci95_low",
             "two_way_mean_excess_vs_long_buy_hold_pp_ci95_low"),
            ("two_way_mean_excess_vs_benchmark_pp_ci95_high",
             "two_way_mean_excess_vs_long_buy_hold_pp_ci95_high"),
            ("two_way_median_excess_vs_benchmark_pp_ci95_low",
             "two_way_median_excess_vs_long_buy_hold_pp_ci95_low"),
            ("two_way_median_excess_vs_benchmark_pp_ci95_high",
             "two_way_median_excess_vs_long_buy_hold_pp_ci95_high"),
            ("two_way_benchmark_win_fraction_pct_ci95_low",
             "two_way_long_buy_hold_win_fraction_pct_ci95_low"),
            ("two_way_benchmark_win_fraction_pct_ci95_high",
             "two_way_long_buy_hold_win_fraction_pct_ci95_high"),
        ):
            if source_key in long_intervals:
                long_interval_aliases[target_key] = long_intervals[source_key]

        output.append({
            "strategy": strategy,
            "cost_scenario": cost,
            "direction_mode": direction,
            "directional_benchmark_role": "1x short hold" if direction == "SHORT" else "1x long hold",
            "status": status,
            "error": error,
            "long_buy_hold_bootstrap_status": long_status,
            "long_buy_hold_bootstrap_error": long_error,
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
            "directional_benchmark_mean_return_pct": round(point["benchmark_mean_return_pct"], 4) if point["benchmark_mean_return_pct"] is not None else "",
            "mean_excess_vs_directional_benchmark_pp": round(point["mean_excess_vs_benchmark_pp"], 4) if point["mean_excess_vs_benchmark_pp"] is not None else "",
            "median_excess_vs_directional_benchmark_pp": round(point["median_excess_vs_benchmark_pp"], 4) if point["median_excess_vs_benchmark_pp"] is not None else "",
            "strategy_win_fraction_vs_directional_benchmark_pct": round(point["benchmark_win_fraction_pct"], 2) if point["benchmark_win_fraction_pct"] is not None else "",
            "long_buy_hold_mean_return_pct": round(long_point["benchmark_mean_return_pct"], 4) if long_point["benchmark_mean_return_pct"] is not None else "",
            "mean_excess_vs_long_buy_hold_pp": round(long_point["mean_excess_vs_benchmark_pp"], 4) if long_point["mean_excess_vs_benchmark_pp"] is not None else "",
            "median_excess_vs_long_buy_hold_pp": round(long_point["median_excess_vs_benchmark_pp"], 4) if long_point["median_excess_vs_benchmark_pp"] is not None else "",
            "strategy_win_fraction_vs_long_buy_hold_pct": round(long_point["benchmark_win_fraction_pct"], 2) if long_point["benchmark_win_fraction_pct"] is not None else "",
            "bootstrap_replicates": replicates,
            "bootstrap_block_length_windows": block_length,
            "bootstrap_seed": run_seed,
            "bootstrap_method": "two_way_market_resampling_plus_contiguous_window_blocks",
            "two_way_directional_benchmark_mean_return_pct_ci95_low": intervals.get("two_way_benchmark_mean_return_pct_ci95_low", ""),
            "two_way_directional_benchmark_mean_return_pct_ci95_high": intervals.get("two_way_benchmark_mean_return_pct_ci95_high", ""),
            "two_way_mean_excess_vs_directional_benchmark_pp_ci95_low": intervals.get("two_way_mean_excess_vs_benchmark_pp_ci95_low", ""),
            "two_way_mean_excess_vs_directional_benchmark_pp_ci95_high": intervals.get("two_way_mean_excess_vs_benchmark_pp_ci95_high", ""),
            **intervals,
            **long_interval_aliases,
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
                 output_dir="walk-forward-results", start_balance=5.0,
                 oos_windows=OOS_WINDOWS):
    warmup_days_available = validate_window_geometry(days, oos_windows)
    if start_balance <= 0:
        raise ValueError("start_balance must be positive")
    os.makedirs(output_dir, exist_ok=True)
    requested_end_ts = int(end_ts or time.time())
    end_ts = floor_to_hour(requested_end_ts)
    start_ts = end_ts - days * 86400
    windows = make_windows(end_ts, count=oos_windows)
    markets = select_research_markets(get_markets(), max_markets)
    metadata = {
        "experiment": "fixed_parameter_hourly_walk_forward_funding_aware_direction_decomposition",
        "selection_rule": "predeclared ordered crypto-major cohort; no OOS-based market selection",
        "direction_modes": list(DIRECTION_MODES),
        "direction_mode_role": "predeclared BOTH/LONG/SHORT decomposition; exploratory diagnosis only, not OOS-based side selection",
        "requested_symbols": list(RESEARCH_SYMBOLS[:max_markets]),
        "markets_tested": [market["symbol"] for market in markets],
        "days_of_history_requested": days,
        "warmup_days_required": MIN_WARMUP_DAYS,
        "warmup_days_available": warmup_days_available,
        "oos_window_count": len(windows),
        "oos_window_days": WINDOW_DAYS,
        "requested_end_timestamp": requested_end_ts,
        "shared_end_timestamp": end_ts,
        "history_start_timestamp": start_ts,
        "end_timestamp_rule": "floored to the last fully closed UTC hourly boundary",
        "start_balance": start_balance,
        "balance_role": "selected diagnostic strategy-capital scenario; independent 5 USD feasibility upper bound is reported below",
        "five_usd_capital_feasibility": capital_feasibility_audit(
            5.0, hourly.RISK_FRACTION, 0.003, hourly.MAX_LEVERAGE, hourly.MIN_NOTIONAL
        ),
        "selected_balance_capital_feasibility": capital_feasibility_audit(
            start_balance, hourly.RISK_FRACTION, 0.003, hourly.MAX_LEVERAGE, hourly.MIN_NOTIONAL
        ),
        "minimum_quote_notional_usd": hourly.MIN_NOTIONAL,
        "minimum_quote_notional_source": "simulator assumption; not independently verified for every market in this run",
        "risk_fraction": hourly.RISK_FRACTION,
        "max_leverage": hourly.MAX_LEVERAGE,
        "cost_scenarios": [
            {"name": name, "fee_per_side": fee, "slippage_per_side": slip}
            for name, fee, slip in COSTS
        ],
        "funding": "included: signed hourly historical funding applied to open positions and both 1x long and 1x short hold benchmarks",
        "directional_benchmark_rule": "BOTH and LONG compare to 1x long hold; SHORT compares to 1x short hold; every row also retains the 1x long hold and cash baselines.",
        "funding_source": "Lighter public /api/v1/fundings?resolution=1h",
        "funding_sign_convention": "positive rate with direction=long means longs pay; direction=short means shorts pay",
        "funding_mark_price_assumption": "funding is applied at its matching hourly candle timestamp using that candle close, mirroring the existing funding-aware simulator; exact settlement-mark alignment remains an assumption",
        "funding_coverage_threshold_pct": MIN_FUNDING_COVERAGE_PCT,
        "funding_coverage_rule": "at least 99% of expected hourly settlement timestamps and zero unexpected timestamps in every market and OOS window; otherwise fail closed",
        "historical_funding_coverage_pct": "reported per market and OOS window in funding_coverage.csv",
        "cash_baseline_return_pct": 0.0,
        "buy_hold_baseline": (
            "not_executable_below_min_notional"
            if start_balance < hourly.MIN_NOTIONAL
            else "1x long perpetual hold; same per-side fees/slippage and signed historical funding"
        ),
        "short_hold_baseline": (
            "not_executable_below_min_notional"
            if start_balance < hourly.MIN_NOTIONAL
            else "1x short perpetual hold; same per-side fees/slippage and inverse signed historical funding"
        ),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "block_length_windows": BOOTSTRAP_BLOCK_LENGTH,
            "seed": BOOTSTRAP_SEED,
            "seed_rule": "base seed plus sorted strategy/cost/direction group index",
            "method": "two_way_market_resampling_plus_contiguous_window_blocks",
        },
        "paper_only": True,
    }
    with open(os.path.join(output_dir, "walk_forward_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    results, failures, coverage_rows = [], [], []
    market_data = []
    for n, market in enumerate(markets, 1):
        symbol = market["symbol"]
        print(f"[{n}/{len(markets)}] prefetching candles and funding for {symbol}")
        try:
            candles = fetch_candles(market["id"], "1h", start_ts, end_ts)
            if len(candles) < days * 24 * 0.8:
                raise RuntimeError(f"incomplete hourly candle history: {len(candles)} bars")
            funding = funding_research.fetch_funding_rates(market["id"], start_ts, end_ts)
            per_window_coverage = {}
            for window_name, window_start, window_end in windows:
                coverage = funding_coverage(funding, window_start, window_end)
                per_window_coverage[window_name] = coverage
                coverage_rows.append({
                    "symbol": symbol,
                    "window": window_name,
                    "window_start_utc": window_start,
                    "window_end_utc": window_end,
                    **coverage,
                })
                if not funding_coverage_is_acceptable(coverage):
                    failures.append({
                        "symbol": symbol,
                        "window": window_name,
                        "error": (
                            f"funding coverage={coverage['funding_coverage_pct']:.3f}% "
                            f"(required >= {MIN_FUNDING_COVERAGE_PCT:.1f}%, "
                            f"unexpected timestamps={coverage['unexpected_funding_rows']})"
                        ),
                    })
            market_data.append({
                "market": market,
                "candles": candles,
                "funding": funding,
                "coverage": per_window_coverage,
            })
        except Exception as exc:
            failures.append({
                "symbol": symbol,
                "error": type(exc).__name__ + ": " + str(exc)[:240],
            })
            print(f"  PREFETCH FAILED: {type(exc).__name__}: {str(exc)[:240]}")
        time.sleep(0.2)

    with open(os.path.join(output_dir, "funding_coverage.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["symbol", "window", "window_start_utc", "window_end_utc",
                  "funding_rows", "expected_funding_hours", "funding_coverage_pct",
                  "missing_funding_hours", "unexpected_funding_rows"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(coverage_rows)
    with open(os.path.join(output_dir, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)

    if failures or len(market_data) != len(markets):
        raise RuntimeError(
            "Funding/candle coverage gate failed; performance metrics and bootstrap were not computed. "
            f"See funding_coverage.csv and failures.json ({len(failures)} issues)."
        )

    for n, item in enumerate(market_data, 1):
        market = item["market"]
        symbol = market["symbol"]
        candles = item["candles"]
        funding = item["funding"]
        print(f"[{n}/{len(market_data)}] simulating {symbol}")
        for strategy in STRATEGIES:
            for direction_mode in DIRECTION_MODES:
                for window_name, window_start, window_end in windows:
                    coverage = item["coverage"][window_name]
                    for cost_name, fee, slip in COSTS:
                        row = simulate_funding_audited(
                            symbol, candles, strategy, window_start, window_end,
                            funding, start_balance, fee, slip, direction_mode
                        )
                        long_benchmark = buy_hold_baseline(
                            candles, window_start, window_end, start_balance,
                            fee, slip, hourly.MIN_NOTIONAL, funding=funding, direction="LONG"
                        )
                        short_benchmark = buy_hold_baseline(
                            candles, window_start, window_end, start_balance,
                            fee, slip, hourly.MIN_NOTIONAL, funding=funding, direction="SHORT"
                        )
                        directional_direction = "SHORT" if direction_mode == "SHORT" else "LONG"
                        directional_benchmark = (
                            short_benchmark if directional_direction == "SHORT" else long_benchmark
                        )
                        long_excess = (
                            row["return_pct"] - long_benchmark["return_pct"]
                            if long_benchmark["executable"] else None
                        )
                        directional_excess = (
                            row["return_pct"] - directional_benchmark["return_pct"]
                            if directional_benchmark["executable"] else None
                        )
                        coverage_ok = funding_coverage_is_acceptable(coverage)
                        row.update({
                            "window": window_name,
                            "window_start_utc": window_start,
                            "window_end_utc": window_end,
                            "window_days": WINDOW_DAYS,
                            "context_days": days,
                            "cost_scenario": cost_name,
                            "direction_mode": direction_mode,
                            "fees_per_side_pct": fee * 100,
                            "slippage_per_side_pct": slip * 100,
                            "funding": "included: signed hourly settlement history",
                            "funding_source": "Lighter public /api/v1/fundings resolution=1h",
                            "funding_sign_convention": "positive long-direction rate: long pays; short receives",
                            "funding_coverage_pct": coverage["funding_coverage_pct"],
                            "historical_funding_coverage_pct": coverage["funding_coverage_pct"],
                            "funding_rows_in_window": coverage["funding_rows"],
                            "expected_funding_hours": coverage["expected_funding_hours"],
                            "funding_missing_hours": coverage["missing_funding_hours"],
                            "funding_unexpected_rows": coverage["unexpected_funding_rows"],
                            "cash_return_pct": 0.0,
                            "cash_baseline_return_pct": 0.0,
                            "excess_vs_cash_pp": row["return_pct"],
                            "buy_hold_executable": long_benchmark["executable"],
                            "buy_hold_status": long_benchmark["status"],
                            "buy_hold_return_pct": long_benchmark["return_pct"],
                            "buy_hold_min_quote_notional_usd": hourly.MIN_NOTIONAL,
                            "buy_hold_method": "1x long perpetual hold; same fees/slippage and signed historical funding",
                            "buy_hold_funding_net_cost_usd": long_benchmark["funding_net_cost_usd"],
                            "buy_hold_funding_events_charged": long_benchmark["funding_events_charged"],
                            "excess_vs_buy_hold_pp": long_excess,
                            "strategy_beats_buy_hold_observed": bool(long_excess is not None and long_excess > 0),
                            "long_buy_hold_paired_evaluable": bool(
                                long_benchmark["executable"] and coverage_ok
                            ),
                            "short_hold_executable": short_benchmark["executable"],
                            "short_hold_status": short_benchmark["status"],
                            "short_hold_return_pct": short_benchmark["return_pct"],
                            "short_hold_method": "1x short perpetual hold; same fees/slippage and inverse signed historical funding",
                            "short_hold_funding_net_cost_usd": short_benchmark["funding_net_cost_usd"],
                            "short_hold_funding_events_charged": short_benchmark["funding_events_charged"],
                            "short_hold_paired_evaluable": bool(
                                short_benchmark["executable"] and coverage_ok
                            ),
                            "directional_benchmark_direction": directional_direction,
                            "directional_benchmark_method": f"1x {directional_direction.lower()} perpetual hold; same fees/slippage and signed funding",
                            "benchmark_status": directional_benchmark["status"],
                            "benchmark_funding_net_cost_usd": directional_benchmark["funding_net_cost_usd"],
                            "benchmark_funding_events_charged": directional_benchmark["funding_events_charged"],
                            "excess_vs_directional_benchmark_pp": directional_excess,
                            "paired_evaluable": bool(directional_benchmark["executable"] and coverage_ok),
                            "strategy_return_pct": row["return_pct"],
                            "strategy_mtm_drawdown_pct": row["max_drawdown_mtm_pct"],
                            "strategy_status": row["status"],
                            "strategy_trades": row["trades"],
                            "strategy_profit_factor": row["profit_factor"],
                            "benchmark_executable": directional_benchmark["executable"],
                            "benchmark_return_pct": directional_benchmark["return_pct"],
                            "observed_excess_return_vs_benchmark_pp": directional_excess,
                            "strategy_beats_benchmark_observed": bool(
                                directional_excess is not None and directional_excess > 0
                            ),
                            "fixed_parameters": True,
                        })
                        results.append(row)
                        print(
                            f"  {strategy}/{direction_mode}/{window_name}/{cost_name}: "
                            f"{row['return_pct']:+.2f}%, funding {row['funding_net_cost_usd']:+.5f} USD, "
                            f"coverage {coverage['funding_coverage_pct']:.2f}%, "
                            f"DD {row['max_drawdown_mtm_pct']:.2f}%, PF {row['profit_factor']}, "
                            f"trades {row['trades']}"
                        )

    if results:
        with open(os.path.join(output_dir, "walk_forward_detail.csv"), "w", newline="", encoding="utf-8") as f:
            fields = sorted({k for row in results for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
    summary = summarize(results)
    with open(os.path.join(output_dir, "walk_forward_summary.csv"), "w", newline="", encoding="utf-8") as f:
        fields = ["strategy", "cost_scenario", "direction_mode", "observations", "tested_market_windows",
                  "low_sample_market_windows", "active_market_windows", "no_trade_market_windows",
                  "mean_return_pct", "median_return_pct", "tested_only_mean_return_pct",
                  "positive_market_windows", "positive_fraction_pct", "cash_baseline_return_pct",
                  "mean_excess_vs_cash_pp", "buy_hold_executable_market_windows",
                  "buy_hold_mean_return_pct", "mean_excess_vs_buy_hold_pp",
                  "strategy_win_fraction_vs_buy_hold_pct", "directional_benchmark_executable_market_windows",
                  "directional_benchmark_mean_return_pct", "mean_excess_vs_directional_benchmark_pp",
                  "strategy_win_fraction_vs_directional_benchmark_pct", "mean_mtm_drawdown_pct", "total_trades",
                  "pf_eligible_windows", "mean_profit_factor", "min_notional_rejections",
                  "historical_funding_coverage_pct", "minimum_funding_coverage_pct",
                  "total_strategy_funding_net_cost_usd", "total_buy_hold_funding_net_cost_usd",
                  "total_directional_benchmark_funding_net_cost_usd", "total_buy_hold_funding_events",
                  "total_directional_benchmark_funding_events", "total_strategy_funding_events"]
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
        benchmark_groups = (
            ("directional_hold", strategy_rows),
            ("long_buy_hold", with_long_buy_hold_benchmark(strategy_rows)),
        )
        for benchmark_role, audit_rows in benchmark_groups:
            strategy_lomo = leave_one_market_out(
                audit_rows, [start_balance], [cost[0] for cost in COSTS],
                list(DIRECTION_MODES), window_names, symbols
            )
            for item in strategy_lomo:
                expected = (len(symbols) if item["omitted_symbol"] == "NONE"
                            else len(symbols) - 1) * len(window_names)
                item["strategy"] = strategy
                item["benchmark_role"] = benchmark_role
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
        print(f"  {row['strategy']}/{row['direction_mode']}/{row['cost_scenario']}: mean {row['mean_return_pct']:+.2f}%, "
              f"positive {row['positive_market_windows']}/{row['observations']}, "
              f"active {row['active_market_windows']}, no-trade {row['no_trade_market_windows']}, "
              f"min-notional rejects {row['min_notional_rejections']}, "
              f"long-hold {row['buy_hold_mean_return_pct']}, excess-long {row['mean_excess_vs_buy_hold_pp']}, "
              f"directional-hold {row['directional_benchmark_mean_return_pct']}, "
              f"excess-directional {row['mean_excess_vs_directional_benchmark_pp']}, "
              f"DD {row['mean_mtm_drawdown_pct']:.2f}%, PF {row['mean_profit_factor']}, "
              f"trades {row['total_trades']}")
    print("TWO-WAY BOOTSTRAP (10,000 replicates; 2-window contiguous blocks):")
    for row in bootstrap_summary:
        if row["status"] == "ok":
            print(f"  {row['strategy']}/{row['direction_mode']}/{row['cost_scenario']}: "
                  f"mean CI95=[{row.get('two_way_mean_return_pct_ci95_low')},"
                  f"{row.get('two_way_mean_return_pct_ci95_high')}], "
                  f"excess-vs-{row['directional_benchmark_role']} CI95=[{row.get('two_way_mean_excess_vs_directional_benchmark_pp_ci95_low')},"
                  f"{row.get('two_way_mean_excess_vs_directional_benchmark_pp_ci95_high')}], "
                  f"excess-vs-long-buy-hold CI95=[{row.get('two_way_mean_excess_vs_long_buy_hold_pp_ci95_low')},"
                  f"{row.get('two_way_mean_excess_vs_long_buy_hold_pp_ci95_high')}]")
        else:
            print(f"  {row['strategy']}/{row['cost_scenario']}: CI not computed ({row['error']})")
    minimum_coverage = min((row["funding_coverage_pct"] for row in coverage_rows), default=0.0)
    print(f"Leave-one-market-out rows={len(lomo_rows)}; minimum funding coverage={minimum_coverage:.3f}%; "
          f"Rows={len(results)}, API/coverage failures={len(failures)}. Paper-only.")
    return results, summary, failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=CONTEXT_DAYS)
    parser.add_argument("--oos-windows", type=int, default=OOS_WINDOWS)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="walk-forward-results")
    parser.add_argument("--start-balance", type=float, default=5.0)
    parser.add_argument("--end-timestamp", type=int, default=None,
                        help="UTC Unix timestamp ending the dataset; use a disjoint historical holdout")
    args = parser.parse_args()
    run_research(args.max_markets, args.days, end_ts=args.end_timestamp,
                 output_dir=args.out, start_balance=args.start_balance,
                 oos_windows=args.oos_windows)


if __name__ == "__main__":
    main()
