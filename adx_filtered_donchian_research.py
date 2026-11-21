"""Paired paper study of a fixed ADX(14) >= 20 gate on Donchian 20/10.

The candidate is predeclared: retain a baseline Donchian breakout only when the
14-period Wilder ADX on the closed signal candle is at least 20. ADX is non-directional.
Both variants share markets, OOS windows, balances, funding and cost assumptions.
No parameter search, no orders, and no production integration.
"""
import argparse
import csv
import json
import os
import time
from unittest.mock import patch

import funding_aware_research as funding
from donchian_benchmark_research import (
    buy_hold,
    calculate_warmup_days,
    select_research_markets,
    validate_oos_history,
)
from hourly_candidate_research import build_model as base_build_model


BASELINE = "donchian_20_10"
CANDIDATE = "donchian_20_10_adx14_ge20"
STRATEGIES = (BASELINE, CANDIDATE)
MAJOR_SYMBOLS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK", "LTC")
ADX_PERIOD = 14
ADX_THRESHOLD = 20.0
MIN_QUOTE_NOTIONAL_USD = 10.0
COST_SCENARIOS = (
    ("base_taker", 0.0005, 0.0005),
    ("double_slippage", 0.0005, 0.0010),
    ("double_fee_and_slippage", 0.0010, 0.0010),
)


def adx_values(candles, period=ADX_PERIOD):
    """Return Wilder ADX values aligned to closed hourly candles; warm-up is None."""
    if period < 2:
        raise ValueError("period must be >= 2")
    count = len(candles)
    values = [None] * count
    if count <= 2 * period - 1:
        return values

    true_range = [0.0] * count
    plus_dm = [0.0] * count
    minus_dm = [0.0] * count
    for index in range(1, count):
        current = candles[index]
        previous = candles[index - 1]
        high_move = float(current["h"]) - float(previous["h"])
        low_move = float(previous["l"]) - float(current["l"])
        plus_dm[index] = high_move if high_move > low_move and high_move > 0 else 0.0
        minus_dm[index] = low_move if low_move > high_move and low_move > 0 else 0.0
        previous_close = float(previous["c"])
        high = float(current["h"])
        low = float(current["l"])
        true_range[index] = max(
            high - low, abs(high - previous_close), abs(low - previous_close)
        )

    smoothed_tr = sum(true_range[1:period + 1])
    smoothed_plus = sum(plus_dm[1:period + 1])
    smoothed_minus = sum(minus_dm[1:period + 1])
    dx_values = [None] * count

    for index in range(period, count):
        if index > period:
            smoothed_tr += true_range[index] - smoothed_tr / period
            smoothed_plus += plus_dm[index] - smoothed_plus / period
            smoothed_minus += minus_dm[index] - smoothed_minus / period

        if smoothed_tr <= 0:
            dx = 0.0
        else:
            plus_di = 100.0 * smoothed_plus / smoothed_tr
            minus_di = 100.0 * smoothed_minus / smoothed_tr
            denominator = plus_di + minus_di
            dx = 100.0 * abs(plus_di - minus_di) / denominator if denominator > 0 else 0.0
        dx_values[index] = dx

        if index == 2 * period - 1:
            seed = [value for value in dx_values[period:2 * period] if value is not None]
            if len(seed) == period:
                values[index] = sum(seed) / period
        elif index > 2 * period - 1 and values[index - 1] is not None:
            values[index] = (
                values[index - 1] * (period - 1) + dx
            ) / period

    return values


def apply_adx_filter(candles, signals, period=ADX_PERIOD, threshold=ADX_THRESHOLD):
    """Keep signals only when closed-candle Wilder ADX(period) >= threshold."""
    if not 0.0 <= float(threshold) <= 100.0:
        raise ValueError("threshold must be between 0 and 100")
    values = adx_values(candles, period=period)
    index_by_time = {int(bar["t"]): index for index, bar in enumerate(candles)}
    filtered = {}
    for timestamp, signal in signals.items():
        index = index_by_time.get(int(timestamp))
        if index is None:
            continue
        value = values[index]
        if value is not None and value >= float(threshold):
            filtered[timestamp] = signal
    return filtered


def build_model_with_adx_filter(candles, strategy):
    """Return the unchanged baseline model or its fixed ADX(14) >= 20 variant."""
    if strategy not in STRATEGIES:
        raise ValueError(f"unsupported strategy: {strategy}")
    signals, exits = base_build_model(candles, BASELINE)
    if strategy == CANDIDATE:
        signals = apply_adx_filter(candles, signals)
    return signals, exits


def write_csv(path, rows, fallback_fields=()):
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else list(fallback_fields)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_balances(raw):
    values = [float(value.strip()) for value in raw.split(",") if value.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("balances must be a comma-separated list of positive numbers")
    return values


def run_research(days=400, windows_count=12, window_days=30, max_markets=10,
                 balances=(5.0, 50.0, 100.0), end_timestamp=None,
                 output_dir="adx-filtered-donchian-results"):
    warmup_days = calculate_warmup_days(days, windows_count, window_days, minimum=30)
    end_timestamp = int(end_timestamp or time.time())
    history_start = end_timestamp - days * 86400
    windows = funding.make_oos_windows(end_timestamp, count=windows_count, window_days=window_days)
    venue_markets = funding.get_markets()
    markets = select_research_markets(
        venue_markets, max_markets=max_markets, requested_symbols=MAJOR_SYMBOLS
    )
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "experiment": "fixed_adx14_ge20_gate_on_donchian_20_10",
        "hypothesis": "A fixed ADX(14) >= 20 gate avoids low-trend-strength Donchian entries and may reduce whipsaw turnover.",
        "selection_rule": "Candidate and baseline are evaluated on the same predeclared crypto-major cohort and OOS windows; no holdout tuning.",
        "baseline_strategy": BASELINE,
        "candidate_strategy": CANDIDATE,
        "strategy_names": list(STRATEGIES),
        "fixed_parameters": {
            "donchian_entry_lookback_hours": 20,
            "donchian_exit_lookback_hours": 10,
            "adx_period_hours": ADX_PERIOD,
            "adx_threshold": ADX_THRESHOLD,
            "candidate_rule": "Wilder ADX(14) at the closed signal candle is >= 20; applies to LONG and SHORT",
            "direction_mode": "BOTH",
            "risk_fraction": 0.005,
            "max_leverage": 3.0,
            "minimum_quote_notional_usd": MIN_QUOTE_NOTIONAL_USD,
        },
        "universe_filter": "explicit_crypto_symbol_allowlist_v2_excludes_tokenized_commodity",
        "requested_symbols": list(MAJOR_SYMBOLS),
        "markets_tested": [market["symbol"] for market in markets],
        "venue_market_count_before_filter": len(venue_markets),
        "days_of_history_requested": days,
        "warmup_days_available": warmup_days,
        "oos_window_count": len(windows),
        "oos_window_days": window_days,
        "oos_coverage_days": windows_count * window_days,
        "shared_end_timestamp": end_timestamp,
        "history_start_timestamp": history_start,
        "balances": list(balances),
        "cost_scenarios": [
            {"name": name, "fee_per_side": fee, "slippage_per_side": slip}
            for name, fee, slip in COST_SCENARIOS
        ],
        "funding_source": "Lighter /api/v1/fundings resolution=1h",
        "funding_sign_convention": "positive rate: long pays, short receives",
        "funding_coverage_denominator": "expected hourly observations within each OOS window",
        "paper_only": True,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    rows, errors = [], []
    print(
        f"ADX(14) >= 20 study: {len(markets)} markets x {len(windows)} OOS windows x "
        f"{len(STRATEGIES)} variants x {len(balances)} balances x {len(COST_SCENARIOS)} cost cases."
    )
    with patch.object(funding, "build_model", side_effect=build_model_with_adx_filter):
        for market_index, market in enumerate(markets, 1):
            symbol, market_id = market["symbol"], market["id"]
            print(f"[{market_index}/{len(markets)}] {symbol}")
            try:
                candles = funding.fetch_candles(market_id, "1h", history_start, end_timestamp)
                funding_rates = funding.fetch_funding_rates(market_id, history_start, end_timestamp)
                history_errors = validate_oos_history(candles, windows)
                if history_errors:
                    errors.append({
                        "symbol": symbol,
                        "error": "insufficient_oos_history:" + ";".join(history_errors),
                    })
                    continue
                if not funding_rates:
                    errors.append({"symbol": symbol, "error": "missing_funding_history"})
                    continue

                for window_name, window_start, window_end in windows:
                    expected_funding_rows = max(1, (window_end - window_start) // 3600)
                    coverage_pct = round(
                        100 * sum(window_start <= ts < window_end for ts in funding_rates)
                        / expected_funding_rows, 2
                    )
                    for balance in balances:
                        for cost_name, fee_rate, slippage_rate in COST_SCENARIOS:
                            benchmark = buy_hold(
                                symbol, candles, window_start, window_end, funding_rates,
                                balance, fee_rate=fee_rate, slippage_rate=slippage_rate,
                            )
                            benchmark_executable = (
                                benchmark.get("status") == "tested"
                                and bool(benchmark.get("min_quote_check_pass"))
                            )
                            for strategy_name in STRATEGIES:
                                result = funding.simulate_with_funding(
                                    symbol, candles, strategy_name, window_start, window_end,
                                    funding_rates, start_balance=balance,
                                    fee_rate=fee_rate, slippage_rate=slippage_rate,
                                    direction_mode="BOTH",
                                )
                                return_pct = float(result["return_pct"])
                                row = {
                                    "symbol": symbol,
                                    "strategy": strategy_name,
                                    "strategy_status": result["status"],
                                    "window": window_name,
                                    "window_start_utc": window_start,
                                    "window_end_utc": window_end,
                                    "shared_end_timestamp": end_timestamp,
                                    "start_balance": balance,
                                    "direction_mode": "BOTH",
                                    "cost_scenario": cost_name,
                                    "fees_per_side_pct": result["fees_per_side_pct"],
                                    "slippage_per_side_pct": result["slippage_per_side_pct"],
                                    "trades": result["trades"],
                                    "long_trades": result["long_trades"],
                                    "short_trades": result["short_trades"],
                                    "return_pct": return_pct,
                                    "end_balance": result["end_balance"],
                                    "max_drawdown_mtm_pct": result["max_drawdown_mtm_pct"],
                                    "profit_factor": result["profit_factor"],
                                    "funding_events_charged": result["funding_events_charged"],
                                    "funding_net_cost_usd": result["funding_net_cost_usd"],
                                    "funding_history_rows": result["funding_history_rows"],
                                    "funding_coverage_pct": coverage_pct,
                                    "min_notional_rejections": result["min_notional_rejections"],
                                    "min_notional_assumption_usd": result["min_notional_assumption_usd"],
                                    "active_trade": int(result["trades"]) > 0,
                                    "eligible_5plus_trades": (
                                        result["status"] == "tested" and int(result["trades"]) >= 5
                                    ),
                                    "cash_return_pct": 0.0,
                                    "excess_vs_cash_pp": round(return_pct, 4),
                                    "buy_hold_status": benchmark.get("status", ""),
                                    "buy_hold_return_pct": benchmark.get("return_pct", ""),
                                    "buy_hold_mtm_drawdown_pct": benchmark.get("max_drawdown_mtm_pct", ""),
                                    "buy_hold_funding_net_cost_usd": benchmark.get("funding_net_cost_usd", ""),
                                    "buy_hold_notional_usd": benchmark.get("min_quote_notional_usd", ""),
                                    "buy_hold_min_quote_check_pass": bool(
                                        benchmark.get("min_quote_check_pass", False)
                                    ),
                                    "buy_hold_executable": benchmark_executable,
                                    "excess_vs_buy_hold_pp": (
                                        round(return_pct - float(benchmark["return_pct"]), 4)
                                        if benchmark_executable else ""
                                    ),
                                    "paper_only": True,
                                }
                                rows.append(row)
                time.sleep(0.15)
            except Exception as exc:
                errors.append({
                    "symbol": symbol,
                    "error": type(exc).__name__ + ": " + str(exc)[:220],
                })
                print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:220]}")

    write_csv(os.path.join(output_dir, "adx_filter_pairs.csv"), rows,
              ("symbol", "strategy", "window", "start_balance", "cost_scenario"))
    write_csv(os.path.join(output_dir, "errors.csv"), errors, ("symbol", "error"))
    print(
        f"Rows={len(rows)} expected={len(markets) * len(windows) * len(balances) * len(COST_SCENARIOS) * len(STRATEGIES)} "
        f"API/data errors={len(errors)}. Funding included; paper-only; no orders placed."
    )
    return rows, errors, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-markets", type=int, default=10)
    parser.add_argument("--balances", default="5,50,100")
    parser.add_argument("--end-timestamp", type=int, default=0)
    parser.add_argument("--out", default="adx-filtered-donchian-results")
    args = parser.parse_args()
    try:
        balances = parse_balances(args.balances)
        run_research(
            days=args.days, windows_count=args.windows, window_days=args.window_days,
            max_markets=args.max_markets, balances=balances,
            end_timestamp=args.end_timestamp, output_dir=args.out,
        )
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
