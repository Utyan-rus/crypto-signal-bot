"""Lighter fee-tier sensitivity study for fixed Donchian 20/10 and ADX(14)>=20.

This is an execution-cost sensitivity analysis, not a strategy parameter search.
It reuses the same predeclared crypto-major cohort, 12 contiguous 30-day OOS
windows, funding series, balances, exits, and two-way bootstrap as the ADX study.
No orders are sent. Account tier is not inferred from the user's account.
"""
import argparse
import csv
import json
import os
import time
from unittest.mock import patch

import adx_filter_bootstrap_audit as bootstrap
import adx_filtered_donchian_research as adx
import funding_aware_research as funding
from donchian_benchmark_research import (
    buy_hold,
    calculate_warmup_days,
    select_research_markets,
    validate_oos_history,
)


# Fee assumptions are per side. Slippage is an explicit, fixed sensitivity
# assumption, not a measured fill estimate. All scenarios are predeclared.
COST_SCENARIOS = (
    ("standard_0bps_fee_5bps_slippage", 0.0, 0.0005, "Standard"),
    ("standard_0bps_fee_10bps_slippage", 0.0, 0.0010, "Standard"),
    ("plus_0_5bps_fee_5bps_slippage", 0.00005, 0.0005, "Plus"),
    ("premium_2_8bps_fee_5bps_slippage", 0.00028, 0.0005, "Premium"),
    ("premium_2_8bps_fee_10bps_slippage", 0.00028, 0.0010, "Premium"),
    ("legacy_5bps_fee_5bps_slippage", 0.0005, 0.0005, "Conservative generic"),
    ("legacy_5bps_fee_10bps_slippage", 0.0005, 0.0010, "Conservative generic"),
)

FEE_TIER_ASSUMPTIONS = (
    {"tier": "Standard", "taker_fee_per_side_pct": 0.0},
    {"tier": "Plus", "taker_fee_per_side_pct": 0.005},
    {"tier": "Premium", "taker_fee_per_side_pct": 0.028},
)


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


def simulate_market_window(
    symbol,
    candles,
    funding_rates,
    window_name,
    window_start,
    window_end,
    shared_end_timestamp,
    balance,
    cost_name,
    fee_rate,
    slippage_rate,
    funding_coverage_pct,
):
    """Run both fixed variants and the matched buy-and-hold baseline for one cell."""
    benchmark = buy_hold(
        symbol,
        candles,
        window_start,
        window_end,
        funding_rates,
        balance,
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
    )
    benchmark_executable = (
        benchmark.get("status") == "tested"
        and bool(benchmark.get("min_quote_check_pass"))
    )
    rows = []
    for strategy_name in adx.STRATEGIES:
        result = funding.simulate_with_funding(
            symbol,
            candles,
            strategy_name,
            window_start,
            window_end,
            funding_rates,
            start_balance=balance,
            fee_rate=fee_rate,
            slippage_rate=slippage_rate,
            direction_mode="BOTH",
        )
        return_pct = float(result["return_pct"])
        benchmark_return = benchmark.get("return_pct")
        rows.append({
            "symbol": symbol,
            "strategy": strategy_name,
            "strategy_status": result["status"],
            "window": window_name,
            "window_start_utc": window_start,
            "window_end_utc": window_end,
            "shared_end_timestamp": shared_end_timestamp,
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
            "funding_coverage_pct": funding_coverage_pct,
            "min_notional_rejections": result["min_notional_rejections"],
            "min_notional_assumption_usd": result["min_notional_assumption_usd"],
            "active_trade": int(result["trades"]) > 0,
            "eligible_5plus_trades": result["status"] == "tested" and int(result["trades"]) >= 5,
            "cash_return_pct": 0.0,
            "excess_vs_cash_pp": round(return_pct, 4),
            "buy_hold_status": benchmark.get("status", ""),
            "buy_hold_return_pct": benchmark_return if benchmark_return is not None else "",
            "buy_hold_mtm_drawdown_pct": benchmark.get("max_drawdown_mtm_pct", ""),
            "buy_hold_funding_net_cost_usd": benchmark.get("funding_net_cost_usd", ""),
            "buy_hold_notional_usd": benchmark.get("min_quote_notional_usd", ""),
            "buy_hold_min_quote_check_pass": bool(benchmark.get("min_quote_check_pass", False)),
            "buy_hold_executable": benchmark_executable,
            "excess_vs_buy_hold_pp": (
                round(return_pct - float(benchmark_return), 4)
                if benchmark_executable and benchmark_return is not None else ""
            ),
            "paper_only": True,
        })
    return rows


def run_research(
    days=400,
    windows_count=12,
    window_days=30,
    max_markets=10,
    balances=(5.0, 50.0, 100.0),
    end_timestamp=None,
    output_dir="lighter-fee-tier-results",
    replicates=10000,
    block_length=2,
    seed=20261102,
):
    if days < windows_count * window_days + 30:
        raise ValueError("history must include all OOS windows plus at least 30 warm-up days")
    if windows_count < 1 or window_days < 1 or max_markets < 1:
        raise ValueError("windows, window days, and max markets must be positive")
    if replicates < 100 or block_length < 1 or block_length > windows_count:
        raise ValueError("invalid bootstrap replicates or block length")

    end_timestamp = int(end_timestamp or time.time())
    warmup_days = calculate_warmup_days(days, windows_count, window_days, minimum=30)
    history_start = end_timestamp - days * 86400
    windows = funding.make_oos_windows(end_timestamp, count=windows_count, window_days=window_days)
    venue_markets = funding.get_markets()
    markets = select_research_markets(
        venue_markets,
        max_markets=max_markets,
        requested_symbols=adx.MAJOR_SYMBOLS,
    )
    if not markets:
        raise RuntimeError("no predeclared crypto-major markets were selected")

    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "experiment": "fixed_donchian20_10_vs_adx14_ge20_lighter_fee_tier_sensitivity",
        "hypothesis": (
            "The previous 5 bps per-side fee assumption may overstate regular-order fees "
            "for Lighter Standard accounts; compare public fee-tier assumptions without "
            "changing strategy parameters."
        ),
        "selection_rule": (
            "Fee scenarios are predeclared from public Lighter tier descriptions; "
            "strategy parameters, cohort, OOS windows, and slippage assumptions are fixed. "
            "No parameter search or best-tier selection."
        ),
        "baseline_strategy": adx.BASELINE,
        "candidate_strategy": adx.CANDIDATE,
        "strategy_names": list(adx.STRATEGIES),
        "fixed_parameters": {
            "donchian_entry_lookback_hours": 20,
            "donchian_exit_lookback_hours": 10,
            "adx_period_hours": adx.ADX_PERIOD,
            "adx_threshold": adx.ADX_THRESHOLD,
            "direction_mode": "BOTH",
            "risk_fraction": 0.005,
            "max_leverage": 3.0,
            "minimum_quote_notional_usd": 10.0,
        },
        "fee_tier_assumptions": list(FEE_TIER_ASSUMPTIONS),
        "fee_schedule_source_urls": [
            "https://prowdex.com/guides/lighter-fees-explained",
            "https://lighterpedia.com/guides/fees/what-a-zero-fee-trade-actually-costs",
        ],
        "user_account_tier_verified": False,
        "slippage_model": "fixed per-side sensitivity assumptions; not measured historical L2 fills",
        "slippage_scenarios_per_side_pct": [0.05, 0.10],
        "universe_filter": "explicit_crypto_symbol_allowlist_v2_excludes_tokenized_commodity",
        "requested_symbols": list(adx.MAJOR_SYMBOLS),
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
            {
                "name": name,
                "fee_per_side": fee,
                "slippage_per_side": slip,
                "fee_tier_assumption": tier,
            }
            for name, fee, slip, tier in COST_SCENARIOS
        ],
        "funding_source": "Lighter /api/v1/fundings resolution=1h",
        "funding_sign_convention": "positive rate: long pays, short receives",
        "funding_coverage_denominator": "expected hourly observations within each OOS window",
        "minimum_quote_notional_usd": 10.0,
        "bootstrap_replicates": replicates,
        "bootstrap_block_length_windows": block_length,
        "bootstrap_seed": seed,
        "bootstrap_method": "two_way_market_resampling_plus_contiguous_window_blocks",
        "paper_only": True,
    }
    with open(os.path.join(output_dir, "run_metadata.json"), "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    rows, errors = [], []
    expected_rows = len(markets) * len(windows) * len(balances) * len(COST_SCENARIOS) * len(adx.STRATEGIES)
    print(
        f"Lighter fee-tier sensitivity: {len(markets)} markets x {len(windows)} OOS windows x "
        f"{len(COST_SCENARIOS)} cost cases x {len(balances)} balances x 2 fixed variants."
    )
    with patch.object(funding, "build_model", side_effect=adx.build_model_with_adx_filter):
        for market_index, market in enumerate(markets, 1):
            symbol, market_id = market["symbol"], market["id"]
            print(f"[{market_index}/{len(markets)}] {symbol}: candles + hourly funding")
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
                        / expected_funding_rows,
                        2,
                    )
                    for balance in balances:
                        for cost_name, fee_rate, slippage_rate, _tier in COST_SCENARIOS:
                            rows.extend(simulate_market_window(
                                symbol=symbol,
                                candles=candles,
                                funding_rates=funding_rates,
                                window_name=window_name,
                                window_start=window_start,
                                window_end=window_end,
                                shared_end_timestamp=end_timestamp,
                                balance=balance,
                                cost_name=cost_name,
                                fee_rate=fee_rate,
                                slippage_rate=slippage_rate,
                                funding_coverage_pct=coverage_pct,
                            ))
                time.sleep(0.15)
            except Exception as exc:
                errors.append({
                    "symbol": symbol,
                    "error": type(exc).__name__ + ": " + str(exc)[:220],
                })
                print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:220]}")

    write_csv(
        os.path.join(output_dir, "fee_tier_pairs.csv"),
        rows,
        ("symbol", "strategy", "window", "start_balance", "cost_scenario"),
    )
    write_csv(os.path.join(output_dir, "errors.csv"), errors, ("symbol", "error"))
    print(f"Rows={len(rows)} expected={expected_rows}; API/data errors={len(errors)}; paper-only.")

    complete = len(rows) == expected_rows and not errors
    if not complete:
        print("Paired bootstrap suppressed because the predeclared market/window grid is incomplete.")
        return False

    summaries, leave_one_out, window_details = bootstrap.audit(
        rows,
        metadata,
        replicates=replicates,
        block_length=block_length,
        seed=seed,
    )
    write_csv(os.path.join(output_dir, "fee_tier_bootstrap_summary.csv"), summaries)
    write_csv(os.path.join(output_dir, "fee_tier_leave_one_market_out.csv"), leave_one_out)
    write_csv(os.path.join(output_dir, "fee_tier_window_detail.csv"), window_details)
    print(
        f"Two-way fee-tier bootstrap complete: rows={len(rows)}, groups={len(summaries)}, "
        f"leave-one-market-out rows={len(leave_one_out)}, replicates={replicates}; "
        "historical cost sensitivity only."
    )
    report_names = {
        "standard_0bps_fee_5bps_slippage",
        "premium_2_8bps_fee_5bps_slippage",
        "legacy_5bps_fee_5bps_slippage",
        "legacy_5bps_fee_10bps_slippage",
    }
    for row in summaries:
        if row["start_balance"] in (50.0, 100.0) and row["cost_scenario"] in report_names:
            print(
                f"balance={row['start_balance']:g} {row['cost_scenario']}: "
                f"candidate={row['candidate_mean_return_pct']:+.3f}% "
                f"CI95=[{row['candidate_mean_return_ci95_low']},"
                f"{row['candidate_mean_return_ci95_high']}] "
                f"delta_vs_Donchian={row['candidate_minus_baseline_mean_pp']:+.3f}pp "
                f"CI95=[{row['candidate_minus_baseline_mean_ci95_low']},"
                f"{row['candidate_minus_baseline_mean_ci95_high']}] "
                f"funding_coverage={row['candidate_funding_coverage_pct']:.3f}%"
            )
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-markets", type=int, default=10)
    parser.add_argument("--balances", default="5,50,100")
    parser.add_argument("--end-timestamp", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--block-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--out", default="lighter-fee-tier-results")
    args = parser.parse_args()
    try:
        balances = parse_balances(args.balances)
        complete = run_research(
            days=args.days,
            windows_count=args.windows,
            window_days=args.window_days,
            max_markets=args.max_markets,
            balances=balances,
            end_timestamp=args.end_timestamp or None,
            output_dir=args.out,
            replicates=args.replicates,
            block_length=args.block_length,
            seed=args.seed,
        )
        if not complete:
            raise SystemExit(2)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
