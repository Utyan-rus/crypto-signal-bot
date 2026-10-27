"""Paired Donchian 20/10 vs buy-and-hold with identical OOS windows and cost stress. Paper-only."""
import argparse
import csv
import json
import os
import statistics
import time
from collections import defaultdict

from backtest import fetch_candles, get_markets
from funding_aware_research import fetch_funding_rates, make_oos_windows, simulate_with_funding
from hourly_candidate_research import FEE, SLIPPAGE

MIN_QUOTE_NOTIONAL_USD = 10.0

# Lighter lists tokenized equities, FX and commodities alongside crypto perps.
# Keep this research universe explicit and conservative; do not silently mix RWAs into crypto results.
CRYPTO_RESEARCH_SYMBOLS = frozenset({
    "0G", "1000BONK", "1000FLOKI", "1000NOT", "1000PEPE", "1000SHIB", "2Z",
    "AAVE", "ADA", "AERO", "APT", "ARB", "ASTER", "AVAX", "AVNT", "AXS",
    "AZTEC", "BCH", "BERA", "BIO", "BNB", "BOT", "BTC", "CASHCAT", "CRO",
    "CRV", "DASH", "DATA", "DOGE", "DOT", "DYDX", "EDGE", "EIGEN", "ENA",
    "ETH", "ETHFI", "FARTCOIN", "FIL", "FOLKS", "GMX", "GRAM", "GRASS",
    "HBAR", "HYPE", "ICP", "JTO", "JUP", "KAITO", "LDO", "LINEA", "LINK",
    "LIT", "LTC", "MNT", "MON", "MORPHO", "NEAR", "NMR", "ONDO", "PAXG",
    "PENDLE", "PENGU", "POL", "POPCAT", "PROVE", "PUMP", "PYTH", "QNT",
    "RAY", "RIVER", "S", "SEI", "SKY", "SKR", "SOL", "STABLE", "STBL",
    "STRK", "SUI", "SYRUP", "TAO", "TIA", "TRUMP", "TRX", "UNI", "USELESS",
    "VIRTUAL", "VVV", "WEN", "WIF", "WLD", "WLFI", "XLM", "XMR", "XPL",
    "XRP", "ZEC", "ZK", "ZORA", "ZRO",
})


def normalize_market_symbol(symbol):
    return str(symbol).upper().replace("-PERP", "").replace("/USDC", "").replace("/USDT", "")


def filter_crypto_markets(markets):
    """Filter venue perps to a versioned, explicit crypto-symbol research universe."""
    return [market for market in markets
            if normalize_market_symbol(market.get("symbol", "")) in CRYPTO_RESEARCH_SYMBOLS]

COST_SCENARIOS = (
    ("base_taker", FEE, SLIPPAGE),
    ("double_slippage", FEE, SLIPPAGE * 2),
    ("double_fee_and_slippage", FEE * 2, SLIPPAGE * 2),
)


def max_drawdown_pct(equity):
    peak = equity[0] if equity else 0.0
    result = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            result = max(result, (peak - value) / peak)
    return round(100 * result, 3)


def buy_hold(symbol, candles, start, end, funding, balance, fee_rate=FEE, slippage_rate=SLIPPAGE):
    bars = [b for b in candles if start <= b["t"] < end]
    if len(bars) < 2:
        return {"symbol": symbol, "strategy": "buy_hold_long_1x", "status": "low_sample",
                "trades": 0, "start_balance": balance}
    entry = bars[0]["o"] * (1 + slippage_rate)
    qty = balance / (entry * (1 + fee_rate))
    entry_fee = qty * entry * fee_rate
    funding_cost, charged = 0.0, 0
    cash_after_entry = balance - qty * entry - entry_fee
    equity = [balance]
    for index, bar in enumerate(bars):
        # The position is opened at bars[0].open, so mark its first candle close too.
        # Funding at the entry timestamp is excluded, matching the strategy simulator.
        if index > 0 and bar["t"] in funding:
            funding_cost += qty * bar["c"] * funding[bar["t"]]
            charged += 1
        mark = bar["c"] * (1 - slippage_rate)
        equity.append(cash_after_entry + qty * mark - qty * mark * fee_rate - funding_cost)
    exit_price = bars[-1]["c"] * (1 - slippage_rate)
    end_balance = balance + qty * (exit_price - entry) - entry_fee - qty * exit_price * fee_rate - funding_cost
    notional = qty * entry
    eligible = notional >= MIN_QUOTE_NOTIONAL_USD
    return {
        "symbol": symbol, "strategy": "buy_hold_long_1x", "status": "tested", "trades": 1,
        "start_balance": balance, "end_balance": round(end_balance, 6),
        "return_pct": round(100 * (end_balance / balance - 1), 3),
        "max_drawdown_mtm_pct": max_drawdown_pct(equity),
        "funding_events_charged": charged, "funding_net_cost_usd": round(funding_cost, 8),
        "min_quote_notional_usd": round(notional, 6), "min_quote_check_pass": eligible,
        "execution_feasibility": "quote_minimum_pass_other_constraints_unverified" if eligible else "below_min_quote",
        "fees_per_side_pct": fee_rate * 100, "slippage_per_side_pct": slippage_rate * 100,
    }


def paired_rows(strategy, benchmark, window, window_start, window_end, end_timestamp, cost_scenario):
    if strategy["symbol"] != benchmark["symbol"]:
        raise ValueError("strategy and benchmark symbols differ")
    if float(strategy["start_balance"]) != float(benchmark["start_balance"]):
        raise ValueError("strategy and benchmark balances differ")
    if window_end <= window_start:
        raise ValueError("OOS window must have positive width")
    sret = float(strategy.get("return_pct", 0) or 0)
    bret = float(benchmark.get("return_pct", 0) or 0)
    trades = int(strategy.get("trades", 0) or 0)
    strategy_observed = strategy.get("status") in ("tested", "low_sample")
    strategy_ok = strategy.get("status") == "tested" and trades >= 5
    benchmark_ok = benchmark.get("status") == "tested" and bool(benchmark.get("min_quote_check_pass"))
    paired = strategy_ok and benchmark_ok
    return {
        "symbol": strategy["symbol"], "window": window, "window_start_utc": window_start,
        "window_end_utc": window_end, "shared_end_timestamp": end_timestamp,
        "start_balance": strategy["start_balance"], "cost_scenario": cost_scenario,
        "fees_per_side_pct": strategy.get("fees_per_side_pct", ""),
        "slippage_per_side_pct": strategy.get("slippage_per_side_pct", ""),
        "strategy": "donchian_20_10", "strategy_status": strategy.get("status", ""),
        "strategy_trades": trades, "strategy_return_pct": sret,
        "strategy_mtm_drawdown_pct": strategy.get("max_drawdown_mtm_pct", ""),
        "strategy_profit_factor": strategy.get("profit_factor", ""),
        "strategy_funding_cost_usd": strategy.get("funding_net_cost_usd", ""),
        "strategy_min_notional_rejections": strategy.get("min_notional_rejections", ""),
        "benchmark_return_pct": bret, "benchmark_mtm_drawdown_pct": benchmark.get("max_drawdown_mtm_pct", ""),
        "benchmark_funding_cost_usd": benchmark.get("funding_net_cost_usd", ""),
        "benchmark_funding_events": benchmark.get("funding_events_charged", ""),
        "benchmark_notional_usd": benchmark.get("min_quote_notional_usd", ""),
        "benchmark_min_notional_pass": benchmark.get("min_quote_check_pass", False),
        "strategy_return_observed": strategy_observed,
        "strategy_active_trade": trades > 0,
        "cash_return_pct": 0.0,
        "strategy_excess_vs_cash_pp": round(sret, 3) if strategy_observed else "",
        "benchmark_executable": benchmark_ok,
        "observed_excess_return_vs_benchmark_pp": (
            round(sret - bret, 3) if strategy_observed and benchmark_ok else ""
        ),
        "strategy_beats_cash": sret > 0 if strategy_observed else False,
        "strategy_beats_benchmark_observed": (
            sret > bret if strategy_observed and benchmark_ok else False
        ),
        "strategy_eligible_5plus_trades": strategy_ok, "benchmark_eligible": benchmark_ok,
        "paired_evaluable": paired,
        "excess_return_pp": round(sret - bret, 3) if paired else "",
        "strategy_beats_benchmark": sret > bret if paired else False,
    }


def summarize(rows, group_keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k, "") for k in group_keys)].append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        pairs = [r for r in group if r["paired_evaluable"]]
        excess = [float(r["excess_return_pp"]) for r in pairs]
        strategy_returns = [float(r["strategy_return_pct"]) for r in pairs]
        benchmark_returns = [float(r["benchmark_return_pct"]) for r in pairs]
        strategy_dd = [float(r["strategy_mtm_drawdown_pct"]) for r in pairs
                       if r["strategy_mtm_drawdown_pct"] not in ("", None)]
        benchmark_dd = [float(r["benchmark_mtm_drawdown_pct"]) for r in pairs
                        if r["benchmark_mtm_drawdown_pct"] not in ("", None)]
        pf = [float(r["strategy_profit_factor"]) for r in pairs
              if r["strategy_profit_factor"] not in ("", None)]
        trades = [int(r["strategy_trades"]) for r in pairs]
        wins = sum(bool(r["strategy_beats_benchmark"]) for r in pairs)
        positive = sum(v > 0 for v in strategy_returns)
        observed = [r for r in group if r.get("strategy_return_observed")]
        executable_benchmarks = [r for r in observed if r.get("benchmark_executable")]
        observed_returns = [float(r["strategy_return_pct"]) for r in observed]
        observed_excess_cash = [float(r["strategy_excess_vs_cash_pp"]) for r in observed
                                if r["strategy_excess_vs_cash_pp"] not in ("", None)]
        observed_excess_benchmark = [
            float(r["observed_excess_return_vs_benchmark_pp"]) for r in executable_benchmarks
            if r["observed_excess_return_vs_benchmark_pp"] not in ("", None)
        ]
        observed_strategy_dd = [
            float(r["strategy_mtm_drawdown_pct"]) for r in observed
            if r["strategy_mtm_drawdown_pct"] not in ("", None)
        ]
        observed_active = [r for r in observed if int(r.get("strategy_trades", 0) or 0) > 0]
        observed_wins_benchmark = sum(bool(r["strategy_beats_benchmark_observed"])
                                      for r in executable_benchmarks)
        row = dict(zip(group_keys, key))
        row.update({
            "rows": len(group), "paired_count": len(pairs),
            "strategy_mean_return_pct": round(statistics.mean(strategy_returns), 3) if strategy_returns else "",
            "strategy_median_return_pct": round(statistics.median(strategy_returns), 3) if strategy_returns else "",
            "benchmark_mean_return_pct": round(statistics.mean(benchmark_returns), 3) if benchmark_returns else "",
            "benchmark_median_return_pct": round(statistics.median(benchmark_returns), 3) if benchmark_returns else "",
            "mean_excess_return_pp": round(statistics.mean(excess), 3) if excess else "",
            "median_excess_return_pp": round(statistics.median(excess), 3) if excess else "",
            "strategy_wins_vs_benchmark": wins,
            "strategy_win_fraction_pct": round(100 * wins / len(pairs), 2) if pairs else "",
            "positive_strategy_windows": positive,
            "positive_strategy_fraction_pct": round(100 * positive / len(pairs), 2) if pairs else "",
            "mean_strategy_mtm_dd_pct": round(statistics.mean(strategy_dd), 3) if strategy_dd else "",
            "median_strategy_mtm_dd_pct": round(statistics.median(strategy_dd), 3) if strategy_dd else "",
            "mean_benchmark_mtm_dd_pct": round(statistics.mean(benchmark_dd), 3) if benchmark_dd else "",
            "median_benchmark_mtm_dd_pct": round(statistics.median(benchmark_dd), 3) if benchmark_dd else "",
            "mean_strategy_profit_factor": round(statistics.mean(pf), 3) if pf else "",
            "median_strategy_profit_factor": round(statistics.median(pf), 3) if pf else "",
            "total_strategy_trades": sum(trades),
            "mean_strategy_trades": round(statistics.mean(trades), 2) if trades else "",
            # Unconditional metrics include every observed OOS row, including low-sample
            # and zero-trade rows; zero trades are cash/no-trade, not silently discarded.
            "strategy_observed_rows": len(observed),
            "strategy_active_windows": len(observed_active),
            "strategy_no_trade_windows": sum(int(r.get("strategy_trades", 0) or 0) == 0 for r in observed),
            "strategy_all_rows_mean_return_pct": (
                round(statistics.mean(observed_returns), 3) if observed_returns else ""
            ),
            "strategy_all_rows_median_return_pct": (
                round(statistics.median(observed_returns), 3) if observed_returns else ""
            ),
            "strategy_all_rows_positive_windows": sum(v > 0 for v in observed_returns),
            "strategy_all_rows_positive_fraction_pct": (
                round(100 * sum(v > 0 for v in observed_returns) / len(observed_returns), 2)
                if observed_returns else ""
            ),
            "strategy_all_rows_mean_excess_vs_cash_pp": (
                round(statistics.mean(observed_excess_cash), 3) if observed_excess_cash else ""
            ),
            "strategy_all_rows_median_excess_vs_cash_pp": (
                round(statistics.median(observed_excess_cash), 3) if observed_excess_cash else ""
            ),
            "strategy_all_rows_wins_vs_cash": sum(v > 0 for v in observed_returns),
            "all_rows_total_strategy_trades": sum(int(r.get("strategy_trades", 0) or 0) for r in observed),
            "all_rows_mean_strategy_mtm_dd_pct": (
                round(statistics.mean(observed_strategy_dd), 3) if observed_strategy_dd else ""
            ),
            "all_rows_median_strategy_mtm_dd_pct": (
                round(statistics.median(observed_strategy_dd), 3) if observed_strategy_dd else ""
            ),
            "benchmark_executable_rows": len(executable_benchmarks),
            "benchmark_executable_mean_return_pct": (
                round(statistics.mean(float(r["benchmark_return_pct"]) for r in executable_benchmarks), 3)
                if executable_benchmarks else ""
            ),
            "benchmark_executable_median_return_pct": (
                round(statistics.median(float(r["benchmark_return_pct"]) for r in executable_benchmarks), 3)
                if executable_benchmarks else ""
            ),
            "observed_strategy_vs_benchmark_mean_excess_pp": (
                round(statistics.mean(observed_excess_benchmark), 3) if observed_excess_benchmark else ""
            ),
            "observed_strategy_vs_benchmark_median_excess_pp": (
                round(statistics.median(observed_excess_benchmark), 3) if observed_excess_benchmark else ""
            ),
            "observed_strategy_wins_vs_benchmark": observed_wins_benchmark,
            "observed_strategy_win_fraction_vs_benchmark_pct": (
                round(100 * observed_wins_benchmark / len(executable_benchmarks), 2)
                if executable_benchmarks else ""
            ),
        })
        result.append(row)
    return result


def write_csv(path, rows, fallback_fields=()):
    fields = sorted({k for row in rows for k in row}) if rows else list(fallback_fields)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def calculate_warmup_days(days, window_count, window_days, minimum=30):
    """Require a pre-OOS history buffer before the first fixed test window."""
    warmup_days = int(days) - int(window_count) * int(window_days)
    if warmup_days < minimum:
        raise ValueError(
            f"Need at least {minimum} days of pre-OOS indicator warm-up; "
            f"requested history leaves {warmup_days} days."
        )
    return warmup_days


def validate_oos_history(candles, windows, min_warmup_bars=200, min_window_coverage=0.90):
    """Require indicator warm-up and adequate hourly candle coverage in every OOS window."""
    timestamps = sorted({int(c["t"]) for c in candles if "t" in c})
    if not windows:
        return ["no_oos_windows"]
    errors = []
    warmup_bars = sum(ts < windows[0][1] for ts in timestamps)
    if warmup_bars < min_warmup_bars:
        errors.append(f"warmup_bars:{warmup_bars}<{min_warmup_bars}")
    for name, start, end in windows:
        expected = max(0, int((end - start) // 3600))
        actual = sum(start <= ts < end for ts in timestamps)
        if expected == 0 or actual < expected * min_window_coverage:
            errors.append(f"{name}_bars:{actual}/{expected}")
    return errors

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=400)
    parser.add_argument("--windows", type=int, default=12)
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--max-markets", type=int, default=10)
    parser.add_argument("--balances", default="5,50,100")
    parser.add_argument("--end-timestamp", type=int, default=0)
    parser.add_argument("--out", default="donchian-benchmark-results")
    args = parser.parse_args()
    if args.days < 1 or args.windows < 1 or args.window_days < 1:
        parser.error("--days, --windows and --window-days must be positive")
    try:
        warmup_days = calculate_warmup_days(args.days, args.windows, args.window_days, minimum=30)
    except ValueError as exc:
        parser.error(str(exc))
    end_timestamp = args.end_timestamp or int(time.time())
    history_start = end_timestamp - args.days * 86400
    balances = [float(x.strip()) for x in args.balances.split(",") if x.strip()]
    if not balances or any(x <= 0 for x in balances):
        parser.error("--balances must contain positive values")
    os.makedirs(args.out, exist_ok=True)
    windows = make_oos_windows(end_timestamp, count=args.windows, window_days=args.window_days)
    venue_markets = get_markets()
    markets = filter_crypto_markets(venue_markets)
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    metadata = {
        "universe_filter": "explicit_crypto_symbol_allowlist_v1",
        "venue_market_count_before_filter": len(venue_markets),
        "recognized_crypto_market_count": len(filter_crypto_markets(venue_markets)),
        "markets_tested": [m["symbol"] for m in markets],
        "days_of_history_requested": args.days,
        "warmup_days_available": warmup_days,
        "minimum_warmup_bars": 200,
        "minimum_oos_candle_coverage_fraction": 0.90,
        "oos_window_count": len(windows),
        "oos_window_days": args.window_days,
        "oos_coverage_days": len(windows) * args.window_days,
        "shared_end_timestamp": end_timestamp,
        "history_start_timestamp": history_start,
        "minimum_quote_notional_usd": MIN_QUOTE_NOTIONAL_USD,
        "cost_scenarios": [{"name": name, "fee_per_side": fee, "slippage_per_side": slip}
                           for name, fee, slip in COST_SCENARIOS],
        "paper_only": True,
    }
    with open(os.path.join(args.out, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Universe=crypto_allowlist venue_markets={len(venue_markets)} recognized_crypto={len(filter_crypto_markets(venue_markets))} tested={len(markets)} symbols={','.join(m['symbol'] for m in markets)}")
    rows, errors = [], []
    for market in markets:
        symbol, market_id = market["symbol"], market["id"]
        try:
            candles = fetch_candles(market_id, "1h", history_start, end_timestamp)
            funding = fetch_funding_rates(market_id, history_start, end_timestamp)
            coverage_errors = validate_oos_history(candles, windows)
            if coverage_errors:
                errors.append({"symbol": symbol, "error": "insufficient_oos_history:" + ";".join(coverage_errors)})
                continue
            if not funding:
                errors.append({"symbol": symbol, "error": "missing_funding_history"})
                continue
            for window, window_start, window_end in windows:
                for balance in balances:
                    for cost_scenario, fee_rate, slippage_rate in COST_SCENARIOS:
                        strategy = simulate_with_funding(
                            symbol, candles, "donchian_20_10", window_start, window_end, funding,
                            start_balance=balance, fee_rate=fee_rate, slippage_rate=slippage_rate)
                        benchmark = buy_hold(
                            symbol, candles, window_start, window_end, funding, balance,
                            fee_rate=fee_rate, slippage_rate=slippage_rate)
                        row = paired_rows(strategy, benchmark, window, window_start, window_end,
                                          end_timestamp, cost_scenario)
                        row["funding_coverage_pct"] = round(
                            100 * sum(window_start <= ts < window_end for ts in funding) /
                            max(1, (window_end-window_start)//3600), 2)
                        rows.append(row)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:200]})
        time.sleep(0.15)
    write_csv(os.path.join(args.out, "donchian_benchmark_pairs.csv"), rows)
    write_csv(os.path.join(args.out, "donchian_benchmark_summary.csv"),
              summarize(rows, ("start_balance", "cost_scenario")))
    write_csv(os.path.join(args.out, "donchian_benchmark_by_market.csv"),
              summarize(rows, ("symbol", "start_balance", "cost_scenario")))
    write_csv(os.path.join(args.out, "donchian_benchmark_by_window.csv"),
              summarize(rows, ("window", "start_balance", "cost_scenario")))
    write_csv(os.path.join(args.out, "errors.csv"), errors, ("symbol", "error"))
    print(f"Rows={len(rows)} paired_evaluable={sum(r['paired_evaluable'] for r in rows)} "
          f"errors={len(errors)} shared_end_timestamp={end_timestamp}. Paper only.")
    for balance in balances:
        for cost_scenario, _, _ in COST_SCENARIOS:
            data = [r for r in rows if r["paired_evaluable"] and
                    float(r["start_balance"]) == balance and r["cost_scenario"] == cost_scenario]
            if data:
                excess = [float(r["excess_return_pp"]) for r in data]
                print(f"balance={balance:g} cost={cost_scenario}: paired={len(data)} "
                      f"mean_excess={statistics.mean(excess):+.3f}pp "
                      f"median_excess={statistics.median(excess):+.3f}pp "
                      f"wins={sum(r['strategy_beats_benchmark'] for r in data)}/{len(data)}")
            else:
                print(f"balance={balance:g} cost={cost_scenario}: no executable paired samples")


if __name__ == "__main__":
    main()
