"""EMA 20/50 funding-aware OOS study with buy-and-hold benchmark. Paper only."""
import argparse
import csv
import os
import statistics
import time

from backtest import fetch_candles, get_markets
from funding_aware_research import fetch_funding_rates, simulate_with_funding
from hourly_candidate_research import FEE, SLIPPAGE


MIN_QUOTE_NOTIONAL_USD = 10.0


def max_drawdown_pct(equity):
    peak = equity[0] if equity else 0.0
    max_dd = 0.0
    for value in equity:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak - value) / peak)
    return round(max_dd * 100, 3)


def buy_hold(symbol, candles, start, end, funding, balance):
    bars = [b for b in candles if start <= b["t"] < end]
    base = {
        "symbol": symbol, "strategy": "buy_hold_long_1x", "trades": 0,
        "start_balance": balance, "fees_per_side_pct": FEE * 100,
        "slippage_per_side_pct": SLIPPAGE * 100,
        "min_quote_notional_usd": 0.0, "min_quote_check_pass": False,
        "execution_feasibility": "not_tested", "funding_events_charged": 0,
        "funding_net_cost_usd": 0.0, "max_drawdown_mtm_pct": 0.0,
    }
    if len(bars) < 2:
        return {**base, "status": "low_sample"}

    entry = bars[0]["o"] * (1 + SLIPPAGE)
    exit_price = bars[-1]["c"] * (1 - SLIPPAGE)
    qty = balance / (entry * (1 + FEE))
    entry_fee = qty * entry * FEE
    cash_after_entry = balance - qty * entry - entry_fee
    funding_cost = 0.0
    charged = 0
    equity = [balance]
    for bar in bars[1:]:
        if bar["t"] in funding:
            funding_cost += qty * bar["c"] * funding[bar["t"]]
            charged += 1
        liquidation_price = bar["c"] * (1 - SLIPPAGE)
        liquidation_fee = qty * liquidation_price * FEE
        equity.append(cash_after_entry + qty * liquidation_price - liquidation_fee - funding_cost)

    exit_fee = qty * exit_price * FEE
    end_balance = balance + qty * (exit_price - entry) - entry_fee - exit_fee - funding_cost
    quote_notional = qty * entry
    quote_minimum_pass = quote_notional >= MIN_QUOTE_NOTIONAL_USD
    return {
        **base,
        "trades": 1,
        "end_balance": round(end_balance, 6),
        "return_pct": round(100 * (end_balance / balance - 1), 3),
        "funding_events_charged": charged,
        "funding_net_cost_usd": round(funding_cost, 8),
        "max_drawdown_mtm_pct": max_drawdown_pct(equity),
        "min_quote_notional_usd": round(quote_notional, 6),
        "min_quote_check_pass": quote_minimum_pass,
        "execution_feasibility": (
            "below_min_quote" if not quote_minimum_pass
            else "quote_minimum_pass_other_constraints_unverified"
        ),
        "status": "tested",
    }


def build_comparison_rows(rows):
    indexed = {}
    for row in rows:
        strategy = row.get("strategy")
        if strategy not in ("ema_20_50_cross", "buy_hold_long_1x"):
            continue
        if row.get("symbol") is None or row.get("window") is None or row.get("start_balance") is None:
            continue
        try:
            balance = float(row["start_balance"])
            float(row["return_pct"])
        except (TypeError, ValueError):
            continue
        key = (str(row["symbol"]), str(row["window"]), balance)
        indexed.setdefault(key, {})["benchmark" if row.get("benchmark") else "ema"] = row

    pairs = []
    for (symbol, window, balance), pair in sorted(indexed.items()):
        ema = pair.get("ema")
        bench = pair.get("benchmark")
        if not ema or not bench:
            continue
        try:
            ema_ret = float(ema["return_pct"])
            bench_ret = float(bench["return_pct"])
        except (KeyError, TypeError, ValueError):
            continue
        ema_trades = int(float(ema.get("trades", 0) or 0))
        ema_eligible = ema.get("status") == "tested" and ema_trades >= 5
        bench_quote_eligible = (
            bench.get("status") == "tested"
            and bool(bench.get("min_quote_check_pass"))
            and bench.get("execution_feasibility") != "below_min_quote"
        )
        paired_evaluable = ema_eligible and bench_quote_eligible
        pairs.append({
            "symbol": symbol,
            "window": window,
            "start_balance": balance,
            "ema_status": ema.get("status", ""),
            "ema_trades": ema_trades,
            "ema_return_pct": ema_ret,
            "ema_max_drawdown_mtm_pct": ema.get("max_drawdown_mtm_pct", ""),
            "ema_profit_factor": ema.get("profit_factor", ""),
            "benchmark_return_pct": bench_ret,
            "benchmark_max_drawdown_mtm_pct": bench.get("max_drawdown_mtm_pct", ""),
            "benchmark_funding_net_cost_usd": bench.get("funding_net_cost_usd", ""),
            "benchmark_min_quote_notional_usd": bench.get("min_quote_notional_usd", ""),
            "benchmark_min_quote_check_pass": bench.get("min_quote_check_pass", False),
            "benchmark_execution_feasibility": bench.get("execution_feasibility", ""),
            "benchmark_quote_eligible": bench_quote_eligible,
            "paired_evaluable": paired_evaluable,
            "ema_minus_benchmark_pct_points": round(ema_ret - bench_ret, 3) if paired_evaluable else "",
            "ema_outperforms_benchmark": (ema_ret > bench_ret) if paired_evaluable else False,
            "ema_eligible_5plus_trades": ema_eligible,
        })
    return pairs


def _summarize_pairs(balance, scope, subset):
    ema_returns = [float(x["ema_return_pct"]) for x in subset]
    benchmark_returns = [float(x["benchmark_return_pct"]) for x in subset]
    evaluable = [x for x in subset if x.get("paired_evaluable")]
    excess = [float(x["ema_minus_benchmark_pct_points"]) for x in evaluable]
    ema_dd = [float(x["ema_max_drawdown_mtm_pct"]) for x in evaluable
              if x.get("ema_max_drawdown_mtm_pct") not in ("", None)]
    benchmark_dd = [float(x["benchmark_max_drawdown_mtm_pct"]) for x in evaluable
                    if x.get("benchmark_max_drawdown_mtm_pct") not in ("", None)]
    return {
        "start_balance": balance,
        "scope": scope,
        "pairs": len(subset),
        "ema_mean_return_pct": round(statistics.mean(ema_returns), 3) if ema_returns else "",
        "ema_median_return_pct": round(statistics.median(ema_returns), 3) if ema_returns else "",
        "ema_positive_pairs": sum(x > 0 for x in ema_returns),
        "benchmark_mean_return_pct": round(statistics.mean(benchmark_returns), 3) if benchmark_returns else "",
        "benchmark_median_return_pct": round(statistics.median(benchmark_returns), 3) if benchmark_returns else "",
        "benchmark_positive_pairs": sum(x > 0 for x in benchmark_returns),
        "mean_excess_return_pct_points": round(statistics.mean(excess), 3) if excess else "",
        "median_excess_return_pct_points": round(statistics.median(excess), 3) if excess else "",
        "ema_outperforms_benchmark_pairs": sum(bool(x["ema_outperforms_benchmark"]) for x in evaluable),
        "ema_eligible_pairs": sum(bool(x["ema_eligible_5plus_trades"]) for x in subset),
        "benchmark_quote_eligible_pairs": sum(bool(x.get("benchmark_quote_eligible")) for x in subset),
        "paired_evaluable_pairs": len(evaluable),
        "mean_ema_mtm_drawdown_pct": round(statistics.mean(ema_dd), 3) if ema_dd else "",
        "mean_benchmark_mtm_drawdown_pct": round(statistics.mean(benchmark_dd), 3) if benchmark_dd else "",
    }


def write_comparison(rows, out_dir):
    pairs = build_comparison_rows(rows)
    detail_fields = [
        "symbol", "window", "start_balance", "ema_status", "ema_trades",
        "ema_return_pct", "ema_max_drawdown_mtm_pct", "ema_profit_factor",
        "benchmark_return_pct", "benchmark_max_drawdown_mtm_pct",
        "benchmark_funding_net_cost_usd", "benchmark_min_quote_notional_usd",
        "benchmark_min_quote_check_pass", "benchmark_execution_feasibility",
        "benchmark_quote_eligible", "paired_evaluable",
        "ema_minus_benchmark_pct_points", "ema_outperforms_benchmark",
        "ema_eligible_5plus_trades",
    ]
    with open(os.path.join(out_dir, "ema_benchmark_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(pairs)

    summaries = []
    balances = sorted({float(x["start_balance"]) for x in pairs})
    for balance in balances:
        balance_rows = [x for x in pairs if float(x["start_balance"]) == balance]
        eligible_rows = [x for x in balance_rows if x["ema_eligible_5plus_trades"]]
        paired_rows = [x for x in balance_rows if x["paired_evaluable"]]
        summaries.append(_summarize_pairs(balance, "all_pairs_including_low_sample", balance_rows))
        summaries.append(_summarize_pairs(balance, "ema_eligible_5plus_trades", eligible_rows))
        summaries.append(_summarize_pairs(balance, "paired_evaluable", paired_rows))
        for symbol in sorted({x["symbol"] for x in balance_rows}):
            symbol_rows = [x for x in balance_rows if x["symbol"] == symbol]
            summaries.append(_summarize_pairs(balance, f"symbol:{symbol}:all", symbol_rows))
            summaries.append(_summarize_pairs(balance, f"symbol:{symbol}:eligible", [x for x in symbol_rows if x["ema_eligible_5plus_trades"]]))
            summaries.append(_summarize_pairs(balance, f"symbol:{symbol}:paired", [x for x in symbol_rows if x["paired_evaluable"]]))
        for window in sorted({x["window"] for x in balance_rows}):
            window_rows = [x for x in balance_rows if x["window"] == window]
            summaries.append(_summarize_pairs(balance, f"window:{window}:all", window_rows))
            summaries.append(_summarize_pairs(balance, f"window:{window}:eligible", [x for x in window_rows if x["ema_eligible_5plus_trades"]]))
            summaries.append(_summarize_pairs(balance, f"window:{window}:paired", [x for x in window_rows if x["paired_evaluable"]]))

    summary_fields = [
        "start_balance", "scope", "pairs", "ema_mean_return_pct", "ema_median_return_pct",
        "ema_positive_pairs", "benchmark_mean_return_pct", "benchmark_median_return_pct",
        "benchmark_positive_pairs", "mean_excess_return_pct_points",
        "median_excess_return_pct_points", "ema_outperforms_benchmark_pairs",
        "ema_eligible_pairs", "benchmark_quote_eligible_pairs", "paired_evaluable_pairs",
        "mean_ema_mtm_drawdown_pct", "mean_benchmark_mtm_drawdown_pct",
    ]
    with open(os.path.join(out_dir, "ema_benchmark_comparison_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    return pairs, summaries

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--max-markets", type=int, default=5)
    p.add_argument("--out", default="ema-benchmark-results")
    p.add_argument("--balances", default="5,50,100")
    args = p.parse_args()
    if args.days < 365:
        p.error("Use at least 365 days for warm-up and eight independent 30-day OOS windows.")
    balances = [float(x.strip()) for x in args.balances.split(",") if x.strip()]
    if not balances or any(x <= 0 for x in balances):
        p.error("--balances must contain positive numbers")
    os.makedirs(args.out, exist_ok=True)
    end = int(time.time())
    start = end - args.days * 86400
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    rows, errors = [], []
    windows = [(f"oos_{i+1:02d}", end - (8-i)*30*86400, end - (7-i)*30*86400) for i in range(8)]
    for market in markets:
        symbol, mid = market["symbol"], market["id"]
        try:
            candles = fetch_candles(mid, "1h", start, end)
            funding = fetch_funding_rates(mid, start, end)
            if len(candles) < args.days * 24 * 0.8:
                rows.append({"symbol": symbol, "strategy": "ALL", "status": "incomplete_candles", "bars_1h": len(candles)})
                continue
            if not funding:
                rows.append({"symbol": symbol, "strategy": "ALL", "status": "missing_funding_history", "bars_1h": len(candles)})
                continue
            for window, lo, hi in windows:
                for balance in balances:
                    row = simulate_with_funding(symbol, candles, "ema_20_50_cross", lo, hi, funding, start_balance=balance)
                    row.update({"window": window, "benchmark": False, "bars_1h": len(candles),
                                "funding_coverage_pct": round(100 * len([t for t in funding if start <= t < end]) / (args.days*24), 2)})
                    rows.append(row)
                    bench = buy_hold(symbol, candles, lo, hi, funding, balance)
                    bench.update({"strategy": "buy_hold_long_1x", "window": window, "benchmark": True,
                                  "funding_coverage_pct": round(100 * len([t for t in funding if lo <= t < hi]) / max(1, ((hi-lo)//3600)), 2)})
                    rows.append(bench)
        except Exception as exc:
            errors.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:200]})
        time.sleep(0.15)
    fields = sorted({k for r in rows for k in r})
    with open(os.path.join(args.out, "ema_benchmark_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(args.out, "errors.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["symbol", "error"])
        writer.writeheader()
        writer.writerows(errors)
    pairs, _ = write_comparison(rows, args.out)
    print(f"Rows={len(rows)}, paired benchmark rows={len(pairs)}, API failures={len(errors)}. Paper-only.")
    for strategy in ("ema_20_50_cross", "buy_hold_long_1x"):
        for balance in balances:
            subset = [r for r in rows if r.get("strategy") == strategy and r.get("start_balance") == balance and r.get("status") == "tested" and r.get("trades", 0) >= (5 if strategy == "ema_20_50_cross" else 1)]
            returns = [r["return_pct"] for r in subset]
            if returns:
                print(f"{strategy} balance={balance:g}: n={len(returns)}, mean={statistics.mean(returns):+.3f}% median={statistics.median(returns):+.3f}% positive={sum(x > 0 for x in returns)}/{len(returns)}")
            else:
                print(f"{strategy} balance={balance:g}: no adequate samples")


if __name__ == "__main__":
    main()
