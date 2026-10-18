"""Robust out-of-sample paper research with continuous indicator warm-up and mark-to-market drawdown.

This is an independent research harness. It does not place orders or alter runtime strategy.
Signals are calculated once on the full chronological candle series; test entries are gated
to the holdout interval. Fees/slippage are stress-tested. Funding remains a disclosed limitation.
"""
import argparse
import csv
import json
import os
import time

import strategy_research as base
from backtest import fetch_candles, get_markets


COST_SCENARIOS = (
    ("base", 0.0005, 0.0005),
    ("double_cost", 0.0010, 0.0010),
    ("triple_cost", 0.0015, 0.0015),
)


def max_drawdown(equity_path):
    peak = float("-inf")
    worst = 0.0
    for equity in equity_path:
        peak = max(peak, equity)
        if peak > 0:
            worst = max(worst, (peak - equity) / peak)
    return worst * 100.0


def simulate_oos(symbol, candles, signals, holdout_start, strategy, cooldown, fresh,
                 fee_rate, slippage_rate, exit_mode="fixed_rr", max_hold_bars=0):
    balance = base.START_BALANCE
    pos = None
    last_exit = -10**9
    trades = []
    equity_path = [balance]
    for i in range(1, len(candles)):
        prev, bar = candles[i - 1], candles[i]
        if bar["t"] < holdout_start:
            continue
        closed = False
        if pos:
            exit_price, reason = None, None
            # Use only the previous completed bar's favorable extreme to update a trailing stop.
            if exit_mode == "trailing_1r" and pos.get("best_price") is not None:
                if pos["side"] == "LONG":
                    pos["stop"] = max(pos["stop"], pos["best_price"] - pos["stop_distance"])
                else:
                    pos["stop"] = min(pos["stop"], pos["best_price"] + pos["stop_distance"])
            if pos["side"] == "LONG":
                if bar["l"] <= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["h"] >= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            else:
                if bar["h"] >= pos["stop"]:
                    exit_price, reason = pos["stop"], "SL"
                elif bar["l"] <= pos["target"]:
                    exit_price, reason = pos["target"], "TP"
            if exit_price is None and max_hold_bars and i - pos["entry_index"] >= max_hold_bars:
                exit_price, reason = bar["c"], "TIME"
            if exit_price is not None:
                exit_price *= (1 - slippage_rate if pos["side"] == "LONG" else 1 + slippage_rate)
                gross = pos["qty"] * (exit_price - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
                pnl = gross - abs(pos["qty"] * exit_price) * fee_rate - pos["entry_fee"]
                balance += pnl
                trades.append({"symbol": symbol, "strategy": strategy, "pnl": pnl, "reason": reason})
                pos = None
                last_exit = i
                closed = True
            elif pos:
                # Update favorable excursion only after deciding this bar's exits.
                pos["best_price"] = (max(pos["best_price"], bar["h"]) if pos["side"] == "LONG"
                                     else min(pos["best_price"], bar["l"]))

        sig = signals.get(prev["t"])
        prevprev = signals.get(candles[i-2]["t"]) if i >= 2 else None
        fresh_now = bool(sig and (not prevprev or prevprev[0] != sig[0]))
        if (pos is None and not closed and i-last_exit > cooldown and sig
                and balance > 0 and (not fresh or fresh_now)):
            side, stop_distance, rr = sig
            entry = bar["o"] * (1 + slippage_rate if side == "LONG" else 1 - slippage_rate)
            qty = min(balance * base.RISK_FRACTION /
                      (stop_distance + entry * 2 * (fee_rate + slippage_rate)),
                      balance * base.LEVERAGE / entry)
            if qty > 0 and qty * entry >= base.MIN_NOTIONAL:
                stop = entry - stop_distance if side == "LONG" else entry + stop_distance
                target = entry + stop_distance * rr if side == "LONG" else entry - stop_distance * rr
                pos = {"side": side, "entry": entry, "qty": qty, "stop": stop,
                       "target": target, "entry_fee": qty * entry * fee_rate,
                       "entry_index": i, "stop_distance": stop_distance, "best_price": entry}
                # Conservative intrabar convention: stop takes precedence if both levels hit.
                if side == "LONG":
                    reason = "SL" if bar["l"] <= stop else ("TP" if bar["h"] >= target else None)
                else:
                    reason = "SL" if bar["h"] >= stop else ("TP" if bar["l"] <= target else None)
                if reason:
                    exit_price = stop if reason == "SL" else target
                    exit_price *= (1 - slippage_rate if side == "LONG" else 1 + slippage_rate)
                    gross = qty * (exit_price - entry) * (1 if side == "LONG" else -1)
                    balance += gross - abs(qty * exit_price) * fee_rate - pos["entry_fee"]
                    trades.append({"symbol": symbol, "strategy": strategy,
                                   "pnl": gross - abs(qty * exit_price) * fee_rate - pos["entry_fee"],
                                   "reason": reason})
                    pos = None
                    last_exit = i

        # Mark-to-market equity includes unrealized PnL and entry fee while position is open.
        equity = balance
        if pos:
            mark = bar["c"]
            unrealized = pos["qty"] * (mark - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
            estimated_exit_cost = abs(pos["qty"] * mark) * (fee_rate + slippage_rate)
            equity += unrealized - pos["entry_fee"] - estimated_exit_cost
        equity_path.append(equity)

    if pos:
        bar = candles[-1]
        exit_price = bar["c"] * (1 - slippage_rate if pos["side"] == "LONG" else 1 + slippage_rate)
        gross = pos["qty"] * (exit_price - pos["entry"]) * (1 if pos["side"] == "LONG" else -1)
        pnl = gross - abs(pos["qty"] * exit_price) * fee_rate - pos["entry_fee"]
        balance += pnl
        trades.append({"symbol": symbol, "strategy": strategy, "pnl": pnl, "reason": "END"})
        equity_path.append(balance)

    wins = sum(t["pnl"] > 0 for t in trades)
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = -sum(t["pnl"] for t in trades if t["pnl"] < 0)
    return {
        "symbol": symbol, "strategy": strategy, "exit_mode": exit_mode,
        "max_hold_bars": max_hold_bars, "trades": len(trades), "wins": wins,
        "win_rate_pct": round(100 * wins / len(trades), 2) if trades else 0.0,
        "start_balance": base.START_BALANCE, "end_balance": round(balance, 6),
        "return_pct": round(100 * (balance / base.START_BALANCE - 1), 3),
        "max_drawdown_mtm_pct": round(max_drawdown(equity_path), 3),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (999.0 if gross_profit else None),
        "funding": "excluded; historical funding series not integrated",
        "status": "tested" if len(trades) >= 5 else "low_sample",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--max-markets", type=int, default=5)
    p.add_argument("--out", default="robust-research-results")
    args = p.parse_args()
    if args.days != 90:
        p.error("Requires 90 days: 60-day train/context + 30-day holdout.")
    os.makedirs(args.out, exist_ok=True)
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    end = int(time.time())
    start = end - 90 * 86400
    holdout_start = end - 30 * 86400
    results, failures = [], []
    for n, market in enumerate(markets, 1):
        symbol = market["symbol"]
        print(f"[{n}/{len(markets)}] {symbol}")
        try:
            candles = fetch_candles(market["id"], "5m", start, end)
            if len(candles) < 90 * 288 * 0.9:
                results.append({"symbol": symbol, "status": "incomplete_history", "bars_5m": len(candles)})
                continue
            # Critical correction: calculate indicators on full history, then gate test entries.
            all_signals = {k: v for k, v in base.signal_map(candles, "rsi_bollinger").items()}
            for strategy in base.STRATEGIES:
                signals = all_signals if strategy == "rsi_bollinger" else base.signal_map(candles, strategy)
                for mode, cooldown, fresh in (("fresh_signal", 6, True),):
                    # Hold entry rules constant while isolating exit management.
                    exit_variants = (("fixed_rr", 0), ("time_12_bars", 12),
                                     ("time_24_bars", 24), ("time_48_bars", 48),
                                     ("trailing_1r", 0))
                    for exit_mode, max_hold in exit_variants:
                        for cost_name, fee, slip in COST_SCENARIOS:
                            row = simulate_oos(symbol, candles, signals, holdout_start, strategy,
                                               cooldown, fresh, fee, slip, exit_mode, max_hold)
                            row.update({"entry_mode": mode, "cooldown_bars": cooldown,
                                        "cost_scenario": cost_name, "fee_per_side_pct": fee*100,
                                        "slippage_per_side_pct": slip*100,
                                        "indicator_context": "full 90d; entries gated to final 30d",
                                        "funding": "excluded; funding stress still required"})
                            results.append(row)
            time.sleep(0.15)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:180]})
            print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:180]}")
    fields = sorted({key for row in results for key in row})
    with open(os.path.join(args.out, "robust_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(args.out, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    tested = [r for r in results if r.get("trades", 0) >= 5]
    print(f"Rows={len(results)}, tested={len(tested)}, API failures={len(failures)}")
    print("Paper-only. Mark-to-market drawdown; full-history indicators; funding excluded.")
if __name__ == "__main__":
    main()
