"""Test market-regime filters on top of existing OHLCV strategies.

Paper-only research: 60-day training / 30-day chronological holdout, 5m closed candles,
next-open entries, conservative SL-first handling, fees and slippage. Does not place orders
or alter the live bot. Filters are fixed in advance, not optimized on the holdout.
"""
import argparse
import csv
import json
import os
import time
from unittest.mock import patch

import strategy_research as base
from backtest import fetch_candles, get_markets, ema

REGIMES = ("none", "trend_alignment", "volatility_band", "trend_and_volatility")
ENTRY_MODES = (("every_signal", 0, False), ("every_signal", 6, False),
               ("fresh_signal", 0, True), ("fresh_signal", 6, True))


def atr_values(candles, period=14):
    out = [None] * len(candles)
    trs = []
    for i, bar in enumerate(candles):
        prev = candles[i-1]["c"] if i else bar["c"]
        trs.append(max(bar["h"]-bar["l"], abs(bar["h"]-prev), abs(bar["l"]-prev)))
        if i >= period:
            out[i] = sum(trs[i-period+1:i+1]) / period
    return out


def filter_signals(candles, signals, regime):
    if regime == "none":
        return signals
    close = [c["c"] for c in candles]
    e200 = ema(close, 200)
    atr = atr_values(candles)
    index = {c["t"]: i for i, c in enumerate(candles)}
    out = {}
    for ts, signal in signals.items():
        i = index.get(ts)
        if i is None or e200[i] is None or atr[i] is None or close[i] <= 0:
            continue
        side, stop_distance, rr = signal
        trend_ok = False
        if i >= 12 and e200[i-12] is not None:
            slope_up = e200[i] > e200[i-12]
            slope_down = e200[i] < e200[i-12]
            trend_ok = ((side == "LONG" and close[i] > e200[i] and slope_up) or
                        (side == "SHORT" and close[i] < e200[i] and slope_down))
        atr_pct = atr[i] / close[i]
        # Fixed broad band, chosen before evaluation; excludes very quiet/noisy extremes.
        vol_ok = 0.0015 <= atr_pct <= 0.012
        if regime == "trend_alignment" and not trend_ok:
            continue
        if regime == "volatility_band" and not vol_ok:
            continue
        if regime == "trend_and_volatility" and not (trend_ok and vol_ok):
            continue
        out[ts] = signal
    return out


def simulate_regime(symbol, candles, strategy, regime, cooldown, fresh):
    raw = base.signal_map(candles, strategy)
    filtered = filter_signals(candles, raw, regime)
    with patch.object(base, "signal_map", return_value=filtered):
        return base.simulate(symbol, candles, strategy, cooldown, fresh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="regime-research-results")
    args = parser.parse_args()
    if args.days != 90:
        parser.error("Requires 90 days (60 train + 30 out-of-sample).")
    os.makedirs(args.out, exist_ok=True)
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    end = int(time.time())
    start = end - 90*86400
    results, failures = [], []
    print(f"Regime study: {len(base.STRATEGIES)} strategies x {len(REGIMES)} regimes x "
          f"{len(ENTRY_MODES)} entry controls x {len(markets)} markets.")
    for n, market in enumerate(markets, 1):
        symbol = market["symbol"]
        print(f"[{n}/{len(markets)}] {symbol}")
        try:
            candles = fetch_candles(market["id"], "5m", start, end)
            if len(candles) < 90*288*0.9:
                results.append({"symbol": symbol, "status": "incomplete_history", "bars_5m": len(candles)})
                continue
            split = base.split_history(candles)
            if not split:
                results.append({"symbol": symbol, "status": "insufficient_split_history"})
                continue
            train, test = split
            for strategy in base.STRATEGIES:
                for regime in REGIMES:
                    for mode, cooldown, fresh in ENTRY_MODES:
                        tr = simulate_regime(symbol, train, strategy, regime, cooldown, fresh)
                        te = simulate_regime(symbol, test, strategy, regime, cooldown, fresh)
                        results.append({
                            "symbol": symbol, "strategy": strategy, "regime": regime,
                            "entry_mode": mode, "cooldown_bars": cooldown,
                            "status": te["status"], "train_trades": tr["trades"],
                            "train_return_pct": tr["return_pct"],
                            "test_trades": te["trades"], "test_wins": te["wins"],
                            "test_win_rate_pct": te["win_rate_pct"],
                            "test_return_pct": te["return_pct"],
                            "test_max_drawdown_pct": te["max_drawdown_pct"],
                            "test_profit_factor": te["profit_factor"],
                            "test_start_balance": base.START_BALANCE,
                            "test_end_balance": te["end_balance"],
                            "fees_per_side_pct": base.FEE*100,
                            "slippage_per_side_pct": base.SLIPPAGE*100,
                            "funding": "not included",
                        })
            time.sleep(0.15)
        except Exception as exc:
            failures.append({"symbol": symbol, "error": type(exc).__name__ + ": " + str(exc)[:180]})
            print(f"FAILED {symbol}: {type(exc).__name__}: {str(exc)[:180]}")
    fields = sorted({key for row in results for key in row})
    with open(os.path.join(args.out, "regime_summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    with open(os.path.join(args.out, "regime_details.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "failures.json"), "w", encoding="utf-8") as f:
        json.dump(failures, f, ensure_ascii=False, indent=2)
    tested = [r for r in results if r.get("test_trades", 0) >= 5]
    positive = [r for r in tested if r["test_return_pct"] > 0]
    print(f"Finished {len(tested)} holdouts; positive={len(positive)}; API failures={len(failures)}.")
    print("No orders placed. Fixed filters; 60d train/30d holdout. Funding excluded. "
          "Results are exploratory and do not approve a live strategy.")


if __name__ == "__main__":
    main()
