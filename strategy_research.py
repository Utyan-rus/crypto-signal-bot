"""Walk-forward parameter research for the Lighter paper strategy.

Uses the first 60 days only to select a take-profit multiple, then evaluates
that single selection on the final 30 days. No live orders are ever sent.
"""
import argparse
import csv
import json
import os
import time

from backtest import backtest, fetch_candles, get_markets

RR_GRID = (1.5, 2.0, 2.5, 3.0)


def split_history(candles5, candles15, test_days=30):
    if not candles5 or not candles15:
        return None
    cutoff = candles5[-1]["t"] - test_days * 86400
    train5 = [c for c in candles5 if c["t"] < cutoff]
    test5 = [c for c in candles5 if c["t"] >= cutoff]
    train15 = [c for c in candles15 if c["t"] < cutoff]
    test15 = [c for c in candles15 if c["t"] >= cutoff]
    if len(train5) < 220 * 3 or len(train15) < 220 or len(test5) < 220 or len(test15) < 220:
        return None
    return (train5, train15, test5, test15)


def select_candidate(rows):
    """Select only among sufficiently sampled candidates; prefer lower drawdown."""
    eligible = [r for r in rows if r["trades"] >= 10 and r["max_drawdown_pct"] <= 20]
    if not eligible:
        return None
    # Avoid selecting a candidate solely on return: penalize drawdown and tiny samples.
    return max(eligible, key=lambda r: (
        r["return_pct"] - 0.5 * r["max_drawdown_pct"],
        r["trades"],
    ))


def research_market(symbol, candles5, candles15):
    split = split_history(candles5, candles15)
    if split is None:
        return {"symbol": symbol, "status": "insufficient_split_history"}
    train5, train15, test5, test15 = split
    candidates = []
    for rr in RR_GRID:
        _, summary = backtest(symbol, train5, train15, rr=rr)
        candidates.append({
            "symbol": symbol, "rr": rr, "trades": summary.get("trades", 0),
            "return_pct": summary.get("return_pct", 0),
            "max_drawdown_pct": summary.get("max_drawdown_pct", 0),
            "end_balance": summary.get("end_balance", 5.0),
        })
    selected = select_candidate(candidates)
    if selected is None:
        return {
            "symbol": symbol, "status": "no_eligible_candidate",
            "training_candidates": candidates,
            "test_trades": 0, "test_return_pct": None,
            "test_max_drawdown_pct": None,
        }
    test_trades, test_summary = backtest(symbol, test5, test15, rr=selected["rr"])
    return {
        "symbol": symbol, "status": "tested",
        "selected_rr": selected["rr"],
        "training_trades": selected["trades"],
        "training_return_pct": selected["return_pct"],
        "training_max_drawdown_pct": selected["max_drawdown_pct"],
        "test_trades": test_summary.get("trades", 0),
        "test_wins": test_summary.get("wins", 0),
        "test_win_rate_pct": test_summary.get("win_rate_pct", 0),
        "test_start_balance": test_summary.get("start_balance", 5.0),
        "test_end_balance": test_summary.get("end_balance", 5.0),
        "test_return_pct": test_summary.get("return_pct", 0),
        "test_max_drawdown_pct": test_summary.get("max_drawdown_pct", 0),
        "test_net_pnl": test_summary.get("net_pnl", 0),
        "training_candidates": candidates,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--max-markets", type=int, default=5)
    parser.add_argument("--out", default="strategy-research-results")
    args = parser.parse_args()
    if args.days != 90:
        parser.error("Walk-forward research currently requires --days 90 (60 train + 30 test).")
    os.makedirs(args.out, exist_ok=True)
    markets = get_markets()
    if args.max_markets > 0:
        markets = markets[:args.max_markets]
    end = int(time.time())
    start = end - 90 * 86400
    results, failures = [], []
    print(f"Walk-forward paper research: {len(markets)} markets, 60 days train + 30 days out-of-sample.")
    for idx, market in enumerate(markets, 1):
        print(f"[{idx}/{len(markets)}] {market['symbol']}")
        try:
            candles5 = fetch_candles(market["id"], "5m", start, end)
            candles15 = fetch_candles(market["id"], "15m", start, end)
            if len(candles5) < 90 * 288 * 0.9 or len(candles15) < 90 * 96 * 0.9:
                results.append({"symbol": market["symbol"], "status": "incomplete_history"})
                continue
            results.append(research_market(market["symbol"], candles5, candles15))
            row = results[-1]
            print(f"  {row['status']}; OOS return={row.get('test_return_pct', 'n/a')}%")
        except Exception as exc:
            failures.append({"symbol": market["symbol"], "error": type(exc).__name__ + ": " + str(exc)[:180]})
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:180]}")
        time.sleep(0.15)
    flat_rows = []
    for result in results:
        flat = {k: v for k, v in result.items() if k != "training_candidates"}
        flat_rows.append(flat)
    fields = sorted({key for row in flat_rows for key in row})
    with open(os.path.join(args.out, "walk_forward_summary.csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flat_rows)
    with open(os.path.join(args.out, "candidate_details.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "failures.json"), "w", encoding="utf-8") as handle:
        json.dump(failures, handle, ensure_ascii=False, indent=2)
    tested = [r for r in results if r.get("status") == "tested"]
    profitable = [r for r in tested if r.get("test_return_pct", 0) > 0]
    print(f"Finished: {len(tested)} tested; {len(profitable)} profitable out-of-sample; {len(failures)} API failures.")
    print("Research only. The holdout result is not a guarantee and no live orders are placed.")


if __name__ == "__main__":
    main()
