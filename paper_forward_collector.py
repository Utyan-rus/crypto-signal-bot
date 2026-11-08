"""Independent, read-only hourly paper-forward journal for the preregistered Lighter holdout.

This collector never submits orders. It advances only newly closed hourly bars,
uses the fixed preregistered strategy set, and persists state plus append-only events.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import funding_aware_research as funding
import hourly_candidate_research as hourly
from backtest import fetch_candles, get_markets
from walk_forward_research import STRATEGIES, floor_to_hour

CONFIG_PATH = Path("research/forward_holdout_2026-11-07.json")
OOS_START = 1794034800
WARMUP_START = 1783234800
START_BALANCE = 50.0
FEE = 0.0005
SLIPPAGE = 0.0005
MIN_NOTIONAL = 10.0
RISK_FRACTION = 0.005
MAX_LEVERAGE = 3.0
DIRECTION = "LONG"
TIMEFRAME = "1h"
STATE_VERSION = 1


def fresh_track():
    return {
        "balance": START_BALANCE, "position": None, "last_bar_ts": OOS_START - 3600,
        "equity_peak": START_BALANCE, "max_drawdown_mtm_pct": 0.0,
        "realized_pnl": 0.0, "funding_cost": 0.0, "funding_missing_hours": 0,
        "trades": 0, "wins": 0, "losses": 0, "gross_profit": 0.0,
        "gross_loss": 0.0, "signals_seen": 0, "min_notional_rejections": 0,
    }


def load_state(path):
    if not path.exists():
        return {"version": STATE_VERSION, "experiment_id": "prospective-120d-holdout-2026-11-07",
                "tracks": {}, "last_run_utc": None}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != STATE_VERSION:
        raise RuntimeError("Unsupported collector state version; refusing to overwrite it")
    return data


_EVENT_IDS = set()


def append_event(events_path, event):
    events_path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(event, sort_keys=True, separators=(",", ":"))
    event_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if event_id in _EVENT_IDS:
        return
    event["event_id"] = event_id
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    _EVENT_IDS.add(event_id)


def load_event_ids(events_path):
    _EVENT_IDS.clear()
    if not events_path.exists():
        return
    with events_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
                if event.get("event_id"):
                    _EVENT_IDS.add(event["event_id"])
            except json.JSONDecodeError:
                raise RuntimeError("Event journal contains malformed JSON; refusing to append")


def mark_to_market(track, close):
    pos = track["position"]
    if not pos:
        return track["balance"]
    sign = 1.0 if pos["side"] == "LONG" else -1.0
    gross = pos["qty"] * (close - pos["entry"]) * sign
    estimated_exit_cost = abs(pos["qty"] * close) * (FEE + SLIPPAGE)
    return track["balance"] + gross - estimated_exit_cost - pos["entry_fee"] - pos["funding_cost"]


def close_position(track, symbol, strategy, bar, raw_exit, reason, events_path):
    pos = track["position"]
    if not pos:
        return
    sign = 1.0 if pos["side"] == "LONG" else -1.0
    exit_price = raw_exit * (1 - SLIPPAGE if sign > 0 else 1 + SLIPPAGE)
    gross = pos["qty"] * (exit_price - pos["entry"]) * sign
    exit_fee = abs(pos["qty"] * exit_price) * FEE
    pnl = gross - exit_fee - pos["entry_fee"] - pos["funding_cost"]
    track["balance"] += pnl
    track["realized_pnl"] += pnl
    track["funding_cost"] += pos["funding_cost"]
    track["trades"] += 1
    if pnl > 0:
        track["wins"] += 1
        track["gross_profit"] += pnl
    elif pnl < 0:
        track["losses"] += 1
        track["gross_loss"] += -pnl
    append_event(events_path, {
        "event": "paper_exit", "timestamp": bar["t"], "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(bar["t"])),
        "symbol": symbol, "strategy": strategy, "side": pos["side"], "entry_time": pos["entry_time"],
        "entry": pos["entry"], "exit": exit_price, "stop": pos["stop"], "take_profit_reference": pos["take_profit_reference"],
        "qty": pos["qty"], "pnl_usd": round(pnl, 8), "funding_cost_usd": round(pos["funding_cost"], 8),
        "funding_complete": pos["funding_complete"], "reason": reason, "balance_after": round(track["balance"], 8),
    })
    track["position"] = None


def process_bar(track, symbol, strategy, candles, i, signals, exits, funding_rates, events_path):
    prev, bar = candles[i - 1], candles[i]
    pos = track["position"]

    if pos and bar["t"] > pos["entry_time"]:
        if bar["t"] in funding_rates:
            rate = float(funding_rates[bar["t"]])
            sign = 1.0 if pos["side"] == "LONG" else -1.0
            pos["funding_cost"] += pos["qty"] * bar["c"] * rate * sign
        else:
            pos["funding_complete"] = False
            track["funding_missing_hours"] += 1
            append_event(events_path, {"event": "funding_missing", "timestamp": bar["t"],
                "symbol": symbol, "strategy": strategy, "position_open": True})

    if pos:
        if pos["side"] == "LONG" and bar["l"] <= pos["stop"]:
            close_position(track, symbol, strategy, bar, pos["stop"], "STOP_LOSS", events_path)
        else:
            lo, hi = exits.get(prev["t"], (None, None))
            if pos and pos["side"] == "LONG" and lo is not None and prev["c"] < lo:
                close_position(track, symbol, strategy, bar, bar["o"], "MODEL_CHANNEL_EXIT", events_path)

    signal = signals.get(prev["t"])
    previous = signals.get(candles[i - 2]["t"]) if i >= 2 else None
    fresh = bool(signal) and (not previous or previous[0] != signal[0])
    if fresh:
        track["signals_seen"] += 1
        side, stop_dist = signal
        if side != DIRECTION:
            append_event(events_path, {"event": "signal_filtered", "timestamp": bar["t"],
                "signal_candle_ts": prev["t"], "symbol": symbol, "strategy": strategy,
                "side": side, "reason": "preregistered_direction_is_long"})
        elif track["position"] is None and track["balance"] > 0:
            entry = bar["o"] * (1 + SLIPPAGE)
            qty = hourly.size_position(track["balance"], entry, stop_dist)
            if qty <= 0 or qty * entry < MIN_NOTIONAL:
                track["min_notional_rejections"] += 1
                append_event(events_path, {"event": "entry_rejected", "timestamp": bar["t"],
                    "signal_candle_ts": prev["t"], "symbol": symbol, "strategy": strategy,
                    "side": side, "reference_entry": entry, "stop_distance": stop_dist,
                    "reason": "risk_budget_or_minimum_notional", "min_notional_usd": MIN_NOTIONAL})
            else:
                stop = entry - stop_dist
                # Diagnostic 3R target only; the fixed strategy exits by stop/channel, not this target.
                target = entry + 3.0 * stop_dist
                track["position"] = {"side": side, "entry": entry, "qty": qty, "stop": stop,
                    "take_profit_reference": target, "entry_time": bar["t"],
                    "entry_fee": qty * entry * FEE, "funding_cost": 0.0, "funding_complete": True}
                append_event(events_path, {"event": "paper_entry", "timestamp": bar["t"],
                    "signal_candle_ts": prev["t"], "symbol": symbol, "strategy": strategy,
                    "side": side, "entry": entry, "stop": stop, "take_profit_reference_3R": target,
                    "take_profit_is_executed": False, "qty": qty, "notional_usd": qty * entry,
                    "risk_budget_usd": track["balance"] * RISK_FRACTION,
                    "fee_per_side": FEE, "slippage_per_side": SLIPPAGE, "min_notional_usd": MIN_NOTIONAL})
                if bar["l"] <= stop:
                    close_position(track, symbol, strategy, bar, stop, "SAME_BAR_STOP_LOSS", events_path)

    mtm = mark_to_market(track, bar["c"])
    track["equity_peak"] = max(track["equity_peak"], mtm)
    if track["equity_peak"] > 0:
        dd = 100.0 * (track["equity_peak"] - mtm) / track["equity_peak"]
        track["max_drawdown_mtm_pct"] = max(track["max_drawdown_mtm_pct"], dd)
    track["last_bar_ts"] = bar["t"]


def collect(output_dir, state_path):
    output_dir = Path(output_dir)
    state_path = Path(state_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_path = output_dir / "events.jsonl"
    load_event_ids(events_path)
    state = load_state(state_path)
    now = int(time.time())
    end_ts = floor_to_hour(now)
    markets = get_markets()
    fixed_symbols = ["BTC", "ETH", "SOL", "BNB", "XRP"]
    def base_symbol(symbol):
        return symbol.upper().replace("-PERP", "").replace("/USDT", "").replace("/USDC", "")
    by_base = {base_symbol(m["symbol"]): m for m in markets}
    selected = [by_base[s] for s in fixed_symbols if s in by_base]
    if [base_symbol(m["symbol"]) for m in selected] != fixed_symbols:
        raise RuntimeError("Preregistered five-market universe is incomplete; collector failed closed")
    failures = []
    for market in selected:
        symbol = market["symbol"]
        key = base_symbol(symbol)
        try:
            candles = fetch_candles(market["id"], TIMEFRAME, WARMUP_START, end_ts)
            if len(candles) < 125 * 24:
                raise RuntimeError(f"insufficient warmup/history: {len(candles)} hourly bars")
            gaps = [(candles[j - 1]["t"], candles[j]["t"]) for j in range(1, len(candles))
                    if candles[j]["t"] - candles[j - 1]["t"] != 3600
                    and candles[j]["t"] >= OOS_START]
            if gaps:
                raise RuntimeError(f"hourly candle history has {len(gaps)} timestamp gaps; first={gaps[0]}")
            tracks_for_market = [f"{key}|{name}" for name in STRATEGIES]
            earliest = min(int(state["tracks"].get(k, fresh_track())["last_bar_ts"]) for k in tracks_for_market)
            rates = funding.fetch_funding_rates(market["id"], max(OOS_START, earliest), end_ts)
            for strategy in STRATEGIES:
                track_key = f"{key}|{strategy}"
                track = state["tracks"].setdefault(track_key, fresh_track())
                signals, exits = funding.build_model(candles, strategy)
                first_new = next((i for i, b in enumerate(candles)
                                  if b["t"] > track["last_bar_ts"]), None)
                if first_new is None:
                    continue
                for i in range(max(1, first_new), len(candles)):
                    bar = candles[i]
                    if bar["t"] < OOS_START or bar["t"] >= end_ts:
                        continue
                    if bar["t"] <= track["last_bar_ts"]:
                        continue
                    process_bar(track, key, strategy, candles, i, signals, exits, rates, events_path)
        except Exception as exc:
            failures.append({"symbol": key, "error": type(exc).__name__ + ": " + str(exc)[:240]})
    state["last_run_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    temp = state_path.with_suffix(state_path.suffix + ".tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(state_path)
    summary = []
    for key, track in sorted(state["tracks"].items()):
        mtm = track["balance"]
        if track["position"]:
            last = next((m for m in selected if base_symbol(m["symbol"]) == key.split("|", 1)[0]), None)
            if last:
                candles = fetch_candles(last["id"], TIMEFRAME, max(WARMUP_START, end_ts - 7200), end_ts)
                if candles:
                    mtm = mark_to_market(track, candles[-1]["c"])
        summary.append({"market_strategy": key, "realized_balance_usd": round(track["balance"], 8),
            "mtm_equity_usd": round(mtm, 8), "realized_pnl_usd": round(track["realized_pnl"], 8),
            "trades": track["trades"], "wins": track["wins"], "losses": track["losses"],
            "profit_factor": round(track["gross_profit"] / track["gross_loss"], 6) if track["gross_loss"] else None,
            "max_drawdown_mtm_pct": round(track["max_drawdown_mtm_pct"], 6),
            "funding_missing_hours": track["funding_missing_hours"],
            "open_position": track["position"], "signals_seen": track["signals_seen"],
            "min_notional_rejections": track["min_notional_rejections"]})
    (output_dir / "summary.json").write_text(json.dumps({
        "experiment_id": state["experiment_id"], "as_of_utc": state["last_run_utc"],
        "oos_start_utc": "2026-11-07T07:00:00Z", "fixed_universe": fixed_symbols,
        "strategies": list(STRATEGIES), "direction": DIRECTION, "timeframe": TIMEFRAME,
        "start_balance_per_market_strategy_usd": START_BALANCE, "risk_fraction": RISK_FRACTION,
        "max_leverage": MAX_LEVERAGE, "min_notional_usd": MIN_NOTIONAL,
        "fee_per_side": FEE, "slippage_per_side": SLIPPAGE,
        "funding_missing_means_pnl_is_incomplete": True,
        "take_profit_note": "3R is logged as a reference only; it is not an execution rule in the preregistered strategies.",
        "real_orders_enabled": False, "failures": failures, "tracks": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise RuntimeError(f"Collector completed with {len(failures)} market errors; see summary.json")
    print(f"Paper-forward collector updated {len(summary)} tracks; OOS bars through {state['last_run_utc']}.")
    print("No real orders. Funding gaps are flagged and make affected PnL incomplete.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="paper-forward-data")
    parser.add_argument("--state", default="paper-forward-data/state.json")
    args = parser.parse_args()
    collect(args.output_dir, args.state)


if __name__ == "__main__":
    main()
