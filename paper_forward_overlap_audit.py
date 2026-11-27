#!/usr/bin/env python3
"""Audit cross-strategy overlap in paper-forward entry events.

This is a diagnostic only: it does not change signals, positions, or OOS parameters.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable


def _finite_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _number_key(value: float) -> str:
    # Normalize harmless JSON float formatting without merging distinct executions.
    return format(value, ".12g")


def _parse_entry(event: dict[str, Any]) -> dict[str, Any]:
    symbol = str(event.get("symbol", "")).strip().upper()
    strategy = str(event.get("strategy", "")).strip()
    side = str(event.get("side", "")).strip().upper()
    if not symbol:
        raise ValueError("symbol is required")
    if not strategy:
        raise ValueError("strategy is required")
    if side not in {"LONG", "SHORT"}:
        raise ValueError("side must be LONG or SHORT")
    try:
        signal_ts = int(event["signal_candle_ts"])
        entry_ts = int(event["timestamp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("signal_candle_ts and timestamp must be integers") from exc
    entry = _finite_number(event.get("entry"), "entry")
    qty = _finite_number(event.get("qty"), "qty")
    stop = _finite_number(event.get("stop"), "stop")
    if entry <= 0 or qty <= 0 or stop <= 0:
        raise ValueError("entry, qty, and stop must be positive")
    return {
        "symbol": symbol, "strategy": strategy, "side": side,
        "signal_candle_ts": signal_ts, "entry_timestamp": entry_ts,
        "entry": entry, "qty": qty, "stop": stop,
    }


def audit_events(events: Iterable[dict[str, Any]], malformed_json_lines: int = 0) -> dict[str, Any]:
    """Return signal-cluster and exact-execution overlap diagnostics."""
    signal_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    execution_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    event_ids: list[str] = []
    total_events = paper_entry_rows_seen = malformed_entry_rows = 0

    for event in events:
        total_events += 1
        if not isinstance(event, dict):
            continue
        event_id = str(event.get("event_id", "")).strip()
        if event_id:
            event_ids.append(event_id)
        if event.get("event") != "paper_entry":
            continue
        paper_entry_rows_seen += 1
        try:
            entry = _parse_entry(event)
        except ValueError:
            malformed_entry_rows += 1
            continue
        signal_key = (entry["symbol"], entry["signal_candle_ts"], entry["side"])
        execution_key = signal_key + (
            entry["entry_timestamp"], _number_key(entry["entry"]),
            _number_key(entry["qty"]), _number_key(entry["stop"]),
        )
        signal_groups[signal_key].append(entry)
        execution_groups[execution_key].append(entry)

    overlap_rows: list[dict[str, Any]] = []
    pair_counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"shared_signal_clusters": 0, "shared_exact_execution_clusters": 0}
    )
    overlapping_signal_clusters = signal_overlap_rows = 0

    for key, group in sorted(signal_groups.items()):
        strategies = sorted({item["strategy"] for item in group})
        signal_overlap_rows += max(0, len(group) - 1)
        if len(strategies) > 1:
            overlapping_signal_clusters += 1
            for left, right in combinations(strategies, 2):
                pair_counts[(left, right)]["shared_signal_clusters"] += 1
            overlap_rows.append({
                "symbol": key[0],
                "signal_candle_ts": key[1],
                "side": key[2],
                "strategy_count": len(strategies),
                "strategies": ";".join(strategies),
                "entry_rows": len(group),
                "unique_execution_fingerprints": len({
                    (item["entry_timestamp"], _number_key(item["entry"]),
                     _number_key(item["qty"]), _number_key(item["stop"]))
                    for item in group
                }),
                "entry_timestamps": ";".join(str(x) for x in sorted({v["entry_timestamp"] for v in group})),
                "entry_prices": ";".join(sorted({_number_key(v["entry"]) for v in group})),
                "quantities": ";".join(sorted({_number_key(v["qty"]) for v in group})),
                "stops": ";".join(sorted({_number_key(v["stop"]) for v in group})),
            })

    exact_duplicate_groups = exact_execution_duplicate_rows = 0
    cross_strategy_exact_duplicate_groups = cross_strategy_duplicate_entry_rows = 0
    for group in execution_groups.values():
        if len(group) > 1:
            exact_duplicate_groups += 1
            exact_execution_duplicate_rows += len(group) - 1
            strategies = sorted({item["strategy"] for item in group})
            if len(strategies) > 1:
                cross_strategy_exact_duplicate_groups += 1
                cross_strategy_duplicate_entry_rows += len(group) - 1
                for left, right in combinations(strategies, 2):
                    pair_counts[(left, right)]["shared_exact_execution_clusters"] += 1

    strategy_names = sorted({
        item["strategy"] for group in signal_groups.values() for item in group
    })
    pair_rows = [
        {"strategy_a": left, "strategy_b": right, **counts}
        for (left, right), counts in sorted(pair_counts.items())
    ]
    duplicate_event_ids = len(event_ids) - len(set(event_ids))
    valid_entry_rows = sum(len(group) for group in signal_groups.values())

    return {
        "schema_version": 1,
        "source_event_rows": total_events,
        "paper_entry_rows_seen": paper_entry_rows_seen,
        "valid_paper_entry_rows": valid_entry_rows,
        "non_entry_events_ignored": total_events - paper_entry_rows_seen,
        "malformed_json_lines": malformed_json_lines,
        "malformed_entry_rows": malformed_entry_rows,
        "duplicate_event_ids": duplicate_event_ids,
        "strategy_count": len(strategy_names),
        "strategy_names": strategy_names,
        "unique_signal_clusters": len(signal_groups),
        "overlapping_signal_clusters": overlapping_signal_clusters,
        "signal_overlap_rows": signal_overlap_rows,
        "unique_execution_fingerprints": len(execution_groups),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_execution_duplicate_rows": exact_execution_duplicate_rows,
        "cross_strategy_exact_duplicate_groups": cross_strategy_exact_duplicate_groups,
        "cross_strategy_duplicate_entry_rows": cross_strategy_duplicate_entry_rows,
        "overlap_groups": overlap_rows,
        "strategy_pair_overlap": pair_rows,
        "interpretation": (
            "Diagnostic only. Shared signal clusters and exact execution duplicates are correlated "
            "across strategy tracks; do not count them as independent portfolio trades. "
            "Per-strategy OOS balances remain separate and unchanged."
        ),
    }


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    events: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(value, dict):
                events.append(value)
            else:
                malformed += 1
    return events, malformed


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in report.items() if k not in {"overlap_groups", "strategy_pair_overlap"}}
    (output_dir / "overlap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    overlap_fields = [
        "symbol", "signal_candle_ts", "side", "strategy_count", "strategies", "entry_rows",
        "unique_execution_fingerprints", "entry_timestamps", "entry_prices", "quantities", "stops",
    ]
    with (output_dir / "overlap_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=overlap_fields)
        writer.writeheader()
        writer.writerows(report["overlap_groups"])
    pair_fields = [
        "strategy_a", "strategy_b", "shared_signal_clusters", "shared_exact_execution_clusters",
    ]
    with (output_dir / "strategy_pair_overlap.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(report["strategy_pair_overlap"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="paper-forward events.jsonl")
    parser.add_argument("--out", type=Path, default=Path("paper-forward-overlap-results"))
    args = parser.parse_args()
    events, malformed = read_jsonl(args.input)
    report = audit_events(events, malformed_json_lines=malformed)
    write_outputs(report, args.out)
    print(
        "PAPER-FORWARD OVERLAP AUDIT: "
        f"entries={report['valid_paper_entry_rows']}, "
        f"unique_signals={report['unique_signal_clusters']}, "
        f"overlapping_signals={report['overlapping_signal_clusters']}, "
        f"exact_duplicate_rows={report['exact_execution_duplicate_rows']}, "
        f"malformed={report['malformed_json_lines'] + report['malformed_entry_rows']}"
    )


if __name__ == "__main__":
    main()
