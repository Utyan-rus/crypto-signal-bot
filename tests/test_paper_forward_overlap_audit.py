import unittest

from paper_forward_overlap_audit import audit_events


def entry(strategy, **overrides):
    event = {
        "event": "paper_entry",
        "event_id": f"id-{strategy}",
        "symbol": "XRP",
        "strategy": strategy,
        "side": "LONG",
        "signal_candle_ts": 100,
        "timestamp": 200,
        "entry": 1.0,
        "qty": 10.0,
        "stop": 0.95,
    }
    event.update(overrides)
    return event


class PaperForwardOverlapAuditTests(unittest.TestCase):
    def test_identical_cross_strategy_entries_are_one_execution_cluster(self):
        report = audit_events([entry("donchian"), entry("ema")])
        self.assertEqual(report["valid_paper_entry_rows"], 2)
        self.assertEqual(report["unique_signal_clusters"], 1)
        self.assertEqual(report["overlapping_signal_clusters"], 1)
        self.assertEqual(report["unique_execution_fingerprints"], 1)
        self.assertEqual(report["exact_execution_duplicate_rows"], 1)
        self.assertEqual(report["cross_strategy_exact_duplicate_groups"], 1)
        self.assertEqual(report["strategy_pair_overlap"][0]["shared_exact_execution_clusters"], 1)

    def test_same_signal_with_different_size_is_not_an_exact_execution_duplicate(self):
        report = audit_events([entry("donchian"), entry("ema", qty=12.0)])
        self.assertEqual(report["unique_signal_clusters"], 1)
        self.assertEqual(report["overlapping_signal_clusters"], 1)
        self.assertEqual(report["unique_execution_fingerprints"], 2)
        self.assertEqual(report["exact_execution_duplicate_rows"], 0)

    def test_four_shared_tracks_produce_all_six_pairwise_overlaps(self):
        report = audit_events([entry(name) for name in ("a", "b", "c", "d")])
        self.assertEqual(len(report["strategy_pair_overlap"]), 6)
        self.assertTrue(all(
            row["shared_signal_clusters"] == 1
            and row["shared_exact_execution_clusters"] == 1
            for row in report["strategy_pair_overlap"]
        ))

    def test_different_market_or_signal_candle_is_not_merged(self):
        report = audit_events([
            entry("a"),
            entry("b", symbol="BTC"),
            entry("c", signal_candle_ts=101),
        ])
        self.assertEqual(report["unique_signal_clusters"], 3)
        self.assertEqual(report["overlapping_signal_clusters"], 0)

    def test_non_entry_events_are_ignored(self):
        report = audit_events([{"event": "signal_filtered", "strategy": "a"}, entry("b")])
        self.assertEqual(report["source_event_rows"], 2)
        self.assertEqual(report["paper_entry_rows_seen"], 1)
        self.assertEqual(report["valid_paper_entry_rows"], 1)
        self.assertEqual(report["non_entry_events_ignored"], 1)

    def test_malformed_entry_is_counted_and_does_not_crash_audit(self):
        report = audit_events([{"event": "paper_entry", "symbol": "XRP"}, entry("valid")])
        self.assertEqual(report["paper_entry_rows_seen"], 2)
        self.assertEqual(report["malformed_entry_rows"], 1)
        self.assertEqual(report["valid_paper_entry_rows"], 1)

    def test_duplicate_event_ids_are_reported(self):
        first = entry("a", event_id="same-id")
        second = entry("b", event_id="same-id")
        report = audit_events([first, second])
        self.assertEqual(report["duplicate_event_ids"], 1)


if __name__ == "__main__":
    unittest.main()
