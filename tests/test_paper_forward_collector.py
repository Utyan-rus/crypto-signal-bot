import json
import tempfile
import unittest
from pathlib import Path

import paper_forward_collector as collector


class PaperForwardCollectorTests(unittest.TestCase):
    def test_state_initializes_with_fixed_paper_balance_and_checkpoint(self):
        track = collector.fresh_track()
        self.assertEqual(track["balance"], 50.0)
        self.assertEqual(track["last_bar_ts"], collector.OOS_START - 3600)
        self.assertIsNone(track["position"])

    def test_state_version_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"version": 999}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                collector.load_state(path)

    def test_mark_to_market_includes_estimated_exit_costs_and_entry_fee(self):
        track = collector.fresh_track()
        track["position"] = {"side": "LONG", "entry": 100.0, "qty": 1.0,
            "entry_fee": 0.05, "funding_cost": 0.0}
        mtm = collector.mark_to_market(track, 101.0)
        self.assertAlmostEqual(mtm, 50.849, places=6)

    def test_event_journal_deduplicates_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            collector.load_event_ids(path)
            event = {"event": "paper_entry", "timestamp": 123, "symbol": "BTC"}
            collector.append_event(path, dict(event))
            collector.append_event(path, dict(event))
            self.assertEqual(len(path.read_text(encoding="utf-8").splitlines()), 1)

    def test_take_profit_is_explicitly_reference_only(self):
        self.assertEqual(collector.DIRECTION, "LONG")
        self.assertEqual(collector.MIN_NOTIONAL, 10.0)
        self.assertEqual(collector.RISK_FRACTION, 0.005)
        self.assertEqual(collector.MAX_LEVERAGE, 3.0)


    def test_funding_coverage_reports_missing_rows_without_open_positions(self):
        track = collector.fresh_track()
        self.assertIsNone(track["position"])
        start = collector.OOS_START
        rates = {start: 0.0001, start + 3600: 0.0001}
        rows = collector.build_funding_coverage_rows("BTC", rates, start + 3 * 3600)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["expected_funding_hours"], 3)
        self.assertEqual(rows[0]["funding_rows"], 2)
        self.assertEqual(rows[0]["missing_funding_hours"], 1)
        self.assertAlmostEqual(rows[0]["funding_coverage_pct"], 66.667, places=3)
        self.assertFalse(rows[0]["window_complete"])
        self.assertEqual(rows[0]["status"], "provisional")
        self.assertIsNone(rows[0]["acceptable"])

    def test_partial_window_with_complete_observed_rows_remains_provisional(self):
        start = collector.OOS_START
        rates = {start + i * 3600: 0.0001 for i in range(3)}
        rows = collector.build_funding_coverage_rows("BTC", rates, start + 3 * 3600)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["funding_rows"], 3)
        self.assertEqual(rows[0]["expected_funding_hours"], 3)
        self.assertEqual(rows[0]["funding_coverage_pct"], 100.0)
        self.assertFalse(rows[0]["window_complete"])
        self.assertEqual(rows[0]["status"], "provisional")
        self.assertIsNone(rows[0]["acceptable"])

    def test_complete_window_with_one_missing_funding_row_passes_99_percent_gate(self):
        start = collector.OOS_START
        rates = {
            start + i * 3600: 0.0001
            for i in range(collector.OOS_WINDOW_SECONDS // 3600)
            if i != 1
        }
        rows = collector.build_funding_coverage_rows(
            "BTC", rates, start + collector.OOS_WINDOW_SECONDS
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["window_complete"])
        self.assertEqual(rows[0]["funding_rows"], 719)
        self.assertEqual(rows[0]["expected_funding_hours"], 720)
        self.assertEqual(rows[0]["status"], "pass")
        self.assertTrue(rows[0]["acceptable"])

    def test_complete_window_below_99_percent_funding_fails(self):
        start = collector.OOS_START
        rates = {
            start + i * 3600: 0.0001
            for i in range(collector.OOS_WINDOW_SECONDS // 3600)
            if i >= 8
        }
        rows = collector.build_funding_coverage_rows(
            "BTC", rates, start + collector.OOS_WINDOW_SECONDS
        )
        self.assertEqual(rows[0]["funding_rows"], 712)
        self.assertEqual(rows[0]["status"], "fail")
        self.assertFalse(rows[0]["acceptable"])

    def test_funding_coverage_reports_only_elapsed_windows(self):
        end_ts = collector.OOS_START + collector.OOS_WINDOW_SECONDS + 2 * 3600
        rows = collector.build_funding_coverage_rows("BTC", {}, end_ts)
        self.assertEqual([row["window"] for row in rows], ["oos_1", "oos_2"])
        self.assertEqual(rows[0]["expected_funding_hours"], 24 * 30)
        self.assertEqual(rows[1]["expected_funding_hours"], 2)
        self.assertEqual(rows[1]["status"], "provisional")
        self.assertIsNone(rows[1]["acceptable"])

    def test_daily_trigger_path_does_not_loop_on_state_or_event_commits(self):
        workflow = Path(".github/workflows/paper-forward-collector.yml").read_text(encoding="utf-8")
        push_paths = workflow.split("  push:", 1)[1].split("  workflow_dispatch:", 1)[0]
        self.assertIn("paper-forward-data/collector-trigger.json", push_paths)
        self.assertNotIn("paper-forward-data/state.json", push_paths)
        self.assertNotIn("paper-forward-data/events.jsonl", push_paths)


if __name__ == "__main__":
    unittest.main()
