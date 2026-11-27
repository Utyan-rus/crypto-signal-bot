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


    def test_prospective_benchmarks_include_cash_and_both_1x_directions(self):
        start = collector.OOS_START
        candles = [
            {"t": start + i * 3600, "o": 100.0 + i, "h": 102.0 + i,
             "l": 99.0 + i, "c": 101.0 + i}
            for i in range(4)
        ]
        rates = {start + i * 3600: 0.0001 for i in range(4)}
        rows = collector.build_prospective_benchmark_rows(
            "BTC", candles, rates, start + 4 * 3600
        )
        by_key = {(row["window"], row["benchmark"]): row for row in rows}
        self.assertEqual(len(rows), 6)
        self.assertEqual(by_key[("oos_to_date", "cash_no_trade")]["return_pct"], 0.0)
        long_row = by_key[("oos_to_date", "1x_long_hold")]
        short_row = by_key[("oos_to_date", "1x_short_hold")]
        self.assertTrue(long_row["funding_complete"])
        self.assertTrue(short_row["funding_complete"])
        self.assertGreater(long_row["funding_net_cost_usd"], 0.0)
        self.assertLess(short_row["funding_net_cost_usd"], 0.0)
        self.assertIsNotNone(long_row["return_pct"])
        self.assertFalse(long_row["decision_eligible"])
        self.assertEqual(long_row["status"], "provisional")

    def test_missing_charged_funding_hides_net_buy_hold_return(self):
        start = collector.OOS_START
        candles = [
            {"t": start + i * 3600, "o": 100.0 + i, "h": 102.0 + i,
             "l": 99.0 + i, "c": 101.0 + i}
            for i in range(4)
        ]
        rates = {
            start: 0.0001,
            start + 3600: 0.0001,
            start + 3 * 3600: 0.0001,
        }
        rows = collector.build_prospective_benchmark_rows(
            "BTC", candles, rates, start + 4 * 3600
        )
        by_key = {(row["window"], row["benchmark"]): row for row in rows}
        long_row = by_key[("oos_to_date", "1x_long_hold")]
        self.assertFalse(long_row["funding_complete"])
        self.assertEqual(long_row["funding_missing_hours"], 1)
        self.assertIsNone(long_row["return_pct"])
        self.assertIsNotNone(long_row["observed_return_excluding_missing_funding_pct"])
        self.assertEqual(long_row["status"], "provisional_funding_pnl_incomplete")
        self.assertEqual(by_key[("oos_to_date", "cash_no_trade")]["return_pct"], 0.0)

    def test_prospective_buy_hold_respects_minimum_notional(self):
        start = collector.OOS_START
        candles = [
            {"t": start + i * 3600, "o": 100.0 + i, "h": 102.0 + i,
             "l": 99.0 + i, "c": 101.0 + i}
            for i in range(4)
        ]
        rates = {start + i * 3600: 0.0001 for i in range(4)}
        rows = collector.build_prospective_benchmark_rows(
            "BTC", candles, rates, start + 4 * 3600, start_balance=5.0
        )
        long_row = next(
            row for row in rows
            if row["window"] == "oos_to_date" and row["benchmark"] == "1x_long_hold"
        )
        self.assertFalse(long_row["executable"])
        self.assertEqual(long_row["status"], "not_executable_below_min_notional")
        self.assertIsNone(long_row["return_pct"])

    def test_paired_comparison_uses_same_oos_to_date_balance_and_boundary(self):
        start = collector.OOS_START
        candles = [
            {"t": start + i * 3600, "o": 100.0 + i, "h": 102.0 + i,
             "l": 99.0 + i, "c": 101.0 + i}
            for i in range(4)
        ]
        rates = {start + i * 3600: 0.0001 for i in range(4)}
        benchmarks = collector.build_prospective_benchmark_rows(
            "BTC", candles, rates, start + 4 * 3600
        )
        tracks = [{
            "market_strategy": "BTC|donchian_long_regime",
            "mtm_equity_usd": 50.0,
            "funding_missing_hours": 0,
        }]
        comparisons = collector.build_paired_benchmark_comparisons(tracks, benchmarks)
        cash = next(row for row in comparisons if row["benchmark"] == "cash_no_trade")
        self.assertEqual(cash["strategy_return_pct"], 0.0)
        self.assertEqual(cash["benchmark_return_pct"], 0.0)
        self.assertEqual(cash["excess_return_pp"], 0.0)
        self.assertEqual(cash["start_balance_usd"], 50.0)
        self.assertFalse(cash["decision_eligible"])
        self.assertEqual(cash["window"], "oos_to_date")


if __name__ == "__main__":
    unittest.main()
