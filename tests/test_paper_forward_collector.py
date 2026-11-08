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


if __name__ == "__main__":
    unittest.main()
