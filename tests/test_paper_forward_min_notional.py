import json
import tempfile
import unittest
from pathlib import Path

import paper_forward_collector as collector


class PaperForwardMinimumNotionalTests(unittest.TestCase):
    def run_signal(self, balance, stop_distance):
        with tempfile.TemporaryDirectory() as tmp:
            events_path = Path(tmp) / "events.jsonl"
            collector.load_event_ids(events_path)

            track = collector.fresh_track()
            track["balance"] = float(balance)
            track["equity_peak"] = float(balance)

            candles = [
                {"t": collector.OOS_START, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
                {"t": collector.OOS_START + 3600, "o": 100.0, "h": 101.0, "l": 99.5, "c": 100.0},
            ]
            signals = {collector.OOS_START: ("LONG", float(stop_distance))}

            collector.process_bar(
                track,
                "BTC",
                "minimum_notional_regression",
                candles,
                1,
                signals,
                {},
                {},
                events_path,
            )
            events = [
                json.loads(line)
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            return track, events

    def test_five_usd_balance_cannot_open_below_minimum_notional(self):
        # With 0.5% risk and a 3x leverage cap, the risk-sized notional is
        # far below the exchange's $10 minimum for this stop distance.
        track, events = self.run_signal(balance=5.0, stop_distance=2.5)

        self.assertIsNone(track["position"])
        self.assertEqual(track["trades"], 0)
        self.assertEqual(track["min_notional_rejections"], 1)
        self.assertEqual([event["event"] for event in events], ["entry_rejected"])
        self.assertEqual(events[0]["reason"], "risk_budget_or_minimum_notional")
        self.assertEqual(events[0]["min_notional_usd"], collector.MIN_NOTIONAL)

    def test_risk_sized_order_above_minimum_notional_is_accepted(self):
        track, events = self.run_signal(balance=50.0, stop_distance=2.5)

        self.assertIsNotNone(track["position"])
        self.assertEqual(track["min_notional_rejections"], 0)
        self.assertEqual([event["event"] for event in events], ["paper_entry"])
        self.assertGreaterEqual(events[0]["notional_usd"], collector.MIN_NOTIONAL)


if __name__ == "__main__":
    unittest.main()
