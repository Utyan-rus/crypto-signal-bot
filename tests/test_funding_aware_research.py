import unittest
from unittest.mock import patch

import funding_aware_research as research


class FundingAwareResearchTests(unittest.TestCase):
    def test_funding_fetch_chunks_and_normalizes_timestamps(self):
        payloads = [
            {"fundings": [
                {"timestamp": 3600, "rate": "0.001", "direction": "long"},
                {"timestamp": 7200, "rate": "0.002", "direction": "short"},
            ]}
        ]
        with patch.object(research, "get_json", side_effect=payloads) as get_json:
            rates = research.fetch_funding_rates(1, 0, 8000)
        self.assertAlmostEqual(rates[3600], 0.00001)
        self.assertAlmostEqual(rates[7200], -0.00002)
        self.assertEqual(get_json.call_count, 1)
        self.assertEqual(get_json.call_args.args[1]["start_timestamp"], 0)
        self.assertEqual(get_json.call_args.args[1]["end_timestamp"], 8000000)

    def test_funding_cost_is_applied_to_open_long(self):
        candles = [
            {"t": 0, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 3600, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 7200, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 10800, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
        ]
        signals = {3600: ("LONG", 0.1)}
        with patch.object(research, "build_model", return_value=(signals, {})):
            result = research.simulate_with_funding("TEST", candles, "stub", 0, 20000, {10800: 0.01})
        self.assertEqual(result["funding_events_charged"], 1)
        self.assertGreater(result["funding_net_cost_usd"], 0)


    def test_direction_mode_keeps_only_requested_side(self):
        candles = [
            {"t": i * 3600, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0, "v": 100.0}
            for i in range(7)
        ]
        signals = {
            0: ("LONG", 2.0), 3600: ("SHORT", 2.0), 7200: ("LONG", 2.0),
            10800: ("SHORT", 2.0), 14400: ("LONG", 2.0), 18000: ("SHORT", 2.0),
        }
        exit_levels = {i * 3600: (101.0, 99.0) for i in range(7)}
        with patch.object(research, "build_model", return_value=(signals, exit_levels)), \
             patch.object(research, "size_position", return_value=1.0):
            long_only = research.simulate_with_funding(
                "TEST", candles, "stub", 0, 25200, {}, start_balance=100.0, direction_mode="LONG")
            short_only = research.simulate_with_funding(
                "TEST", candles, "stub", 0, 25200, {}, start_balance=100.0, direction_mode="SHORT")
            both = research.simulate_with_funding(
                "TEST", candles, "stub", 0, 25200, {}, start_balance=100.0, direction_mode="BOTH")
            with self.assertRaisesRegex(ValueError, "direction_mode"):
                research.simulate_with_funding(
                    "TEST", candles, "stub", 0, 25200, {}, start_balance=100.0, direction_mode="FLAT")
        self.assertGreater(long_only["long_trades"], 0)
        self.assertEqual(long_only["short_trades"], 0)
        self.assertGreater(short_only["short_trades"], 0)
        self.assertEqual(short_only["long_trades"], 0)
        self.assertGreater(both["long_trades"], 0)
        self.assertGreater(both["short_trades"], 0)

    def test_make_oos_windows_are_non_overlapping_and_chronological(self):
        end = 2_000_000_000
        windows = research.make_oos_windows(end, count=8, window_days=30)
        self.assertEqual(len(windows), 8)
        self.assertEqual(windows[0][0], "oos_01")
        self.assertEqual(windows[-1][0], "oos_08")
        self.assertEqual(windows[0][1], end - 240 * 86400)
        self.assertEqual(windows[-1][2], end)
        for index, (_, lo, hi) in enumerate(windows):
            self.assertEqual(hi - lo, 30 * 86400)
            if index:
                self.assertEqual(windows[index - 1][2], lo)

    def test_make_oos_windows_rejects_zero_count(self):
        with self.assertRaises(ValueError):
            research.make_oos_windows(2_000_000_000, count=0)

    def test_funding_receipt_offsets_short_position_cost(self):
        candles = [
            {"t": 0, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 3600, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 7200, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
            {"t": 10800, "o": 100, "h": 100.01, "l": 99.99, "c": 100, "v": 100},
        ]
        signals = {3600: ("SHORT", 0.1)}
        with patch.object(research, "build_model", return_value=(signals, {})):
            result = research.simulate_with_funding("TEST", candles, "stub", 0, 20000, {10800: 0.01})
        self.assertEqual(result["funding_events_charged"], 1)
        self.assertLess(result["funding_net_cost_usd"], 0)


if __name__ == "__main__":
    unittest.main()
