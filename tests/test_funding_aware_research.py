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
