import unittest

from ema_benchmark_research import buy_hold


class EmaBenchmarkResearchTests(unittest.TestCase):
    def test_buy_hold_applies_two_sided_costs_and_funding(self):
        candles = [
            {"t": 0, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "h": 111.0, "l": 99.0, "c": 110.0},
        ]
        result = buy_hold("TEST", candles, 0, 7200, {3600: 0.01}, 100.0)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["funding_events_charged"], 1)
        self.assertGreater(result["funding_net_cost_usd"], 0)
        self.assertLess(result["return_pct"], 10.0)

    def test_buy_hold_rejects_too_short_window(self):
        result = buy_hold("TEST", [{"t": 0, "o": 100.0, "c": 100.0}], 0, 3600, {}, 100.0)
        self.assertEqual(result["status"], "low_sample")
        self.assertEqual(result["trades"], 0)


if __name__ == "__main__":
    unittest.main()
