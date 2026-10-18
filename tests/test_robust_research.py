import unittest

from robust_research import max_drawdown, simulate_oos


class RobustResearchTests(unittest.TestCase):
    def test_mark_to_market_drawdown_includes_intratrade_equity_dips(self):
        self.assertAlmostEqual(max_drawdown([5.0, 4.0, 4.5, 5.5, 5.0]), 20.0)

    def test_flat_equity_has_no_drawdown(self):
        self.assertEqual(max_drawdown([5.0, 5.0, 5.0]), 0.0)

    def test_recovery_does_not_erase_prior_drawdown(self):
        self.assertAlmostEqual(max_drawdown([10.0, 8.0, 12.0]), 20.0)

    def test_time_exit_closes_after_configured_holding_bars(self):
        candles = [
            {"t": i * 300, "o": 100.0, "h": 100.2, "l": 99.8, "c": 100.0, "v": 10.0}
            for i in range(5)
        ]
        signals = {0: ("LONG", 1.0, 1.5)}
        result = simulate_oos("TEST", candles, signals, 0, "test", 0, True,
                              0.0005, 0.0005, "time_exit", 1)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["max_hold_bars"], 1)


if __name__ == "__main__":
    unittest.main()
