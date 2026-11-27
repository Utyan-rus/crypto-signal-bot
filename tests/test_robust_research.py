import unittest

from robust_research import MIN_NOTIONAL, max_drawdown, simulate_oos


class RobustResearchTests(unittest.TestCase):
    def test_mark_to_market_drawdown_includes_intratrade_equity_dips(self):
        self.assertAlmostEqual(max_drawdown([5.0, 4.0, 4.5, 5.5, 5.0]), 20.0)

    def test_flat_equity_has_no_drawdown(self):
        self.assertEqual(max_drawdown([5.0, 5.0, 5.0]), 0.0)

    def test_recovery_does_not_erase_prior_drawdown(self):
        self.assertAlmostEqual(max_drawdown([10.0, 8.0, 12.0]), 20.0)

    def test_time_exit_closes_after_configured_holding_bars(self):
        candles = [
            {"t": i * 300, "o": 100.0, "h": 100.06, "l": 100.045, "c": 100.05, "v": 10.0}
            for i in range(5)
        ]
        signals = {0: ("LONG", 0.01, 1.5)}
        result = simulate_oos("TEST", candles, signals, 0, "test", 0, True,
                              0.0005, 0.0005, "time_exit", 1)
        self.assertEqual(result["trades"], 1)
        self.assertEqual(result["max_hold_bars"], 1)


    def test_five_dollar_risk_budget_cannot_meet_minimum_notional(self):
        candles = [
            {"t": i * 300, "o": 100.0, "h": 100.1, "l": 99.9, "c": 100.0, "v": 10.0}
            for i in range(5)
        ]
        signals = {0: ("LONG", 0.2, 1.5)}
        result = simulate_oos("TEST", candles, signals, 0, "test", 0, True,
                              0.0005, 0.0005, "fixed_rr", 0, 0.0, "LONG")
        self.assertEqual(MIN_NOTIONAL, 10.0)
        self.assertEqual(result["entry_attempts"], 1)
        self.assertEqual(result["min_notional_rejections"], 1)
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["status"], "not_executable")
        self.assertEqual(result["cash_baseline_return_pct"], 0.0)
        self.assertEqual(result["buy_hold_status"], "not_executable_at_1x_below_min_notional")
        self.assertEqual(result["historical_funding_coverage_pct"], 0.0)

    def test_direction_filter_excludes_opposite_signals(self):
        candles = [
            {"t": i * 300, "o": 100.0, "h": 100.2, "l": 99.8, "c": 100.0, "v": 10.0}
            for i in range(5)
        ]
        signals = {0: ("SHORT", 1.0, 1.5)}
        result = simulate_oos("TEST", candles, signals, 0, "test", 0, True,
                              0.0005, 0.0005, "fixed_rr", 0, 0.0, "LONG")
        self.assertEqual(result["trades"], 0)


if __name__ == "__main__":
    unittest.main()
