import unittest

from hourly_candidate_research import build_model, max_dd, simulate, size_position


class HourlyCandidateResearchTests(unittest.TestCase):
    def setUp(self):
        self.candles = []
        price = 100.0
        for i in range(500):
            # Deterministic rising market with small oscillations.
            price *= 1.001
            self.candles.append({
                "t": i*3600, "o": price*0.999, "h": price*1.002,
                "l": price*0.998, "c": price, "v": 100.0
            })

    def test_indicator_context_produces_no_future_candle_signal(self):
        signals, _ = build_model(self.candles, "donchian_20_10")
        candle_times = {b["t"] for b in self.candles}
        self.assertTrue(set(signals).issubset(candle_times))
        self.assertTrue(all(t < self.candles[-1]["t"] for t in signals))

    def test_minimum_notional_never_forces_excess_risk(self):
        # With $5 capital and a wide stop, the risk-sized order is below $1.
        self.assertEqual(size_position(5.0, 100.0, 10.0), 0.0)
        # A feasible order stays within the defined risk budget.
        qty = size_position(5.0, 100.0, 0.1)
        self.assertGreater(qty, 0.0)
        self.assertLessEqual(qty * 0.1, 5.0 * 0.005 + 1e-12)

    def test_drawdown_is_mark_to_market(self):
        self.assertAlmostEqual(max_dd([5.0, 4.0, 4.5, 5.5]), 20.0)

    def test_simulation_returns_required_audit_fields(self):
        result = simulate("TEST", self.candles, "ema_20_50_cross", 350*3600, 450*3600)
        for key in ("return_pct", "max_drawdown_mtm_pct", "profit_factor", "trades"):
            self.assertIn(key, result)

    def test_strategy_names_are_supported(self):
        for name in ("donchian_20_10", "ema_trend_pullback", "ema_20_50_cross", "ema_20_50_cross_spread"):
            signals, exits = build_model(self.candles, name)
            self.assertIsInstance(signals, dict)
            self.assertIsInstance(exits, dict)


if __name__ == "__main__":
    unittest.main()
