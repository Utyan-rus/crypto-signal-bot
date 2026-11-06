import unittest
from unittest.mock import patch

from hourly_candidate_research import MIN_NOTIONAL, build_model, max_dd, simulate, size_position


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
        candle_times = {b["t"] for b in self.candles}
        for strategy in ("donchian_20_10", "mean_reversion_bollinger", "adaptive_regime_switch",
                         "donchian_long_regime", "donchian_long_regime_vol"):
            signals, _ = build_model(self.candles, strategy)
            self.assertTrue(set(signals).issubset(candle_times))
            self.assertTrue(all(t < self.candles[-1]["t"] for t in signals))

    def test_minimum_notional_never_forces_excess_risk(self):
        self.assertEqual(MIN_NOTIONAL, 10.0)
        # A wide stop sizes below the venue's $10 minimum and must be rejected.
        self.assertEqual(size_position(5.0, 100.0, 10.0), 0.0)
        # A feasible $15 notional order remains inside both leverage and risk caps.
        qty = size_position(5.0, 100.0, 0.1)
        self.assertGreater(qty, 0.0)
        self.assertGreaterEqual(qty * 100.0, MIN_NOTIONAL)
        self.assertLessEqual(qty * 100.0, 5.0 * 3.0 + 1e-12)
        self.assertLessEqual(qty * 0.1, 5.0 * 0.005 + 1e-12)

    def test_simulation_counts_minimum_notional_rejections(self):
        candles = [
            {"t":i*3600, "o":100.0, "h":100.2, "l":99.8, "c":100.0, "v":100.0}
            for i in range(4)
        ]
        with patch("hourly_candidate_research.build_model",
                   return_value=({0:("LONG", 10.0)}, {})):
            result = simulate("TEST", candles, "donchian_20_10", 3600, 4*3600)
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["min_notional_rejections"], 1)
        self.assertEqual(result["min_notional_assumption_usd"], 10.0)

    def test_drawdown_is_mark_to_market(self):
        self.assertAlmostEqual(max_dd([5.0, 4.0, 4.5, 5.5]), 20.0)

    def test_simulation_returns_required_audit_fields(self):
        result = simulate("TEST", self.candles, "ema_20_50_cross", 350*3600, 450*3600)
        for key in ("return_pct", "max_drawdown_mtm_pct", "profit_factor", "trades"):
            self.assertIn(key, result)

    def test_strategy_names_are_supported(self):
        for name in ("donchian_20_10", "ema_trend_pullback", "ema_20_50_cross",
                     "ema_20_50_cross_spread", "mean_reversion_bollinger",
                     "adaptive_regime_switch", "donchian_long_regime",
                     "donchian_long_regime_vol"):
            signals, exits = build_model(self.candles, name)
            self.assertIsInstance(signals, dict)
            self.assertIsInstance(exits, dict)

    def test_new_regime_strategies_are_long_only(self):
        for name in ("donchian_long_regime", "donchian_long_regime_vol"):
            signals, _ = build_model(self.candles, name)
            self.assertTrue(all(side == "LONG" for side, _ in signals.values()))


if __name__ == "__main__":
    unittest.main()
