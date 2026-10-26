import unittest
from unittest.mock import patch

from donchian_benchmark_research import (
    buy_hold, paired_rows, summarize, COST_SCENARIOS,
)
from funding_aware_research import make_oos_windows, simulate_with_funding


class DonchianBenchmarkTests(unittest.TestCase):
    def test_pair_requires_trade_and_notional_eligibility(self):
        strategy = {"symbol": "BTC", "status": "tested", "trades": 4,
                    "start_balance": 5, "return_pct": 10, "max_drawdown_mtm_pct": 1}
        benchmark = {"symbol": "BTC", "status": "tested", "trades": 1,
                     "return_pct": -10, "start_balance": 5, "min_quote_check_pass": False, "min_quote_notional_usd": 5}
        row = paired_rows(strategy, benchmark, "oos_01", 0, 100, 100, "base_taker")
        self.assertFalse(row["paired_evaluable"])
        self.assertEqual(row["excess_return_pp"], "")
        self.assertFalse(row["strategy_beats_benchmark"])

    def test_valid_pair_excess_and_summary(self):
        strategy = {"symbol": "ETH", "status": "tested", "trades": 8, "start_balance": 50,
                    "return_pct": 3, "max_drawdown_mtm_pct": 4, "profit_factor": 1.2,
                    "fees_per_side_pct": 0.05, "slippage_per_side_pct": 0.05}
        benchmark = {"symbol": "ETH", "status": "tested", "trades": 1, "return_pct": 1, "start_balance": 50,
                     "min_quote_check_pass": True, "min_quote_notional_usd": 49,
                     "max_drawdown_mtm_pct": 6}
        row = paired_rows(strategy, benchmark, "oos_01", 100, 200, 200, "base_taker")
        self.assertTrue(row["paired_evaluable"])
        self.assertEqual(row["excess_return_pp"], 2)
        summary = summarize([row], ("start_balance", "cost_scenario"))[0]
        self.assertEqual(summary["paired_count"], 1)
        self.assertEqual(summary["mean_excess_return_pp"], 2)
        self.assertEqual(summary["strategy_wins_vs_benchmark"], 1)
        self.assertEqual(summary["mean_strategy_profit_factor"], 1.2)
        self.assertEqual(row["window_start_utc"], 100)
        self.assertEqual(row["window_end_utc"], 200)
        self.assertEqual(row["shared_end_timestamp"], 200)

    def test_small_balance_is_below_minimum_notional(self):
        candles = [
            {"t": 0, "o": 100, "h": 101, "l": 99, "c": 100},
            {"t": 3600, "o": 100, "h": 101, "l": 99, "c": 100},
        ]
        result = buy_hold("BTC", candles, 0, 7200, {}, 5)
        self.assertFalse(result["min_quote_check_pass"])
        self.assertEqual(result["execution_feasibility"], "below_min_quote")

    def test_buy_hold_drawdown_includes_entry_candle_dip(self):
        candles = [
            {"t": 0, "o": 100, "h": 101, "l": 79, "c": 80},
            {"t": 3600, "o": 80, "h": 111, "l": 79, "c": 110},
        ]
        result = buy_hold("BTC", candles, 0, 7200, {}, 100)
        self.assertGreater(result["max_drawdown_mtm_pct"], 15.0)

    def test_cost_scenarios_are_monotonically_more_conservative(self):
        self.assertEqual([x[0] for x in COST_SCENARIOS],
                         ["base_taker", "double_slippage", "double_fee_and_slippage"])
        self.assertLess(COST_SCENARIOS[0][1], COST_SCENARIOS[2][1])
        self.assertLess(COST_SCENARIOS[0][2], COST_SCENARIOS[1][2])

    def test_oos_windows_are_contiguous_and_share_exact_end_timestamp(self):
        end = 1_800_000_000
        windows = make_oos_windows(end, count=12, window_days=30)
        self.assertEqual(windows[-1][2], end)
        self.assertEqual(len(windows), 12)
        self.assertEqual(windows[0][1], end - 360 * 86400)
        for previous, current in zip(windows, windows[1:]):
            self.assertEqual(previous[2], current[1])
            self.assertEqual(previous[2] - previous[1], 30 * 86400)

    @patch("funding_aware_research.size_position", return_value=1.0)
    @patch("funding_aware_research.build_model")
    def test_higher_fee_and_slippage_lower_strategy_return(self, build_model, _size_position):
        candles = [
            {"t": 0, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "h": 111.0, "l": 99.0, "c": 110.0},
            {"t": 7200, "o": 110.0, "h": 111.0, "l": 109.0, "c": 110.0},
        ]
        build_model.return_value = ({0: ("LONG", 50.0)}, {})
        base = simulate_with_funding("TEST", candles, "donchian_20_10", 0, 7200, {},
                                     start_balance=100.0, fee_rate=0.0005, slippage_rate=0.0005)
        stress = simulate_with_funding("TEST", candles, "donchian_20_10", 0, 7200, {},
                                       start_balance=100.0, fee_rate=0.001, slippage_rate=0.001)
        self.assertLess(stress["return_pct"], base["return_pct"])
        self.assertEqual(stress["fees_per_side_pct"], 0.1)
        self.assertEqual(stress["slippage_per_side_pct"], 0.1)

    def test_buy_hold_cost_stress_reduces_return(self):
        candles = [
            {"t": 0, "o": 100, "h": 101, "l": 99, "c": 100},
            {"t": 3600, "o": 100, "h": 111, "l": 99, "c": 110},
        ]
        base = buy_hold("BTC", candles, 0, 7200, {}, 100, 0.0005, 0.0005)
        stress = buy_hold("BTC", candles, 0, 7200, {}, 100, 0.001, 0.001)
        self.assertLess(stress["return_pct"], base["return_pct"])


if __name__ == "__main__":
    unittest.main()
