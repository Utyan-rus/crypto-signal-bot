import unittest
from unittest.mock import patch

import lighter_fee_tier_research as research


class LighterFeeTierResearchTests(unittest.TestCase):
    def test_fee_and_slippage_scenarios_are_predeclared_and_unique(self):
        actual = {
            name: (fee, slippage, tier)
            for name, fee, slippage, tier in research.COST_SCENARIOS
        }
        expected = {
            "standard_0bps_fee_5bps_slippage": (0.0, 0.0005, "Standard"),
            "standard_0bps_fee_10bps_slippage": (0.0, 0.0010, "Standard"),
            "plus_0_5bps_fee_5bps_slippage": (0.00005, 0.0005, "Plus"),
            "premium_2_8bps_fee_5bps_slippage": (0.00028, 0.0005, "Premium"),
            "premium_2_8bps_fee_10bps_slippage": (0.00028, 0.0010, "Premium"),
            "legacy_5bps_fee_5bps_slippage": (0.0005, 0.0005, "Conservative generic"),
            "legacy_5bps_fee_10bps_slippage": (0.0005, 0.0010, "Conservative generic"),
        }
        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), len(research.COST_SCENARIOS))
        self.assertTrue(all(fee >= 0 and slippage >= 0 for fee, slippage, _ in actual.values()))

    def test_same_fee_and_slippage_are_applied_to_both_variants_and_buy_hold(self):
        benchmark = {
            "status": "tested",
            "return_pct": 0.50,
            "max_drawdown_mtm_pct": 2.0,
            "funding_net_cost_usd": 0.01,
            "min_quote_notional_usd": 50.0,
            "min_quote_check_pass": True,
        }
        simulated = {
            "status": "tested",
            "trades": 6,
            "long_trades": 3,
            "short_trades": 3,
            "return_pct": 1.25,
            "end_balance": 50.625,
            "max_drawdown_mtm_pct": 1.5,
            "profit_factor": 1.10,
            "funding_events_charged": 12,
            "funding_net_cost_usd": 0.02,
            "funding_history_rows": 100,
            "min_notional_rejections": 1,
            "min_notional_assumption_usd": 10.0,
            "fees_per_side_pct": 0.028,
            "slippage_per_side_pct": 0.10,
        }
        with patch.object(research, "buy_hold", return_value=benchmark) as buy_hold_mock:
            with patch.object(research.funding, "simulate_with_funding", return_value=simulated) as sim_mock:
                rows = research.simulate_market_window(
                    symbol="BTC",
                    candles=[],
                    funding_rates={},
                    window_name="oos_01",
                    window_start=1000,
                    window_end=2000,
                    shared_end_timestamp=2000,
                    balance=50.0,
                    cost_name="premium_2_8bps_fee_10bps_slippage",
                    fee_rate=0.00028,
                    slippage_rate=0.0010,
                    funding_coverage_pct=99.9,
                )

        buy_hold_mock.assert_called_once()
        self.assertEqual(buy_hold_mock.call_args.kwargs["fee_rate"], 0.00028)
        self.assertEqual(buy_hold_mock.call_args.kwargs["slippage_rate"], 0.0010)
        self.assertEqual(sim_mock.call_count, 2)
        for call in sim_mock.call_args_list:
            self.assertEqual(call.kwargs["fee_rate"], 0.00028)
            self.assertEqual(call.kwargs["slippage_rate"], 0.0010)
            self.assertEqual(call.kwargs["direction_mode"], "BOTH")
        self.assertEqual({row["strategy"] for row in rows}, set(research.adx.STRATEGIES))
        self.assertTrue(all(row["buy_hold_executable"] for row in rows))
        self.assertEqual([row["fees_per_side_pct"] for row in rows], [0.028, 0.028])
        self.assertEqual([row["slippage_per_side_pct"] for row in rows], [0.10, 0.10])
        self.assertEqual([row["excess_vs_buy_hold_pp"] for row in rows], [0.75, 0.75])

    def test_below_minimum_buy_hold_is_not_counted_as_executable(self):
        benchmark = {
            "status": "tested",
            "return_pct": 2.0,
            "max_drawdown_mtm_pct": 3.0,
            "funding_net_cost_usd": 0.0,
            "min_quote_notional_usd": 5.0,
            "min_quote_check_pass": False,
        }
        simulated = {
            "status": "low_sample",
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "return_pct": 0.0,
            "end_balance": 5.0,
            "max_drawdown_mtm_pct": 0.0,
            "profit_factor": 0.0,
            "funding_events_charged": 0,
            "funding_net_cost_usd": 0.0,
            "funding_history_rows": 100,
            "min_notional_rejections": 4,
            "min_notional_assumption_usd": 10.0,
            "fees_per_side_pct": 0.0,
            "slippage_per_side_pct": 0.05,
        }
        with patch.object(research, "buy_hold", return_value=benchmark):
            with patch.object(research.funding, "simulate_with_funding", return_value=simulated):
                rows = research.simulate_market_window(
                    symbol="BTC",
                    candles=[],
                    funding_rates={},
                    window_name="oos_01",
                    window_start=1000,
                    window_end=2000,
                    shared_end_timestamp=2000,
                    balance=5.0,
                    cost_name="standard_0bps_fee_5bps_slippage",
                    fee_rate=0.0,
                    slippage_rate=0.0005,
                    funding_coverage_pct=99.9,
                )
        self.assertTrue(all(not row["buy_hold_executable"] for row in rows))
        self.assertTrue(all(row["excess_vs_buy_hold_pp"] == "" for row in rows))
        self.assertTrue(all(row["buy_hold_min_quote_check_pass"] is False for row in rows))


if __name__ == "__main__":
    unittest.main()
