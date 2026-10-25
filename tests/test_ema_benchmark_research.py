import unittest

from ema_benchmark_research import buy_hold, build_comparison_rows, write_comparison


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
        self.assertEqual(result["fees_per_side_pct"], 0.05)
        self.assertEqual(result["slippage_per_side_pct"], 0.05)
        self.assertTrue(result["min_quote_check_pass"])
        self.assertGreaterEqual(result["max_drawdown_mtm_pct"], 0.0)

    def test_buy_hold_rejects_too_short_window(self):
        result = buy_hold("TEST", [{"t": 0, "o": 100.0, "c": 100.0}], 0, 3600, {}, 100.0)
        self.assertEqual(result["status"], "low_sample")
        self.assertEqual(result["trades"], 0)

    def test_buy_hold_marks_small_balance_below_quote_minimum(self):
        candles = [
            {"t": 0, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
        ]
        result = buy_hold("TEST", candles, 0, 7200, {}, 5.0)
        self.assertFalse(result["min_quote_check_pass"])
        self.assertEqual(result["execution_feasibility"], "below_min_quote")

    def test_buy_hold_marks_to_market_drawdown(self):
        candles = [
            {"t": 0, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "h": 101.0, "l": 79.0, "c": 80.0},
            {"t": 7200, "o": 80.0, "h": 111.0, "l": 79.0, "c": 110.0},
            {"t": 10800, "o": 110.0, "h": 111.0, "l": 109.0, "c": 110.0},
        ]
        result = buy_hold("TEST", candles, 0, 14400, {}, 100.0)
        self.assertGreater(result["max_drawdown_mtm_pct"], 10.0)

    def test_paired_comparison_uses_same_market_window_and_balance(self):
        rows = [
            {"symbol": "BTC", "strategy": "ema_20_50_cross", "benchmark": False,
             "window": "oos_01", "start_balance": 50.0, "status": "tested",
             "trades": 6, "return_pct": -1.0, "max_drawdown_mtm_pct": 2.0, "profit_factor": 0.8},
            {"symbol": "BTC", "strategy": "buy_hold_long_1x", "benchmark": True,
             "window": "oos_01", "start_balance": 50.0, "status": "tested",
             "trades": 1, "return_pct": 2.0, "max_drawdown_mtm_pct": 3.0,
             "funding_net_cost_usd": 0.2, "min_quote_notional_usd": 49.0,
             "min_quote_check_pass": True, "execution_feasibility": "quote_minimum_pass_other_constraints_unverified"},
        ]
        pairs = build_comparison_rows(rows)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["ema_minus_benchmark_pct_points"], -3.0)
        self.assertTrue(pairs[0]["ema_eligible_5plus_trades"])
        self.assertTrue(pairs[0]["paired_evaluable"])
        self.assertTrue(pairs[0]["benchmark_quote_eligible"])

    def test_comparison_files_include_window_and_balance_splits(self):
        import os
        import tempfile
        rows = [
            {"symbol": "BTC", "strategy": "ema_20_50_cross", "benchmark": False,
             "window": "oos_01", "start_balance": 50.0, "status": "tested",
             "trades": 6, "return_pct": -1.0, "max_drawdown_mtm_pct": 2.0, "profit_factor": 0.8},
            {"symbol": "BTC", "strategy": "buy_hold_long_1x", "benchmark": True,
             "window": "oos_01", "start_balance": 50.0, "status": "tested",
             "trades": 1, "return_pct": 2.0, "max_drawdown_mtm_pct": 3.0,
             "funding_net_cost_usd": 0.2, "min_quote_notional_usd": 49.0,
             "min_quote_check_pass": True, "execution_feasibility": "quote_minimum_pass_other_constraints_unverified"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pairs, summary = write_comparison(rows, tmp)
            self.assertEqual(len(pairs), 1)
            self.assertTrue(any(x["scope"] == "window:oos_01:eligible" for x in summary))
            self.assertTrue(os.path.exists(os.path.join(tmp, "ema_benchmark_comparison.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "ema_benchmark_comparison_summary.csv")))


    def test_ineligible_pairs_do_not_count_as_benchmark_outperformance(self):
        import os
        import tempfile
        rows = [
            {"symbol": "BTC", "strategy": "ema_20_50_cross", "benchmark": False,
             "window": "oos_01", "start_balance": 5.0, "status": "low_sample",
             "trades": 0, "return_pct": 0.0, "max_drawdown_mtm_pct": 0.0, "profit_factor": 0.0},
            {"symbol": "BTC", "strategy": "buy_hold_long_1x", "benchmark": True,
             "window": "oos_01", "start_balance": 5.0, "status": "tested",
             "trades": 1, "return_pct": -3.0, "max_drawdown_mtm_pct": 5.0,
             "funding_net_cost_usd": 0.1, "min_quote_notional_usd": 4.9,
             "min_quote_check_pass": False, "execution_feasibility": "below_min_quote"},
        ]
        pairs = build_comparison_rows(rows)
        self.assertEqual(len(pairs), 1)
        self.assertFalse(pairs[0]["ema_eligible_5plus_trades"])
        self.assertFalse(pairs[0]["benchmark_quote_eligible"])
        self.assertFalse(pairs[0]["paired_evaluable"])
        self.assertFalse(pairs[0]["ema_outperforms_benchmark"])
        self.assertEqual(pairs[0]["ema_minus_benchmark_pct_points"], "")
        with tempfile.TemporaryDirectory() as tmp:
            _, summary = write_comparison(rows, tmp)
            overall = next(x for x in summary if x["scope"] == "all_pairs_including_low_sample")
            self.assertEqual(overall["ema_outperforms_benchmark_pairs"], 0)
            self.assertEqual(overall["paired_evaluable_pairs"], 0)
            self.assertEqual(overall["mean_excess_return_pct_points"], "")
            with open(os.path.join(tmp, "ema_benchmark_comparison.csv"), encoding="utf-8") as detail_file:
                detail = detail_file.read()
            self.assertIn("paired_evaluable", detail.splitlines()[0])


if __name__ == "__main__":
    unittest.main()
