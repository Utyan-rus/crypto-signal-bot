import unittest

from walk_forward_research import (COSTS, RESEARCH_SYMBOLS, bootstrap_summaries, buy_hold_baseline, make_windows, select_research_markets, summarize)


class WalkForwardResearchTests(unittest.TestCase):
    def test_windows_are_chronological_non_overlapping_and_cover_240_days(self):
        end = 2_000_000_000
        windows = make_windows(end)
        self.assertEqual(len(windows), 8)
        self.assertTrue(all(windows[i][2] == windows[i + 1][1] for i in range(3)))
        self.assertEqual(windows[-1][2], end)
        self.assertEqual(windows[-1][2] - windows[0][1], 240 * 86400)

    def test_cost_scenarios_increase_costs(self):
        self.assertEqual(len(COSTS), 4)
        self.assertLess(COSTS[0][1], COSTS[1][1])
        self.assertLess(COSTS[1][2], COSTS[2][2])
        self.assertLess(COSTS[2][1] + COSTS[2][2], COSTS[3][1] + COSTS[3][2])

    def test_summary_includes_low_sample_and_no_trade_rows(self):
        rows = [
            {"strategy":"s", "cost_scenario":"base", "status":"tested", "symbol":"BTC",
             "window":"oos_1", "return_pct":2, "max_drawdown_mtm_pct":3,
             "trades":10, "profit_factor":1.2, "min_notional_rejections":0,
             "buy_hold_executable":True, "buy_hold_return_pct":1.0,
             "paired_evaluable":True, "excess_vs_buy_hold_pp":1.0,
             "strategy_beats_benchmark_observed":True},
            {"strategy":"s", "cost_scenario":"base", "status":"tested", "symbol":"ETH",
             "window":"oos_1", "return_pct":-1, "max_drawdown_mtm_pct":4,
             "trades":8, "profit_factor":0.8, "min_notional_rejections":0,
             "buy_hold_executable":True, "buy_hold_return_pct":1.0,
             "paired_evaluable":True, "excess_vs_buy_hold_pp":-2.0,
             "strategy_beats_benchmark_observed":False},
            {"strategy":"s", "cost_scenario":"base", "status":"low_sample", "symbol":"SOL",
             "window":"oos_1", "return_pct":99, "max_drawdown_mtm_pct":1,
             "trades":1, "profit_factor":9, "min_notional_rejections":2,
             "buy_hold_executable":True, "buy_hold_return_pct":1.0,
             "paired_evaluable":True, "excess_vs_buy_hold_pp":98.0,
             "strategy_beats_benchmark_observed":True},
            {"strategy":"s", "cost_scenario":"base", "status":"low_sample", "symbol":"ADA",
             "window":"oos_1", "return_pct":0, "max_drawdown_mtm_pct":0,
             "trades":0, "profit_factor":0, "min_notional_rejections":3,
             "buy_hold_executable":True, "buy_hold_return_pct":1.0,
             "paired_evaluable":True, "excess_vs_buy_hold_pp":-1.0,
             "strategy_beats_benchmark_observed":False},
        ]
        result = summarize(rows)[0]
        self.assertEqual(result["observations"], 4)
        self.assertEqual(result["tested_market_windows"], 2)
        self.assertEqual(result["low_sample_market_windows"], 2)
        self.assertEqual(result["active_market_windows"], 3)
        self.assertEqual(result["no_trade_market_windows"], 1)
        self.assertEqual(result["positive_market_windows"], 2)
        self.assertEqual(result["total_trades"], 19)
        self.assertEqual(result["mean_return_pct"], 25.0)
        self.assertEqual(result["tested_only_mean_return_pct"], 0.5)
        self.assertEqual(result["min_notional_rejections"], 5)
        self.assertEqual(result["mean_profit_factor"], 1.0)
        self.assertEqual(result["buy_hold_executable_market_windows"], 4)
        self.assertEqual(result["buy_hold_mean_return_pct"], 1.0)
        self.assertEqual(result["mean_excess_vs_buy_hold_pp"], 24.0)
        self.assertEqual(result["strategy_win_fraction_vs_buy_hold_pct"], 50.0)


    def test_buy_hold_baseline_respects_minimum_notional_and_costs(self):
        candles = [
            {"t": 0, "o": 100.0, "c": 100.0},
            {"t": 1, "o": 110.0, "c": 110.0},
            {"t": 2, "o": 120.0, "c": 120.0},
        ]
        small = buy_hold_baseline(candles, 0, 3, 5.0, 0.0, 0.0, 10.0)
        self.assertFalse(small["executable"])
        self.assertEqual(small["status"], "not_executable_below_min_notional")
        large = buy_hold_baseline(candles, 0, 3, 50.0, 0.0, 0.0, 10.0)
        self.assertTrue(large["executable"])
        self.assertAlmostEqual(large["return_pct"], 20.0)
        costed = buy_hold_baseline(candles, 0, 3, 50.0, 0.0005, 0.0005, 10.0)
        self.assertLess(costed["return_pct"], large["return_pct"])

    def test_two_way_bootstrap_is_reproducible_and_rejects_incomplete_grid(self):
        symbols = ["BTC", "ETH"]
        windows = ["w1", "w2", "w3", "w4"]
        rows = []
        for symbol_index, symbol in enumerate(symbols):
            for window_index, window in enumerate(windows):
                strategy_return = 1.0 + symbol_index + 0.2 * (window_index % 2)
                benchmark_return = 0.5
                excess = strategy_return - benchmark_return
                rows.append({
                    "strategy": "donchian_20_10", "cost_scenario": "base",
                    "symbol": symbol, "window": window, "start_balance": 50.0,
                    "direction_mode": "BOTH", "strategy_return_pct": strategy_return,
                    "strategy_mtm_drawdown_pct": 3.0, "strategy_status": "tested",
                    "strategy_trades": 10, "strategy_profit_factor": 1.2,
                    "benchmark_executable": True, "benchmark_return_pct": benchmark_return,
                    "paired_evaluable": True, "observed_excess_return_vs_benchmark_pp": excess,
                    "strategy_beats_benchmark_observed": excess > 0,
                })
        first = bootstrap_summaries(rows, windows, symbols, replicates=100,
                                    block_length=2, seed=123)
        second = bootstrap_summaries(rows, windows, symbols, replicates=100,
                                     block_length=2, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first[0]["status"], "ok")
        self.assertEqual(first[0]["market_window_rows"], 8)
        incomplete = bootstrap_summaries(rows[:-1], windows, symbols, replicates=100,
                                         block_length=2, seed=123)
        self.assertEqual(incomplete[0]["status"], "incomplete_grid")

    def test_summary_ignores_incomplete_market_rows_without_cost_scenario(self):
        self.assertEqual(summarize([{
            "symbol": "BTC", "strategy": "ALL", "status": "incomplete_history"
        }]), [])

    def test_research_cohort_is_predeclared_and_ordered(self):
        markets = [{"symbol": symbol, "id": index}
                   for index, symbol in enumerate(reversed(RESEARCH_SYMBOLS), 1)]
        selected = select_research_markets(markets, 5)
        self.assertEqual([market["symbol"] for market in selected],
                         ["BTC", "ETH", "SOL", "BNB", "XRP"])
        with self.assertRaises(RuntimeError):
            select_research_markets(markets[:-1], 5)


if __name__ == "__main__":
    unittest.main()
