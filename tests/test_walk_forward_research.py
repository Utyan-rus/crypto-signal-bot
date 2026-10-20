import unittest

from walk_forward_research import COSTS, make_windows, summarize


class WalkForwardResearchTests(unittest.TestCase):
    def test_windows_are_chronological_non_overlapping_and_cover_120_days(self):
        end = 2_000_000_000
        windows = make_windows(end)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(windows[i][2] == windows[i + 1][1] for i in range(3)))
        self.assertEqual(windows[-1][2], end)
        self.assertEqual(windows[-1][2] - windows[0][1], 120 * 86400)

    def test_cost_scenarios_increase_costs(self):
        self.assertEqual(len(COSTS), 4)
        self.assertLess(COSTS[0][1], COSTS[1][1])
        self.assertLess(COSTS[1][2], COSTS[2][2])
        self.assertLess(COSTS[2][1] + COSTS[2][2], COSTS[3][1] + COSTS[3][2])

    def test_summary_aggregates_only_tested_rows(self):
        rows = [
            {"strategy":"s", "cost_scenario":"base", "status":"tested", "symbol":"BTC",
             "window":"oos_1", "return_pct":2, "max_drawdown_mtm_pct":3,
             "trades":10, "profit_factor":1.2},
            {"strategy":"s", "cost_scenario":"base", "status":"tested", "symbol":"ETH",
             "window":"oos_1", "return_pct":-1, "max_drawdown_mtm_pct":4,
             "trades":8, "profit_factor":0.8},
            {"strategy":"s", "cost_scenario":"base", "status":"low_sample", "symbol":"SOL",
             "window":"oos_1", "return_pct":99, "max_drawdown_mtm_pct":1,
             "trades":1, "profit_factor":9},
        ]
        result = summarize(rows)[0]
        self.assertEqual(result["observations"], 2)
        self.assertEqual(result["positive_market_windows"], 1)
        self.assertEqual(result["total_trades"], 18)
        self.assertEqual(result["mean_return_pct"], 0.5)


if __name__ == "__main__":
    unittest.main()
