import unittest

from slope_filter_bootstrap_audit import audit


def fixture():
    end = 1_900_000_000
    symbols = ["BTC", "ETH", "SOL"]
    window_days = 30
    first = end - 4 * window_days * 86400
    windows = [
        (f"oos_{index + 1:02d}", first + index * window_days * 86400,
         first + (index + 1) * window_days * 86400)
        for index in range(4)
    ]
    metadata = {
        "markets_tested": symbols,
        "strategy_names": ["donchian_20_10", "donchian_20_10_ema200_slope12"],
        "balances": [50.0],
        "cost_scenarios": [{"name": "base_taker"}],
        "oos_window_count": 4,
        "oos_window_days": window_days,
        "shared_end_timestamp": end,
    }
    rows = []
    for window_index, (window, start, stop) in enumerate(windows):
        for symbol_index, symbol in enumerate(symbols):
            baseline_return = 0.5 + 0.1 * window_index + 0.05 * symbol_index
            for strategy, result_return in (
                ("donchian_20_10", baseline_return),
                ("donchian_20_10_ema200_slope12", baseline_return + 1.0),
            ):
                rows.append({
                    "symbol": symbol,
                    "strategy": strategy,
                    "window": window,
                    "window_start_utc": start,
                    "window_end_utc": stop,
                    "shared_end_timestamp": end,
                    "start_balance": 50.0,
                    "cost_scenario": "base_taker",
                    "strategy_status": "tested",
                    "trades": 6,
                    "return_pct": result_return,
                    "max_drawdown_mtm_pct": 2.0,
                    "profit_factor": 1.3,
                    "funding_coverage_pct": 99.8,
                    "funding_events_charged": 12,
                    "funding_net_cost_usd": 0.01,
                    "min_notional_rejections": 0,
                    "buy_hold_status": "tested",
                    "buy_hold_return_pct": -2.0,
                    "buy_hold_min_quote_check_pass": "true",
                    "buy_hold_executable": "true",
                    "excess_vs_buy_hold_pp": result_return + 2.0,
                })
    return rows, metadata


class SlopeFilterBootstrapTests(unittest.TestCase):
    def test_two_way_bootstrap_and_leave_one_market_out(self):
        rows, metadata = fixture()
        summary, lomo, window_detail = audit(
            rows, metadata, replicates=500, block_length=2, seed=31
        )
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(lomo), 4)
        self.assertEqual(len(window_detail), 4)
        self.assertAlmostEqual(summary[0]["candidate_minus_baseline_mean_pp"], 1.0)
        self.assertAlmostEqual(summary[0]["candidate_minus_baseline_mean_ci95_low"], 1.0)
        self.assertAlmostEqual(summary[0]["candidate_minus_baseline_mean_ci95_high"], 1.0)
        self.assertGreater(summary[0]["candidate_mean_return_pct"], 0.0)
        self.assertEqual(summary[0]["buy_hold_executable_market_windows"], 12)

    def test_bootstrap_is_reproducible(self):
        rows, metadata = fixture()
        first = audit(rows, metadata, replicates=500, block_length=2, seed=37)
        second = audit(rows, metadata, replicates=500, block_length=2, seed=37)
        self.assertEqual(first, second)

    def test_incomplete_market_window_strategy_grid_is_rejected(self):
        rows, metadata = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete result grid"):
            audit(rows[:-1], metadata, replicates=100, block_length=2, seed=41)

    def test_buy_hold_ineligible_balance_has_no_buy_hold_excess(self):
        rows, metadata = fixture()
        for row in rows:
            row["buy_hold_min_quote_check_pass"] = "false"
            row["buy_hold_executable"] = "false"
            row["excess_vs_buy_hold_pp"] = ""
        summary, _, _ = audit(rows, metadata, replicates=100, block_length=2, seed=43)
        self.assertEqual(summary[0]["buy_hold_executable_market_windows"], 0)
        self.assertEqual(summary[0]["candidate_mean_excess_vs_buy_hold_pp"], "")


if __name__ == "__main__":
    unittest.main()
