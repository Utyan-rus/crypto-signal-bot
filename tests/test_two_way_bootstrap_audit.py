import unittest

from two_way_bootstrap_audit import audit, two_way_bootstrap


def fixture():
    end = 1_800_000_000
    symbols = ["BTC", "ETH"]
    first = end - 4 * 30 * 86400
    windows = [
        (f"oos_{i + 1:02d}", first + i * 30 * 86400, first + (i + 1) * 30 * 86400)
        for i in range(4)
    ]
    metadata = {
        "markets_tested": symbols,
        "oos_window_count": 4,
        "oos_window_days": 30,
        "shared_end_timestamp": end,
        "cost_scenarios": [{"name": "base_taker"}],
        "direction_modes": ["BOTH"],
    }
    rows = []
    for index, (window, start, stop) in enumerate(windows):
        for symbol_index, symbol in enumerate(symbols):
            ret = 1.0 + index + 0.5 * symbol_index
            rows.append({
                "symbol": symbol,
                "window": window,
                "window_start_utc": start,
                "window_end_utc": stop,
                "shared_end_timestamp": end,
                "start_balance": 50.0,
                "cost_scenario": "base_taker",
                "direction_mode": "BOTH",
                "strategy_status": "tested",
                "strategy_trades": 6,
                "strategy_return_pct": ret,
                "strategy_mtm_drawdown_pct": 2.0 + 0.1 * index,
                "strategy_profit_factor": 1.2 + 0.1 * index,
                "benchmark_return_pct": 0.5,
                "benchmark_executable": "true",
                "paired_evaluable": "true",
                "observed_excess_return_vs_benchmark_pp": ret - 0.5,
                "strategy_beats_benchmark_observed": str(ret > 0.5).lower(),
            })
    return rows, metadata


class TwoWayBootstrapAuditTests(unittest.TestCase):
    def test_two_way_bootstrap_is_reproducible(self):
        rows, metadata = fixture()
        windows = sorted({row["window"] for row in rows})
        symbols = metadata["markets_tested"]
        first = two_way_bootstrap(rows, windows, symbols, replicates=500, block_length=2, seed=17)
        second = two_way_bootstrap(rows, windows, symbols, replicates=500, block_length=2, seed=17)
        self.assertEqual(first, second)
        self.assertGreater(first["two_way_mean_return_pct_ci95_low"], 0)
        self.assertGreater(first["two_way_mean_excess_vs_benchmark_pp_ci95_low"], 0)

    def test_audit_emits_full_cohort_and_leave_one_market_out(self):
        rows, metadata = fixture()
        summary, lomo = audit(rows, metadata, replicates=500, block_length=2, seed=18)
        self.assertEqual(len(summary), 1)
        self.assertEqual(len(lomo), 3)
        full = next(row for row in lomo if row["omitted_symbol"] == "NONE")
        without_btc = next(row for row in lomo if row["omitted_symbol"] == "BTC")
        without_eth = next(row for row in lomo if row["omitted_symbol"] == "ETH")
        self.assertAlmostEqual(full["strategy_mean_return_pct"], 2.75, places=4)
        self.assertAlmostEqual(without_btc["strategy_mean_return_pct"], 3.0, places=4)
        self.assertAlmostEqual(without_eth["strategy_mean_return_pct"], 2.5, places=4)
        self.assertEqual(full["paired_evaluable_market_windows"], 8)

    def test_incomplete_market_window_grid_is_rejected(self):
        rows, metadata = fixture()
        with self.assertRaisesRegex(ValueError, "incomplete result grid"):
            audit(rows[:-1], metadata, replicates=100, block_length=2, seed=19)


if __name__ == "__main__":
    unittest.main()
