import unittest

from paired_bootstrap_audit import audit_rows, moving_block_sample, validate_rows
import random


def fixture():
    end = 1_800_000_000
    symbols = ["BTC", "ETH"]
    windows = []
    first = end - 4 * 30 * 86400
    for i in range(4):
        start = first + i * 30 * 86400
        windows.append((f"oos_{i + 1:02d}", start, start + 30 * 86400))
    metadata = {
        "markets_tested": symbols,
        "oos_window_count": 4,
        "oos_window_days": 30,
        "shared_end_timestamp": end,
        "cost_scenarios": [{"name": "base_taker"}],
    }
    rows = []
    for window_index, (name, start, stop) in enumerate(windows):
        for symbol_index, symbol in enumerate(symbols):
            for balance in (5.0, 50.0):
                executable = balance >= 50
                ret = (1.0 + window_index + symbol_index * 0.25) if executable else 0.0
                benchmark_ret = 0.5 if executable else -2.0
                rows.append({
                    "symbol": symbol, "window": name,
                    "window_start_utc": start, "window_end_utc": stop,
                    "shared_end_timestamp": end, "start_balance": balance,
                    "cost_scenario": "base_taker", "strategy_status": "tested" if executable else "low_sample",
                    "strategy_trades": 6 if executable else 0, "strategy_return_pct": ret,
                    "strategy_mtm_drawdown_pct": 2.0 if executable else 0.0,
                    "strategy_profit_factor": 1.2 if executable else 0.0,
                    "benchmark_return_pct": benchmark_ret, "benchmark_mtm_drawdown_pct": 5.0,
                    "benchmark_executable": str(executable),
                    "observed_excess_return_vs_benchmark_pp": (ret - benchmark_ret) if executable else "",
                    "strategy_beats_benchmark_observed": str(executable and ret > benchmark_ret),
                    "paired_evaluable": str(executable),
                })
    return rows, metadata


class PairedBootstrapAuditTests(unittest.TestCase):
    def test_moving_block_sample_is_reproducible_and_preserves_contiguous_blocks(self):
        names = ["a", "b", "c", "d", "e", "f"]
        a = moving_block_sample(names, 2, random.Random(7))
        b = moving_block_sample(names, 2, random.Random(7))
        self.assertEqual(a, b)
        self.assertEqual(len(a), len(names))
        self.assertTrue(all(name in names for name in a))

    def test_bootstrap_flags_five_usd_no_trade_as_not_executable(self):
        rows, metadata = fixture()
        summary, _ = audit_rows(rows, metadata, replicates=500, block_length=2, seed=1)
        small = next(row for row in summary if row["start_balance"] == 5.0)
        self.assertEqual(small["evaluation_status"], "no_executable_samples")
        self.assertEqual(small["total_trades"], 0)
        self.assertEqual(small["strategy_mean_return_pct"], 0.0)
        self.assertEqual(small["eligible_strategy_market_windows"], 0)
        self.assertEqual(small["benchmark_executable_market_windows"], 0)

    def test_positive_fixture_has_positive_bootstrap_mean_interval(self):
        rows, metadata = fixture()
        summary, detail = audit_rows(rows, metadata, replicates=500, block_length=2, seed=2)
        large = next(row for row in summary if row["start_balance"] == 50.0)
        self.assertGreater(large["mean_return_ci95_low"], 0.0)
        self.assertGreater(large["excess_vs_benchmark_mean_ci95_low"], 0.0)
        self.assertEqual(len(detail), 4)

    def test_rejects_symbol_cohort_mismatch_and_incomplete_grid(self):
        rows, metadata = fixture()
        wrong = [dict(row) for row in rows]
        wrong[0]["symbol"] = "AAPL"
        with self.assertRaisesRegex(ValueError, "symbol cohort mismatch"):
            validate_rows(wrong, metadata)
        with self.assertRaisesRegex(ValueError, "incomplete result grid"):
            validate_rows(rows[:-1], metadata)

    def test_rejects_noncontiguous_oos_windows(self):
        rows, metadata = fixture()
        broken = [dict(row) for row in rows]
        for row in broken:
            if row["window"] == "oos_03":
                row["window_start_utc"] += 3600
                row["window_end_utc"] += 3600
        with self.assertRaisesRegex(ValueError, "contiguous"):
            validate_rows(broken, metadata)


if __name__ == "__main__":
    unittest.main()
