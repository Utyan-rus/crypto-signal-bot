import unittest
from unittest.mock import patch

import funding_aware_research as funding_research
import hourly_candidate_research as hourly
from walk_forward_research import (COSTS, DIRECTION_MODES, MIN_FUNDING_COVERAGE_PCT, RESEARCH_SYMBOLS, bootstrap_summaries, buy_hold_baseline, capital_feasibility_audit, funding_coverage, funding_coverage_is_acceptable, floor_to_hour, make_windows, select_research_markets, simulate_funding_audited, summarize, validate_window_geometry, with_long_buy_hold_benchmark)


class WalkForwardResearchTests(unittest.TestCase):
    def test_end_timestamp_uses_last_fully_closed_hour(self):
        self.assertEqual(floor_to_hour(1793692773), 1793689200)
        self.assertEqual(floor_to_hour(1793689200), 1793689200)

    def test_windows_are_chronological_non_overlapping_and_cover_240_days(self):
        end = 2_000_000_000
        windows = make_windows(end)
        self.assertEqual(len(windows), 8)
        self.assertTrue(all(windows[i][2] == windows[i + 1][1] for i in range(3)))
        self.assertEqual(windows[-1][2], end)
        self.assertEqual(windows[-1][2] - windows[0][1], 240 * 86400)

    def test_extended_windows_cover_600_days_without_overlap(self):
        end = 2_000_000_000
        windows = make_windows(end, count=20)
        self.assertEqual(len(windows), 20)
        self.assertTrue(all(windows[i][2] == windows[i + 1][1]
                            for i in range(len(windows) - 1)))
        self.assertEqual(windows[-1][2], end)
        self.assertEqual(windows[-1][2] - windows[0][1], 600 * 86400)

    def test_history_geometry_requires_125_day_warmup(self):
        self.assertEqual(validate_window_geometry(725, 20), 125)
        with self.assertRaises(ValueError):
            validate_window_geometry(724, 20)
        with self.assertRaises(ValueError):
            validate_window_geometry(725, 0)

    def test_disjoint_245_day_holdout_has_fixed_warmup_and_four_windows(self):
        # This OOS period ends before the later paired Donchian dataset begins.
        end = 1759366800  # 2025-10-02 01:00 UTC
        self.assertEqual(validate_window_geometry(245, 4), 125)
        windows = make_windows(end, count=4)
        self.assertEqual(len(windows), 4)
        self.assertTrue(all(windows[i][2] == windows[i + 1][1]
                            for i in range(len(windows) - 1)))
        self.assertEqual(windows[0][1], end - 120 * 86400)
        self.assertEqual(windows[-1][2], end)
        self.assertLessEqual(end, 1759369802)  # paired dataset history start

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


    def test_buy_hold_baseline_applies_signed_funding(self):
        candles = [
            {"t": 0, "o": 100.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "c": 100.0},
            {"t": 7200, "o": 100.0, "c": 100.0},
        ]
        result = buy_hold_baseline(
            candles, 0, 10800, 50.0, 0.0, 0.0, 10.0,
            funding={3600: 0.01, 7200: -0.005}
        )
        self.assertTrue(result["executable"])
        self.assertEqual(result["funding_events_charged"], 2)
        self.assertAlmostEqual(result["funding_net_cost_usd"], 0.25)
        self.assertAlmostEqual(result["return_pct"], -0.5)

    def test_short_hold_baseline_applies_inverse_price_and_signed_funding(self):
        flat = [
            {"t": 0, "o": 100.0, "c": 100.0},
            {"t": 3600, "o": 100.0, "c": 100.0},
            {"t": 7200, "o": 100.0, "c": 100.0},
        ]
        funded = buy_hold_baseline(
            flat, 0, 10800, 50.0, 0.0, 0.0, 10.0,
            funding={3600: 0.01, 7200: -0.005}, direction="SHORT"
        )
        self.assertTrue(funded["executable"])
        self.assertEqual(funded["status"], "executable_1x_short_with_funding")
        self.assertEqual(funded["funding_events_charged"], 2)
        self.assertAlmostEqual(funded["funding_net_cost_usd"], -0.25)
        self.assertAlmostEqual(funded["return_pct"], 0.5)

        falling = [
            {"t": 0, "o": 100.0, "c": 100.0},
            {"t": 3600, "o": 90.0, "c": 80.0},
        ]
        rising = [
            {"t": 0, "o": 100.0, "c": 100.0},
            {"t": 3600, "o": 110.0, "c": 120.0},
        ]
        short_win = buy_hold_baseline(
            falling, 0, 7200, 50.0, 0.0, 0.0, 10.0, direction="SHORT"
        )
        short_loss = buy_hold_baseline(
            rising, 0, 7200, 50.0, 0.0, 0.0, 10.0, direction="SHORT"
        )
        self.assertAlmostEqual(short_win["return_pct"], 20.0)
        self.assertAlmostEqual(short_loss["return_pct"], -20.0)
        costed = buy_hold_baseline(
            falling, 0, 7200, 50.0, 0.0005, 0.0005, 10.0, direction="SHORT"
        )
        self.assertLess(costed["return_pct"], short_win["return_pct"])

    def test_hold_baseline_rejects_unknown_direction(self):
        with self.assertRaises(ValueError):
            buy_hold_baseline([], 0, 3600, 50.0, 0.0, 0.0, 10.0, direction="BOTH")

    def test_long_benchmark_mapping_preserves_long_hold_fields(self):
        mapped = with_long_buy_hold_benchmark([{
            "buy_hold_executable": True,
            "buy_hold_return_pct": 1.25,
            "excess_vs_buy_hold_pp": -0.5,
            "strategy_beats_buy_hold_observed": False,
            "long_buy_hold_paired_evaluable": True,
            "benchmark_executable": True,
            "benchmark_return_pct": -2.0,
            "observed_excess_return_vs_benchmark_pp": 1.5,
            "strategy_beats_benchmark_observed": True,
        }])[0]
        self.assertTrue(mapped["benchmark_executable"])
        self.assertEqual(mapped["benchmark_return_pct"], 1.25)
        self.assertEqual(mapped["observed_excess_return_vs_benchmark_pp"], -0.5)
        self.assertFalse(mapped["strategy_beats_benchmark_observed"])
        self.assertTrue(mapped["paired_evaluable"])


    def test_funding_coverage_counts_exact_hourly_rows_and_reports_gaps(self):
        full = funding_coverage({0: 0.0, 3600: 0.001, 7200: -0.001}, 0, 10800)
        partial = funding_coverage({0: 0.0, 3600: 0.001}, 0, 10800)
        self.assertEqual(full["funding_rows"], 3)
        self.assertEqual(full["expected_funding_hours"], 3)
        self.assertEqual(full["funding_coverage_pct"], 100.0)
        self.assertEqual(partial["missing_funding_hours"], 1)
        self.assertAlmostEqual(partial["funding_coverage_pct"], 200 / 3, places=2)
        self.assertEqual(MIN_FUNDING_COVERAGE_PCT, 99.0)
        self.assertTrue(funding_coverage_is_acceptable(full))
        self.assertFalse(funding_coverage_is_acceptable(partial))
        misaligned = funding_coverage({0: 0.0, 3600: 0.001, 7200: -0.001, 1800: 0.0001}, 0, 10800)
        self.assertFalse(funding_coverage_is_acceptable(misaligned))

    def test_funding_aware_simulator_matches_no_funding_hourly_baseline(self):
        candles = [
            {"t": 0, "o": 100.0, "h": 101.0, "l": 99.0, "c": 100.0, "v": 100.0},
            {"t": 3600, "o": 100.0, "h": 101.0, "l": 99.0, "c": 101.0, "v": 100.0},
            {"t": 7200, "o": 101.0, "h": 102.0, "l": 100.0, "c": 101.5, "v": 100.0},
        ]
        signals = {0: ("LONG", 2.0)}
        with patch.object(hourly, "build_model", return_value=(signals, {})), \
             patch.object(funding_research, "build_model", return_value=(signals, {})):
            original_balance = hourly.START_BALANCE
            hourly.START_BALANCE = 50.0
            try:
                baseline = hourly.simulate("TEST", candles, "stub", 0, 10800)
                funded = simulate_funding_audited(
                    "TEST", candles, "stub", 0, 10800, {}, 50.0,
                    hourly.FEE, hourly.SLIPPAGE
                )
            finally:
                hourly.START_BALANCE = original_balance
        self.assertEqual(funded["trades"], baseline["trades"])
        self.assertAlmostEqual(funded["return_pct"], baseline["return_pct"], places=3)
        self.assertAlmostEqual(funded["max_drawdown_mtm_pct"], baseline["max_drawdown_mtm_pct"], places=3)

    def test_five_usd_balance_cannot_meet_min_notional_without_breaking_risk_cap(self):
        audit = capital_feasibility_audit(5.0, 0.005, 0.003, 3.0, 10.0)
        self.assertAlmostEqual(audit["maximum_risk_sized_notional_usd"], 8.333333, places=5)
        self.assertEqual(audit["maximum_leverage_notional_usd"], 15.0)
        self.assertFalse(audit["minimum_notional_reachable_at_minimum_stop"])
        self.assertEqual(audit["minimum_balance_usd_at_minimum_stop"], 6.0)
        self.assertTrue(audit["upper_bound_only"])
        larger = capital_feasibility_audit(50.0, 0.005, 0.003, 3.0, 10.0)
        self.assertTrue(larger["minimum_notional_reachable_at_minimum_stop"])

    def test_funding_aware_simulator_counts_zero_sized_min_notional_rejections(self):
        candles = [
            {"t": 0, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0, "v": 100.0},
            {"t": 3600, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0, "v": 100.0},
            {"t": 7200, "o": 100.0, "h": 100.5, "l": 99.5, "c": 100.0, "v": 100.0},
        ]
        with patch.object(funding_research, "build_model", return_value=({0: ("LONG", 2.0)}, {})):
            result = simulate_funding_audited(
                "TEST", candles, "stub", 0, 10800, {}, 5.0, 0.0, 0.0
            )
        self.assertEqual(result["trades"], 0)
        self.assertEqual(result["min_notional_rejections"], 1)

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

    def test_bootstrap_reports_directional_and_long_buy_hold_excess_separately(self):
        symbols = ["BTC", "ETH"]
        windows = ["w1", "w2", "w3", "w4"]
        rows = []
        for direction in ("LONG", "SHORT"):
            for symbol_index, symbol in enumerate(symbols):
                for window_index, window in enumerate(windows):
                    strategy_return = (1.0 if direction == "LONG" else -1.0) + 0.1 * (window_index % 2)
                    long_benchmark = 0.25
                    directional_benchmark = 0.25 if direction == "LONG" else -0.5
                    rows.append({
                        "strategy": "donchian_20_10", "cost_scenario": "base",
                        "symbol": symbol, "window": window, "start_balance": 50.0,
                        "direction_mode": direction, "strategy_return_pct": strategy_return,
                        "strategy_mtm_drawdown_pct": 2.0, "strategy_status": "tested",
                        "strategy_trades": 10, "strategy_profit_factor": 1.2,
                        "buy_hold_executable": True, "buy_hold_return_pct": long_benchmark,
                        "long_buy_hold_paired_evaluable": True,
                        "excess_vs_buy_hold_pp": strategy_return - long_benchmark,
                        "strategy_beats_buy_hold_observed": strategy_return > long_benchmark,
                        "benchmark_executable": True, "benchmark_return_pct": directional_benchmark,
                        "paired_evaluable": True,
                        "observed_excess_return_vs_benchmark_pp": strategy_return - directional_benchmark,
                        "strategy_beats_benchmark_observed": strategy_return > directional_benchmark,
                    })
        result = bootstrap_summaries(rows, windows, symbols, replicates=100,
                                     block_length=2, seed=321)
        by_direction = {row["direction_mode"]: row for row in result}
        short = by_direction["SHORT"]
        self.assertAlmostEqual(short["mean_excess_vs_directional_benchmark_pp"], -0.45)
        self.assertAlmostEqual(short["mean_excess_vs_long_buy_hold_pp"], -1.20)
        self.assertIn("two_way_mean_excess_vs_directional_benchmark_pp_ci95_low", short)
        self.assertIn("two_way_mean_excess_vs_long_buy_hold_pp_ci95_low", short)
        self.assertEqual(short["directional_benchmark_role"], "1x short hold")


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


    def test_direction_modes_are_predeclared_and_ordered(self):
        self.assertEqual(DIRECTION_MODES, ("BOTH", "LONG", "SHORT"))

    def test_simulate_funding_audited_passes_direction_mode(self):
        fake_result = {
            "symbol": "TEST", "strategy": "stub", "trades": 0,
            "return_pct": 0.0, "max_drawdown_mtm_pct": 0.0,
            "profit_factor": 0.0, "min_notional_rejections": 0,
        }
        with patch.object(funding_research, "simulate_with_funding",
                          return_value=fake_result) as simulator:
            result = simulate_funding_audited(
                "TEST", [], "stub", 0, 3600, {}, 50.0, 0.0, 0.0,
                direction_mode="SHORT"
            )
        self.assertEqual(simulator.call_args.kwargs["direction_mode"], "SHORT")
        self.assertEqual(result["trades"], 0)

    def test_summary_keeps_long_and_short_direction_rows_separate(self):
        common = {
            "strategy": "s", "cost_scenario": "base", "status": "tested",
            "symbol": "BTC", "window": "oos_1", "max_drawdown_mtm_pct": 2,
            "trades": 10, "profit_factor": 1.2, "min_notional_rejections": 0,
            "buy_hold_executable": True, "buy_hold_return_pct": 0.5,
            "paired_evaluable": True,
        }
        rows = [
            {**common, "direction_mode": "LONG", "return_pct": 1.0,
             "excess_vs_buy_hold_pp": 0.5, "strategy_beats_benchmark_observed": True},
            {**common, "direction_mode": "SHORT", "return_pct": -1.0,
             "excess_vs_buy_hold_pp": -1.5, "strategy_beats_benchmark_observed": False},
        ]
        result = summarize(rows)
        by_direction = {row["direction_mode"]: row for row in result}
        self.assertEqual(set(by_direction), {"LONG", "SHORT"})
        self.assertEqual(by_direction["LONG"]["mean_return_pct"], 1.0)
        self.assertEqual(by_direction["SHORT"]["mean_return_pct"], -1.0)

    def test_summary_reports_directional_and_long_benchmarks_separately(self):
        common = {
            "strategy": "s", "cost_scenario": "base", "status": "tested",
            "symbol": "BTC", "window": "oos_1", "max_drawdown_mtm_pct": 2,
            "trades": 10, "profit_factor": 1.2, "min_notional_rejections": 0,
            "buy_hold_executable": True, "buy_hold_return_pct": 1.0,
            "long_buy_hold_paired_evaluable": True, "paired_evaluable": True,
        }
        rows = [
            {**common, "direction_mode": "LONG", "return_pct": 1.5,
             "excess_vs_buy_hold_pp": 0.5, "strategy_beats_buy_hold_observed": True,
             "benchmark_executable": True, "benchmark_return_pct": 1.0,
             "observed_excess_return_vs_benchmark_pp": 0.5,
             "strategy_beats_benchmark_observed": True},
            {**common, "direction_mode": "SHORT", "return_pct": -1.0,
             "excess_vs_buy_hold_pp": -2.0, "strategy_beats_buy_hold_observed": False,
             "benchmark_executable": True, "benchmark_return_pct": -0.25,
             "observed_excess_return_vs_benchmark_pp": -0.75,
             "strategy_beats_benchmark_observed": False},
        ]
        result = summarize(rows)
        by_direction = {row["direction_mode"]: row for row in result}
        short = by_direction["SHORT"]
        self.assertEqual(short["buy_hold_mean_return_pct"], 1.0)
        self.assertEqual(short["mean_excess_vs_buy_hold_pp"], -2.0)
        self.assertEqual(short["directional_benchmark_mean_return_pct"], -0.25)
        self.assertEqual(short["mean_excess_vs_directional_benchmark_pp"], -0.75)
        self.assertEqual(short["strategy_win_fraction_vs_directional_benchmark_pct"], 0.0)


    def test_bootstrap_summaries_keep_direction_grids_separate(self):
        symbols = ["BTC", "ETH"]
        windows = ["w1", "w2", "w3", "w4"]
        rows = []
        for direction in ("LONG", "SHORT"):
            for symbol_index, symbol in enumerate(symbols):
                for window_index, window in enumerate(windows):
                    strategy_return = (1.0 if direction == "LONG" else -1.0) + 0.1 * (window_index % 2)
                    benchmark_return = 0.25
                    rows.append({
                        "strategy": "donchian_20_10", "cost_scenario": "base",
                        "symbol": symbol, "window": window, "start_balance": 50.0,
                        "direction_mode": direction, "strategy_return_pct": strategy_return,
                        "strategy_mtm_drawdown_pct": 2.0, "strategy_status": "tested",
                        "strategy_trades": 10, "strategy_profit_factor": 1.2,
                        "benchmark_executable": True, "benchmark_return_pct": benchmark_return,
                        "paired_evaluable": True,
                        "observed_excess_return_vs_benchmark_pp": strategy_return - benchmark_return,
                        "strategy_beats_benchmark_observed": strategy_return > benchmark_return,
                    })
        result = bootstrap_summaries(rows, windows, symbols, replicates=100,
                                     block_length=2, seed=123)
        by_direction = {row["direction_mode"]: row for row in result}
        self.assertEqual(set(by_direction), {"LONG", "SHORT"})
        self.assertEqual(by_direction["LONG"]["market_window_rows"], 8)
        self.assertEqual(by_direction["SHORT"]["market_window_rows"], 8)
        self.assertGreater(by_direction["LONG"]["mean_return_pct"],
                           by_direction["SHORT"]["mean_return_pct"])
        self.assertEqual(by_direction["LONG"]["status"], "ok")
        self.assertEqual(by_direction["SHORT"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
