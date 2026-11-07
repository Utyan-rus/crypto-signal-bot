import json
import unittest
from pathlib import Path

import hourly_candidate_research as hourly
from walk_forward_research import (
    BOOTSTRAP_BLOCK_LENGTH, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    COSTS, DIRECTION_MODES, MIN_FUNDING_COVERAGE_PCT, RESEARCH_SYMBOLS,
    STRATEGIES, floor_to_hour, make_windows, validate_window_geometry,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "research" / "forward_holdout_2026-11-07.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "walk-forward-research.yml"


class ProspectiveHoldoutPreregistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cls.workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    def test_future_oos_is_disjoint_from_previously_inspected_research_data(self):
        data = self.config["data_window"]
        self.assertGreater(data["oos_start_timestamp"], data["previous_research_data_end_timestamp"])
        self.assertEqual(data["oos_start_timestamp"], data["oos_end_timestamp"] - 120 * 86400)
        self.assertEqual(floor_to_hour(data["oos_end_timestamp"]), data["oos_end_timestamp"])
        self.assertEqual(data["warmup_start_timestamp"], data["oos_end_timestamp"] - 245 * 86400)

    def test_window_geometry_is_fixed_and_non_overlapping(self):
        data = self.config["data_window"]
        self.assertEqual(validate_window_geometry(245, 4), 125)
        windows = make_windows(data["oos_end_timestamp"], count=4)
        self.assertEqual(len(windows), 4)
        self.assertEqual(windows[0][1], data["oos_start_timestamp"])
        self.assertEqual(windows[-1][2], data["oos_end_timestamp"])
        self.assertTrue(all(windows[i][2] == windows[i + 1][1] for i in range(3)))
        self.assertTrue(all(end - start == 30 * 86400 for _, start, end in windows))

    def test_preregistered_universe_strategy_set_and_direction_match_code(self):
        self.assertEqual(self.config["universe"]["ordered_symbols"], list(RESEARCH_SYMBOLS[:5]))
        self.assertEqual([item["name"] for item in self.config["strategy_set"]], list(STRATEGIES))
        self.assertEqual(self.config["direction_modes"], list(DIRECTION_MODES))
        self.assertEqual(self.config["primary_direction"], "LONG")
        self.assertEqual(self.config["primary_strategy"], "donchian_long_regime")
        self.assertEqual(self.config["secondary_strategy"], "donchian_long_regime_vol")
        roles = {item["name"]: item["role"] for item in self.config["strategy_set"]}
        self.assertEqual(roles["donchian_long_regime"], "primary")
        self.assertEqual(roles["donchian_long_regime_vol"], "secondary_diagnostic_only")

    def test_cost_funding_and_bootstrap_parameters_match_code(self):
        expected_costs = [
            {"name": name, "fee_per_side": fee, "slippage_per_side": slippage}
            for name, fee, slippage in COSTS
        ]
        self.assertEqual(self.config["cost_scenarios"], expected_costs)
        self.assertEqual(self.config["funding"]["minimum_coverage_pct_per_market_window"], MIN_FUNDING_COVERAGE_PCT)
        bootstrap = self.config["bootstrap"]
        self.assertEqual(bootstrap["replicates"], BOOTSTRAP_REPLICATES)
        self.assertEqual(bootstrap["contiguous_window_block_length"], BOOTSTRAP_BLOCK_LENGTH)
        self.assertEqual(bootstrap["seed"], BOOTSTRAP_SEED)

    def test_risk_caps_and_verified_minimum_notional_are_consistent(self):
        execution = self.config["capital_and_execution"]
        self.assertEqual(execution["risk_fraction"], hourly.RISK_FRACTION)
        self.assertEqual(execution["max_leverage"], hourly.MAX_LEVERAGE)
        self.assertEqual(execution["minimum_quote_notional_usd"], hourly.MIN_NOTIONAL)
        five = execution["five_usd_feasibility"]
        self.assertFalse(five["minimum_notional_reachable"])
        self.assertLess(five["maximum_allowed_notional_usd"], execution["minimum_quote_notional_usd"])

    def test_primary_promotion_gate_is_predeclared_and_paper_only(self):
        gate = self.config["primary_promotion_criteria"]
        self.assertEqual(gate["strategy"], self.config["primary_strategy"])
        self.assertEqual(gate["direction"], self.config["primary_direction"])
        self.assertEqual(gate["require_two_way_ci95_lower_mean_return_pct_greater_than"], 0.0)
        self.assertEqual(gate["require_two_way_ci95_lower_mean_excess_vs_1x_long_hold_pp_greater_than"], 0.0)
        self.assertTrue(self.config["paper_only"])
        self.assertFalse(self.config["real_orders_enabled"])

    def test_workflow_cannot_start_backtest_before_preregistered_end(self):
        self.assertIn('default: "1804402800"', self.workflow)
        self.assertIn("id: holdout_gate", self.workflow)
        self.assertIn("if: steps.holdout_gate.outputs.ready == 'true'", self.workflow)
        self.assertIn("if: steps.holdout_gate.outputs.ready != 'true'", self.workflow)
        self.assertIn("test_forward_holdout_preregistration.py", self.workflow)
        self.assertIn("--days 245 --oos-windows 4", self.workflow)
        self.assertIn("--end-timestamp", self.workflow)
        self.assertIn("1804402800", self.workflow)


if __name__ == "__main__":
    unittest.main()
