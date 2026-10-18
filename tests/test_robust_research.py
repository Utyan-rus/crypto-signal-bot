import unittest

from robust_research import max_drawdown


class RobustResearchTests(unittest.TestCase):
    def test_mark_to_market_drawdown_includes_intratrade_equity_dips(self):
        self.assertAlmostEqual(max_drawdown([5.0, 4.0, 4.5, 5.5, 5.0]), 20.0)

    def test_flat_equity_has_no_drawdown(self):
        self.assertEqual(max_drawdown([5.0, 5.0, 5.0]), 0.0)

    def test_recovery_does_not_erase_prior_drawdown(self):
        self.assertAlmostEqual(max_drawdown([10.0, 8.0, 12.0]), 20.0)


if __name__ == "__main__":
    unittest.main()
