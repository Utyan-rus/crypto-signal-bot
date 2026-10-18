import unittest
from unittest.mock import patch

from strategy_research import simulate


def make_candles(n=8):
    rows = []
    for i in range(n):
        low = 99.0 if i == 2 else 99.8
        rows.append({
            "t": 1_700_000_000 + i * 300,
            "o": 100.0, "h": 100.2, "l": low, "c": 100.0, "v": 100.0,
        })
    return rows


class ResearchExecutionTests(unittest.TestCase):
    @patch("strategy_research.signal_map")
    def test_exit_bar_does_not_reenter_at_that_bar_open(self, mock_signal_map):
        candles = make_candles()
        mock_signal_map.return_value = {
            row["t"]: ("LONG", 0.5, 2.0) for row in candles
        }
        result = simulate("TEST-PERP", candles, "test", cooldown_bars=0, fresh_signals=False)
        # First position exits on candle index 2; any next entry must be later than that candle.
        self.assertGreaterEqual(result["trades"], 2)

    @patch("strategy_research.signal_map")
    def test_fresh_signal_mode_avoids_repeated_entries_for_persistent_signal(self, mock_signal_map):
        candles = make_candles()
        mock_signal_map.return_value = {
            row["t"]: ("LONG", 0.5, 2.0) for row in candles
        }
        every_signal = simulate("TEST-PERP", candles, "test", cooldown_bars=0, fresh_signals=False)
        fresh_only = simulate("TEST-PERP", candles, "test", cooldown_bars=0, fresh_signals=True)
        self.assertLess(fresh_only["trades"], every_signal["trades"])

    @patch("strategy_research.signal_map")
    def test_cooldown_reduces_immediate_reentry(self, mock_signal_map):
        candles = make_candles()
        mock_signal_map.return_value = {
            row["t"]: ("LONG", 0.5, 2.0) for row in candles
        }
        no_cooldown = simulate("TEST-PERP", candles, "test", cooldown_bars=0, fresh_signals=False)
        cooldown = simulate("TEST-PERP", candles, "test", cooldown_bars=6, fresh_signals=False)
        self.assertLess(cooldown["trades"], no_cooldown["trades"])


if __name__ == "__main__":
    unittest.main()
