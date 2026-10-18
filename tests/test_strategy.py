import math
import unittest

import pandas as pd

from analysis import analyze, STRATEGIES
from backtest import signal_series


def candles(n=260, flat=True):
    rows = []
    for i in range(n):
        close = 100.0 if flat else 100 + 0.025 * i + 0.35 * math.sin(i / 4.0) + 0.12 * math.sin(i / 1.7)
        rows.append({
            "t": 1_700_000_000 + i * 300,
            "o": close - 0.05, "h": close + 0.25, "l": close - 0.25,
            "c": close, "v": 100.0 + 5.0 * math.sin(i / 3.0),
        })
    return rows


class GuardedStrategyTests(unittest.TestCase):
    def test_guarded_is_registered(self):
        self.assertIn("guarded", STRATEGIES)

    def test_insufficient_history_is_neutral(self):
        frame = pd.DataFrame([{
            "open": 100, "high": 101, "low": 99, "close": 100, "volume": 10
        }] * 100)
        self.assertEqual(analyze(frame, strategy="guarded").direction, "NEUTRAL")

    def test_flat_market_does_not_force_a_trade(self):
        frame = pd.DataFrame([{
            "open": x["o"], "high": x["h"], "low": x["l"],
            "close": x["c"], "volume": x["v"]
        } for x in candles(flat=True)])
        self.assertEqual(analyze(frame, strategy="guarded").direction, "NEUTRAL")

    def test_backtest_signal_rows_include_atr_stop_distance(self):
        signals = signal_series(candles(flat=True))
        self.assertTrue(signals)
        for signal in signals.values():
            self.assertEqual(len(signal), 4)
            self.assertIn(signal[0], ("LONG", "SHORT", "NEUTRAL"))
            self.assertGreaterEqual(signal[3], 0)


if __name__ == "__main__":
    unittest.main()
