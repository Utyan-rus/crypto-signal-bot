import unittest
from unittest.mock import patch

import slope_filtered_donchian_research as research


def make_candles(count=240, drift=0.1):
    rows = []
    price = 100.0
    for index in range(count):
        price += drift
        rows.append({
            "t": 1_700_000_000 + index * 3600,
            "o": price,
            "h": price + 0.2,
            "l": price - 0.2,
            "c": price,
            "v": 100.0,
        })
    return rows


class SlopeFilteredDonchianTests(unittest.TestCase):
    def test_rising_ema200_keeps_long_and_rejects_short(self):
        rows = make_candles(drift=0.1)
        timestamp = rows[-1]["t"]
        signals = {
            timestamp: ("LONG", 2.0),
            rows[-2]["t"]: ("SHORT", 2.0),
        }
        filtered = research.apply_ema200_slope_filter(rows, signals)
        self.assertIn(timestamp, filtered)
        self.assertNotIn(rows[-2]["t"], filtered)

    def test_falling_ema200_keeps_short_and_rejects_long(self):
        rows = make_candles(drift=-0.1)
        timestamp = rows[-1]["t"]
        signals = {timestamp: ("SHORT", 2.0), rows[-2]["t"]: ("LONG", 2.0)}
        filtered = research.apply_ema200_slope_filter(rows, signals)
        self.assertIn(timestamp, filtered)
        self.assertNotIn(rows[-2]["t"], filtered)

    def test_filter_uses_only_closed_candles_available_at_signal_time(self):
        rows = make_candles(count=221, drift=0.1)
        signal_time = rows[-1]["t"]
        signals = {signal_time: ("LONG", 2.0)}
        original = research.apply_ema200_slope_filter(rows, signals)
        future = rows + [{
            "t": rows[-1]["t"] + 3600,
            "o": 1_000_000.0,
            "h": 1_000_001.0,
            "l": 999_999.0,
            "c": 1_000_000.0,
            "v": 100.0,
        }]
        extended = research.apply_ema200_slope_filter(future, signals)
        self.assertEqual(original, extended)

    def test_missing_ema_slope_warmup_fails_closed(self):
        rows = make_candles(count=205, drift=0.1)
        signals = {rows[-1]["t"]: ("LONG", 2.0)}
        self.assertEqual(research.apply_ema200_slope_filter(rows, signals), {})

    def test_candidate_reuses_baseline_exits_and_only_filters_signals(self):
        rows = make_candles()
        timestamp = rows[-1]["t"]
        signals = {timestamp: ("LONG", 2.0)}
        exits = {timestamp: (99.0, 101.0)}
        with patch.object(research, "base_build_model", return_value=(signals, exits)):
            candidate_signals, candidate_exits = research.build_model_with_slope_filter(
                rows, research.CANDIDATE
            )
            baseline_signals, baseline_exits = research.build_model_with_slope_filter(
                rows, research.BASELINE
            )
        self.assertEqual(candidate_signals, signals)
        self.assertEqual(candidate_exits, exits)
        self.assertEqual(baseline_signals, signals)
        self.assertEqual(baseline_exits, exits)

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported strategy"):
            research.build_model_with_slope_filter(make_candles(), "unregistered")


if __name__ == "__main__":
    unittest.main()
