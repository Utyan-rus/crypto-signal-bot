import unittest
from unittest.mock import patch

import adx_filtered_donchian_research as research


def make_candles(count=240, drift=0.1):
    rows = []
    price = 100.0
    for index in range(count):
        price += drift
        rows.append({
            "t": 1_700_000_000 + index * 3600,
            "o": price,
            "h": price + 0.5,
            "l": price - 0.5,
            "c": price,
            "v": 100.0,
        })
    return rows


class AdxFilteredDonchianTests(unittest.TestCase):
    def test_wilder_adx_is_high_in_a_persistent_trend(self):
        values = research.adx_values(make_candles(drift=0.1))
        self.assertIsNone(values[26])
        self.assertGreaterEqual(values[-1], 20.0)

    def test_wilder_adx_is_low_without_directional_movement(self):
        values = research.adx_values(make_candles(drift=0.0))
        self.assertEqual(values[-1], 0.0)

    def test_filter_is_non_directional_when_adx_is_high(self):
        rows = make_candles(drift=0.1)
        timestamp = rows[-1]["t"]
        signals = {
            timestamp: ("LONG", 2.0),
            rows[-2]["t"]: ("SHORT", 2.0),
        }
        filtered = research.apply_adx_filter(rows, signals)
        self.assertEqual(filtered, signals)

    def test_low_adx_rejects_long_and_short(self):
        rows = make_candles(drift=0.0)
        signals = {
            rows[-1]["t"]: ("LONG", 2.0),
            rows[-2]["t"]: ("SHORT", 2.0),
        }
        self.assertEqual(research.apply_adx_filter(rows, signals), {})

    def test_filter_uses_only_candles_available_at_signal_time(self):
        rows = make_candles(count=80, drift=0.1)
        timestamp = rows[-1]["t"]
        signals = {timestamp: ("LONG", 2.0)}
        original = research.apply_adx_filter(rows, signals)
        future = rows + [{
            "t": rows[-1]["t"] + 3600,
            "o": 1_000_000.0,
            "h": 1_000_001.0,
            "l": 999_999.0,
            "c": 1_000_000.0,
            "v": 100.0,
        }]
        self.assertEqual(original, research.apply_adx_filter(future, signals))

    def test_missing_adx_warmup_fails_closed(self):
        rows = make_candles(count=25, drift=0.1)
        signals = {rows[-1]["t"]: ("LONG", 2.0)}
        self.assertEqual(research.apply_adx_filter(rows, signals), {})

    def test_invalid_period_or_threshold_is_rejected(self):
        rows = make_candles()
        with self.assertRaisesRegex(ValueError, "period"):
            research.adx_values(rows, period=1)
        with self.assertRaisesRegex(ValueError, "threshold"):
            research.apply_adx_filter(rows, {}, threshold=101.0)

    def test_candidate_reuses_baseline_exits_and_only_filters_signals(self):
        rows = make_candles()
        timestamp = rows[-1]["t"]
        signals = {timestamp: ("LONG", 2.0)}
        exits = {timestamp: (99.0, 101.0)}
        with patch.object(research, "base_build_model", return_value=(signals, exits)):
            candidate_signals, candidate_exits = research.build_model_with_adx_filter(
                rows, research.CANDIDATE
            )
            baseline_signals, baseline_exits = research.build_model_with_adx_filter(
                rows, research.BASELINE
            )
        self.assertEqual(candidate_exits, exits)
        self.assertEqual(baseline_exits, exits)
        self.assertEqual(baseline_signals, signals)
        self.assertEqual(candidate_signals, research.apply_adx_filter(rows, signals))

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported strategy"):
            research.build_model_with_adx_filter(make_candles(), "unregistered")


if __name__ == "__main__":
    unittest.main()
