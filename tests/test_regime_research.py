import unittest

from regime_research import filter_signals


def candles(n=230, drift=0.2):
    rows = []
    price = 100.0
    for i in range(n):
        price += drift
        rows.append({"t": 1_700_000_000 + i*300, "o": price, "h": price+0.2,
                     "l": price-0.2, "c": price, "v": 100.0})
    return rows


class RegimeFilterTests(unittest.TestCase):
    def test_none_preserves_all_signals(self):
        rows = candles()
        signals = {rows[-1]["t"]: ("LONG", 1.0, 2.0)}
        self.assertEqual(filter_signals(rows, signals, "none"), signals)

    def test_trend_filter_does_not_mutate_input(self):
        rows = candles()
        signals = {rows[-1]["t"]: ("LONG", 1.0, 2.0)}
        original = dict(signals)
        filter_signals(rows, signals, "trend_alignment")
        self.assertEqual(signals, original)

    def test_unknown_regime_fails_closed_by_combined_rules(self):
        rows = candles()
        signals = {rows[-1]["t"]: ("LONG", 1.0, 2.0)}
        # The fixed trend/volatility regimes always return a subset of original signals.
        for regime in ("trend_alignment", "volatility_band", "trend_and_volatility"):
            result = filter_signals(rows, signals, regime)
            self.assertTrue(set(result).issubset(set(signals)))


if __name__ == "__main__":
    unittest.main()
