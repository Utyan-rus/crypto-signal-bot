"""
analysis.py
Computes technical indicators and turns them into a Long/Short/Neutral
suggestion with a confidence score and a risk read (volatility-based).

This is a rule-based heuristic over public market data. It is NOT
financial advice, it does not predict the future, and it can be
wrong — markets can move against any signal at any time.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    mid = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


class Signal:
    def __init__(self, direction: str, confidence: int, risk_level: str, reasons: list,
                 entry: float, stop_loss: float, take_profit: float, atr_value: float):
        self.direction = direction          # "LONG", "SHORT", "NEUTRAL"
        self.confidence = confidence        # 0-100
        self.risk_level = risk_level        # "LOW", "MEDIUM", "HIGH"
        self.reasons = reasons
        self.entry = entry
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr_value = atr_value


def analyze(df: pd.DataFrame, order_book_imbalance: float = 0.0,
            funding_rate: float | None = None) -> Signal:
    close = df["close"]
    last_price = close.iloc[-1]

    ema_fast = ema(close, 20).iloc[-1]
    ema_slow = ema(close, 50).iloc[-1]
    rsi_val = rsi(close).iloc[-1]
    macd_line, signal_line, hist = macd(close)
    macd_hist_val = hist.iloc[-1]
    atr_series = atr(df)
    atr_val = atr_series.iloc[-1]
    upper_bb, mid_bb, lower_bb = bollinger_bands(close)

    score = 0
    reasons = []

    # Trend: EMA crossover
    if ema_fast > ema_slow:
        score += 2
        reasons.append("EMA20 above EMA50 — short-term trend is up")
    else:
        score -= 2
        reasons.append("EMA20 below EMA50 — short-term trend is down")

    # Momentum: RSI
    if rsi_val < 30:
        score += 2
        reasons.append(f"RSI {rsi_val:.1f} — oversold, possible bounce")
    elif rsi_val > 70:
        score -= 2
        reasons.append(f"RSI {rsi_val:.1f} — overbought, possible pullback")
    else:
        reasons.append(f"RSI {rsi_val:.1f} — neutral zone")

    # Momentum: MACD histogram
    if macd_hist_val > 0:
        score += 1
        reasons.append("MACD histogram positive — bullish momentum")
    else:
        score -= 1
        reasons.append("MACD histogram negative — bearish momentum")

    # Mean reversion: Bollinger position
    if last_price <= lower_bb.iloc[-1]:
        score += 1
        reasons.append("Price at/below lower Bollinger Band")
    elif last_price >= upper_bb.iloc[-1]:
        score -= 1
        reasons.append("Price at/above upper Bollinger Band")

    # Order book imbalance
    if order_book_imbalance > 0.1:
        score += 1
        reasons.append(f"Order book skewed to buy-side ({order_book_imbalance:.2f})")
    elif order_book_imbalance < -0.1:
        score -= 1
        reasons.append(f"Order book skewed to sell-side ({order_book_imbalance:.2f})")

    # Funding rate (perpetuals): very positive funding = crowded longs = contrarian short lean
    if funding_rate is not None:
        if funding_rate > 0.0005:
            score -= 1
            reasons.append(f"Funding rate high ({funding_rate:.4%}) — longs crowded/expensive")
        elif funding_rate < -0.0005:
            score += 1
            reasons.append(f"Funding rate negative ({funding_rate:.4%}) — shorts crowded/expensive")

    # Direction + confidence
    max_score = 8
    confidence = min(100, int(abs(score) / max_score * 100))
    if score >= 2:
        direction = "LONG"
    elif score <= -2:
        direction = "SHORT"
    else:
        direction = "NEUTRAL"
        confidence = min(confidence, 40)  # cap confidence when signal is weak/mixed

    # Risk level from volatility (ATR as % of price)
    atr_pct = (atr_val / last_price) * 100 if last_price else 0
    if atr_pct < 1:
        risk_level = "LOW"
    elif atr_pct < 3:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # Suggested stop-loss / take-profit based on ATR (2x ATR stop, 3x ATR target)
    if direction == "LONG":
        stop_loss = last_price - 2 * atr_val
        take_profit = last_price + 3 * atr_val
    elif direction == "SHORT":
        stop_loss = last_price + 2 * atr_val
        take_profit = last_price - 3 * atr_val
    else:
        stop_loss = last_price - 2 * atr_val
        take_profit = last_price + 2 * atr_val

    return Signal(
        direction=direction,
        confidence=confidence,
        risk_level=risk_level,
        reasons=reasons,
        entry=last_price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        atr_value=atr_val,
    )


def backtest(df: pd.DataFrame, lookback: int = 60, horizon: int = 24) -> dict:
    """
    Retrospective check: walks a rolling window through the given candles,
    recomputes this same rule-based signal at each point using only data
    available up to that point, then checks whether price moved in the
    signal's direction `horizon` candles later.

    This is purely a statistic over historical data with these exact
    rules — not a prediction, and no guarantee future performance will
    look similar.
    """
    results = {"LONG": {"count": 0, "wins": 0}, "SHORT": {"count": 0, "wins": 0}}
    close = df["close"]
    n = len(df)

    for i in range(lookback, n - horizon):
        window = df.iloc[: i + 1]
        sig = analyze(window)
        if sig.direction == "NEUTRAL":
            continue

        entry_price = close.iloc[i]
        future_price = close.iloc[i + horizon]

        bucket = results[sig.direction]
        bucket["count"] += 1
        if sig.direction == "LONG" and future_price > entry_price:
            bucket["wins"] += 1
        elif sig.direction == "SHORT" and future_price < entry_price:
            bucket["wins"] += 1

    return results
