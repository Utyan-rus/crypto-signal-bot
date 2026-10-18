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


def adx(df: pd.DataFrame, period: int = 14):
    """Average Directional Index + the two directional indicators it's built from."""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    smoothed_tr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / smoothed_tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / smoothed_tr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def donchian_channel(df: pd.DataFrame, period: int = 20):
    """Highest high / lowest low over the PRIOR `period` bars (shifted, so the
    current bar can't look at itself — avoids lookahead bias on a breakout)."""
    upper = df["high"].shift(1).rolling(period).max()
    lower = df["low"].shift(1).rolling(period).min()
    return upper, lower


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


def analyze_guarded(df: pd.DataFrame, order_book_imbalance: float = 0.0,
                    funding_rate: float | None = None) -> Signal:
    """Conservative trend-following filter; neutral is preferred to weak setups.

    Uses closed-candle OHLCV only (ExchangeClient drops the currently forming
    candle). The confidence value is a rule-strength score, not a probability.
    """
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns) or len(df) < 220:
        price = float(df["close"].iloc[-1]) if len(df) else 0.0
        return Signal("NEUTRAL", 0, "HIGH", ["Нужно минимум 220 закрытых свечей"], price, price, price, 0.0)

    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    e20, e50, e200 = ema(close, 20), ema(close, 50), ema(close, 200)
    rsi_value = float(rsi(close).iloc[-1])
    atr_value = float(atr(df, 14).iloc[-1])
    adx_values, plus_di, minus_di = adx(df, 14)
    adx_value = float(adx_values.iloc[-1])
    macd_line, macd_signal, macd_hist = macd(close)
    hist = float(macd_hist.iloc[-1])
    volume_avg = float(df["volume"].rolling(20).mean().iloc[-1])
    volume_now = float(df["volume"].iloc[-1])

    vals = [last, float(e20.iloc[-1]), float(e50.iloc[-1]), float(e200.iloc[-1]),
            rsi_value, atr_value, adx_value, hist, volume_avg]
    if not all(np.isfinite(x) for x in vals) or last <= 0 or atr_value <= 0:
        return Signal("NEUTRAL", 0, "HIGH", ["Индикаторы ещё не сформировались"], last, last, last, 0.0)

    reasons = [
        f"EMA20/50/200: {e20.iloc[-1]:.6g} / {e50.iloc[-1]:.6g} / {e200.iloc[-1]:.6g}",
        f"ADX {adx_value:.1f}; RSI {rsi_value:.1f}",
    ]
    long_trend = last > e200.iloc[-1] and e20.iloc[-1] > e50.iloc[-1] and e50.iloc[-1] > e200.iloc[-1]
    short_trend = last < e200.iloc[-1] and e20.iloc[-1] < e50.iloc[-1] and e50.iloc[-1] < e200.iloc[-1]
    volume_ok = volume_avg > 0 and volume_now >= volume_avg * 0.8
    long_momentum = 52 <= rsi_value <= 68 and hist > 0 and float(plus_di.iloc[-1]) > float(minus_di.iloc[-1])
    short_momentum = 32 <= rsi_value <= 48 and hist < 0 and float(minus_di.iloc[-1]) > float(plus_di.iloc[-1])

    direction = "NEUTRAL"
    score = 0
    if adx_value >= 18 and volume_ok and long_trend and long_momentum:
        direction, score = "LONG", 4
        reasons.append("Тренд, моментум и объём согласованы")
    elif adx_value >= 18 and volume_ok and short_trend and short_momentum:
        direction, score = "SHORT", 4
        reasons.append("Тренд, моментум и объём согласованы")
    else:
        reasons.append("Нет полного подтверждения — вход пропущен")
        if adx_value < 18:
            reasons.append("ADX ниже 18: недостаточно выраженный тренд")
        if not volume_ok:
            reasons.append("Объём ниже фильтра")
        if not (long_trend or short_trend):
            reasons.append("EMA20/50/200 не подтверждают единое направление")
        if order_book_imbalance > 0.2:
            reasons.append("Стакан смещён к покупателям, но сам по себе не создаёт сигнал")
        elif order_book_imbalance < -0.2:
            reasons.append("Стакан смещён к продавцам, но сам по себе не создаёт сигнал")

    atr_pct = atr_value / last * 100
    risk_level = "LOW" if atr_pct < 1 else ("MEDIUM" if atr_pct < 3 else "HIGH")
    stop_mult, target_mult = 1.5, 3.0
    if direction == "LONG":
        stop_loss, take_profit = last - stop_mult * atr_value, last + target_mult * atr_value
    elif direction == "SHORT":
        stop_loss, take_profit = last + stop_mult * atr_value, last - target_mult * atr_value
    else:
        stop_loss, take_profit = last - stop_mult * atr_value, last + stop_mult * atr_value
    confidence = 75 if direction != "NEUTRAL" else 25
    return Signal(direction, confidence, risk_level, reasons, last, stop_loss, take_profit, atr_value)


STRATEGIES = {"guarded": analyze_guarded}
STRATEGY_NAMES = {"guarded": "Фильтр-тренд"}


def analyze(df: pd.DataFrame, order_book_imbalance: float = 0.0,
            funding_rate: float | None = None, strategy: str = "guarded") -> Signal:
    func = STRATEGIES.get(strategy, analyze_guarded)
    return func(df, order_book_imbalance=order_book_imbalance, funding_rate=funding_rate)


def backtest(df: pd.DataFrame, lookback: int = 60, horizon: int = 24,
             strategy: str = "guarded") -> dict:
    """
    Retrospective check: walks a rolling window through the given candles,
    recomputes this same rule-based signal at each point using only data
    available up to that point, then checks whether price moved in the
    signal's direction `horizon` candles later.

    This is purely a statistic over historical data with these exact
    rules — not a prediction, and no guarantee future performance will
    look similar.
    """
    func = STRATEGIES.get(strategy, analyze_guarded)
    results = {"LONG": {"count": 0, "wins": 0}, "SHORT": {"count": 0, "wins": 0}}
    close = df["close"]
    n = len(df)

    for i in range(lookback, n - horizon):
        window = df.iloc[: i + 1]
        sig = func(window)
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
