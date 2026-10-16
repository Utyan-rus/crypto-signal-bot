"""
exchange.py
Fetches OHLCV market data from a crypto exchange via ccxt.
Read-only — only queries public market data, never places orders
and never touches a wallet or API keys with trading permissions.
"""

import ccxt
import pandas as pd


class ExchangeClient:
    def __init__(self, exchange_id: str = "binance"):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 200) -> pd.DataFrame:
        """
        Fetch OHLCV candles and return as a DataFrame.
        symbol example: 'BTC/USDT'
        """
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def fetch_funding_rate(self, symbol: str):
        """
        Fetch the current funding rate for a perpetual futures symbol, if supported.
        Returns None if the exchange/market doesn't expose it.
        """
        try:
            futures_symbol = symbol if ":" in symbol else f"{symbol}:USDT"
            fr = self.exchange.fetch_funding_rate(futures_symbol)
            return fr.get("fundingRate")
        except Exception:
            return None

    def fetch_order_book_imbalance(self, symbol: str, depth: int = 50) -> float:
        """
        Simple bid/ask volume imbalance in the order book.
        Positive = more buy-side depth, negative = more sell-side depth.
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=depth)
            bid_vol = sum(b[1] for b in ob["bids"])
            ask_vol = sum(a[1] for a in ob["asks"])
            total = bid_vol + ask_vol
            if total == 0:
                return 0.0
            return (bid_vol - ask_vol) / total
        except Exception:
            return 0.0
