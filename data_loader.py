# data_loader.py
"""
data_loader.py
Funciones para obtener OHLCV (CCXT -> CoinGecko fallback) y normalizar.
Función principal: fetch_ohlcv(symbol, timeframe='1d', start=None, end=None, limit=1000)
- symbol puede ser: 'BINANCE:BTCUSDT', 'BTCUSDT', 'BTC/USDT', 'BTC-USD'
Devuelve pandas.DataFrame indexado por fecha con columnas: open, high, low, close, vol
"""

from datetime import datetime, timedelta
from typing import Optional
import time
import re
import os

try:
    import ccxt
except Exception:
    ccxt = None

try:
    import requests
except Exception:
    requests = None

import pandas as pd
import numpy as np


def _normalize_symbol_for_ccxt(symbol: str) -> str:
    s = symbol.upper().replace("BINANCE:", "").replace("-", "/")
    if "/" not in s:
        # try map USDT or USD
        if s.endswith("USDT"):
            base = s[:-4]
            s = f"{base}/USDT"
        elif s.endswith("USD"):
            base = s[:-3]
            s = f"{base}/USD"
    return s


def fetch_ohlcv_ccxt(symbol: str, timeframe: str = "1d", since: Optional[str] = None, limit: int = 1000):
    """Intento CCXT (Binance)"""
    if ccxt is None:
        return None
    try:
        exchange = ccxt.binance() if hasattr(ccxt, "binance") else None
        if exchange is None:
            return None
        s = _normalize_symbol_for_ccxt(symbol)
        since_ms = None
        if since:
            since_ms = int(pd.Timestamp(since).timestamp() * 1000)
        all_rows = []
        fetched = exchange.fetch_ohlcv(s, timeframe=timeframe, since=since_ms, limit=limit)
        if not fetched:
            return None
        df = pd.DataFrame(fetched, columns=["ts", "open", "high", "low", "close", "vol"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "vol"]]
    except Exception:
        return None


def fetch_ohlcv_coingecko(symbol: str, vs_currency: str = "usd", days: int = 3650):
    """Fallback CoinGecko (usa ids simples por mapeo)"""
    if requests is None:
        return None
    mapping = {"BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin", "SOL": "solana", "ADA": "cardano"}
    s = symbol.upper().replace("BINANCE:", "").replace("/", "").replace("-", "")
    cg_id = mapping.get(s, s.lower())
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
    params = {"vs_currency": vs_currency, "days": days}
    try:
        r = requests.get(url, params=params, timeout=30)
        data = r.json()
        prices = data.get("prices", [])
        if not prices:
            return None
        df = pd.DataFrame(prices, columns=["ts", "close"])
        df["date"] = pd.to_datetime(df["ts"], unit="ms")
        df = df.set_index("date")
        # coinGecko only gives prices; use close as open/high/low and vol=0
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["vol"] = 0
        return df[["open", "high", "low", "close", "vol"]]
    except Exception:
        return None


def fetch_ohlcv(symbol: str, timeframe: str = "1d", start: Optional[str] = None, end: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
    """
    Wrapper: intenta CCXT (Binance) y luego CoinGecko.
    start/end aceptan formatos 'YYYY-MM-DD' o None.
    Devuelve DataFrame con índice DatetimeIndex.
    """
    # try CCXT
    df = fetch_ohlcv_ccxt(symbol, timeframe=timeframe, since=start, limit=limit)
    if df is not None and len(df) > 0:
        # filter by start/end
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]
        return df
    # fallback
    df = fetch_ohlcv_coingecko(symbol, days=3650)
    if df is None:
        raise ValueError(f"No price data available for {symbol}")
    if start:
        df = df[df.index >= pd.to_datetime(start)]
    if end:
        df = df[df.index <= pd.to_datetime(end)]
    return df
