"""
Helper functions for complex technical indicators.
These are standalone functions that can be reused across the application.
"""

import pandas as pd
import ta


def ichimoku_tenkan(df: pd.DataFrame) -> pd.Series:
    """Tenkan-sen (Conversion Line) = (Max high + Min low) / 2 for 9 periods"""
    high_9 = df["high"].rolling(window=9).max()
    low_9 = df["low"].rolling(window=9).min()
    return (high_9 + low_9) / 2


def ichimoku_kijun(df: pd.DataFrame) -> pd.Series:
    """Kijun-sen (Base Line) = (Max high + Min low) / 2 for 26 periods"""
    high_26 = df["high"].rolling(window=26).max()
    low_26 = df["low"].rolling(window=26).min()
    return (high_26 + low_26) / 2


def ichimoku_senkou_a(df: pd.DataFrame) -> pd.Series:
    """Senkou A (Leading Span A) = (Tenkan + Kijun) / 2"""
    tenkan = (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2
    kijun = (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2
    return (tenkan + kijun) / 2


def ichimoku_senkou_b(df: pd.DataFrame) -> pd.Series:
    """Senkou B (Leading Span B) = (Max high + Min low) / 2 for 52 periods"""
    high_52 = df["high"].rolling(window=52).max()
    low_52 = df["low"].rolling(window=52).min()
    return (high_52 + low_52) / 2


def ichimoku_chikou(df: pd.DataFrame) -> pd.Series:
    """Chikou Span (Lagging Span) = Close shifted -26 periods"""
    return df["close"].shift(-26)


def keltner_upper(df: pd.DataFrame, ema_window: int = 20, atr_multiplier: float = 2.0) -> pd.Series:
    """Keltner Channel Upper = EMA + (ATR * multiplier)"""
    ema = ta.trend.EMAIndicator(df["close"], window=ema_window).ema_indicator()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ema_window).average_true_range()
    return ema + (atr * atr_multiplier)


def keltner_middle(df: pd.DataFrame, ema_window: int = 20) -> pd.Series:
    """Keltner Channel Middle = EMA"""
    return ta.trend.EMAIndicator(df["close"], window=ema_window).ema_indicator()


def keltner_lower(df: pd.DataFrame, ema_window: int = 20, atr_multiplier: float = 2.0) -> pd.Series:
    """Keltner Channel Lower = EMA - (ATR * multiplier)"""
    ema = ta.trend.EMAIndicator(df["close"], window=ema_window).ema_indicator()
    atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=ema_window).average_true_range()
    return ema - (atr * atr_multiplier)


def sqz_mom(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Squeeze Momentum indicator."""
    src = df["close"]

    tr1 = df["high"] - df["low"]
    tr2 = df["high"] - df["close"].shift()
    tr3 = df["low"] - df["close"].shift()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    kc_ma = src.rolling(length).mean()
    kc_range = tr.rolling(length).mean()

    highest_hl = df["high"].rolling(length).max()
    lowest_hl = df["low"].rolling(length).min()
    sma_close = src.rolling(length).mean()
    midline = (highest_hl + lowest_hl + sma_close) / 3

    return (src - midline).rolling(5).mean()


def stoch_rsi_d(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
    """Stochastic RSI %D = SMA of %K"""
    rsi = ta.momentum.RSIIndicator(df["close"], window=rsi_period).rsi()
    rsi_min = rsi.rolling(window=stoch_period).min()
    rsi_max = rsi.rolling(window=stoch_period).max()
    stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
    return stoch_rsi.rolling(window=3).mean()


def elder_ray_bull(df: pd.DataFrame, ema_period: int = 13) -> pd.Series:
    """Elder Ray Bull Power = Close - EMA"""
    ema = ta.trend.EMAIndicator(df["close"], window=ema_period).ema_indicator()
    return df["close"] - ema


def elder_ray_bear(df: pd.DataFrame, ema_period: int = 13) -> pd.Series:
    """Elder Ray Bear Power = Close - EMA - (High - EMA)"""
    ema = ta.trend.EMAIndicator(df["close"], window=ema_period).ema_indicator()
    return df["close"] - ema - (df["high"].rolling(ema_period).max() - ema)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price"""
    return (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()


def bollinger_width(df: pd.DataFrame) -> pd.Series:
    """Bollinger Band Width = (BB Upper - BB Lower) / BB Middle"""
    bb = ta.volatility.BollingerBands(df["close"])
    return (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()