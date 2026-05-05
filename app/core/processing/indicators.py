"""
RF-6: Technical indicator calculation using the ``ta`` library.

Organizado por grupos:
- VELOCIDAD: MON, ROC, RSI (6,14,24 + EMAs), Stochastic, Williams %R, CCI
- TENDENCIA: MACD, ADX, DI+/DI-, EMA 22/50/100, Ichimoku
- AMPLITUD: Bollinger Bands, Keltner Channels
- LIQUIDEZ: CMF, OBV, Elder Ray, EOM, VWAP
"""

import pandas as pd
import ta
from loguru import logger


class IndicatorEngine:
    """Calculate technical indicators on OHLCV DataFrames organized by groups.

    Cada indicador se calcula por timeframe y se añade como columna independiente.
    """

    # ===================== VELOCIDAD =====================
    # ~12 indicadores por timeframe
    _VELOCIDAD: list[tuple[str, callable]] = [
        ("MON", lambda df: ta.momentum.ROCIndicator(df["close"], window=12).roc()),
        ("ROC", lambda df: ta.momentum.ROCIndicator(df["close"], window=14).roc()),
        ("RSI_6", lambda df: ta.momentum.RSIIndicator(df["close"], window=6).rsi()),
        ("RSI_14", lambda df: ta.momentum.RSIIndicator(df["close"], window=14).rsi()),
        ("RSI_24", lambda df: ta.momentum.RSIIndicator(df["close"], window=24).rsi()),
        ("RSI_EMA_6", lambda df: ta.momentum.RSIIndicator(df["close"], window=6).rsi().ewm(span=6).mean()),
        ("RSI_EMA_14", lambda df: ta.momentum.RSIIndicator(df["close"], window=14).rsi().ewm(span=14).mean()),
        ("RSI_EMA_24", lambda df: ta.momentum.RSIIndicator(df["close"], window=24).rsi().ewm(span=24).mean()),
        ("STOCH_K", lambda df: ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()),
        ("STOCH_D", lambda df: ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch_signal()),
        ("WILLIAMS_R", lambda df: ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"]).williams_r()),
        ("CCI", lambda df: ta.trend.CCIIndicator(df["high"], df["low"], df["close"]).cci()),
    ]

    # ===================== TENDENCIA =====================
    # ~17 indicadores por timeframe
    _TENDENCIA: list[tuple[str, callable]] = [
        ("MACD_LINE", lambda df: ta.trend.MACD(df["close"]).macd()),
        ("MACD_SIGNAL", lambda df: ta.trend.MACD(df["close"]).macd_signal()),
        ("MACD_HIST", lambda df: ta.trend.MACD(df["close"]).macd_diff()),
        ("ADX", lambda df: ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()),
        ("DI_PLUS", lambda df: ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx_pos()),
        ("DI_MINUS", lambda df: ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx_neg()),
        ("EMA_22", lambda df: ta.trend.EMAIndicator(df["close"], window=22).ema_indicator()),
        ("EMA_50", lambda df: ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()),
        ("EMA_100", lambda df: ta.trend.EMAIndicator(df["close"], window=100).ema_indicator()),
        ("ICHIMOKU_TENKAN", lambda df: _ichimoku_tenkan(df)),
        ("ICHIMOKU_KIJUN", lambda df: _ichimoku_kijun(df)),
        ("ICHIMOKU_SA", lambda df: _ichimoku_senkou_a(df)),
        ("ICHIMOKU_SB", lambda df: _ichimoku_senkou_b(df)),
        ("ICHIMOKU_CHIKOU", lambda df: _ichimoku_chikou(df)),
    ]

    # ===================== AMPLITUD =====================
    # ~7 indicadores por timeframe
    _AMPLITUD: list[tuple[str, callable]] = [
        ("BB_UPPER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_hband()),
        ("BB_MIDDLE", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_mavg()),
        ("BB_LOWER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_lband()),
        ("BB_WIDTH", lambda df: (ta.volatility.BollingerBands(df["close"]).bollinger_hband() - ta.volatility.BollingerBands(df["close"]).bollinger_lband()) / ta.volatility.BollingerBands(df["close"]).bollinger_mavg()),
        ("KELTNER_UPPER", lambda df: _keltner_upper(df)),
        ("KELTNER_MIDDLE", lambda df: ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()),
        ("KELTNER_LOWER", lambda df: _keltner_lower(df)),
    ]

    # ===================== LIQUIDEZ =====================
    # ~6 indicadores por timeframe
    _LIQUIDEZ: list[tuple[str, callable]] = [
        ("CMF", lambda df: ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()),
        ("OBV", lambda df: ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()),
        ("ELDER_BULL", lambda df: df["close"] - ta.trend.EMAIndicator(df["close"], window=13).ema_indicator()),
        ("ELDER_BEAR", lambda df: df["close"] - ta.trend.EMAIndicator(df["close"], window=13).ema_indicator() - (df["high"].rolling(13).max() - ta.trend.EMAIndicator(df["close"], window=13).ema_indicator())),
        ("EOM", lambda df: ta.volume.EaseOfMovementIndicator(df["high"], df["low"], df["volume"]).ease_of_movement()),
        ("VWAP", lambda df: (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()),
    ]

    # Mapping de grupos
    _GROUPS = {
        "velocidad": _VELOCIDAD,
        "tendencia": _TENDENCIA,
        "amplitud": _AMPLITUD,
        "liquidez": _LIQUIDEZ,
    }

    # ── Helpers para indicadores complejos ─────────────────────

    @staticmethod
    def _ichimoku_tenkan(df: pd.DataFrame) -> pd.Series:
        """Tenkan-sen (Conversion Line) = (Max high + Min low) / 2 for 9 periods"""
        high_9 = df["high"].rolling(window=9).max()
        low_9 = df["low"].rolling(window=9).min()
        return (high_9 + low_9) / 2

    @staticmethod
    def _ichimoku_kijun(df: pd.DataFrame) -> pd.Series:
        """Kijun-sen (Base Line) = (Max high + Min low) / 2 for 26 periods"""
        high_26 = df["high"].rolling(window=26).max()
        low_26 = df["low"].rolling(window=26).min()
        return (high_26 + low_26) / 2

    @staticmethod
    def _ichimoku_senkou_a(df: pd.DataFrame) -> pd.Series:
        """Senkou A (Leading Span A) = (Tenkan + Kijun) / 2"""
        tenkan = (df["high"].rolling(window=9).max() + df["low"].rolling(window=9).min()) / 2
        kijun = (df["high"].rolling(window=26).max() + df["low"].rolling(window=26).min()) / 2
        return (tenkan + kijun) / 2

    @staticmethod
    def _ichimoku_senkou_b(df: pd.DataFrame) -> pd.Series:
        """Senkou B (Leading Span B) = (Max high + Min low) / 2 for 52 periods"""
        high_52 = df["high"].rolling(window=52).max()
        low_52 = df["low"].rolling(window=52).min()
        return (high_52 + low_52) / 2

    @staticmethod
    def _ichimoku_chikou(df: pd.DataFrame) -> pd.Series:
        """Chikou Span (Lagging Span) = Close shifted -26 periods"""
        return df["close"].shift(-26)

    @staticmethod
    def _keltner_upper(df: pd.DataFrame) -> pd.Series:
        """Keltner Channel Upper = EMA + (ATR * 2)"""
        ema = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
        return ema + (atr * 2)

    @staticmethod
    def _keltner_lower(df: pd.DataFrame) -> pd.Series:
        """Keltner Channel Lower = EMA - (ATR * 2)"""
        ema = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
        atr = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=20).average_true_range()
        return ema - (atr * 2)

    # ── Public API ───────────────────────────────────────────

    @classmethod
    def compute(cls, df: pd.DataFrame, timeframe_suffix: str = "") -> pd.DataFrame:
        """Compute all indicators (all groups) and append them to df."""
        result = df.copy()
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""

        total_indicators = 0

        for group_name, indicators in cls._GROUPS.items():
            for label, fn in indicators:
                col_name = f"{label}{suffix}"
                try:
                    result[col_name] = fn(result)
                except Exception as exc:
                    logger.warning("Indicator {} failed: {}", col_name, exc)
                    result[col_name] = float("nan")
                total_indicators += 1

        logger.debug(
            "Computed {} indicators for {} (suffix={})",
            total_indicators,
            group_name,
            suffix,
        )
        return result

    @classmethod
    def compute_by_group(cls, df: pd.DataFrame, group: str, timeframe_suffix: str = "") -> pd.DataFrame:
        """Compute indicators for a specific group only."""
        result = df.copy()
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""

        if group not in cls._GROUPS:
            raise ValueError(f"Grupo desconocido: {group}. Available: {list(cls._GROUPS.keys())}")

        for label, fn in cls._GROUPS[group]:
            col_name = f"{label}{suffix}"
            try:
                result[col_name] = fn(result)
            except Exception as exc:
                logger.warning("Indicator {} failed: {}", col_name, exc)
                result[col_name] = float("nan")

        return result

    @classmethod
    def compute_multi_timeframe(cls, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Compute indicators for each timeframe in the dict."""
        return {tf: cls.compute(df, timeframe_suffix=tf) for tf, df in data.items()}

    @classmethod
    def get_group_columns(cls, group: str, timeframe_suffix: str = "") -> list[str]:
        """Get list of column names for a specific group."""
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
        return [f"{label}{suffix}" for label, _ in cls._GROUPS.get(group, [])]

    @classmethod
    def get_all_groups_columns(cls, timeframe_suffix: str = "") -> dict[str, list[str]]:
        """Get columns for all groups."""
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
        return {
            group: [f"{label}{suffix}" for label, _ in indicators]
            for group, indicators in cls._GROUPS.items()
        }

    @classmethod
    def available_indicators(cls) -> dict[str, int]:
        """List available indicators by group."""
        return {group: len(indicators) for group, indicators in cls._GROUPS.items()}