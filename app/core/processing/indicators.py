"""
RF-6: Technical indicator calculation using the ``ta`` library.

Organizado por grupos:
- VELOCIDAD: MON, ROC, RSI (6,14,24 + EMAs), Stochastic, Williams %R, CCI
- TENDENCIA: MACD, ADX, DI+/DI-, EMA 22/50/100, Ichimoku
- AMPLITUD: Bollinger Bands, Keltner Channels
- LIQUIDEZ: CMF, OBV, Elder Ray, EOM, VWAP
"""

import numpy as np
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
        ("SQZ_MOM", lambda df: IndicatorEngine._sqz_mom(df, length=20)),
        ("RSI_6", lambda df: ta.momentum.RSIIndicator(df["close"], window=6).rsi()),
        ("RSI_14", lambda df: ta.momentum.RSIIndicator(df["close"], window=14).rsi()),
        ("RSI_24", lambda df: ta.momentum.RSIIndicator(df["close"], window=24).rsi()),
        ("RSI_EMA_6", lambda df: ta.momentum.RSIIndicator(df["close"], window=6).rsi().ewm(span=6).mean()),
        ("RSI_EMA_14", lambda df: ta.momentum.RSIIndicator(df["close"], window=14).rsi().ewm(span=14).mean()),
        ("RSI_EMA_24", lambda df: ta.momentum.RSIIndicator(df["close"], window=24).rsi().ewm(span=24).mean()),
        ("STOCH_K", lambda df: ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3).stochrsi_k()),
        ("STOCH_D", lambda df: ta.momentum.StochRSIIndicator(df["close"], window=14, smooth1=3, smooth2=3).stochrsi_d()),
        ("WILLIAMS_R", lambda df: ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=22).williams_r()),
        ("CCI", lambda df: ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=50).cci()),
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
        ("EMA_7", lambda df: ta.trend.EMAIndicator(df["close"], window=7).ema_indicator()),
        ("EMA_22", lambda df: ta.trend.EMAIndicator(df["close"], window=22).ema_indicator()),
        ("EMA_99", lambda df: ta.trend.EMAIndicator(df["close"], window=99).ema_indicator()),
        ("ICHIMOKU_TENKAN", lambda df: IndicatorEngine._ichimoku_tenkan(df)),
        ("ICHIMOKU_KIJUN", lambda df: IndicatorEngine._ichimoku_kijun(df)),
        ("ICHIMOKU_SA", lambda df: IndicatorEngine._ichimoku_senkou_a(df)),
        ("ICHIMOKU_SB", lambda df: IndicatorEngine._ichimoku_senkou_b(df)),
        ("ICHIMOKU_CHIKOU", lambda df: IndicatorEngine._ichimoku_chikou(df)),
    ]

    # ===================== AMPLITUD =====================
    # ~7 indicadores por timeframe
    _AMPLITUD: list[tuple[str, callable]] = [
        ("BB_UPPER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_hband()),
        ("BB_MIDDLE", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_mavg()),
        ("BB_LOWER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_lband()),
        ("BB_WIDTH", lambda df: (ta.volatility.BollingerBands(df["close"]).bollinger_hband() - ta.volatility.BollingerBands(df["close"]).bollinger_lband()) / ta.volatility.BollingerBands(df["close"]).bollinger_mavg()),
        ("KELTNER_UPPER", lambda df: IndicatorEngine._keltner_upper(df)),
        ("KELTNER_MIDDLE", lambda df: ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()),
        ("KELTNER_LOWER", lambda df: IndicatorEngine._keltner_lower(df)),
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

    @staticmethod
    def _sqz_mom(df: pd.DataFrame, length: int = 20) -> pd.Series:
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

    @staticmethod
    def _stoch_rsi_d(df: pd.DataFrame, rsi_period: int = 14, stoch_period: int = 14) -> pd.Series:
        """Stochastic RSI %D = SMA of %K"""
        rsi = ta.momentum.RSIIndicator(df["close"], window=rsi_period).rsi()
        rsi_min = rsi.rolling(window=stoch_period).min()
        rsi_max = rsi.rolling(window=stoch_period).max()
        stoch_rsi = ((rsi - rsi_min) / (rsi_max - rsi_min)) * 100
        return stoch_rsi.rolling(window=3).mean()

    # ── Public API ───────────────────────────────────────────

    @classmethod
    def compute(cls, df: pd.DataFrame, timeframe_suffix: str = "") -> pd.DataFrame:
        """Compute all indicators (all groups) and append them to df."""
        result = df.copy()
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""

        # Skip heavy indicators if not enough data
        min_rows = 60
        has_enough_data = len(df) >= min_rows
        
        if not has_enough_data:
            # logger.warning(f"DataFrame has only {len(df)} rows, need {min_rows} for full indicators")
            pass

        total_indicators = 0

        for group_name, indicators in cls._GROUPS.items():
            for label, fn in indicators:
                col_name = f"{label}{suffix}"
                try:
                    # Skip indicators requiring more data than available
                    if label in ("EMA_50", "EMA_100", "ICHIMOKU_SB", "ADX", "DI_PLUS", "DI_MINUS", 
                               "KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"):
                        required_rows = {"EMA_50": 50, "EMA_100": 100, "ICHIMOKU_SB": 52,
                                        "ADX": 14, "DI_PLUS": 14, "DI_MINUS": 14,
                                        "KELTNER_UPPER": 20, "KELTNER_MIDDLE": 20, "KELTNER_LOWER": 20}
                        if len(df) < required_rows.get(label, 60):
                            result[col_name] = float("nan")
                            total_indicators += 1
                            continue
                    
                    result[col_name] = fn(result)
                except Exception as exc:
                    # logger.warning("Indicator {} failed: {}", col_name, exc)
                    result[col_name] = float("nan")
                total_indicators += 1

        # logger.debug(
        #     "Computed {} indicators for {} (suffix={})",
        #     total_indicators,
        #     group_name,
        #     suffix,
        # )
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
        result = {}
        for tf, df in data.items():
            df_with_indicators = cls.compute(df, timeframe_suffix=tf)
            numeric_cols = df_with_indicators.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            df_with_indicators[numeric_cols] = df_with_indicators[numeric_cols].round(2)
            result[tf] = df_with_indicators
        return result

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