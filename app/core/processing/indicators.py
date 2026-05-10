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

from app.core.processing.normalization import DataNormalizer

from app.utils.indicator_helpers import (
    ichimoku_tenkan,
    ichimoku_kijun,
    ichimoku_senkou_a,
    ichimoku_senkou_b,
    ichimoku_chikou,
    keltner_upper,
    keltner_middle,
    keltner_lower,
    sqz_mom,
    stoch_rsi_d,
    elder_ray_bull,
    elder_ray_bear,
    vwap,
    bollinger_width,
)


class IndicatorEngine:
    """Calculate technical indicators on OHLCV DataFrames organized by groups.

    Cada indicador se calcula por timeframe y se añade como columna independiente.
    """

    # ===================== VELOCIDAD =====================
    # ~12 indicadores por timeframe
    _VELOCIDAD: list[tuple[str, callable]] = [
        ("MON", lambda df: ta.momentum.ROCIndicator(df["close"], window=12).roc()),
        ("ROC", lambda df: ta.momentum.ROCIndicator(df["close"], window=14).roc()),
        ("SQZ_MOM", lambda df: sqz_mom(df, length=20)),
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
        ("ICHIMOKU_TENKAN", lambda df: ichimoku_tenkan(df)),
        ("ICHIMOKU_KIJUN", lambda df: ichimoku_kijun(df)),
        ("ICHIMOKU_SA", lambda df: ichimoku_senkou_a(df)),
        ("ICHIMOKU_SB", lambda df: ichimoku_senkou_b(df)),
        ("ICHIMOKU_CHIKOU", lambda df: ichimoku_chikou(df)),
    ]

    # ===================== AMPLITUD =====================
    # ~7 indicadores por timeframe
    _AMPLITUD: list[tuple[str, callable]] = [
        ("BB_UPPER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_hband()),
        ("BB_MIDDLE", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_mavg()),
        ("BB_LOWER", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_lband()),
        ("BB_WIDTH", lambda df: bollinger_width(df)),
        ("KELTNER_UPPER", lambda df: keltner_upper(df)),
        ("KELTNER_MIDDLE", lambda df: keltner_middle(df)),
        ("KELTNER_LOWER", lambda df: keltner_lower(df)),
    ]

    # ===================== LIQUIDEZ =====================
    # ~6 indicadores por timeframe
    _LIQUIDEZ: list[tuple[str, callable]] = [
        ("CMF", lambda df: ta.volume.ChaikinMoneyFlowIndicator(df["high"], df["low"], df["close"], df["volume"], window=20).chaikin_money_flow()),
        ("OBV", lambda df: ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()),
        ("ELDER_BULL", lambda df: elder_ray_bull(df)),
        ("ELDER_BEAR", lambda df: elder_ray_bear(df)),
        ("EOM", lambda df: ta.volume.EaseOfMovementIndicator(df["high"], df["low"], df["volume"]).ease_of_movement()),
        ("VWAP", lambda df: vwap(df)),
    ]

    # Mapping de grupos
    _GROUPS = {
        "velocidad": _VELOCIDAD,
        "tendencia": _TENDENCIA,
        "amplitud": _AMPLITUD,
        "liquidez": _LIQUIDEZ,
    }

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
    def compute_multi_timeframe(cls, data: dict[str, pd.DataFrame], normalized: bool = False) -> dict[str, pd.DataFrame]:
        """Compute indicators for each timeframe in the dict."""
        result = {}
        for tf, df in data.items():
            df_with_indicators = cls.compute(df, timeframe_suffix=tf)
            
            if normalized:
                df_with_indicators = DataNormalizer.normalize_indicators(df_with_indicators, suffix=f"_{tf}")
                
            numeric_cols = df_with_indicators.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
            df_with_indicators[numeric_cols] = df_with_indicators[numeric_cols].round(4)
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