"""
RF-6: Technical indicator calculation using the ``ta`` library.

Computes indicators per timeframe and appends them as independent features
(columns) to the OHLCV DataFrame.  Indicators are labelled anonymously
(Ind1, Ind2, …) to satisfy RF-11 column-standard requirements.
"""

import pandas as pd
import ta
from loguru import logger


class IndicatorEngine:
    """Calculate technical indicators on OHLCV DataFrames.

    Each indicator group is computed independently per timeframe and
    added as new columns.  The engine is extensible — add new indicators
    by registering them in ``_REGISTRY``.

    Satisfies RF-6 acceptance criteria:
    - Indicators calculated per timeframe.
    - Each indicator is an independent feature.
    - Multiple indicators per semantic group supported.
    """

    # Registry: (anonymous_label, callable(df) -> pd.Series)
    # Extend this list to add new indicators without rewriting the pipeline (RNF-3).
    _REGISTRY: list[tuple[str, callable]] = [
        ("Ind1", lambda df: ta.trend.SMAIndicator(df["close"], window=14).sma_indicator()),
        ("Ind2", lambda df: ta.trend.EMAIndicator(df["close"], window=14).ema_indicator()),
        ("Ind3", lambda df: ta.momentum.RSIIndicator(df["close"], window=14).rsi()),
        ("Ind4", lambda df: ta.trend.MACD(df["close"]).macd()),
        ("Ind5", lambda df: ta.trend.MACD(df["close"]).macd_signal()),
        ("Ind6", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_hband()),
        ("Ind7", lambda df: ta.volatility.BollingerBands(df["close"]).bollinger_lband()),
        ("Ind8", lambda df: ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()),
        ("Ind9", lambda df: ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()),
        ("Ind10", lambda df: ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"]).stoch()),
    ]

    # ── Public API ──────────────────────────────────

    @classmethod
    def compute(cls, df: pd.DataFrame, timeframe_suffix: str = "") -> pd.DataFrame:
        """Compute all registered indicators and append them to *df*.

        Args:
            df: OHLCV DataFrame (must contain ``open, high, low, close, volume``).
            timeframe_suffix: Optional suffix for column names (e.g. ``"_1h"``).

        Returns:
            DataFrame with indicator columns appended.
        """
        result = df.copy()
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""

        for label, fn in cls._REGISTRY:
            col_name = f"{label}{suffix}"
            try:
                result[col_name] = fn(result)
            except Exception as exc:
                logger.warning("Indicator {} failed: {}", col_name, exc)
                result[col_name] = float("nan")

        logger.debug(
            "Computed {} indicators{}",
            len(cls._REGISTRY),
            f" (suffix={suffix})" if suffix else "",
        )
        return result

    @classmethod
    def compute_multi_timeframe(
        cls, data: dict[str, pd.DataFrame]
    ) -> dict[str, pd.DataFrame]:
        """Compute indicators for each timeframe in the dict.

        Args:
            data: ``{timeframe: DataFrame}``.

        Returns:
            Same structure with indicator columns appended.
        """
        return {tf: cls.compute(df, timeframe_suffix=tf) for tf, df in data.items()}

    @classmethod
    def available_indicators(cls) -> list[str]:
        """List the anonymous labels of all registered indicators."""
        return [label for label, _ in cls._REGISTRY]
