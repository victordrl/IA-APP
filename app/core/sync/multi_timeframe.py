"""
RF-7: Multi-timeframe synchronization.

Aligns 1h, 4h, and daily candle data into a single coherent row
so that every row of the tensor represents the same market instant.
"""

import pandas as pd
from loguru import logger


class MultiTimeframeSync:
    """Synchronize OHLCV DataFrames from different timeframes into one aligned table.

    Alignment rules (RF-7):
    - 4 candles of 1h  ≡ 1 candle of 4h.
    - 24 candles of 1h ≡ 1 candle of 1d.
    - Each row represents the *same* market instant.

    Strategy: use the 1h timeframe as the master clock and forward-fill
    the higher timeframes so each 1h row carries the most recent 4h/1d values.
    """

    # Mapping from standard config labels to canonical names
    _TF_CANONICAL = {"1h": "1h", "4h": "4h", "1d": "1d", "1D": "1d"}

    def __init__(self, base_timeframe: str = "1h"):
        self._base = base_timeframe

    # ── Public API ──────────────────────────────────

    def synchronize(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge multiple timeframes into a single aligned DataFrame.

        Args:
            data: ``{timeframe: DataFrame}`` — each DF must have a
                  ``timestamp`` column and feature columns.

        Returns:
            A single DataFrame indexed by the base (1h) timestamps with
            higher-timeframe columns forward-filled.
        """
        if self._base not in data:
            raise ValueError(f"Base timeframe '{self._base}' not found in data keys: {list(data.keys())}")

        base_df = data[self._base].copy()
        base_df = base_df.set_index("timestamp").sort_index()

        # Rename base columns with suffix
        base_df = base_df.add_suffix(f"_{self._base}")

        for tf, df in data.items():
            canonical = self._TF_CANONICAL.get(tf, tf)
            if canonical == self._base:
                continue

            higher = df.copy()
            higher = higher.set_index("timestamp").sort_index()
            higher = higher.add_suffix(f"_{canonical}")

            # Reindex to the base clock — forward-fill so each 1h row
            # carries the most recently known higher-TF value.
            higher = higher.reindex(base_df.index, method="ffill")
            base_df = base_df.join(higher, how="left")

        # Forward-fill any remaining NaNs from the join
        base_df = base_df.ffill()

        logger.info(
            "Synchronized {} timeframes → {} rows × {} cols",
            len(data),
            len(base_df),
            len(base_df.columns),
        )
        return base_df

    @staticmethod
    def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
        """Append global features required by RF-11.

        Adds:
        - ``precio_actual``: latest close from the 1h column.
        - ``tiempo_normalizado``: hour-of-day / 24, capturing intraday position.
        """
        result = df.copy()

        # Current price = most recent 1h close at that row
        close_1h_col = [c for c in result.columns if c.startswith("close") and "1h" in c]
        if close_1h_col:
            result["precio_actual"] = result[close_1h_col[0]]

        # Normalized time (hour / 24)
        result["tiempo_normalizado"] = result.index.hour / 24.0

        return result
