"""
Utility functions for data processing.
Shared helpers for OHLCV data manipulation.
"""

from datetime import datetime, timezone

import pandas as pd


def calculate_progress_vela(
    timestamps: pd.Series,
    timeframe: str,
    is_realtime: bool = False,
) -> pd.Series:
    """Calculate candle progress (0-1) based on current time.

    For historical data (is_realtime=False), all candles are closed (progress=1.0).
    For real-time data (is_realtime=True), the last candle may be unclosed.

    Args:
        timestamps: Series of candle timestamps.
        timeframe: Timeframe string (1h, 4h, 1d).
        is_realtime: If True, calculate actual progress for current candle.

    Returns:
        Series with progress values (0-1).
    """
    if not is_realtime:
        return pd.Series(1.0, index=timestamps.index)

    now = datetime.now(timezone.utc)
    result = pd.Series(1.0, index=timestamps.index)

    if timeframe == "1h":
        minutes_in_hour = 60
        current_minute = now.hour * 60 + now.minute
        current_progress = (current_minute % minutes_in_hour) / minutes_in_hour
        result.iloc[-1] = current_progress
    elif timeframe == "4h":
        hours_in_4h = 4
        hour_in_period = now.hour % hours_in_4h
        current_progress = (hour_in_period * 60 + now.minute) / (hours_in_4h * 60)
        result.iloc[-1] = current_progress
    elif timeframe == "1d":
        current_progress = (now.hour * 60 + now.minute) / (24 * 60)
        result.iloc[-1] = current_progress

    return result