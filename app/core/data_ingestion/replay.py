"""
RF-5: Market Replay — simulate real-time data from historical records.

Uses Darts TimeSeries to create a sliding-window replay that respects
temporal ordering and mimics the same refresh cadence as real-time mode.
"""

import asyncio
from typing import AsyncGenerator

import pandas as pd
from darts import TimeSeries
from loguru import logger

from app.config import settings


class MarketReplay:
    """Replay historical OHLCV data as if it were arriving in real time.

    Satisfies RF-5 acceptance criteria:
    - Respects original temporal order.
    - Simulates the configured refresh interval.
    - Can be activated/deactivated via configuration or endpoint.
    """

    def __init__(
        self,
        data: dict[str, pd.DataFrame],
        window_size: int | None = None,
        speed_multiplier: float | None = None,
        refresh_seconds: float | None = None,
    ):
        """
        Args:
            data: ``{timeframe: DataFrame}`` — historical data per timeframe.
            window_size: Number of rows per sliding window (default from config).
            speed_multiplier: Speed factor for replay (1.0 = real-time speed).
            refresh_seconds: Base interval between emissions in seconds.
        """
        self._window_size = window_size or settings.tensor_window_size
        self._speed = speed_multiplier or settings.replay_speed_multiplier
        self._refresh = refresh_seconds or settings.replay_refresh_seconds
        self._active = False

        # Convert DataFrames → Darts TimeSeries for structured windowing
        self._series: dict[str, TimeSeries] = {}
        for tf, df in data.items():
            ts_df = df.copy()
            ts_df = ts_df.set_index("timestamp").sort_index()
            # Darts requires a DatetimeIndex with a frequency
            ts_df.index = pd.DatetimeIndex(ts_df.index)
            self._series[tf] = TimeSeries.from_dataframe(
                ts_df,
                value_cols=["open", "high", "low", "close", "volume"],
                fill_missing_dates=True,
                freq=None,  # let Darts infer
            )

        # Determine max replay steps from the shortest series
        min_len = min(len(s) for s in self._series.values())
        self._max_steps = max(0, min_len - self._window_size)

        logger.info(
            "MarketReplay initialized — window={}, speed={}×, steps={}",
            self._window_size,
            self._speed,
            self._max_steps,
        )

    # ── Public API ──────────────────────────────────

    async def stream(self) -> AsyncGenerator[dict[str, pd.DataFrame], None]:
        """Async generator that yields one window per step.

        Yields:
            ``{timeframe: DataFrame}`` — sliding window for each timeframe.
        """
        self._active = True
        delay = self._refresh / self._speed

        for step in range(self._max_steps):
            if not self._active:
                logger.info("Replay stopped at step {}/{}", step, self._max_steps)
                return

            window: dict[str, pd.DataFrame] = {}
            for tf, ts in self._series.items():
                sliced = ts[step : step + self._window_size]
                window[tf] = sliced.pd_dataframe()

            logger.debug("Replay step {}/{}", step + 1, self._max_steps)
            yield window
            await asyncio.sleep(delay)

        self._active = False
        logger.success("Replay completed — {} steps emitted", self._max_steps)

    def stop(self) -> None:
        """Stop the replay mid-stream."""
        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if replay is currently running."""
        return self._active

    @property
    def total_steps(self) -> int:
        """Total number of replay steps available."""
        return self._max_steps
