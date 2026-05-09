"""
RF-4: Real-time market data extraction.

Provides periodic polling of current OHLCV candles (including
unclosed candles with provisional close = current price).
Auto-refreshes every 15 minutes as specified in RF-4.
"""

import asyncio
from datetime import datetime, timezone

import ccxt
import pandas as pd
from loguru import logger

from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.utils.data_utils import calculate_progress_vela


class RealTimeDataFetcher:
    """Periodically fetch the latest OHLCV candles for all configured timeframes.

    Satisfies RF-4 acceptance criteria:
    - Auto-refresh every 15 minutes.
    - Retrieves unclosed candles (1h, 4h, 1d) with current price as provisional close.
    """

    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
    REFRESH_INTERVAL_SECONDS = 15 * 60  # 15 minutes

    def __init__(self, exchange_id: str | None = None):
        # Reuse the same fallback logic from HistoricalDataFetcher
        self._exchange = HistoricalDataFetcher._init_exchange(
            exchange_id or settings.exchange_id
        )
        self._latest: dict[str, pd.DataFrame] = {}
        self._running = False
        logger.info("RealTimeDataFetcher ready — using {}", self._exchange.id)

    # ── Public API ──────────────────────────────────

    def fetch_latest(
        self,
        symbol: str | None = None,
        timeframe: str = "1h",
        limit: int = 30,
    ) -> pd.DataFrame:
        """Fetch the most recent *limit* candles (includes unclosed current candle).

        Args:
            symbol: Trading pair.
            timeframe: Candle interval.
            limit: Number of candles to retrieve.

        Returns:
            DataFrame with the latest candles.
        """
        symbol = symbol or settings.default_symbol
        raw = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=self.OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["progress_vela"] = calculate_progress_vela(df["timestamp"], timeframe, is_realtime=True)
        return df

    def fetch_latest_multi_timeframe(
        self,
        symbol: str | None = None,
        timeframes: list[str] | None = None,
        limit: int = 30,
    ) -> dict[str, pd.DataFrame]:
        """Fetch latest candles across all configured timeframes."""
        symbol = symbol or settings.default_symbol
        timeframes = timeframes or settings.timeframes_list

        result: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            result[tf] = self.fetch_latest(symbol=symbol, timeframe=tf, limit=limit)
        return result

    # ── Background Polling Loop ─────────────────────

    async def start_polling(
        self,
        symbol: str | None = None,
        timeframes: list[str] | None = None,
    ) -> None:
        """Start an async loop that refreshes data every 15 min."""
        symbol = symbol or settings.default_symbol
        timeframes = timeframes or settings.timeframes_list
        self._running = True

        logger.info("Real-time polling started — every {}s", self.REFRESH_INTERVAL_SECONDS)
        while self._running:
            try:
                self._latest = self.fetch_latest_multi_timeframe(
                    symbol=symbol, timeframes=timeframes
                )
                logger.debug(
                    "Refreshed real-time data at {}",
                    datetime.now(timezone.utc).isoformat(),
                )
            except Exception as exc:
                logger.error("Real-time fetch error: {}", exc)
            await asyncio.sleep(self.REFRESH_INTERVAL_SECONDS)

    def stop_polling(self) -> None:
        """Stop the background polling loop."""
        self._running = False
        logger.info("Real-time polling stopped")

    @property
    def latest_data(self) -> dict[str, pd.DataFrame]:
        """Return the most recently cached data."""
        return self._latest
