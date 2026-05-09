"""
RF-3: Historical market data extraction via CCXT.

Fetches OHLCV candle data from Binance (or any CCXT-supported exchange)
with automatic pagination, chronological ordering, and reproducibility.

Includes fallback exchanges and retry logic for geo-restricted regions.
"""

import time
from datetime import datetime

import ccxt
import pandas as pd
from loguru import logger

from app.config import settings
from app.utils.data_utils import calculate_progress_vela

# Ordered fallback list — tried in sequence if the primary fails.
_EXCHANGE_FALLBACKS = ["binance", "binanceus", "bybit", "okx", "kraken"]


class HistoricalDataFetcher:
    """Fetch historical OHLCV data from a cryptocurrency exchange.

    Handles automatic pagination, rate-limiting, chronological ordering,
    and exchange fallback to satisfy RF-3 acceptance criteria.
    """

    OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

    def __init__(self, exchange_id: str | None = None):
        preferred = exchange_id or settings.exchange_id
        self._exchange = self._init_exchange(preferred)
        logger.info(
            "HistoricalDataFetcher ready — using {}",
            self._exchange.id,
        )

    # ── Exchange Initialization ─────────────────────

    @staticmethod
    def _init_exchange(preferred: str) -> ccxt.Exchange:
        """Try to initialize and load markets for the preferred exchange.

        Falls back through ``_EXCHANGE_FALLBACKS`` if the preferred one
        is unreachable (geo-block, DNS failure, etc.).
        """
        candidates = [preferred] + [
            ex for ex in _EXCHANGE_FALLBACKS if ex != preferred
        ]

        for eid in candidates:
            try:
                exchange_class = getattr(ccxt, eid, None)
                if exchange_class is None:
                    continue
                exchange = exchange_class(
                    {"enableRateLimit": settings.exchange_rate_limit}
                )
                exchange.load_markets()
                logger.debug("Exchange {} connected successfully", eid)
                return exchange
            except Exception as exc:
                logger.warning(
                    "Exchange {} unavailable ({}), trying next…", eid, exc
                )

        raise RuntimeError(
            f"No exchange reachable. Tried: {candidates}. "
            "Check your internet connection or configure a VPN."
        )

    # ── Public API ──────────────────────────────────

    def fetch(
        self,
        symbol: str | None = None,
        timeframe: str = "1h",
        since: str | datetime | None = None,
        until: str | datetime | None = None,
        limit_per_request: int = 1000,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch paginated OHLCV data and return a clean DataFrame.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            timeframe: Candle interval (``1h``, ``4h``, ``1d``).
            since: Start datetime or ISO-8601 string.
            until: End datetime or ISO-8601 string (defaults to now).
            limit_per_request: Max candles per API call (exchange limit).
            max_retries: Number of retry attempts per request on failure.

        Returns:
            ``pd.DataFrame`` with columns ``[timestamp, open, high, low, close, volume]``,
            sorted chronologically.
        """
        symbol = symbol or settings.default_symbol
        since_ms = self._to_ms(since) if since else None
        until_ms = self._to_ms(until) if until else None

        logger.info(
            "Fetching {} {} | since={} until={}",
            symbol,
            timeframe,
            since,
            until,
        )

        all_candles: list[list] = []
        while True:
            batch = self._fetch_with_retry(
                symbol, timeframe, since_ms, limit_per_request, max_retries
            )
            if not batch:
                break

            # Filter candles that exceed *until*
            if until_ms:
                batch = [c for c in batch if c[0] <= until_ms]
                if not batch:
                    break

            all_candles.extend(batch)

            # Advance cursor past the last candle
            since_ms = batch[-1][0] + 1

            if len(batch) < limit_per_request:
                break

        df = pd.DataFrame(all_candles, columns=self.OHLCV_COLUMNS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = (
            df.drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        df["progress_vela"] = calculate_progress_vela(df["timestamp"], timeframe, is_realtime=False)

        logger.success("{} candles fetched for {} {}", len(df), symbol, timeframe)
        return df

    # ── Helpers ──────────────────────────────────────

    def _fetch_with_retry(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int | None,
        limit: int,
        max_retries: int,
    ) -> list[list]:
        """Fetch a single batch with exponential backoff retry."""
        for attempt in range(1, max_retries + 1):
            try:
                return self._exchange.fetch_ohlcv(
                    symbol, timeframe, since=since_ms, limit=limit
                )
            except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as exc:
                wait = 2**attempt
                logger.warning(
                    "Retry {}/{} for {} {} — {} — waiting {}s",
                    attempt,
                    max_retries,
                    symbol,
                    timeframe,
                    exc,
                    wait,
                )
                time.sleep(wait)
            except ccxt.ExchangeError as exc:
                logger.error("Exchange error (non-retryable): {}", exc)
                raise
        logger.error("All {} retries exhausted", max_retries)
        return []

    def _to_ms(self, dt: str | datetime) -> int:
        """Convert a datetime or ISO string to Unix milliseconds."""
        if isinstance(dt, str):
            return self._exchange.parse8601(dt)
        return int(dt.timestamp() * 1000)

    def fetch_multi_timeframe(
        self,
        symbol: str | None = None,
        timeframes: list[str] | None = None,
        since: str | datetime | None = None,
        until: str | datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV for multiple timeframes.

        Returns:
            Dictionary ``{timeframe: DataFrame}``.
        """
        symbol = symbol or settings.default_symbol
        timeframes = timeframes or settings.timeframes_list

        results: dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            results[tf] = self.fetch(
                symbol=symbol, timeframe=tf, since=since, until=until
            )
        return results
