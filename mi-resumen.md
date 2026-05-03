This file is a merged representation of the entire codebase, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
.env.example
.gitignore
app/__init__.py
app/api/__init__.py
app/api/routes/__init__.py
app/api/routes/data.py
app/api/routes/replay.py
app/api/routes/tensor.py
app/api/schemas.py
app/config.py
app/core/__init__.py
app/core/data_ingestion/__init__.py
app/core/data_ingestion/historical.py
app/core/data_ingestion/realtime.py
app/core/data_ingestion/replay.py
app/core/processing/__init__.py
app/core/processing/indicators.py
app/core/processing/normalizer.py
app/core/sync/__init__.py
app/core/sync/multi_timeframe.py
app/core/tensor/__init__.py
app/core/tensor/builder.py
app/main.py
app/utils/__init__.py
app/utils/logger.py
README.md
requirements.txt
requisitos ia 1.docx
run.py
tests/__init__.py
tests/test_api_health.py
tests/test_normalizer.py
tests/test_tensor_builder.py
```

# Files

## File: .env.example
````
# ── Server ──────────────────────────────────────────
APP_HOST=0.0.0.0
APP_PORT=8000
APP_DEBUG=true

# ── Exchange (CCXT) ─────────────────────────────────
# No API keys needed for public market data
EXCHANGE_ID=binance
EXCHANGE_RATE_LIMIT=true

# ── Data Defaults ───────────────────────────────────
DEFAULT_SYMBOL=BTC/USDT
DEFAULT_TIMEFRAMES=1h,4h,1d

# ── Tensor ──────────────────────────────────────────
TENSOR_WINDOW_SIZE=30

# ── Replay ──────────────────────────────────────────
REPLAY_SPEED_MULTIPLIER=1.0
REPLAY_REFRESH_SECONDS=5
````

## File: .gitignore
````
# ── Python ──────────────────────────────────────────
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
*.egg

# ── Virtual Environment ────────────────────────────
.venv/
venv/
env/

# ── Environment Variables ──────────────────────────
.env

# ── IDE ─────────────────────────────────────────────
.vscode/
.idea/
*.swp
*.swo

# ── Data Cache ──────────────────────────────────────
data/cache/
*.parquet
*.h5

# ── Logs ────────────────────────────────────────────
logs/
*.log

# ── OS ──────────────────────────────────────────────
.DS_Store
Thumbs.db
````

## File: app/__init__.py
````python
"""IA-APP: AI Trading Data Pipeline — Phase 1."""

__version__ = "0.1.0"
````

## File: app/api/__init__.py
````python
"""API package: routes and schemas."""
````

## File: app/api/routes/__init__.py
````python
"""API route modules."""
````

## File: app/api/routes/data.py
````python
"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import HistoricalRequest
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.realtime import RealTimeDataFetcher

router = APIRouter(prefix="/data", tags=["Data Ingestion"])

_historical = HistoricalDataFetcher()
_realtime = RealTimeDataFetcher()


@router.post("/historical")
def fetch_historical(req: HistoricalRequest):
    """Fetch historical OHLCV data for the given parameters.

    Returns a dict ``{timeframe: [{timestamp, open, high, low, close, volume}, …]}``.
    """
    try:
        result = _historical.fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )
        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/realtime")
def fetch_realtime(symbol: str = "BTC/USDT", limit: int = 30):
    """Fetch the latest *limit* candles across all default timeframes."""
    try:
        result = _realtime.fetch_latest_multi_timeframe(symbol=symbol, limit=limit)
        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
````

## File: app/api/routes/replay.py
````python
"""
Replay endpoints — start, stop, and status of market replay sessions.
Covers RF-5 (Market Replay) exposure via API.
"""

import asyncio

from fastapi import APIRouter, HTTPException

from app.api.schemas import PipelineStatus, ReplayRequest
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.replay import MarketReplay

router = APIRouter(prefix="/replay", tags=["Market Replay"])

_historical = HistoricalDataFetcher()
_current_replay: MarketReplay | None = None
_replay_step: int = 0


@router.post("/start")
async def start_replay(req: ReplayRequest):
    """Start a new market replay session from historical data.

    Fetches the requested range, creates a ``MarketReplay`` instance,
    and begins streaming windows in the background.
    """
    global _current_replay, _replay_step

    if _current_replay and _current_replay.is_active:
        raise HTTPException(status_code=409, detail="Replay already running — stop it first")

    try:
        raw = _historical.fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )
        _current_replay = MarketReplay(
            data=raw,
            speed_multiplier=req.speed_multiplier,
        )
        _replay_step = 0

        # Run replay in the background
        asyncio.create_task(_run_replay())

        return {
            "status": "started",
            "total_steps": _current_replay.total_steps,
            "speed": req.speed_multiplier,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop")
def stop_replay():
    """Stop the currently running replay."""
    global _current_replay
    if _current_replay:
        _current_replay.stop()
        return {"status": "stopped", "step": _replay_step}
    return {"status": "no_replay_running"}


@router.get("/status", response_model=PipelineStatus)
def replay_status():
    """Return the current replay status."""
    active = _current_replay.is_active if _current_replay else False
    return PipelineStatus(
        mode="replay" if active else "idle",
        replay_active=active,
        replay_step=_replay_step if active else None,
        replay_total_steps=_current_replay.total_steps if _current_replay else None,
    )


# ── Background Task ────────────────────────────────

async def _run_replay():
    """Consume the replay stream in the background."""
    global _replay_step
    if not _current_replay:
        return
    async for _window in _current_replay.stream():
        _replay_step += 1
````

## File: app/api/routes/tensor.py
````python
"""
Tensor endpoints — full pipeline execution and metadata inspection.
Covers RF-8, RF-10, RF-11 and RNF-4 (observability).
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import HistoricalRequest, TensorMeta
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.processing.indicators import IndicatorEngine
from app.core.processing.normalizer import Normalizer
from app.core.sync.multi_timeframe import MultiTimeframeSync
from app.core.tensor.builder import TensorBuilder

router = APIRouter(prefix="/tensor", tags=["Tensor Pipeline"])

_historical = HistoricalDataFetcher()
_sync = MultiTimeframeSync()
_normalizer = Normalizer()
_builder = TensorBuilder()


@router.post("/build", response_model=TensorMeta)
def build_tensor(req: HistoricalRequest):
    """Execute the full pipeline: fetch → indicators → sync → normalize → tensor.

    Returns tensor metadata (shape, features). The tensor itself is kept
    server-side for downstream consumers.
    """
    try:
        # 1. Fetch historical data
        raw = _historical.fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )

        # 2. Compute indicators per timeframe
        with_indicators = IndicatorEngine.compute_multi_timeframe(raw)

        # 3. Synchronize into a single DataFrame
        synced = _sync.synchronize(with_indicators)
        synced = MultiTimeframeSync.add_global_features(synced)

        # 4. Normalize
        normalized = _normalizer.fit_transform(synced)

        # 5. Build tensor & return metadata
        meta = _builder.describe(normalized)
        _builder.build(normalized)  # validates and logs

        return TensorMeta(**meta)

    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/describe", response_model=TensorMeta)
def describe_tensor(req: HistoricalRequest):
    """Dry-run: returns what the tensor shape would be without building it."""
    try:
        raw = _historical.fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )
        with_indicators = IndicatorEngine.compute_multi_timeframe(raw)
        synced = _sync.synchronize(with_indicators)
        synced = MultiTimeframeSync.add_global_features(synced)
        return TensorMeta(**_builder.describe(synced))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/indicators")
def list_indicators():
    """List all available indicator labels."""
    return {"indicators": IndicatorEngine.available_indicators()}
````

## File: app/api/schemas.py
````python
"""
Pydantic schemas for API request / response models.
"""

from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────


class HistoricalRequest(BaseModel):
    """Parameters for fetching historical OHLCV data."""
    symbol: str = Field("BTC/USDT", description="Trading pair")
    timeframes: list[str] = Field(["1h", "4h", "1d"], description="Candle intervals")
    since: str | None = Field(None, description="Start date ISO-8601 (e.g. 2024-01-01T00:00:00Z)")
    until: str | None = Field(None, description="End date ISO-8601")


class ReplayRequest(BaseModel):
    """Parameters for starting a market replay session."""
    symbol: str = Field("BTC/USDT")
    timeframes: list[str] = Field(["1h", "4h", "1d"])
    since: str | None = Field(None)
    until: str | None = Field(None)
    speed_multiplier: float = Field(1.0, ge=0.1, le=100.0)


# ── Responses ───────────────────────────────────────


class TensorMeta(BaseModel):
    """Metadata describing a generated tensor."""
    window_size: int
    num_features: int
    num_rows: int
    num_windows: int
    tensor_shape: list[int]
    feature_columns: list[str]


class PipelineStatus(BaseModel):
    """Current status of the data pipeline."""
    mode: str  # "idle" | "historical" | "realtime" | "replay"
    replay_active: bool
    replay_step: int | None = None
    replay_total_steps: int | None = None


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    version: str
````

## File: app/config.py
````python
"""
Centralized application configuration.
Loads settings from .env file and environment variables.
Covers RF-1 (environment isolation) requirements.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ──────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True

    # ── Exchange ────────────────────────────────────
    exchange_id: str = "binance"
    exchange_rate_limit: bool = True

    # ── Data ────────────────────────────────────────
    default_symbol: str = "BTC/USDT"
    default_timeframes: str = "1h,4h,1d"

    # ── Tensor ──────────────────────────────────────
    tensor_window_size: int = 30

    # ── Replay ──────────────────────────────────────
    replay_speed_multiplier: float = 1.0
    replay_refresh_seconds: float = 5.0

    @property
    def timeframes_list(self) -> list[str]:
        """Return timeframes as a clean list."""
        return [tf.strip() for tf in self.default_timeframes.split(",")]


settings = Settings()
````

## File: app/core/__init__.py
````python
"""Core business logic package."""
````

## File: app/core/data_ingestion/__init__.py
````python
"""Data ingestion sub-package: historical, real-time & replay."""
````

## File: app/core/data_ingestion/historical.py
````python
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
````

## File: app/core/data_ingestion/realtime.py
````python
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
````

## File: app/core/data_ingestion/replay.py
````python
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
````

## File: app/core/processing/__init__.py
````python
"""Data processing sub-package: indicators & normalization."""
````

## File: app/core/processing/indicators.py
````python
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
````

## File: app/core/processing/normalizer.py
````python
"""
RF-9: Data normalization strategies.

Applies the appropriate normalization method depending on feature type:
- Bounded indicators (RSI, Stochastic) → Min-Max [0, 1]
- Prices and returns                   → Z-Score
- Volumes                              → Log scaling + standardization
"""

import numpy as np
import pandas as pd
from loguru import logger


class Normalizer:
    """Apply column-aware normalization to a feature DataFrame.

    Stores fitted parameters (mean, std, min, max) so the same transform
    can be applied identically on future data (RNF-2 reproducibility).
    """

    # Columns containing these substrings use specific strategies.
    _BOUNDED_KEYWORDS = {"Ind3", "Ind10"}  # RSI, Stochastic → Min-Max
    _VOLUME_KEYWORDS = {"volume", "Ind9"}  # Volume, OBV → Log-scale
    # Everything else → Z-Score

    def __init__(self):
        self._params: dict[str, dict] = {}  # {col: {method, ...fitted_values}}

    # ── Public API ──────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and return the normalized copy.

        Args:
            df: DataFrame of numeric features (excludes ``timestamp``).

        Returns:
            Normalized DataFrame with the same shape and columns.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            method = self._detect_method(col)
            result[col] = self._apply(result[col], col, method, fit=True)

        logger.debug("Fit & transformed {} numeric columns", len(numeric_cols))
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform *df* using previously fitted parameters.

        Args:
            df: DataFrame with the same columns used in ``fit_transform``.

        Returns:
            Normalized DataFrame.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self._params:
                logger.warning("Column {} was not fitted — skipping", col)
                continue
            method = self._params[col]["method"]
            result[col] = self._apply(result[col], col, method, fit=False)

        return result

    # ── Internal ────────────────────────────────────

    def _detect_method(self, col: str) -> str:
        """Heuristic to select normalization method based on column name."""
        for kw in self._BOUNDED_KEYWORDS:
            if kw in col:
                return "minmax"
        for kw in self._VOLUME_KEYWORDS:
            if kw.lower() in col.lower():
                return "log"
        return "zscore"

    def _apply(self, series: pd.Series, col: str, method: str, fit: bool) -> pd.Series:
        if method == "minmax":
            return self._minmax(series, col, fit)
        elif method == "log":
            return self._log_scale(series, col, fit)
        else:
            return self._zscore(series, col, fit)

    def _minmax(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        if fit:
            mn, mx = s.min(), s.max()
            self._params[col] = {"method": "minmax", "min": mn, "max": mx}
        else:
            mn = self._params[col]["min"]
            mx = self._params[col]["max"]
        rng = mx - mn
        if rng == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / rng

    def _zscore(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        if fit:
            mean, std = s.mean(), s.std()
            self._params[col] = {"method": "zscore", "mean": mean, "std": std}
        else:
            mean = self._params[col]["mean"]
            std = self._params[col]["std"]
        if std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mean) / std

    def _log_scale(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        logged = np.log1p(s.clip(lower=0))
        if fit:
            mean, std = logged.mean(), logged.std()
            self._params[col] = {"method": "log", "mean": mean, "std": std}
        else:
            mean = self._params[col]["mean"]
            std = self._params[col]["std"]
        if std == 0:
            return pd.Series(0.0, index=s.index)
        return (logged - mean) / std

    @property
    def fitted_params(self) -> dict[str, dict]:
        """Return a copy of the fitted normalization parameters."""
        return dict(self._params)
````

## File: app/core/sync/__init__.py
````python
"""Synchronization sub-package: multi-timeframe alignment."""
````

## File: app/core/sync/multi_timeframe.py
````python
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
````

## File: app/core/tensor/__init__.py
````python
"""Tensor construction sub-package."""
````

## File: app/core/tensor/builder.py
````python
"""
RF-8 / RF-10 / RF-11: Tensor construction with sliding windows.

Builds a PyTorch tensor of shape ``(num_windows, window_size, num_features)``
from a fully synchronized and normalized DataFrame.  Each window is a
30 × N slice ready for neural network consumption.
"""

import numpy as np
import pandas as pd
import torch
from loguru import logger

from app.config import settings


class TensorBuilder:
    """Construct sliding-window tensors from a synchronized feature DataFrame.

    Satisfies:
    - RF-8:  Tensor shape ``30 × N`` with sliding windows.
    - RF-10: No out-of-range values, no temporal misalignment.
    - RF-11: Standard column layout per timeframe block + global vars.
    """

    def __init__(self, window_size: int | None = None):
        self._window_size = window_size or settings.tensor_window_size

    # ── Public API ──────────────────────────────────

    def build(self, df: pd.DataFrame) -> torch.Tensor:
        """Create a 3-D tensor from the synchronized DataFrame.

        Args:
            df: Fully synchronized, normalized DataFrame
                (index = timestamps, columns = features).

        Returns:
            ``torch.Tensor`` of shape ``(num_windows, window_size, num_features)``.

        Raises:
            ValueError: If the DataFrame has fewer rows than the window size.
        """
        self._validate(df)

        values = df.select_dtypes(include=[np.number]).values  # (T, N)
        num_rows, num_features = values.shape
        num_windows = num_rows - self._window_size + 1

        # Sliding-window view — zero-copy where possible
        windows = np.lib.stride_tricks.sliding_window_view(values, self._window_size, axis=0)
        # windows shape: (num_windows, num_features, window_size) → transpose
        windows = windows.transpose(0, 2, 1)  # → (num_windows, window_size, num_features)

        tensor = torch.tensor(windows, dtype=torch.float32)

        logger.success(
            "Tensor built — shape {} (windows={}, steps={}, features={})",
            list(tensor.shape),
            num_windows,
            self._window_size,
            num_features,
        )
        return tensor

    def build_single_window(self, df: pd.DataFrame) -> torch.Tensor:
        """Build a single ``(1, window_size, N)`` tensor from the last rows.

        Useful for real-time inference where only the latest window matters.
        """
        tail = df.tail(self._window_size)
        self._validate(tail)
        values = tail.select_dtypes(include=[np.number]).values
        tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        return tensor

    # ── Validation ──────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """RF-10 structural validation."""
        if len(df) < self._window_size:
            raise ValueError(
                f"DataFrame has {len(df)} rows but window_size={self._window_size}"
            )

        numeric = df.select_dtypes(include=[np.number])

        # Check for NaN / Inf
        if numeric.isnull().any().any():
            nan_cols = numeric.columns[numeric.isnull().any()].tolist()
            logger.warning("NaN detected in columns: {} — filling with 0", nan_cols)
            df[nan_cols] = df[nan_cols].fillna(0)

        if np.isinf(numeric.values).any():
            raise ValueError("Infinite values detected in feature DataFrame")

    # ── Metadata ────────────────────────────────────

    def describe(self, df: pd.DataFrame) -> dict:
        """Return a summary of what the tensor would look like.

        Useful for the observability endpoint (RNF-4).
        """
        numeric = df.select_dtypes(include=[np.number])
        num_rows = len(df)
        num_features = len(numeric.columns)
        num_windows = max(0, num_rows - self._window_size + 1)
        return {
            "window_size": self._window_size,
            "num_features": num_features,
            "num_rows": num_rows,
            "num_windows": num_windows,
            "tensor_shape": [num_windows, self._window_size, num_features],
            "feature_columns": numeric.columns.tolist(),
        }
````

## File: app/main.py
````python
"""
IA-APP — FastAPI Application Entry Point.

Initializes the server, registers all routes, and configures
middleware.  Covers RF-2 (Backend Initialization).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import data, replay, tensor
from app.api.schemas import HealthResponse
from app.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup & shutdown hooks."""
    setup_logging()
    yield


app = FastAPI(
    title="IA-APP — AI Trading Data Pipeline",
    description=(
        "Fase 1: Infraestructura, Servidor y Pipeline de Datos. "
        "Obtiene, sincroniza, normaliza y entrega tensores de mercado "
        "multi-temporalidad (1h, 4h, 1d) listos para redes neuronales."
    ),
    version=__version__,
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────
app.include_router(data.router)
app.include_router(tensor.router)
app.include_router(replay.router)


# ── Health ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Endpoint de salud del servidor."""
    return HealthResponse(status="ok", version=__version__)
````

## File: app/utils/__init__.py
````python
"""Utility sub-package."""
````

## File: app/utils/logger.py
````python
"""
Centralized logging configuration using Loguru.

Provides structured, colored console output and optional file rotation.
Satisfies RNF-4 (Observability).
"""

import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """Configure Loguru for the application."""
    # Remove default handler
    logger.remove()

    # Console handler — colorized, with context
    log_level = "DEBUG" if settings.app_debug else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — rotated daily, kept 7 days
    logger.add(
        "logs/ia_app_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    )

    logger.info("Logging initialized — level={}", log_level)
````

## File: requirements.txt
````
# ── Server ──────────────────────────────────────────
fastapi>=0.115.0
uvicorn[standard]>=0.34.0
pydantic>=2.11.0
pydantic-settings>=2.9.0
python-dotenv>=1.0.0

# ── Data Ingestion ──────────────────────────────────
ccxt>=4.5.0

# ── Data Processing & Analysis ──────────────────────
pandas>=2.2.0
numpy>=2.1.0
ta>=0.11.0

# ── Time-Series & Replay ───────────────────────────
darts>=0.32.0

# ── Tensor / Deep Learning Backend ──────────────────
torch>=2.6.0

# ── Observability & Logging ─────────────────────────
loguru>=0.7.0

# ── Testing ─────────────────────────────────────────
pytest>=8.0.0
httpx>=0.28.0
````

## File: run.py
````python
"""
Convenience script to launch the server.
Usage: python run.py
"""

import uvicorn

from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
    )
````

## File: tests/__init__.py
````python
"""Tests package."""
````

## File: tests/test_api_health.py
````python
"""
Integration test for the FastAPI health endpoint.
"""

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "version" in body
````

## File: tests/test_normalizer.py
````python
"""
Unit tests for the Normalizer.
Validates RF-9 normalization strategies.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.processing.normalizer import Normalizer


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic feature DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "close_1h": np.random.uniform(90, 110, n),    # → Z-Score
            "Ind3_1h": np.random.uniform(0, 100, n),      # → Min-Max (RSI)
            "volume_1h": np.random.uniform(1000, 5000, n), # → Log-scale
            "Ind10_4h": np.random.uniform(0, 100, n),     # → Min-Max (Stoch)
        }
    )


class TestNormalizer:
    def test_fit_transform_shape(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        assert result.shape == feature_df.shape

    def test_minmax_range(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        # RSI column → should be in [0, 1]
        assert result["Ind3_1h"].min() >= -1e-9
        assert result["Ind3_1h"].max() <= 1.0 + 1e-9

    def test_zscore_mean_std(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        # Z-scored column → mean ≈ 0, std ≈ 1
        assert abs(result["close_1h"].mean()) < 0.1
        assert abs(result["close_1h"].std() - 1.0) < 0.1

    def test_transform_uses_fitted_params(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        norm.fit_transform(feature_df)
        # Transform same data again → should give same result
        result2 = norm.transform(feature_df)
        assert result2.shape == feature_df.shape

    def test_fitted_params_stored(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        norm.fit_transform(feature_df)
        params = norm.fitted_params
        assert "close_1h" in params
        assert params["close_1h"]["method"] == "zscore"
        assert params["Ind3_1h"]["method"] == "minmax"
````

## File: tests/test_tensor_builder.py
````python
"""
Unit tests for the Tensor Builder.
Validates RF-8, RF-10 structural integrity.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.tensor.builder import TensorBuilder


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a synthetic OHLCV-like DataFrame with 50 rows."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "open_1h": np.random.uniform(90, 110, n),
            "close_1h": np.random.uniform(90, 110, n),
            "high_1h": np.random.uniform(100, 120, n),
            "low_1h": np.random.uniform(80, 100, n),
            "volume_1h": np.random.uniform(1000, 5000, n),
            "Ind1_1h": np.random.uniform(90, 110, n),
            "Ind2_1h": np.random.uniform(90, 110, n),
            "Ind3_1h": np.random.uniform(0, 100, n),
            "precio_actual": np.random.uniform(90, 110, n),
            "tiempo_normalizado": np.linspace(0, 1, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


class TestTensorBuilder:
    def test_build_shape(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        tensor = builder.build(sample_df)
        # 50 rows, window 30 → 21 windows, 10 features
        assert tensor.shape == (21, 30, 10)

    def test_build_single_window(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        tensor = builder.build_single_window(sample_df)
        assert tensor.shape == (1, 30, 10)

    def test_too_few_rows_raises(self):
        small_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        builder = TensorBuilder(window_size=10)
        with pytest.raises(ValueError, match="window_size"):
            builder.build(small_df)

    def test_describe_metadata(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        meta = builder.describe(sample_df)
        assert meta["window_size"] == 30
        assert meta["num_features"] == 10
        assert meta["tensor_shape"] == [21, 30, 10]
````

## File: README.md
````markdown
# IA-APP: Estrategia de Inversión Cuantitativa

Este proyecto implementa una **Inteligencia Artificial** especializada en mercados financieros, diseñada para generar señales de compra y venta con objetivos de rentabilidad superiores al 10%. El sistema integra técnicas avanzadas de Deep Learning con metodologías cuantitativas de vanguardia.

## 🚀 Características Principales

- **Deep Reinforcement Learning (DRL):** El agente aprende directamente de los resultados financieros, optimizando la estrategia en entornos simulados antes de operar en real.
- **Arquitectura Multimodal (xLSTM):** Integra información de precios, volumen, indicadores técnicos y sentimiento de mercado para tomar decisiones holísticas.
- **Gestión de Riesgo Avanzada:** Implementa el **Triple Barrier Method** y **Differentiable Sharpe Ratio** para asegurar ratios riesgo/beneficio favorables (objetivo 1:4) y evitar el _overfitting_.
- **Estructura Modular (RF-1 a RF-11):** El código sigue una arquitectura estricta que separa la recolección de datos, cálculo de indicadores, construcción de tensores y lógica de inferencia.

## 🏗️ Arquitectura Técnica

El sistema está organizado en los siguientes módulos principales:

1.  **Data Fetcher:** Obtiene datos históricos de múltiples temporalidades (1h, 4h, 1d) desde fuentes fiables (Binance, CCXT).
2.  **Indicator Engine:** Calcula indicadores técnicos (SMA, EMA, MACD, Bollinger Bands, etc.) utilizando la librería `ta`.
3.  **Tensor Builder:** Estructura los datos en tensores 3D para el entrenamiento de redes neuronales, preservando el orden temporal.
4.  **Model Architecture:** Implementación de redes basadas en **Transformers** y **LSTM** adaptadas a datos financieros (xLSTM, PatchTST).
5.  **Risk Manager:** Define las condiciones de salida y profit (10%) y stop-loss (2%) para guiar el entrenamiento.

## 🛠️ Instalación y Ejecución

### Requisitos

- Python 3.9+
- CUDA (para entrenamiento GPU)

### Instalación de Dependencias

```bash
git clone https://github.com/tuusuario/IA-APP.git
cd IA-APP
pip install -r requirements.txt
```

### Ejecución

- **Entrenamiento:**
  ```bash
  python train.py --symbol BTC/USDT --epochs 50 --window-size 30
  ```
- **Inferencia (Modo Replay):**
  ```bash
  python main.py --mode replay --symbol BTC/USDT --speed 2.0
  ```

## 📊 Estrategia de Mercado

- **Símbolo:** BTC/USDT
- **Objetivo de Profit:** +10% por operación.
- **Stop Loss:** -2%.
- **Horizonte:** Corto / Medio plazo.
- **Temporalidades:** 1h, 4h, 1d (sincronizadas).
````
