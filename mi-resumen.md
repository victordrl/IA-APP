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
app/api/routes/data.py
app/api/routes/replay.py
app/api/routes/tensor.py
app/api/schemas.py
app/config.py
app/core/data_ingestion/historical.py
app/core/data_ingestion/realtime.py
app/core/data_ingestion/replay_backtrader.py
app/core/processing/indicators.py
app/core/processing/normalizer.py
app/core/sync/multi_timeframe.py
app/core/tensor/builder.py
app/main.py
app/utils/logger.py
README.md
repomix-output.xml
requirements.txt
requisitos ia 1.docx
run_visualizer.bat
run.py
tests/test_api_health.py
tests/test_normalizer.py
tests/test_replay_visualizer.py
tests/test_tensor_builder.py
```

# Files

## File: .env.example
`````
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
`````

## File: .gitignore
`````
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
`````

## File: app/api/routes/data.py
`````python
"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.

Incluye opción de sincronización multi-timeframe:
- sync_type: timeframe | merged | semantic
- sync_version: ohlcv | indicators
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import HistoricalRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.realtime import RealTimeDataFetcher
from app.core.processing.indicators import IndicatorEngine
from app.core.sync.multi_timeframe import MultiTimeframeSync

router = APIRouter(prefix="/data", tags=["Data Ingestion"])

_historical = HistoricalDataFetcher()
_realtime = RealTimeDataFetcher()
_sync = MultiTimeframeSync()


def _apply_sync(data: dict, sync_type: str, sync_version: str):
    """Aplicar sync a los datos."""
    with_indicators = IndicatorEngine.compute_multi_timeframe(data)
    synced = _sync.synchronize(with_indicators, sync_type=sync_type, sync_version=sync_version)
    return synced


@router.post("/historical")
def fetch_historical(req: HistoricalRequest):
    """Fetch historical OHLCV data for the given parameters.

    Returns a dict with timeframe data, optionally synchronized.

    Params:
        - include_sync: If True, returns synchronized data (single DataFrame)
        - sync_type: timeframe | merged | semantic (default from config)
        - sync_version: ohlcv | indicators (default from config)
    """
    try:
        result = _historical.fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )

        sync_type = req.sync_type or settings.sync_type
        sync_version = req.sync_version or settings.sync_version

        if req.include_sync:
            synced = _apply_sync(result, sync_type, sync_version)
            return synced.to_dict(orient="records")

        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/realtime")
def fetch_realtime(
    symbol: str = "BTC/USDT",
    limit: int = 30,
    sync_type: str = None,
    sync_version: str = None,
    include_sync: bool = False,
):
    """Fetch the latest *limit* candles across all default timeframes.

    Params:
        - sync_type: timeframe | merged | semantic (default from config)
        - sync_version: ohlcv | indicators (default from config)
        - include_sync: If True, returns synchronized data
    """
    try:
        result = _realtime.fetch_latest_multi_timeframe(symbol=symbol, limit=limit)

        if include_sync:
            use_sync_type = sync_type or settings.sync_type
            use_sync_version = sync_version or settings.sync_version
            synced = _apply_sync(result, use_sync_type, use_sync_version)
            return synced.to_dict(orient="records")

        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
`````

## File: app/api/routes/tensor.py
`````python
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
`````

## File: repomix-output.xml
`````xml
This file is a merged representation of the entire codebase, combined into a single document by Repomix.

<file_summary>
This section contains a summary of this file.

<purpose>
This file contains a packed representation of the entire repository's contents.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.
</purpose>

<file_format>
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  - File path as an attribute
  - Full contents of the file
</file_format>

<usage_guidelines>
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.
</usage_guidelines>

<notes>
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)
</notes>

</file_summary>

<directory_structure>
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
mi-resumen.md
README.md
requirements.txt
requisitos ia 1.docx
run.py
tests/__init__.py
tests/test_api_health.py
tests/test_normalizer.py
tests/test_tensor_builder.py
</directory_structure>

<files>
This section contains the contents of the repository's files.

<file path=".env.example">
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
</file>

<file path=".gitignore">
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
</file>

<file path="app/__init__.py">
"""IA-APP: AI Trading Data Pipeline — Phase 1."""

__version__ = "0.1.0"
</file>

<file path="app/api/__init__.py">
"""API package: routes and schemas."""
</file>

<file path="app/api/routes/__init__.py">
"""API route modules."""
</file>

<file path="app/api/routes/data.py">
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
</file>

<file path="app/api/routes/replay.py">
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
</file>

<file path="app/api/routes/tensor.py">
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
</file>

<file path="app/api/schemas.py">
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
</file>

<file path="app/config.py">
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
</file>

<file path="app/core/__init__.py">
"""Core business logic package."""
</file>

<file path="app/core/data_ingestion/__init__.py">
"""Data ingestion sub-package: historical, real-time & replay."""
</file>

<file path="app/core/data_ingestion/historical.py">
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
</file>

<file path="app/core/data_ingestion/realtime.py">
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
</file>

<file path="app/core/data_ingestion/replay.py">
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
</file>

<file path="app/core/processing/__init__.py">
"""Data processing sub-package: indicators & normalization."""
</file>

<file path="app/core/processing/indicators.py">
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
</file>

<file path="app/core/processing/normalizer.py">
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
</file>

<file path="app/core/sync/__init__.py">
"""Synchronization sub-package: multi-timeframe alignment."""
</file>

<file path="app/core/sync/multi_timeframe.py">
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
</file>

<file path="app/core/tensor/__init__.py">
"""Tensor construction sub-package."""
</file>

<file path="app/core/tensor/builder.py">
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
</file>

<file path="app/main.py">
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
</file>

<file path="app/utils/__init__.py">
"""Utility sub-package."""
</file>

<file path="app/utils/logger.py">
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
</file>

<file path="mi-resumen.md">
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
</file>

<file path="requirements.txt">
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
</file>

<file path="run.py">
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
</file>

<file path="tests/__init__.py">
"""Tests package."""
</file>

<file path="tests/test_api_health.py">
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
</file>

<file path="tests/test_normalizer.py">
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
</file>

<file path="tests/test_tensor_builder.py">
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
</file>

<file path="README.md">
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
</file>

</files>
`````

## File: run_visualizer.bat
`````batch
@echo off
cd /d C:\Users\PILON\Desktop\Proyectos\IA-APP
python tests\test_replay_visualizer.py --mock
pause
`````

## File: run.py
`````python
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
`````

## File: tests/test_api_health.py
`````python
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
`````

## File: tests/test_normalizer.py
`````python
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
`````

## File: tests/test_replay_visualizer.py
`````python
"""
Replay Visualizer - Muestra el replay en vivo con matplotlib.

Usage:
    python tests/test_replay_visualizer.py --mock    # Testing sin API
    python tests/test_replay_visualizer.py       # Con API real

Abre ventana con 3 subplots (1h, 4h, 1d) mostrando:
- Velas OHLCV
- Indicadores (RSI, MACD, EMA, Bollinger)  
- Progress de la vela
"""

import time
import threading
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Any, Optional

API_BASE = "http://localhost:8000"

PAYLOAD = {
    "symbol": "BTC/USDT",
    "since": "2026-02-01T00:00:00",
    "until": "2026-05-05T09:00:00",
    "speed_multiplier": 10,
}

_api_data: Dict[str, Any] = {
    "active": False,
    "step": 0,
    "total": 0,
    "candles_1h": None,
    "candles_4h": None,
    "candles_1d": None,
    "progress": {"1h": 1.0, "4h": 0.5, "1d": 0.25},
}
_stop_event = threading.Event()


def start_replay() -> Dict:
    """Inicia el replay."""
    print(f"Iniciando replay: {PAYLOAD}")
    resp = requests.post(f"{API_BASE}/replay/start", json=PAYLOAD, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"Replay iniciado: total_steps={data.get('total_steps')}, window={data.get('window_size')}")
    _api_data["total"] = data.get("total_steps", 0)
    _api_data["active"] = True
    return data


def stop_replay() -> None:
    """Detiene el replay."""
    try:
        requests.post(f"{API_BASE}/replay/stop", timeout=10)
    except:
        pass
    _api_data["active"] = False


def poll_loop() -> None:
    """Hace polling de los datos del replay."""
    while not _stop_event.is_set():
        try:
            status_resp = requests.get(f"{API_BASE}/replay/status", timeout=10)
            status = status_resp.json()

            if not status.get("replay_active"):
                print("Replay terminado")
                break

            step = status.get("replay_step", 0)
            _api_data["step"] = step

            print(f"Step {step}/{_api_data['total']}")

            if step % 10 == 0:
                time.sleep(0.5)
            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"Error polling: {e}")
            time.sleep(0.5)

    _api_data["active"] = False


def create_sample_data(tf: str, n: int = 60) -> pd.DataFrame:
    """Crea datos de ejemplo para testing."""
    np.random.seed(42)

    start_date = datetime(2026, 2, 1)
    freq_map = {"1h": "h", "4h": "4h", "1d": "D"}
    dates = pd.date_range(start=start_date, periods=n, freq=freq_map.get(tf, "h"))

    base_price = 67000
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = close * (1 + np.random.uniform(-0.005, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.01)))
        volume = np.random.randint(500, 2000)

        data.append({
            "timestamp": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


def plot_candlestick(ax: plt.Axes, df: pd.DataFrame, title: str, progress: float = 1.0) -> None:
    """Grafica velas japonesas."""
    ax.clear()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=12)
        return

    n = min(len(df), 60)
    df = df.tail(n).reset_index(drop=True)

    for i in range(n):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"

        ax.plot([i, i], [l, h], color=color, linewidth=0.8)

        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.1:
            body_height = max(h - l) * 0.3

        ax.add_patch(plt.Rectangle(
            (i - 0.35, body_bottom),
            0.7,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        ))

    ax.set_xlim(-1, n)
    y_min, y_max = df["low"].min() * 0.995, df["high"].max() * 1.005
    ax.set_ylim(y_min, y_max)

    step = _api_data.get("step", 0)
    total = _api_data.get("total", 0)
    ax.set_title(f"{title} | Step: {step}/{total} | Progress: {progress:.0%}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price", fontsize=10)
    ax.grid(True, alpha=0.3)

    ylabel = ax.set_yticks([])
    for label in ax.get_yticklabels():
        label.set_fontsize(8)


def plot_indicators(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Grafica RSI."""
    ax.clear()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=10)
        return

    n = min(len(df), 60)
    df = df.tail(n).reset_index(drop=True)

    if "close" in df.columns:
        close = df["close"].values
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        window = 14
        if len(close) >= window:
            avg_gain = np.convolve(gain, np.ones(window)/window, mode="valid")
            avg_loss = np.convolve(loss, np.ones(window)/window, mode="valid")
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            ax.plot(rsi, color="#26a69a", linewidth=1.5, label="RSI(14)")
            ax.axhline(y=70, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(y=30, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.fill_between(range(len(rsi)), 30, 70, alpha=0.1, color="gray")

    ax.set_xlim(-1, n)
    ax.set_ylim(0, 100)
    ax.set_title(f"{title}", fontsize=10)
    ax.set_ylabel("RSI", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def run_mock() -> None:
    """Ejecuta en modo mock sin API."""
    print("Ejecutando en MODO MOCK...")

    df_1h = create_sample_data("1h", 60)
    df_4h = create_sample_data("4h", 60)
    df_1d = create_sample_data("1d", 60)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Replay Visualizer - Modo Mock (Test)", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(3, 1, 1)
    plot_candlestick(ax1, df_1h, "1H", progress=1.0)

    ax2 = fig.add_subplot(3, 1, 2)
    plot_candlestick(ax2, df_4h, "4H", progress=0.5)

    ax3 = fig.add_subplot(3, 1, 3)
    plot_candlestick(ax3, df_1d, "1D", progress=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_with_api() -> None:
    """Ejecuta con la API real."""
    print("Iniciando visualización con API...")

    start_replay()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Replay Visualizer - BTC/USDT", fontsize=14, fontweight="bold")

    ax1h = fig.add_subplot(3, 1, 1)
    ax4h = fig.add_subplot(3, 1, 2)
    ax1d = fig.add_subplot(3, 1, 3)

    plt.ion()
    plt.show(block=False)

    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    try:
        while _api_data["active"] and poll_thread.is_alive():
            step = _api_data.get("step", 0)
            progress_1h = (step % 100) / 100 if step < 100 else 1.0
            progress_4h = (step % 4) / 4 if step < 100 else min((step % 4 + 1) / 4, 1.0)
            progress_1d = (step % 24) / 24 if step < 100 else min((step % 24 + 1) / 24, 1.0)

            plot_candlestick(ax1h, create_sample_data("1h", 60), "1H", progress_1h)
            plot_candlestick(ax4h, create_sample_data("4h", 60), "4H", progress_4h)
            plot_candlestick(ax1d, create_sample_data("1d", 60), "1D", progress_1d)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.pause(0.5)

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        stop_replay()
        plt.ioff()
        print("Visualización terminada")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay Visualizer")
    parser.add_argument("--mock", action="store_true", help="Modo test sin API")
    args = parser.parse_args()

    if args.mock:
        run_mock()
        return

    try:
        run_with_api()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_replay()


if __name__ == "__main__":
    main()
`````

## File: tests/test_tensor_builder.py
`````python
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
`````

## File: app/api/schemas.py
`````python
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
    sync_type: str | None = Field(None, description="timeframe | merged | semantic (default from config)")
    sync_version: str | None = Field(None, description="ohlcv | indicators (default from config)")
    include_sync: bool = Field(False, description="Apply sync to the data")


class ReplayRequest(BaseModel):
    """Parameters for starting a market replay session."""
    symbol: str = Field("BTC/USDT")
    timeframes: list[str] = Field(["1h", "4h", "1d"])
    since: str | None = Field(None)
    until: str | None = Field(None)
    speed_multiplier: float = Field(1.0, ge=0.1, le=100.0)
    sync_type: str | None = Field(None, description="timeframe | merged | semantic (default from config)")
    sync_version: str | None = Field(None, description="ohlcv | indicators (default from config)")


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
`````

## File: app/core/data_ingestion/historical.py
`````python
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

        df["progress_vela"] = self._calculate_progress_vela(df["timestamp"], timeframe)

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

    def _calculate_progress_vela(self, timestamps: pd.Series, timeframe: str) -> pd.Series:
        """Calculate candle progress (0-1) for each timestamp.

        Historical data from Binance always comes with closed candles,
        so progress is always 1.0.

        Args:
            timestamps: Series of candle timestamps.
            timeframe: Timeframe string (1h, 4h, 1d).

        Returns:
            Series with progress values (0-1).
        """
        if timeframe == "1h":
            return pd.Series(1.0, index=timestamps.index)
        elif timeframe == "4h":
            return pd.Series(1.0, index=timestamps.index)
        elif timeframe == "1d":
            return pd.Series(1.0, index=timestamps.index)
        return pd.Series(1.0, index=timestamps.index)

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
`````

## File: app/core/data_ingestion/realtime.py
`````python
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
        df["progress_vela"] = self._calculate_progress_vela(df["timestamp"], timeframe)
        return df

    def _calculate_progress_vela(self, timestamps: pd.Series, timeframe: str) -> pd.Series:
        """Calculate candle progress (0-1) based on current time.

        For real-time data, calculates how complete the current candle is:
        - The last candle may be unclosed, so progress < 1.0
        - Previous candles are closed (progress = 1.0)

        Args:
            timestamps: Series of candle timestamps.
            timeframe: Timeframe string (1h, 4h, 1d).

        Returns:
            Series with progress values (0-1).
        """
        now = datetime.now(timezone.utc)
        result = pd.Series(1.0, index=timestamps.index)

        if timeframe == "1h":
            minutes_in_hour = 60
            current_minute = now.hour * 60 + now.minute
            current_progress = (current_minute % 60) / minutes_in_hour
            result.iloc[-1] = current_progress
        elif timeframe == "4h":
            hours_in_4h = 4
            current_hour = now.hour
            hour_in_period = current_hour % hours_in_4h
            current_progress = (hour_in_period * 60 + now.minute) / (hours_in_4h * 60)
            result.iloc[-1] = current_progress
        elif timeframe == "1d":
            current_hour = now.hour
            current_progress = (current_hour * 60 + now.minute) / (24 * 60)
            result.iloc[-1] = current_progress

        return result

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
`````

## File: app/core/processing/normalizer.py
`````python

`````

## File: app/core/sync/multi_timeframe.py
`````python
"""
RF-7: Multi-timeframe synchronization.

Múltiples modos de sincronización:
- TYPE 1 (timeframe): 1h|indicadores|4h|indicadores|1d|indicadores
- TYPE 2 (merged): todos los indicadores juntos sin separación por timeframe
- TYPE 3 (semantic): por grupos (VELOCIDAD|1h,4h,1d|TENDENCIA|...)

Cada tipo tiene 2 versiones:
- ohlcv: incluye OHLCV, volume, progress
- indicators: solo indicadores

Bug fix: sincronización correcta sin reindex/ffill problemático.
"""

import pandas as pd
from loguru import logger

from app.config import settings


class MultiTimeframeSync:
    """Synchronize OHLCV DataFrames con múltiples modos y versiones."""

    _TF_CANONICAL = {"1h": "1h", "4h": "4h", "1d": "1d", "1D": "1d"}

    OHLCV_COLS = ["open", "high", "low", "close", "volume", "progress_vela"]

    GROUPS = {
        "VELOCIDAD": ["MON", "ROC", "RSI_6", "RSI_14", "RSI_24", "RSI_EMA_6", "RSI_EMA_14",
                     "RSI_EMA_24", "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI"],
        "TENDENCIA": ["MACD_LINE", "MACD_SIGNAL", "MACD_HIST", "ADX", "DI_PLUS", "DI_MINUS",
                     "EMA_22", "EMA_50", "EMA_100", "ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN",
                     "ICHIMOKU_SA", "ICHIMOKU_SB", "ICHIMOKU_CHIKOU"],
        "AMPLITUD": ["BB_UPPER", "BB_MIDDLE", "BB_LOWER", "BB_WIDTH",
                     "KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"],
        "LIQUIDEZ": ["CMF", "OBV", "ELDER_BULL", "ELDER_BEAR", "EOM", "VWAP"],
    }

    def __init__(self, base_timeframe: str = "1h"):
        self._base = base_timeframe

    def synchronize(self, data: dict[str, pd.DataFrame], sync_type: str = None,
                    sync_version: str = None, include_global: bool = True) -> pd.DataFrame:
        """Main synchronization method.

        Args:
            data: dict with timeframe keys and DataFrame values
            sync_type: "timeframe" | "merged" | "semantic" (default from config)
            sync_version: "ohlcv" | "indicators" (default from config)
            include_global: add global features (precio_actual, tiempo_normalizado)
        """
        sync_type = sync_type or settings.sync_type
        sync_version = sync_version or settings.sync_version

        if self._base not in data:
            raise ValueError(f"Base timeframe '{self._base}' not found: {list(data.keys())}")

        base_df = data[self._base].copy()
        base_df = base_df.set_index("timestamp").sort_index()

        for tf, df in data.items():
            canonical = self._TF_CANONICAL.get(tf, tf)
            if canonical != self._base:
                df_copy = df.copy().set_index("timestamp").sort_index()
                # Agregar sufijo a las columnas OHLCV para evitar overlap
                for col in self.OHLCV_COLS:
                    if col in df_copy.columns:
                        df_copy = df_copy.rename(columns={col: f"{col}_{canonical}"})
                df_copy = df_copy.reindex(base_df.index, method="ffill")
                base_df = base_df.join(df_copy, how="left")

        base_df = base_df.ffill()

        result = base_df.copy()

        if sync_type == "timeframe":
            result = self._sync_timeframe(result, sync_version)
        elif sync_type == "merged":
            result = self._sync_merged(result, sync_version)
        elif sync_type == "semantic":
            result = self._sync_semantic(result, sync_version)
        else:
            raise ValueError(f"Unknown sync_type: {sync_type}")

        if include_global:
            result = self.add_global_features(result)

        return result

    def _sync_timeframe(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 1: Cada timeframe con sus indicadores juntos."""
        result = pd.DataFrame(index=df.index)

        for tf in ["1h", "4h", "1d"]:
            tf_cols = [c for c in df.columns if f"_{tf}" in c]
            if not tf_cols:
                continue

            ohlcv_tf = [c for c in tf_cols if any(c.endswith(f"_{tf}") and c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
            indicators_tf = [c for c in tf_cols if c not in ohlcv_tf]

            if version == "ohlcv":
                for col in ohlcv_tf:
                    result[col] = df[col]
            elif version == "indicators":
                pass

            for col in indicators_tf:
                result[col] = df[col]

        for col in df.columns:
            if not any(f"_{tf}" in col for tf in ["1h", "4h", "1d"]):
                result[col] = df[col]

        return result

    def _sync_merged(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 2: Todos los indicadores juntos sin separación por timeframe."""
        result = pd.DataFrame(index=df.index)

        if version == "ohlcv":
            for tf in ["1h", "4h", "1d"]:
                ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and any(c.replace(f"_{tf}", "") in self.OHLCV_COLS for _ in [1])]
                for col in ohlcv_cols:
                    result[col] = df[col]

        all_indicators = []
        for tf in ["1h", "4h", "1d"]:
            tf_indicators = [c for c in df.columns if f"_{tf}" in c and not any(c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
            all_indicators.extend(tf_indicators)

        for col in all_indicators:
            indicator_name = col.rsplit("_", 1)[0]
            for tf in ["1h", "4h", "1d"]:
                full_col = f"{indicator_name}_{tf}"
                if full_col in df.columns:
                    result[full_col] = df[full_col]

        for col in df.columns:
            if col not in result.columns:
                result[col] = df[col]

        return result

    def _sync_semantic(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 3: Por grupos semánticos (VELOCIDAD, TENDENCIA, AMPLITUD, LIQUIDEZ)."""
        result = pd.DataFrame(index=df.index)

        if version == "ohlcv":
            for tf in ["1h", "4h", "1d"]:
                ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and any(c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
                for col in ohlcv_cols:
                    result[col] = df[col]

        for group_name, indicators in self.GROUPS.items():
            group_cols = []
            for indicator in indicators:
                for tf in ["1h", "4h", "1d"]:
                    full_col = f"{indicator}_{tf}"
                    if full_col in df.columns:
                        group_cols.append(full_col)

            for col in group_cols:
                result[col] = df[col]

        for col in df.columns:
            if col not in result.columns:
                result[col] = df[col]

        return result

    @staticmethod
    def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features globales."""
        result = df.copy()

        close_1h_col = [c for c in result.columns if c.startswith("close") and "1h" in c]
        if close_1h_col:
            result["precio_actual"] = result[close_1h_col[0]]

        if "tiempo_normalizado" not in result.columns:
            result["tiempo_normalizado"] = result.index.hour / 24.0

        return result

    @staticmethod
    def get_sync_types() -> list[str]:
        """Retorna los tipos de sync disponibles."""
        return ["timeframe", "merged", "semantic"]

    @staticmethod
    def get_versions() -> list[str]:
        """Retorna las versiones disponibles."""
        return ["ohlcv", "indicators"]
`````

## File: app/utils/logger.py
`````python
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
`````

## File: README.md
`````markdown
# IA-APP: Estrategia de Inversión Cuantitativa

Este proyecto implementa una **Inteligencia Artificial** especializada en mercados financieros, diseñada para generar señales de compra y venta con objetivos de rentabilidad superiores al 10%. El sistema integra técnicas avanzadas de Deep Learning con metodologías cuantitativas de vanguardia.

## Características Principales

- **Deep Reinforcement Learning (DRL):** El agente aprende directamente de los resultados financieros, optimizando la estrategia en entornos simulados antes de operar en real.
- **Arquitectura Multimodal (xLSTM):** Integra información de precios, volumen, indicadores técnicos y sentimiento de mercado para tomar decisiones holísticas.
- **Gestión de Riesgo Avanzada:** Implementa el **Triple Barrier Method** y **Differentiable Sharpe Ratio** para asegurar ratios riesgo/beneficio favorables (objetivo 1:4) y evitar el _overfitting_.
- **Estructura Modular (RF-1 a RF-11):** El código sigue una arquitectura estricta que separa la recolección de datos, cálculo de indicadores, construcción de tensores y lógica de inferencia.

## Arquitectura Técnica

El sistema está organizado en los siguientes módulos principales:

1.  **Data Fetcher:** Obtiene datos históricos de múltiples temporalidades (1h, 4h, 1d) desde fuentes fiables (Binance, CCXT).
2.  **Indicator Engine:** Calcula indicadores técnicos (SMA, EMA, MACD, Bollinger Bands, etc.) utilizando la librería `ta`.
3.  **Tensor Builder:** Estructura los datos en tensores 3D para el entrenamiento de redes neuronales, preservando el orden temporal.
4.  **Model Architecture:** Implementación de redes basadas en **Transformers** y **LSTM** adaptadas a datos financieros (xLSTM, PatchTST).
5.  **Risk Manager:** Define las condiciones de salida y profit (10%) y stop-loss (2%) para guiar el entrenamiento.

## Instalación y Ejecución

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
`````

## File: requirements.txt
`````
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
backtrader>=1.9.78.123

# ── Tensor / Deep Learning Backend ──────────────────
torch>=2.6.0

# ── Observability & Logging ─────────────────────────
loguru>=0.7.0

# ── Testing ─────────────────────────────────────────
pytest>=8.0.0
httpx>=0.28.0
`````

## File: app/config.py
`````python
"""
Centralized application configuration.
Loads settings from .env file and environment variables.
Covers RF-1 (environment isolation) requirements.
"""

__version__ = "0.1.0"

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
    tensor_window_size: int = 100

    # ── Replay ──────────────────────────────────────
    replay_speed_multiplier: float = 1.0
    replay_refresh_seconds: float = 5.0
    replay_indicators_warmup: int = 2400  # ~100 días de 1h para indicadores completos

    # ── Sync ──────────────────────────────────────────
    # Tipos: "timeframe" | "merged" | "semantic"
    sync_type: str = "timeframe"
    # Versión: "ohlcv" | "indicators"
    sync_version: str = "indicators"

    @property
    def timeframes_list(self) -> list[str]:
        """Return timeframes as a clean list."""
        return [tf.strip() for tf in self.default_timeframes.split(",")]


settings = Settings()
`````

## File: app/core/data_ingestion/replay_backtrader.py
`````python
"""
RF-5: Backtrader Data Replay - Reconstruct higher timeframes from 1h data.

LOGICA DEL REPLAY:
1. Fetch 1h del rango completo (ej: 2026-01-01 a 2026-05-01)
2. Warmup: primeros 2400 datos (~100 días) para indicadores completos
3. Por cada step posterior:
   - Agregar 1 vela 1h al buffer
   - Acumular datos en vela 4h (cerrar cada 4 steps)
   - Acumular datos en vela 1d (cerrar cada 24 steps)
4. Retornar buffers con TODOS los datos acumulados + vela en progreso
5. Los indicadores usan todos los datos del buffer
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import pandas as pd
from loguru import logger

from app.config import settings


class BacktraderReplay:
    """Replay 1h data and reconstruct 4h/1d with dynamic buffers.

    Los buffers startan con warmup de datos históricos para indicadores completos.
    La vela en progreso siempre es parte del buffer (desde el inicio).
    """

    def __init__(
        self,
        data_1h: pd.DataFrame,
        window_size: int | None = None,
        speed_multiplier: float = 1.0,
        refresh_seconds: float = 5.0,
    ):
        self._window_size = window_size or settings.tensor_window_size
        self._indicators_warmup = settings.replay_indicators_warmup  # 2400
        self._speed = speed_multiplier
        self._refresh = refresh_seconds
        self._active = False

        # Datos 1h - asegurar que tiene índice de timestamp
        self._data_1h = data_1h.copy()
        if "timestamp" in self._data_1h.columns:
            self._data_1h = self._data_1h.set_index("timestamp").sort_index()
        self._data_1h.index = pd.DatetimeIndex(self._data_1h.index)

        # Total de datos disponibles para replay
        self._total_1h = len(self._data_1h)

        # Validar que hay suficientes datos para warmup + replay
        min_required = self._indicators_warmup + 1  # al menos 1 step
        if self._total_1h < min_required:
            raise ValueError(
                f"Insufficient data: {self._total_1h} rows. "
                f"Need at least {self._indicators_warmup} for indicators warmup (~100 days). "
                f"Requested range provides {self._total_1h - self._indicators_warmup} replay steps."
            )

        # Warmup: primeros 2400 datos para indicadores
        self._warmup_end = self._indicators_warmup
        self._max_steps = self._total_1h - self._warmup_end

        # Fecha real de inicio del step 1
        self._first_step_date = self._data_1h.index[self._warmup_end]

        # Buffers dinámicos - startan con warmup data
        self._buffer_1h = self._data_1h.iloc[:self._warmup_end].copy().reset_index()
        self._buffer_1h["progress_vela"] = 1.0

        # Pre-calcular 4h y 1d del warmup
        self._buffer_4h = self._build_warmup_timeframes(self._buffer_1h, "4h")
        self._buffer_1d = self._build_warmup_timeframes(self._buffer_1h, "1d")

        # Inicializar velas en progreso con el primer dato del replay
        first_replay_idx = self._data_1h.index[self._warmup_end]
        first_row = self._data_1h.iloc[self._warmup_end]
        self._current_4h = self._init_4h_candle(first_replay_idx, first_row)
        self._current_1d = self._init_1d_candle(first_replay_idx, first_row)

        # Agregar vela en progreso actual al buffer para visualización
        self._buffer_4h = pd.concat([self._buffer_4h, pd.DataFrame([self._current_4h])], ignore_index=True)
        self._buffer_1d = pd.concat([self._buffer_1d, pd.DataFrame([self._current_1d])], ignore_index=True)

        logger.info(
            "BacktraderReplay initialized — warmup={} (~100 days), steps={}, total_1h={}",
            self._warmup_end,
            self._max_steps,
            self._total_1h,
        )
        logger.info(
            "Replay range: {} to {}",
            self._data_1h.index[0],
            self._data_1h.index[-1],
        )
        logger.info(
            "Step 1 starts at: {} (date: {})",
            self._warmup_end,
            self._first_step_date,
        )

    def _init_4h_candle(self, timestamp, row) -> dict:
        """Inicializar vela de 4h."""
        return {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "progress_vela": 0.25,
        }

    def _init_1d_candle(self, timestamp, row) -> dict:
        """Inicializar vela de 1d."""
        return {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "progress_vela": 1 / 24,
        }

    def _build_warmup_timeframes(self, df_1h: pd.DataFrame, target: str) -> pd.DataFrame:
        """Build 4h or 1d from warmup 1h data."""
        compression = 4 if target == "4h" else 24
        rows = []

        for i in range(compression, len(df_1h) + 1, compression):
            chunk = df_1h.iloc[i - compression:i]
            if chunk.empty:
                continue
            rows.append({
                "timestamp": chunk["timestamp"].iloc[-1],
                "open": chunk["open"].iloc[0],
                "high": chunk["high"].max(),
                "low": chunk["low"].min(),
                "close": chunk["close"].iloc[-1],
                "volume": chunk["volume"].sum(),
                "progress_vela": 1.0,
            })

        return pd.DataFrame(rows)

    @property
    def first_step_date(self):
        """Retorna la fecha real donde comienza el step 1."""
        return self._first_step_date

    async def stream(self) -> AsyncGenerator[dict, None]:
        """Async generator que yield buffers dinámicos por cada step.

        Yields:
            - buffer_1h: DataFrame con todas las velas 1h acumuladas
            - buffer_4h: DataFrame con velas 4h (cerradas + en progreso)
            - buffer_1d: DataFrame con velas 1d (cerradas + en progreso)
            - step: número de step actual
        """
        self._active = True
        delay = self._refresh / self._speed

        # El replay starts desde warmup_end
        for step in range(self._max_steps):
            if not self._active:
                logger.info("Replay stopped at step {}/{}", step, self._max_steps)
                return

            # Índice del dato 1h actual (del range total)
            data_idx = self._warmup_end + step
            current_timestamp = self._data_1h.index[data_idx]
            current_row = self._data_1h.iloc[data_idx]

            # Agregar al buffer 1h
            new_1h_row = {
                "timestamp": current_timestamp,
                "open": float(current_row["open"]),
                "high": float(current_row["high"]),
                "low": float(current_row["low"]),
                "close": float(current_row["close"]),
                "volume": float(current_row["volume"]),
                "progress_vela": 1.0,
            }
            self._buffer_1h = pd.concat([self._buffer_1h, pd.DataFrame([new_1h_row])], ignore_index=True)

            # Actualizar 4h
            step_in_4h = step % 4
            progress_4h = (step_in_4h + 1) / 4.0

            if self._current_4h is not None:
                self._current_4h["high"] = max(self._current_4h["high"], new_1h_row["high"])
                self._current_4h["low"] = min(self._current_4h["low"], new_1h_row["low"])
                self._current_4h["close"] = new_1h_row["close"]
                self._current_4h["volume"] += new_1h_row["volume"]
                self._current_4h["progress_vela"] = progress_4h
                self._current_4h["timestamp"] = current_timestamp

                # Si es el último del grupo de 4, cerrar y crear nueva
                if step_in_4h == 3:
                    self._current_4h["progress_vela"] = 1.0
                    closed_4h = self._current_4h.copy()
                    self._buffer_4h = pd.concat([self._buffer_4h, pd.DataFrame([closed_4h])], ignore_index=True)

                    # Crear nueva vela
                    if data_idx + 1 < self._total_1h:
                        next_timestamp = self._data_1h.index[data_idx + 1]
                        next_row = self._data_1h.iloc[data_idx + 1]
                        self._current_4h = self._init_4h_candle(next_timestamp, next_row)

            # Actualizar 1d
            step_in_1d = step % 24
            progress_1d = (step_in_1d + 1) / 24.0

            if self._current_1d is not None:
                self._current_1d["high"] = max(self._current_1d["high"], new_1h_row["high"])
                self._current_1d["low"] = min(self._current_1d["low"], new_1h_row["low"])
                self._current_1d["close"] = new_1h_row["close"]
                self._current_1d["volume"] += new_1h_row["volume"]
                self._current_1d["progress_vela"] = progress_1d
                self._current_1d["timestamp"] = current_timestamp

                # Si es el último del grupo de 24, cerrar y crear nueva
                if step_in_1d == 23:
                    self._current_1d["progress_vela"] = 1.0
                    closed_1d = self._current_1d.copy()
                    self._buffer_1d = pd.concat([self._buffer_1d, pd.DataFrame([closed_1d])], ignore_index=True)

                    # Crear nueva vela
                    if data_idx + 1 < self._total_1h:
                        next_timestamp = self._data_1h.index[data_idx + 1]
                        next_row = self._data_1h.iloc[data_idx + 1]
                        self._current_1d = self._init_1d_candle(next_timestamp, next_row)

            # Log simplificado
            logger.info(
                f"[Step {step + 1}/{self._max_steps}] | "
                f"1h: {len(self._buffer_1h)} | "
                f"4h: {len(self._buffer_4h)} (p:{progress_4h:.2f}) | "
                f"1d: {len(self._buffer_1d)} (p:{progress_1d:.2f})"
            )

            yield {
                "buffer_1h": self._buffer_1h.copy(),
                "buffer_4h": self._buffer_4h.copy(),
                "buffer_1d": self._buffer_1d.copy(),
                "step": step,
            }

            await asyncio.sleep(delay)

        self._active = False
        logger.success("Replay completed — {} steps emitted", self._max_steps)

    def stop(self) -> None:
        """Stop the replay mid-stream."""
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def total_steps(self) -> int:
        return self._max_steps
`````

## File: app/core/processing/indicators.py
`````python
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
`````

## File: app/core/tensor/builder.py
`````python

`````

## File: app/main.py
`````python
"""
IA-APP — FastAPI Application Entry Point.

Initializes the server, registers all routes, and configures
middleware.  Covers RF-2 (Backend Initialization).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import __version__
from app.api.routes import data, replay  # tensor deshabilitado
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
app.include_router(replay.router)
# app.include_router(tensor.router)  # deshabilitado


# ── Health ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Endpoint de salud del servidor."""
    return HealthResponse(status="ok", version=__version__)
`````

## File: app/api/routes/replay.py
`````python
"""
Replay endpoints — start, stop, and status of market replay sessions.
Covers RF-5 (Market Replay) exposure via API.

Usa BacktraderReplay con buffers dinámicos:
- buffer_1h: todas las velas 1h acumuladas (warmup + progresivas)
- buffer_4h: velas 4h (cerradas + en progreso)
- buffer_1d: velas 1d (cerradas + en progreso)

El replay usa TODOS los datos acumulados para indicadores.
"""

import asyncio

from fastapi import APIRouter, HTTPException
import pandas as pd

from app.api.schemas import PipelineStatus, ReplayRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.replay_backtrader import BacktraderReplay
from app.core.processing.indicators import IndicatorEngine
from app.core.sync.multi_timeframe import MultiTimeframeSync

router = APIRouter(prefix="/replay", tags=["Market Replay"])

_historical = HistoricalDataFetcher()
_sync = MultiTimeframeSync()

_current_replay: BacktraderReplay | None = None
_current_step: int = 0
_replay_active: bool = False
_current_sync_type = "timeframe"
_current_sync_version = "indicators"


@router.post("/start")
async def start_replay(req: ReplayRequest):
    """Start a new market replay session.

    Fetch 1h del rango especificado (since - until).
    Por cada step:
    - Agregar vela 1h al buffer
    - Reconstruir 4h (cierra cada 4 steps)
    - Reconstruir 1d (cierra cada 24 steps)
    - Calcular indicadores con TODOS los datos acumulados
    - sincronizar timeframes

    El replay es indefinido - recorre todos los datos del rango.
    """
    global _current_replay, _current_step, _replay_active, _current_sync_type, _current_sync_version

    if _current_replay and _current_replay.is_active:
        raise HTTPException(status_code=409, detail="Replay already running — stop it first")

    try:
        since = req.since or (req.until if req.until else None)
        until = req.until

        # Fetch 1h del rango completo
        raw_1h = _historical.fetch(
            symbol=req.symbol,
            timeframe="1h",
            since=since,
            until=until,
        )

        min_required = settings.replay_indicators_warmup + 1
        if len(raw_1h) < min_required:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(raw_1h)} rows. Need at least {min_required} (~{min_required//24} days). "
                f"Indicators warmup uses {settings.replay_indicators_warmup} rows (~{settings.replay_indicators_warmup//24} days). "
                f"This leaves {len(raw_1h) - settings.replay_indicators_warmup} replay steps."
            )

        # Crear replay con warmup de 2400 datos para indicadores completos
        _current_replay = BacktraderReplay(
            data_1h=raw_1h,
            window_size=settings.tensor_window_size,
            speed_multiplier=req.speed_multiplier,
            refresh_seconds=settings.replay_refresh_seconds,
        )
        _current_step = 0
        _replay_active = True
        _current_sync_type = req.sync_type or settings.sync_type
        _current_sync_version = req.sync_version or settings.sync_version

        asyncio.create_task(_run_replay())

        return {
            "status": "started",
            "total_steps": _current_replay.total_steps,
            "speed": req.speed_multiplier,
            "warmup_1h_rows": settings.replay_indicators_warmup,
            "warmup_days": settings.replay_indicators_warmup // 24,
            "total_1h_data": len(raw_1h),
            "replay_start_date": str(_current_replay.first_step_date),
            "sync_type": _current_sync_type,
            "sync_version": _current_sync_version,
            "note": f"Indicators warmup uses {settings.replay_indicators_warmup} rows (~100 days). Step 1 starts at: {_current_replay.first_step_date}",
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/stop")
def stop_replay():
    """Stop the currently running replay."""
    global _current_replay, _replay_active
    if _current_replay:
        _current_replay.stop()
        _replay_active = False
        return {
            "status": "stopped",
            "step": _current_step,
        }
    return {"status": "no_replay_running"}


@router.get("/status", response_model=PipelineStatus)
def replay_status():
    """Return the current replay status."""
    return PipelineStatus(
        mode="replay" if _replay_active else "idle",
        replay_active=_replay_active,
        replay_step=_current_step if _replay_active else None,
        replay_total_steps=_current_replay.total_steps if _current_replay else None,
    )


async def _run_replay():
    """Consume the replay stream and process with indicators."""
    global _current_step, _replay_active, _current_sync_type, _current_sync_version

    if not _current_replay:
        return

    # Asegurar valores por defecto del sync
    if not _current_sync_type:
        _current_sync_type = settings.sync_type
    if not _current_sync_version:
        _current_sync_version = settings.sync_version

    # Mensaje de inicio con fecha exacta
    first_date = _current_replay.first_step_date
    window_size = settings.tensor_window_size
    window_end = first_date + pd.Timedelta(hours=window_size)

    logger.info("=== REPLAY START ===")
    logger.info("Total steps: {}".format(_current_replay.total_steps))
    logger.info("Warmup rows: {} (~100 days)".format(settings.replay_indicators_warmup))
    logger.info("Step 1 starts at: {}".format(first_date))
    logger.info("Window size: {} hours".format(window_size))
    logger.info("First window (100 steps): {} to {}".format(first_date, window_end))
    logger.info("Sync type: {} (version: {})".format(_current_sync_type, _current_sync_version))
    logger.info("=" * 40)

    try:
        async for window in _current_replay.stream():
            if not _replay_active:
                break

            step = window["step"]
            _current_step = step + 1

            try:
                buffers = {
                    "1h": window["buffer_1h"],
                    "4h": window["buffer_4h"],
                    "1d": window["buffer_1d"],
                }

                # Obtener datos DIRECTAMENTE de los buffers - NO del sync
                buf_1h = buffers["1h"]
                buf_4h = buffers["4h"]
                buf_1d = buffers["1d"]

                last_1h = buf_1h.iloc[-1].to_dict() if not buf_1h.empty else {}
                last_4h = buf_4h.iloc[-1].to_dict() if not buf_4h.empty else {}
                last_1d = buf_1d.iloc[-1].to_dict() if not buf_1d.empty else {}

                # Log de reconstrucción - usando valores de los buffers directamente
                logger.info("=== Step {} ===".format(step))
                logger.info("1h: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
                    last_1h.get("open", 0),
                    last_1h.get("high", 0),
                    last_1h.get("low", 0),
                    last_1h.get("close", 0),
                    last_1h.get("volume", 0),
                    last_1h.get("progress_vela", 1.0),
                ))
                logger.info("4h: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
                    last_4h.get("open", 0),
                    last_4h.get("high", 0),
                    last_4h.get("low", 0),
                    last_4h.get("close", 0),
                    last_4h.get("volume", 0),
                    last_4h.get("progress_vela", 0.0),
                ))
                logger.info("1d: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
                    last_1d.get("open", 0),
                    last_1d.get("high", 0),
                    last_1d.get("low", 0),
                    last_1d.get("close", 0),
                    last_1d.get("volume", 0),
                    last_1d.get("progress_vela", 0.0),
                ))

                # Sincronizar para uso interno (tensor)
                # Debug: mostrar tamaño de buffers antes del sync
                # logger.debug("Before sync - buffers: 1h:{}, 4h:{}, 1d:{}".format(
                #     len(buffers["1h"]), len(buffers["4h"]), len(buffers["1d"])))

                try:
                    with_indicators = IndicatorEngine.compute_multi_timeframe(buffers)
                    synced = _sync.synchronize(
                        with_indicators,
                        sync_type=_current_sync_type,
                        sync_version=_current_sync_version
                    )
                    synced = MultiTimeframeSync.add_global_features(synced)

                    logger.debug("Sync completed - shape: {}".format(synced.shape))

                    if synced.empty:
                        logger.warning("Sync returned empty DataFrame")
                        continue

                except Exception as sync_error:
                    logger.error("Sync failed at step {}: {}", _current_step, sync_error)
                    continue

                # Log del sync generado - TODAS las columnas
                logger.info("=== SYNC (type:{}, ver:{}) | rows:{} | cols:{} ===".format(
                    _current_sync_type, _current_sync_version, synced.shape[0], synced.shape[1]))
                for col in synced.columns:
                    val = synced.iloc[-1][col]
                    if pd.notna(val):
                        logger.info("  {}: {}".format(col, val))

                last_row = synced.iloc[-1].round(2)

                # Nueva fila completa con TODOS los indicadores por temporalidad
                def get_indicator_cols(prefix):
                    cols = []
                    for col in last_row.index:
                        if f"_{prefix}" in col:
                            val = last_row[col]
                            if pd.notna(val):
                                indicator_name = col.replace(f"_{prefix}", "")
                                cols.append("{}:{}".format(indicator_name, val))
                    return " | ".join(cols) if cols else "N/A"

                logger.info("=== FULL ROW - INDICATORS ===")
                logger.info("1h: |{}|".format(get_indicator_cols("1h")))
                logger.info("4h: |{}|".format(get_indicator_cols("4h")))
                logger.info("1d: |{}|".format(get_indicator_cols("1d")))

            except Exception as e:
                logger.error("Error at step {}: {}", _current_step, e)

        _replay_active = False
        logger.success("Replay finished — {} steps completed".format(_current_step))

    except Exception as e:
        logger.error("Replay error: {}", e)
        _replay_active = False


from loguru import logger
`````
