"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.

Incluye opción de sincronización multi-timeframe:
- sync_type: timeframe | merged | semantic
- sync_version: ohlcv | indicators
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import HistoricalRequest
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

        valid_sync_types = ["timeframe", "merged", "semantic"]
        valid_sync_versions = ["ohlcv", "indicators"]
        sync_type = req.sync_type if req.sync_type in valid_sync_types else "timeframe"
        sync_version = req.sync_version if req.sync_version in valid_sync_versions else "ohlcv"

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
            valid_sync_types = ["timeframe", "merged", "semantic"]
            valid_sync_versions = ["ohlcv", "indicators"]
            use_sync_type = sync_type if sync_type in valid_sync_types else "timeframe"
            use_sync_version = sync_version if sync_version in valid_sync_versions else "ohlcv"
            synced = _apply_sync(result, use_sync_type, use_sync_version)
            return synced.to_dict(orient="records")

        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))