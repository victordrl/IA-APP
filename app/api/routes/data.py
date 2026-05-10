"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd

from app.api.schemas import HistoricalRequest
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.realtime import RealTimeDataFetcher
from app.core.processing.indicators import IndicatorEngine

router = APIRouter(prefix="/data", tags=["Data Ingestion"])

_historical: HistoricalDataFetcher | None = None
_realtime = RealTimeDataFetcher()

def _get_historical() -> HistoricalDataFetcher:
    """Lazy initialization of historical fetcher."""
    global _historical
    if _historical is None:
        _historical = HistoricalDataFetcher()
    return _historical

@router.post("/historical")
def fetch_historical(req: HistoricalRequest):
    """Fetch historical OHLCV data, optionally with technical indicators.

    If include_indicators=True, appends technical indicators.
    NaN values appear where the lookback window is insufficient.
    """
    try:
        result = _get_historical().fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )
        if req.include_indicators:
            result = IndicatorEngine.compute_multi_timeframe(result)
        return {
            tf: (
                df.replace({float("nan"): None})
                .to_dict(orient="records")
            )
            for tf, df in result.items()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.get("/ultimas_velas")
def fetch_ultimas_velas(
    symbol: str = "BTC/USDT",
    limit: int = 30,
    include_indicators: bool = False,
    timeframes: str = "1h,4h,1d",
):
    """Fetch the latest *limit* candles across all default timeframes.
    
    Optional: include_indicators=True to append technical indicators.
    """
    try:
        tf_list = [t.strip() for t in timeframes.split(",")]
        result = _realtime.fetch_latest_multi_timeframe(
            symbol=symbol, limit=limit, timeframes=tf_list
        )
        if include_indicators:
            result = IndicatorEngine.compute_multi_timeframe(result)
            return {
                tf: (
                    df.replace({float("nan"): None})
                    .to_dict(orient="records")
                )
                for tf, df in result.items()
            }
        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))