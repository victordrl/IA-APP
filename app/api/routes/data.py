"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.
"""

from fastapi import APIRouter, HTTPException

from app.api.schemas import HistoricalRequest
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.realtime import RealTimeDataFetcher
from app.core.processing.indicators import IndicatorEngine

router = APIRouter(prefix="/data", tags=["Data Ingestion"])

_historical = HistoricalDataFetcher()
_realtime = RealTimeDataFetcher()


@router.post("/historical")
def fetch_historical(req: HistoricalRequest):
    """Fetch historical OHLCV data, optionally with technical indicators.

    If include_indicators=True, appends ~40 indicators per timeframe
    (VELOCIDAD, TENDENCIA, AMPLITUD, LIQUIDEZ groups).
    NaN values appear where the lookback window is insufficient.
    """
    try:
        result = _historical.fetch_multi_timeframe(
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


@router.get("/realtime")
def fetch_realtime(
    symbol: str = "BTC/USDT",
    limit: int = 30,
):
    """Fetch the latest *limit* candles across all default timeframes."""
    try:
        result = _realtime.fetch_latest_multi_timeframe(symbol=symbol, limit=limit)
        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))