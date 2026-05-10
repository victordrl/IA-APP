"""
Replay endpoints — start of market replay sessions.
Uses MarketReplayEngine to simulate historical data sequentially without sleeps.
"""

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool
from app.api.schemas import ReplayRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.market_engine import MarketReplayEngine

router = APIRouter(prefix="/replay", tags=["Market Replay"])
HISTORICAL = HistoricalDataFetcher()

SYNC_TYPES = ["timeframe", "merged", "semantic"]
SYNC_VERSIONS = ["base", "ohlcv", "indicators"]

@router.post("/run")
async def run_replay(req: ReplayRequest):
    """
    Ejecuta el Motor de Replay de manera ininterrumpida y devuelve 
    toda la lista de steps generados al finalizar.
    """
    try:
        since = req.since or (req.until if req.until else None)
        until = req.until

        raw_1h = await run_in_threadpool(
            HISTORICAL.fetch,
            symbol=req.symbol,
            timeframe="1h",
            since=since,
            until=until,
        )

        warmup = settings.replay_indicators_warmup
        if len(raw_1h) <= warmup:
            raise HTTPException(
                status_code=400,
                detail=f"Datos insuficientes. Se requiere un warmup de {warmup} velas."
            )

        sync_type = req.sync_type if req.sync_type in SYNC_TYPES else "timeframe"
        sync_version = req.sync_version if req.sync_version in SYNC_VERSIONS else "ohlcv"

        engine = MarketReplayEngine(data_1h=raw_1h, warmup_size=warmup)
        
        results = await run_in_threadpool(
            engine.run_replay,
            sync_type=sync_type, 
            sync_version=sync_version,
            normalized=req.normalized
        )

        return {
            "status": "finished",
            "total_steps": len(results),
            "sync_type": sync_type,
            "sync_version": sync_version,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))