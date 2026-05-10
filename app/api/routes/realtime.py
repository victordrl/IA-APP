"""
Realtime endpoints.
Uses MarketRealtimeEngine to process real-time market data with a bounded FIFO buffer.
"""

from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from app.api.schemas import RealtimeRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.market_engine import MarketRealtimeEngine

router = APIRouter(prefix="/realtime", tags=["Market Realtime"])
HISTORICAL = HistoricalDataFetcher()

SYNC_TYPES = ["timeframe", "merged", "semantic"]
SYNC_VERSIONS = ["base", "ohlcv", "indicators"]

@router.post("/run")
async def run_realtime(req: RealtimeRequest):
    """
    Simula una sesión de realtime instantánea obteniendo datos hasta la fecha actual 
    y procesando en memoria hasta devolver la ventana con los últimos N steps.
    """
    try:
        # 4 meses atrás por defecto para warmup
        until_dt = datetime.utcnow()
        since_dt = until_dt - timedelta(days=120)
        
        since_str = since_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        until_str = until_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        
        raw_1h = HISTORICAL.fetch(
            symbol=req.symbol,
            timeframe="1h",
            since=since_str,
            until=until_str,
        )
        
        warmup = settings.replay_indicators_warmup
        if len(raw_1h) <= warmup:
            raise HTTPException(
                status_code=400,
                detail=f"Datos recientes insuficientes. Se requiere un warmup de {warmup} velas."
            )
            
        sync_type = req.sync_type if req.sync_type in SYNC_TYPES else "timeframe"
        sync_version = req.sync_version if req.sync_version in SYNC_VERSIONS else "ohlcv"
        
        engine = MarketRealtimeEngine(data_1h=raw_1h, warmup_size=warmup, n_steps=req.n_steps)
        
        total_steps_to_run = engine.total_rows - engine.warmup_size
        
        # Procesar todos los steps históricos hasta el presente
        for step in range(total_steps_to_run):
            try:
                engine.add_next_step(step, sync_type, sync_version)
            except ValueError:
                break
                
        results = engine.get_latest_steps()

        return {
            "status": "finished",
            "n_steps": len(results),
            "sync_type": sync_type,
            "sync_version": sync_version,
            "results": results
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
