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
from datetime import timedelta

from fastapi import APIRouter, HTTPException

from app.api.schemas import PipelineStatus, ReplayRequest, ReplayConfigRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.replay_backtrader import BacktraderReplay
from app.core.processing.indicators import IndicatorEngine
from app.core.sync.multi_timeframe import MultiTimeframeSync

router = APIRouter(prefix="/replay", tags=["Market Replay"])

HISTORICAL = HistoricalDataFetcher()
_SYNC = MultiTimeframeSync()

SYNC_TYPES = ["timeframe", "merged", "semantic"]
SYNC_VERSIONS = ["ohlcv", "indicators"]

_current_replay: BacktraderReplay | None = None
_current_step: int = 0
_replay_active: bool = False
_current_sync_type = "timeframe"
_current_sync_version = "indicators"
_completed_steps: list = []
_replay_finished: bool = False


def _validate_sync(sync_type: str | None, sync_version: str | None) -> tuple[str, str]:
    """Validate and return sync_type and sync_version with defaults."""
    return (
        sync_type if sync_type in SYNC_TYPES else "timeframe",
        sync_version if sync_version in SYNC_VERSIONS else "ohlcv",
    )


def _log_step(step: int, last_1h: dict, last_4h: dict, last_1d: dict, 
              progress_4h: float, progress_1d: float):
    """Log step reconstruction with OHLCV data."""
    logger.info("=== Step {} ===".format(step))
    logger.info("1h: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
        last_1h.get("open", 0), last_1h.get("high", 0), last_1h.get("low", 0),
        last_1h.get("close", 0), last_1h.get("volume", 0), last_1h.get("progress_vela", 1.0),
    ))
    logger.info("4h: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
        last_4h.get("open", 0), last_4h.get("high", 0), last_4h.get("low", 0),
        last_4h.get("close", 0), last_4h.get("volume", 0), progress_4h,
    ))
    logger.info("1d: O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f} P:{:.2f}".format(
        last_1d.get("open", 0), last_1d.get("high", 0), last_1d.get("low", 0),
        last_1d.get("close", 0), last_1d.get("volume", 0), progress_1d,
    ))


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

        raw_1h = HISTORICAL.fetch(
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

        _current_replay = BacktraderReplay(
            data_1h=raw_1h,
            window_size=settings.tensor_window_size,
            speed_multiplier=req.speed_multiplier,
            refresh_seconds=settings.replay_refresh_seconds,
        )
        _current_step = 0
        _replay_active = True

        _current_sync_type, _current_sync_version = _validate_sync(req.sync_type, req.sync_version)

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
        return {"status": "stopped", "step": _current_step}
    return {"status": "no_replay_running"}


@router.get("/status", response_model=PipelineStatus)
def replay_status():
    """Return the current replay status."""
    return PipelineStatus(
        mode="replay" if _replay_active else "idle",
        replay_active=_replay_active,
        replay_step=_current_step if _replay_active else None,
        replay_total_steps=_current_replay.total_steps if _current_replay else None,
        sync_type=_current_sync_type,
        sync_version=_current_sync_version,
    )


@router.get("/results")
def get_replay_results(
    limit: int = 0,
    sync_type: str = "timeframe",
    sync_version: str = "ohlcv"
):
    """Return all completed steps from the last replay session.
    
    Args:
        limit: If > 0, return only the last N steps. If 0, return all steps.
        sync_type: Requested sync format (timeframe|merged|semantic). 
                   If different from stored, data still contains all columns.
        sync_version: Requested version (ohlcv|indicators|base).
    
    Returns:
        Dict with steps, total_steps, current_step, and status.
    """
    if not _completed_steps:
        return {"status": "no_results", "steps": [], "total_steps": 0, "current_step": 0}
    
    # Validate sync params
    sync_type = sync_type if sync_type in SYNC_TYPES else "timeframe"
    sync_version = sync_version if sync_version in SYNC_VERSIONS else "ohlcv"
    
    # Check if requested format matches stored format
    stored_format = _completed_steps[0].get("sync_type") if _completed_steps else None
    format_warning = None
    if stored_format and stored_format != sync_type:
        format_warning = f"Requested '{sync_type}' differs from stored '{stored_format}'. Data contains all columns."
    
    steps_to_return = _completed_steps if limit <= 0 else _completed_steps[-limit:]
    
    # Clean up NaN/inf values for JSON
    clean_steps = []
    for step in steps_to_return:
        clean_step = {k: (None if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')) else v) 
                      for k, v in step.items()}
        clean_steps.append(clean_step)
    
    result = {
        "status": "finished" if _replay_finished else "running",
        "steps": clean_steps,
        "total_steps": len(_completed_steps),
        "current_step": _current_step,
        "sync_type": sync_type,
        "sync_version": sync_version,
    }
    if format_warning:
        result["warning"] = format_warning
    
    return result


@router.patch("/config")
def update_replay_config(config: ReplayConfigRequest):
    """Actualizar sync_type y sync_version durante replay activo."""
    global _current_sync_type, _current_sync_version
    _current_sync_type, _current_sync_version = _validate_sync(config.sync_type, config.sync_version)
    return {"status": "updated", "sync_type": _current_sync_type, "sync_version": _current_sync_version}


async def _run_replay():
    """Consume the replay stream and process with indicators."""
    global _current_step, _replay_active, _current_sync_type, _current_sync_version
    global _completed_steps, _replay_finished

    _completed_steps = []
    _replay_finished = False

    if not _current_replay:
        return

    _current_sync_type, _current_sync_version = _validate_sync(_current_sync_type, _current_sync_version)

    first_date = _current_replay.first_step_date
    window_end = first_date + timedelta(hours=settings.tensor_window_size)

    logger.info("=== REPLAY START ===")
    logger.info("Total steps: {}".format(_current_replay.total_steps))
    logger.info("Warmup rows: {} (~100 days)".format(settings.replay_indicators_warmup))
    logger.info("Step 1 starts at: {}".format(first_date))
    logger.info("Window size: {} hours".format(settings.tensor_window_size))
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
                buffers = {"1h": window["buffer_1h"], "4h": window["buffer_4h"], "1d": window["buffer_1d"]}

                # Calcular indicadores con TODOS los datos acumulados
                with_indicators = IndicatorEngine.compute_multi_timeframe(buffers)
                
                # Sincronizar según sync_type y sync_version
                synced = _SYNC.synchronize(
                    with_indicators,
                    sync_type=_current_sync_type,
                    sync_version=_current_sync_version
                )
                synced = MultiTimeframeSync.add_global_features(synced)
                
                # Obtener última fila sincronizada con todos los datos (indicadores + OHLCV)
                last_row = synced.iloc[-1].round(2)
                
                # Obtener timestamp del step
                ts = last_row.get("timestamp") or (buffers["1h"].iloc[-1]["timestamp"] if not buffers["1h"].empty else None)
                
                # Log del step
                progress_4h = window.get("progress_4h", 0.0)
                progress_1d = window.get("progress_1d", 0.0)
                
                last_1h = buffers["1h"].iloc[-1].to_dict() if not buffers["1h"].empty else {}
                last_4h = buffers["4h"].iloc[-1].to_dict() if not buffers["4h"].empty else {}
                last_1d = buffers["1d"].iloc[-1].to_dict() if not buffers["1d"].empty else {}
                _log_step(step, last_1h, last_4h, last_1d, progress_4h, progress_1d)

                # Guardar step con datos sincronizados (contiene indicadores si sync_version=indicators)
                step_data = {
                    "step": step,
                    "timestamp": str(ts) if ts else None,
                    "sync_type": _current_sync_type,
                    "sync_version": _current_sync_version,
                    "data": last_row.to_dict()
                }
                _completed_steps.append(step_data)

            except Exception as e:
                logger.error("Error at step {}: {}", _current_step, e)

        _replay_active = False
        _replay_finished = True
        logger.success("Replay finished — {} steps completed".format(_current_step))

    except Exception as e:
        logger.error("Replay error: {}", e)
        _replay_active = False


from loguru import logger