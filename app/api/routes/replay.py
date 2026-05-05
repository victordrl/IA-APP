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
_current_sync_type: str = "timeframe"
_current_sync_version: str = "ohlcv"


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
                with_indicators = IndicatorEngine.compute_multi_timeframe(buffers)
                synced = _sync.synchronize(
                    with_indicators,
                    sync_type=_current_sync_type,
                    sync_version=_current_sync_version
                )
                synced = MultiTimeframeSync.add_global_features(synced)

                if synced.empty:
                    continue

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