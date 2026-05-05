"""
Replay endpoints — start, stop, and status of market replay sessions.
Covers RF-5 (Market Replay) exposure via API.

Uses BacktraderReplay to reconstruct 4h and 1d candles from 1h data,
providing more realistic backtesting than pre-built candles.

Each replay step integrates with the full tensor pipeline:
fetch 1h → reconstruct 4h/1d → indicators → sync → normalize → tensor.
"""

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException
import torch

from app.api.schemas import PipelineStatus, ReplayRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.replay_backtrader import BacktraderReplay
from app.core.processing.indicators import IndicatorEngine
from app.core.processing.normalizer import Normalizer
from app.core.sync.multi_timeframe import MultiTimeframeSync
from app.core.tensor.builder import TensorBuilder

router = APIRouter(prefix="/replay", tags=["Market Replay"])

_historical = HistoricalDataFetcher()
_sync = MultiTimeframeSync()
_normalizer = Normalizer()
_builder = TensorBuilder()

_current_replay: BacktraderReplay | None = None
_current_step: int = 0
_tensors_grouped: list[dict[str, torch.Tensor]] = []
_replay_active: bool = False


@router.post("/start")
async def start_replay(req: ReplayRequest):
    """Start a new market replay session using Backtrader data replay.

    Fetches 1h historical data, then uses Backtrader to reconstruct 4h and 1d
    candles in real-time as the replay progresses.

    Each step executes the full pipeline:
    1. Take 1h window
    2. Reconstruct 4h via backtrader replay
    3. Reconstruct 1d via backtrader replay
    4. Compute indicators
    5. Synchronize timeframes
    6. Normalize
    7. Build tensor

    Returns:
        status, total_steps, speed multiplier
    """
    global _current_replay, _current_step, _tensors, _replay_active

    if _current_replay and _current_replay.is_active:
        raise HTTPException(status_code=409, detail="Replay already running — stop it first")

    try:
        raw_1h = _historical.fetch(
            symbol=req.symbol,
            timeframe="1h",
            since=req.since,
            until=req.until,
        )

        _current_replay = BacktraderReplay(
            data_1h=raw_1h,
            window_size=settings.tensor_window_size,
            speed_multiplier=req.speed_multiplier,
            refresh_seconds=settings.replay_refresh_seconds,
        )
        _current_step = 0
        _tensors_grouped = []
        _replay_active = True

        asyncio.create_task(_run_replay())

        return {
            "status": "started",
            "total_steps": _current_replay.total_steps,
            "speed": req.speed_multiplier,
            "note": "Using Backtrader replay: 1h → 4h/1d reconstruction",
        }
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
            "tensors_built": len(_tensors_grouped),
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


@router.get("/tensors")
def get_tensors():
    """Return the tensors built during replay (grouped by indicator category)."""
    if not _tensors_grouped:
        return {"count": 0, "shapes": {}}

    shapes = []
    for tg in _tensors_grouped:
        step_shapes = {k: list(v.shape) if v is not None else None for k, v in tg.items()}
        shapes.append(step_shapes)

    return {
        "count": len(_tensors_grouped),
        "shapes": shapes,
    }


@router.get("/tensor/{step}/{grupo}")
def get_tensor_by_group(step: int, grupo: str):
    """Get tensor for a specific step and group (velocidad/tendencia/amplitud/liquidez)."""
    valid_groups = ["velocidad", "tendencia", "amplitud", "liquidez"]
    if grupo not in valid_groups:
        raise HTTPException(status_code=400, detail=f"Grupo inválido. Available: {valid_groups}")

    if step < 0 or step >= len(_tensors_grouped):
        raise HTTPException(status_code=404, detail=f"Step {step} not found")

    tensor = _tensors_grouped[step].get(grupo)
    if tensor is None:
        raise HTTPException(status_code=404, detail=f"Tensor for grupo '{grupo}' at step {step} is None")

    return {
        "step": step,
        "grupo": grupo,
        "shape": list(tensor.shape),
        "data": tensor.numpy().tolist(),
    }


async def _run_replay():
    """Consume the replay stream and build tensors."""
    global _current_step, _replay_active

    if not _current_replay:
        return

    try:
        async for window in _current_replay.stream():
            if not _replay_active:
                break

            _current_step += 1

            try:
                data = {
                    "1h": window["1h"],
                    "4h": window["4h_recon"],
                    "1d": window["1d_recon"],
                }

                with_indicators = IndicatorEngine.compute_multi_timeframe(data)

                synced = _sync.synchronize(with_indicators)
                synced = MultiTimeframeSync.add_global_features(synced)

                normalized = _normalizer.fit_transform(synced)

                tensors_grouped = _builder.build_grouped(normalized)
                _tensors_grouped.append(tensors_grouped)

                # Debug output con shapes por grupo
                shapes_str = ", ".join([f"{k}:{v.shape[1:] if v is not None else 'None'}" for k, v in tensors_grouped.items()])
                logger.info(
                    "[Step {}/{}] Tensores: {}",
                    _current_step,
                    _current_replay.total_steps,
                    shapes_str,
                )

            except Exception as e:
                logger.error("Error building tensor at step {}: {}", _current_step, e)

        _replay_active = False
        logger.success("Replay finished — {} steps completed", len(_tensors_grouped))

    except Exception as e:
        logger.error("Replay error: {}", e)
        _replay_active = False


from loguru import logger