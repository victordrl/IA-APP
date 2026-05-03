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