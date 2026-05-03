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
