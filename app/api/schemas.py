"""
Pydantic schemas for API request / response models.
"""

from pydantic import BaseModel, Field


# ── Requests ────────────────────────────────────────


class HistoricalRequest(BaseModel):
    """Parameters for fetching historical OHLCV data."""
    symbol: str = Field("BTC/USDT", description="Trading pair")
    timeframes: list[str] = Field(["1h", "4h", "1d"], description="Candle intervals")
    since: str | None = Field(None, description="Start date ISO-8601 (e.g. 2024-01-01T00:00:00Z)")
    until: str | None = Field(None, description="End date ISO-8601")


class ReplayRequest(BaseModel):
    """Parameters for starting a market replay session."""
    symbol: str = Field("BTC/USDT")
    timeframes: list[str] = Field(["1h", "4h", "1d"])
    since: str | None = Field(None)
    until: str | None = Field(None)
    speed_multiplier: float = Field(1.0, ge=0.1, le=100.0)


# ── Responses ───────────────────────────────────────


class TensorMeta(BaseModel):
    """Metadata describing a generated tensor."""
    window_size: int
    num_features: int
    num_rows: int
    num_windows: int
    tensor_shape: list[int]
    feature_columns: list[str]


class PipelineStatus(BaseModel):
    """Current status of the data pipeline."""
    mode: str  # "idle" | "historical" | "realtime" | "replay"
    replay_active: bool
    replay_step: int | None = None
    replay_total_steps: int | None = None


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    version: str
