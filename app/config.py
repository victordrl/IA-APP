"""
Centralized application configuration.
Loads settings from .env file and environment variables.
Covers RF-1 (environment isolation) requirements.
"""

__version__ = "0.1.0"

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from .env / environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Server ──────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = True

    # ── Exchange ────────────────────────────────────
    exchange_id: str = "binance"
    exchange_rate_limit: bool = True

    # ── Data ────────────────────────────────────────
    default_symbol: str = "BTC/USDT"
    default_timeframes: str = "1h,4h,1d"

    # ── Tensor ──────────────────────────────────────
    tensor_window_size: int = 30

    # ── Replay ──────────────────────────────────────
    replay_speed_multiplier: float = 1.0
    replay_refresh_seconds: float = 5.0

    @property
    def timeframes_list(self) -> list[str]:
        """Return timeframes as a clean list."""
        return [tf.strip() for tf in self.default_timeframes.split(",")]


settings = Settings()
