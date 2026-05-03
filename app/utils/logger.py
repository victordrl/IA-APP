"""
Centralized logging configuration using Loguru.

Provides structured, colored console output and optional file rotation.
Satisfies RNF-4 (Observability).
"""

import sys

from loguru import logger

from app.config import settings


def setup_logging() -> None:
    """Configure Loguru for the application."""
    # Remove default handler
    logger.remove()

    # Console handler — colorized, with context
    log_level = "DEBUG" if settings.app_debug else "INFO"
    logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler — rotated daily, kept 7 days
    logger.add(
        "logs/ia_app_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    )

    logger.info("Logging initialized — level={}", log_level)
