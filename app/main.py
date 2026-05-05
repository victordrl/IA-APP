"""
IA-APP — FastAPI Application Entry Point.

Initializes the server, registers all routes, and configures
middleware.  Covers RF-2 (Backend Initialization).
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import __version__
from app.api.routes import data, replay, tensor
from app.api.schemas import HealthResponse
from app.utils.logger import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup & shutdown hooks."""
    setup_logging()
    yield


app = FastAPI(
    title="IA-APP — AI Trading Data Pipeline",
    description=(
        "Fase 1: Infraestructura, Servidor y Pipeline de Datos. "
        "Obtiene, sincroniza, normaliza y entrega tensores de mercado "
        "multi-temporalidad (1h, 4h, 1d) listos para redes neuronales."
    ),
    version=__version__,
    lifespan=lifespan,
)

# ── CORS ────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ──────────────────────────────────────────
app.include_router(data.router)
app.include_router(tensor.router)
app.include_router(replay.router)


# ── Health ──────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Endpoint de salud del servidor."""
    return HealthResponse(status="ok", version=__version__)
