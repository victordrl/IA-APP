"""
Laplace model training endpoints.
Uses replay data to train the agent model (laplace_v3).
"""

import os
import json
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import joblib
from keras import layers
from loguru import logger

from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.market_engine import MarketReplayEngine

router = APIRouter(prefix="/laplace", tags=["Laplace Training"])

# Default config - must be defined before model loading
DEFAULT_WINDOW_SIZE = 24  # 24 timesteps
DEFAULT_FEATURES = 20     # 20 features per timestep

# Global model storage
_laplace_model = None

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, 'laplace_f3.keras')
CACHE_DIR = os.path.join(BASE_DIR, 'data', 'cache')
TRAIN_CACHE_FILE = os.path.join(CACHE_DIR, 'laplace_train_cache.json')

# Load model if exists
if os.path.exists(MODEL_PATH):
    try:
        _laplace_model = keras.models.load_model(MODEL_PATH)
        logger.info(f"Loaded existing model from {MODEL_PATH}")
    except Exception as e:
        logger.warning(f"Could not load model: {e}")
        _laplace_model = None

HISTORICAL = HistoricalDataFetcher()


def _crear_modelo(ventana_tiempo: int = DEFAULT_WINDOW_SIZE, features: int = DEFAULT_FEATURES):
    """Create the Laplace autoencoder model (laplace_v3 architecture)."""
    modelo = keras.Sequential([
        # Encoder
        layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same', 
                      input_shape=(ventana_tiempo, features)),
        layers.MaxPool1D(2),
        layers.LSTM(64, return_sequences=False),
        layers.Dense(16, activation='relu', name="capa_procesamiento"),
        
        # Bridge
        layers.RepeatVector(ventana_tiempo),
        
        # Decoder
        layers.LSTM(64, return_sequences=True),
        layers.TimeDistributed(layers.Dense(features))
    ])
    
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return modelo


@router.get("/current")
async def laplace_current():
    """Get current model info. Returns hardcoded list for now."""
    return {
        "models": [
            {
                "name": "laplace_v3",
                "version": "3.0",
                "window_size": 24,
                "features": 20,
                "status": "ready" if _laplace_model is not None else "not_created",
                "created_at": "2026-01-01T00:00:00Z",
                "last_trained": None
            }
        ]
    }


@router.post("/train")
async def laplace_train(
    symbol: str = "BTC/USDT",
    timeframes: list[str] = ["1h", "4h", "1d"],
    days: int = 365,
    sync_type: str = "merged",
    sync_version: str = "base",
    normalized: bool = True,
    epochs: int = 50,
    batch_size: int = 32
):
    """
    Fetch normalized replay data AND train the model in one step.
    Uses the SAME logic as /replay/run to get the data.
    Loads from cache if exists, otherwise fetches and saves to cache.
    """
    global _laplace_model
    
    try:
        from datetime import datetime, timedelta, timezone
        
        # Check if cache file exists and is not empty
        cache_loaded = False
        if os.path.exists(TRAIN_CACHE_FILE) and os.path.getsize(TRAIN_CACHE_FILE) > 0:
            logger.info(f"Loading cached training data from {TRAIN_CACHE_FILE}")
            with open(TRAIN_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
            
            X_train = np.array(cache_data['X_train'])
            results = cache_data['results']
            logger.info(f"Loaded {len(X_train)} samples from cache")
            cache_loaded = True
        else:
            # Calculate since/until from days
            now = datetime.now(timezone.utc)
            until = now
            since = now - timedelta(days=days)
            
            since_str = since.isoformat().replace("+00:00", "Z")
            until_str = until.isoformat().replace("+00:00", "Z")
            
            logger.info(f"Fetching normalized data: {symbol} {since_str} to {until_str}")
            
            # Fetch 1h data (same as replay)
            raw_1h = await run_in_threadpool(
                HISTORICAL.fetch,
                symbol=symbol,
                timeframe="1h",
                since=since_str,
                until=until_str,
            )
            
            warmup = settings.replay_indicators_warmup
            
            if len(raw_1h) <= warmup:
                raise HTTPException(
                    status_code=400,
                    detail=f"Datos insuficientes. Se requiere un warmup de {warmup} velas."
                )
            
            # Use the SAME MarketReplayEngine as /replay/run
            engine = MarketReplayEngine(data_1h=raw_1h, warmup_size=warmup)
            
            results = await run_in_threadpool(
                engine.run_replay,
                sync_type=sync_type,
                sync_version=sync_version,
                normalized=normalized
            )
            
            # If not loaded from cache, build training sequences
        if not cache_loaded:
            # Feature columns
            feature_cols = [
                'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'progress_vela_1h',
                'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'progress_vela_4h',
                'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'progress_vela_1d',
                'precio_actual', 'tiempo_normalizado'
            ]
            
            train_sequences = []
            for i in range(len(results) - DEFAULT_WINDOW_SIZE + 1):
                window = results[i:i + DEFAULT_WINDOW_SIZE]
                sequence = []
                for step in window:
                    row_features = []
                    for col in feature_cols:
                        val = step.get(col, 0.0)
                        if val is None:
                            val = 0.0
                        row_features.append(float(val))
                    sequence.append(row_features)
                train_sequences.append(sequence)
            
            X_train = np.array(train_sequences)
        
        # Save to cache only if we fetched fresh data (not loaded from cache)
        if not cache_loaded:
            os.makedirs(CACHE_DIR, exist_ok=True)
            cache_data = {
                'X_train': X_train.tolist(),
                'results': results,
                'metadata': {
                    'symbol': symbol,
                    'days': days,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            }
            with open(TRAIN_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f)
            
            logger.info(f"Cached {len(X_train)} samples to {TRAIN_CACHE_FILE}")
        
        logger.info(f"Prepared {len(X_train)} samples, shape {X_train.shape}")
        
        # Create model if not exists
        if _laplace_model is None:
            _laplace_model = _crear_modelo(ventana_tiempo=DEFAULT_WINDOW_SIZE, features=DEFAULT_FEATURES)
        
        # Train the model
        logger.info(f"Training laplace_v3 with {len(X_train)} samples, {epochs} epochs")
        
        history = _laplace_model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Save model
        _laplace_model.save(MODEL_PATH)
        
        # Save scaler
        scaler_path = os.path.join(BASE_DIR, 'escalador_f3.pkl')
        joblib.dump({"means": None, "stds": None}, scaler_path)
        
        logger.success(f"Laplace_v3 trained and saved to {MODEL_PATH}")
        
        return {
            "status": "trained",
            "model": "laplace_v3",
            "model_path": MODEL_PATH,
            "epochs": epochs,
            "batch_size": batch_size,
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "tensor_shape": list(X_train.shape),
            "results": results  # Same data as /replay/run
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Laplace train error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/predict")
async def laplace_predict(
    symbol: str = "BTC/USDT",
    timeframes: list[str] = ["1h", "4h", "1d"],
    days: int = 120,
    sync_type: str = "merged",
    sync_version: str = "base",
    normalized: bool = True
):
    """
    Predict using laplace_v3 with normalized replay data.
    Uses the trained model to reconstruct/predict.
    """
    global _laplace_model
    
    try:
        if _laplace_model is None:
            raise HTTPException(
                status_code=400,
                detail="Model not trained. Call /laplace/train first."
            )
        
        from datetime import datetime, timedelta, timezone
        
        now = datetime.now(timezone.utc)
        until = now
        since = now - timedelta(days=days)
        
        since_str = since.isoformat().replace("+00:00", "Z")
        until_str = until.isoformat().replace("+00:00", "Z")
        
        # Fetch and normalize using SAME logic as /replay/run
        raw_1h = await run_in_threadpool(
            HISTORICAL.fetch,
            symbol=symbol,
            timeframe="1h",
            since=since_str,
            until=until_str,
        )
        
        # Use smaller warmup for predict (just need enough for 24h window + some buffer)
        warmup = 30
        
        engine = MarketReplayEngine(data_1h=raw_1h, warmup_size=warmup)
        
        results = await run_in_threadpool(
            engine.run_replay,
            sync_type=sync_type,
            sync_version=sync_version,
            normalized=normalized
        )
        
        # Extract features
        feature_cols = [
            'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'progress_vela_1h',
            'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'progress_vela_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'progress_vela_1d',
            'precio_actual', 'tiempo_normalizado'
        ]
        
        # Get last 24 timesteps
        if len(results) < DEFAULT_WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {DEFAULT_WINDOW_SIZE} samples, got {len(results)}"
            )
        
        window = results[-DEFAULT_WINDOW_SIZE:]
        sequence = []
        for step in window:
            row_features = []
            for col in feature_cols:
                val = step.get(col, 0.0)
                if val is None:
                    val = 0.0
                row_features.append(float(val))
            sequence.append(row_features)
        
        X_input = np.array([sequence])

        def _predict_bottleneck_and_full():
            # Loaded Sequential models often have no `.input` until called; wire
            # the same layer instances through an explicit Input (weights stay shared).
            inp = keras.Input(shape=(DEFAULT_WINDOW_SIZE, DEFAULT_FEATURES))
            x = inp
            bottleneck = None
            for layer in _laplace_model.layers:
                x = layer(x)
                if layer.name == "capa_procesamiento":
                    bottleneck = x
            if bottleneck is None:
                raise ValueError(
                    'Model has no layer named "capa_procesamiento"; retrain or fix architecture.'
                )
            dual_model = keras.Model(inputs=inp, outputs=[bottleneck, x])
            return dual_model.predict(X_input, verbose=0)

        capa_procesamiento_output, final_output = await run_in_threadpool(
            _predict_bottleneck_and_full
        )
        
        mse = np.mean((X_input - final_output) ** 2)
        
        return {
            "status": "predicted",
            "model": "laplace_v3",
            "input_shape": list(X_input.shape),
            "input": X_input[0].tolist(),  # First sample
            "capa_procesamiento": {
                "shape": list(capa_procesamiento_output.shape),
                "features": capa_procesamiento_output[0].tolist()  # 16 features
            },
            "prediction": {
                "shape": list(final_output.shape),
                "data": final_output[0].tolist()  # 24 timesteps x 20 features
            },
            "reconstruction_error": float(mse),
            "anomaly_score": "high" if mse > 0.01 else "normal", 
            "timestamp": results[-1].get("timestamp") if results else None
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Laplace predict error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/status")
async def laplace_status():
    """Check model status."""
    return {
        "model_loaded": _laplace_model is not None,
        "window_size": DEFAULT_WINDOW_SIZE,
        "features_per_step": DEFAULT_FEATURES
    }