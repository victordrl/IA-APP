"""
Laplace model training endpoints.
Uses replay data to train the agent model (laplace_v3).
"""

import os
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

router = APIRouter(prefix="/laplace", tags=["Laplace Training"])

# Global model storage
_laplace_model = None
_training_data = None
_training_results = None  # Store raw results for reference

# Default config
DEFAULT_WINDOW_SIZE = 24  # 24 timesteps
DEFAULT_FEATURES = 18     # 18 features per timestep

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


def _generate_normalized_data(data_1h, warmup: int = 100) -> list[dict]:
    """Generate normalized replay data from 1h OHLCV using Z-score."""
    import pandas as pd
    
    df = data_1h.iloc[warmup:].copy().reset_index(drop=True)
    total_rows = len(df)
    
    # Z-score normalization
    ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
    means = df[ohlcv_cols].mean()
    stds = df[ohlcv_cols].std()
    normalized = (df[ohlcv_cols] - means) / stds
    
    results = []
    
    for i in range(DEFAULT_WINDOW_SIZE - 1, total_rows):
        curr = normalized.iloc[i]
        
        # 1h features (6)
        row = {
            "open_1h": float(curr['open']),
            "high_1h": float(curr['high']),
            "low_1h": float(curr['low']),
            "close_1h": float(curr['close']),
            "volume_1h": float(curr['volume']),
            "progress_vela_1h": 1.0,
        }
        
        # 4h features (every 4th candle)
        idx_4h = i // 4
        if idx_4h > 0:
            curr_4h = normalized.iloc[idx_4h]
            progress_4h = (idx_4h % DEFAULT_WINDOW_SIZE) / DEFAULT_WINDOW_SIZE
            row.update({
                "open_4h": float(curr_4h['open']),
                "high_4h": float(curr_4h['high']),
                "low_4h": float(curr_4h['low']),
                "close_4h": float(curr_4h['close']),
                "volume_4h": float(curr_4h['volume']),
                "progress_vela_4h": float(progress_4h),
            })
        else:
            row.update({f"{c}_4h": 0.0 for c in ['open', 'high', 'low', 'close', 'volume']})
            row["progress_vela_4h"] = 0.0
        
        # 1d features (every 24th candle)
        idx_1d = i // 24
        if idx_1d > 0:
            curr_1d = normalized.iloc[idx_1d]
            progress_1d = (idx_1d % DEFAULT_WINDOW_SIZE) / DEFAULT_WINDOW_SIZE
            row.update({
                "open_1d": float(curr_1d['open']),
                "high_1d": float(curr_1d['high']),
                "low_1d": float(curr_1d['low']),
                "close_1d": float(curr_1d['close']),
                "volume_1d": float(curr_1d['volume']),
                "progress_vela_1d": float(progress_1d),
            })
        else:
            row.update({f"{c}_1d": 0.0 for c in ['open', 'high', 'low', 'close', 'volume']})
            row["progress_vela_1d"] = 0.0
        
        # Global features
        if i > 0:
            prev = normalized.iloc[i-1]
            row["precio_actual"] = float(curr['close'] - prev['close'])
        else:
            row["precio_actual"] = 0.0
        
        row["tiempo_normalizado"] = float(i / total_rows)
        row["timestamp"] = str(df.iloc[i]["timestamp"])
        
        results.append(row)
    
    return results


@router.get("/current")
async def laplace_current():
    """Get current model info. Returns hardcoded list for now."""
    return {
        "models": [
            {
                "name": "laplace_v3",
                "version": "3.0",
                "window_size": 24,
                "features": 18,
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
    days: int = 120,
    sync_type: str = "merged",
    sync_version: str = "base",
    normalized: bool = True
):
    """
    Fetch and normalize replay data for training.
    Returns the normalized data (24 timesteps × 18 features per row).
    
    Call /fit after this to actually train the model.
    """
    global _training_data, _training_results
    
    try:
        from datetime import datetime, timedelta, timezone
        
        # Calculate since/until from days
        now = datetime.now(timezone.utc)
        until = now
        since = now - timedelta(days=days)
        
        since_str = since.isoformat().replace("+00:00", "Z")
        until_str = until.isoformat().replace("+00:00", "Z")
        
        logger.info(f"Fetching normalized data: {symbol} {since_str} to {until_str}")
        
        # Fetch 1h data
        raw_1h = await run_in_threadpool(
            HISTORICAL.fetch,
            symbol=symbol,
            timeframe="1h",
            since=since_str,
            until=until_str,
        )
        
        warmup = settings.replay_indicators_warmup
        
        if len(raw_1h) <= warmup + DEFAULT_WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data. Need at least {warmup + DEFAULT_WINDOW_SIZE} rows, got {len(raw_1h)}"
            )
        
        # Generate normalized data
        results = await run_in_threadpool(
            _generate_normalized_data,
            raw_1h,
            warmup
        )
        
        # Prepare training sequences (24 timesteps each)
        feature_cols = [
            'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'progress_vela_1h',
            'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'progress_vela_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'progress_vela_1d',
            'precio_actual', 'tiempo_normalizado'
        ]
        
        X = np.array([[r[c] for c in feature_cols] for r in results])
        
        # Create sequences of 24 timesteps
        train_sequences = []
        for i in range(len(X) - DEFAULT_WINDOW_SIZE + 1):
            train_sequences.append(X[i:i + DEFAULT_WINDOW_SIZE])
        
        X_train = np.array(train_sequences)
        
        # Store for fit
        _training_data = X_train
        _training_results = results
        
        logger.info(f"Prepared {len(X_train)} samples, shape {X_train.shape}")
        
        return {
            "status": "ready",
            "model": "laplace_v3",
            "window_size": DEFAULT_WINDOW_SIZE,
            "features_per_step": DEFAULT_FEATURES,
            "total_samples": len(X_train),
            "tensor_shape": list(X_train.shape),
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Laplace train error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/fit")
async def laplace_fit(epochs: int = 10, batch_size: int = 32):
    """
    Fit/Train the model using data from /train endpoint.
    Creates model if not exists, then trains.
    """
    global _laplace_model, _training_data
    
    try:
        if _training_data is None:
            raise HTTPException(
                status_code=400,
                detail="No training data. Call /laplace/train first."
            )
        
        X_train = _training_data
        
        # Create model if not exists
        if _laplace_model is None:
            _laplace_model = _crear_modelo(ventana_tiempo=DEFAULT_WINDOW_SIZE, features=DEFAULT_FEATURES)
        
        logger.info(f"Training laplace_v3 with {len(X_train)} samples, {epochs} epochs")
        
        history = _laplace_model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1
        )
        
        # Save model
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(base_dir, 'laplace_f3.keras')
        _laplace_model.save(model_path)
        
        # Save scaler
        scaler_path = os.path.join(base_dir, 'escalador_f3.pkl')
        joblib.dump({"means": None, "stds": None}, scaler_path)
        
        # Clear training data
        _training_data = None
        
        logger.success(f"Laplace_v3 trained and saved to {model_path}")
        
        return {
            "status": "trained",
            "model": "laplace_v3",
            "model_path": model_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "final_loss": float(history.history['loss'][-1]),
            "final_val_loss": float(history.history['val_loss'][-1]),
            "tensor_shape": list(X_train.shape)
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Laplace fit error: {exc}")
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
                detail="Model not trained. Call /laplace/train and /laplace/fit first."
            )
        
        from datetime import datetime, timedelta, timezone
        
        now = datetime.now(timezone.utc)
        until = now
        since = now - timedelta(days=days)
        
        since_str = since.isoformat().replace("+00:00", "Z")
        until_str = until.isoformat().replace("+00:00", "Z")
        
        raw_1h = await run_in_threadpool(
            HISTORICAL.fetch,
            symbol=symbol,
            timeframe="1h",
            since=since_str,
            until=until_str,
        )
        
        warmup = settings.replay_indicators_warmup
        
        results = await run_in_threadpool(
            _generate_normalized_data,
            raw_1h,
            warmup
        )
        
        # Prepare input
        feature_cols = [
            'open_1h', 'high_1h', 'low_1h', 'close_1h', 'volume_1h', 'progress_vela_1h',
            'open_4h', 'high_4h', 'low_4h', 'close_4h', 'volume_4h', 'progress_vela_4h',
            'open_1d', 'high_1d', 'low_1d', 'close_1d', 'volume_1d', 'progress_vela_1d',
            'precio_actual', 'tiempo_normalizado'
        ]
        
        X = np.array([[r[c] for c in feature_cols] for r in results])
        
        if len(X) < DEFAULT_WINDOW_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least {DEFAULT_WINDOW_SIZE} samples, got {len(X)}"
            )
        
        X_input = X[-DEFAULT_WINDOW_SIZE:].reshape(1, DEFAULT_WINDOW_SIZE, DEFAULT_FEATURES)
        
        prediction = _laplace_model.predict(X_input, verbose=0)
        
        mse = np.mean((X_input - prediction) ** 2)
        
        return {
            "status": "predicted",
            "model": "laplace_v3",
            "input_shape": list(X_input.shape),
            "prediction_shape": list(prediction.shape),
            "reconstruction_error": float(mse),
            "anomaly_score": "high" if mse > 0.01 else "normal",
            "timestamp": results[-1]["timestamp"] if results else None
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
        "training_data_ready": _training_data is not None,
        "window_size": DEFAULT_WINDOW_SIZE,
        "features_per_step": DEFAULT_FEATURES
    }