"""
Data endpoints — historical fetch and real-time latest candles.
Covers RF-3 (historical) and RF-4 (real-time) exposure via API.
"""

from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np

from app.api.schemas import HistoricalRequest
from app.config import settings
from app.core.data_ingestion.historical import HistoricalDataFetcher
from app.core.data_ingestion.realtime import RealTimeDataFetcher
from app.core.data_ingestion.replay_backtrader import BacktraderReplay
from app.core.processing.indicators import IndicatorEngine
from app.core.sync.multi_timeframe import MultiTimeframeSync

router = APIRouter(prefix="/data", tags=["Data Ingestion"])

_historical: HistoricalDataFetcher | None = None
_realtime = RealTimeDataFetcher()
_sync = MultiTimeframeSync()


OHLCV_COLS = ["open", "high", "low", "close", "volume"]
PROGRESS_COLS = ["progress_vela"]
TIMEFRAMES = ["1h", "4h", "1d"]

VELOCIDAD_INDICATORS = [
    "progress_vela", "volume",
    "MON", "ROC", "SQZ_MOM", "RSI_6", "RSI_14", "RSI_24", "RSI_EMA_6", "RSI_EMA_14", "RSI_EMA_24",
    "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI"
]
TENDENCIA_INDICATORS = [
    "MACD_LINE", "MACD_SIGNAL", "MACD_HIST", "ADX", "DI_PLUS", "DI_MINUS",
    "EMA_7", "EMA_22", "EMA_99", "ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN", "ICHIMOKU_SA", "ICHIMOKU_SB", "ICHIMOKU_CHIKOU"
]
AMPLITUD_INDICATORS = ["BB_UPPER", "BB_MIDDLE", "BB_LOWER", "BB_WIDTH", "KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"]
LIQUIDEZ_INDICATORS = ["CMF", "OBV", "ELDER_BULL", "ELDER_BEAR", "EOM", "VWAP"]

SEMANTIC_GROUPS = {
    "velocidad": VELOCIDAD_INDICATORS,
    "tendencia": TENDENCIA_INDICATORS,
    "amplitud": AMPLITUD_INDICATORS,
    "liquidez": LIQUIDEZ_INDICATORS,
}


def _extract_group(data_1h: dict, data_4h: dict, data_1d: dict, group_name: str) -> dict:
    """Extract indicators for a semantic group from the three timeframe data."""
    if group_name not in SEMANTIC_GROUPS:
        return {}
    
    indicators = SEMANTIC_GROUPS[group_name]
    result = {}
    
    for indicator in indicators:
        indicator_data = {}
        
        key_1h = f"{indicator}_1h" if not indicator.endswith("_1h") else indicator
        for key in data_1h:
            if key == indicator or key == key_1h:
                indicator_data["1h"] = data_1h[key]
                break
        
        key_4h = f"{indicator}_4h" if not indicator.endswith("_4h") else indicator
        for key in data_4h:
            if key == indicator or key == key_4h:
                indicator_data["4h"] = data_4h[key]
                break
        
        key_1d = f"{indicator}_1d" if not indicator.endswith("_1d") else indicator
        for key in data_1d:
            if key == indicator or key == key_1d:
                indicator_data["1d"] = data_1d[key]
                break
        
        if indicator_data:
            result[indicator] = indicator_data
    
    return result


def _filter_columns_by_version(df: pd.DataFrame, sync_version: str) -> pd.DataFrame:
    """Filter columns based on sync_version: base, ohlcv, or indicators."""
    all_cols = set(df.columns)
    
    ohlcv_cols = {f"{col}_{tf}" for col in OHLCV_COLS for tf in TIMEFRAMES}
    ohlcv_cols.update({col for col in all_cols if col in OHLCV_COLS})
    
    progress_cols = {f"{col}_{tf}" for col in PROGRESS_COLS for tf in TIMEFRAMES}
    progress_cols.update({col for col in all_cols if col in PROGRESS_COLS})
    
    indicator_cols = all_cols - ohlcv_cols - progress_cols - {"timestamp", "tf", "precio_actual", "tiempo_normalizado"}
    
    if sync_version == "base":
        keep_cols = {"timestamp"} | ohlcv_cols | progress_cols
    elif sync_version == "ohlcv":
        keep_cols = all_cols
    elif sync_version == "indicators":
        keep_cols = {"timestamp"} | progress_cols | indicator_cols
    else:
        keep_cols = all_cols
    
    return df[[c for c in df.columns if c in keep_cols]]


def _format_timeframe(steps: list[dict]) -> list[dict]:
    """Format steps as nested by timeframe: {1h: {}, 4h: {}, 1d: {}}."""
    result = []
    for step in steps:
        new_step = {}
        for tf in TIMEFRAMES:
            tf_data = {}
            for key, value in step.items():
                if key.endswith(f"_{tf}"):
                    tf_key = key.replace(f"_{tf}", "")
                    tf_data[tf_key] = value
                elif key == "timestamp":
                    tf_data[key] = value
                elif key == f"progress_vela" and tf == "1h":
                    tf_data[key] = value
            new_step[tf] = tf_data
        result.append(new_step)
    return result


def _format_semantic(steps: list[dict], sync_version: str = "ohlcv") -> list[dict]:
    """Format steps as nested by semantic group: {velocidad: {}, tendencia: {}, amplitud: {}, liquidez: {}}."""
    include_ohlc = sync_version == "ohlcv"
    
    result = []
    for step in steps:
        new_step = {}
        
        if include_ohlc:
            new_step["ohlcv"] = {}
            for tf in TIMEFRAMES:
                ohlc_data = {}
                for col in OHLCV_COLS + ["progress_vela"]:
                    key = f"{col}_{tf}"
                    if key in step:
                        ohlc_data[col] = step[key]
                new_step["ohlcv"][tf] = ohlc_data
        
        for group_name, indicators in SEMANTIC_GROUPS.items():
            group_data = {}
            for indicator in indicators:
                indicator_data = {}
                for tf in TIMEFRAMES:
                    key = f"{indicator}_{tf}"
                    if key in step:
                        indicator_data[tf] = step[key]
                if indicator_data:
                    group_data[indicator] = indicator_data
            new_step[group_name] = group_data
        
        if "timestamp" in step:
            new_step["timestamp"] = step["timestamp"]
        
        result.append(new_step)
    return result


def _get_historical() -> HistoricalDataFetcher:
    """Lazy initialization of historical fetcher."""
    global _historical
    if _historical is None:
        _historical = HistoricalDataFetcher()
    return _historical


def _build_timeframes_from_1h(df_1h: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build 4h and 1d DataFrames from 1h data (same logic as replay)."""
    buffer_1h = df_1h.copy()
    buffer_1h["progress_vela"] = 1.0
    
    buffer_4h = _build_warmup_timeframes(buffer_1h, "4h")
    buffer_1d = _build_warmup_timeframes(buffer_1h, "1d")
    
    return {
        "1h": buffer_1h,
        "4h": buffer_4h,
        "1d": buffer_1d,
    }


def _build_warmup_timeframes(df_1h: pd.DataFrame, target: str) -> pd.DataFrame:
    """Build 4h or 1d from 1h data (same as replay_backtrader)."""
    compression = 4 if target == "4h" else 24
    rows = []
    
    for i in range(compression, len(df_1h) + 1, compression):
        chunk = df_1h.iloc[i - compression:i]
        if chunk.empty:
            continue
        rows.append({
            "timestamp": chunk["timestamp"].iloc[-1],
            "open": chunk["open"].iloc[0],
            "high": chunk["high"].max(),
            "low": chunk["low"].min(),
            "close": chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
            "progress_vela": 1.0,
        })
    
    return pd.DataFrame(rows)


def _has_timeframe_suffix(col: str) -> bool:
    return col.endswith("_1h") or col.endswith("_4h") or col.endswith("_1d")


def _build_steps_from_full_data(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_1d: pd.DataFrame,
    limit: int,
    sync_type: str,
    sync_version: str
) -> list[dict]:
    """Build steps from all available data (not replay logic).
    
    Uses all data to build timeframes, calculate indicators, then maps to progressive steps.
    """
    total_1h = len(df_1h)
    total_4h = len(df_4h)
    total_1d = len(df_1d)
    
    buffer_tensor = []
    
    for step_idx in range(total_1h - limit, total_1h):
        step_in_4h = step_idx % 4
        step_in_1d = step_idx % 24
        
        progress_1h = 1.0
        progress_4h = (step_in_4h + 1) / 4.0
        progress_1d = (step_in_1d + 1) / 24.0
        
        row_1h = df_1h.iloc[step_idx]
        ts = row_1h["timestamp"]
        
        data_1h = {"timestamp": ts}
        for col in df_1h.columns:
            if col == "timestamp":
                continue
            if _has_timeframe_suffix(col) or col in OHLCV_COLS or col in PROGRESS_COLS:
                data_1h[f"{col}_1h"] = row_1h[col]
            else:
                data_1h[col] = row_1h[col]
        data_1h["progress_vela_1h"] = progress_1h
        
        idx_4h = step_idx // 4
        if idx_4h < total_4h:
            data_4h = df_4h.iloc[idx_4h].to_dict()
        else:
            data_4h = {"progress_vela_4h": progress_4h}
        data_4h["progress_vela_4h"] = progress_4h
        
        idx_1d = step_idx // 24
        if idx_1d < total_1d:
            data_1d = df_1d.iloc[idx_1d].to_dict()
        else:
            data_1d = {"progress_vela_1d": progress_1d}
        data_1d["progress_vela_1d"] = progress_1d
        
        step = {"timestamp": ts}
        
        if sync_type == "merged":
            for k, v in data_1h.items():
                step[k] = v
            for k, v in data_4h.items():
                if k != "timestamp":
                    step[k] = v
            for k, v in data_1d.items():
                if k != "timestamp":
                    step[k] = v
        elif sync_type == "timeframe":
            step["1h"] = {k: v for k, v in data_1h.items() if k != "timestamp"}
            step["4h"] = {k: v for k, v in data_4h.items() if k != "timestamp"}
            step["1d"] = {k: v for k, v in data_1d.items() if k != "timestamp"}
        elif sync_type == "semantic":
            step["velocidad"] = {}
            step["tendencia"] = {}
            step["amplitud"] = {}
            step["liquidez"] = {}
            for k, v in data_1h.items():
                for group, indicators in SEMANTIC_GROUPS.items():
                    for ind in indicators:
                        if k == ind or k.startswith(f"{ind}_"):
                            step[group][k] = v
                            break
            if sync_version == "ohlcv":
                step["ohlcv"] = {
                    "1h": {"open": data_1h.get("open_1h"), "high": data_1h.get("high_1h"), "low": data_1h.get("low_1h"), "close": data_1h.get("close_1h"), "volume": data_1h.get("volume_1h"), "progress_vela": data_1h.get("progress_vela_1h")},
                    "4h": {"open": data_4h.get("open"), "high": data_4h.get("high"), "low": data_4h.get("low"), "close": data_4h.get("close"), "volume": data_4h.get("volume"), "progress_vela": data_4h.get("progress_vela_4h")},
                    "1d": {"open": data_1d.get("open"), "high": data_1d.get("high"), "low": data_1d.get("low"), "close": data_1d.get("close"), "volume": data_1d.get("volume"), "progress_vela": data_1d.get("progress_vela_1d")},
                }
        
        buffer_tensor.append(step)
    
    return buffer_tensor


def _build_steps_from_replay_logic(
    df_1h_full: pd.DataFrame,
    df_4h_warmup: pd.DataFrame,
    df_1d_warmup: pd.DataFrame,
    limit: int,
    sync_type: str,
    sync_version: str,
    warmup_end: int
) -> list[dict]:
    """Build steps using exactly the same logic as BacktraderReplay.
    
    1. Use warmup data (first N rows) for indicators
    2. Simulate from warmup_end onwards, building 4h/1d progressively
    3. Store only last 'limit' steps in buffer_tensor
    4. Apply sync during construction
    """
    total_1h = len(df_1h_full)
    max_steps = total_1h - warmup_end
    
    if max_steps <= 0:
        return []
    
    df_1h_warmup = df_1h_full.iloc[:warmup_end].reset_index(drop=True)
    df_1h_replay = df_1h_full.iloc[warmup_end:].reset_index(drop=True)
    
    buffers_4h = df_4h_warmup.reset_index(drop=True)
    buffers_1d = df_1d_warmup.reset_index(drop=True)
    
    current_4h = None
    current_1d = None
    buffer_tensor = []
    
    for step_idx in range(len(df_1h_replay)):
        step_in_4h = step_idx % 4
        step_in_1d = step_idx % 24
        
        progress_1h = 1.0
        progress_4h = (step_in_4h + 1) / 4.0
        progress_1d = (step_in_1d + 1) / 24.0
        
        row_1h = df_1h_replay.iloc[step_idx]
        ts = row_1h["timestamp"]
        
        data_1h = {
            "timestamp": ts,
            "open": row_1h["open"],
            "high": row_1h["high"],
            "low": row_1h["low"],
            "close": row_1h["close"],
            "volume": row_1h["volume"],
            "progress_vela": progress_1h,
        }
        for col in df_1h_warmup.columns:
            if col not in ["timestamp", "open", "high", "low", "close", "volume", "progress_vela"]:
                if _has_timeframe_suffix(col):
                    data_1h[col] = row_1h[col]
                elif col in ["tf", "precio_actual", "tiempo_normalizado"]:
                    pass
                else:
                    data_1h[f"{col}_1h"] = row_1h[col]
        
        if current_4h is None:
            current_4h = {
                "timestamp": ts,
                "open": row_1h["open"],
                "high": row_1h["high"],
                "low": row_1h["low"],
                "close": row_1h["close"],
                "volume": row_1h["volume"],
                "progress_vela": progress_4h,
            }
        else:
            current_4h["high"] = max(current_4h["high"], row_1h["high"])
            current_4h["low"] = min(current_4h["low"], row_1h["low"])
            current_4h["close"] = row_1h["close"]
            current_4h["volume"] += row_1h["volume"]
            current_4h["progress_vela"] = progress_4h
            current_4h["timestamp"] = ts
        
        if step_in_4h == 3:
            current_4h["progress_vela"] = 1.0
            data_4h = current_4h.copy()
            idx_4h = step_idx // 4
            if idx_4h < len(buffers_4h):
                row_4h = buffers_4h.iloc[idx_4h]
                for col in row_4h.index:
                    if col != "timestamp":
                        data_4h[col] = row_4h[col]
            current_4h = None
        else:
            data_4h = current_4h.copy()
        
        if current_1d is None:
            current_1d = {
                "timestamp": ts,
                "open": row_1h["open"],
                "high": row_1h["high"],
                "low": row_1h["low"],
                "close": row_1h["close"],
                "volume": row_1h["volume"],
                "progress_vela": progress_1d,
            }
        else:
            current_1d["high"] = max(current_1d["high"], row_1h["high"])
            current_1d["low"] = min(current_1d["low"], row_1h["low"])
            current_1d["close"] = row_1h["close"]
            current_1d["volume"] += row_1h["volume"]
            current_1d["progress_vela"] = progress_1d
            current_1d["timestamp"] = ts
        
        if step_in_1d == 23:
            current_1d["progress_vela"] = 1.0
            data_1d = current_1d.copy()
            idx_1d = step_idx // 24
            if idx_1d < len(buffers_1d):
                row_1d = buffers_1d.iloc[idx_1d]
                for col in row_1d.index:
                    if col != "timestamp":
                        data_1d[col] = row_1d[col]
            current_1d = None
        else:
            data_1d = current_1d.copy()
        
        step = {"timestamp": ts}
        
        if sync_type == "merged":
            for k, v in data_1h.items():
                step[k] = v
            for k, v in data_4h.items():
                if k != "timestamp":
                    if _has_timeframe_suffix(k):
                        step[k] = v
                    else:
                        step[f"{k}_4h"] = v
            for k, v in data_1d.items():
                if k != "timestamp":
                    if _has_timeframe_suffix(k):
                        step[k] = v
                    else:
                        step[f"{k}_1d"] = v
        elif sync_type == "timeframe":
            step["1h"] = data_1h
            step["4h"] = data_4h
            step["1d"] = data_1d
        elif sync_type == "semantic":
            step["velocidad"] = _extract_group(data_1h, data_4h, data_1d, "velocidad")
            step["tendencia"] = _extract_group(data_1h, data_4h, data_1d, "tendencia")
            step["amplitud"] = _extract_group(data_1h, data_4h, data_1d, "amplitud")
            step["liquidez"] = _extract_group(data_1h, data_4h, data_1d, "liquidez")
            if sync_version == "ohlcv":
                step["ohlcv"] = {
                    "1h": {k: v for k, v in data_1h.items() if k in ["open", "high", "low", "close", "volume", "progress_vela"]},
                    "4h": {k: v for k, v in data_4h.items() if k in ["open", "high", "low", "close", "volume", "progress_vela"]},
                    "1d": {k: v for k, v in data_1d.items() if k in ["open", "high", "low", "close", "volume", "progress_vela"]},
                }
        
        buffer_tensor.append(step)
        
        if len(buffer_tensor) > limit:
            buffer_tensor.pop(0)
    
    return buffer_tensor


def _build_4h_from_1h_with_progress(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build 4h from 1h with progressive progress_vela per step (like replay)."""
    rows = []
    current_4h = None
    
    for step_idx in range(len(df_1h)):
        step_in_4h = step_idx % 4
        progress_4h = (step_in_4h + 1) / 4.0
        
        current_row = df_1h.iloc[step_idx]
        
        if current_4h is None:
            current_4h = {
                "timestamp": current_row["timestamp"],
                "open": current_row["open"],
                "high": current_row["high"],
                "low": current_row["low"],
                "close": current_row["close"],
                "volume": current_row["volume"],
                "progress_vela": progress_4h,
            }
        else:
            current_4h["high"] = max(current_4h["high"], current_row["high"])
            current_4h["low"] = min(current_4h["low"], current_row["low"])
            current_4h["close"] = current_row["close"]
            current_4h["volume"] += current_row["volume"]
            current_4h["progress_vela"] = progress_4h
            current_4h["timestamp"] = current_row["timestamp"]
        
        if step_in_4h == 3:
            current_4h["progress_vela"] = 1.0
            rows.append(current_4h.copy())
            current_4h = None
        else:
            rows.append(current_4h.copy())
    
    return pd.DataFrame(rows)


def _build_1d_from_1h_with_progress(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Build 1d from 1h with progressive progress_vela per step (like replay)."""
    rows = []
    current_1d = None
    
    for step_idx in range(len(df_1h)):
        step_in_1d = step_idx % 24
        progress_1d = (step_in_1d + 1) / 24.0
        
        current_row = df_1h.iloc[step_idx]
        
        if current_1d is None:
            current_1d = {
                "timestamp": current_row["timestamp"],
                "open": current_row["open"],
                "high": current_row["high"],
                "low": current_row["low"],
                "close": current_row["close"],
                "volume": current_row["volume"],
                "progress_vela": progress_1d,
            }
        else:
            current_1d["high"] = max(current_1d["high"], current_row["high"])
            current_1d["low"] = min(current_1d["low"], current_row["low"])
            current_1d["close"] = current_row["close"]
            current_1d["volume"] += current_row["volume"]
            current_1d["progress_vela"] = progress_1d
            current_1d["timestamp"] = current_row["timestamp"]
        
        if step_in_1d == 23:
            current_1d["progress_vela"] = 1.0
            rows.append(current_1d.copy())
            current_1d = None
        else:
            rows.append(current_1d.copy())
    
    return pd.DataFrame(rows)


def _add_global_features_to_buffers(buffers: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Add global features (precio_actual, tiempo_normalizado) to each timeframe."""
    base_len = len(buffers["1h"])
    
    for tf in ["1h", "4h", "1d"]:
        df = buffers[tf].copy()
        
        df["precio_actual"] = df["close"]
        
        total_steps = len(buffers["1h"])
        df["tiempo_normalizado"] = [i / total_steps for i in range(len(df))]
        
        buffers[tf] = df
    
    return buffers


def _merge_timeframes_to_steps(df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame) -> list[dict]:
    """Merge 3 timeframes into steps, matching by index position."""
    steps = []
    min_len = min(len(df_1h), len(df_4h), len(df_1d))
    
    def _add_suffix(col: str, suffix: str) -> str:
        if col.endswith(suffix):
            return col
        return f"{col}{suffix}"
    
    for i in range(min_len):
        step = {}
        
        for col in df_1h.columns:
            if col == "timestamp":
                step["timestamp"] = df_1h.iloc[i][col]
            elif col in OHLCV_COLS or col == "progress_vela":
                step[_add_suffix(col, "_1h")] = df_1h.iloc[i][col]
            elif col not in ["tf", "precio_actual", "tiempo_normalizado"]:
                step[_add_suffix(col, "_1h")] = df_1h.iloc[i][col]
        
        for col in df_4h.columns:
            if col == "timestamp":
                continue
            elif col in OHLCV_COLS or col == "progress_vela":
                step[_add_suffix(col, "_4h")] = df_4h.iloc[i][col]
            elif col not in ["tf", "precio_actual", "tiempo_normalizado"]:
                step[_add_suffix(col, "_4h")] = df_4h.iloc[i][col]
        
        for col in df_1d.columns:
            if col == "timestamp":
                continue
            elif col in OHLCV_COLS or col == "progress_vela":
                step[_add_suffix(col, "_1d")] = df_1d.iloc[i][col]
            elif col not in ["tf", "precio_actual", "tiempo_normalizado"]:
                step[_add_suffix(col, "_1d")] = df_1d.iloc[i][col]
        
        steps.append(step)
    
    return steps


def _filter_to_base(steps: list[dict]) -> list[dict]:
    """Filter to only OHLCVP columns."""
    ohlcvp_suffixes = ["_1h", "_4h", "_1d"]
    base_cols = ["open", "high", "low", "close", "volume", "progress_vela"]
    
    result = []
    for step in steps:
        filtered = {}
        for key in step:
            base = key.rsplit("_", 1)[0] if "_" in key else key
            if base in base_cols:
                filtered[key] = step[key]
        result.append(filtered)
    
    return result


def _filter_to_indicators(steps: list[dict]) -> list[dict]:
    """Filter to only indicator columns (no OHLCV)."""
    base_cols = ["open", "high", "low", "close", "volume", "progress_vela", "timestamp", "tf", "precio_actual", "tiempo_normalizado"]
    
    result = []
    for step in steps:
        filtered = {"timestamp": step.get("timestamp")}
        for key, value in step.items():
            base = key.rsplit("_", 1)[0] if "_" in key else key
            if base not in base_cols and not key.startswith("progress"):
                filtered[key] = value
        result.append(filtered)
    
    return result


@router.post("/historical")
def fetch_historical(req: HistoricalRequest):
    """Fetch historical OHLCV data, optionally with technical indicators.

    If include_indicators=True, appends ~40 indicators per timeframe
    (VELOCIDAD, TENDENCIA, AMPLITUD, LIQUIDEZ groups).
    NaN values appear where the lookback window is insufficient.
    """
    try:
        result = _get_historical().fetch_multi_timeframe(
            symbol=req.symbol,
            timeframes=req.timeframes,
            since=req.since,
            until=req.until,
        )
        if req.include_indicators:
            result = IndicatorEngine.compute_multi_timeframe(result)
        return {
            tf: (
                df.replace({float("nan"): None})
                .to_dict(orient="records")
            )
            for tf, df in result.items()
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/realtime")
def fetch_realtime(
    symbol: str = "BTC/USDT",
    limit: int = 30,
    sync_type: str = "timeframe",
    sync_version: str = "ohlcv",
    return_tensor: bool = False,
):
    """Fetch latest N steps with 4h and 1d reconstructed from 1h.

    Uses replay_indicators_warmup (2400) candles to calculate indicators
    and reconstruct higher timeframes with progressive progress_vela,
    then returns the last N steps (like replay stream).

    Parameters:
        symbol: Trading pair (default: BTC/USDT)
        limit: Number of steps to return (default: 30)
        sync_type: timeframe | merged | semantic (default: timeframe)
        sync_version: ohlcv | indicators | base (default: ohlcv)
        return_tensor: Return tensor format instead of JSON (default: false)
    """
    try:
        lookback = settings.replay_indicators_warmup
        
        result = _get_historical().fetch_multi_timeframe(
            symbol=symbol,
            timeframes=["1h"],
            since=None,
            until=None,
        )
        raw_1h = result["1h"]
        
        if len(raw_1h) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {len(raw_1h)} rows. Need at least 100."
            )
        
        raw_1h = raw_1h.reset_index(drop=True)
        
        df_1h = raw_1h.copy()
        df_1h["progress_vela"] = 1.0
        
        warmup_end = min(settings.replay_indicators_warmup, len(df_1h))
        df_1h_warmup = df_1h.iloc[:warmup_end]
        
        df_4h = _build_warmup_timeframes(df_1h_warmup, "4h")
        df_1d = _build_warmup_timeframes(df_1h_warmup, "1d")
        
        if sync_version != "base":
            buffers = {
                "1h": df_1h_warmup,
                "4h": df_4h,
                "1d": df_1d,
            }
            buffers = IndicatorEngine.compute_multi_timeframe(buffers)
            df_1h = buffers["1h"]
            df_4h = buffers["4h"]
            df_1d = buffers["1d"]
        
        last_n_steps = _build_steps_from_full_data(
            df_1h, df_4h, df_1d, limit, sync_type, sync_version
        )
        
        last_n_steps = [{k: (None if isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')) else v) for k, v in step.items()} for step in last_n_steps]
        
        steps_dict = last_n_steps
        
        if return_tensor:
            all_keys = set()
            for step in steps_dict:
                all_keys.update(step.keys())
            
            tensor_3d = []
            for step in steps_dict:
                row = [step.get(k, None) for k in sorted(all_keys)]
                tensor_3d.append(row)
            
            return {
                "window_size": limit,
                "sync_type": sync_type,
                "sync_version": sync_version,
                "tensor": tensor_3d,
                "columns": sorted(list(all_keys)),
                "shape": [1, limit, len(all_keys)],
            }
        
        return {
            "steps": steps_dict,
            "summary": {
                "total_steps": len(steps_dict),
                "sync_type": sync_type,
                "sync_version": sync_version,
            }
        }
        
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/ultimas_velas")
def fetch_ultimas_velas(
    symbol: str = "BTC/USDT",
    limit: int = 30,
    include_indicators: bool = False,
    timeframes: str = "1h,4h,1d",
):
    """Fetch the latest *limit* candles across all default timeframes.
    
    Optional: include_indicators=True to append technical indicators (VELOCIDAD,
    TENDENCIA, AMPLITUD, LIQUIDEZ groups).
    """
    try:
        tf_list = [t.strip() for t in timeframes.split(",")]
        result = _realtime.fetch_latest_multi_timeframe(
            symbol=symbol, limit=limit, timeframes=tf_list
        )
        if include_indicators:
            result = IndicatorEngine.compute_multi_timeframe(result)
            return {
                tf: (
                    df.replace({float("nan"): None})
                    .to_dict(orient="records")
                )
                for tf, df in result.items()
            }
        return {tf: df.to_dict(orient="records") for tf, df in result.items()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))