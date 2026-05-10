import pandas as pd
from typing import Dict, Any, List
from collections import deque
from loguru import logger

from app.core.processing.indicators import IndicatorEngine
from app.core.sync.multi_timeframe import MultiTimeframeSync


class MarketEngineBase:
    """
    Base logic for market data reconstruction and synchronization.
    Recalculates progressive 4h and 1d candles from 1h data.
    """

    def __init__(self, data_1h: pd.DataFrame, warmup_size: int = 2400):
        self.data_1h = data_1h.copy()
        if "timestamp" in self.data_1h.columns:
            self.data_1h = self.data_1h.set_index("timestamp").sort_index()
        self.data_1h.index = pd.DatetimeIndex(self.data_1h.index)

        self.warmup_size = warmup_size
        self.total_rows = len(self.data_1h)

        if self.total_rows <= self.warmup_size:
            raise ValueError(f"Not enough data. Needed > {self.warmup_size}, got {self.total_rows}")

        # Build initial buffers (Warmup)
        self.buffer_1h = self.data_1h.iloc[:self.warmup_size].copy().reset_index()
        self.buffer_1h["progress_vela"] = 1.0

        self.buffer_4h = self._build_warmup(self.buffer_1h, 4)
        self.buffer_1d = self._build_warmup(self.buffer_1h, 24)

        # Initialize current candles for the next steps
        first_step_idx = self.warmup_size
        first_row = self.data_1h.iloc[first_step_idx]
        first_ts = self.data_1h.index[first_step_idx]

        self.current_4h = self._init_candle(first_ts, first_row, 0.25)
        self.current_1d = self._init_candle(first_ts, first_row, 1 / 24.0)

        self.sync_engine = MultiTimeframeSync()

    def _init_candle(self, timestamp, row, progress) -> dict:
        return {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "progress_vela": progress
        }

    def _build_warmup(self, df_1h: pd.DataFrame, compression: int) -> pd.DataFrame:
        rows = []
        for i in range(compression, len(df_1h) + 1, compression):
            chunk = df_1h.iloc[i - compression:i]
            if chunk.empty:
                continue
            rows.append({
                "timestamp": chunk["timestamp"].iloc[-1],
                "open": float(chunk["open"].iloc[0]),
                "high": float(chunk["high"].max()),
                "low": float(chunk["low"].min()),
                "close": float(chunk["close"].iloc[-1]),
                "volume": float(chunk["volume"].sum()),
                "progress_vela": 1.0
            })
        return pd.DataFrame(rows)

    def process_step(self, step: int, sync_type: str, sync_version: str) -> dict:
        """
        Processes a single 1h step, updates 4h/1d progressive values, 
        calculates indicators, synchronizes and returns it.
        """
        data_idx = self.warmup_size + step
        ts = self.data_1h.index[data_idx]
        row = self.data_1h.iloc[data_idx]

        # 1. Update 1h Buffer
        new_1h = self._init_candle(ts, row, 1.0)
        self.buffer_1h = pd.concat([self.buffer_1h, pd.DataFrame([new_1h])], ignore_index=True)

        # 2. Update 4h Progress
        step_in_4h = step % 4
        self.current_4h["high"] = max(self.current_4h["high"], float(row["high"]))
        self.current_4h["low"] = min(self.current_4h["low"], float(row["low"]))
        self.current_4h["close"] = float(row["close"])
        self.current_4h["volume"] += float(row["volume"])
        self.current_4h["progress_vela"] = (step_in_4h + 1) / 4.0
        self.current_4h["timestamp"] = ts

        # 3. Update 1d Progress
        step_in_1d = step % 24
        self.current_1d["high"] = max(self.current_1d["high"], float(row["high"]))
        self.current_1d["low"] = min(self.current_1d["low"], float(row["low"]))
        self.current_1d["close"] = float(row["close"])
        self.current_1d["volume"] += float(row["volume"])
        self.current_1d["progress_vela"] = (step_in_1d + 1) / 24.0
        self.current_1d["timestamp"] = ts

        # 4. Build Temp Buffers (including in-progress candles)
        step_buf_1h = self.buffer_1h.copy()
        step_buf_4h = pd.concat([self.buffer_4h, pd.DataFrame([self.current_4h])], ignore_index=True)
        step_buf_1d = pd.concat([self.buffer_1d, pd.DataFrame([self.current_1d])], ignore_index=True)

        buffers_dict = {"1h": step_buf_1h, "4h": step_buf_4h, "1d": step_buf_1d}

        # 5. Compute Indicators
        with_indicators = IndicatorEngine.compute_multi_timeframe(buffers_dict)

        # 6. Synchronization
        synced = self.sync_engine.synchronize(
            with_indicators,
            sync_type=sync_type,
            sync_version=sync_version
        )
        synced = MultiTimeframeSync.add_global_features(synced)

        last_row = synced.iloc[-1].round(4).to_dict()

        # Clean NaN/Inf for JSON
        clean_row = {
            k: (None if pd.isna(v) or v == float('inf') or v == float('-inf') else v)
            for k, v in last_row.items()
        }

        # Make sure timestamp is firmly established at the root
        clean_row["timestamp"] = str(ts)
        
        formatted_row = self._format_nested(clean_row, sync_type)

        # 7. Close candles and create new ones if needed
        if step_in_4h == 3:
            self.buffer_4h = pd.concat([self.buffer_4h, pd.DataFrame([self.current_4h])], ignore_index=True)
            if data_idx + 1 < self.total_rows:
                next_ts = self.data_1h.index[data_idx + 1]
                next_row = self.data_1h.iloc[data_idx + 1]
                self.current_4h = self._init_candle(next_ts, next_row, 0.25)

        if step_in_1d == 23:
            self.buffer_1d = pd.concat([self.buffer_1d, pd.DataFrame([self.current_1d])], ignore_index=True)
            if data_idx + 1 < self.total_rows:
                next_ts = self.data_1h.index[data_idx + 1]
                next_row = self.data_1h.iloc[data_idx + 1]
                self.current_1d = self._init_candle(next_ts, next_row, 1 / 24.0)

        return formatted_row

    def _format_nested(self, flat_dict: dict, sync_type: str) -> dict:
        """Nests the flat dictionary based on the required sync type."""
        res = {"timestamp": flat_dict.get("timestamp")}
        
        if sync_type == "merged":
            return flat_dict
            
        elif sync_type == "timeframe":
            res["1h"] = {}
            res["4h"] = {}
            res["1d"] = {}
            for k, v in flat_dict.items():
                if k == "timestamp": continue
                if k.endswith("_1h"):
                    res["1h"][k.replace("_1h", "")] = v
                elif k.endswith("_4h"):
                    res["4h"][k.replace("_4h", "")] = v
                elif k.endswith("_1d"):
                    res["1d"][k.replace("_1d", "")] = v
                else:
                    res["1h"][k] = v  # Fallbacks
            return res
            
        elif sync_type == "semantic":
            GROUPS = MultiTimeframeSync.GROUPS
            ohlcv_cols = MultiTimeframeSync.OHLCV_COLS
            
            # Identify shared/global keys that must exist in every semantic group
            shared_keys = {}
            for k, v in flat_dict.items():
                if k == "timestamp": continue
                base_k = k.replace("_1h", "").replace("_4h", "").replace("_1d", "")
                if base_k in ohlcv_cols or k in ["precio_actual", "tiempo_normalizado"]:
                    shared_keys[k] = v

            # Force initialize the 4 exact categories
            for group in GROUPS.keys():
                res[group.lower()] = {}

            for k, v in flat_dict.items():
                if k == "timestamp" or k in shared_keys: continue
                
                base_k = k.replace("_1h", "").replace("_4h", "").replace("_1d", "")
                
                for group, inds in GROUPS.items():
                    if base_k in inds:
                        res[group.lower()][k] = v
                        break
            
            # Inject shared keys into the 4 semantic groups
            for group_key in res.keys():
                if group_key != "timestamp":
                    res[group_key].update(shared_keys)
                    
            return res
            
        return flat_dict


class MarketReplayEngine(MarketEngineBase):
    """Engine for unlimited market replay execution."""

    def run_replay(self, sync_type: str, sync_version: str) -> List[dict]:
        """
        Executes without interruptions and returns all steps 
        collected in an unbounded global buffer.
        """
        global_buffer = []
        total_steps_to_run = self.total_rows - self.warmup_size
        logger.info(f"Starting Replay for {total_steps_to_run} steps...")

        for step in range(total_steps_to_run):
            res = self.process_step(step, sync_type, sync_version)
            global_buffer.append(res)

            if step % 50 == 0:
                logger.info(f"Replay progress: step {step}/{total_steps_to_run}")

        logger.success("Replay finished!")
        return global_buffer


class MarketRealtimeEngine(MarketEngineBase):
    """Engine for realtime bounded buffer execution."""

    def __init__(self, data_1h: pd.DataFrame, warmup_size: int = 2400, n_steps: int = 50):
        super().__init__(data_1h, warmup_size)
        self.n_steps = n_steps
        # Sliding buffer logic: FIFO limited size
        self.global_buffer = deque(maxlen=self.n_steps)

    def add_next_step(self, step: int, sync_type: str, sync_version: str) -> dict:
        """Processes the next step and pushes it to the bounded buffer."""
        if step >= self.total_rows - self.warmup_size:
            raise ValueError("No more realtime data available in the dataset.")

        res = self.process_step(step, sync_type, sync_version)
        self.global_buffer.append(res)
        
        # Memory Optimization: Prune raw buffers to prevent memory leak in realtime
        # We only need the last ~warmup_size data to compute indicators effectively.
        max_1h = self.warmup_size + 100
        if len(self.buffer_1h) > max_1h:
            self.buffer_1h = self.buffer_1h.tail(self.warmup_size).reset_index(drop=True)
            
        max_4h = (self.warmup_size // 4) + 25
        if len(self.buffer_4h) > max_4h:
            self.buffer_4h = self.buffer_4h.tail(self.warmup_size // 4).reset_index(drop=True)
            
        max_1d = (self.warmup_size // 24) + 5
        if len(self.buffer_1d) > max_1d:
            self.buffer_1d = self.buffer_1d.tail(self.warmup_size // 24).reset_index(drop=True)
            
        return res

    def get_latest_steps(self) -> List[dict]:
        """Returns the items currently stored in the memory buffer."""
        return list(self.global_buffer)
