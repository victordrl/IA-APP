"""
RF-7: Multi-timeframe synchronization.

Múltiples modos de sincronización:
- TYPE 1 (timeframe): 1h|indicadores|4h|indicadores|1d|indicadores
- TYPE 2 (merged): todos los indicadores juntos sin separación por timeframe
- TYPE 3 (semantic): por grupos (VELOCIDAD|1h,4h,1d|TENDENCIA|...)

Cada tipo tiene 2 versiones:
- ohlcv: incluye OHLCV, volume, progress
- indicators: solo indicadores

Bug fix: sincronización correcta sin reindex/ffill problemático.
"""

import pandas as pd
from loguru import logger

from app.config import settings


class MultiTimeframeSync:
    """Synchronize OHLCV DataFrames con múltiples modos y versiones."""

    _TF_CANONICAL = {"1h": "1h", "4h": "4h", "1d": "1d", "1D": "1d"}

    OHLC_COLS = ["open", "high", "low", "close"]
    VOLUME_PROGRESS_COLS = ["volume", "progress_vela"]
    OHLCV_COLS = OHLC_COLS + VOLUME_PROGRESS_COLS

    GROUPS = {
        "VELOCIDAD": ["MON", "ROC", "SQZ_MOM", "RSI_6", "RSI_14", "RSI_24", "RSI_EMA_6", "RSI_EMA_14",
                     "RSI_EMA_24", "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI"],
        "TENDENCIA": ["MACD_LINE", "MACD_SIGNAL", "MACD_HIST", "ADX", "DI_PLUS", "DI_MINUS",
                     "EMA_7", "EMA_22", "EMA_50", "EMA_99", "EMA_100", "ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN",
                     "ICHIMOKU_SA", "ICHIMOKU_SB", "ICHIMOKU_CHIKOU"],
        "AMPLITUD": ["BB_UPPER", "BB_MIDDLE", "BB_LOWER", "BB_WIDTH",
                     "KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"],
        "LIQUIDEZ": ["CMF", "OBV", "ELDER_BULL", "ELDER_BEAR", "EOM", "VWAP"],
    }

    def __init__(self, base_timeframe: str = "1h"):
        self._base = base_timeframe

    def synchronize(self, data: dict[str, pd.DataFrame], sync_type: str = None,
                    sync_version: str = None, include_global: bool = True) -> pd.DataFrame:
        """Main synchronization method."""
        sync_type = sync_type or "timeframe"
        sync_version = sync_version or "ohlcv"

        if self._base not in data:
            raise ValueError(f"Base timeframe '{self._base}' not found: {list(data.keys())}")

        base_df = data[self._base].copy()
        base_df = base_df.set_index("timestamp").sort_index()

        # Agregar sufijo a TODAS las columnas OHLCV del base timeframe también
        for col in self.OHLCV_COLS:
            if col in base_df.columns:
                base_df = base_df.rename(columns={col: f"{col}_{self._base}"})

        for tf, df in data.items():
            canonical = self._TF_CANONICAL.get(tf, tf)
            if canonical != self._base:
                df_copy = df.copy().set_index("timestamp").sort_index()
                # Agregar sufijo a las columnas OHLCV para evitar overlap
                for col in self.OHLCV_COLS:
                    if col in df_copy.columns:
                        df_copy = df_copy.rename(columns={col: f"{col}_{canonical}"})
                df_copy = df_copy.reindex(base_df.index, method="ffill")
                base_df = base_df.join(df_copy, how="left")

        base_df = base_df.ffill()
        result = base_df.copy()

        # Generate global features FIRST so they exist for all versions
        if include_global:
            result = self.add_global_features(result)

        # Apply filtering based on sync_version
        cols_to_keep = []
        for col in result.columns:
            if col in ["precio_actual", "tiempo_normalizado"]:
                cols_to_keep.append(col)
                continue
                
            base_col = col.replace("_1h", "").replace("_4h", "").replace("_1d", "")
            is_ohlc = base_col in self.OHLC_COLS
            is_vp = base_col in self.VOLUME_PROGRESS_COLS
            is_indicator = not is_ohlc and not is_vp
            
            if sync_version == "base":
                if is_ohlc or is_vp: cols_to_keep.append(col)
            elif sync_version == "indicators":
                if is_vp or is_indicator: cols_to_keep.append(col)
            else: # "ohlcv"
                cols_to_keep.append(col)
                
        result = result[cols_to_keep]

        # Order columns based on sync_type
        if sync_type == "timeframe":
            result = self._sync_timeframe(result)
        elif sync_type == "merged":
            result = self._sync_merged(result)
        elif sync_type == "semantic":
            result = self._sync_semantic(result)
        else:
            raise ValueError(f"Unknown sync_type: {sync_type}")

        return result

    def _sync_timeframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tipo 1: Cada timeframe con sus indicadores juntos."""
        result_cols = []

        for tf in ["1h", "4h", "1d"]:
            tf_cols = [c for c in df.columns if f"_{tf}" in c]
            if not tf_cols:
                continue

            ohlc_tf = [c for c in tf_cols if c.endswith(f"_{tf}") and c.replace(f"_{tf}", "") in self.OHLC_COLS]
            vol_prog_tf = [c for c in tf_cols if c.endswith(f"_{tf}") and c.replace(f"_{tf}", "") in self.VOLUME_PROGRESS_COLS]
            indicators_tf = [c for c in tf_cols if c not in ohlc_tf and c not in vol_prog_tf]

            result_cols.extend(ohlc_tf)
            result_cols.extend(vol_prog_tf)
            result_cols.extend(indicators_tf)

        for col in df.columns:
            if col not in result_cols:
                result_cols.append(col)

        return df[result_cols].copy()

    def _sync_merged(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tipo 2: Todos los indicadores juntos sin separación por timeframe."""
        result_cols = []

        # 1. OHLCV
        for tf in ["1h", "4h", "1d"]:
            ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and c.replace(f"_{tf}", "") in self.OHLCV_COLS]
            result_cols.extend(ohlcv_cols)

        # 2. Indicators
        all_indicators = set()
        for tf in ["1h", "4h", "1d"]:
            tf_indicators = [c for c in df.columns if f"_{tf}" in c and c.replace(f"_{tf}", "") not in self.OHLC_COLS and c.replace(f"_{tf}", "") not in self.VOLUME_PROGRESS_COLS]
            all_indicators.update(tf_indicators)

        result_cols.extend(sorted(all_indicators))

        for col in df.columns:
            if col not in result_cols:
                result_cols.append(col)

        return df[result_cols].copy()

    def _sync_semantic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tipo 3: Por grupos semánticos (VELOCIDAD, TENDENCIA, AMPLITUD, LIQUIDEZ)."""
        result_cols = []

        # Group logic is actually mainly handled downstream by market_engine formatting
        # Here we just order them loosely.
        
        # 1. OHLCV
        for tf in ["1h", "4h", "1d"]:
            ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and c.replace(f"_{tf}", "") in self.OHLCV_COLS]
            result_cols.extend(ohlcv_cols)

        # 2. Semantic Groups
        for group_name, indicators in self.GROUPS.items():
            for indicator in indicators:
                for tf in ["1h", "4h", "1d"]:
                    full_col = f"{indicator}_{tf}"
                    if full_col in df.columns and full_col not in result_cols:
                        result_cols.append(full_col)

        for col in df.columns:
            if col not in result_cols:
                result_cols.append(col)

        return df[result_cols].copy()

    @staticmethod
    def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
        """Agregar features globales."""
        result = df.copy()

        close_1h_col = [c for c in result.columns if c.startswith("close") and "1h" in c]
        if close_1h_col:
            result["precio_actual"] = result[close_1h_col[0]]

        if "tiempo_normalizado" not in result.columns:
            result["tiempo_normalizado"] = result.index.hour / 24.0

        return result

    @staticmethod
    def get_sync_types() -> list[str]:
        """Retorna los tipos de sync disponibles."""
        return ["timeframe", "merged", "semantic"]

    @staticmethod
    def get_versions() -> list[str]:
        """Retorna las versiones disponibles."""
        return ["ohlcv", "indicators"]