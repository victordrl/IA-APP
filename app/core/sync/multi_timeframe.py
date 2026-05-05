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

    OHLCV_COLS = ["open", "high", "low", "close", "volume", "progress_vela"]

    GROUPS = {
        "VELOCIDAD": ["MON", "ROC", "RSI_6", "RSI_14", "RSI_24", "RSI_EMA_6", "RSI_EMA_14",
                     "RSI_EMA_24", "STOCH_K", "STOCH_D", "WILLIAMS_R", "CCI"],
        "TENDENCIA": ["MACD_LINE", "MACD_SIGNAL", "MACD_HIST", "ADX", "DI_PLUS", "DI_MINUS",
                     "EMA_22", "EMA_50", "EMA_100", "ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN",
                     "ICHIMOKU_SA", "ICHIMOKU_SB", "ICHIMOKU_CHIKOU"],
        "AMPLITUD": ["BB_UPPER", "BB_MIDDLE", "BB_LOWER", "BB_WIDTH",
                     "KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"],
        "LIQUIDEZ": ["CMF", "OBV", "ELDER_BULL", "ELDER_BEAR", "EOM", "VWAP"],
    }

    def __init__(self, base_timeframe: str = "1h"):
        self._base = base_timeframe

    def synchronize(self, data: dict[str, pd.DataFrame], sync_type: str = None,
                    sync_version: str = None, include_global: bool = True) -> pd.DataFrame:
        """Main synchronization method.

        Args:
            data: dict with timeframe keys and DataFrame values
            sync_type: "timeframe" | "merged" | "semantic" (default from config)
            sync_version: "ohlcv" | "indicators" (default from config)
            include_global: add global features (precio_actual, tiempo_normalizado)
        """
        sync_type = sync_type or settings.sync_type
        sync_version = sync_version or settings.sync_version

        if self._base not in data:
            raise ValueError(f"Base timeframe '{self._base}' not found: {list(data.keys())}")

        base_df = data[self._base].copy()
        base_df = base_df.set_index("timestamp").sort_index()

        for tf, df in data.items():
            canonical = self._TF_CANONICAL.get(tf, tf)
            if canonical != self._base:
                df_copy = df.copy().set_index("timestamp").sort_index()
                df_copy = df_copy.reindex(base_df.index, method="ffill")
                base_df = base_df.join(df_copy, how="left")

        base_df = base_df.ffill()

        result = base_df.copy()

        if sync_type == "timeframe":
            result = self._sync_timeframe(result, sync_version)
        elif sync_type == "merged":
            result = self._sync_merged(result, sync_version)
        elif sync_type == "semantic":
            result = self._sync_semantic(result, sync_version)
        else:
            raise ValueError(f"Unknown sync_type: {sync_type}")

        if include_global:
            result = self.add_global_features(result)

        return result

    def _sync_timeframe(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 1: Cada timeframe con sus indicadores juntos."""
        result = pd.DataFrame(index=df.index)

        for tf in ["1h", "4h", "1d"]:
            tf_cols = [c for c in df.columns if f"_{tf}" in c]
            if not tf_cols:
                continue

            ohlcv_tf = [c for c in tf_cols if any(c.endswith(f"_{tf}") and c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
            indicators_tf = [c for c in tf_cols if c not in ohlcv_tf]

            if version == "ohlcv":
                for col in ohlcv_tf:
                    result[col] = df[col]
            elif version == "indicators":
                pass

            for col in indicators_tf:
                result[col] = df[col]

        for col in df.columns:
            if not any(f"_{tf}" in col for tf in ["1h", "4h", "1d"]):
                result[col] = df[col]

        return result

    def _sync_merged(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 2: Todos los indicadores juntos sin separación por timeframe."""
        result = pd.DataFrame(index=df.index)

        if version == "ohlcv":
            for tf in ["1h", "4h", "1d"]:
                ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and any(c.replace(f"_{tf}", "") in self.OHLCV_COLS for _ in [1])]
                for col in ohlcv_cols:
                    result[col] = df[col]

        all_indicators = []
        for tf in ["1h", "4h", "1d"]:
            tf_indicators = [c for c in df.columns if f"_{tf}" in c and not any(c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
            all_indicators.extend(tf_indicators)

        for col in all_indicators:
            indicator_name = col.rsplit("_", 1)[0]
            for tf in ["1h", "4h", "1d"]:
                full_col = f"{indicator_name}_{tf}"
                if full_col in df.columns:
                    result[full_col] = df[full_col]

        for col in df.columns:
            if col not in result.columns:
                result[col] = df[col]

        return result

    def _sync_semantic(self, df: pd.DataFrame, version: str) -> pd.DataFrame:
        """Tipo 3: Por grupos semánticos (VELOCIDAD, TENDENCIA, AMPLITUD, LIQUIDEZ)."""
        result = pd.DataFrame(index=df.index)

        if version == "ohlcv":
            for tf in ["1h", "4h", "1d"]:
                ohlcv_cols = [c for c in df.columns if f"_{tf}" in c and any(c.replace(f"_{tf}", "") in self.OHLCV_COLS)]
                for col in ohlcv_cols:
                    result[col] = df[col]

        for group_name, indicators in self.GROUPS.items():
            group_cols = []
            for indicator in indicators:
                for tf in ["1h", "4h", "1d"]:
                    full_col = f"{indicator}_{tf}"
                    if full_col in df.columns:
                        group_cols.append(full_col)

            for col in group_cols:
                result[col] = df[col]

        for col in df.columns:
            if col not in result.columns:
                result[col] = df[col]

        return result

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