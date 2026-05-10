import numpy as np
import pandas as pd
from loguru import logger

class DataNormalizer:
    """
    Normalizes OHLCV and indicator data using efficient Pandas rolling operations
    to prevent look-ahead bias and handle non-stationary financial data.
    """
    
    # Rango de ventana móvil para estadísticas (mediana, IQR)
    ROLLING_WINDOW = 100

    @classmethod
    def apply_normalization(cls, df: pd.DataFrame, timeframe_suffix: str = "") -> pd.DataFrame:
        """
        Aplica reglas de normalización a las columnas de un DataFrame.
        Esta función se ejecuta después de que todos los indicadores han sido calculados.
        """
        result = df.copy()
        suffix = f"_{timeframe_suffix}" if timeframe_suffix else ""
        
        # 1. PRECIOS OHLC (Log-Returns o Retornos Logarítmicos)
        # rt = ln(P_t / P_{t-1})
        for col in ["open", "high", "low", "close"]:
            col_name = f"{col}{suffix}"
            if col_name in result.columns:
                result[col_name] = cls._log_returns(result[col_name])

        # El volumen original o actual se requiere a veces para otros cálculos, pero ya llegamos tarde.
        # 2. VOLUMEN (Log-Transform + RobustScaler)
        vol_col = f"volume{suffix}"
        if vol_col in result.columns:
            # Primero log transform para aplastar picos extremos: log(1 + V)
            log_vol = np.log1p(result[vol_col])
            # Luego Robust Scaler móvil
            result[vol_col] = cls._rolling_robust_scaler(log_vol)

        # 3. INDICADORES (Por familia)
        for col in result.columns:
            if suffix and not col.endswith(suffix):
                continue
                
            base_col = col.replace(suffix, "") if suffix else col
            
            # --- Rango Limitado: Min-Max ---
            if any(base_col.startswith(p) for p in ["RSI_", "STOCH_K", "STOCH_D"]):
                # De 0 a 100 -> de 0 a 1
                result[col] = result[col] / 100.0
                
            elif base_col == "WILLIAMS_R":
                # De -100 a 0 -> de -1 a 0 (o de 0 a 1 si sumamos 100 y dividimos)
                result[col] = (result[col] / 100.0) # -> Rango [-1, 0]
                
            elif base_col == "CMF":
                # Ya va de -1 a 1, lo dejamos igual o normalizamos un poco. Ya está acotado.
                pass
                
            # --- Distancia al Cierre ---
            elif any(base_col.startswith(p) for p in ["EMA_", "ICHIMOKU_", "VWAP", "BB_UPPER", "BB_LOWER", "BB_MIDDLE", "KELTNER_"]):
                # (Valor / Close) - 1
                close_col = f"close{suffix}"
                # ATENCION: El close ya se convirtió a Log-Return arriba.
                # Necesitamos usar el close original. Como lo sobrescribimos, mejor usar una copia
                # Guardaremos el close_original temporalmente si es posible.
                pass # Lo corregiremos en la reestructuración abajo.

        return result

    @classmethod
    def normalize_indicators(cls, df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        """
        Función principal de normalización.
        """
        result = df.copy()
        
        # En esta etapa, el OHLCV no tiene el suffix todavía. Solo los indicadores lo tienen.
        close_col = "close"
        # Guardar close original temporalmente para cálculos relativos (EMA, VWAP, etc.)
        close_original = result[close_col].copy() if close_col in result.columns else None

        for col in result.columns:
            # 1. Normalizar OHLCV directamente (no tienen suffix todavía)
            if col in ["open", "high", "low", "close"]:
                result[col] = cls._log_returns(result[col])
                continue
            elif col == "volume":
                log_vol = np.log1p(result[col])
                result[col] = cls._rolling_robust_scaler(log_vol)
                continue

            # 2. Filtrar otras columnas que no sean indicadores generados para este timeframe
            if suffix and not col.endswith(suffix):
                continue
                
            base_col = col.replace(suffix, "") if suffix else col

            # 3. OSCILADORES LIMITADOS (/ 100)
            if any(base_col.startswith(p) for p in ["RSI_", "STOCH_"]):
                result[col] = result[col] / 100.0
            elif base_col == "WILLIAMS_R":
                result[col] = result[col] / 100.0

            # 4. SEGUIDORES DE TENDENCIA (Relativo al Cierre)
            elif any(base_col.startswith(p) for p in ["EMA_", "ICHIMOKU_", "VWAP", "BB_UPPER", "BB_MIDDLE", "BB_LOWER", "KELTNER_"]):
                if close_original is not None:
                    # (Valor / Precio) - 1
                    result[col] = np.where(close_original > 0, (result[col] / close_original) - 1.0, 0.0)

            # 5. OSCILADORES NO LIMITADOS (Rolling Robust Scaler)
            elif any(base_col.startswith(p) for p in ["MON", "ROC", "SQZ_MOM", "CCI", "MACD", "OBV", "ELDER_", "EOM", "BB_WIDTH", "ADX", "DI_"]):
                result[col] = cls._rolling_robust_scaler(result[col])
                
        # Reemplazar infinitos por NaN y luego rellenar con 0 (log-returns iniciales, etc)
        result.replace([np.inf, -np.inf], np.nan, inplace=True)
        result.fillna(0, inplace=True)
        
        return result

    @staticmethod
    def _log_returns(series: pd.Series) -> pd.Series:
        """Calcula el log-return: ln(P_t / P_{t-1})"""
        return np.log(series / series.shift(1))

    @staticmethod
    def _pct_change(series: pd.Series) -> pd.Series:
        """Calcula el porcentaje de cambio: (P_t / P_{t-1}) - 1"""
        return series.pct_change()

    @classmethod
    def _rolling_robust_scaler(cls, series: pd.Series) -> pd.Series:
        """
        Escala usando la mediana y el rango intercuartílico (IQR) de los últimos N periodos.
        (X - Median) / (Q3 - Q1)
        """
        # Calcular Rolling Median
        rolling_median = series.rolling(window=cls.ROLLING_WINDOW, min_periods=1).median()
        
        # Calcular Cuartiles Q1 (25%) y Q3 (75%)
        q1 = series.rolling(window=cls.ROLLING_WINDOW, min_periods=1).quantile(0.25)
        q3 = series.rolling(window=cls.ROLLING_WINDOW, min_periods=1).quantile(0.75)
        iqr = q3 - q1
        
        # Evitar división por cero si el IQR es 0 (ej. datos estáticos)
        iqr = iqr.replace(0, 1e-8)
        
        return (series - rolling_median) / iqr
