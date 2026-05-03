"""
RF-9: Data normalization strategies.

Applies the appropriate normalization method depending on feature type:
- Bounded indicators (RSI, Stochastic) → Min-Max [0, 1]
- Prices and returns                   → Z-Score
- Volumes                              → Log scaling + standardization
"""

import numpy as np
import pandas as pd
from loguru import logger


class Normalizer:
    """Apply column-aware normalization to a feature DataFrame.

    Stores fitted parameters (mean, std, min, max) so the same transform
    can be applied identically on future data (RNF-2 reproducibility).
    """

    # Columns containing these substrings use specific strategies.
    _BOUNDED_KEYWORDS = {"Ind3", "Ind10"}  # RSI, Stochastic → Min-Max
    _VOLUME_KEYWORDS = {"volume", "Ind9"}  # Volume, OBV → Log-scale
    # Everything else → Z-Score

    def __init__(self):
        self._params: dict[str, dict] = {}  # {col: {method, ...fitted_values}}

    # ── Public API ──────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on *df* and return the normalized copy.

        Args:
            df: DataFrame of numeric features (excludes ``timestamp``).

        Returns:
            Normalized DataFrame with the same shape and columns.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            method = self._detect_method(col)
            result[col] = self._apply(result[col], col, method, fit=True)

        logger.debug("Fit & transformed {} numeric columns", len(numeric_cols))
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform *df* using previously fitted parameters.

        Args:
            df: DataFrame with the same columns used in ``fit_transform``.

        Returns:
            Normalized DataFrame.
        """
        result = df.copy()
        numeric_cols = result.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col not in self._params:
                logger.warning("Column {} was not fitted — skipping", col)
                continue
            method = self._params[col]["method"]
            result[col] = self._apply(result[col], col, method, fit=False)

        return result

    # ── Internal ────────────────────────────────────

    def _detect_method(self, col: str) -> str:
        """Heuristic to select normalization method based on column name."""
        for kw in self._BOUNDED_KEYWORDS:
            if kw in col:
                return "minmax"
        for kw in self._VOLUME_KEYWORDS:
            if kw.lower() in col.lower():
                return "log"
        return "zscore"

    def _apply(self, series: pd.Series, col: str, method: str, fit: bool) -> pd.Series:
        if method == "minmax":
            return self._minmax(series, col, fit)
        elif method == "log":
            return self._log_scale(series, col, fit)
        else:
            return self._zscore(series, col, fit)

    def _minmax(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        if fit:
            mn, mx = s.min(), s.max()
            self._params[col] = {"method": "minmax", "min": mn, "max": mx}
        else:
            mn = self._params[col]["min"]
            mx = self._params[col]["max"]
        rng = mx - mn
        if rng == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / rng

    def _zscore(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        if fit:
            mean, std = s.mean(), s.std()
            self._params[col] = {"method": "zscore", "mean": mean, "std": std}
        else:
            mean = self._params[col]["mean"]
            std = self._params[col]["std"]
        if std == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mean) / std

    def _log_scale(self, s: pd.Series, col: str, fit: bool) -> pd.Series:
        logged = np.log1p(s.clip(lower=0))
        if fit:
            mean, std = logged.mean(), logged.std()
            self._params[col] = {"method": "log", "mean": mean, "std": std}
        else:
            mean = self._params[col]["mean"]
            std = self._params[col]["std"]
        if std == 0:
            return pd.Series(0.0, index=s.index)
        return (logged - mean) / std

    @property
    def fitted_params(self) -> dict[str, dict]:
        """Return a copy of the fitted normalization parameters."""
        return dict(self._params)
