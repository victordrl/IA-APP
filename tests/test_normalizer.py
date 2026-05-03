"""
Unit tests for the Normalizer.
Validates RF-9 normalization strategies.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.processing.normalizer import Normalizer


@pytest.fixture
def feature_df() -> pd.DataFrame:
    """Synthetic feature DataFrame."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame(
        {
            "close_1h": np.random.uniform(90, 110, n),    # → Z-Score
            "Ind3_1h": np.random.uniform(0, 100, n),      # → Min-Max (RSI)
            "volume_1h": np.random.uniform(1000, 5000, n), # → Log-scale
            "Ind10_4h": np.random.uniform(0, 100, n),     # → Min-Max (Stoch)
        }
    )


class TestNormalizer:
    def test_fit_transform_shape(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        assert result.shape == feature_df.shape

    def test_minmax_range(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        # RSI column → should be in [0, 1]
        assert result["Ind3_1h"].min() >= -1e-9
        assert result["Ind3_1h"].max() <= 1.0 + 1e-9

    def test_zscore_mean_std(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        result = norm.fit_transform(feature_df)
        # Z-scored column → mean ≈ 0, std ≈ 1
        assert abs(result["close_1h"].mean()) < 0.1
        assert abs(result["close_1h"].std() - 1.0) < 0.1

    def test_transform_uses_fitted_params(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        norm.fit_transform(feature_df)
        # Transform same data again → should give same result
        result2 = norm.transform(feature_df)
        assert result2.shape == feature_df.shape

    def test_fitted_params_stored(self, feature_df: pd.DataFrame):
        norm = Normalizer()
        norm.fit_transform(feature_df)
        params = norm.fitted_params
        assert "close_1h" in params
        assert params["close_1h"]["method"] == "zscore"
        assert params["Ind3_1h"]["method"] == "minmax"
