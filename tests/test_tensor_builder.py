"""
Unit tests for the Tensor Builder.
Validates RF-8, RF-10 structural integrity.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.tensor.builder import TensorBuilder


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a synthetic OHLCV-like DataFrame with 50 rows."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame(
        {
            "open_1h": np.random.uniform(90, 110, n),
            "close_1h": np.random.uniform(90, 110, n),
            "high_1h": np.random.uniform(100, 120, n),
            "low_1h": np.random.uniform(80, 100, n),
            "volume_1h": np.random.uniform(1000, 5000, n),
            "Ind1_1h": np.random.uniform(90, 110, n),
            "Ind2_1h": np.random.uniform(90, 110, n),
            "Ind3_1h": np.random.uniform(0, 100, n),
            "precio_actual": np.random.uniform(90, 110, n),
            "tiempo_normalizado": np.linspace(0, 1, n),
        },
        index=pd.date_range("2024-01-01", periods=n, freq="h"),
    )


class TestTensorBuilder:
    def test_build_shape(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        tensor = builder.build(sample_df)
        # 50 rows, window 30 → 21 windows, 10 features
        assert tensor.shape == (21, 30, 10)

    def test_build_single_window(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        tensor = builder.build_single_window(sample_df)
        assert tensor.shape == (1, 30, 10)

    def test_too_few_rows_raises(self):
        small_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        builder = TensorBuilder(window_size=10)
        with pytest.raises(ValueError, match="window_size"):
            builder.build(small_df)

    def test_describe_metadata(self, sample_df: pd.DataFrame):
        builder = TensorBuilder(window_size=30)
        meta = builder.describe(sample_df)
        assert meta["window_size"] == 30
        assert meta["num_features"] == 10
        assert meta["tensor_shape"] == [21, 30, 10]
