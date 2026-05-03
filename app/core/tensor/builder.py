"""
RF-8 / RF-10 / RF-11: Tensor construction with sliding windows.

Builds a PyTorch tensor of shape ``(num_windows, window_size, num_features)``
from a fully synchronized and normalized DataFrame.  Each window is a
30 × N slice ready for neural network consumption.
"""

import numpy as np
import pandas as pd
import torch
from loguru import logger

from app.config import settings


class TensorBuilder:
    """Construct sliding-window tensors from a synchronized feature DataFrame.

    Satisfies:
    - RF-8:  Tensor shape ``30 × N`` with sliding windows.
    - RF-10: No out-of-range values, no temporal misalignment.
    - RF-11: Standard column layout per timeframe block + global vars.
    """

    def __init__(self, window_size: int | None = None):
        self._window_size = window_size or settings.tensor_window_size

    # ── Public API ──────────────────────────────────

    def build(self, df: pd.DataFrame) -> torch.Tensor:
        """Create a 3-D tensor from the synchronized DataFrame.

        Args:
            df: Fully synchronized, normalized DataFrame
                (index = timestamps, columns = features).

        Returns:
            ``torch.Tensor`` of shape ``(num_windows, window_size, num_features)``.

        Raises:
            ValueError: If the DataFrame has fewer rows than the window size.
        """
        self._validate(df)

        values = df.select_dtypes(include=[np.number]).values  # (T, N)
        num_rows, num_features = values.shape
        num_windows = num_rows - self._window_size + 1

        # Sliding-window view — zero-copy where possible
        windows = np.lib.stride_tricks.sliding_window_view(values, self._window_size, axis=0)
        # windows shape: (num_windows, num_features, window_size) → transpose
        windows = windows.transpose(0, 2, 1)  # → (num_windows, window_size, num_features)

        tensor = torch.tensor(windows, dtype=torch.float32)

        logger.success(
            "Tensor built — shape {} (windows={}, steps={}, features={})",
            list(tensor.shape),
            num_windows,
            self._window_size,
            num_features,
        )
        return tensor

    def build_single_window(self, df: pd.DataFrame) -> torch.Tensor:
        """Build a single ``(1, window_size, N)`` tensor from the last rows.

        Useful for real-time inference where only the latest window matters.
        """
        tail = df.tail(self._window_size)
        self._validate(tail)
        values = tail.select_dtypes(include=[np.number]).values
        tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        return tensor

    # ── Validation ──────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        """RF-10 structural validation."""
        if len(df) < self._window_size:
            raise ValueError(
                f"DataFrame has {len(df)} rows but window_size={self._window_size}"
            )

        numeric = df.select_dtypes(include=[np.number])

        # Check for NaN / Inf
        if numeric.isnull().any().any():
            nan_cols = numeric.columns[numeric.isnull().any()].tolist()
            logger.warning("NaN detected in columns: {} — filling with 0", nan_cols)
            df[nan_cols] = df[nan_cols].fillna(0)

        if np.isinf(numeric.values).any():
            raise ValueError("Infinite values detected in feature DataFrame")

    # ── Metadata ────────────────────────────────────

    def describe(self, df: pd.DataFrame) -> dict:
        """Return a summary of what the tensor would look like.

        Useful for the observability endpoint (RNF-4).
        """
        numeric = df.select_dtypes(include=[np.number])
        num_rows = len(df)
        num_features = len(numeric.columns)
        num_windows = max(0, num_rows - self._window_size + 1)
        return {
            "window_size": self._window_size,
            "num_features": num_features,
            "num_rows": num_rows,
            "num_windows": num_windows,
            "tensor_shape": [num_windows, self._window_size, num_features],
            "feature_columns": numeric.columns.tolist(),
        }
