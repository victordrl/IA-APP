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
from app.core.processing.indicators import IndicatorEngine


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

    def build_grouped(self, df: pd.DataFrame) -> dict[str, torch.Tensor]:
        """Build separate tensors for each indicator group.

        Each tensor contains:
        - Indicator columns for that group (from all timeframes)
        - progress_vela columns (1h, 4h, 1d) - shared across all groups

        Args:
            df: Fully synchronized, normalized DataFrame.

        Returns:
            Dict with keys: "velocidad", "tendencia", "amplitud", "liquidez"
            Each value is a torch.Tensor of shape (1, window_size, num_features).
        """
        self._validate(df)

        # Columns de progreso de vela (comunes a todos los grupos)
        progress_cols = [
            "progress_vela_1h_1h",
            "progress_vela_4h",
            "progress_vela_1d",
        ]
        available_progress = [c for c in progress_cols if c in df.columns]

        # Obtener columnas de indicadores por grupo
        groups_info = IndicatorEngine.get_all_groups_columns()

        result = {}
        for group_name, indicator_cols in groups_info.items():
            # Filtrar solo las columnas que existen en el DataFrame
            available_indicators = [c for c in indicator_cols if c in df.columns]

            # Combinar indicadores + progress_vela
            group_cols = available_indicators + available_progress

            if not group_cols:
                logger.warning(f"No columns found for group {group_name}")
                result[group_name] = None
                continue

            # Extraer valores
            group_df = df[group_cols].select_dtypes(include=[np.number])
            values = group_df.values

            # Validar y hacer sliding window
            if len(values) < self._window_size:
                logger.warning(f"Group {group_name}: not enough rows ({len(values)} < {self._window_size})")
                result[group_name] = None
                continue

            num_rows = len(values)
            num_windows = max(0, num_rows - self._window_size + 1)

            # Sliding window view
            windows = np.lib.stride_tricks.sliding_window_view(values, self._window_size, axis=0)
            windows = windows.transpose(0, 2, 1)  # (num_windows, window_size, num_features)

            tensor = torch.tensor(windows, dtype=torch.float32)

            logger.info(
                "Grupo {}: shape {} (indicators={}, progress={})",
                group_name,
                list(tensor.shape),
                len(available_indicators),
                len(available_progress),
            )

            result[group_name] = tensor

        return result

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
