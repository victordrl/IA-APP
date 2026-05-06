"""
RF-5: Backtrader Data Replay - Reconstruct higher timeframes from 1h data.

LOGICA DEL REPLAY:
1. Fetch 1h del rango completo (ej: 2026-01-01 a 2026-05-01)
2. Warmup: primeros 2400 datos (~100 días) para indicadores completos
3. Por cada step posterior:
   - Agregar 1 vela 1h al buffer
   - Acumular datos en vela 4h (cerrar cada 4 steps)
   - Acumular datos en vela 1d (cerrar cada 24 steps)
4. Retornar buffers con TODOS los datos acumulados + vela en progreso
5. Los indicadores usan todos los datos del buffer
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import pandas as pd
from loguru import logger

from app.config import settings


class BacktraderReplay:
    """Replay 1h data and reconstruct 4h/1d with dynamic buffers.

    Los buffers startan con warmup de datos históricos para indicadores completos.
    La vela en progreso siempre es parte del buffer (desde el inicio).
    """

    def __init__(
        self,
        data_1h: pd.DataFrame,
        window_size: int | None = None,
        speed_multiplier: float = 1.0,
        refresh_seconds: float = 5.0,
    ):
        self._window_size = window_size or settings.tensor_window_size
        self._indicators_warmup = settings.replay_indicators_warmup  # 2400
        self._speed = speed_multiplier
        self._refresh = refresh_seconds
        self._active = False

        # Datos 1h - asegurar que tiene índice de timestamp
        self._data_1h = data_1h.copy()
        if "timestamp" in self._data_1h.columns:
            self._data_1h = self._data_1h.set_index("timestamp").sort_index()
        self._data_1h.index = pd.DatetimeIndex(self._data_1h.index)

        # Total de datos disponibles para replay
        self._total_1h = len(self._data_1h)

        # Validar que hay suficientes datos para warmup + replay
        min_required = self._indicators_warmup + 1  # al menos 1 step
        if self._total_1h < min_required:
            raise ValueError(
                f"Insufficient data: {self._total_1h} rows. "
                f"Need at least {self._indicators_warmup} for indicators warmup (~100 days). "
                f"Requested range provides {self._total_1h - self._indicators_warmup} replay steps."
            )

        # Warmup: primeros 2400 datos para indicadores
        self._warmup_end = self._indicators_warmup
        self._max_steps = self._total_1h - self._warmup_end

        # Fecha real de inicio del step 1
        self._first_step_date = self._data_1h.index[self._warmup_end]

        # Buffers dinámicos - startan con warmup data
        self._buffer_1h = self._data_1h.iloc[:self._warmup_end].copy().reset_index()
        self._buffer_1h["progress_vela"] = 1.0

        # Pre-calcular 4h y 1d del warmup
        self._buffer_4h = self._build_warmup_timeframes(self._buffer_1h, "4h")
        self._buffer_1d = self._build_warmup_timeframes(self._buffer_1h, "1d")

        # Inicializar velas en progreso con el primer dato del replay
        first_replay_idx = self._data_1h.index[self._warmup_end]
        first_row = self._data_1h.iloc[self._warmup_end]
        self._current_4h = self._init_4h_candle(first_replay_idx, first_row)
        self._current_1d = self._init_1d_candle(first_replay_idx, first_row)

        logger.info(
            "BacktraderReplay initialized — warmup={} (~100 days), steps={}, total_1h={}",
            self._warmup_end,
            self._max_steps,
            self._total_1h,
        )
        logger.info(
            "Replay range: {} to {}",
            self._data_1h.index[0],
            self._data_1h.index[-1],
        )
        logger.info(
            "Step 1 starts at: {} (date: {})",
            self._warmup_end,
            self._first_step_date,
        )

    def _init_4h_candle(self, timestamp, row) -> dict:
        """Inicializar vela de 4h."""
        return {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "progress_vela": 0.25,
        }

    def _init_1d_candle(self, timestamp, row) -> dict:
        """Inicializar vela de 1d."""
        return {
            "timestamp": timestamp,
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
            "progress_vela": 1 / 24,
        }

    def _build_warmup_timeframes(self, df_1h: pd.DataFrame, target: str) -> pd.DataFrame:
        """Build 4h or 1d from warmup 1h data."""
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

    @property
    def first_step_date(self):
        """Retorna la fecha real donde comienza el step 1."""
        return self._first_step_date

    async def stream(self) -> AsyncGenerator[dict, None]:
        """Async generator que yield buffers dinámicos por cada step.

        Yields:
            - buffer_1h: DataFrame con todas las velas 1h acumuladas
            - buffer_4h: DataFrame con velas 4h (cerradas + en progreso)
            - buffer_1d: DataFrame con velas 1d (cerradas + en progreso)
            - step: número de step actual
        """
        self._active = True
        delay = self._refresh / self._speed

        # Agregar las velas en progreso actuales al buffer
        self._buffer_4h = pd.concat([self._buffer_4h, pd.DataFrame([self._current_4h])], ignore_index=True)
        self._buffer_1d = pd.concat([self._buffer_1d, pd.DataFrame([self._current_1d])], ignore_index=True)

        # El replay starts desde warmup_end
        for step in range(self._max_steps):
            if not self._active:
                logger.info("Replay stopped at step {}/{}", step, self._max_steps)
                return

            # Índice del dato 1h actual (del range total)
            data_idx = self._warmup_end + step
            current_timestamp = self._data_1h.index[data_idx]
            current_row = self._data_1h.iloc[data_idx]

            # Agregar al buffer 1h
            new_1h_row = {
                "timestamp": current_timestamp,
                "open": float(current_row["open"]),
                "high": float(current_row["high"]),
                "low": float(current_row["low"]),
                "close": float(current_row["close"]),
                "volume": float(current_row["volume"]),
                "progress_vela": 1.0,
            }
            self._buffer_1h = pd.concat([self._buffer_1h, pd.DataFrame([new_1h_row])], ignore_index=True)

            # Actualizar 4h
            step_in_4h = step % 4
            progress_4h = (step_in_4h + 1) / 4.0

            if self._current_4h is not None:
                self._current_4h["high"] = max(self._current_4h["high"], new_1h_row["high"])
                self._current_4h["low"] = min(self._current_4h["low"], new_1h_row["low"])
                self._current_4h["close"] = new_1h_row["close"]
                self._current_4h["volume"] += new_1h_row["volume"]
                self._current_4h["progress_vela"] = progress_4h
                self._current_4h["timestamp"] = current_timestamp

                # Si es el último del grupo de 4, cerrar y crear nueva
                if step_in_4h == 3:
                    self._current_4h["progress_vela"] = 1.0
                    closed_4h = self._current_4h.copy()
                    self._buffer_4h = pd.concat([self._buffer_4h, pd.DataFrame([closed_4h])], ignore_index=True)

                    # Crear nueva vela con progress inicial
                    if data_idx + 1 < self._total_1h:
                        next_timestamp = self._data_1h.index[data_idx + 1]
                        next_row = self._data_1h.iloc[data_idx + 1]
                        self._current_4h = self._init_4h_candle(next_timestamp, next_row)
                        self._buffer_4h = pd.concat([self._buffer_4h, pd.DataFrame([self._current_4h])], ignore_index=True)
                else:
                    # Actualizar la última fila del buffer con los valores actuales de la vela en progreso
                    if len(self._buffer_4h) > 0:
                        self._buffer_4h.iloc[-1] = pd.Series(self._current_4h)

            # Actualizar 1d
            step_in_1d = step % 24
            progress_1d = (step_in_1d + 1) / 24.0

            if self._current_1d is not None:
                self._current_1d["high"] = max(self._current_1d["high"], new_1h_row["high"])
                self._current_1d["low"] = min(self._current_1d["low"], new_1h_row["low"])
                self._current_1d["close"] = new_1h_row["close"]
                self._current_1d["volume"] += new_1h_row["volume"]
                self._current_1d["progress_vela"] = progress_1d
                self._current_1d["timestamp"] = current_timestamp

                # Si es el último del grupo de 24, cerrar y crear nueva
                if step_in_1d == 23:
                    self._current_1d["progress_vela"] = 1.0
                    closed_1d = self._current_1d.copy()
                    self._buffer_1d = pd.concat([self._buffer_1d, pd.DataFrame([closed_1d])], ignore_index=True)

                    # Crear nueva vela con progress inicial
                    if data_idx + 1 < self._total_1h:
                        next_timestamp = self._data_1h.index[data_idx + 1]
                        next_row = self._data_1h.iloc[data_idx + 1]
                        self._current_1d = self._init_1d_candle(next_timestamp, next_row)
                        self._buffer_1d = pd.concat([self._buffer_1d, pd.DataFrame([self._current_1d])], ignore_index=True)
                else:
                    # Actualizar la última fila del buffer con los valores actuales de la vela en progreso
                    if len(self._buffer_1d) > 0:
                        self._buffer_1d.iloc[-1] = pd.Series(self._current_1d)

            # Log simplificado
            logger.info(
                f"[Step {step + 1}/{self._max_steps}] | "
                f"1h: {len(self._buffer_1h)} | "
                f"4h: {len(self._buffer_4h)} (p:{progress_4h:.2f}) | "
                f"1d: {len(self._buffer_1d)} (p:{progress_1d:.2f})"
            )

            yield {
                "buffer_1h": self._buffer_1h.copy(),
                "buffer_4h": self._buffer_4h.copy(),
                "buffer_1d": self._buffer_1d.copy(),
                "step": step,
                "progress_4h": progress_4h,
                "progress_1d": progress_1d,
            }

            await asyncio.sleep(delay)

        self._active = False
        logger.success("Replay completed — {} steps emitted", self._max_steps)

    def stop(self) -> None:
        """Stop the replay mid-stream."""
        self._active = False

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def total_steps(self) -> int:
        return self._max_steps