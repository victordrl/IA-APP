"""
RF-5: Backtrader Data Replay - Reconstruct higher timeframes from 1h data.

Uses backtrader's replay feature to simulate how 4h and 1d candles develop
from 1h input data, providing more realistic backtesting than pre-built candles.
"""

import asyncio
from datetime import datetime
from typing import AsyncGenerator

import backtrader as bt
import pandas as pd
from loguru import logger


class PandasDataFrameFeed(bt.feeds.PandasData):
    """Custom data feed that accepts a DataFrame directly."""

    params = (
        ("datetime", None),
        ("open", "open"),
        ("high", "high"),
        ("low", "low"),
        ("close", "close"),
        ("volume", "volume"),
        ("openinterest", -1),
    )


class _EmptyStrategy(bt.Strategy):
    """Estrategia vacía requerida para que backtrader procese los datos."""

    def __init__(self):
        pass

    def next(self):
        pass


class BacktraderReplay:
    """Replay 1h data and reconstruct 4h and 1d using backtrader's replay feature.

    Instead of fetching pre-built 4h/1d candles, this class takes 1h candles and
    uses backtrader to reconstruct how the higher timeframes would have formed
    in real-time.

    Flow per step:
    1. Take 1h window of data
    2. Reconstruct 4h via replay (4 candles = 1 x 4h)
    3. Reconstruct 1d via replay (24 candles = 1 x 1d)
    4. Yield {1h: df, "4h_recon": df, "1d_recon": df} with debug output
    """

    def __init__(
        self,
        data_1h: pd.DataFrame,
        window_size: int | None = None,
        speed_multiplier: float = 1.0,
        refresh_seconds: float = 5.0,
    ):
        """
        Args:
            data_1h: DataFrame with 1h OHLCV data (timestamp, open, high, low, close, volume)
            window_size: Number of rows per sliding window (default: 30)
            speed_multiplier: Speed factor for replay (1.0 = real-time speed)
            refresh_seconds: Base interval between emissions in seconds
        """
        self._window_size = window_size or 30
        self._speed = speed_multiplier
        self._refresh = refresh_seconds
        self._active = False

        self._data_1h = data_1h.copy()
        self._data_1h = self._data_1h.set_index("timestamp").sort_index()
        self._data_1h.index = pd.DatetimeIndex(self._data_1h.index)

        self._max_steps = max(0, len(self._data_1h) - self._window_size)

        logger.info(
            "BacktraderReplay initialized — window={}, speed={}×, steps={}, total_1h_rows={}",
            self._window_size,
            self._speed,
            self._max_steps,
            len(self._data_1h),
        )

    def _reconstruct_timeframe(self, df: pd.DataFrame, target_timeframe: str, compression: int) -> pd.DataFrame:
        """Use backtrader to reconstruct a higher timeframe from 1h data."""
        df_reset = df.copy()
        
        if "timestamp" in df_reset.columns:
            df_reset["datetime"] = pd.to_datetime(df_reset["timestamp"])
        elif "index" in df_reset.columns:
            df_reset["datetime"] = pd.to_datetime(df_reset["index"])
        else:
            df_reset["datetime"] = df_reset.index
            if hasattr(df_reset.index, 'to_pydatetime'):
                df_reset["datetime"] = df_reset.index.to_pydatetime()
            else:
                df_reset["datetime"] = pd.to_datetime(df_reset["datetime"])

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df_reset.columns:
                df_reset[col] = df_reset[col].astype(float)

        data_feed = PandasDataFrameFeed(
            dataname=df_reset,
            datetime=0,
            open="open",
            high="high",
            low="low",
            close="close",
            volume="volume",
            openinterest=-1,
        )

        cerebro = bt.Cerebro()
        cerebro.adddata(data_feed)

        if target_timeframe == "4h":
            cerebro.replaydata(
                data_feed,
                timeframe=bt.TimeFrame.Minutes,
                compression=240,
            )
        elif target_timeframe == "1d":
            cerebro.replaydata(
                data_feed,
                timeframe=bt.TimeFrame.Days,
                compression=24,
            )

        cerebro.addstrategy(_EmptyStrategy)

        cerebro.run()
        
        result_rows = []
        
        # En backtrader con replaydata, solo se puede acceder a la barra actual (índice 0)
        # ya que el replay va construyendo la barra en tiempo real
        if len(cerebro.datas) > 1:
            replay_data = cerebro.datas[1]
            data_len = len(replay_data)
            logger.debug(f"Reconstructed {target_timeframe}: {data_len} bars")
            
            # Solo accedemos a la barra 0 (la barra actual en construcción)
            if data_len > 0:
                try:
                    result_rows.append(
                        {
                            "timestamp": replay_data.datetime[0],
                            "open": float(replay_data.open[0]),
                            "high": float(replay_data.high[0]),
                            "low": float(replay_data.low[0]),
                            "close": float(replay_data.close[0]),
                            "volume": float(replay_data.volume[0] if replay_data.volume[0] else 0),
                        }
                    )
                except Exception as e:
                    logger.warning("Error accessing replay data: {}", e)

        if result_rows:
            result_df = pd.DataFrame(result_rows)
            # Convertir a datetime64[ms, UTC] para consistencia con datos de Binance
            result_df["timestamp"] = pd.to_datetime(result_df["timestamp"]).dt.tz_localize("UTC")
            return result_df

        return pd.DataFrame(
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

    async def stream(self) -> AsyncGenerator[dict[str, pd.DataFrame], None]:
        """Async generator that yields one window per step with reconstructed timeframes."""
        self._active = True
        delay = self._refresh / self._speed

        for step in range(self._max_steps):
            if not self._active:
                logger.info("Replay stopped at step {}/{}", step, self._max_steps)
                return

            window_1h = self._data_1h.iloc[step : step + self._window_size].copy()
            window_1h = window_1h.reset_index()

            logger.debug("Replay step {}/{}", step + 1, self._max_steps)

            logger.info(f"[Step {step + 1}/{self._max_steps}]")
            logger.info("  1h:        {} | O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f}".format(
                window_1h["timestamp"].iloc[0].strftime("%Y-%m-%d %H:%M"),
                window_1h["open"].iloc[0],
                window_1h["high"].iloc[0],
                window_1h["low"].iloc[0],
                window_1h["close"].iloc[0],
                window_1h["volume"].iloc[0],
            ))

            df_4h = self._reconstruct_timeframe(window_1h, "4h", 240)
            if not df_4h.empty:
                logger.info("  4h_recon:  {} | O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f}".format(
                    df_4h["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                    df_4h["open"].iloc[0],
                    df_4h["high"].iloc[0],
                    df_4h["low"].iloc[0],
                    df_4h["close"].iloc[0],
                    df_4h["volume"].iloc[0],
                ))

            df_1d = self._reconstruct_timeframe(window_1h, "1d", 24)
            if not df_1d.empty:
                logger.info("  1d_recon:  {} | O:{:.2f} H:{:.2f} L:{:.2f} C:{:.2f} V:{:.0f}".format(
                    df_1d["timestamp"].iloc[0].strftime("%Y-%m-%d"),
                    df_1d["open"].iloc[0],
                    df_1d["high"].iloc[0],
                    df_1d["low"].iloc[0],
                    df_1d["close"].iloc[0],
                    df_1d["volume"].iloc[0],
                ))

            yield {
                "1h": window_1h,
                "4h_recon": df_4h,
                "1d_recon": df_1d,
            }

            await asyncio.sleep(delay)

        self._active = False
        logger.success("Replay completed — {} steps emitted", self._max_steps)

    def stop(self) -> None:
        """Stop the replay mid-stream."""
        self._active = False

    @property
    def is_active(self) -> bool:
        """Check if replay is currently running."""
        return self._active

    @property
    def total_steps(self) -> int:
        """Total number of replay steps available."""
        return self._max_steps