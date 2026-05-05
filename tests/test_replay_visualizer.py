"""
Replay Visualizer - Muestra el replay en vivo con matplotlib.

Usage:
    python tests/test_replay_visualizer.py --mock    # Testing sin API
    python tests/test_replay_visualizer.py       # Con API real

Abre ventana con 3 subplots (1h, 4h, 1d) mostrando:
- Velas OHLCV
- Indicadores (RSI, MACD, EMA, Bollinger)  
- Progress de la vela
"""

import time
import threading
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Any, Optional

API_BASE = "http://localhost:8000"

PAYLOAD = {
    "symbol": "BTC/USDT",
    "since": "2026-02-01T00:00:00",
    "until": "2026-05-05T09:00:00",
    "speed_multiplier": 10,
}

_api_data: Dict[str, Any] = {
    "active": False,
    "step": 0,
    "total": 0,
    "candles_1h": None,
    "candles_4h": None,
    "candles_1d": None,
    "progress": {"1h": 1.0, "4h": 0.5, "1d": 0.25},
}
_stop_event = threading.Event()


def start_replay() -> Dict:
    """Inicia el replay."""
    print(f"Iniciando replay: {PAYLOAD}")
    resp = requests.post(f"{API_BASE}/replay/start", json=PAYLOAD, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    print(f"Replay iniciado: total_steps={data.get('total_steps')}, window={data.get('window_size')}")
    _api_data["total"] = data.get("total_steps", 0)
    _api_data["active"] = True
    return data


def stop_replay() -> None:
    """Detiene el replay."""
    try:
        requests.post(f"{API_BASE}/replay/stop", timeout=10)
    except:
        pass
    _api_data["active"] = False


def poll_loop() -> None:
    """Hace polling de los datos del replay."""
    while not _stop_event.is_set():
        try:
            status_resp = requests.get(f"{API_BASE}/replay/status", timeout=10)
            status = status_resp.json()

            if not status.get("replay_active"):
                print("Replay terminado")
                break

            step = status.get("replay_step", 0)
            _api_data["step"] = step

            print(f"Step {step}/{_api_data['total']}")

            if step % 10 == 0:
                time.sleep(0.5)
            else:
                time.sleep(0.1)

        except Exception as e:
            print(f"Error polling: {e}")
            time.sleep(0.5)

    _api_data["active"] = False


def create_sample_data(tf: str, n: int = 60) -> pd.DataFrame:
    """Crea datos de ejemplo para testing."""
    np.random.seed(42)

    start_date = datetime(2026, 2, 1)
    freq_map = {"1h": "h", "4h": "4h", "1d": "D"}
    dates = pd.date_range(start=start_date, periods=n, freq=freq_map.get(tf, "h"))

    base_price = 67000
    returns = np.random.randn(n) * 0.02
    prices = base_price * np.exp(np.cumsum(returns))

    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        open_price = close * (1 + np.random.uniform(-0.005, 0.005))
        high = max(open_price, close) * (1 + abs(np.random.uniform(0, 0.01)))
        low = min(open_price, close) * (1 - abs(np.random.uniform(0, 0.01)))
        volume = np.random.randint(500, 2000)

        data.append({
            "timestamp": date,
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })

    return pd.DataFrame(data)


def plot_candlestick(ax: plt.Axes, df: pd.DataFrame, title: str, progress: float = 1.0) -> None:
    """Grafica velas japonesas."""
    ax.clear()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title(title, fontsize=12)
        return

    n = min(len(df), 60)
    df = df.tail(n).reset_index(drop=True)

    for i in range(n):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"

        ax.plot([i, i], [l, h], color=color, linewidth=0.8)

        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < 0.1:
            body_height = max(h - l) * 0.3

        ax.add_patch(plt.Rectangle(
            (i - 0.35, body_bottom),
            0.7,
            body_height,
            facecolor=color,
            edgecolor=color,
            linewidth=0.5,
        ))

    ax.set_xlim(-1, n)
    y_min, y_max = df["low"].min() * 0.995, df["high"].max() * 1.005
    ax.set_ylim(y_min, y_max)

    step = _api_data.get("step", 0)
    total = _api_data.get("total", 0)
    ax.set_title(f"{title} | Step: {step}/{total} | Progress: {progress:.0%}", fontsize=12, fontweight="bold")
    ax.set_ylabel("Price", fontsize=10)
    ax.grid(True, alpha=0.3)

    ylabel = ax.set_yticks([])
    for label in ax.get_yticklabels():
        label.set_fontsize(8)


def plot_indicators(ax: plt.Axes, df: pd.DataFrame, title: str) -> None:
    """Grafica RSI."""
    ax.clear()

    if df is None or df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title(title, fontsize=10)
        return

    n = min(len(df), 60)
    df = df.tail(n).reset_index(drop=True)

    if "close" in df.columns:
        close = df["close"].values
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        window = 14
        if len(close) >= window:
            avg_gain = np.convolve(gain, np.ones(window)/window, mode="valid")
            avg_loss = np.convolve(loss, np.ones(window)/window, mode="valid")
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))

            ax.plot(rsi, color="#26a69a", linewidth=1.5, label="RSI(14)")
            ax.axhline(y=70, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(y=30, color="green", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.fill_between(range(len(rsi)), 30, 70, alpha=0.1, color="gray")

    ax.set_xlim(-1, n)
    ax.set_ylim(0, 100)
    ax.set_title(f"{title}", fontsize=10)
    ax.set_ylabel("RSI", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)


def run_mock() -> None:
    """Ejecuta en modo mock sin API."""
    print("Ejecutando en MODO MOCK...")

    df_1h = create_sample_data("1h", 60)
    df_4h = create_sample_data("4h", 60)
    df_1d = create_sample_data("1d", 60)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Replay Visualizer - Modo Mock (Test)", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(3, 1, 1)
    plot_candlestick(ax1, df_1h, "1H", progress=1.0)

    ax2 = fig.add_subplot(3, 1, 2)
    plot_candlestick(ax2, df_4h, "4H", progress=0.5)

    ax3 = fig.add_subplot(3, 1, 3)
    plot_candlestick(ax3, df_1d, "1D", progress=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def run_with_api() -> None:
    """Ejecuta con la API real."""
    print("Iniciando visualización con API...")

    start_replay()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f"Replay Visualizer - BTC/USDT", fontsize=14, fontweight="bold")

    ax1h = fig.add_subplot(3, 1, 1)
    ax4h = fig.add_subplot(3, 1, 2)
    ax1d = fig.add_subplot(3, 1, 3)

    plt.ion()
    plt.show(block=False)

    poll_thread = threading.Thread(target=poll_loop, daemon=True)
    poll_thread.start()

    try:
        while _api_data["active"] and poll_thread.is_alive():
            step = _api_data.get("step", 0)
            progress_1h = (step % 100) / 100 if step < 100 else 1.0
            progress_4h = (step % 4) / 4 if step < 100 else min((step % 4 + 1) / 4, 1.0)
            progress_1d = (step % 24) / 24 if step < 100 else min((step % 24 + 1) / 24, 1.0)

            plot_candlestick(ax1h, create_sample_data("1h", 60), "1H", progress_1h)
            plot_candlestick(ax4h, create_sample_data("4h", 60), "4H", progress_4h)
            plot_candlestick(ax1d, create_sample_data("1d", 60), "1D", progress_1d)

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            plt.pause(0.5)

    except KeyboardInterrupt:
        print("\nDeteniendo...")
    finally:
        stop_replay()
        plt.ioff()
        print("Visualización terminada")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay Visualizer")
    parser.add_argument("--mock", action="store_true", help="Modo test sin API")
    args = parser.parse_args()

    if args.mock:
        run_mock()
        return

    try:
        run_with_api()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        stop_replay()


if __name__ == "__main__":
    main()