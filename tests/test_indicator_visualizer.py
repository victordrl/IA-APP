"""
Indicator Visualizer — Abre una ventana separada por grupo de indicadores.

Grupos (uno por ventana):
  1. OHLCV — velas 1h / 4h / 1d
  2. VELOCIDAD — MON, ROC, RSI, STOCH, WILLIAMS_R, CCI
  3. TENDENCIA — MACD, ADX, EMA, ICHIMOKU
  4. AMPLITUD — Bollinger Bands, Keltner Channel
  5. LIQUIDEZ — CMF, OBV, ELDER, VWAP, EOM, VOL PROFILE

Usage:
    python tests/test_indicator_visualizer.py
    python tests/test_indicator_visualizer.py --rows 100
    python tests/test_indicator_visualizer.py --mode realtime         # Ultimas 150 velas en tiempo real
    python tests/test_indicator_visualizer.py --symbol ETH/USDT --mode realtime
"""

import argparse
import os
import requests
import pandas as pd
import numpy as np
import matplotlib
from datetime import datetime
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

API_BASE = "http://localhost:8000"

COLORS = {
    "MON": "#FF5722",
    "ROC": "#03A9F4",
    "RSI_6": "#FF6B6B",
    "RSI_14": "#FF9F43",
    "RSI_24": "#FECA57",
    "RSI_EMA_6": "#FF6B6B",
    "RSI_EMA_14": "#FF9F43",
    "RSI_EMA_24": "#FECA57",
    "STOCH_K": "#26A69A",
    "STOCH_D": "#7EC8E3",
    "WILLIAMS_R": "#9B59B6",
    "CCI": "#E91E63",
    "MACD_LINE": "#00BCD4",
    "MACD_SIGNAL": "#0097A7",
    "MACD_HIST": "#607D8B",
    "ADX": "#2196F3",
    "DI_PLUS": "#4CAF50",
    "DI_MINUS": "#F44336",
    "EMA_7": "#FF9800",
    "EMA_22": "#E65100",
    "EMA_99": "#BF360C",
    "ICHIMOKU_TENKAN": "#FF9800",
    "ICHIMOKU_KIJUN": "#2196F3",
    "ICHIMOKU_SA": "#9C27B0",
    "ICHIMOKU_SB": "#3F51B5",
    "ICHIMOKU_CHIKOU": "#00BCD4",
    "BB_UPPER": "#8BC34A",
    "BB_MIDDLE": "#CDDC39",
    "BB_LOWER": "#8BC34A",
    "BB_WIDTH": "#689F38",
    "KELTNER_UPPER": "#795548",
    "KELTNER_MIDDLE": "#A1887F",
    "KELTNER_LOWER": "#795548",
    "CMF": "#607D8B",
    "OBV": "#9C27B0",
    "ELDER_BULL": "#4CAF50",
    "ELDER_BEAR": "#F44336",
    "VWAP": "#FFEB3B",
    "EOM": "#9E9E9E",
}


def create_mock_data(symbol: str, n_candles: int = 80) -> dict:
    """Crea datos simulados (mock) para testing sin API."""
    np.random.seed(42)
    base_price = 67000 if "BTC" in symbol else 3500

    def make_tf_data(n: int) -> list:
        dates = pd.date_range(start="2026-01-01", periods=n, freq="h" if n > 24 else "D")
        prices = [base_price]
        for _ in range(n - 1):
            prices.append(prices[-1] * (1 + np.random.randn() * 0.02))

        data = []
        for i, (ts, close) in enumerate(zip(dates, prices)):
            open_p = close * (1 + np.random.uniform(-0.01, 0.01))
            high = max(open_p, close) * (1 + abs(np.random.uniform(0, 0.015)))
            low = min(open_p, close) * (1 - abs(np.random.uniform(0, 0.015)))
            vol = np.random.randint(500, 5000)
            data.append({
                "timestamp": ts.isoformat(),
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "volume": vol
            })
        return data

    data = {}
    for tf in ["1h", "4h", "1d"]:
        n = n_candles if tf == "1h" else n_candles // (4 if tf == "4h" else 24)
        n = max(n, 20)
        data[tf] = make_tf_data(n)

    print(f"Mock data creado: 1h={len(data['1h'])} velas, 4h={len(data['4h'])}, 1d={len(data['1d'])}")
    return data


def fetch_realtime_data(host: str, symbol: str, n_steps: int, normalized: bool) -> dict:
    print(f"Llamando {host}/realtime/run (normalized={normalized}) ...")
    payload = {
        "symbol": symbol,
        "n_steps": n_steps,
        "sync_type": "timeframe",
        "sync_version": "indicators",
        "normalized": normalized
    }
    resp = requests.post(f"{host}/realtime/run", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    # Convert list of steps into dict of timeframes
    parsed_data = {"1h": [], "4h": [], "1d": []}
    for step in data.get("results", []):
        ts = step.get("timestamp")
        for tf in ["1h", "4h", "1d"]:
            if tf in step:
                row = step[tf].copy()
                row["timestamp"] = ts
                parsed_data[tf].append(row)
                
    dfs = {}
    for tf, records in parsed_data.items():
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        dfs[tf] = df
    return dfs


def plot_candlestick(ax, df, n_show):
    ax.clear()
    if df is None or df.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center", transform=ax.transAxes)
        return
    df = df.tail(n_show).reset_index(drop=True)
    n = len(df)
    price_range = df["high"].max() - df["low"].min()
    for i in range(n):
        row = df.iloc[i]
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        color = "#26a69a" if c >= o else "#ef5350"
        ax.plot([i, i], [l, h], color=color, linewidth=0.8)
        body_bottom = min(o, c)
        body_height = abs(c - o)
        if body_height < price_range * 0.005:
            body_height = price_range * 0.01
        ax.add_patch(plt.Rectangle((i - 0.35, body_bottom), 0.7, body_height,
                                   facecolor=color, edgecolor=color, linewidth=0.5))
    ax.set_xlim(-1, n)
    ax.grid(True, alpha=0.3)


def plot_lines(ax, df, cols, n_show, ylim=None, title=""):
    ax.clear()
    for col in cols:
        if col not in df.columns:
            continue
        series = df[col].tail(n_show).dropna()
        if series.empty:
            continue
        color = COLORS.get(col, "#888888")
        ax.plot(range(len(series)), series.values, color=color, linewidth=1.3, label=col)
    ax.set_xlim(-1, n_show)
    if ylim:
        ax.set_ylim(*ylim)
    if cols:
        ax.legend(loc="upper right", fontsize=7)
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=7)


def plot_macd(ax, df, n_show):
    ax.clear()
    line = df["MACD_LINE"].tail(n_show).dropna()
    signal = df["MACD_SIGNAL"].tail(n_show).dropna()
    hist = df["MACD_HIST"].tail(n_show).dropna()

    if not line.empty:
        ax.plot(range(len(line)), line.values, color=COLORS["MACD_LINE"], linewidth=1.3, label="MACD")
    if not signal.empty:
        ax.plot(range(len(signal)), signal.values, color=COLORS["MACD_SIGNAL"], linewidth=1.3, label="Signal")

    if not hist.empty:
        x = np.arange(len(hist))
        pos = hist.values >= 0
        ax.bar(x[pos], hist.values[pos], color=COLORS["MACD_HIST"], alpha=0.5, width=0.8)
        neg = hist.values < 0
        ax.bar(x[neg], hist.values[neg], color="#F44336", alpha=0.5, width=0.8)

    ax.set_xlim(-1, n_show)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("MACD LINE / SIGNAL / HIST", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=7)


def plot_ohlcv_window(dfs: dict, n_show: int, symbol: str, since: str, until: str, prefix: str = ""):
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.canvas.manager.set_window_title(f"{prefix}OHLCV — 1h / 4h / 1d")
    tf_labels = ["1h", "4h", "1d"]
    for ax, tf in zip(axs, tf_labels):
        plot_candlestick(ax, dfs.get(tf), n_show)
        ax.set_ylabel("Price", fontsize=8)
        ax.set_title(f"{tf.upper()} — BTC/USDT", fontsize=10, fontweight="bold")

        ts = dfs[tf]["timestamp"].tail(n_show) if tf in dfs else []
        if len(ts) > 0:
            tick_every = max(1, len(ts) // 8)
            ticks = range(0, len(ts), tick_every)
            ax.set_xticks(list(ticks))
            ax.set_xticklabels([ts.iloc[i].strftime("%m-%d") for i in ticks], rotation=45, fontsize=7)

    fig.suptitle(f"OHLCV — {symbol} | {since[:10]} -> {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def plot_sqz_histogram(ax, df, n_show):
    ax.clear()
    if "SQZ_MOM" not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return
    series = df["SQZ_MOM"].tail(n_show).dropna().reset_index(drop=True)
    if series.empty:
        return
    n = len(series)
    vals = series.values
    colors = ["#4CAF50" if v >= 0 else "#FF0000" for v in vals]
    ax.bar(range(n), vals, color=colors, width=0.8, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.set_xlim(-1, n)
    ymin, ymax = vals.min(), vals.max()
    if ymin == ymax:
        ymin, ymax = -1, 1
    margin = (ymax - ymin) * 0.1
    ax.set_ylim(ymin - margin, ymax + margin)
    ax.grid(True, alpha=0.25)
    ax.tick_params(axis="both", labelsize=7)


def plot_velocidad_window(dfs: dict, n_show: int, symbol: str, since: str, until: str, prefix: str = ""):
    rows = [
        ("SQZ MOM (histograma)", "sqz", None),
        ("MON / ROC", ["MON", "ROC"], None),
        ("RSI (6 / 14 / 24)", ["RSI_6", "RSI_14", "RSI_24"], (0, 1) if "NORM" in prefix else (0, 100)),
        ("RSI EMA (6 / 14 / 24)", ["RSI_EMA_6", "RSI_EMA_14", "RSI_EMA_24"], (0, 1) if "NORM" in prefix else (0, 100)),
        ("STOCH %K / %D", ["STOCH_K", "STOCH_D"], (0, 1)),
        ("WILLIAMS %R", ["WILLIAMS_R"], (-1, 0) if "NORM" in prefix else (-100, 0)),
        ("CCI", ["CCI"], None),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title(f"{prefix}VELOCIDAD — MON, ROC, RSI, STOCH, WILLIAMS, CCI")
    tf_list = ["1h", "4h", "1d"]

    for row_i, (title, cols, ylim) in enumerate(rows):
        for ax, tf in zip(axs[row_i], tf_list):
            df = dfs.get(tf)
            if df is None:
                continue
            if cols == "sqz":
                plot_sqz_histogram(ax, df, n_show)
                if row_i == 0:
                    ax.set_title(f"{tf.upper()} — {title}", fontsize=9, fontweight="bold")
                ax.tick_params(axis="both", labelsize=7)
                continue
            for col in cols:
                if col not in df.columns:
                    continue
                series = df[col].tail(n_show).dropna()
                if series.empty:
                    continue
                color = COLORS.get(col, "#888888")
                ax.plot(range(len(series)), series.values, color=color, linewidth=1.3, label=col)
            ax.set_xlim(-1, n_show)
            if ylim:
                ax.set_ylim(*ylim)
                ax.fill_between(range(n_show), ylim[0], ylim[1], alpha=0.05, color="gray")
            if row_i == 0:
                ax.set_title(f"{tf.upper()} — {title}", fontsize=9, fontweight="bold")
            elif tf == "1h":
                ax.set_ylabel(cols[0], fontsize=7)
            ax.legend(loc="upper right", fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=7)
            if row_i == n_rows - 1:
                ts = df["timestamp"].tail(n_show)
                tick_every = max(1, len(ts) // 8)
                ticks = list(range(0, len(ts), tick_every))
                ax.set_xticks(ticks)
                ax.set_xticklabels([ts.iloc[i].strftime("%m-%d") for i in ticks if i < len(ts)], rotation=45, fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.suptitle(f"VELOCIDAD — {symbol} | {since[:10]} -> {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_tendencia_window(dfs: dict, n_show: int, symbol: str, since: str, until: str, prefix: str = ""):
    rows = [
        ("MACD", None),
        ("ADX / DI+ / DI-", ["ADX", "DI_PLUS", "DI_MINUS"]),
        ("EMA (7 / 22 / 99)", ["EMA_7", "EMA_22", "EMA_99"]),
        ("ICHIMOKU TENKAN / KIJUN", ["ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN"]),
        ("ICHIMOKU SA / SB", ["ICHIMOKU_SA", "ICHIMOKU_SB"]),
        ("ICHIMOKU CHIKOU", ["ICHIMOKU_CHIKOU"]),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title(f"{prefix}TENDENCIA — MACD, ADX, EMA, ICHIMOKU")
    tf_list = ["1h", "4h", "1d"]

    for row_i, (title, cols) in enumerate(rows):
        for ax, tf in zip(axs[row_i], tf_list):
            df = dfs.get(tf)
            if df is None:
                continue
            if cols is None:
                plot_macd(ax, df, n_show)
            else:
                for col in cols:
                    if col not in df.columns:
                        continue
                    series = df[col].tail(n_show).dropna()
                    if series.empty:
                        continue
                    color = COLORS.get(col, "#888888")
                    ax.plot(range(len(series)), series.values, color=color, linewidth=1.3, label=col)
                ax.set_xlim(-1, n_show)
                ax.legend(loc="upper right", fontsize=6)
                ax.grid(True, alpha=0.25)
            if row_i == 0:
                ax.set_title(f"{tf.upper()} — {title}", fontsize=9, fontweight="bold")
            elif tf == "1h":
                ax.set_ylabel(title.split(" ")[0], fontsize=7)
            ax.tick_params(axis="both", labelsize=7)
            if row_i == n_rows - 1:
                ts = df["timestamp"].tail(n_show)
                tick_every = max(1, len(ts) // 8)
                ticks = list(range(0, len(ts), tick_every))
                ax.set_xticks(ticks)
                ax.set_xticklabels([ts.iloc[i].strftime("%m-%d") for i in ticks if i < len(ts)], rotation=45, fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.suptitle(f"TENDENCIA — {symbol} | {since[:10]} -> {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_amplitud_window(dfs: dict, n_show: int, symbol: str, since: str, until: str, prefix: str = ""):
    rows = [
        ("Bollinger Bands — Upper / Middle / Lower", ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]),
        ("Bollinger Width", ["BB_WIDTH"], None),
        ("Keltner Channel — Upper / Middle / Lower", ["KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"]),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 2), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title(f"{prefix}AMPLITUD — Bollinger, Keltner")
    tf_list = ["1h", "4h", "1d"]

    for row_i, (title, cols, *ylim_extra) in enumerate(rows):
        ylim = ylim_extra[0] if ylim_extra else None
        for ax, tf in zip(axs[row_i], tf_list):
            df = dfs.get(tf)
            if df is None:
                continue
            for col in cols:
                if col not in df.columns:
                    continue
                series = df[col].tail(n_show).dropna()
                if series.empty:
                    continue
                color = COLORS.get(col, "#888888")
                ax.plot(range(len(series)), series.values, color=color, linewidth=1.3, label=col)
            ax.set_xlim(-1, n_show)
            if ylim:
                ax.set_ylim(*ylim)
            if row_i == 0:
                ax.set_title(f"{tf.upper()} — {title}", fontsize=9, fontweight="bold")
            elif tf == "1h":
                ax.set_ylabel(cols[0].split("_")[0], fontsize=7)
            ax.legend(loc="upper right", fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=7)
            if row_i == n_rows - 1:
                ts = df["timestamp"].tail(n_show)
                tick_every = max(1, len(ts) // 8)
                ticks = list(range(0, len(ts), tick_every))
                ax.set_xticks(ticks)
                ax.set_xticklabels([ts.iloc[i].strftime("%m-%d") for i in ticks if i < len(ts)], rotation=45, fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.suptitle(f"AMPLITUD — {symbol} | {since[:10]} -> {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_liquidez_window(dfs: dict, n_show: int, symbol: str, since: str, until: str, prefix: str = ""):
    rows = [
        ("CMF", ["CMF"], (-1, 1)),
        ("OBV", ["OBV"], None),
        ("ELDER BULL / BEAR", ["ELDER_BULL", "ELDER_BEAR"], None),
        ("VWAP", ["VWAP"], None),
        ("EOM", ["EOM"], None),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title(f"{prefix}LIQUIDEZ — CMF, OBV, ELDER, VWAP, EOM")
    tf_list = ["1h", "4h", "1d"]

    for row_i, (title, cols, ylim) in enumerate(rows):
        for ax, tf in zip(axs[row_i], tf_list):
            df = dfs.get(tf)
            if df is None:
                continue
            for col in cols:
                if col not in df.columns:
                    continue
                series = df[col].tail(n_show).dropna()
                if series.empty:
                    continue
                color = COLORS.get(col, "#888888")
                ax.plot(range(len(series)), series.values, color=color, linewidth=1.3, label=col)
            ax.set_xlim(-1, n_show)
            if ylim:
                ax.set_ylim(*ylim)
            if row_i == 0:
                ax.set_title(f"{tf.upper()} — {title}", fontsize=9, fontweight="bold")
            elif tf == "1h":
                ax.set_ylabel(cols[0], fontsize=7)
            ax.legend(loc="upper right", fontsize=6)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=7)
            if row_i == n_rows - 1:
                ts = df["timestamp"].tail(n_show)
                tick_every = max(1, len(ts) // 8)
                ticks = list(range(0, len(ts), tick_every))
                ax.set_xticks(ticks)
                ax.set_xticklabels([ts.iloc[i].strftime("%m-%d") for i in ticks if i < len(ts)], rotation=45, fontsize=6)
            else:
                ax.set_xticklabels([])

    fig.suptitle(f"LIQUIDEZ — {symbol} | {since[:10]} -> {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def save_all_figures(output_dir: str):
    """Guarda todas las figuras abiertas como PNG en la carpeta especificada."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directorio creado: {output_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved_count = 0

    for fig_num in plt.get_fignums():
        fig = plt.figure(fig_num)
        window_title = fig.canvas.manager.get_window_title()
        safe_name = window_title.replace(" — ", "_").replace(" ", "_").replace("|", "_").replace("/", "_")
        filename = f"{timestamp}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f"Guardado: {filepath}")
        saved_count += 1

    return saved_count


def main():
    parser = argparse.ArgumentParser(description="Indicator Visualizer")
    parser.add_argument("--host", default=API_BASE)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--since", default="2026-01-01T00:00:00Z")
    parser.add_argument("--until", default="2026-11-07T00:00:00Z")
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--mode", default="realtime", choices=["historical", "realtime", "mock"])
    parser.add_argument("--output-dir", default="outputs", help="Directorio para guardar PNGs")
    parser.add_argument("--save-png", action="store_true", help="Guardar imágenes como PNG antes de mostrar ventanas")
    args = parser.parse_args()

    if args.mode == "mock":
        print(f"Modo MOCK: Generando datos simulados para {args.symbol} ...")
        # El resto del modo mock se omite por brevedad o lo deshabilitamos temporalmente
        pass
    elif args.mode == "realtime":
        print(f"Descargando {args.symbol} (realtime, ultimas 10 velas) NORMALIZED vs RAW ...")
        
        # 1. Fetch RAW (Sin normalizar)
        dfs_raw = fetch_realtime_data(args.host, args.symbol, n_steps=10, normalized=False)
        
        # 2. Fetch NORMALIZED
        dfs_norm = fetch_realtime_data(args.host, args.symbol, n_steps=10, normalized=True)
        
        if "1h" in dfs_raw and not dfs_raw["1h"].empty:
            ts = dfs_raw["1h"]["timestamp"]
            since_calc = ts.min().strftime("%Y-%m-%dT%H:%M:%SZ")
            until_calc = ts.max().strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            since_calc, until_calc = "realtime", "realtime"
            
        n_rows = 10
        print("Abriendo 10 ventanas (5 RAW + 5 NORMALIZED)...")
        
        # Plot RAW
        plot_ohlcv_window(dfs_raw, n_rows, args.symbol, since_calc, until_calc, prefix="[RAW] ")
        plot_velocidad_window(dfs_raw, n_rows, args.symbol, since_calc, until_calc, prefix="[RAW] ")
        plot_tendencia_window(dfs_raw, n_rows, args.symbol, since_calc, until_calc, prefix="[RAW] ")
        plot_amplitud_window(dfs_raw, n_rows, args.symbol, since_calc, until_calc, prefix="[RAW] ")
        plot_liquidez_window(dfs_raw, n_rows, args.symbol, since_calc, until_calc, prefix="[RAW] ")
        
        # Plot NORMALIZED
        plot_ohlcv_window(dfs_norm, n_rows, args.symbol, since_calc, until_calc, prefix="[NORM] ")
        plot_velocidad_window(dfs_norm, n_rows, args.symbol, since_calc, until_calc, prefix="[NORM] ")
        plot_tendencia_window(dfs_norm, n_rows, args.symbol, since_calc, until_calc, prefix="[NORM] ")
        plot_amplitud_window(dfs_norm, n_rows, args.symbol, since_calc, until_calc, prefix="[NORM] ")
        plot_liquidez_window(dfs_norm, n_rows, args.symbol, since_calc, until_calc, prefix="[NORM] ")

    elif args.mode == "historical":
        print("Historical mode disabled for this demo.")

    print("5 ventanas abiertas — cerralas para terminar.")

    if args.save_png:
        print(f"\nGuardando imágenes PNG en: {args.output_dir}")
        saved = save_all_figures(args.output_dir)
        print(f"Total de imágenes guardadas: {saved}")

    plt.show()


if __name__ == "__main__":
    main()