"""
Indicator Visualizer — Abre una ventana separada por grupo de indicadores.

Grupos (uno por ventana):
  1. OHLCV — velas 1h / 4h / 1d
  2. VELOCIDAD — MON, ROC, RSI, STOCH, WILLIAMS_R, CCI
  3. TENDENCIA — MACD, ADX, EMA, ICHIMOKU
  4. AMPLITUD — Bollinger Bands, Keltner Channel
  5. LIQUIDEZ — CMF, OBV, ELDER, VWAP, EOM

Usage:
    python tests/test_indicator_visualizer.py
    python tests/test_indicator_visualizer.py --rows 100
    python tests/test_indicator_visualizer.py --symbol ETH/USDT --since 2026-01-01T00:00:00Z
"""

import argparse
import requests
import pandas as pd
import numpy as np
import matplotlib
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


def fetch_data(host: str, payload: dict) -> dict:
    print(f"Llamando {host}/data/historical ...")
    resp = requests.post(f"{host}/data/historical", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    for tf, records in data.items():
        print(f"  {tf}: {len(records)} velas")
    return data


def build_dfs(data: dict) -> dict[str, pd.DataFrame]:
    dfs = {}
    for tf, records in data.items():
        df = pd.DataFrame(records)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        suffix = f"_{tf}"
        rename = {c: c[:-len(suffix)] for c in df.columns if c.endswith(suffix)}
        if rename:
            df = df.rename(columns=rename)
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


def plot_ohlcv_window(dfs: dict, n_show: int, symbol: str, since: str, until: str):
    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=False)
    fig.canvas.manager.set_window_title("OHLCV — 1h / 4h / 1d")
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

    fig.suptitle(f"OHLCV — {symbol} | {since[:10]} → {until[:10]}", fontsize=12, fontweight="bold")
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


def plot_velocidad_window(dfs: dict, n_show: int, symbol: str, since: str, until: str):
    rows = [
        ("SQZ MOM (histograma)", "sqz", None),
        ("MON / ROC", ["MON", "ROC"], None),
        ("RSI (6 / 14 / 24)", ["RSI_6", "RSI_14", "RSI_24"], (0, 100)),
        ("RSI EMA (6 / 14 / 24)", ["RSI_EMA_6", "RSI_EMA_14", "RSI_EMA_24"], (0, 100)),
        ("STOCH %K / %D", ["STOCH_K", "STOCH_D"], (0, 1)),
        ("WILLIAMS %R", ["WILLIAMS_R"], (-100, 0)),
        ("CCI", ["CCI"], (-200, 200)),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title("VELOCIDAD — MON, ROC, RSI, STOCH, WILLIAMS, CCI")
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

    fig.suptitle(f"VELOCIDAD — {symbol} | {since[:10]} → {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_tendencia_window(dfs: dict, n_show: int, symbol: str, since: str, until: str):
    rows = [
        ("MACD", None),
        ("ADX / DI+ / DI-", ["ADX", "DI_PLUS", "DI_MINUS"]),
        ("EMA (22 / 50 / 100)", ["EMA_22", "EMA_50", "EMA_100"]),
        ("ICHIMOKU TENKAN / KIJUN", ["ICHIMOKU_TENKAN", "ICHIMOKU_KIJUN"]),
        ("ICHIMOKU SA / SB", ["ICHIMOKU_SA", "ICHIMOKU_SB"]),
        ("ICHIMOKU CHIKOU", ["ICHIMOKU_CHIKOU"]),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title("TENDENCIA — MACD, ADX, EMA, ICHIMOKU")
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

    fig.suptitle(f"TENDENCIA — {symbol} | {since[:10]} → {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_amplitud_window(dfs: dict, n_show: int, symbol: str, since: str, until: str):
    rows = [
        ("Bollinger Bands — Upper / Middle / Lower", ["BB_UPPER", "BB_MIDDLE", "BB_LOWER"]),
        ("Bollinger Width", ["BB_WIDTH"], (-0.5, 5)),
        ("Keltner Channel — Upper / Middle / Lower", ["KELTNER_UPPER", "KELTNER_MIDDLE", "KELTNER_LOWER"]),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 2), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title("AMPLITUD — Bollinger, Keltner")
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

    fig.suptitle(f"AMPLITUD — {symbol} | {since[:10]} → {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def plot_liquidez_window(dfs: dict, n_show: int, symbol: str, since: str, until: str):
    rows = [
        ("CMF", ["CMF"], (-1, 1)),
        ("OBV", ["OBV"], None),
        ("ELDER BULL / BEAR", ["ELDER_BULL", "ELDER_BEAR"], None),
        ("VWAP", ["VWAP"], None),
        ("EOM", ["EOM"], None),
    ]
    n_rows = len(rows)
    fig, axs = plt.subplots(n_rows, 3, figsize=(14, n_rows * 1.8), sharex=False, squeeze=False)
    fig.canvas.manager.set_window_title("LIQUIDEZ — CMF, OBV, ELDER, VWAP, EOM")
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

    fig.suptitle(f"LIQUIDEZ — {symbol} | {since[:10]} → {until[:10]}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])


def main():
    parser = argparse.ArgumentParser(description="Indicator Visualizer")
    parser.add_argument("--host", default=API_BASE)
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--since", default="2026-01-01T00:00:00Z")
    parser.add_argument("--until", default="2026-07-07T00:00:00Z")
    parser.add_argument("--rows", type=int, default=80)
    args = parser.parse_args()

    payload = {
        "symbol": args.symbol,
        "timeframes": ["1h", "4h", "1d"],
        "since": args.since,
        "until": args.until,
        "include_indicators": True,
    }

    print(f"Descargando {args.symbol} ({args.since} → {args.until}) ...")
    data = fetch_data(args.host, payload)
    dfs = build_dfs(data)

    print("Abriendo ventanas...")
    plot_ohlcv_window(dfs, args.rows, args.symbol, args.since, args.until)
    plot_velocidad_window(dfs, args.rows, args.symbol, args.since, args.until)
    plot_tendencia_window(dfs, args.rows, args.symbol, args.since, args.until)
    plot_amplitud_window(dfs, args.rows, args.symbol, args.since, args.until)
    plot_liquidez_window(dfs, args.rows, args.symbol, args.since, args.until)

    print("5 ventanas abiertas — cerralas para terminar.")
    plt.show()


if __name__ == "__main__":
    main()