"""
Label Gantt Chart — API-fetched data, same time axis
=====================================================
Fetches data from APIs (not DB), runs model predictions, plots Gantt chart.
"""

from __future__ import annotations

import logging
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from owl.config import (
    INPUT_WINDOW_BARS_HOURLY,
    INPUT_WINDOW_MINUTES,
    NUM_CATEGORIES,
    PREDICTION_WINDOW_BARS_HOURLY,
    PREDICTION_WINDOW_MINUTES,
    REPORT_DPI,
    WINDOW_STRIDE,
    WINDOW_STRIDE_HOURLY,
)
from owl.data.query_engine import fetch_and_merge
from owl.data.ticker_config import get_symbol_to_sector_idx, get_tickers
from owl.preprocessing.pipeline import PreprocessingPipeline

logger = logging.getLogger(__name__)
_PALETTE = plt.cm.RdYlGn(np.linspace(0.1, 0.9, NUM_CATEGORIES))


def fetch_symbol_data(
    symbol: str,
    start_date: date,
    end_date: date,
    frequency: str = "minute",
) -> pd.DataFrame | None:
    """Fetch merged price + fundamentals from API (no DB write)."""
    case_id = f"gantt_{symbol}_{start_date}"
    return fetch_and_merge(symbol, start_date, end_date, case_id, frequency=frequency)


def build_windows_and_predict(
    df: pd.DataFrame,
    pipeline: PreprocessingPipeline,
    model: torch.nn.Module,
    device: torch.device,
    symbol: str,
    sector_idx: int = 0,
    input_window: int | None = None,
    prediction_window: int | None = None,
    stride: int | None = None,
) -> list[tuple[pd.Timestamp, int]]:
    """Return list of (timestamp, predicted_label) for each rolling window."""
    inp_win = input_window or INPUT_WINDOW_MINUTES
    pred_win = prediction_window or PREDICTION_WINDOW_MINUTES
    stride_val = stride if stride is not None else WINDOW_STRIDE
    total = inp_win + pred_win

    cols = pipeline.feature_columns if pipeline else [
        c for c in df.columns
        if c not in ("id", "case_id", "symbol", "timestamp", "frequency")
    ]
    results = []
    for start in range(0, max(len(df) - total + 1, 0), stride_val if stride_val > 0 else total):
        end = start + total
        window = df.iloc[start:end]
        inp = window.iloc[:inp_win]
        if len(inp) < inp_win:
            continue
        arr = inp[cols].values.astype(np.float32)
        np.nan_to_num(arr, copy=False)
        X = torch.from_numpy(arr).unsqueeze(0).to(device)
        sector = torch.tensor([sector_idx], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(X, sector_idx=sector)
        pred = int(logits.argmax(dim=1).item())
        ts = pd.to_datetime(inp["timestamp"].iloc[-1]) if "timestamp" in inp.columns else pd.Timestamp.now()
        results.append((ts, pred))
    return results


def plot_label_gantt(
    segments_by_symbol: dict[str, list[tuple[pd.Timestamp, int]]],
    save_path: Path | None = None,
) -> plt.Figure:
    """Plot horizontal bar (Gantt) chart: y=symbols, x=time, color=label."""
    symbols = list(segments_by_symbol.keys())
    if not symbols:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, ax = plt.subplots(figsize=(14, max(6, len(symbols) * 0.5)))
    y_height = 0.8
    all_ts = [ts for sym, segs in segments_by_symbol.items() for ts, _ in segs]
    if not all_ts:
        ax.text(0.5, 0.5, "No segments", ha="center", va="center", transform=ax.transAxes)
        return fig
    t_min, t_max = min(all_ts), max(all_ts)
    span_days = mdates.date2num(t_max) - mdates.date2num(t_min)
    segment_width = max(span_days / 150, 1 / 96)

    for i, sym in enumerate(symbols):
        for ts, label in segments_by_symbol[sym]:
            x = mdates.date2num(ts)
            color = _PALETTE[label % len(_PALETTE)]
            ax.barh(i, segment_width, left=x, height=y_height, color=color, edgecolor="none")

    ax.set_yticks(range(len(symbols)))
    ax.set_yticklabels(symbols, fontsize=9)
    ax.set_ylim(-0.5, len(symbols) - 0.5)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.set_xlabel("Time (shared)")
    ax.set_title("Label Gantt — Predicted category over time (API-fetched)")
    ax.invert_yaxis()
    fig.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=REPORT_DPI, bbox_inches="tight")
        logger.info("Label Gantt → %s", save_path)
    return fig


def run_label_gantt(
    trainer,
    pipeline: PreprocessingPipeline,
    n_symbols: int = 20,
    days: int = 14,
    save_path: Path | None = None,
    frequency: str = "minute",
    input_window: int | None = None,
    prediction_window: int | None = None,
    stride: int | None = None,
) -> bool:
    """Fetch from API, predict, plot Gantt. Returns True if successful.

    Use frequency='hour' and hourly window params for hourly pipeline.
    """
    tickers = get_tickers()
    if not tickers:
        return False
    symbols = random.sample(tickers, min(n_symbols, len(tickers)))
    end_date = min(date.today(), datetime.strptime("2025-12-31", "%Y-%m-%d").date())
    start_date = end_date - timedelta(days=days)
    sym_to_sector = get_symbol_to_sector_idx()

    if frequency == "hour":
        inp_win = input_window or INPUT_WINDOW_BARS_HOURLY
        pred_win = prediction_window or PREDICTION_WINDOW_BARS_HOURLY
        stride_val = stride if stride is not None else WINDOW_STRIDE_HOURLY
    else:
        inp_win = input_window or INPUT_WINDOW_MINUTES
        pred_win = prediction_window or PREDICTION_WINDOW_MINUTES
        stride_val = stride if stride is not None else WINDOW_STRIDE
    total = inp_win + pred_win

    segments_by_symbol = {}
    for sym in symbols:
        df = fetch_symbol_data(sym, start_date, end_date, frequency=frequency)
        if df is None or len(df) < total:
            continue
        df = pipeline.transform(df)
        segs = build_windows_and_predict(
            df, pipeline, trainer.model, trainer.device, sym,
            sym_to_sector.get(sym, 0),
            input_window=inp_win,
            prediction_window=pred_win,
            stride=stride_val,
        )
        if segs:
            segments_by_symbol[sym] = segs

    if not segments_by_symbol:
        return False
    from owl.config import REPORT_DIR
    path = save_path if save_path is not None else REPORT_DIR / "cnn" / "label_gantt.png"
    plot_label_gantt(segments_by_symbol, save_path=path)
    return True
