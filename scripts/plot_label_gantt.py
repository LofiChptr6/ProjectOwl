#!/usr/bin/env python
"""
Label Gantt Chart — API-fetched data, same time axis
======================================================
Fetches data from APIs (not DB) for a defined date range, runs model predictions,
and plots a horizontal bar (Gantt) chart: y=symbols, x=time, color=predicted label.

Shows correlation between symbols and cyclical label behavior over time.

Usage::

    python scripts/plot_label_gantt.py
    python scripts/plot_label_gantt.py --model cnn --symbols 15 --days 14
"""

import argparse
import logging
import random
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import torch

# Add project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from owl.config import CHECKPOINT_DIR, REPORT_DIR
from owl.data.ticker_config import get_tickers
from owl.preprocessing.pipeline import PreprocessingPipeline
from owl.visualization.gantt import (
    build_windows_and_predict,
    fetch_symbol_data,
    plot_label_gantt,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("plot_label_gantt")

from owl.config import INPUT_WINDOW_MINUTES, PREDICTION_WINDOW_MINUTES, WINDOW_STRIDE

_TOTAL_WINDOW = INPUT_WINDOW_MINUTES + PREDICTION_WINDOW_MINUTES


def main():
    parser = argparse.ArgumentParser(description="Plot label Gantt chart from API data")
    parser.add_argument("--model", choices=["cnn", "transformer"], default="cnn")
    parser.add_argument("--symbols", type=int, default=20, help="Number of random symbols")
    parser.add_argument("--days", type=int, default=14, help="Days of data to fetch")
    parser.add_argument("--checkpoint", type=str, default="best")
    args = parser.parse_args()

    tickers = get_tickers()
    if not tickers:
        logger.error("No tickers — check config/tickers_by_sector.csv")
        sys.exit(1)
    symbols = random.sample(tickers, min(args.symbols, len(tickers)))
    end_date = min(date.today(), datetime.strptime("2025-12-31", "%Y-%m-%d").date())
    start_date = end_date - timedelta(days=args.days)

    logger.info("Fetching %d symbols from API for %s → %s", len(symbols), start_date, end_date)
    pipeline = PreprocessingPipeline()

    df_sample = fetch_symbol_data(symbols[0], start_date, end_date)
    if df_sample is None or len(df_sample) < _TOTAL_WINDOW:
        logger.error("First symbol %s has insufficient data — try different symbols/dates", symbols[0])
        sys.exit(1)
    pipeline.transform(df_sample)
    in_features = len(pipeline.feature_columns)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model == "cnn":
        from owl.models.cnn_model import CNNTrainer
        trainer = CNNTrainer(in_features=in_features)
    else:
        from owl.models.transformer_model import TransformerTrainer
        trainer = TransformerTrainer(in_features=in_features)
    trainer.model_name = args.model
    ckpt_path = CHECKPOINT_DIR / args.model / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s — run training first", ckpt_path)
        sys.exit(1)
    trainer.load(tag=args.checkpoint)
    trainer.model.eval().to(device)

    from owl.data.ticker_config import get_symbol_to_sector_idx
    sym_to_sector = get_symbol_to_sector_idx()

    segments_by_symbol: dict[str, list[tuple[pd.Timestamp, int]]] = {}
    for sym in symbols:
        df = fetch_symbol_data(sym, start_date, end_date)
        if df is None or len(df) < _TOTAL_WINDOW:
            logger.warning("Skipping %s — insufficient data", sym)
            continue
        df = pipeline.transform(df)
        sector_idx = sym_to_sector.get(sym, 0)
        segs = build_windows_and_predict(
            df, pipeline, trainer.model, device, sym, sector_idx
        )
        if segs:
            segments_by_symbol[sym] = segs
    if not segments_by_symbol:
        logger.error("No data — check API keys and date range")
        sys.exit(1)

    save_path = REPORT_DIR / args.model / "label_gantt.png"
    plot_label_gantt(segments_by_symbol, save_path=save_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
