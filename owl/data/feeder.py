"""
Data Feeder — PyTorch Dataset + DataLoader
============================================
Reads pre-populated cases from PostgreSQL, applies the preprocessing
pipeline, produces ``(input, target_category)`` pairs ready for the
CNN / Transformer models.

When ``WINDOW_STRIDE == 0`` each case maps to exactly one sample
(no rolling window).  Otherwise a rolling stride is used.

Multi-worker DataLoader is used so the DB reads and preprocessing run
concurrently on separate processes, keeping the GPU saturated.
"""

import logging
from collections import OrderedDict
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from owl.config import (
    BATCH_SIZE,
    INPUT_WINDOW_BARS_HOURLY,
    INPUT_WINDOW_MINUTES,
    NUM_CATEGORIES,
    NUM_WORKERS,
    PREDICTION_WINDOW_BARS_HOURLY,
    PREDICTION_WINDOW_MINUTES,
    TRAINING_TABLE,
    TRAINING_TABLE_HOURLY,
    VALIDATION_TABLE,
    VALIDATION_TABLE_HOURLY,
    WINDOW_STRIDE,
    WINDOW_STRIDE_HOURLY,
)
from owl.data.db import get_case_ids, get_case_row_count, load_case_data
from owl.data.ticker_config import get_symbol_to_sector_idx

logger = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────

def categorize_return(ret: float, thresholds: list[float] | None = None) -> int:
    """Map a scalar return to an integer category.

    When thresholds is None, uses NUM_CATEGORIES evenly spaced bins
    from -3% to +3% (no pre-set thresholds in config).
    """
    if thresholds is None:
        thresholds = list(
            np.linspace(-0.03, 0.03, NUM_CATEGORIES - 1)
        )
    for i, t in enumerate(thresholds):
        if ret < t:
            return i
    return len(thresholds)


# ── LRU-like case cache (per-worker) ─────────────────────────────────────

class _CaseCache:
    """In-process LRU cache for preprocessed DataFrames."""

    def __init__(self, maxsize: int = 50):
        self._data: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._maxsize = maxsize

    def get(self, case_id: str, table_name: str,
            pipeline=None) -> pd.DataFrame:
        if case_id in self._data:
            self._data.move_to_end(case_id)
            return self._data[case_id]

        df = load_case_data(case_id, table_name)
        if pipeline is not None:
            df = pipeline.transform(df)

        # evict oldest if full
        while len(self._data) >= self._maxsize:
            self._data.popitem(last=False)

        self._data[case_id] = df
        return df


# ── PyTorch Dataset ───────────────────────────────────────────────────────

class OwlDataset(Dataset):
    """Map-style dataset that yields ``(input_tensor, category_label)`` pairs
    from rolling windows of preprocessed case data.

    Parameters
    ----------
    table_name : which Postgres table to read from
    case_ids   : explicit list (if *None*, all IDs in *table_name* are used)
    pipeline   : a ``PreprocessingPipeline`` instance (or *None* for raw data)
    cache_size : how many full cases to keep in memory per DataLoader worker
    input_window, prediction_window, stride : bar counts (default: minute config)
    """

    def __init__(self, table_name: str, case_ids: list[str] | None = None,
                 pipeline=None, cache_size: int = 50, use_sector: bool = False,
                 input_window: int | None = None, prediction_window: int | None = None,
                 stride: int | None = None):
        self.table_name = table_name
        self.case_ids   = case_ids or get_case_ids(table_name)
        self.pipeline   = pipeline
        self._cache     = _CaseCache(maxsize=cache_size)
        self.use_sector = use_sector
        self._symbol_to_idx = get_symbol_to_sector_idx() if use_sector else {}

        self.input_window     = input_window or INPUT_WINDOW_MINUTES
        self.prediction_window = prediction_window or PREDICTION_WINDOW_MINUTES
        self.stride           = stride if stride is not None else WINDOW_STRIDE
        total_window         = self.input_window + self.prediction_window

        # build a flat index  →  (case_id, window_start_row)
        self.windows: list[tuple[str, int]] = []
        for cid in self.case_ids:
            n = get_case_row_count(cid, table_name)
            if self.stride <= 0:
                if n >= total_window:
                    self.windows.append((cid, 0))
            else:
                for start in range(0, max(n - total_window + 1, 0), self.stride):
                    self.windows.append((cid, start))

        logger.info("OwlDataset: %d windows from %d cases (%s)",
                     len(self.windows), len(self.case_ids), table_name)

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        case_id, start = self.windows[idx]
        df = self._cache.get(case_id, self.table_name, self.pipeline)

        total = self.input_window + self.prediction_window
        end = start + total
        if end > len(df):
            end = len(df)
            start = max(end - total, 0)

        window = df.iloc[start:end]

        # split into input / prediction
        inp = window.iloc[:self.input_window]
        tgt = window.iloc[self.input_window:]

        # target category = return over the prediction horizon
        if "close" in inp.columns and len(tgt) > 0:
            p0 = inp["close"].iloc[-1]
            p1 = tgt["close"].iloc[-1]
            ret = (p1 / p0) - 1 if p0 != 0 else 0.0
        else:
            ret = 0.0
        category = categorize_return(ret)

        # select feature columns (pipeline provides the list)
        if self.pipeline is not None:
            cols = self.pipeline.feature_columns
        else:
            cols = [c for c in inp.columns
                    if c not in ("id", "case_id", "symbol", "timestamp",
                                 "frequency")]

        inp_arr = inp[cols].values.astype(np.float32)
        np.nan_to_num(inp_arr, copy=False)

        if self.use_sector:
            symbol = str(inp["symbol"].iloc[0]) if "symbol" in inp.columns else ""
            sector_idx = self._symbol_to_idx.get(symbol, 0)
            return torch.from_numpy(inp_arr), category, sector_idx
        return torch.from_numpy(inp_arr), category

    # ──────────────────────────────────────────────────────────────────────

    def get_feature_dim(self) -> int:
        """Number of features per time-step."""
        sample = self[0][0]
        return sample.shape[-1]

    def get_sample_for_plot(self, idx: int) -> tuple:
        """Return (inp_arr, tgt_arr, symbol, start_ts, end_ts, category) for viz.

        inp_arr: (T_in, F) input window; tgt_arr: (T_tgt, F) future window.
        """
        case_id, start = self.windows[idx]
        df = self._cache.get(case_id, self.table_name, self.pipeline)

        total = self.input_window + self.prediction_window
        end = start + total
        if end > len(df):
            end = len(df)
            start = max(end - total, 0)

        window = df.iloc[start:end]
        inp = window.iloc[:self.input_window]
        tgt = window.iloc[self.input_window:]

        if self.pipeline is not None:
            cols = self.pipeline.feature_columns
        else:
            cols = [c for c in inp.columns
                    if c not in ("id", "case_id", "symbol", "timestamp", "frequency")]

        inp_arr = inp[cols].values.astype(np.float32)
        tgt_arr = tgt[cols].values.astype(np.float32) if len(tgt) > 0 else np.zeros((0, len(cols)))
        np.nan_to_num(inp_arr, copy=False)
        np.nan_to_num(tgt_arr, copy=False)

        symbol = str(inp["symbol"].iloc[0]) if "symbol" in inp.columns else ""
        start_ts = inp["timestamp"].iloc[0] if "timestamp" in inp.columns else None
        end_ts = inp["timestamp"].iloc[-1] if "timestamp" in inp.columns else None

        if "close" in inp.columns and len(tgt) > 0:
            p0 = inp["close"].iloc[-1]
            p1 = tgt["close"].iloc[-1]
            ret = (p1 / p0) - 1 if p0 != 0 else 0.0
        else:
            ret = 0.0
        category = categorize_return(ret)

        return inp_arr, tgt_arr, symbol, start_ts, end_ts, category


# ── DataLoader factory ────────────────────────────────────────────────────

def make_dataloaders(
    pipeline=None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    train_case_ids: list[str] | None = None,
    val_case_ids: list[str] | None = None,
    use_sector: bool = True,
    training_table: str | None = None,
    validation_table: str | None = None,
    input_window: int | None = None,
    prediction_window: int | None = None,
    stride: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``. When use_sector=True, batches are (X, y, sector_idx).

    Use training_table=TRAINING_TABLE_HOURLY, validation_table=VALIDATION_TABLE_HOURLY,
    input_window=INPUT_WINDOW_BARS_HOURLY, prediction_window=PREDICTION_WINDOW_BARS_HOURLY,
    stride=WINDOW_STRIDE_HOURLY for the hourly pipeline.
    """
    t_train = training_table or TRAINING_TABLE
    t_val   = validation_table or VALIDATION_TABLE
    train_ds = OwlDataset(
        t_train, train_case_ids, pipeline, use_sector=use_sector,
        input_window=input_window, prediction_window=prediction_window, stride=stride,
    )
    val_ds   = OwlDataset(
        t_val, val_case_ids, pipeline, use_sector=use_sector,
        input_window=input_window, prediction_window=prediction_window, stride=stride,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
