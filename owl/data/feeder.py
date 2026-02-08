"""
Data Feeder — PyTorch Dataset + DataLoader
============================================
Reads pre-populated cases from PostgreSQL, applies the preprocessing
pipeline, produces rolling-window ``(input, target_category)`` pairs
ready for the CNN / Transformer models.

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
    CATEGORY_THRESHOLDS,
    INPUT_WINDOW_MINUTES,
    NUM_WORKERS,
    PREDICTION_WINDOW_MINUTES,
    TRAINING_TABLE,
    VALIDATION_TABLE,
    WINDOW_STRIDE,
)
from owl.data.db import get_case_ids, get_case_row_count, load_case_data

logger = logging.getLogger(__name__)

_TOTAL_WINDOW = INPUT_WINDOW_MINUTES + PREDICTION_WINDOW_MINUTES


# ── helpers ───────────────────────────────────────────────────────────────

def categorize_return(ret: float, thresholds: list[float] | None = None) -> int:
    """Map a scalar return to an integer category.

    Default thresholds (from config):
        strong_down | down | flat | up | strong_up
          <-2%       -0.5%  +0.5%  +2%
    """
    thresholds = thresholds or CATEGORY_THRESHOLDS
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
    """

    def __init__(self, table_name: str, case_ids: list[str] | None = None,
                 pipeline=None, cache_size: int = 50):
        self.table_name = table_name
        self.case_ids   = case_ids or get_case_ids(table_name)
        self.pipeline   = pipeline
        self._cache     = _CaseCache(maxsize=cache_size)

        # build a flat index  →  (case_id, window_start_row)
        self.windows: list[tuple[str, int]] = []
        for cid in self.case_ids:
            n = get_case_row_count(cid, table_name)
            for start in range(0, max(n - _TOTAL_WINDOW + 1, 0), WINDOW_STRIDE):
                self.windows.append((cid, start))

        logger.info("OwlDataset: %d windows from %d cases (%s)",
                     len(self.windows), len(self.case_ids), table_name)

    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        case_id, start = self.windows[idx]
        df = self._cache.get(case_id, self.table_name, self.pipeline)

        end = start + _TOTAL_WINDOW
        if end > len(df):
            end = len(df)
            start = max(end - _TOTAL_WINDOW, 0)

        window = df.iloc[start:end]

        # split into input / prediction
        inp = window.iloc[:INPUT_WINDOW_MINUTES]
        tgt = window.iloc[INPUT_WINDOW_MINUTES:]

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
        # replace remaining NaN with 0
        np.nan_to_num(inp_arr, copy=False)

        return torch.from_numpy(inp_arr), category

    # ──────────────────────────────────────────────────────────────────────

    def get_feature_dim(self) -> int:
        """Number of features per time-step."""
        sample, _ = self[0]
        return sample.shape[-1]


# ── DataLoader factory ────────────────────────────────────────────────────

def make_dataloaders(
    pipeline=None,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    train_case_ids: list[str] | None = None,
    val_case_ids: list[str] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader)``."""
    train_ds = OwlDataset(TRAINING_TABLE, train_case_ids, pipeline)
    val_ds   = OwlDataset(VALIDATION_TABLE, val_case_ids, pipeline)

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
