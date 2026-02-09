"""
Preprocessing Pipeline
=======================
Chains normalisation → feature engineering → NaN handling → column selection
into a single callable that the data feeder applies to each case.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from owl.config import (
    EMA_WINDOWS,
    FUNDAMENTAL_COLUMNS,
    MA_WINDOWS,
    PRICE_COLUMNS,
    WMA_WINDOWS,
)
from owl.preprocessing.features import add_all_features
from owl.preprocessing.normalizer import (
    normalize_fundamentals,
    normalize_num_transactions,
    normalize_prices,
    normalize_volume,
)

logger = logging.getLogger(__name__)

# Columns that should *not* be fed to the model (metadata only)
_META_COLS = {"id", "case_id", "symbol", "timestamp", "frequency"}


class PreprocessingPipeline:
    """Stateless transformer: ``df_out = pipeline.transform(df_in)``

    After calling ``transform`` at least once, ``feature_columns`` contains
    the ordered list of numeric columns the model will receive.
    """

    def __init__(self):
        self._feature_columns: list[str] | None = None

    # ──────────────────────────────────────────────────────────────────────

    @property
    def feature_columns(self) -> list[str]:
        if self._feature_columns is None:
            raise RuntimeError("Call transform() first so columns are known.")
        return self._feature_columns

    # ──────────────────────────────────────────────────────────────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the full pipeline to a single-case DataFrame."""
        if df.empty:
            return df

        df = df.copy()

        # 1 ── normalise prices ────────────────────────────────────────────
        df, _price_meta = normalize_prices(df)

        # 2 ── normalise volume ────────────────────────────────────────────
        df, _vol_meta = normalize_volume(df)

        # 3 ── normalise num_transactions (log1p + z-score) ─────────────────
        df = normalize_num_transactions(df)

        # 4 ── normalise fundamentals (z-score per window) ─────────────────
        fund_cols = [c for c in FUNDAMENTAL_COLUMNS if c in df.columns]
        df = normalize_fundamentals(df, fund_cols)

        # 5 ── feature engineering (MAs, ARMA, GARCH, time, …) ─────────────
        df = add_all_features(df)

        # 6 ── fill remaining NaNs ─────────────────────────────────────────
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill().fillna(0)

        # 7 ── determine feature columns (once, so all cases use same columns)
        if self._feature_columns is None:
            self._feature_columns = [
                c for c in df.columns
                if c not in _META_COLS and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)
            ]
        # Ensure every case has the same feature columns (fill missing with 0)
        for c in self._feature_columns:
            if c not in df.columns:
                df[c] = 0.0

        return df

    # ──────────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable summary of what the pipeline does."""
        lines = [
            "PreprocessingPipeline",
            f"  price norm  : {__import__('owl.config', fromlist=['NORMALIZATION_METHOD']).NORMALIZATION_METHOD}",
            f"  volume norm : {__import__('owl.config', fromlist=['VOLUME_NORMALIZATION']).VOLUME_NORMALIZATION}",
            f"  SMA windows : {MA_WINDOWS}",
            f"  WMA windows : {WMA_WINDOWS}",
            f"  EMA windows : {EMA_WINDOWS}",
            f"  features    : {len(self._feature_columns or [])} columns",
        ]
        return "\n".join(lines)
