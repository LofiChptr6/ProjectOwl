"""
Normalization strategies for financial time-series
====================================================
Four approaches are implemented and selectable via ``config.NORMALIZATION_METHOD``:

1. **revol** — Return-Volatility normalisation (inspired by ReVol, 2025).
   Removes per-window return and volatility characteristics so the model
   sees *shape* rather than *scale*.
2. **log_return** — Simple first-difference of log prices.
3. **zscore** — Per-window z-score of each column.
4. **minmax** — Per-window min–max scaling to [0, 1].

Volume is always handled separately via ``config.VOLUME_NORMALIZATION``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from owl.config import NORMALIZATION_METHOD, VOLUME_NORMALIZATION

_EPS = 1e-8


# ══════════════════════════════════════════════════════════════════════════
#  Price normalization
# ══════════════════════════════════════════════════════════════════════════

def normalize_prices(df: pd.DataFrame, method: str | None = None,
                     ) -> tuple[pd.DataFrame, dict]:
    """Normalize OHLC columns **in-place-ish** and return ``(df, meta)``."""
    method = method or NORMALIZATION_METHOD
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if not price_cols:
        return df, {}

    dispatch = {
        "revol":      _revol,
        "log_return": _log_return,
        "zscore":     _zscore_prices,
        "minmax":     _minmax_prices,
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown normalization method: {method}")
    return fn(df, price_cols)


# ── ReVol ─────────────────────────────────────────────────────────────────

def _revol(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Return-volatility normalisation.

    1. Compute log-returns on ``close``.
    2. Measure window return & volatility.
    3. Normalise all price cols: ``(price / base_price - 1) / volatility``.
    """
    df = df.copy()
    close = df["close"].values.astype(np.float64)

    log_ret = np.diff(np.log(np.maximum(close, _EPS)))
    log_ret = np.insert(log_ret, 0, 0.0)

    base       = close[0] if close[0] != 0 else 1.0
    win_return = log_ret.sum()
    win_vol    = log_ret.std() + _EPS

    for c in cols:
        df[c] = (df[c].values / base - 1.0) / win_vol

    meta = {
        "base_price":       base,
        "window_return":    win_return,
        "window_volatility": win_vol,
    }
    # add log-return as an extra column
    df["log_return"] = log_ret / win_vol
    return df, meta


# ── Log-return ────────────────────────────────────────────────────────────

def _log_return(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    for c in cols:
        arr = df[c].values.astype(np.float64)
        lr  = np.diff(np.log(np.maximum(arr, _EPS)))
        df[c] = np.insert(lr, 0, 0.0)
    df["log_return"] = df["close"]          # alias
    return df, {}


# ── Z-score ───────────────────────────────────────────────────────────────

def _zscore_prices(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    meta: dict = {}
    for c in cols:
        mu  = df[c].mean()
        std = df[c].std() + _EPS
        df[c] = (df[c] - mu) / std
        meta[f"{c}_mean"] = mu
        meta[f"{c}_std"]  = std
    df["log_return"] = np.insert(
        np.diff(np.log(np.maximum(df["close"].values + _EPS, _EPS))), 0, 0.0)
    return df, meta


# ── Min-max ───────────────────────────────────────────────────────────────

def _minmax_prices(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    meta: dict = {}
    for c in cols:
        mn = df[c].min()
        mx = df[c].max()
        rng = mx - mn + _EPS
        df[c] = (df[c] - mn) / rng
        meta[f"{c}_min"] = mn
        meta[f"{c}_max"] = mx
    df["log_return"] = np.insert(
        np.diff(np.log(np.maximum(df["close"].values + _EPS, _EPS))), 0, 0.0)
    return df, meta


# ══════════════════════════════════════════════════════════════════════════
#  Volume normalization
# ══════════════════════════════════════════════════════════════════════════

def normalize_volume(df: pd.DataFrame, method: str | None = None,
                     ) -> tuple[pd.DataFrame, dict]:
    method = method or VOLUME_NORMALIZATION
    if "volume" not in df.columns:
        return df, {}

    df = df.copy()
    vol = df["volume"].values.astype(np.float64)

    if method == "log_zscore":
        log_vol = np.log1p(np.maximum(vol, 0))
        mu, std = log_vol.mean(), log_vol.std() + _EPS
        df["volume"] = (log_vol - mu) / std
        return df, {"vol_log_mean": mu, "vol_log_std": std}

    if method == "zscore":
        mu, std = vol.mean(), vol.std() + _EPS
        df["volume"] = (vol - mu) / std
        return df, {"vol_mean": mu, "vol_std": std}

    if method == "minmax":
        mn, mx = vol.min(), vol.max()
        rng = mx - mn + _EPS
        df["volume"] = (vol - mn) / rng
        return df, {"vol_min": mn, "vol_max": mx}

    raise ValueError(f"Unknown volume normalization: {method}")


# ══════════════════════════════════════════════════════════════════════════
#  Fundamental normalization (z-score per column within the window)
# ══════════════════════════════════════════════════════════════════════════

def normalize_fundamentals(df: pd.DataFrame,
                           columns: list[str]) -> pd.DataFrame:
    """Z-score each fundamental column independently."""
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        vals = df[c].values.astype(np.float64)
        mu  = np.nanmean(vals)
        std = np.nanstd(vals) + _EPS
        df[c] = (vals - mu) / std
    return df
