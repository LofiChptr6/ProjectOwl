"""
Normalization strategies for financial time-series
====================================================
Five approaches are implemented and selectable via ``config.NORMALIZATION_METHOD``:

1. **first_open** — Divide all prices by the first bar's open. All curves start ~1.
2. **revol** — Return-Volatility normalisation (inspired by ReVol, 2025).
   Removes per-window return and volatility characteristics so the model
   sees *shape* rather than *scale*.
3. **log_return** — Simple first-difference of log prices.
4. **zscore** — Per-window z-score of each column.
5. **minmax** — Per-window min–max scaling to [0, 1].

Volume is always handled separately via ``config.VOLUME_NORMALIZATION``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from owl.config import (
    MARKETCAP_PE_NORMALIZATION,
    NORMALIZATION_METHOD,
    NUM_TRANSACTIONS_NORMALIZATION,
    VOLUME_NORMALIZATION,
)

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
        "first_open": _first_open,
        "revol":      _revol,
        "log_return": _log_return,
        "zscore":     _zscore_prices,
        "minmax":     _minmax_prices,
    }
    fn = dispatch.get(method)
    if fn is None:
        raise ValueError(f"Unknown normalization method: {method}")
    return fn(df, price_cols)


# ── First-open ────────────────────────────────────────────────────────────

def _first_open(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Divide all price columns by the first bar's open. Curves start ~1."""
    df = df.copy()
    base = float(df["open"].iloc[0]) if "open" in df.columns else 1.0
    base = base if base != 0 else 1.0
    for c in cols:
        df[c] = df[c].values / base
    close = df["close"].values.astype(np.float64)
    log_ret = np.diff(np.log(np.maximum(close, _EPS)))
    df["log_return"] = np.insert(log_ret, 0, 0.0)
    return df, {"base_open": base}


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

    if method == "log_first60":
        log_vol = np.log1p(np.maximum(vol, 0))
        n = min(60, len(log_vol))
        avg_first = log_vol[:n].mean() + _EPS
        df["volume"] = log_vol / avg_first
        return df, {"vol_log_avg_first60": float(avg_first)}

    raise ValueError(f"Unknown volume normalization: {method}")


# ══════════════════════════════════════════════════════════════════════════
#  Num-transactions normalization
# ══════════════════════════════════════════════════════════════════════════

def normalize_num_transactions(df: pd.DataFrame, method: str | None = None) -> pd.DataFrame:
    """Normalize num_transactions (per-window). Method from config or override."""
    if "num_transactions" not in df.columns:
        return df
    method = method or NUM_TRANSACTIONS_NORMALIZATION
    df = df.copy()
    n = df["num_transactions"].values.astype(np.float64)

    if method == "log_zscore":
        log_n = np.log1p(np.maximum(n, 0))
        mu, std = log_n.mean(), log_n.std() + _EPS
        df["num_transactions"] = (log_n - mu) / std
    elif method == "zscore":
        mu, std = n.mean(), n.std() + _EPS
        df["num_transactions"] = (n - mu) / std
    elif method == "minmax":
        mn, mx = n.min(), n.max()
        rng = mx - mn + _EPS
        df["num_transactions"] = (n - mn) / rng
    elif method == "log_first60":
        log_n = np.log1p(np.maximum(n, 0))
        n_first = min(60, len(log_n))
        avg_first = log_n[:n_first].mean() + _EPS
        df["num_transactions"] = log_n / avg_first
    else:
        raise ValueError(f"Unknown num_transactions normalization: {method}")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Fundamental normalization (marketcap/pe: configurable; others: z-score)
# ══════════════════════════════════════════════════════════════════════════

_MARKETCAP_PE_COLS = {"marketcap", "pe"}


def _apply_marketcap_pe_norm(vals: np.ndarray, method: str) -> np.ndarray:
    """Apply configurable normalization for marketcap/pe (handles negatives via max(0) for log)."""
    vals = np.asarray(vals, dtype=np.float64)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

    if method == "log_first60":
        log_v = np.log1p(np.maximum(vals, 0))
        n = min(60, len(log_v))
        avg_first = log_v[:n].mean() + _EPS
        return log_v / avg_first
    if method == "log_zscore":
        log_v = np.log1p(np.maximum(vals, 0))
        mu, std = np.nanmean(log_v), np.nanstd(log_v) + _EPS
        return (log_v - mu) / std
    if method == "zscore":
        mu, std = np.nanmean(vals), np.nanstd(vals) + _EPS
        return (vals - mu) / std
    if method == "minmax":
        mn, mx = np.nanmin(vals), np.nanmax(vals)
        rng = mx - mn + _EPS
        return (vals - mn) / rng
    raise ValueError(f"Unknown marketcap/pe normalization: {method}")


def normalize_fundamentals(df: pd.DataFrame,
                           columns: list[str],
                           method: str | None = None) -> pd.DataFrame:
    """Normalize fundamentals. marketcap/pe use MARKETCAP_PE_NORMALIZATION; others z-score."""
    method = method or MARKETCAP_PE_NORMALIZATION
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            continue
        vals = df[c].values.astype(np.float64)
        if c in _MARKETCAP_PE_COLS:
            df[c] = _apply_marketcap_pe_norm(vals, method)
        else:
            mu = np.nanmean(vals)
            std = np.nanstd(vals) + _EPS
            df[c] = (vals - mu) / std
    return df
