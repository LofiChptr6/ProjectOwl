"""
Feature engineering
====================
Computes technical indicators and temporal features that are appended as
new columns to the DataFrame **after** normalisation.

All window sizes are imported from ``owl.config`` so they are easy to change
in one place.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from owl.config import (
    ARMA_ORDER,
    BOLLINGER_STD,
    BOLLINGER_WINDOW,
    EMA_WINDOWS,
    GARCH_ORDER,
    MA_WINDOWS,
    RSI_WINDOW,
    WMA_WINDOWS,
)

# Suppress convergence warnings from statsmodels / arch during batch runs
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ══════════════════════════════════════════════════════════════════════════
#  Moving averages
# ══════════════════════════════════════════════════════════════════════════

def add_sma(df: pd.DataFrame, col: str = "close",
            windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or MA_WINDOWS
    for w in windows:
        df[f"sma_{w}"] = df[col].rolling(w, min_periods=1).mean()
    return df


def add_wma(df: pd.DataFrame, col: str = "close",
            windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or WMA_WINDOWS
    for w in windows:
        weights = np.arange(1, w + 1, dtype=float)
        df[f"wma_{w}"] = (
            df[col]
            .rolling(w, min_periods=1)
            .apply(lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(),
                   raw=True)
        )
    return df


def add_ema(df: pd.DataFrame, col: str = "close",
            windows: list[int] | None = None) -> pd.DataFrame:
    windows = windows or EMA_WINDOWS
    for w in windows:
        df[f"ema_{w}"] = df[col].ewm(span=w, min_periods=1, adjust=False).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Bollinger Bands
# ══════════════════════════════════════════════════════════════════════════

def add_bollinger(df: pd.DataFrame, col: str = "close",
                  window: int | None = None,
                  num_std: float | None = None) -> pd.DataFrame:
    window  = window  or BOLLINGER_WINDOW
    num_std = num_std or BOLLINGER_STD
    rolling = df[col].rolling(window, min_periods=1)
    mid   = rolling.mean()
    std   = rolling.std().fillna(0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    # normalised position within the band  (0 = lower, 1 = upper)
    df["boll_pos"] = ((df[col] - lower) / (upper - lower + 1e-8)).clip(-1, 2)
    df["boll_width"] = (upper - lower) / (mid + 1e-8)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  RSI
# ══════════════════════════════════════════════════════════════════════════

def add_rsi(df: pd.DataFrame, col: str = "close",
            window: int | None = None) -> pd.DataFrame:
    window = window or RSI_WINDOW
    delta  = df[col].diff()
    gain   = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    loss   = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs     = gain / (loss + 1e-8)
    df["rsi"] = 100 - 100 / (1 + rs)
    # normalise to [-1, 1]
    df["rsi"] = (df["rsi"] - 50) / 50
    return df


# ══════════════════════════════════════════════════════════════════════════
#  ARMA residuals
# ══════════════════════════════════════════════════════════════════════════

def add_arma_residuals(df: pd.DataFrame, col: str = "log_return",
                       order: tuple[int, int] | None = None) -> pd.DataFrame:
    """Fit ARMA on *col* and add residuals as ``arma_resid``."""
    order = order or ARMA_ORDER
    if col not in df.columns:
        df["arma_resid"] = 0.0
        return df
    try:
        from statsmodels.tsa.arima.model import ARIMA
        series = df[col].dropna().values
        if len(series) < max(order) + 10:
            raise ValueError("Too short")
        model = ARIMA(series, order=(order[0], 0, order[1]))
        result = model.fit(method_kwargs={"maxiter": 200})
        resid = np.full(len(df), 0.0)
        resid[df[col].notna().values] = result.resid
        df["arma_resid"] = resid
    except Exception:
        df["arma_resid"] = 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
#  GARCH conditional volatility
# ══════════════════════════════════════════════════════════════════════════

def add_garch_vol(df: pd.DataFrame, col: str = "log_return",
                  order: tuple[int, int] | None = None) -> pd.DataFrame:
    """Fit GARCH and add conditional volatility as ``garch_vol``."""
    order = order or GARCH_ORDER
    if col not in df.columns:
        df["garch_vol"] = 0.0
        return df
    try:
        from arch import arch_model
        series = df[col].dropna().values * 100        # arch expects % scale
        if len(series) < 20:
            raise ValueError("Too short")
        am = arch_model(series, vol="Garch", p=order[0], q=order[1],
                        mean="Zero", rescale=False)
        res = am.fit(disp="off", show_warning=False)
        vol = np.full(len(df), np.nanstd(series) / 100)
        vol[df[col].notna().values] = res.conditional_volatility / 100
        df["garch_vol"] = vol
    except Exception:
        df["garch_vol"] = df[col].rolling(20, min_periods=1).std().fillna(0).values \
            if col in df.columns else 0.0
    return df


# ══════════════════════════════════════════════════════════════════════════
#  Temporal features  (cyclical encoding)
# ══════════════════════════════════════════════════════════════════════════

def add_time_features(df: pd.DataFrame,
                      ts_col: str = "timestamp") -> pd.DataFrame:
    """Add cyclical (sin/cos) encodings of month, day-of-week, hour, minute."""
    if ts_col not in df.columns:
        return df
    ts = pd.to_datetime(df[ts_col])

    def _cyc(values, period):
        rad = 2 * np.pi * values / period
        return np.sin(rad), np.cos(rad)

    s, c = _cyc(ts.dt.month, 12)
    df["month_sin"], df["month_cos"] = s, c

    s, c = _cyc(ts.dt.dayofweek, 7)
    df["dow_sin"], df["dow_cos"] = s, c

    s, c = _cyc(ts.dt.hour, 24)
    df["hour_sin"], df["hour_cos"] = s, c

    s, c = _cyc(ts.dt.minute, 60)
    df["minute_sin"], df["minute_cos"] = s, c

    # binary: is the bar inside regular trading hours?
    hour_frac = ts.dt.hour + ts.dt.minute / 60.0
    df["is_market_open"] = ((hour_frac >= 9.5) & (hour_frac < 16.0)).astype(float)

    return df


# ══════════════════════════════════════════════════════════════════════════
#  Convenience: apply ALL features at once
# ══════════════════════════════════════════════════════════════════════════

def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run every feature-engineering step in order."""
    df = add_sma(df)
    df = add_wma(df)
    df = add_ema(df)
    df = add_bollinger(df)
    df = add_rsi(df)
    df = add_arma_residuals(df)
    df = add_garch_vol(df)
    df = add_time_features(df)
    return df
