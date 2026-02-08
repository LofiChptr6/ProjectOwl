"""
SHARADAR / Nasdaq-Data-Link client
====================================
Wraps the ``nasdaqdatalink`` package to pull:

  • **SHARADAR/DAILY** — daily valuation ratios (PE, PB, PS, market-cap)
  • **SHARADAR/SF1**   — quarterly fundamentals (revenue, EBITDA, margins …)
  • **SHARADAR/TICKERS** — ticker universe for market-cap filtering
"""

import logging
from datetime import date

import pandas as pd

from owl.config import (
    NASDAQ_DATA_LINK_API_KEY,
    SHARADAR_DAILY_COLUMNS,
    SHARADAR_SF1_COLUMNS,
)

logger = logging.getLogger(__name__)

# Lazy-import so the rest of the app can load even when the package is missing
_ndl = None


def _get_ndl():
    global _ndl
    if _ndl is None:
        import nasdaqdatalink
        nasdaqdatalink.ApiConfig.api_key = NASDAQ_DATA_LINK_API_KEY
        _ndl = nasdaqdatalink
    return _ndl


# ── public helpers ────────────────────────────────────────────────────────

def get_daily_metrics(
    ticker: str,
    start_date: str | date,
    end_date: str | date,
) -> pd.DataFrame:
    """Return daily valuation metrics for *ticker* over the given range.

    Columns returned: ``date`` + everything listed in
    ``config.SHARADAR_DAILY_COLUMNS``.
    """
    ndl = _get_ndl()
    try:
        df = ndl.get_table(
            "SHARADAR/DAILY",
            ticker=ticker,
            date={"gte": str(start_date), "lte": str(end_date)},
            paginate=True,
        )
    except Exception as exc:
        logger.error("SHARADAR/DAILY query failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    keep = ["date"] + [c for c in SHARADAR_DAILY_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_fundamentals(
    ticker: str,
    start_date: str | date | None = None,
    end_date: str | date | None = None,
    dimension: str = "MRQ",
) -> pd.DataFrame:
    """Return quarterly fundamentals from SF1.

    The ``datekey`` column indicates the filing date (when the data became
    publicly available).  We use it to implement the 1-day lag later.
    """
    ndl = _get_ndl()
    filters: dict = {"ticker": ticker, "dimension": dimension}
    if start_date:
        filters["datekey"] = {"gte": str(start_date)}
    if end_date:
        filters.setdefault("datekey", {})
        if isinstance(filters["datekey"], dict):
            filters["datekey"]["lte"] = str(end_date)

    try:
        df = ndl.get_table("SHARADAR/SF1", paginate=True, **filters)
    except Exception as exc:
        logger.error("SHARADAR/SF1 query failed for %s: %s", ticker, exc)
        return pd.DataFrame()

    if df.empty:
        return df

    keep = ["datekey"] + [c for c in SHARADAR_SF1_COLUMNS if c in df.columns]
    df = df[keep].copy()
    df["datekey"] = pd.to_datetime(df["datekey"])
    df.sort_values("datekey", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def get_ticker_universe(
    min_market_cap: float = 1e9,
    top_n: int = 1000,
) -> list[str]:
    """Return top *top_n* tickers by market-cap from SHARADAR/TICKERS.

    Falls back to a hard-coded default list when the API call fails so that
    development can proceed offline.
    """
    ndl = _get_ndl()
    try:
        df = ndl.get_table(
            "SHARADAR/TICKERS",
            table="SF1",
            paginate=True,
        )
    except Exception as exc:
        logger.warning("SHARADAR/TICKERS query failed: %s — using fallback", exc)
        return _FALLBACK_TICKERS[:top_n]

    if df.empty:
        return _FALLBACK_TICKERS[:top_n]

    # Filter active US equities
    if "isdelisted" in df.columns:
        df = df[df["isdelisted"] == "N"]
    if "category" in df.columns:
        df = df[df["category"] == "Domestic Common Stock"]
    if "lastpricedate" in df.columns:
        df = df.dropna(subset=["lastpricedate"])

    if "scalemarketcap" in df.columns:
        # scalemarketcap encodes: 1=Nano … 6=Mega
        df = df.sort_values("scalemarketcap", ascending=False)
    elif "famaindustry" in df.columns:
        pass  # can't sort by cap; return everything

    tickers = df["ticker"].unique().tolist()[:top_n]
    logger.info("Loaded %d tickers from SHARADAR/TICKERS", len(tickers))
    return tickers


# ── hard-coded fallback (top US large-caps, 2024 vintage) ────────────────
_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "AVGO", "HD", "CVX",
    "MRK", "ABBV", "LLY", "PEP", "KO", "COST", "ADBE", "WMT", "MCD",
    "CSCO", "CRM", "BAC", "PFE", "TMO", "ACN", "NFLX", "AMD", "LIN",
    "ABT", "ORCL", "CMCSA", "DHR", "NKE", "TXN", "WFC", "UPS", "PM",
    "INTC", "RTX", "QCOM", "NEE", "LOW",
]
