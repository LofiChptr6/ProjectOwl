"""
Data Query Engine
==================
Generates random (symbol, date-window) cases, fetches price + fundamental
data from both APIs, merges / aligns frequencies, and stores the result
in the PostgreSQL training or validation table.

Key operations
--------------
1. ``generate_random_cases()`` — pick random symbols & date ranges
2. ``fetch_and_merge()``       — pull Massive bars + SHARADAR daily/SF1,
                                  lag fundamentals by 1 day, forward-fill
3. ``populate_database()``     — orchestrate (1) + (2), write to DB
"""

import hashlib
import logging
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm

from owl.config import (
    CASE_WINDOW_CALENDAR_DAYS,
    CASE_WINDOW_TRADING_DAYS,
    DATA_END_DATE,
    DATA_START_DATE,
    FUNDAMENTAL_COLUMNS,
    PRICE_FREQUENCY,
    PRICE_MULTIPLIER,
    RANDOM_CASES_TRAINING,
    RANDOM_CASES_VALIDATION,
    SHARADAR_DAILY_COLUMNS,
    SHARADAR_SF1_COLUMNS,
    TOP_STOCKS_COUNT,
    TRAINING_TABLE,
    VALIDATION_TABLE,
)
from owl.data.db import insert_case_data, insert_metadata
from owl.data.massive_client import MassiveClient
from owl.data.sharadar_client import (
    get_daily_metrics,
    get_fundamentals,
    get_ticker_universe,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
#  Random case generation
# ══════════════════════════════════════════════════════════════════════════

def _random_trading_date(start: str, end: str) -> date:
    """Return a random *weekday* between *start* and *end* (YYYY-MM-DD)."""
    s = datetime.strptime(start, "%Y-%m-%d").date()
    e = datetime.strptime(end, "%Y-%m-%d").date()
    # leave room for the case window
    e = e - timedelta(days=CASE_WINDOW_CALENDAR_DAYS + 5)
    if e <= s:
        e = s + timedelta(days=1)
    while True:
        d = s + timedelta(days=random.randint(0, (e - s).days))
        if d.weekday() < 5:          # Mon–Fri
            return d


def _case_id(symbol: str, start: date) -> str:
    raw = f"{symbol}_{start.isoformat()}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def generate_random_cases(
    n_train: int | None = None,
    n_val: int | None = None,
    tickers: list[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train_cases, val_cases)`` — each a list of dicts with keys
    ``symbol``, ``start_date``, ``end_date``, ``case_id``, ``split``.
    """
    n_train = n_train or RANDOM_CASES_TRAINING
    n_val   = n_val   or RANDOM_CASES_VALIDATION
    tickers = tickers or get_ticker_universe(top_n=TOP_STOCKS_COUNT)

    if not tickers:
        raise RuntimeError("Ticker universe is empty — check API keys.")

    seen: set[str] = set()
    cases: list[dict] = []

    for split, n in [("train", n_train), ("validation", n_val)]:
        for _ in range(n):
            sym = random.choice(tickers)
            sd  = _random_trading_date(DATA_START_DATE, DATA_END_DATE)
            ed  = sd + timedelta(days=CASE_WINDOW_CALENDAR_DAYS)
            cid = _case_id(sym, sd)
            if cid in seen:
                continue
            seen.add(cid)
            cases.append({
                "symbol":     sym,
                "start_date": sd,
                "end_date":   ed,
                "case_id":    cid,
                "split":      split,
            })

    train = [c for c in cases if c["split"] == "train"]
    val   = [c for c in cases if c["split"] == "validation"]
    logger.info("Generated %d training + %d validation cases", len(train), len(val))
    return train, val


# ══════════════════════════════════════════════════════════════════════════
#  Fetch & merge
# ══════════════════════════════════════════════════════════════════════════

def fetch_and_merge(
    symbol: str,
    start_date: date,
    end_date: date,
    case_id: str,
    massive_client: MassiveClient | None = None,
) -> pd.DataFrame | None:
    """Query Massive + SHARADAR, align frequencies, return merged DataFrame.

    Returns ``None`` when either source yields no data.
    """
    mc = massive_client or MassiveClient()

    # 1 ── minute-level price bars ─────────────────────────────────────────
    price_df = mc.get_aggs(
        ticker=symbol,
        multiplier=PRICE_MULTIPLIER,
        timespan=PRICE_FREQUENCY,
        from_date=str(start_date),
        to_date=str(end_date),
    )
    if price_df.empty:
        logger.warning("No price data for %s %s→%s", symbol, start_date, end_date)
        return None

    # keep only regular-hours bars (configurable via config)
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)

    # 2 ── SHARADAR daily metrics ──────────────────────────────────────────
    # pull a few extra days back so we can lag
    sharadar_start = start_date - timedelta(days=7)
    daily_df = get_daily_metrics(symbol, sharadar_start, end_date)

    # 3 ── SHARADAR SF1 fundamentals ───────────────────────────────────────
    sf1_df = get_fundamentals(symbol, start_date=str(sharadar_start - timedelta(days=365)),
                              end_date=str(end_date))

    # 4 ── align & merge ──────────────────────────────────────────────────
    merged = _merge_sources(price_df, daily_df, sf1_df)
    if merged is None or merged.empty:
        return None

    merged.insert(0, "case_id", case_id)
    merged.insert(1, "symbol", symbol)
    merged.insert(3, "frequency", PRICE_FREQUENCY)
    return merged


def _merge_sources(
    price_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    sf1_df: pd.DataFrame,
) -> pd.DataFrame | None:
    """Merge daily & quarterly data into minute-level price bars.

    • SHARADAR daily → lagged 1 day, then joined by date & forward-filled.
    • SHARADAR SF1  → lagged 1 day from ``datekey``, forward-filled.
    """
    if price_df.empty:
        return None

    # Ensure timestamp is datetime
    price_df = price_df.copy()
    price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
    price_df["_date"] = price_df["timestamp"].dt.normalize()

    # ── merge daily metrics (lagged 1 day) ────────────────────────────
    if not daily_df.empty:
        daily_df = daily_df.copy()
        daily_df["date"] = pd.to_datetime(daily_df["date"])
        # lag: yesterday's report is available today
        daily_df["_date"] = daily_df["date"] + pd.Timedelta(days=1)
        daily_df.drop(columns=["date"], inplace=True)
        # keep only the columns we want
        daily_cols = [c for c in SHARADAR_DAILY_COLUMNS if c in daily_df.columns]
        daily_df = daily_df[["_date"] + daily_cols].drop_duplicates(subset=["_date"])

        price_df = price_df.merge(daily_df, on="_date", how="left")
    else:
        for c in SHARADAR_DAILY_COLUMNS:
            price_df[c] = np.nan

    # ── merge SF1 fundamentals (lagged 1 day from datekey) ────────────
    if not sf1_df.empty:
        sf1_df = sf1_df.copy()
        sf1_df["datekey"] = pd.to_datetime(sf1_df["datekey"])
        sf1_df["_date"] = sf1_df["datekey"] + pd.Timedelta(days=1)
        sf1_df.drop(columns=["datekey"], inplace=True)
        sf1_cols = [c for c in SHARADAR_SF1_COLUMNS if c in sf1_df.columns]
        sf1_df = sf1_df[["_date"] + sf1_cols].drop_duplicates(subset=["_date"])

        price_df = price_df.merge(sf1_df, on="_date", how="left")
    else:
        for c in SHARADAR_SF1_COLUMNS:
            if c not in price_df.columns:
                price_df[c] = np.nan

    # ── forward-fill fundamental columns across minute bars ───────────
    price_df.sort_values("timestamp", inplace=True)
    for col in FUNDAMENTAL_COLUMNS:
        if col in price_df.columns:
            price_df[col] = price_df[col].ffill()

    price_df.drop(columns=["_date"], inplace=True, errors="ignore")
    price_df.reset_index(drop=True, inplace=True)
    return price_df


# ══════════════════════════════════════════════════════════════════════════
#  Populate database
# ══════════════════════════════════════════════════════════════════════════

def populate_database(
    n_train: int | None = None,
    n_val: int | None = None,
    tickers: list[str] | None = None,
) -> None:
    """End-to-end: generate cases → fetch → merge → store in PostgreSQL."""
    train_cases, val_cases = generate_random_cases(n_train, n_val, tickers)
    mc = MassiveClient()

    for case in tqdm(train_cases + val_cases, desc="Populating DB"):
        table = TRAINING_TABLE if case["split"] == "train" else VALIDATION_TABLE
        try:
            df = fetch_and_merge(
                symbol=case["symbol"],
                start_date=case["start_date"],
                end_date=case["end_date"],
                case_id=case["case_id"],
                massive_client=mc,
            )
            if df is not None and not df.empty:
                insert_case_data(df, table)
                insert_metadata(
                    case_id=case["case_id"],
                    symbol=case["symbol"],
                    start_date=case["start_date"],
                    end_date=case["end_date"],
                    split=case["split"],
                    num_rows=len(df),
                )
        except Exception:
            logger.exception("Failed case %s (%s)", case["case_id"], case["symbol"])
