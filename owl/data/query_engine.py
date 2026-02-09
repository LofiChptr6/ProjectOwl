"""
Data Query Engine
==================
Generates random (symbol, date-window) cases, fetches price + fundamental
data from both APIs, merges / aligns frequencies, and stores the result
in the PostgreSQL training or validation table.

Key operations
--------------
1. ``generate_random_cases()`` — pick random date snapshots *per symbol*
2. ``fetch_and_merge()``       — pull Massive bars + SHARADAR daily/SF1,
                                  lag fundamentals by 1 day, forward-fill
3. ``populate_database()``     — orchestrate (1) + (2) with a thread-pool
"""

import hashlib
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from threading import Lock

import numpy as np
import pandas as pd
from tqdm import tqdm

from owl.config import (
    CASE_WINDOW_CALENDAR_DAYS,
    DATA_END_DATE,
    HISTORIC_LOOKBACK_YEARS,
    FUNDAMENTAL_COLUMNS,
    POPULATE_MAX_WORKERS,
    PRICE_FREQUENCY,
    PRICE_FREQUENCY_HOURLY,
    PRICE_MULTIPLIER,
    PRICE_MULTIPLIER_HOURLY,
    RANDOM_CASES_PER_SYMBOL,
    RANDOM_CASES_TRAINING,
    RANDOM_CASES_VALIDATION,
    SHARADAR_DAILY_COLUMNS,
    SHARADAR_SF1_COLUMNS,
    TOP_STOCKS_COUNT,
    TRAIN_VAL_RATIO,
    TRAINING_TABLE,
    TRAINING_TABLE_HOURLY,
    VALIDATION_TABLE,
    VALIDATION_TABLE_HOURLY,
)
from owl.data.db import insert_case_data, insert_metadata
from owl.data.massive_client import MassiveClient
from owl.data.sharadar_client import (
    get_daily_metrics,
    get_fundamentals,
    get_ticker_universe,
)
from owl.data.ticker_config import get_tickers as get_tickers_from_csv

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


def _overlapping_start(sd1: date, overlap_days: int, start_d: date, end_d: date) -> date | None:
    """Return a random weekday sd2 such that [sd1, sd1+W] and [sd2, sd2+W] overlap >= overlap_days."""
    W = CASE_WINDOW_CALENDAR_DAYS
    lo = max(start_d, sd1 - timedelta(days=overlap_days))
    hi = min(end_d - timedelta(days=W + 5), sd1 + timedelta(days=overlap_days))
    if lo > hi:
        return None
    for _ in range(30):
        d = lo + timedelta(days=random.randint(0, max(0, (hi - lo).days)))
        if d.weekday() < 5:
            return d
    # fallback: first weekday in [lo, hi]
    d = lo
    while d <= hi:
        if d.weekday() < 5:
            return d
        d += timedelta(days=1)
    return None


def generate_random_cases(
    n_train: int | None = None,
    n_val: int | None = None,
    tickers: list[str] | None = None,
    cases_per_symbol: int | None = None,
    overlap_pairs: bool = False,
    overlap_fraction: float = 0.5,
) -> tuple[list[dict], list[dict]]:
    """Return ``(train_cases, val_cases)``.

    Strategy:
    - Generate *cases_per_symbol* random date snapshots for **every** ticker.
    - If *overlap_pairs* is True: cases are generated in pairs; each pair overlaps
      by at least *overlap_fraction* of the case window (e.g. 0.5 = 50% overlap).
    - Split 80/20 into train / val (per symbol).
    - If *n_train* / *n_val* are explicitly provided, fall back to global-count.
    """
    tickers = tickers or get_tickers_from_csv()
    if not tickers:
        tickers = get_ticker_universe(top_n=TOP_STOCKS_COUNT)
    if not tickers:
        raise RuntimeError("Ticker universe is empty — check API keys.")

    today = date.today()
    end_d = min(datetime.strptime(DATA_END_DATE, "%Y-%m-%d").date(), today)
    start_d = today - timedelta(days=HISTORIC_LOOKBACK_YEARS * 365)
    start_str = start_d.strftime("%Y-%m-%d")
    end_str = end_d.strftime("%Y-%m-%d")

    seen: set[str] = set()
    all_cases: list[dict] = []

    # ── per-symbol strategy ─────────────────────────────────────────────
    if n_train is None and n_val is None:
        cps = cases_per_symbol or RANDOM_CASES_PER_SYMBOL
        overlap_days = int(CASE_WINDOW_CALENDAR_DAYS * overlap_fraction)

        for sym in tickers:
            sym_cases: list[dict] = []

            if overlap_pairs:
                # Generate pairs: each pair overlaps by >= overlap_fraction
                n_pairs = max(1, cps // 2)
                for _ in range(n_pairs):
                    sd1 = _random_trading_date(start_str, end_str)
                    ed1 = sd1 + timedelta(days=CASE_WINDOW_CALENDAR_DAYS)
                    cid1 = _case_id(sym, sd1)
                    if cid1 not in seen:
                        seen.add(cid1)
                        sym_cases.append({"symbol": sym, "start_date": sd1, "end_date": ed1, "case_id": cid1})
                    sd2 = _overlapping_start(sd1, overlap_days, start_d, end_d)
                    if sd2 is not None:
                        ed2 = sd2 + timedelta(days=CASE_WINDOW_CALENDAR_DAYS)
                        cid2 = _case_id(sym, sd2)
                        if cid2 not in seen:
                            seen.add(cid2)
                            sym_cases.append({"symbol": sym, "start_date": sd2, "end_date": ed2, "case_id": cid2})
            else:
                attempts = 0
                while len(sym_cases) < cps and attempts < cps * 5:
                    attempts += 1
                    sd = _random_trading_date(start_str, end_str)
                    ed = sd + timedelta(days=CASE_WINDOW_CALENDAR_DAYS)
                    cid = _case_id(sym, sd)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    sym_cases.append({"symbol": sym, "start_date": sd, "end_date": ed, "case_id": cid})

            # split per symbol: first N train, rest val
            random.shuffle(sym_cases)
            n_train_sym = max(1, int(len(sym_cases) * TRAIN_VAL_RATIO))
            for i, c in enumerate(sym_cases):
                c["split"] = "train" if i < n_train_sym else "validation"
            all_cases.extend(sym_cases)

    # ── legacy global-count strategy ──────────────────────────────────
    else:
        _n_train = n_train or (RANDOM_CASES_TRAINING or 500)
        _n_val   = n_val   or (RANDOM_CASES_VALIDATION or 100)
        for split, n in [("train", _n_train), ("validation", _n_val)]:
            for _ in range(n):
                sym = random.choice(tickers)
                sd  = _random_trading_date(start_str, end_str)
                ed  = sd + timedelta(days=CASE_WINDOW_CALENDAR_DAYS)
                cid = _case_id(sym, sd)
                if cid in seen:
                    continue
                seen.add(cid)
                all_cases.append({
                    "symbol":     sym,
                    "start_date": sd,
                    "end_date":   ed,
                    "case_id":    cid,
                    "split":      split,
                })

    train = [c for c in all_cases if c["split"] == "train"]
    val   = [c for c in all_cases if c["split"] == "validation"]
    logger.info("Generated %d training + %d validation cases from %d tickers",
                len(train), len(val), len(tickers))
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
    frequency: str = "minute",
) -> pd.DataFrame | None:
    """Query Massive + SHARADAR, align frequencies, return merged DataFrame.

    Returns ``None`` when either source yields no data.
    Use frequency='hour' for 1-hour bars (separate tables).
    """
    mc = massive_client or MassiveClient()
    if frequency == "hour":
        mult, span = PRICE_MULTIPLIER_HOURLY, PRICE_FREQUENCY_HOURLY
    else:
        mult, span = PRICE_MULTIPLIER, PRICE_FREQUENCY

    # 1 ── price bars: query 7 days before start for smooth merge with daily ─
    PRICE_PADDING_DAYS = 7
    price_from = start_date - timedelta(days=PRICE_PADDING_DAYS)
    price_df = mc.get_aggs(
        ticker=symbol,
        multiplier=mult,
        timespan=span,
        from_date=str(price_from),
        to_date=str(end_date),
    )
    if price_df.empty:
        logger.warning("No price data for %s %s→%s", symbol, price_from, end_date)
        return None

    # keep only regular-hours bars (configurable via config)
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)

    # 2 ── SHARADAR daily metrics ──────────────────────────────────────────
    # pull enough older data so leading bars (after 1-day lag) get filled
    sharadar_start = price_from - timedelta(days=30)
    daily_df = get_daily_metrics(symbol, sharadar_start, end_date)

    # 3 ── SHARADAR SF1 fundamentals ───────────────────────────────────────
    sf1_df = get_fundamentals(symbol, start_date=str(sharadar_start - timedelta(days=365)),
                              end_date=str(end_date))

    # 4 ── align & merge ──────────────────────────────────────────────────
    merged = _merge_sources(price_df, daily_df, sf1_df)
    if merged is None or merged.empty:
        return None

    # 5 ── drop padding: keep only rows from start_date onwards ─────────────
    merged["timestamp"] = pd.to_datetime(merged["timestamp"])
    merged = merged[merged["timestamp"].dt.date >= start_date].copy()
    merged.reset_index(drop=True, inplace=True)

    merged.insert(0, "case_id", case_id)
    merged.insert(1, "symbol", symbol)
    merged.insert(3, "frequency", span)
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
        raise ValueError("No daily metrics found")

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

    # ── forward-fill: use last available non-NaN for any gap ────────────────
    price_df.sort_values("timestamp", inplace=True)
    for col in FUNDAMENTAL_COLUMNS:
        if col in price_df.columns:
            price_df[col] = price_df[col].ffill()

    price_df.drop(columns=["_date"], inplace=True, errors="ignore")
    price_df.reset_index(drop=True, inplace=True)
    return price_df


# ══════════════════════════════════════════════════════════════════════════
#  Single-case worker (called in thread pool)
# ══════════════════════════════════════════════════════════════════════════

_db_lock = Lock()          # serialise DB writes (avoid concurrent insert clashes)


def _process_one_case(
    case: dict,
    massive_client: MassiveClient,
    frequency: str = "minute",
    training_table: str | None = None,
    validation_table: str | None = None,
    meta_table: str = "case_metadata",
) -> dict:
    """Fetch, merge, and insert a single case.  Returns a status dict."""
    t_train = training_table or TRAINING_TABLE
    t_val = validation_table or VALIDATION_TABLE
    table = t_train if case["split"] == "train" else t_val
    try:
        df = fetch_and_merge(
            symbol=case["symbol"],
            start_date=case["start_date"],
            end_date=case["end_date"],
            case_id=case["case_id"],
            massive_client=massive_client,
            frequency=frequency,
        )
        if df is not None and not df.empty:
            with _db_lock:
                insert_case_data(df, table)
                insert_metadata(
                    case_id=case["case_id"],
                    symbol=case["symbol"],
                    start_date=case["start_date"],
                    end_date=case["end_date"],
                    split=case["split"],
                    num_rows=len(df),
                    meta_table=meta_table,
                )
            return {"case_id": case["case_id"], "status": "ok", "rows": len(df)}
        return {"case_id": case["case_id"], "status": "empty"}
    except Exception as exc:
        logger.exception("Failed case %s (%s)", case["case_id"], case["symbol"])
        return {"case_id": case["case_id"], "status": "error", "error": str(exc)}


# ══════════════════════════════════════════════════════════════════════════
#  Populate database  (multi-worker)
# ══════════════════════════════════════════════════════════════════════════

def populate_database(
    n_train: int | None = None,
    n_val: int | None = None,
    tickers: list[str] | None = None,
    cases_per_symbol: int | None = None,
    max_workers: int | None = None,
    frequency: str = "minute",
    overlap_pairs: bool = False,
    overlap_fraction: float = 0.5,
) -> None:
    """End-to-end: generate cases → fetch → merge → store in PostgreSQL.

    Uses a thread-pool of *max_workers* threads so multiple API calls and
    DB inserts run concurrently.

    Use frequency='hour' for by-hour data → writes to training_cases_hourly,
    validation_cases_hourly, case_metadata_hourly.

    Use overlap_pairs=True so that for each symbol, cases are generated in
    pairs that overlap by at least overlap_fraction (e.g. 0.5 = 50%) of the window.
    """
    train_cases, val_cases = generate_random_cases(
        n_train, n_val, tickers, cases_per_symbol,
        overlap_pairs=overlap_pairs,
        overlap_fraction=overlap_fraction,
    )
    all_cases = train_cases + val_cases
    workers = max_workers or POPULATE_MAX_WORKERS

    if frequency == "hour":
        t_train, t_val = TRAINING_TABLE_HOURLY, VALIDATION_TABLE_HOURLY
        meta_table = "case_metadata_hourly"
    else:
        t_train, t_val = TRAINING_TABLE, VALIDATION_TABLE
        meta_table = "case_metadata"

    # one MassiveClient per worker (re-uses its requests.Session)
    clients = [MassiveClient() for _ in range(workers)]

    ok = 0
    empty = 0
    errors = 0

    logger.info("Populating DB (%s) with %d cases using %d workers …",
                frequency, len(all_cases), workers)

    def _worker(case, mc):
        return _process_one_case(
            case, mc,
            frequency=frequency,
            training_table=t_train,
            validation_table=t_val,
            meta_table=meta_table,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_worker, case, clients[i % workers]): case
            for i, case in enumerate(all_cases)
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Populating DB"):
            result = future.result()
            if result["status"] == "ok":
                ok += 1
            elif result["status"] == "empty":
                empty += 1
            else:
                errors += 1

    logger.info("Population complete: %d ok, %d empty, %d errors",
                ok, empty, errors)
