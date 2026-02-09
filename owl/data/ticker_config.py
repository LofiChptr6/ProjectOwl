"""
Ticker config — load sector/symbol CSV
=======================================
Provides ticker list and symbol→sector mapping for use by the query engine
and sector embedding in models.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from owl.config import TICKERS_CSV_PATH

logger = logging.getLogger(__name__)

_sector_list: list[str] | None = None
_symbol_to_sector: dict[str, str] | None = None
_symbol_to_idx: dict[str, int] | None = None


def load_tickers_csv(path: Path | None = None) -> pd.DataFrame:
    """Load the tickers_by_sector CSV. Returns DataFrame with columns sector, symbol."""
    p = path or TICKERS_CSV_PATH
    if not p.exists():
        logger.warning("Ticker CSV not found: %s", p)
        return pd.DataFrame()
    df = pd.read_csv(p)
    if "sector" not in df.columns or "symbol" not in df.columns:
        logger.warning("CSV must have columns [sector, symbol]")
        return pd.DataFrame()
    return df.dropna(subset=["sector", "symbol"])


def get_tickers(path: Path | None = None) -> list[str]:
    """Return list of symbols from the CSV."""
    df = load_tickers_csv(path)
    return df["symbol"].astype(str).str.strip().unique().tolist()


def get_sector_list(path: Path | None = None) -> list[str]:
    """Return ordered list of unique sectors (deterministic order for embedding indices)."""
    global _sector_list
    if _sector_list is not None:
        return _sector_list
    df = load_tickers_csv(path)
    if df.empty:
        _sector_list = []
        return []
    _sector_list = sorted(df["sector"].astype(str).str.strip().unique().tolist())
    return _sector_list


def get_symbol_to_sector(path: Path | None = None) -> dict[str, str]:
    """Return mapping symbol → sector. Last occurrence wins for duplicates."""
    global _symbol_to_sector
    if _symbol_to_sector is not None:
        return _symbol_to_sector
    df = load_tickers_csv(path)
    if df.empty:
        _symbol_to_sector = {}
        return {}
    _symbol_to_sector = df.set_index("symbol")["sector"].astype(str).str.strip().to_dict()
    return _symbol_to_sector


def get_symbol_to_sector_idx(path: Path | None = None) -> dict[str, int]:
    """Return mapping symbol → sector index (0-based). Unknown symbols → 0 (padding)."""
    global _symbol_to_idx
    if _symbol_to_idx is not None:
        return _symbol_to_idx
    sectors = get_sector_list(path)
    sym_to_sector = get_symbol_to_sector(path)
    sector_to_idx = {s: i + 1 for i, s in enumerate(sectors)}  # 0 = unknown/padding
    _symbol_to_idx = {sym: sector_to_idx.get(sec, 0) for sym, sec in sym_to_sector.items()}
    return _symbol_to_idx


def get_num_sectors(path: Path | None = None) -> int:
    """Number of sectors + 1 (for padding index 0)."""
    return len(get_sector_list(path)) + 1
