"""
Massive / Polygon REST client
===============================
Thin wrapper around the Polygon-compatible REST API for:
  • Aggregate bars  (OHLCV at any frequency)
  • Ticker listing  (filtered by market-cap)

All HTTP details are isolated here so the rest of the code-base is decoupled
from the upstream API shape.
"""

import logging
import time
from datetime import date, datetime

import pandas as pd
import requests

from owl.config import MASSIVE_API_KEY, MASSIVE_BASE_URL

logger = logging.getLogger(__name__)

_RATE_LIMIT_SLEEP = 0.15          # seconds between paginated requests
_MAX_RETRIES      = 3
_RETRY_BACKOFF    = 2.0           # exponential back-off multiplier


class MassiveClient:
    """Stateless REST client — instantiate once, reuse across threads."""

    def __init__(self, api_key: str | None = None,
                 base_url: str | None = None):
        self.api_key  = api_key  or MASSIVE_API_KEY
        self.base_url = base_url or MASSIVE_BASE_URL
        self.session  = requests.Session()

    # ── aggregates ────────────────────────────────────────────────────────

    def get_aggs(
        self,
        ticker: str,
        multiplier: int,
        timespan: str,
        from_date: str | date,
        to_date: str | date,
        adjusted: bool = True,
        limit: int = 50_000,
    ) -> pd.DataFrame:
        """Return OHLCV bars as a DataFrame.

        Parameters
        ----------
        ticker      : e.g. "AAPL"
        multiplier  : bar size  (1 → 1-min, 5 → 5-min, …)
        timespan    : "minute" | "hour" | "day" | …
        from_date   : YYYY-MM-DD or date object
        to_date     : YYYY-MM-DD or date object
        """
        url = (f"{self.base_url}/v2/aggs/ticker/{ticker}/range/"
               f"{multiplier}/{timespan}/{from_date}/{to_date}")
        params: dict = {
            "apiKey":   self.api_key,
            "adjusted": str(adjusted).lower(),
            "sort":     "asc",
            "limit":    limit,
        }

        all_results: list[dict] = []
        page = 0
        while url:
            data = self._get_json(url, params)
            if data is None:
                break
            results = data.get("results", [])
            all_results.extend(results)
            page += 1
            logger.debug("%s — page %d, rows so far: %d",
                         ticker, page, len(all_results))
            # pagination
            next_url = data.get("next_url")
            if next_url:
                url = next_url
                params = {"apiKey": self.api_key}
                time.sleep(_RATE_LIMIT_SLEEP)
            else:
                url = None                              # type: ignore[assignment]

        if not all_results:
            logger.warning("No aggregate data returned for %s (%s→%s)",
                           ticker, from_date, to_date)
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        rename = {"o": "open", "h": "high", "l": "low", "c": "close",
                  "v": "volume", "vw": "vwap", "n": "num_transactions",
                  "t": "timestamp"}
        df.rename(columns={k: v for k, v in rename.items() if k in df.columns},
                  inplace=True)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms",
                                             utc=True)
            df["timestamp"] = df["timestamp"].dt.tz_convert(
                "America/New_York").dt.tz_localize(None)
        return df

    # ── tickers ───────────────────────────────────────────────────────────

    def get_tickers(
        self,
        market: str = "stocks",
        active: bool = True,
        limit: int = 1000,
        order: str = "desc",
    ) -> pd.DataFrame:
        """Return a DataFrame of tickers (with market_cap when available)."""
        url = f"{self.base_url}/v3/reference/tickers"
        params: dict = {
            "apiKey": self.api_key,
            "market": market,
            "active": str(active).lower(),
            "limit":  limit,
            "order":  order,
            "sort":   "market_cap" if market == "stocks" else "ticker",
        }
        all_results: list[dict] = []
        while url:
            data = self._get_json(url, params)
            if data is None:
                break
            all_results.extend(data.get("results", []))
            next_url = data.get("next_url")
            if next_url and len(all_results) < limit:
                url = next_url
                params = {"apiKey": self.api_key}
                time.sleep(_RATE_LIMIT_SLEEP)
            else:
                url = None                              # type: ignore[assignment]

        return pd.DataFrame(all_results) if all_results else pd.DataFrame()

    # ── internal helpers ──────────────────────────────────────────────────

    def _get_json(self, url: str, params: dict) -> dict | None:
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=30)
                if resp.status_code == 429:             # rate-limited
                    wait = _RETRY_BACKOFF ** attempt
                    logger.warning("Rate-limited, sleeping %.1fs", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as exc:
                logger.error("Request failed (attempt %d/%d): %s",
                             attempt, _MAX_RETRIES, exc)
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_BACKOFF ** attempt)
        return None
