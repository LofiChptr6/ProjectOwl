#!/usr/bin/env python
"""Populate the hourly training & validation tables with random cases.

Same logic as populate_data.py but fetches by-hour bars from Polygon and
stores in training_cases_hourly / validation_cases_hourly / case_metadata_hourly.
For each symbol, cases are generated in pairs that overlap by >= 50% of the window.

Run before run_example_hourly.py.

Usage::

    python scripts/populate_data_hourly.py
    python scripts/populate_data_hourly.py --cases-per-symbol 12
    python scripts/populate_data_hourly.py --tickers AAPL MSFT NVDA
    python scripts/populate_data_hourly.py --workers 16
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.data.db import create_tables
from owl.data.query_engine import populate_database


def main():
    parser = argparse.ArgumentParser(description="Populate ProjectOwl hourly DB")
    parser.add_argument("--cases-per-symbol", type=int, default=None,
                        help="Random snapshots per ticker (default from config)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max concurrent fetch+insert workers")
    parser.add_argument("--n-train", type=int, default=None,
                        help="(Legacy) Total training cases")
    parser.add_argument("--n-val", type=int, default=None,
                        help="(Legacy) Total validation cases")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Restrict to specific ticker symbols")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    create_tables()

    populate_database(
        n_train=args.n_train,
        n_val=args.n_val,
        tickers=args.tickers,
        cases_per_symbol=args.cases_per_symbol,
        max_workers=args.workers,
        frequency="hour",
        overlap_pairs=True,
        overlap_fraction=0.5,
    )
    logging.info("Hourly population complete.")


if __name__ == "__main__":
    main()
