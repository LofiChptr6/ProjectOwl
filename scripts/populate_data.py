#!/usr/bin/env python
"""Populate the training & validation tables with random cases.

Usage::

    python scripts/populate_data.py                        # defaults from config
    python scripts/populate_data.py --n-train 50 --n-val 10  # quick test run
    python scripts/populate_data.py --tickers AAPL MSFT NVDA  # specific stocks
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.data.db import create_tables
from owl.data.query_engine import populate_database


def main():
    parser = argparse.ArgumentParser(description="Populate ProjectOwl DB")
    parser.add_argument("--n-train", type=int, default=None,
                        help="Number of training cases (default from config)")
    parser.add_argument("--n-val", type=int, default=None,
                        help="Number of validation cases (default from config)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Restrict to specific ticker symbols")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    # ensure tables exist
    create_tables()

    populate_database(
        n_train=args.n_train,
        n_val=args.n_val,
        tickers=args.tickers,
    )
    logging.info("Population complete.")


if __name__ == "__main__":
    main()
