#!/usr/bin/env python
"""Create (or recreate) all PostgreSQL tables used by ProjectOwl.

Usage::

    python scripts/init_db.py          # create tables
    python scripts/init_db.py --drop   # drop + recreate
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.data.db import create_tables, drop_tables


def main():
    parser = argparse.ArgumentParser(description="Initialise ProjectOwl database")
    parser.add_argument("--drop", action="store_true",
                        help="Drop all tables before recreating")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.drop:
        logging.info("Dropping all tables…")
        drop_tables()

    logging.info("Creating tables…")
    create_tables()
    logging.info("Done.")


if __name__ == "__main__":
    main()
