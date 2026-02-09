#!/usr/bin/env python
"""Clear all rows from ProjectOwl tables. Tables remain; only data is removed.

Usage::

    python scripts/clear_tables.py
"""

import logging
import sys
from pathlib import Path

from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from owl.config import (
    META_TABLE,
    META_TABLE_HOURLY,
    METRICS_TABLE,
    TRAINING_TABLE,
    TRAINING_TABLE_HOURLY,
    VALIDATION_TABLE,
    VALIDATION_TABLE_HOURLY,
)
from owl.data.db import get_engine

TABLES = [
    TRAINING_TABLE, VALIDATION_TABLE, META_TABLE,
    TRAINING_TABLE_HOURLY, VALIDATION_TABLE_HOURLY, META_TABLE_HOURLY,
    METRICS_TABLE,
]


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    engine = get_engine()
    with engine.connect() as conn:
        for table in TABLES:
            result = conn.execute(text(f"DELETE FROM {table}"))
            conn.commit()
            logging.info("Cleared %s (%d rows)", table, result.rowcount)

    logging.info("Done. All tables emptied.")


if __name__ == "__main__":
    main()
