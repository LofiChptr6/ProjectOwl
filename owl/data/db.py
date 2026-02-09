"""
Database utilities — PostgreSQL via SQLAlchemy
================================================
- Defines the ORM tables for training / validation cases and metrics.
- Provides helpers for CRUD, bulk insert, and session management.
"""

import logging
from contextlib import contextmanager

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import QueuePool

from owl.config import DATABASE_URL

logger = logging.getLogger(__name__)
Base = declarative_base()

_engine = None
_SessionFactory = None


# ── connection helpers ────────────────────────────────────────────────────

def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            echo=False,
        )
    return _engine


def get_session_factory():
    global _SessionFactory
    if _SessionFactory is None:
        _SessionFactory = sessionmaker(bind=get_engine())
    return _SessionFactory


@contextmanager
def get_session():
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ── ORM table definitions ────────────────────────────────────────────────

class _CaseBase(Base):
    """Abstract base — every column shared by training & validation."""
    __abstract__ = True

    id               = Column(Integer, primary_key=True, autoincrement=True)
    case_id          = Column(String(64), nullable=False, index=True)
    symbol           = Column(String(16), nullable=False, index=True)
    timestamp        = Column(DateTime, nullable=False)
    frequency        = Column(String(16), nullable=False, default="minute")

    # ── Massive (price) columns ──
    open             = Column(Float)
    high             = Column(Float)
    low              = Column(Float)
    close            = Column(Float)
    volume           = Column(BigInteger)
    vwap             = Column(Float)
    num_transactions = Column(Integer)

    # ── SHARADAR daily valuation ──
    marketcap        = Column(Float)
    pe               = Column(Float)
    pb               = Column(Float)
    ps               = Column(Float)

    # ── SHARADAR SF1 quarterly fundamentals (forward-filled, lagged 1 day) ──
    revenue          = Column(Float)
    ebitda           = Column(Float)
    eps              = Column(Float)
    netinc           = Column(Float)
    netmargin        = Column(Float)
    grossmargin      = Column(Float)
    operatingmargin  = Column(Float)
    de               = Column(Float)
    roe              = Column(Float)
    roa              = Column(Float)
    roic             = Column(Float)
    currentratio     = Column(Float)
    debtratio        = Column(Float)
    fcf              = Column(Float)
    workingcapital   = Column(Float)
    sharesbas        = Column(Float)
    shareswa         = Column(Float)


class TrainingCase(_CaseBase):
    __tablename__ = "training_cases"
    __table_args__ = (
        UniqueConstraint("case_id", "symbol", "timestamp", name="uq_train_case"),
        Index("idx_train_case_id", "case_id"),
        Index("idx_train_symbol", "symbol"),
    )


class ValidationCase(_CaseBase):
    __tablename__ = "validation_cases"
    __table_args__ = (
        UniqueConstraint("case_id", "symbol", "timestamp", name="uq_val_case"),
        Index("idx_val_case_id", "case_id"),
        Index("idx_val_symbol", "symbol"),
    )


class TrainingCaseHourly(_CaseBase):
    __tablename__ = "training_cases_hourly"
    __table_args__ = (
        UniqueConstraint("case_id", "symbol", "timestamp", name="uq_train_hourly"),
        Index("idx_train_hourly_case_id", "case_id"),
        Index("idx_train_hourly_symbol", "symbol"),
    )


class ValidationCaseHourly(_CaseBase):
    __tablename__ = "validation_cases_hourly"
    __table_args__ = (
        UniqueConstraint("case_id", "symbol", "timestamp", name="uq_val_hourly"),
        Index("idx_val_hourly_case_id", "case_id"),
        Index("idx_val_hourly_symbol", "symbol"),
    )


class CaseMetadata(Base):
    __tablename__ = "case_metadata"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    case_id    = Column(String(64), nullable=False, unique=True, index=True)
    symbol     = Column(String(16), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date   = Column(DateTime, nullable=False)
    split      = Column(String(16), nullable=False)   # "train" | "validation"
    num_rows   = Column(Integer)


class CaseMetadataHourly(Base):
    __tablename__ = "case_metadata_hourly"
    id         = Column(Integer, primary_key=True, autoincrement=True)
    case_id    = Column(String(64), nullable=False, unique=True, index=True)
    symbol     = Column(String(16), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date   = Column(DateTime, nullable=False)
    split      = Column(String(16), nullable=False)   # "train" | "validation"
    num_rows   = Column(Integer)


class TrainingMetrics(Base):
    __tablename__ = "training_metrics"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    timestamp    = Column(DateTime, server_default=text("NOW()"))
    model_name   = Column(String(64))
    epoch        = Column(Integer)
    batch        = Column(Integer)
    metric_name  = Column(String(64))
    metric_value = Column(Float)
    phase        = Column(String(16))   # "data_pull" | "preprocess" | "train" | "validate"


# ── DDL helpers ───────────────────────────────────────────────────────────

def create_tables():
    engine = get_engine()
    Base.metadata.create_all(engine)
    logger.info("All database tables created.")


def drop_tables():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    logger.info("All database tables dropped.")


# ── query helpers ─────────────────────────────────────────────────────────

def get_case_ids(table_name: str) -> list[str]:
    engine = get_engine()
    df = pd.read_sql(
        f"SELECT DISTINCT case_id FROM {table_name} ORDER BY case_id", engine
    )
    return df["case_id"].tolist()


def get_symbol_for_case(case_id: str, meta_table: str = "case_metadata") -> str | None:
    """Return symbol for *case_id* from metadata table, or None if not found."""
    engine = get_engine()
    df = pd.read_sql(
        f"SELECT symbol FROM {meta_table} WHERE case_id = %(cid)s",
        engine, params={"cid": case_id},
    )
    return df["symbol"].iloc[0] if not df.empty else None


def get_case_row_count(case_id: str, table_name: str) -> int:
    engine = get_engine()
    df = pd.read_sql(
        f"SELECT COUNT(*) AS cnt FROM {table_name} WHERE case_id = %(cid)s",
        engine, params={"cid": case_id},
    )
    return int(df["cnt"].iloc[0])


def load_case_data(case_id: str, table_name: str) -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(
        f"SELECT * FROM {table_name} WHERE case_id = %(cid)s ORDER BY timestamp",
        engine, params={"cid": case_id},
    )
    return df


def insert_case_data(df: pd.DataFrame, table_name: str,
                     if_exists: str = "append") -> None:
    engine = get_engine()
    df.to_sql(table_name, engine, if_exists=if_exists, index=False,
              method="multi", chunksize=1000)
    logger.debug("Inserted %d rows into %s", len(df), table_name)


def insert_metadata(case_id, symbol, start_date, end_date, split, num_rows,
                    meta_table: str = "case_metadata"):
    """Insert case metadata. Use meta_table='case_metadata_hourly' for hourly pipeline."""
    if meta_table == "case_metadata_hourly":
        cls = CaseMetadataHourly
    else:
        cls = CaseMetadata
    with get_session() as sess:
        sess.add(cls(
            case_id=case_id, symbol=symbol,
            start_date=start_date, end_date=end_date,
            split=split, num_rows=num_rows,
        ))


def clear_training_metrics() -> None:
    """Truncate training_metrics so dashboard shows only the current run."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE training_metrics"))
        conn.commit()


def log_metric(model_name: str, epoch: int, batch: int,
               metric_name: str, metric_value: float,
               phase: str = "train") -> None:
    with get_session() as sess:
        sess.add(TrainingMetrics(
            model_name=model_name, epoch=epoch, batch=batch,
            metric_name=metric_name, metric_value=metric_value,
            phase=phase,
        ))


def get_recent_metrics(model_name: str | None = None,
                       limit: int = 1000) -> pd.DataFrame:
    engine = get_engine()
    q = "SELECT * FROM training_metrics"
    if model_name:
        q += " WHERE model_name = %(mn)s"
    q += " ORDER BY id DESC LIMIT %(lim)s"
    return pd.read_sql(q, engine, params={"mn": model_name, "lim": limit})
