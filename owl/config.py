"""
ProjectOwl — Central Configuration
====================================
All tuneable scales, frequencies, window sizes, and model hyper-parameters
live here.  When you revisit a parameter later, change it in **one** place.

Sections
--------
1. API / DB credentials  (loaded from .env)
2. Time scales & frequencies
3. Stock universe
4. Technical-indicator windows
5. Normalization
6. Model architecture
7. Training
8. Visualisation / t-SNE
9. Orchestration dashboard
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── locate project root & load .env ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# ═════════════════════════════════════════════════════════════════════════
# 1.  API KEYS & DATABASE
# ═════════════════════════════════════════════════════════════════════════
MASSIVE_API_KEY        = os.getenv("MASSIVE_API_KEY")
NASDAQ_DATA_LINK_API_KEY = os.getenv("NASDAQ_DATA_LINK_API_KEY")
DATABASE_URL           = os.getenv("DATABASE_URL",
                                   "postgresql://postgres:5369@localhost:5432/postgres")

# Base URL for the Massive / Polygon REST API
MASSIVE_BASE_URL = os.getenv("MASSIVE_BASE_URL", "https://api.polygon.io")

# ═════════════════════════════════════════════════════════════════════════
# 2.  TIME SCALES & FREQUENCIES   ◀── EASY TO CHANGE
# ═════════════════════════════════════════════════════════════════════════
# How large each random "case" window is
CASE_WINDOW_TRADING_DAYS  = 10          # ~2 weeks of trading
CASE_WINDOW_CALENDAR_DAYS = 14          # calendar-day span used in API calls

# Price bar frequency fed to Massive API
PRICE_FREQUENCY   = "minute"            # "minute" | "hour" | "day"
PRICE_MULTIPLIER  = 1                   # 1-min bars

# US equity market hours (Eastern Time)
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR  = 16
MARKET_CLOSE_MINUTE = 0
TRADING_MINUTES_PER_DAY = 390           # 09:30 → 16:00

# ── Model input / output windows (bar counts)
INPUT_WINDOW_MINUTES      = 1440        # 24 hours of minute bars
PREDICTION_WINDOW_MINUTES = 360         # 6 hours to predict
WINDOW_STRIDE             = 60          # rolling-window stride (minutes)

# ═════════════════════════════════════════════════════════════════════════
# 3.  STOCK UNIVERSE
# ═════════════════════════════════════════════════════════════════════════
TOP_STOCKS_COUNT          = 1000
MIN_MARKET_CAP            = 1e9         # USD
DATA_START_DATE           = "2016-01-01"
DATA_END_DATE             = "2025-12-31"
RANDOM_CASES_TRAINING     = 500
RANDOM_CASES_VALIDATION   = 100

# ═════════════════════════════════════════════════════════════════════════
# 4.  TECHNICAL INDICATORS   ◀── EASY TO CHANGE
# ═════════════════════════════════════════════════════════════════════════
MA_WINDOWS   = [5, 10, 20, 50, 100, 200]
WMA_WINDOWS  = [5, 10, 20, 50]
EMA_WINDOWS  = [5, 10, 20, 50, 100]
ARMA_ORDER   = (2, 2)                   # (p, q) for ARMA
GARCH_ORDER  = (1, 1)                   # (p, q) for GARCH
BOLLINGER_WINDOW = 20
BOLLINGER_STD    = 2
RSI_WINDOW       = 14

# ═════════════════════════════════════════════════════════════════════════
# 5.  NORMALIZATION   ◀── EASY TO CHANGE
# ═════════════════════════════════════════════════════════════════════════
NORMALIZATION_METHOD  = "revol"         # "revol" | "zscore" | "minmax" | "log_return"
VOLUME_NORMALIZATION  = "log_zscore"    # "log_zscore" | "zscore" | "minmax"

# ═════════════════════════════════════════════════════════════════════════
# 6.  MODEL ARCHITECTURE   ◀── EASY TO CHANGE
# ═════════════════════════════════════════════════════════════════════════
NUM_CATEGORIES       = 5
CATEGORY_THRESHOLDS  = [-0.02, -0.005, 0.005, 0.02]
CATEGORY_NAMES       = ["strong_down", "down", "flat", "up", "strong_up"]

# CNN
CNN_CHANNELS     = [64, 128, 256, 512]
CNN_KERNEL_SIZES = [3, 5, 7, 11]        # multi-scale kernels

# Transformer
TRANSFORMER_D_MODEL    = 256
TRANSFORMER_NHEAD      = 8
TRANSFORMER_NUM_LAYERS = 6
TRANSFORMER_DIM_FF     = 1024
TRANSFORMER_DROPOUT    = 0.1

# Shared
LATENT_DIM = 128                        # embedding dimension for t-SNE

# ═════════════════════════════════════════════════════════════════════════
# 7.  TRAINING
# ═════════════════════════════════════════════════════════════════════════
BATCH_SIZE     = 64
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-5
NUM_EPOCHS     = 100
NUM_WORKERS    = 4                      # DataLoader workers
DEVICE         = "cuda"                 # auto-detected at runtime
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"

# ═════════════════════════════════════════════════════════════════════════
# 8.  VISUALISATION / t-SNE
# ═════════════════════════════════════════════════════════════════════════
TSNE_PERPLEXITY      = 30
TSNE_N_ITER          = 1000
TSNE_LEARNING_RATE   = 200.0
REPORT_DIR           = PROJECT_ROOT / "outputs" / "reports"
REPORT_DPI           = 150
VIDEO_DIR            = PROJECT_ROOT / "outputs" / "videos"
VIDEO_FPS            = 30
VIDEO_DURATION_SEC   = 30

# ═════════════════════════════════════════════════════════════════════════
# 9.  ORCHESTRATION DASHBOARD
# ═════════════════════════════════════════════════════════════════════════
DASHBOARD_PORT              = 8050
DASHBOARD_UPDATE_INTERVAL_MS = 1000
METRICS_QUEUE_SIZE          = 10_000

# ═════════════════════════════════════════════════════════════════════════
# 10. DATABASE TABLE NAMES
# ═════════════════════════════════════════════════════════════════════════
TRAINING_TABLE   = "training_cases"
VALIDATION_TABLE = "validation_cases"
META_TABLE       = "case_metadata"
METRICS_TABLE    = "training_metrics"

# ═════════════════════════════════════════════════════════════════════════
# 11. SHARADAR COLUMNS TO PULL   ◀── EASY TO CHANGE
# ═════════════════════════════════════════════════════════════════════════
SHARADAR_DAILY_COLUMNS = [
    "marketcap", "pe", "pb", "ps",
]

SHARADAR_SF1_COLUMNS = [
    "revenue", "ebitda", "eps", "netinc",
    "netmargin", "grossmargin", "operatingmargin",
    "de", "roe", "roa", "roic",
    "currentratio", "debtratio",
    "fcf", "workingcapital",
    "sharesbas", "shareswa",
]

# All fundamental columns (daily + quarterly) that appear in the DB
FUNDAMENTAL_COLUMNS = SHARADAR_DAILY_COLUMNS + SHARADAR_SF1_COLUMNS

# Price columns from Massive
PRICE_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "num_transactions"]
