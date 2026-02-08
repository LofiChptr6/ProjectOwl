# ProjectOwl — Time-Series Characterisation & Stock Price Prediction

An ML pipeline that characterises multi-dimensional financial time-series
and predicts stock price movements using CNN and Transformer architectures,
with t-SNE visualisation of learned embeddings.

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .               # install owl package in editable mode

# 3. Configure .env  (already present — verify your API keys)
#    MASSIVE_API_KEY=...
#    NASDAQ_DATA_LINK_API_KEY=...
#    DATABASE_URL=postgresql://postgres:5369@localhost:5432/postgres

# 4. Initialise database
python scripts/init_db.py

# 5. Populate with sample data (small run)
python scripts/populate_data.py --n-train 10 --n-val 3 --tickers AAPL MSFT NVDA

# 6. Train a model
python scripts/train.py --model cnn --epochs 10 --dashboard

# 7. Generate t-SNE visualisations
python scripts/run_tsne.py --model cnn --video

# Or run everything end-to-end:
python run_example.py --model cnn --dashboard
```

---

## Project Structure

```
ProjectOwl/
├── .env                          # API keys & DB connection (DO NOT COMMIT)
├── requirements.txt
├── setup.py
├── run_example.py                # End-to-end smoke test
│
├── owl/                          # Main package
│   ├── config.py                 # ◀ ALL tuneable parameters live here
│   ├── data/
│   │   ├── db.py                 # PostgreSQL ORM + helpers
│   │   ├── massive_client.py     # Massive/Polygon REST client
│   │   ├── sharadar_client.py    # SHARADAR SF1 + DAILY client
│   │   ├── query_engine.py       # Random case generation + data merging
│   │   └── feeder.py             # PyTorch Dataset + DataLoader
│   ├── preprocessing/
│   │   ├── normalizer.py         # ReVol, z-score, min-max, log-return
│   │   ├── features.py           # SMA, WMA, EMA, RSI, ARMA, GARCH, time
│   │   └── pipeline.py           # Chains everything into one transform
│   ├── models/
│   │   ├── base.py               # Shared training loop + checkpointing
│   │   ├── cnn_model.py          # Multi-scale 1D CNN
│   │   ├── transformer_model.py  # Transformer encoder with CLS token
│   │   └── tsne_viz.py           # t-SNE computation + scatter plots
│   ├── visualization/
│   │   ├── reports.py            # Category examples, feature importance
│   │   └── tsne_video.py         # Animated 2D/3D t-SNE trajectory videos
│   └── orchestration/
│       └── dashboard.py          # Dash web UI for live monitoring
│
├── scripts/
│   ├── init_db.py                # Create / reset tables
│   ├── populate_data.py          # Fetch data from APIs → PostgreSQL
│   ├── train.py                  # Train CNN or Transformer
│   └── run_tsne.py               # Generate t-SNE images + videos
│
└── outputs/                      # Generated artifacts
    ├── checkpoints/              # Model .pt files
    ├── reports/                  # PNG report images
    └── videos/                   # MP4 t-SNE trajectory videos
```

---

## Changing Scales & Frequencies

**All tuneable parameters are centralised in `owl/config.py`.**  Key sections:

| What to change | Config variable(s) |
|---|---|
| Bar frequency (1-min, 5-min, hourly…) | `PRICE_FREQUENCY`, `PRICE_MULTIPLIER` |
| Case window length | `CASE_WINDOW_TRADING_DAYS`, `CASE_WINDOW_CALENDAR_DAYS` |
| Model input window | `INPUT_WINDOW_MINUTES` |
| Prediction horizon | `PREDICTION_WINDOW_MINUTES` |
| Rolling-window stride | `WINDOW_STRIDE` |
| Moving-average windows | `MA_WINDOWS`, `WMA_WINDOWS`, `EMA_WINDOWS` |
| ARMA / GARCH orders | `ARMA_ORDER`, `GARCH_ORDER` |
| Classification thresholds | `CATEGORY_THRESHOLDS`, `NUM_CATEGORIES` |
| Normalisation method | `NORMALIZATION_METHOD` |
| Stock universe | `TOP_STOCKS_COUNT`, `MIN_MARKET_CAP` |

---

## Components

### 1. Data Query Engine (`owl/data/query_engine.py`)
- Generates random (symbol, date-window) pairs from the top ~1000 US stocks.
- Fetches minute-level OHLCV from Massive API.
- Fetches daily valuation + quarterly fundamentals from SHARADAR.
- **Frequency alignment**: SHARADAR data is lagged 1 day and forward-filled
  into minute bars.
- Writes merged results into `training_cases` / `validation_cases` tables.

### 2. Data Feeder (`owl/data/feeder.py`)
- PyTorch `Dataset` with LRU per-worker case cache.
- Produces rolling windows: 24 h input + 6 h prediction target.
- Multi-worker `DataLoader` keeps GPU saturated.

### 3. Preprocessing (`owl/preprocessing/`)
- **Normaliser**: ReVol (return-volatility), z-score, min-max, log-return.
- **Features**: SMA/WMA/EMA, Bollinger bands, RSI, ARMA residuals, GARCH
  conditional volatility, cyclical time encoding.
- **Pipeline**: single `.transform(df)` call applies everything.

### 4. Models (`owl/models/`)
- **CNN**: multi-scale 1-D convolutions → residual blocks → global pool → latent.
- **Transformer**: linear projection → learnable positional encoding →
  encoder stack → CLS-token latent.
- Both have a classification head and expose `.encode()` for t-SNE.

### 5. Visualisation
- Category example grids, gradient-based feature importance, loss/accuracy
  curves (`owl/visualization/reports.py`).
- t-SNE 2D/3D scatter plots (`owl/models/tsne_viz.py`).
- Animated MP4 trajectory videos showing a stock's rolling-window path
  through embedding space (`owl/visualization/tsne_video.py`).

### 6. Orchestration Dashboard (`owl/orchestration/dashboard.py`)
- Dash/Plotly web app at `http://localhost:8050`.
- Live training loss, validation accuracy, pipeline throughput, bottleneck chart.
- Launch: `python -m owl.orchestration.dashboard` or pass `--dashboard` to train.

---

## Data Sources

| Source | What | Frequency | Lag |
|---|---|---|---|
| Massive (Polygon) API | OHLCV + VWAP + txn count | by-minute | real-time |
| SHARADAR/DAILY | Market-cap, P/E, P/B, P/S | daily | 1 day |
| SHARADAR/SF1 (MRQ) | Revenue, EBITDA, EPS, margins, ratios | quarterly | 1 day |

---

## Prerequisites

- Python ≥ 3.9
- PostgreSQL (local, running on port 5432)
- CUDA-capable GPU recommended (falls back to CPU)
- Massive API key & Nasdaq Data Link API key (in `.env`)
