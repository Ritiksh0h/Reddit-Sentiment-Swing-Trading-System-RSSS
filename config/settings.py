"""
Module: config/settings.py
Purpose: Central configuration — environment variables, resolved paths, constants.
         All values come from .env or OS environment. Never hardcoded here.
Phase: All
Last modified: 2026-06-11
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).parent.parent.resolve()

MODEL_REGISTRY_PATH: Path = Path(
    os.getenv("MODEL_REGISTRY_PATH", str(PROJECT_ROOT / "models" / "registry"))
)
RAW_DATA_PATH: Path = PROJECT_ROOT / "data" / "raw"
LOG_PATH: Path = PROJECT_ROOT / "logs"

# Ensure critical directories exist at import time
for _path in [MODEL_REGISTRY_PATH, RAW_DATA_PATH, LOG_PATH]:
    _path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reddit API (PRAW)
# ---------------------------------------------------------------------------
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "rsss/0.1")

# Subreddits monitored — per §4.1, no additions without spec update
MONITORED_SUBREDDITS: list[str] = [
    "wallstreetbets",
    "stocks",
    "investing",
    "options",
    "valueinvesting",
]

# ---------------------------------------------------------------------------
# Market Data
# ---------------------------------------------------------------------------
POLYGON_API_KEY: str = os.getenv("POLYGON_API_KEY", "")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
DB_URL: str = os.getenv("DB_URL", "sqlite:///rsss_dev.db")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# FinBERT
# ---------------------------------------------------------------------------
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
FINBERT_MAX_LENGTH: int = 512
FINBERT_BATCH_SIZE: int = 32
# "cuda", "mps", or "cpu" — resolved at runtime in NLP module; never hardcode GPU
FINBERT_DEVICE: str = os.getenv("FINBERT_DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Market Calendar & Timezone
# ---------------------------------------------------------------------------
MARKET_TIMEZONE: str = "America/New_York"
MARKET_OPEN_HOUR_ET: int = 9
MARKET_OPEN_MINUTE_ET: int = 30
MARKET_CLOSE_HOUR_ET: int = 16

# ---------------------------------------------------------------------------
# Train / Test Split — NON-NEGOTIABLE per §3.2
# ---------------------------------------------------------------------------
TRAIN_END_DATE: str = "2023-12-31"
TEST_START_DATE: str = "2024-01-01"

# ---------------------------------------------------------------------------
# Phase 1 — Research Pipeline  (aliases used by pipeline/ scripts)
# ---------------------------------------------------------------------------
DATA_RAW: Path = RAW_DATA_PATH
DATA_PROC: Path = PROJECT_ROOT / "data" / "processed"
DATA_FEAT: Path = PROJECT_ROOT / "data" / "features"
MODELS_DIR: Path = MODEL_REGISTRY_PATH
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

SENTIMENT_PARQUET: Path      = DATA_RAW / "merged_with_sentiment.parquet"
FEATURES_FULL_PATH: Path     = DATA_FEAT / "features_full.parquet"
FEATURES_COMPLETE_PATH: Path = DATA_FEAT / "features_complete.parquet"
FEATURES_PARQUET: Path       = FEATURES_FULL_PATH  # pipeline/ alias → features_full

EXPERIMENT_RESULTS_DIR: Path = PROJECT_ROOT / "experiments"

# ---------------------------------------------------------------------------
# Ticker Universe Paths
# ---------------------------------------------------------------------------
TICKERS_TRADE_PATH: Path = PROJECT_ROOT / 'config' / 'tickers_trade.txt'
TICKERS_WATCH_PATH: Path = PROJECT_ROOT / 'config' / 'tickers_watch.txt'
TICKERS_DROP_PATH:  Path = PROJECT_ROOT / 'config' / 'tickers_drop.txt'


def load_tickers(path) -> list[str]:
    """
    Load ticker list from a text file.
    Skips blank lines and comment lines starting with #.
    Returns uppercase deduplicated list in file order.
    """
    p = Path(path)
    if not p.exists():
        return []
    tickers = []
    seen: set = set()
    for line in p.read_text().splitlines():
        t = line.strip().upper()
        if t and not t.startswith('#') and t not in seen:
            tickers.append(t)
            seen.add(t)
    return tickers

# Phase 1 train/test split (same dates, shorter alias names for pipeline scripts)
TRAIN_END: str = TRAIN_END_DATE    # "2023-12-31"
TEST_START: str = TEST_START_DATE  # "2024-01-01"

# Market cutoff (09:30 ET = feature boundary)
MARKET_TZ: str = MARKET_TIMEZONE
CUTOFF_HOUR: int = MARKET_OPEN_HOUR_ET    # 9
CUTOFF_MIN: int = MARKET_OPEN_MINUTE_ET   # 30

# Create Phase 1 directories on import
for _p1_path in [DATA_PROC, DATA_FEAT, REPORTS_DIR]:
    _p1_path.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Phase 0 — Signal Validation
# ---------------------------------------------------------------------------
# Ordered by WSB mention frequency — highest first for best IC signal chance
PHASE0_TICKERS: list[str] = [
    "NVDA", "TSLA", "AMD", "AAPL",
    "GME", "AMC", "PLTR", "MARA", "COIN",
    "SPY",   # benchmark
]

# Known HuggingFace dataset candidates (ranked by fit for this project).
# Web search was unavailable at scaffold time — verify IDs before use.
# See scripts/phase0_validate.py for loading logic.
# ---------------------------------------------------------------------------
# Dynamic risk-budget engine — TASK 3
# ---------------------------------------------------------------------------
BASE_RISK_PCT        = 0.005    # fractional Kelly base (0.5% of equity per trade)
BASE_RISK_PCT_MAX    = 0.0075   # hard ceiling regardless of regime/confidence
ATR_STOP_MULT        = 2.5      # stop = -(ATR_STOP_MULT × atr_pct), clamped below
ATR_STOP_MIN         = -0.12    # widest allowed stop (-12%)
ATR_STOP_MAX         = -0.04    # tightest allowed stop (-4%)
ATR_STOP_DEFAULT     = -0.08    # fallback when ATR unavailable

POS_CAP_HIGH         = 0.20     # max single position in bull regime (20% of equity)
POS_CAP_MED          = 0.15     # max single position in neutral regime
POS_CAP_LOW          = 0.10     # max single position in bear regime

MAX_POSITIONS_BULL   = 6        # max concurrent positions in bull
MAX_POSITIONS_BEAR   = 3        # max concurrent positions in bear
MAX_POSITIONS_CHOPPY = 2        # max concurrent positions in choppy/neutral

HEAT_BUDGET_BULL     = 0.06     # max total portfolio risk deployed in bull (6%)
HEAT_BUDGET_BEAR     = 0.03     # max total portfolio risk deployed in bear (3%)
HEAT_BUDGET_CHOPPY   = 0.02     # max total portfolio risk deployed in choppy (2%)

DEPLOY_MAX_BULL      = 0.80     # max equity deployed as positions in bull
DEPLOY_MAX_BEAR      = 0.40     # max equity deployed as positions in bear
DEPLOY_MAX_CHOPPY    = 0.20     # max equity deployed as positions in choppy

MAX_BOOK_CORR        = 0.70     # max pairwise 60-day return correlation
MAX_CORR_CLUSTER     = 2        # max positions from semiconductor cluster
SEMI_CLUSTER: set    = {'NVDA', 'AMD', 'MU', 'INTC', 'ARM'}
SEMI_MAX_EXPOSURE    = 0.35     # max portfolio weight in SEMI_CLUSTER combined

PHASE0_HF_DATASETS: list[dict] = [
    {
        "id": "Lelon/reddit-wsb-posts",
        "notes": "Pushshift-derived WSB posts 2012-2022, ~500k+. Best fit.",
        "text_col": "title",
        "body_col": "selftext",
        "score_col": "score",
        "ts_col": "created_utc",
        "author_col": "author",
        "post_id_col": "id",
        "subreddit_col": "subreddit",
    },
    {
        "id": "RomanBlanco/reddit_wsb_2021",
        "notes": "WSB-specific, focused on GME squeeze ~Jan-mid 2021. Narrow window.",
        "text_col": "title",
        "body_col": "selftext",
        "score_col": "score",
        "ts_col": "created_utc",
        "author_col": "author",
        "post_id_col": "id",
        "subreddit_col": "subreddit",
    },
    {
        "id": "SocialGrep/one-million-reddit-comments",
        "notes": "Comments only (no title/selftext). Fallback if post datasets unavailable.",
        "text_col": "body",
        "body_col": None,
        "score_col": "score",
        "ts_col": "created_utc",
        "author_col": "author",
        "post_id_col": "id",
        "subreddit_col": "subreddit",
    },
]
