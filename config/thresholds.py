"""
Module: config/thresholds.py
Purpose: All signal thresholds, risk parameters, and quality constants.
         Every magic number in the system lives here. No exceptions.
Phase: 3/4 (production)
Last modified: 2026-06-17
"""

# ── Phase 3 Feature Set (14 features) ──────────────────────────────────────
# 8 market + 3 attention + 1 news + 2 StockTwits
# Locked via experiments/phase3_locked_architecture.json
# L1 Granger: Reddit sentiment 0/6 years significant → dropped.
# News + StockTwits added Phase 4. Historical rows default to 0.0.
PHASE3_FEATURES: list = [
    # Market (8)
    'returns_1d', 'returns_5d', 'returns_20d',
    'rsi_14', 'atr_14', 'relative_volume',
    'dist_from_20ma', 'dist_from_50ma',
    # Attention (3)
    'post_count_1d', 'mention_growth_1d', 'mention_growth_7d',
    # News (1)
    'news_sentiment_1d',
    # StockTwits (2)
    'st_sentiment_1d', 'st_bull_pct',
]

CLEAN_FEATURES = PHASE3_FEATURES  # backward-compat alias (used by experiments/)

# Sentiment features dropped per L1 Granger (0/6 years significant)
SENTIMENT_FEATURES_DROPPED: list = [
    'avg_sentiment_1d', 'avg_sentiment_3d', 'weighted_sentiment',
    'sentiment_std', 'sentiment_accel', 'bullish_ratio',
]

# ── Density Gate ────────────────────────────────────────────────────────────
DENSITY_GATE: int = 10          # post_count_1d must be >= this to pass filter

# ── Drop Tickers ────────────────────────────────────────────────────────────
# Insufficient training rows or persistent data quality issues
DROP_TICKERS: list = ['ASTS', 'LCID', 'MSTR', 'RIOT', 'RIVN', 'SMCI', 'WMT']

# ── Portfolio Rules ─────────────────────────────────────────────────────────
MAX_POSITIONS: int        = 3      # never open more than 3 simultaneously
HOLD_DAYS: int            = 5      # standard hold period (trading days)
TAKE_PROFIT_CAP: float    = 0.15   # close early if unrealized gain >= 15%
MIN_PRED_RETURN: float    = 0.01   # minimum |predicted_5d| to enter position
TICKER_COOLDOWN_DAYS: int = 7      # days before re-entering same ticker

# ── Position Sizing ─────────────────────────────────────────────────────────
TARGET_RISK_PCT: float  = 0.02   # risk 2% of portfolio per trade (ATR-based)
MAX_POSITION_PCT: float = 0.25   # never exceed 25% of portfolio per position

# ── Regime Sizing ───────────────────────────────────────────────────────────
REGIME_SIZING: dict = {
    'positive': 1.00,   # SPY above 200MA + positive 60d return
    'neutral':  0.75,
    'negative': 0.50,
}

# ── Signal Classification ───────────────────────────────────────────────────
BULLISH_THRESHOLD: float  =  0.03   # predicted_5d >= 3% → BULLISH
BEARISH_THRESHOLD: float  = -0.03   # predicted_5d <= -3% → BEARISH

# ── Live Monitoring Gates ───────────────────────────────────────────────────
IC_GREEN_GATE: float = 0.03    # 30-day live IC above this → model working
IC_AMBER_GATE: float = 0.01    # 30-day live IC in [0.01, 0.03) → watch closely
IC_RED_GATE: float   = 0.01    # 30-day live IC below this → trigger Fix 3 after 2 weeks

# ── Slippage (dynamic) ──────────────────────────────────────────────────────
SLIPPAGE_BASE: float       = 0.001   # 0.1% base slippage
SLIPPAGE_GROWTH_COEF: float = 0.0005  # additional per unit of mention_growth_7d
SLIPPAGE_GROWTH_CAP: float  = 3.0    # cap mention_growth_7d at 3.0 for slippage calc

# ── Dataset Quality Filters (used by pipeline/01_feature_builder.py) ───────
MIN_POST_COUNT: int         = 10     # exclude (ticker, date) rows below this
MIN_CONFIDENCE: float       = 0.8    # FinBERT confidence gate for avg_sentiment_hc
MAX_RETURN_WINSORISE: float = 0.50   # clip |target_return_5d| > 50%

# ── Statistical Validation (used by pipeline/03,05) ────────────────────────
MIN_IC_THRESHOLD: float   = 0.03
REDDIT_ADDS_VALUE_IC: float = 0.005
N_PERMUTATIONS: int       = 100
N_BOOTSTRAP: int          = 50
PVALUE_THRESHOLD: float   = 0.05

# ── Backtest Mechanics (used by pipeline/02,04 and experiments/) ────────────
STARTING_CAPITAL: float  = 1_000.0
SLIPPAGE: float          = 0.001    # flat slippage for historical backtests
FEE_PER_LEG: float       = 0.0005  # 0.05% per leg

# ── Experiment C paths (used by experiments/experiment_c/) ──────────────────
ATTENTION_FILTER_MIN_POSTS: int  = 10
EXPANDED_PARQUET_PATH: str  = 'data/raw/merged_with_sentiment_expanded.parquet'
EXPANDED_FEATURES_PATH: str = 'data/features/features_expanded.parquet'

# ── IC Classification (used by experiments/) ────────────────────────────────
IC_WEAK_SIGNAL: float     = 0.05
IC_MEANINGFUL: float      = 0.10
IC_STRONG: float          = 0.15
IC_OVERFIT_WARNING: float = 0.25
EXPERIMENT_MIN_IC: float  = 0.05
EXPERIMENT_MIN_SHARPE: float = 1.0
